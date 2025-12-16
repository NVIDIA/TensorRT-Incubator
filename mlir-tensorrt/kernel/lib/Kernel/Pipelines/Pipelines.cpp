//===- Pipelines.cpp ------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// Definitions for passes and pipelines associated with the KernelDialect or
/// GPU code generation.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "mlir-kernel/Kernel/TransformSchedules/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::kernel;

void kernel::buildTransformIRPipeline(OpPassManager &pm, StringRef funcFilter,
                                      int64_t computeCapability,
                                      int64_t maxSharedMemoryPerBlockKb,
                                      uint64_t maxRegistersPerBlock,
                                      ArrayRef<std::string> generatorBenefit) {
  kernel::InitialTransformSchedulePassOptions opts{};
  opts.funcFilter = funcFilter.str();
  opts.computeCapability = computeCapability;
  opts.maxSharedMemoryPerBlockKb = maxSharedMemoryPerBlockKb;
  opts.maxRegistersPerBlock = maxRegistersPerBlock;
  opts.generatorBenefit = SmallVector<std::string>(generatorBenefit.begin(),
                                                   generatorBenefit.end());

  pm.addPass(kernel::createInitialTransformSchedulePass(opts));
  pm.addPass(kernel::createApplyTransformsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/// This function is used as a callback for the `promote-buffers-to-stack` pass.
/// It returns true if the given value is produced by a `memref.alloc`
/// representing a GPU private memory allocation.
///
/// Note that we currently lower any non-shared alloc to alloca. Failure to
/// lower promote a non-shared memref.alloc due to e.g. size limits imposed here
/// or due to it escaping control flow is an error.
static bool isPromotableAllocation(Value alloc) {
  auto type = dyn_cast<MemRefType>(alloc.getType());
  if (!type || !alloc.getDefiningOp<memref::AllocOp>())
    return false;

  if (!type.hasStaticShape()) {
    /// TODO: we could lower dynamic sizes if we can prove that an upper bound
    /// on the shape using e.g. integer range analysis of
    /// ValueBoundsOpInterface.
    return false;
  }

  if (auto addrSpace = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(
          type.getMemorySpace())) {
    if (addrSpace.getValue() == gpu::AddressSpace::Workgroup)
      return false;
  }

  auto parentOp = alloc.getDefiningOp()->getParentOfType<func::FuncOp>();
  if (!parentOp)
    return false;

  /// TODO: shouldn't have to repeatedly query the DataLayout. The transform
  /// should use data layout analysis and pass it to the callback.
  unsigned byteSize = mlir::DataLayout::closest(alloc.getDefiningOp())
                          .getTypeSize(type.getElementType());

  return type.getNumElements() * byteSize <= 256;
}

void kernel::buildKernelMemRefOptimizationPipeline(
    OpPassManager &kernelModulePM) {

  kernelModulePM.addNestedPass<func::FuncOp>(
      mlir::bufferization::createPromoteBuffersToStackPass(
          /*isSmallAlloc=*/isPromotableAllocation));
  auto &funcPM = kernelModulePM.nest<func::FuncOp>();
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(mlir::createCSEPass());
  funcPM.addPass(mlir::createSROA());
  funcPM.addPass(mlir::createMem2Reg());

  // Lower any lingering linalg ops to loops. This must be done after
  // bufferization since this transform doesn't work on tensors. This can occur
  // if a linalg op fails to vectorize. It happens when, for example, a linalg
  // op operates on complex number types.
  kernelModulePM.addPass(mlir::createConvertLinalgToLoopsPass());

  // Do post-bufferization optimization.
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(memref::createFoldMemRefAliasOpsPass());
  kernelModulePM.addPass(memref::createExpandStridedMetadataPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(mlir::createConvertVectorToSCFPass(
      VectorTransferToSCFOptions().enableFullUnroll()));
  kernelModulePM.addPass(createCanonicalizerPass());
}

void kernel::buildKernelLowerToPTXPipeline(OpPassManager &kernelModulePM,
                                           StringRef dumpPtxPath) {
  // Handle shared memory allocations by converting to globals.
  kernelModulePM.addPass(createKernelExpandMemRefArgsPass());
  kernelModulePM.addPass(createSharedAllocToGlobalPass());

  // Eliminate any extraneous linalg operations that could not be vectorized.
  // This can happen for e.g. gather/scatter-type linalg ops that were not
  // vectorized.
  kernelModulePM.addNestedPass<func::FuncOp>(
      mlir::createConvertLinalgToLoopsPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      createConvertComplexToStandardPass());

  kernelModulePM.addPass(affine::createAffineExpandIndexOpsAsAffinePass());
  kernelModulePM.addPass(createLowerAffinePass());
  kernelModulePM.addPass(arith::createArithExpandOpsPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(createCanonicalizerPass());

  // Lower to NVVM and translate to PTX.
  kernelModulePM.addPass(arith::createArithEmulateUnsupportedFloats(
      arith::ArithEmulateUnsupportedFloatsOptions{{"f8E4M3FN"}, "f16"}));
  kernelModulePM.addPass(kernel::createKernelSpecialFloatsTypeConversionPass());
  kernelModulePM.addPass(
      kernel::createKernelNormalizeQuantizedConversionsPass());
  kernel::LowerToNVVMPassOptions lowerToNVVMOpts{};
  lowerToNVVMOpts.preserveStructuredControlFlow = false;
  kernelModulePM.addPass(kernel::createLowerToNVVMPass(lowerToNVVMOpts));
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(createCanonicalizerPass());

  TranslateNVVMToPTXPassOptions opts{};
  opts.dumpPtxPath = dumpPtxPath;
  kernelModulePM.addPass(kernel::createTranslateNVVMToPTXPass(opts));
}

namespace {

struct KernelLowerToPTXPipelineOptions
    : public PassPipelineOptions<KernelLowerToPTXPipelineOptions> {
  Option<int64_t> deviceComputeCapability{
      *this, "device-compute-capability",
      llvm::cl::desc("target device compute capability (SM version)"),
      llvm::cl::init(60)};

  Option<std::string> dumpPtxPath{
      *this, "dump-ptx",
      llvm::cl::desc(
          "dump generated PTX as files in this directory; the directory "
          "will be created if it does not exist"),
      llvm::cl::init("")};
};

} // namespace

void kernel::registerKernelPipelines() {

  PassPipelineRegistration<KernelLowerToPTXPipelineOptions>(
      "kernel-lower-to-ptx-pipeline",
      "lowers bufferized and optimized GPU kernel IR to NVVM and translates to "
      "PTX",
      [](OpPassManager &pm,
         const KernelLowerToPTXPipelineOptions &pipelineOptions) {
        buildKernelLowerToPTXPipeline(pm, pipelineOptions.dumpPtxPath);
      });
}
