//===- ExecuteConstantFoldableSubgraphs.cpp -------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of the `plan-execute-constant-foldable-subgraphs` pass.
///
//===----------------------------------------------------------------------===//

#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-kernel/InitAllDialects.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-tensorrt-common/Utils/PassManagerUtils.h"
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-tensorrt/Backends/Kernel/Passes.h"
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mtrt {
#define GEN_PASS_DEF_PLANEXECUTECONSTANTFOLDABLESUBGRAPHSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

//===----------------------------------------------------------------------===//
// SubgraphExecutor
//===----------------------------------------------------------------------===//

namespace {
/// This is a stateless and thread-safe class that provides an interface to
/// execute `ModuleOp` (containing executor IR) by wrapping necessary runtime
/// methods. This class is expected to be used to execute constant foldable
/// modules with no input arguments.
class SubgraphExecutor {
public:
  /// Executes `main` function from `op` (containing executor IR) and returns
  /// `RuntimeValue` results of execution after converting them to `Attribute`.
  /// Returned `Attribute` is of type `DenseElementsAttr` if `RuntimeValue` is
  /// of type `MemRefValue` else `IntegerAttr` or `FloatAttr` is returned. Entry
  /// function `main` in `op` should not have any input arguments.
  StatusOr<llvm::SmallVector<Attribute>> execute(MLIRContext &ctx,
                                                 Ref<RuntimeClient> client,
                                                 Ref<Stream> stream,
                                                 ModuleOp op);

private:
  /// Assuming `data` resides on the host, creates and returns
  /// `DenseElementsAttr` of shape `shape` and element type `code`.
  static StatusOr<Attribute>
  getDenseElementsAttrFromHostRawBuffer(MLIRContext &ctx, ScalarTypeCode code,
                                        ArrayRef<int64_t> shape, void *data,
                                        int64_t numBytes);

  /// Converts host visible scalar runtime value to IntegerAttr OR FloatAttr and
  /// returns it.
  static StatusOr<Attribute> convertScalarValueToAttr(MLIRContext &ctx,
                                                      ScalarValue *value);

  /// Converts device visible runtime value to `DenseElementsAttr` and returns
  /// it.
  static StatusOr<Attribute>
  convertMemRefValueToDenseElementsAttr(MLIRContext &ctx,
                                        Ref<RuntimeClient> client,
                                        Ref<Stream> stream, MemRefValue *value);

  /// Converts a list of `RuntimeValue`s to a list of `Attribute`s.
  static StatusOr<llvm::SmallVector<Attribute>>
  convertRuntimeValuesToAttributes(
      MLIRContext &ctx, Ref<RuntimeClient> client, Ref<Stream> stream,
      llvm::SmallVectorImpl<std::unique_ptr<RuntimeValue>> &results);
};

StatusOr<llvm::SmallVector<Attribute>>
SubgraphExecutor::execute(MLIRContext &ctx, Ref<RuntimeClient> client,
                          Ref<Stream> stream, ModuleOp op) {
  // Translate to runtime executable.
  mlir::FailureOr<std::unique_ptr<mtrt::ExecutableStorage>> exeStorage =
      mlir::translateToRuntimeExecutable(op);
  if (failed(exeStorage))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to translate compiled MLIR module to a "
                            "MLIR-TensorRT runtime Executable");
  auto executable = std::make_unique<mtrt::Executable>(std::move(*exeStorage));
  // Create lua runtime session.
  auto opts = RuntimeSessionOptions::getSPMDOptions();
  opts.enableFeatures({"core", "cuda"});
  MTRT_ASSIGN_OR_RETURN(
      std::unique_ptr<LuaRuntimeSession> luaRuntimeSession,
      LuaRuntimeSession::create(client, opts, executable->getView()));
  // Execute `main` function.
  MTRT_RETURN_IF_ERROR(luaRuntimeSession->setStream(stream));
  MTRT_ASSIGN_OR_RETURN(
      llvm::SmallVector<std::unique_ptr<RuntimeValue>> results,
      luaRuntimeSession->executeFunction("main",
                                         llvm::ArrayRef<RuntimeValue *>{},
                                         llvm::ArrayRef<RuntimeValue *>{}));
  return SubgraphExecutor::convertRuntimeValuesToAttributes(ctx, client, stream,
                                                            results);
}

StatusOr<Attribute> SubgraphExecutor::getDenseElementsAttrFromHostRawBuffer(
    MLIRContext &ctx, ScalarTypeCode code, ArrayRef<int64_t> shape, void *data,
    int64_t numBytes) {
  switch (code) {
  case ScalarTypeCode::i1:
    // i1 is a special case because its packed during storage while all other
    // types are 8 bit aligned.
    return DenseElementsAttr::get(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 1)),
        ArrayRef<bool>(reinterpret_cast<const bool *>(data), numBytes));
  case ScalarTypeCode::i4:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 4)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::i8:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 8)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::ui8:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 8)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::i16:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 16)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::i32:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 32)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::i64:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, IntegerType::get(&ctx, 64)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::f8e4m3fn:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, Float8E4M3FNType::get(&ctx)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::f16:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, Float16Type::get(&ctx)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::bf16:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, BFloat16Type::get(&ctx)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::f32:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, Float32Type::get(&ctx)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::f64:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, Float64Type::get(&ctx)),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::complex32:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, ComplexType::get(Float32Type::get(&ctx))),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  case ScalarTypeCode::complex64:
    return DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(shape, ComplexType::get(Float64Type::get(&ctx))),
        ArrayRef<char>(reinterpret_cast<const char *>(data), numBytes));
  default:
    return getInvalidArgStatus(
        "Unsupported scalar type code to create DenseElementsAttr from "
        "`MemRefValue`: ",
        mtrt::flat::EnumNameScalarTypeCode(code));
  }
}

StatusOr<Attribute>
SubgraphExecutor::convertScalarValueToAttr(MLIRContext &ctx,
                                           ScalarValue *value) {
  ScalarTypeCode code = value->getType().getCode();
  switch (code) {
  case ScalarTypeCode::i1:
    return IntegerAttr::get(IntegerType::get(&ctx, 1), value->get<bool>());
  case ScalarTypeCode::i4:
    return IntegerAttr::get(IntegerType::get(&ctx, 4),
                            APInt(8, value->get<uint8_t>()).trunc(4));
  case ScalarTypeCode::i8:
    return IntegerAttr::get(IntegerType::get(&ctx, 8), value->get<int8_t>());
  case ScalarTypeCode::ui8:
    return IntegerAttr::get(IntegerType::get(&ctx, 8), value->get<uint8_t>());
  case ScalarTypeCode::i16:
    return IntegerAttr::get(IntegerType::get(&ctx, 16), value->get<int16_t>());
  case ScalarTypeCode::i32:
    return IntegerAttr::get(IntegerType::get(&ctx, 32), value->get<int32_t>());
  case ScalarTypeCode::i64:
    return IntegerAttr::get(IntegerType::get(&ctx, 64), value->get<int64_t>());
  case ScalarTypeCode::f8e4m3fn:
    return FloatAttr::get(
        Float8E4M3FNType::get(&ctx),
        APFloat(APFloat::Float8E4M3FN(), APInt(8, value->get<uint8_t>())));
  case ScalarTypeCode::f16:
    return FloatAttr::get(
        Float16Type::get(&ctx),
        APFloat(APFloat::IEEEhalf(), APInt(16, value->get<uint16_t>())));
  case ScalarTypeCode::bf16:
    return FloatAttr::get(BFloat16Type::get(&ctx),
                          APFloat(APFloat::BFloat(), value->get<uint16_t>()));
  case ScalarTypeCode::f32:
    return FloatAttr::get(Float32Type::get(&ctx), value->get<float>());
  case ScalarTypeCode::f64:
    return FloatAttr::get(Float64Type::get(&ctx), value->get<double>());
  case ScalarTypeCode::complex32: {
    std::complex<float> v = value->getComplex<float>();
    return ArrayAttr::get(&ctx,
                          {FloatAttr::get(Float32Type::get(&ctx), v.real()),
                           FloatAttr::get(Float32Type::get(&ctx), v.imag())});
  }
  case ScalarTypeCode::complex64: {
    std::complex<double> v = value->getComplex<double>();
    return ArrayAttr::get(&ctx,
                          {FloatAttr::get(Float64Type::get(&ctx), v.real()),
                           FloatAttr::get(Float64Type::get(&ctx), v.imag())});
  }
  default:
    return getInvalidArgStatus(
        "Unsupported scalar type code to create attribute from `ScalarValue`: "
        "{0} ",
        mtrt::flat::EnumNameScalarTypeCode(code));
  }
}

StatusOr<Attribute> SubgraphExecutor::convertMemRefValueToDenseElementsAttr(
    MLIRContext &ctx, Ref<RuntimeClient> client, Ref<Stream> stream,
    MemRefValue *value) {
  PointerInfo pointerInfo = value->getPointerInfo(PointerOwner::internal);
  if (pointerInfo.isHostVisible()) {
    return SubgraphExecutor::getDenseElementsAttrFromHostRawBuffer(
        ctx, value->getScalarType(), value->getShape(), value->getVoidPtr(),
        value->getTotalFootprintInBytes());
  }
  if (pointerInfo.isDeviceVisible()) {
    StatusOr<std::unique_ptr<MemRefValue>> hostMemRef =
        client->copyToHost(*value, stream);
    if (hostMemRef.isError())
      return hostMemRef.getStatus();
    return SubgraphExecutor::getDenseElementsAttrFromHostRawBuffer(
        ctx, (*hostMemRef)->getScalarType(), (*hostMemRef)->getShape(),
        (*hostMemRef)->getVoidPtr(), (*hostMemRef)->getTotalFootprintInBytes());
  }
  return getStatusWithMsg(StatusCode::InternalError,
                          "MemRefValue is neither host not device visible.");
}

StatusOr<llvm::SmallVector<Attribute>>
SubgraphExecutor::convertRuntimeValuesToAttributes(
    MLIRContext &ctx, Ref<RuntimeClient> client, Ref<Stream> stream,
    llvm::SmallVectorImpl<std::unique_ptr<RuntimeValue>> &results) {
  llvm::SmallVector<Attribute, 8> r;
  for (const std::unique_ptr<RuntimeValue> &e : results) {
    if (e->getKind() == RuntimeValue::Kind::Scalar) {
      ScalarValue *scalarValue = static_cast<ScalarValue *>(e.get());
      StatusOr<Attribute> dea =
          SubgraphExecutor::convertScalarValueToAttr(ctx, scalarValue);
      if (dea.isError())
        return dea.getStatus();
      r.push_back(*dea);
    } else {
      MemRefValue *memRefValue = static_cast<MemRefValue *>(e.get());
      StatusOr<Attribute> dea =
          SubgraphExecutor::convertMemRefValueToDenseElementsAttr(
              ctx, client, stream, memRefValue);
      if (dea.isError())
        return dea.getStatus();
      r.push_back(*dea);
    }
  }
  return StatusOr<llvm::SmallVector<Attribute>>(std::move(r));
}
} // namespace

//===----------------------------------------------------------------------===//
// Outlining
//===----------------------------------------------------------------------===//

/// Creates and returns `ModuleOp` which has a single function `main` whose
/// region is cloned from `originalFunc`. The new module is created nested
/// inside the module containing `originalFunc`.
static StatusOr<ModuleOp> outlineFuncToModule(IRRewriter &rewriter,
                                              func::FuncOp originalFunc) {
  ModuleOp parentModule = originalFunc->getParentOfType<ModuleOp>();
  if (!parentModule)
    return getStatusWithMsg(StatusCode::InvalidArgument,
                            "function is not inside a module.");

  // Create the new module nested inside the parent module.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&parentModule.getBodyRegion().front());
  ModuleOp newModule = rewriter.create<ModuleOp>(originalFunc->getLoc());
  Region *moduleRegion = &newModule.getBodyRegion();
  Block *entryBlock = &moduleRegion->getBlocks().front();
  if (!entryBlock)
    return getStatusWithMsg(StatusCode::InvalidArgument,
                            "function doesn't have entry level block.");
  newModule.setName(("const_fold_" + originalFunc.getName()).str());

  rewriter.setInsertionPointToStart(&moduleRegion->getBlocks().front());
  func::FuncOp mainFunc = rewriter.create<func::FuncOp>(
      originalFunc->getLoc(), "main", originalFunc.getFunctionType());
  IRMapping mapping;
  originalFunc.getBody().cloneInto(&mainFunc.getBody(), mapping);
  return newModule;
}

//===----------------------------------------------------------------------===//
// Compilation Pipeline
//===----------------------------------------------------------------------===//
struct SubgraphPipelineOptions : public CLOptionScope {
  SubgraphPipelineOptions()
      : CLOptionScope(CLOptionScope::LocalScope{}),
        deviceOptions(std::make_unique<DeviceOptions>(*this)),
        executorOptions(std::make_unique<ExecutorOptions>(*this)) {}

  Option<std::string> dumpPtxDir{*this, "dump-ptx-dir"};
  ListOption<std::string> generatorBenefit{
      *this, "generator-benefit",
      llvm::cl::desc("A list of 'name:benefit' pairs to adjust generator "
                     "benefits for kernel generation.")};

  std::unique_ptr<DeviceOptions> deviceOptions;
  std::unique_ptr<ExecutorOptions> executorOptions;
};

static void populateSubgraphCompilationPipeline(OpPassManager &pm) {
  pm.addPass(executor::createExecutorGenerateABIWrappersPass(
      executor::ExecutorGenerateABIWrappersPassOptions{
          /*forceUndefOutputArgs=*/true,
      }));

  //===----------------------------------------------------------------------===//
  // To linalg conversion and preprocessing
  //===----------------------------------------------------------------------===//
  {
    OpPassManager &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(mlir::createStablehloToKernelPass());
    funcPM.addPass(mlir::createStablehloToLinalgPass());
    funcPM.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    funcPM.addPass(mtrt::createLinalgElementwiseFusionPass());
    funcPM.addPass(mtrt::createLinalgSimplifyExtractSlicePass());
    funcPM.addPass(mtrt::createTensorExtPadToInsertSlicePass());
    funcPM.addPass(mlir::createCSEPass());
    funcPM.addPass(mlir::createCanonicalizerPass());
  }

  pm.addPass(mtrt::compiler::createKernelSegmentationPass());

  //===----------------------------------------------------------------------===//
  // Kernel Generation
  //===----------------------------------------------------------------------===//
  auto subgraphPipelineOptions = std::make_unique<SubgraphPipelineOptions>();
  const auto &deviceOpts = *subgraphPipelineOptions->deviceOptions;

  kernel::buildTransformIRPipeline(
      pm, mtrt::compiler::getKernelGenClusterAttrName(),
      deviceOpts.computeCapability, deviceOpts.maxSharedMemoryPerBlockKb,
      deviceOpts.maxRegistersPerBlock,
      subgraphPipelineOptions->generatorBenefit);

  // Lower kernel.sort operations to generated merge sort kernels
  pm.addPass(kernel::createLowerKernelSortPass());

  // Populate target information.
  kernel::SetGPUTargetPassOptions setTargetOptions{};
  setTargetOptions.inferTargetFromHost = deviceOpts.shouldInferFromHost;
  setTargetOptions.chip = "sm_" + std::to_string(deviceOpts.computeCapability);
  setTargetOptions.maxRegistersPerBlock = deviceOpts.maxRegistersPerBlock;
  setTargetOptions.maxSharedMemoryPerBlockKb =
      deviceOpts.maxSharedMemoryPerBlockKb;
  pm.addPass(kernel::createSetGPUTargetPass(setTargetOptions));
  pm.addPass(kernel::createAnnotateKernelEntrypointsPass());

  // Run pre-bufferization transformations on all GPU modules. This pass
  // dispatches dynamic pipelines as directed by the GPU module's attributes.
  pm.addNestedPass<gpu::GPUModuleOp>(
      kernel::createDispatchGPUModuleCompilationPass(
          kernel::DispatchGPUModuleCompilationPassOptions{
              kernel::GPUModuleLoweringPhase::PreBufferization}));

  //===----------------------------------------------------------------------===//
  // Bufferization
  //===----------------------------------------------------------------------===//

  // Jointly bufferize all modules.
  plan::PlanBufferizationOptions bufferizationOpts{};
  bufferizationOpts.forceEntrypointsReturnAllocs = true;
  plan::buildPlanBufferizationPipeline(pm, bufferizationOpts);

  // Run post-bufferization transformations on all GPU modules. This pass
  // dispatches dynamic pipelines as directed by the GPU module's attributes.
  pm.addNestedPass<gpu::GPUModuleOp>(
      kernel::createDispatchGPUModuleCompilationPass(
          kernel::DispatchGPUModuleCompilationPassOptions{
              kernel::GPUModuleLoweringPhase::PostBufferization}));

  //===----------------------------------------------------------------------===//
  // Post-bufferization device code lowering.
  //===----------------------------------------------------------------------===//

  // Refine argument types based on uses.
  pm.addPass(kernel::createKernelRefineArgumentLayoutsPass());
  pm.addNestedPass<gpu::GPUModuleOp>(
      kernel::createKernelExpandMemRefArgsPass());

  // Lower code in `gpu.module` to NVVM.
  pm.addNestedPass<gpu::GPUModuleOp>(
      kernel::createDispatchGPUModuleCompilationPass(
          kernel::DispatchGPUModuleCompilationPassOptions{
              kernel::GPUModuleLoweringPhase::LowerToNVVM,
              /*debug=*/subgraphPipelineOptions->dumpPtxDir}));

  //===----------------------------------------------------------------------===//
  // Host Code Lowering
  //===----------------------------------------------------------------------===//
  pm.addPass(createConvertKernelToCUDAPass());
  pm.addPass(createConvertMemRefToCUDAPass());
  pm.addNestedPass<func::FuncOp>(cuda::createCUDAScheduleAsyncPass());
  // Insert host-side syncs after stream scheduling so any tokens/events we
  // create are tied to the final stream assignment of async copies.
  pm.addPass(cuda::createCUDAInsertHostSyncPass());
  addNestedPasses<func::FuncOp>(pm, [](OpPassManager &pm) {
    pm.addPass(mlir::createCSEPass());
    pm.addPass(cuda::createCUDASimplifyStreamWaitPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cuda::createCUDAExpandOpsPass());
  });

  // We can now safely drop any nested 'gpu.module'.
  pm.addPass(createDropNestedModulesPass());

  pm.addPass(createConvertPlanToExecutorPass());
  pm.addPass(executor::createExecutorAllocsToGlobalsPass());

  ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
  cudaToExecutorOpts.indexBitwidth =
      subgraphPipelineOptions->executorOptions->indexBitwidth;
  pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth =
      subgraphPipelineOptions->executorOptions->indexBitwidth;
  mlir::executor::buildExecutorLoweringPipeline(pm, stdToExecOpts);
}

//===----------------------------------------------------------------------===//
// Constant foldable subgraph processing
//===----------------------------------------------------------------------===//

/// This function first outlines `func` to a standalone `ModuleOp`.
/// Outlined standalone `ModuleOp` is compiled and executed by
/// `SubgraphExecutor`, returning results as vector of `Attribute`s. For
/// each result, `arith.constant` op is created inside `func`. Finally,
/// we replace operands of terminator of `func` (i.e. `func.Return`) with the
/// newly created `arith.constant` ops. This way, when `func` is inlined
/// at call site, we get result of constant compilation i.e. number
/// `arith.constant` ops equal to results of `func`.
/// For example,
/// 1. original constant foldable function is outlined by
/// `planOutlineConstantFoldableSubgraphs` pass.
/// ```
/// module {
///  func.func @test(% arg0 : tensor<4xf32>, % arg1 :
///  tensor<4xf32>)->tensor<4xf32> {
///   %0 = call @constant_subgraph(): ()->tensor<4xf32>
///   %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
///   return %0: tensor<4xf32>
///  }
///  func.func private @constant_subgraph() -> tensor<4xf32> attributes
///   {plan.constant_foldable} {
///  %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///  %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
///  %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///  %0 = stablehlo.add % cst_1, % cst_0 : tensor<4xf32>
///  %1 = stablehlo.subtract % 0, % cst_1 : tensor<4xf32>
///  return %1 : tensor<4xf32>
///  }
/// }
/// ```
/// 2. Constant foldable function (private function with
/// `plan.constant_foldable` attribute) i.e. @constant_subgraph is outlined to a
/// module.
/// ```
/// module {
///  func.func @main()->tensor<4xf32> {
///   %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
///   %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///   %0 = stablehlo.add %cst_1, %cst_0 : tensor<4xf32>
///   %1 = stablehlo.subtract %0, %cst_1 : tensor<4xf32>
///  return % 1 : tensor<4xf32>
///  }
/// }
/// 3. Module is compiled and executed by passing it to `compileAndExecute`
/// method of `SubgraphExecutor` class. It returns result of `main` as a
/// vector of `Attribute`s. At this point, work of outlined module is complete.
/// 4. For each returned `Attribute` in 3, one `arith.constant` op is created in
/// function whose execution result `Attribute` represents. In this example, it
/// is @constant_subgraph. Finally, `func.return` op operands in  are replaced
/// with these newly created `arith.constant` ops.
/// module {
///  func.func private @constant_subgraph() -> tensor<4xf32> attributes
///  {plan.constant_foldable} {
///    %cst = arith.constant dense<3.000000e+00> : tensor<4xf32>
///    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
///    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
///    %0 = stablehlo.add %cst_2, %cst_1 : tensor<4xf32>
///    %1 = stablehlo.subtract %0, %cst_2 : tensor<4xf32>
///   return %cst : tensor<4xf32>
///  }
/// }
/// 5. After this stage, when `call @constant_subgraph(): ()->tensor<4xf32>` is
/// inlined and canonicalized, we are left with the result/s of constant folding
/// outlined subgraph.
/// module {
///  func.func @test(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>)
///  ->tensor<4xf32> {
///   %cst = arith.constant dense<3.000000e+00> : tensor<4xf32>
///   return %cst: tensor<4xf32>
///  }
/// }
static Status processConstantFoldableFunc(
    IRRewriter &rewriter, MLIRContext &ctx, Ref<RuntimeClient> client,
    SubgraphExecutor &executor, func::FuncOp func,
    llvm::StringMap<ModuleOp> &funcToOutlinedModuleMap) {
  if (!funcToOutlinedModuleMap.contains(func.getSymName()))
    return getInternalErrorStatus(
        "failed to find outlined module for func: {0}", func.getSymName());
  ModuleOp outlinedModule = funcToOutlinedModuleMap[func.getSymName()];
  Ref<Stream> stream = client->getDevices().front()->getStream();
  auto result = executor.execute(ctx, client, stream, outlinedModule);
  if (result.isError())
    return result.getStatus();

  auto entryBlock = &func.getBody().getBlocks().front();
  if (!entryBlock)
    return getInternalErrorStatus("failed to get entry block for function: {0}",
                                  func->getName().getStringRef());

  auto term = dyn_cast<func::ReturnOp>(entryBlock->getTerminator());
  if (!term)
    return getInternalErrorStatus("failed to get terminator for function: {0}",
                                  func->getName().getStringRef());

  if (term->getNumOperands() != (*result).size())
    return getInternalErrorStatus(
        "Mismatch in result counts. Function has {0} results, but execution "
        "returned {1} results.",
        term->getNumOperands(), result->size());

  llvm::SmallVector<Value, 8> constantOps;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    for (const Attribute e : *result) {
      // Complex scalar is not supported by `arith.constant` op so create
      // `complex.constant` op for such cases. For complex scalar, `e` is
      // an `ArrayAttr` of two `FloatAttr`s where first represents the real part
      // and second represents the imaginary part of a complex number.
      if (auto complexNumArray = dyn_cast<ArrayAttr>(e)) {
        FloatAttr realPart = cast<FloatAttr>(complexNumArray[0]);
        Type resultType = realPart.getType().isF32()
                              ? ComplexType::get(Float32Type::get(&ctx))
                              : ComplexType::get(Float64Type::get(&ctx));
        constantOps.push_back(rewriter
                                  .create<complex::ConstantOp>(term->getLoc(),
                                                               resultType,
                                                               complexNumArray)
                                  .getResult());
      } else {
        constantOps.push_back(
            rewriter
                .create<arith::ConstantOp>(term->getLoc(), cast<TypedAttr>(e))
                .getResult());
      }
    }
  }

  rewriter.modifyOpInPlace(term, [&]() { term->setOperands(constantOps); });

  return Status::getOk();
}

namespace {
class PlanExecuteConstantFoldableSubgraphsPass
    : public ::mtrt::impl::PlanExecuteConstantFoldableSubgraphsPassBase<
          PlanExecuteConstantFoldableSubgraphsPass> {
public:
  using Base::Base;

  PlanExecuteConstantFoldableSubgraphsPass() : Base() {
    dynamicPM = OpPassManager("builtin.module");
    OpPassManager &nestedModulesPM = dynamicPM.nest<ModuleOp>();
    populateSubgraphCompilationPipeline(nestedModulesPM);
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    registerLuaRuntimeExtensions();
    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    dynamicPM.getDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp parentModuleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect constant foldable outlined funcs.
    SmallVector<func::FuncOp> constFoldableFuncs;
    for (func::FuncOp func : parentModuleOp.getOps<func::FuncOp>()) {
      if (func.isPrivate() && func->hasAttr("plan.constant_foldable"))
        constFoldableFuncs.push_back(func);
    }

    // Outline each constant foldable function collected above to a new module
    // inside `parentModuleOp`.
    llvm::StringMap<ModuleOp> funcToOutlinedModuleMap;
    for (func::FuncOp func : constFoldableFuncs) {
      StatusOr<ModuleOp> outlinedModule = outlineFuncToModule(rewriter, func);
      if (outlinedModule.isError()) {
        emitError(parentModuleOp->getLoc())
            << "failed to outline func: " << func->getName()
            << " to module, with error: "
            << outlinedModule.getStatus().getMessage();
        return signalPassFailure();
      }
      funcToOutlinedModuleMap[func.getSymName()] = *outlinedModule;
    }

    // Lower newly added modules to the executor IR. OpPassManager `dynamicPM`
    // is nested to ONLY process module ops nested inside `parentModuleOp` and
    // NOT `parentModuleOp` itself.
    if (failed(runPipeline(dynamicPM, parentModuleOp))) {
      emitError(parentModuleOp->getLoc())
          << "failed to run subgraph compilation pipeline";
      return signalPassFailure();
    }

    StatusOr<Ref<RuntimeClient>> runtimeClient = RuntimeClient::create();
    // Return if runtime client creation failed OR no CUDA device found.
    if (runtimeClient.isError() || (*runtimeClient)->getDevices().empty()) {
      emitError(parentModuleOp->getLoc())
          << "failed to create runtime client: "
          << runtimeClient.getStatus().getMessage();
      return signalPassFailure();
    }

    // Execute `main` functions in compiled module ops inside `parentModuleOp`
    // and create constant/s for results in original function for which nested
    // module was created.
    for (func::FuncOp func : constFoldableFuncs) {
      Status r =
          processConstantFoldableFunc(rewriter, getContext(), *runtimeClient,
                                      executor, func, funcToOutlinedModuleMap);
      if (r.isError()) {
        emitError(func->getLoc()) << "failed to execute constant folding on "
                                  << func.getName() << ": " << r.getMessage();
        return signalPassFailure();
      }
    }

    // Delete newly created modules since they were just for compilation purpose
    // and won't be useful after this pass.
    for (auto &entry : funcToOutlinedModuleMap) {
      rewriter.eraseOp(entry.second);
    }
  }

  SubgraphExecutor executor;
  OpPassManager dynamicPM;
};
} // namespace
