//===- FallbackSchedule.cpp -----------------------------------------------===//
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
/// Schedule for all the other linalg generic ops that do not have a
/// specialized transform schedule.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformBuilder.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformSchedules.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir-kernel/Utils/TilingUtils.h"
#include "mlir-tensorrt-common/Support/ADTExtras.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

#define DEBUG_TYPE "kernel-fallback-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

using namespace mlir;
using namespace mlir::kernel;

namespace {
/// This schedule handles all the other operations.
/// Note: it is not designed to be used in the end-to-end codegen workflow. It
/// is only meant to be used for tiling to distribute work to CTAs, currently
/// for simulation and cost analysis purposes.
class FallbackSchedule
    : public LinalgTransformSchedule<FallbackSchedule,
                                     FallbackScheduleParametersAttr> {
public:
  using LinalgTransformSchedule::LinalgTransformSchedule;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FallbackSchedule)

  StringRef getMnemonic() const override { return "fallback"; }

  /// Returns true if this generator can handle the given linalg op.
  bool isSupported(const TransformScheduleOptions &options,
                   linalg::LinalgOp op) const override;

  FailureOr<FallbackScheduleParametersAttr>
  decideParameters(linalg::LinalgOp rootOp,
                   const TransformScheduleOptions &options) const override {
    FailureOr<SmallVector<FallbackScheduleParametersAttr>> attrs =
        enumerateParameters(rootOp, options);
    if (failed(attrs) || attrs->empty())
      return failure();
    return attrs->front();
  }

  FailureOr<SmallVector<FallbackScheduleParametersAttr>>
  enumerateParameters(linalg::LinalgOp rootOp,
                      const TransformScheduleOptions &options) const override;

  FailureOr<Value>
  generateSchedule(TransformIRBuilder &b, linalg::LinalgOp rootOp,
                   Value funcHandle, FallbackScheduleParametersAttr parameters,
                   const TransformScheduleOptions &options) const override;
};
} // namespace

bool FallbackSchedule::isSupported(const TransformScheduleOptions &options,
                                   linalg::LinalgOp op) const {
  return succeeded(this->decideParameters(op, options));
}

/// Get handles to the fusable operands.
/// The handles are assigned to nullptr if the operand is a scalar or does not
/// have a `TilingInterface`.
static void getFusableOperandHandles(TransformIRBuilder &b,
                                     linalg::LinalgOp rootOp,
                                     Value rootOpHandle,
                                     SmallVectorImpl<Value> &inputHandles,
                                     SmallVectorImpl<Value> &initHandles) {

  // Now, search backward to find more ops to fuse.
  std::deque<std::pair<Operation *, Value>> producers;

  // First, separate the handles for producers of the root op's operands based
  // on whether they are connected to DPS input or inits of the root op.
  llvm::SmallDenseSet<Operation *> seenOps;
  for (OpOperand &operand : rootOp.getOperation()->getOpOperands()) {
    if (rootOp.isScalar(&operand))
      continue;
    auto defOp = operand.get().getDefiningOp<TilingInterface>();
    if (!defOp)
      continue;
    if (seenOps.contains(defOp))
      continue;
    seenOps.insert(defOp);
    if (rootOp.isDpsInput(&operand)) {
      inputHandles.push_back(
          b.operandHandle(rootOpHandle, operand.getOperandNumber()));
      producers.push_back(
          std::make_pair(operand.get().getDefiningOp(), inputHandles.back()));
    } else {
      initHandles.push_back(
          b.operandHandle(rootOpHandle, operand.getOperandNumber()));
    }
  }

  while (!producers.empty()) {
    auto [producerOp, producerHandle] = producers.front();
    producers.pop_front();
    for (OpOperand &operand : producerOp->getOpOperands()) {
      if (!isa<RankedTensorType>(operand.get().getType()))
        continue;
      auto defOp = operand.get().getDefiningOp<TilingInterface>();
      if (!defOp)
        continue;
      if (seenOps.contains(defOp))
        continue;
      seenOps.insert(defOp);
      inputHandles.push_back(
          b.operandHandle(producerHandle, operand.getOperandNumber()));
      producers.push_back(
          std::make_pair(operand.get().getDefiningOp(), inputHandles.back()));
    }
  }
}

/// Returns true if `array` is non-empty and contains at least one non-zero
/// element
static bool nonEmptyAndNonZero(ArrayRef<int64_t> array) {
  return !array.empty() && llvm::any_of(array, [](int64_t x) { return x > 0; });
}

FailureOr<Value> FallbackSchedule::generateSchedule(
    TransformIRBuilder &b, linalg::LinalgOp rootOp, Value funcHandle,
    FallbackScheduleParametersAttr parameters,
    const TransformScheduleOptions &options) const {
  MLIRContext *ctx = b.getContext();

  FailureOr<ArrayAttr> ctaMapping =
      getCTADistributionMappingAttr(rootOp.getLoc(), parameters.getGridShape());
  if (failed(ctaMapping))
    return failure();

  // Fixup the tile shapes by replacing no-op tile sizes with zero.
  SmallVector<int64_t> threadTileShape = tiling_utils::fixupTileShape(
      parameters.getCtaBlockingShape(), parameters.getThreadTileShape());
  SmallVector<int64_t> ctaBlockingTileShape = tiling_utils::fixupTileShape(
      parameters.getCtaWorkloadShape(), parameters.getCtaBlockingShape());

  FailureOr<ArrayAttr> threadMapping =
      getThreadDistributionMappingAttr(rootOp.getLoc(), threadTileShape);
  if (failed(threadMapping))
    return failure();

  funcHandle =
      b.sequence(funcHandle, [&](TransformIRBuilder &b,
                                 BlockArgument funcHandle) {
         // Match the root.
         Value linalgRootOpH = b.structuredMatch<linalg::LinalgOp>(
             funcHandle,

             {b.getNamedAttr(kRootGenericAttrName, UnitAttr::get(ctx))});

         SmallVector<Value> inputProducerHandles;
         SmallVector<Value> initProducerHandles;
         getFusableOperandHandles(b, rootOp, linalgRootOpH,
                                  inputProducerHandles, initProducerHandles);

         // Create the top-level `scf.forall` representing the grid. We need
         // to choose two different ops depending on whether the parameters
         // Create the top-level `scf.forall` representing the grid. We need
         // to choose two different ops depending on whether the parameters
         // specified a no-op tile. The
         // `transform.structured.tile_using_forall` requires a non-trivial
         // tile.
         Value tiledCTAsForallH{};
         if (!parameters.getGridShape().empty()) {
           transform::TileUsingForallOp tiledCTAs =
               b.tileToForall(linalgRootOpH, parameters.getGridShape(),
                              transform::NumThreadsSpec());
           tiledCTAs.setMappingAttr(*ctaMapping);
           tiledCTAsForallH = tiledCTAs.getForallOp();
           linalgRootOpH = tiledCTAs.getTiledOp();
         } else {
           auto nestOp =
               b.create<transform::NestScalarLinalgInForallOp>(linalgRootOpH);
           tiledCTAsForallH = nestOp.getForallOp();
           linalgRootOpH = nestOp.getLinalgOp();
         }

         // Fuse greedily into the `scf.forall`.
         tiledCTAsForallH = b.fuseProducersIntoContainingOp(
             tiledCTAsForallH, inputProducerHandles);
         tiledCTAsForallH = b.fuseProducersIntoContainingOp(
             tiledCTAsForallH, initProducerHandles);

         // Create the `scf.for` loop nests.
         if (nonEmptyAndNonZero(ctaBlockingTileShape)) {
           auto tileUsingForOp = b.create<transform::TileUsingForOp>(
               linalgRootOpH, ctaBlockingTileShape);
           linalgRootOpH = tileUsingForOp.getTiledLinalgOp();

           b.applyPatterns<transform::ApplyTilingCanonicalizationPatternsOp,
                           transform::ApplyFoldTensorEmptyPatternsOp>(
               funcHandle);

           b.fuseProducersIntoContainingOp(tileUsingForOp.getLoops().back(),
                                           inputProducerHandles);
         }

         if (nonEmptyAndNonZero(threadTileShape)) {
           b.applyPatterns<transform::ApplyTilingCanonicalizationPatternsOp,
                           transform::ApplyFoldTensorEmptyPatternsOp>(
               funcHandle);
           auto threadForall = b.tileToForall(linalgRootOpH, threadTileShape);
           linalgRootOpH = threadForall.getTiledOp();
           threadForall.setMappingAttr(*threadMapping);
           b.fuseProducersIntoContainingOp(threadForall.getForallOp(),
                                           inputProducerHandles);
         }

         // Run for/forall interchange.
         b.applyPatterns<transform::ApplyInterchangeForAndForallPatternsOp>(
             funcHandle);

         // Fuse greedily on the root again.
         Value parentForallOp = b.getParentForallOp(linalgRootOpH);
         b.fuseProducersIntoContainingOp(parentForallOp, initProducerHandles);

         // Perform cleanup.
         b.applyPatterns<transform::ApplyCanonicalizationPatternsOp,
                         transform::ApplyTilingCanonicalizationPatternsOp,
                         transform::ApplyFoldTensorEmptyPatternsOp>(funcHandle);

         //  Post-tiling verification
         b.create<transform::VerifyPostTilingOp>(tiledCTAsForallH);
         b.create<transform::YieldOp>(funcHandle);
       }).getResult(0);

  const int64_t numThreads = tiling_utils::ctaVolume(parameters.getCtaShape());

  Value kernelFuncHandle =
      b.sequence(funcHandle, [&](TransformIRBuilder &b,
                                 BlockArgument funcHandle) {
         auto getMappingAttr = [&](ArrayAttr level) {
           return b.getNamedAttr("mapping", level);
         };

         Value kernelFuncHandle =
             b.create<transform::ForallToKernelOp>(
                  b.structuredMatch<scf::ForallOp>(funcHandle,
                                                   getMappingAttr(*ctaMapping)),
                  /*numThreads=*/numThreads,
                  /*reuse_existing_gpu_module=*/false,
                  /*extra_module_attrs=*/
                  ArrayRef<NamedAttribute>{b.getNamedAttr(
                      KernelDialect::getGpuModuleKindAttrName(),
                      kernel::DefaultGPUModuleKindAttr::get(ctx))})
                 .getKernelFunc();

         // Distribute the second `scf.forall` to threads.
         b.create<transform::ForallToSubgroupsOp>(
             b.anyOpType,
             b.structuredMatch<scf::ForallOp>(kernelFuncHandle,
                                              getMappingAttr(*threadMapping)),
             b.getI64IntegerAttr(1));

         b.create<transform::YieldOp>(kernelFuncHandle);
       }).getResult(0);

  // Run CSE + canonicalization patterns.
  b.cse(kernelFuncHandle);
  b.applyPatterns<transform::ApplyCanonicalizationPatternsOp,
                  transform::ApplyTilingCanonicalizationPatternsOp,
                  transform::ApplyFoldTensorEmptyPatternsOp>(kernelFuncHandle);

  if (parameters.getDisableVectorization())
    // Skip Vectorization
    return funcHandle;

  // Perform vectorization.
  if (linalg::hasVectorizationImpl(rootOp.getOperation()) &&
      !linalg::isaConvolutionOpInterface(rootOp))
    kernelFuncHandle =
        b.create<transform::KernelVectorizeChildrenAndApplyPatternsOp>(
            kernelFuncHandle);

  b.cse(kernelFuncHandle);

  // Perform basic vectorization.
  b.applyPatterns<transform::ApplyCanonicalizationPatternsOp,
                  transform::ApplyCastAwayVectorLeadingOneDimPatternsOp,
                  transform::ApplyRankReducingSubviewPatternsOp,
                  transform::ApplyLowerShapeCastPatternsOp,
                  transform::ApplyLowerContractionPatternsOp,
                  transform::ApplyFoldTensorEmptyPatternsOp>(kernelFuncHandle);

  // In the case where `scf.for` are created, we must hoist subset
  // insert/extract operations in order to avoid repeatedly reading/writing
  // accumulators to memory. We want the iteration arguments to only be the
  // `vector` type, not the tensor type.
  Value forLoopHandle = b.structuredMatch<scf::ForOp>(kernelFuncHandle);
  b.create<transform::ApplyLoopInvariantCodeMotionOp>(forLoopHandle);
  b.create<transform::HoistLoopInvariantSubsetsOp>(forLoopHandle);

  // Perform basic vectorization.
  b.applyPatterns<
      transform::ApplyCanonicalizationPatternsOp,
      transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp,
      transform::ApplyRewriteVectorTransferReadToConstantPatternOp,
      transform::ApplyLowerOuterProductPatternsOp,
      transform::ApplyLowerMultiReductionPatternsOp,
      transform::ApplyFoldTensorEmptyPatternsOp,
      transform::ApplyLowerGatherPatternsOp>(kernelFuncHandle);

  return funcHandle;
}

FailureOr<SmallVector<FallbackScheduleParametersAttr>>
FallbackSchedule::enumerateParameters(
    linalg::LinalgOp op, const TransformScheduleOptions &options) const {
  if (ShapedType::isDynamicShape(op.getStaticLoopRanges())) {
    LLVM_DEBUG(DBGS() << "detected dynamic shape in root op: " << op << "\n");
    return failure();
  }

  /// Upstream tiling utilities don't do well when the indexing maps contain
  /// negative multiplication coefficients. This can appear in reversals, e.g.
  /// "(d0) -> (128 - d0)". Such maps will result in asserts/crashes during
  /// tiling.
  /// TODO: introduce a rewrite to make these patterns to use `linalg.index`.
  /// TODO: fix this upstream if possible
  if (linalg_ext::hasNegativeMultiplicationCoefficients(
          op.getIndexingMapsArray()))
    return failure();

  // Generate configurations with both power-of-2 and non-power-of-2 tile sizes
  // If onlyPowerOfTwoConfigs is enabled, generate only power-of-2
  // If enumeratePowerOfTwoConfigs is enabled, generate both; otherwise just
  // non-power-of-2
  SmallVector<FallbackScheduleParametersAttr> allConfigs;

  SmallVector<bool> powerOfTwoOptions;
  if (options.onlyPowerOfTwoConfigs) {
    powerOfTwoOptions = {true}; // ONLY power-of-2
  } else if (options.enumeratePowerOfTwoConfigs) {
    powerOfTwoOptions = {false, true}; // Both non-pow2 and pow2
  } else {
    powerOfTwoOptions = {false}; // ONLY non-power-of-2
  }

  for (bool usePowerOfTwo : powerOfTwoOptions) {
    tiling_utils::TileShapeSelectionConfig tileShapeSelectionConfig;
    tileShapeSelectionConfig.deviceNumSMs = options.numMultiProcessors;
    tileShapeSelectionConfig.registersPerBlockBytes =
        options.registersPerBlockLimit * 4;
    tileShapeSelectionConfig.sharedMemoryPerBlockBytes =
        options.sharedMemoryPerBlockLimitBytes;
    tileShapeSelectionConfig.getPowerOfTwoTiles = usePowerOfTwo;
    tileShapeSelectionConfig.numStages = options.numStages;

    FailureOr<tiling_utils::TileShapeSelectionResult> tileShapes =
        tiling_utils::simpleGpuTileShapeSelection(op, tileShapeSelectionConfig);
    if (failed(tileShapes)) {
      continue; // Try the other configuration
    }

    auto params = kernel::FallbackScheduleParametersAttr::get(
        op->getContext(), options.gpuTargetInfo, tileShapes->ctaWorkloadShape,
        tileShapes->ctaBlockingShape, tileShapes->threadTileShape,
        tileShapes->gridShape, tileShapes->threadBlockShape,
        /*disableVectorization=*/true);
    if (params) {
      allConfigs.push_back(params);
    }
  } // end for loop over power-of-2 options

  if (allConfigs.empty()) {
    return failure();
  }

  // Deduplicate: if both configs have the same ctaWorkloadShape, keep only one
  if (allConfigs.size() == 2) {
    auto shape0 = allConfigs[0].getCtaWorkloadShape();
    auto shape1 = allConfigs[1].getCtaWorkloadShape();

    if (shape0.size() == shape1.size() &&
        std::equal(shape0.begin(), shape0.end(), shape1.begin())) {
      allConfigs.resize(1);
    }
  }

  return allConfigs;
}

//===----------------------------------------------------------------------===//
// Registration Functions
//===----------------------------------------------------------------------===//

void kernel::registerFallbackTransformSchedule(DialectRegistry &registry) {
  mlir::kernel::addTransformScheduleGeneratorExtension<FallbackSchedule>(
      registry);
}

std::unique_ptr<TransformScheduleBase>
kernel::createFallbackTransformSchedule(MLIRContext *context,
                                        PatternBenefit benefit) {
  return std::make_unique<FallbackSchedule>(context, benefit);
}
