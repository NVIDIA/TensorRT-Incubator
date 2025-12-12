//===- ScatterSchedule.cpp ------------------------------------------------===//
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
/// Schedule for 'kernel.scatter' operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformBuilder.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformSchedules.h"
#include "mlir-kernel/Utils/TilingUtils.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <numeric>

#define DEBUG_TYPE "kernel-scatter-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

using namespace mlir;
using namespace mlir::kernel;

namespace {
/// This schedule handles 'kernel.scatter' operations.
class ScatterSchedule
    : public TransformSchedule<ScatterSchedule, ScatterOp,
                               ScatterScheduleParametersAttr> {
public:
  using TransformSchedule::TransformSchedule;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScatterSchedule)

  StringRef getMnemonic() const override { return "scatter"; }

  /// Returns true if this generator can handle the given linalg op.
  bool isSupported(const TransformScheduleOptions &options,
                   ScatterOp op) const override {
    return true;
  }

  FailureOr<ScatterScheduleParametersAttr>
  decideParameters(ScatterOp rootOp,
                   const TransformScheduleOptions &options) const override {
    FailureOr<SmallVector<ScatterScheduleParametersAttr>> attrs =
        enumerateParameters(rootOp, options);
    if (failed(attrs) || attrs->empty())
      return failure();
    return attrs->front();
  }

  FailureOr<SmallVector<ScatterScheduleParametersAttr>>
  enumerateParameters(ScatterOp rootOp,
                      const TransformScheduleOptions &options) const override;

  FailureOr<Value>
  generateSchedule(TransformIRBuilder &b, ScatterOp rootOp, Value funcHandle,
                   ScatterScheduleParametersAttr parameters,
                   const TransformScheduleOptions &options) const override;
};

} // namespace

template <typename RangeTy>
int64_t ctaVolume(RangeTy &&input) {
  auto range = make_filter_range(std::forward<RangeTy>(input),
                                 [](int64_t x) { return x != 0; });
  return std::accumulate(range.begin(), range.end(), 1, std::multiplies<>());
}

static SmallVector<int64_t> fixupTileShape(ArrayRef<int64_t> iterSpaceShape,
                                           ArrayRef<int64_t> tileShape) {
  SmallVector<int64_t> result;
  for (auto [l, r] : llvm::zip_equal(iterSpaceShape, tileShape))
    result.push_back(l == r ? 0 : r);
  return result;
}

FailureOr<Value> ScatterSchedule::generateSchedule(
    TransformIRBuilder &b, ScatterOp rootOp, Value funcHandle,
    ScatterScheduleParametersAttr parameters,
    const TransformScheduleOptions &options) const {
  MLIRContext *ctx = b.getContext();
  const Type anyOpType = b.anyOpType;

  FailureOr<ArrayAttr> ctaMapping =
      getCTADistributionMappingAttr(rootOp.getLoc(), parameters.getGridShape());
  if (failed(ctaMapping))
    return failure();

  // Forall-to-threads. Fixup the tile shape by replacing no-op tile
  // sizes with zero.
  SmallVector<int64_t> ctaBlockingShape = fixupTileShape(
      parameters.getCtaWorkloadShape(), parameters.getCtaBlockingShape());
  SmallVector<int64_t> threadTileShape = fixupTileShape(
      parameters.getCtaBlockingShape(), parameters.getThreadTileShape());
  FailureOr<ArrayAttr> threadMapping =
      getThreadDistributionMappingAttr(rootOp.getLoc(), threadTileShape);
  if (failed(threadMapping))
    return failure();

  funcHandle =
      b.sequence(funcHandle, [&](TransformIRBuilder &b,
                                 BlockArgument funcHandle) {
         // Match the root.
         Value scatterOpH = b.structuredMatch<ScatterOp>(
             funcHandle,
             {b.getNamedAttr(kRootGenericAttrName, UnitAttr::get(ctx))});

         // Create the top-level `scf.forall` representing the grid.
         Value tiledCTAsForallH{};
         if (!parameters.getGridShape().empty()) {
           transform::TileUsingForallOp tiledCTAs =
               b.tileToForall(scatterOpH, parameters.getGridShape(),
                              transform::NumThreadsSpec());
           tiledCTAs.setMappingAttr(*ctaMapping);
           tiledCTAsForallH = tiledCTAs.getForallOp();
           scatterOpH = tiledCTAs.getTiledOp();
         } else {
           auto nestOp =
               b.create<transform::NestScalarLinalgInForallOp>(scatterOpH);
           tiledCTAsForallH = nestOp.getForallOp();
           scatterOpH = nestOp.getLinalgOp();
         }

         // Create the `scf.for` loop nests.
         if (llvm::any_of(ctaBlockingShape, [](int64_t x) { return x > 0; })) {
           scatterOpH =
               b.create<transform::TileUsingForOp>(scatterOpH, ctaBlockingShape)
                   .getTiledLinalgOp();
         }

         if (llvm::any_of(threadTileShape, [](int64_t x) { return x > 0; })) {
           // Perform tiling canonicalization to enable better fusion when
           // we fuse into forall thread mapping below.
           b.applyPatterns<transform::ApplyTilingCanonicalizationPatternsOp>(
               funcHandle);

           auto threadForall = b.tileToForall(scatterOpH, threadTileShape);
           scatterOpH = threadForall.getTiledOp();
           threadForall.setMappingAttr(*threadMapping);
         }

         auto matchOp = b.structuredMatch<ScatterOp>(funcHandle);
         // The current implementation of `transform.lower_to_loops` only
         // returns the outer most loop (if any loops are created).
         SmallVector<Type, 1> lowerToLoopsType;
         if (!parameters.getGridShape().empty())
           lowerToLoopsType.push_back(anyOpType);
         b.create<transform::LowerToLoopsOp>(lowerToLoopsType, matchOp);

         // Run for/forall interchange.
         b.applyPatterns<transform::ApplyTilingCanonicalizationPatternsOp,
                         transform::ApplyInterchangeForAndForallPatternsOp>(
             funcHandle);

         // Perform cleanup.
         b.applyPatterns<transform::ApplyTilingCanonicalizationPatternsOp,
                         transform::ApplyCanonicalizationPatternsOp>(
             funcHandle);

         //  Post-tiling verification
         b.create<transform::VerifyPostTilingOp>(tiledCTAsForallH);

         b.create<transform::YieldOp>(funcHandle);
       }).getResult(0);

  const int64_t numThreads = ctaVolume(parameters.getCtaShape());

  Value kernelFuncHandle =
      b.sequence(funcHandle, [&](TransformIRBuilder &b,
                                 BlockArgument funcHandle) {
         MLIRContext *ctx = b.getContext();
         auto getMappingAttr = [&](ArrayAttr level) {
           return b.getNamedAttr("mapping", level);
         };

         Value kernelFuncHandle =
             b.create<transform::ForallToKernelOp>(
                  b.structuredMatch<scf::ForallOp>(
                      funcHandle, {getMappingAttr(*ctaMapping)}),
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
                  transform::ApplyTilingCanonicalizationPatternsOp>(
      kernelFuncHandle);

  // In the case where `scf.for` are created, we must hoist subset
  // insert/extract operations in order to avoid repeatedly reading/writing
  // accumulators to memory. We want the iteration arguments to only be the
  // `vector` type, not the tensor type.
  Value forLoopHandle = b.structuredMatch<scf::ForOp>(kernelFuncHandle);
  b.create<transform::ApplyLoopInvariantCodeMotionOp>(forLoopHandle);
  b.create<transform::HoistLoopInvariantSubsetsOp>(forLoopHandle);

  return funcHandle;
}

FailureOr<SmallVector<ScatterScheduleParametersAttr>>
ScatterSchedule::enumerateParameters(
    ScatterOp op, const TransformScheduleOptions &options) const {
  auto updateType = cast<ShapedType>(op.getUpdates().front().getType());
  auto inputType = cast<ShapedType>(op.getInits().front().getType());
  ShapedType indicesType = op.getIndices().getType();
  auto tilingOp = cast<TilingInterface>(*op);

  // TODO: support dynamic shapes
  if (ShapedType::isDynamicShape(updateType.getShape()))
    return failure();

  SmallVector<utils::IteratorType> iteratorTypes =
      tilingOp.getLoopIteratorTypes();

  // Scatter is I/O bound. We don't want the algorithm to choose a CTA shape
  // based on SMEM capacity or RF capacity since the focus should instead be on
  // distributing the workload across the GPU. Therefore, we artificially lower
  // the SMEM and RF limits.
  // TODO: account for I/O distribution in the algorithm itself.
  tiling_utils::TileShapeSelectionConfig tileShapeSelectionConfig;
  tileShapeSelectionConfig.deviceNumSMs = options.numMultiProcessors;
  tileShapeSelectionConfig.registersPerBlockBytes =
      static_cast<uint64_t>(0.5f * options.registersPerBlockLimit * 4);
  tileShapeSelectionConfig.sharedMemoryPerBlockBytes =
      static_cast<uint64_t>(0.5f * options.sharedMemoryPerBlockLimitBytes);

  // Approximate the loop nest information needed by the tile-shape selection
  // model.
  MLIRContext *ctx = op.getContext();
  SmallVector<AffineMap> fakeAccessMaps(
      3, AffineMap::getMultiDimIdentityMap(updateType.getRank(), ctx));

  // If we have at least two dimensions in the iteration space, permute the
  // dimensions of the input map in order to prevent the tile shape selector
  // from thinking accesses to the input can be coalesced.
  if (fakeAccessMaps.front().getNumResults() >= 2) {
    MutableAffineMap fakeInputMap(fakeAccessMaps.front());
    AffineExpr backDim = fakeInputMap.getResults().back();
    AffineExpr frontDim = fakeInputMap.getResults().front();
    fakeInputMap.setResult(fakeInputMap.getNumResults() - 1, frontDim);
    fakeInputMap.setResult(0, backDim);
    fakeAccessMaps[0] = fakeInputMap.getAffineMap();
  }

  SmallVector<Type> fakeOperandTypes = {
      updateType.clone(inputType.getElementType()),
      updateType.clone(indicesType.getElementType()),
      updateType.clone(inputType.getElementType()),
  };

  FailureOr<tiling_utils::TileShapeSelectionResult> tileShapeSelectionResult =
      tiling_utils::simpleGpuTileShapeSelection(
          fakeAccessMaps, iteratorTypes, /*operandTypes=*/fakeOperandTypes,
          op.getNumDpsInits(),
          /*loopRanges=*/updateType.getShape(), tileShapeSelectionConfig);

  SmallVector<int64_t> ctaWorkloadShape, ctaBlockingShape, threadTileShape,
      gridShape, threadBlockShape;

  auto params = kernel::ScatterScheduleParametersAttr::get(
      op->getContext(), options.gpuTargetInfo,
      tileShapeSelectionResult->ctaWorkloadShape,
      tileShapeSelectionResult->ctaBlockingShape,
      tileShapeSelectionResult->threadTileShape,
      tileShapeSelectionResult->gridShape,
      tileShapeSelectionResult->threadBlockShape);
  if (!params)
    return failure();

  return SmallVector<ScatterScheduleParametersAttr>{params};
}

void kernel::registerScatterTransformSchedule(DialectRegistry &registry) {
  mlir::kernel::addTransformScheduleGeneratorExtension<ScatterSchedule>(
      registry);
}

std::unique_ptr<TransformScheduleBase>
kernel::createScatterTransformSchedule(MLIRContext *context,
                                       PatternBenefit benefit) {
  return std::make_unique<ScatterSchedule>(context, benefit);
}
