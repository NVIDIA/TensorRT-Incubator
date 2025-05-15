//===- GatherToSlice.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of `stablehlo-gather-to-slice` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_GATHERTOSLICEPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo;

static bool isMonotonic(ArrayRef<int64_t> array) {
  if (array.empty())
    return true;
  int64_t prev = array.front();
  for (int64_t next : array.drop_front()) {
    if (next <= prev)
      return false;
    prev = next;
  }
  return true;
}

/// Returns the pair of offsets and strides for slice-like indices of a gather
/// op.
static FailureOr<std::pair<SmallVector<DenseIntElementsAttr>,
                           SmallVector<DenseIntElementsAttr>>>
isSliceLikeIndices(Value v) {
  DenseIntElementsAttr offset{}, stride{};
  auto indexPattern = m_Op<stablehlo::BroadcastInDimOp>(m_Op<stablehlo::AddOp>(
      m_Constant(&offset),
      m_Op<stablehlo::MulOp>(m_Constant(&stride), m_Op<stablehlo::IotaOp>())));
  if (matchPattern(v, indexPattern))
    return std::make_pair(SmallVector<DenseIntElementsAttr>{offset},
                          SmallVector<DenseIntElementsAttr>{stride});

  SmallVector<DenseIntElementsAttr> offsets{}, strides{};
  auto concatOp = v.getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concatOp || static_cast<int64_t>(concatOp.getDimension()) !=
                       concatOp.getType().getRank() - 1)
    return failure();
  for (Value v : concatOp.getOperands()) {
    if (!matchPattern(v, indexPattern))
      return failure();
    offsets.push_back(offset);
    strides.push_back(stride);
  }
  return std::make_pair(std::move(offsets), std::move(strides));
}

/// Return whether the types and dimension numbers of the given gather op can be
/// interpreted as a `stablehlo.slice` if the indices are also "index-like".
/// Note that this function does not match the indices as being slice-like.
static LogicalResult sliceLikeGather(stablehlo::GatherOp op) {
  stablehlo::GatherDimensionNumbersAttr config = op.getDimensionNumbers();
  RankedTensorType inputType = op.getOperand().getType();
  RankedTensorType resultType = op.getType();
  RankedTensorType indexType = op.getStartIndices().getType();
  ArrayRef<int64_t> collapsedSliceDims = config.getCollapsedSliceDims();
  ArrayRef<int64_t> startIndexMap = config.getStartIndexMap();
  ArrayRef<int64_t> offsetDims = config.getOffsetDims();
  ArrayRef<int64_t> sliceSizes = op.getSliceSizes();

  // The input rank and result rank should match.
  if (inputType.getRank() != resultType.getRank())
    return failure();

  // The `index_vector_dim` should be the last dim in indices.
  if (indexType.getRank() - 1 != config.getIndexVectorDim())
    return failure();

  // The `start_index_map` corresponds to the sliced dims and
  // equal to `collapsed_sliced_dims`. They should be monotonic.
  if (collapsedSliceDims != startIndexMap || !isMonotonic(startIndexMap))
    return failure();

  // The `offset_dims` refers to non-sliced dimensions. Check
  // that the `slice_sizes` will have `1` in all sliced dims, full
  // extent in all other positions.
  if (!isMonotonic(offsetDims))
    return failure();
  if (sliceSizes.size() != offsetDims.size() + startIndexMap.size())
    return failure();
  unsigned offsetDimsPos = 0;
  unsigned startIndexMapPos = 0;
  for (auto [idx, val] : llvm::enumerate(sliceSizes)) {
    // `offset_dims` should be non-sliced dimensions
    if (offsetDims[offsetDimsPos] == static_cast<int64_t>(idx)) {
      if (val != resultType.getDimSize(idx))
        return failure();
      offsetDimsPos++;
      continue;
    }
    // `start_index_map` should refer to sliced dimensions
    if (startIndexMap[startIndexMapPos] != static_cast<int64_t>(idx))
      return failure();
    if (val != 1)
      return failure();
    startIndexMapPos++;
  }

  // The leading dimensions of the indices shape should be
  // equal to the sizes of the sliced dimensions.
  for (auto [idx, extent] : llvm::enumerate(indexType.getShape().drop_back())) {
    // Get the sliced dimension this should correspond to.
    assert(idx < startIndexMap.size() &&
           "index out-of-bounds with respect to start_index_map");
    int64_t slicedResultDim = startIndexMap[idx];
    if (resultType.getDimSize(slicedResultDim) != extent)
      return failure();
  }

  return success();
}

static int64_t getIntConstant(DenseIntElementsAttr els) {
  assert(els.isSplat() && "expected constant value with splat elements attr");
  return els.getSplatValue<APInt>().getSExtValue();
}

template <typename T>
Operation *constructSliceFromSliceLikeGather(RewriterBase &rewriter,
                                             stablehlo::GatherOp op,
                                             ArrayRef<T> offsetValues,
                                             ArrayRef<T> strideValues) {
  int64_t rank = op.getType().getRank();
  SmallVector<int64_t> offsets(rank, 0), limits(op.getType().getShape()),
      strides(rank, 1);
  auto cfg = op.getDimensionNumbers();
  for (auto [slicedDim, offsetVal, strideVal] :
       llvm::zip_equal(cfg.getStartIndexMap(), offsetValues, strideValues)) {
    int64_t offset = getIntConstant(offsetVal);
    int64_t stride = getIntConstant(strideVal);
    offsets[slicedDim] = offset;
    strides[slicedDim] = stride;
    limits[slicedDim] =
        offset + (op.getType().getDimSize(slicedDim) - 1) * stride + 1;
  }
  return rewriter.create<stablehlo::SliceOp>(op.getLoc(), op.getOperand(),
                                             offsets, limits, strides);
}

namespace {

struct RewriteSliceLikeGather : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::pair<SmallVector<DenseIntElementsAttr>,
                        SmallVector<DenseIntElementsAttr>>>
        offsetsAndStrides = isSliceLikeIndices(op.getStartIndices());
    if (failed(offsetsAndStrides) || failed(sliceLikeGather(op)))
      return failure();

    rewriter.replaceOp(op,
                       constructSliceFromSliceLikeGather<DenseIntElementsAttr>(
                           rewriter, op, std::get<0>(*offsetsAndStrides),
                           std::get<1>(*offsetsAndStrides)));

    return success();
  }
};

class GatherToSlicePass
    : public stablehlo_ext::impl::GatherToSlicePassBase<GatherToSlicePass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteSliceLikeGather>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply rewrite patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
