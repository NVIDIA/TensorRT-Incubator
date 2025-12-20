//===- CanonicalizeScatter.cpp  ------------------------------------------===//
//
// The canonicalize gather pass logic is adapted from the XLA project
// `xla/mlir_hlo/mhlo/transforms/mhlo_canonicalize_scatter/mhlo_canonicalize_scatter.cc`
// and has the original license: Apache License v2.0. See
// https://github.com/openxla/xla/blob/main/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the `stablehlo-ext-canonicalize-scatter` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/GatherScatterUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::stablehlo_ext;

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZESCATTERPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

static SmallVector<Value> transposeTensors(OpBuilder &b, Location loc,
                                           ValueRange tensors,
                                           ArrayRef<int64_t> permutation) {
  if (llvm::equal(permutation, llvm::seq<int64_t>(0, permutation.size())))
    return tensors;

  auto permutationAttr = b.getDenseI64ArrayAttr(permutation);
  SmallVector<Value> transposedTensors;
  for (Value tensor : tensors)
    transposedTensors.push_back(
        b.create<TransposeOp>(loc, tensor, permutationAttr));

  return transposedTensors;
}

// Transposes updates to align with the dims of operands.
static SmallVector<Value> transposeUpdatesAccordingToScatterDimsMap(
    OpBuilder &b, Location loc, ArrayRef<Value> updates,
    ArrayRef<int64_t> scatterDimsToOperandDims) {
  auto updatesType = cast<RankedTensorType>(updates.front().getType());
  int64_t updatesRank = updatesType.getRank();
  int64_t operandRank = updatesRank - 1;

  // For the updates, we need to add the scatter dimension to the permutation.
  SmallVector<int64_t> permutation{0};
  for (int64_t i : scatterDimsToOperandDims)
    permutation.push_back(i + 1);

  for (int64_t i = 0; i < operandRank; ++i)
    if (!llvm::is_contained(scatterDimsToOperandDims, i))
      permutation.push_back(i + 1);

  return transposeTensors(b, loc, updates, permutation);
}

// Makes window dimensions of `updates` the innermost ones.
static SmallVector<Value> transposeUpdatesToMoveWindowDimensionsInside(
    OpBuilder &b, Location loc, ArrayRef<Value> updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = cast<RankedTensorType>(updates.front().getType());
  int64_t updatesRank = updatesType.getRank();

  // Move update dimensions to the back
  SmallVector<int64_t> permutation;
  for (int i = 0; i < updatesRank; ++i)
    if (!llvm::is_contained(updateWindowDims, i))
      permutation.push_back(i);

  permutation.append(updateWindowDims.begin(), updateWindowDims.end());
  return transposeTensors(b, loc, updates, permutation);
}

static SmallVector<Value> reshapeUpdatesToEnsureSingleScatterDimension(
    OpBuilder &b, Location loc, ValueRange updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = cast<RankedTensorType>(updates.front().getType());
  int64_t updatesRank = updatesType.getRank();

  // Collapse scatter dimensions to 1D if there are more than 1 or prepend a
  // size-1 dimension if there are no explicit scatter dims.
  size_t numScatterDims = updatesRank - updateWindowDims.size();
  if (numScatterDims > 1) {
    SmallVector<ReassociationIndices> reassociation{
        llvm::to_vector<2>(llvm::seq<int64_t>(0, numScatterDims))};
    for (int i = numScatterDims, e = updatesRank; i < e; ++i)
      reassociation.push_back({i});

    return llvm::map_to_vector(updates, [&](Value update) -> Value {
      return createCollapsingReshape(b, loc, update, reassociation);
    });
  }
  if (numScatterDims == 0) {
    return llvm::map_to_vector(updates, [&](Value update) -> Value {
      return insertDegenerateDimensions(
          b, loc, cast<TypedValue<TensorType>>(update), {0});
    });
  }
  return updates;
}

// Inserts size-1 dimensions to get rid of `insertedWindowDims` attribute.
static SmallVector<Value>
reshapeUpdatesToMatchOperandShape(OpBuilder &b, Location loc,
                                  ArrayRef<Value> updates,
                                  ArrayRef<int64_t> insertedWindowDims) {
  size_t numScatterDims = insertedWindowDims.size();
  if (numScatterDims == 0)
    return to_vector(updates);

  SmallVector<int64_t> shiftedScatterDimsToOperandDims;
  for (int64_t i : insertedWindowDims)
    shiftedScatterDimsToOperandDims.push_back(i + 1);

  return llvm::map_to_vector(updates, [&](Value update) -> Value {
    return insertDegenerateDimensions(b, loc,
                                      cast<TypedValue<TensorType>>(update),
                                      shiftedScatterDimsToOperandDims);
  });
}

// Inserts transposes and reshapes to make window/slice dimensions become the
// innermost dimensions of updates. Also insert degenerate size-1 dimensions to
// match the shape of the slice and the shape of the operand.
static SmallVector<Value>
canonicalizeUpdates(OpBuilder &b, Location loc, SmallVector<Value> updates,
                    ArrayRef<int64_t> scatterDimsToOperandDims,
                    ArrayRef<int64_t> updateWindowDims,
                    ArrayRef<int64_t> insertedWindowDims) {
  updates = transposeUpdatesToMoveWindowDimensionsInside(b, loc, updates,
                                                         updateWindowDims);
  updates = reshapeUpdatesToEnsureSingleScatterDimension(b, loc, updates,
                                                         updateWindowDims);
  updates =
      reshapeUpdatesToMatchOperandShape(b, loc, updates, insertedWindowDims);
  return transposeUpdatesAccordingToScatterDimsMap(b, loc, updates,
                                                   scatterDimsToOperandDims);
}

namespace {

/// This pattern rewrites scatter into a transposes, reshapes and a simpler
/// scatter.
///
/// It transposes and reshapes updates, scatterIndices and operands to get to
/// the following characteristics:
///
/// - scatter_indices is a two-dimensional tensor
/// - index_vector_dim is 1
/// - inserted_window_dims is []
/// - update_window_dims is [1, 2, ...]
/// - scatter_dims_to_operand_dims is [0, 1, ...]
struct CanonicalizeScatterPattern : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    if (isCanonicalScatter(scatterOp) ||
        !checkUpdateComputationReturnsUpdateValues(scatterOp) ||
        isCanonicalScatterNd(scatterOp))
      return failure();

    // This pattern does not account for batching dimensions, which are a
    // featurea added in stablehlo v1.1.0.
    if (!scatterOp.getScatterDimensionNumbers()
             .getScatterIndicesBatchingDims()
             .empty())
      return failure();

    Location loc = scatterOp.getLoc();
    ScatterDimensionNumbersAttr dimsAttrs =
        scatterOp.getScatterDimensionNumbers();

    auto operandType =
        cast<RankedTensorType>(scatterOp.getInputs().front().getType());
    int64_t operandRank = operandType.getRank();
    auto [operandPermutation, operandPermutationInverse] =
        makeOperandStartIndexPermutations(
            dimsAttrs.getScatterDimsToOperandDims(), operandRank);

    Value canonicalIndices =
        canonicalizeStartIndices(rewriter, loc, scatterOp.getScatterIndices(),
                                 dimsAttrs.getIndexVectorDim());

    SmallVector<Value> canonicalOperands = transposeTensors(
        rewriter, loc, scatterOp.getInputs(), operandPermutation);

    SmallVector<Value> canonicalUpdates = canonicalizeUpdates(
        rewriter, loc, scatterOp.getUpdates(),
        dimsAttrs.getScatterDimsToOperandDims(),
        dimsAttrs.getUpdateWindowDims(), dimsAttrs.getInsertedWindowDims());

    int64_t scatterIndicesVectorSize =
        cast<TensorType>(canonicalIndices.getType()).getDimSize(1);
    auto canonicalDimsAttrs = ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*updateWindowDims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(1, operandRank + 1)),
        /*insertedWindowDims=*/{}, /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(0, scatterIndicesVectorSize)),
        /*indexVectorDim=*/1);

    auto newScatterOp = rewriter.create<ScatterOp>(
        loc, TypeRange(ValueRange(canonicalOperands)), canonicalOperands,
        canonicalIndices, canonicalUpdates, canonicalDimsAttrs);
    Region &region = newScatterOp.getUpdateComputation();
    rewriter.inlineRegionBefore(scatterOp.getUpdateComputation(), region,
                                region.end());

    SmallVector<Value> transposedResults = transposeTensors(
        rewriter, loc, newScatterOp.getResults(), operandPermutationInverse);
    rewriter.replaceOp(scatterOp, transposedResults);
    return success();
  }
};
} // namespace

// Ensure that there are enough "inserted window dimensions" so that the window
// update and the batch dims access disjoint areas of the result index space.
// This ensures we can map to `tensorr.scatter` or `onnx.scatter_nd`. This must
// be called after ensuring that there is a single scatter batch dimension.
static FailureOr<SmallVector<Value>>
stablehloReshapeScatterUpdatesToAddInsertedDims(OpBuilder &b, Location loc,
                                                ValueRange updates,
                                                int64_t indexDepth,
                                                int64_t inputRank) {
  assert(indexDepth >= 1 && "expected non-zero index depth");
  const size_t numScatterBatchDims = 1;
  RankedTensorType updateType =
      cast<RankedTensorType>(updates.front().getType());
  const int64_t currUpdateSliceRank =
      updateType.getRank() - numScatterBatchDims;
  const int64_t expectedUpdateSliceRank = inputRank - indexDepth;
  const int64_t expectedInsertWindowDims = indexDepth;
  assert(expectedInsertWindowDims >= 1 &&
         "expected positive number of window insert dims");

  // No need to do anything.
  if (expectedUpdateSliceRank == currUpdateSliceRank)
    return llvm::to_vector(updates);

  // Otherwise, we need to drop leading dimensions (hopefully). If leading
  // dimensions are not unit dims, then we can't proceed.
  if (currUpdateSliceRank > expectedUpdateSliceRank) {
    const int64_t dimToDrop = currUpdateSliceRank - expectedUpdateSliceRank;

    if (!llvm::all_equal(
            updateType.getShape().drop_front(1).take_front(dimToDrop)) ||
        updateType.getDimSize(1) != 1)
      return failure();

    RankedTensorType newShape =
        RankedTensorType::Builder(updateType)
            .setShape(updateType.getShape().drop_front(1 + dimToDrop))
            .insertDim(updateType.getDimSize(0), 0);
    return llvm::to_vector(llvm::map_range(updates, [&](Value update) -> Value {
      return b.create<stablehlo::ReshapeOp>(loc, newShape, update);
    }));
  }

  return failure();
}

namespace {

/// Simplify `stablehlo.scatter` to conform with `tensorrt.scatter`.
struct StablehloCanonicalizeScatterToTensorRtScatterNdFormat
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {

    // Only proceed if we are in the StableHLO canonical form. This is covered
    // by the "stablehlo-ext-canonicalize-scatter" pass that runs before this
    // pass. All scatter ops should be in stablehlo canonical form at this
    // point.
    if (!stablehlo_ext::isCanonicalScatter(op) ||
        !checkUpdateComputationReturnsUpdateValues(op) ||
        stablehlo_ext::isCanonicalScatterNd(op))
      return failure();

    RankedTensorType canonicalizedInputType =
        cast<RankedTensorType>(op.getInputs().front().getType());
    RankedTensorType canonicalizedIndexType =
        cast<RankedTensorType>(op.getScatterIndices().getType());
    RankedTensorType updateType =
        cast<RankedTensorType>(op.getUpdates().front().getType());

    // Check that the update slices are "full" slices.
    unsigned updateWindowStart = 1 + canonicalizedIndexType.getShape().back();
    unsigned inputWindowStart = canonicalizedIndexType.getShape().back();
    if (canonicalizedInputType.getShape().drop_front(inputWindowStart) !=
        updateType.getShape().drop_front(updateWindowStart))
      return failure();

    // Reshape the updates if possible.
    int64_t inputRank = canonicalizedInputType.getRank();
    int64_t indexDepth = canonicalizedIndexType.getDimSize(1);
    FailureOr<SmallVector<Value>> canonicalizedUpdates =
        stablehloReshapeScatterUpdatesToAddInsertedDims(
            rewriter, op.getLoc(), op.getUpdates(), indexDepth, inputRank);
    if (failed(canonicalizedUpdates))
      return rewriter.notifyMatchFailure(op, "failed to canonicalize updates");

    // Create the new scatter op.
    auto canonicalizedUpdatesType =
        cast<RankedTensorType>(canonicalizedUpdates->front().getType());
    assert(((canonicalizedInputType.getRank() - indexDepth) +
            (canonicalizedIndexType.getRank() - 1)) ==
               canonicalizedUpdatesType.getRank() &&
           "expected slice size to equal inputRank - index_depth");
    auto newConfig = stablehlo::ScatterDimensionNumbersAttr::get(
        getContext(),
        /*updateWindowDims=*/
        llvm::to_vector(
            llvm::seq<int64_t>(1, canonicalizedUpdatesType.getRank())),
        /*insertedWindowDims=*/
        llvm::to_vector(llvm::seq<int64_t>(0, indexDepth)),
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/
        llvm::to_vector(llvm::seq<int64_t>(0, indexDepth)), 1);
    auto scatterOp = rewriter.create<stablehlo::ScatterOp>(
        op.getLoc(), TypeRange(ValueRange(op.getInputs())), op.getInputs(),
        op.getScatterIndices(), *canonicalizedUpdates, newConfig);
    Region &region = scatterOp.getUpdateComputation();
    rewriter.inlineRegionBefore(op.getUpdateComputation(), region,
                                region.end());
    rewriter.replaceOp(op, scatterOp.getResults());

    scatterOp->setAttr("tensorrt.canonicalized_scatter",
                       rewriter.getUnitAttr());

    return success();
  }
};
} // namespace

void stablehlo_ext::populateCanonicalizeStablehloScatterPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<CanonicalizeScatterPattern,
                  StablehloCanonicalizeScatterToTensorRtScatterNdFormat>(
      patterns.getContext());
}

namespace {

struct CanonicalizeScatterPass
    : stablehlo_ext::impl::CanonicalizeScatterPassBase<
          CanonicalizeScatterPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateCanonicalizeStablehloScatterPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};

} // namespace
