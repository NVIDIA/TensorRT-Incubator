//===- TilingInterfaceImpl.cpp --------------------------------------------===//
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
/// This file contains the TilingInterface implementation for operations in
/// the Kernel dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic error "-Wunused-parameter"
#endif

#define DEBUG_TYPE "kernel-tiling-interface"
#define DBGV_IMPL(fmt, ...)                                                    \
  do {                                                                         \
    llvm::dbgs() << llvm::formatv(fmt "\n", __VA_ARGS__);                      \
  } while (false)
#define DBGV(fmt, ...) LLVM_DEBUG(DBGV_IMPL(fmt, __VA_ARGS__))

using namespace mlir;
using namespace mlir::kernel;
using namespace mlir::memref;

using llvm::SmallSetVector;

/// Returns the size of the given dimension of the given tensor as a static
/// integer if possible, otherwise returns a Value from `tensor.dim`.
static OpFoldResult getTensorDimSize(OpBuilder &b, Location loc, Value tensor,
                                     int64_t dim) {
  auto type = cast<ShapedType>(tensor.getType());
  assert(type.hasRank() && dim < type.getRank() && "dim is out of bounds");
  if (!ShapedType::isDynamic(type.getDimSize(dim)))
    return b.getIndexAttr(type.getDimSize(dim));
  return b
      .create<tensor::DimOp>(loc, tensor,
                             b.create<arith::ConstantIndexOp>(loc, dim))
      .getResult();
}

/// Return the sizes as static integers.
static SmallVector<int64_t>
getCorrespondingShape(ArrayRef<OpFoldResult> sizes) {
  return llvm::map_to_vector(sizes, [](OpFoldResult size) -> int64_t {
    if (std::optional<int64_t> constInt = getConstantIntValue(size))
      return *constInt;
    return ShapedType::kDynamic;
  });
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

/// Return the volume of "scatter updates", which is just the product of
/// dimensions of the updates tensor shape not including "window"  dimensions.
static int64_t getNumUpdates(ScatterOp op) {
  auto updatesType = cast<ShapedType>(op.getUpdates().front().getType());
  auto updateWindowDims = op.getUpdateWindowDims();
  int64_t numUpdates = 1;
  for (int64_t d : llvm::seq<int64_t>(0, updatesType.getRank())) {
    if (llvm::is_contained(updateWindowDims, d))
      continue;
    if (ShapedType::isDynamic(updatesType.getDimSize(d)))
      return ShapedType::kDynamic;
    numUpdates *= updatesType.getDimSize(d);
  }
  return numUpdates;
}

/// For a given `kernel.scatter` op and loop iteration space offsets and sizes
/// (which are in the "update" tensor space), return the corresponding tile
/// offsets and sizes for the indices tensor.
static LogicalResult
getIndexTilePosition(OpBuilder &b, ScatterOp op, ArrayRef<OpFoldResult> offsets,
                     ArrayRef<OpFoldResult> sizes,
                     SmallVectorImpl<OpFoldResult> &indicesOffsets,
                     SmallVectorImpl<OpFoldResult> &indicesSizes) {
  ShapedType indicesType = op.getIndices().getType();
  ShapedType updatesType = cast<ShapedType>(op.getUpdates().front().getType());

  ArrayRef<int64_t> updateWindowDims = op.getUpdateWindowDims();
  for (int64_t d : llvm::seq<int64_t>(0, updatesType.getRank())) {
    if (!llvm::is_contained(updateWindowDims, d)) {
      indicesSizes.push_back(sizes[d]);
      indicesOffsets.push_back(offsets[d]);
    }
  }

  const int64_t indexVectorDim = op.getIndexVectorDim();
  if (indexVectorDim < indicesType.getRank()) {
    indicesOffsets.insert(indicesOffsets.begin() + indexVectorDim,
                          b.getIndexAttr(0));
    indicesSizes.insert(
        indicesSizes.begin() + indexVectorDim,
        getTensorDimSize(b, op->getLoc(), op.getIndices(), indexVectorDim));
  }
  return success();
}

/// Returns true if the ScatterOp implements a windowed scatter. Example:
/// - Input is tensor<10xf32> and the `stablehlo.scatter` issues two
///   `tensor<3xf32>` updates at different starting indices.
/// This is potential semantic of `stablehlo.scatter` that tends not
/// to be represented in other tensor operator sets (e.g. ONNX scatter
/// variants).  The specific condition is that "full window indices" are not
/// disjoint from the input "full starting indices" (see the last four columns
/// in the exammple given by the spec's diagram at
/// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter).
/// This is equivalent to the "scatter dims to operand dims" != "inserted window
/// dims", which is simpler to check.
static bool isWindowedScatter(ScatterOp op) {
  return op.getScatterDimsToOperandDims() != op.getInsertedWindowDims();
}

/// Retrieve dimensions of the input/result that correspond to the scatter
/// update window dimensions. Note that if the op is a "windowed" scatter,
/// then these dimensions may still be indexed into by the scatter start
/// index.
static SmallVector<int64_t, 4> getInputWindowDims(ScatterOp op) {
  auto inputType = cast<ShapedType>(op.getInits().front().getType());
  ArrayRef<int64_t> insertedWindowDims = op.getInsertedWindowDims();
  ArrayRef<int64_t> inputBatchingDims = op.getInputBatchingDims();
  SmallVector<int64_t, 4> inputWindowDims;
  for (int64_t d : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (!llvm::is_contained(inputBatchingDims, d) &&
        !llvm::is_contained(insertedWindowDims, d))
      inputWindowDims.push_back(d);
  }
  return inputWindowDims;
}

/// Retrieve dimensions of the updates that correspond to the scatter
/// update window dimensions except where they overlap
static SmallVector<int64_t, 4> getUpdatePureWindowDims(ScatterOp op) {
  ArrayRef<int64_t> updateWindowDims = op.getUpdateWindowDims();
  SmallVector<int64_t, 4> inputWindowDims = getInputWindowDims(op);
  ArrayRef<int64_t> scatterDimsToOperandDims = op.getScatterDimsToOperandDims();
  SmallVector<int64_t, 4> updatePureWindowDims;
  for (auto [updateWindowDim, inputWindowDim] :
       llvm::zip_equal(updateWindowDims, inputWindowDims)) {
    if (!llvm::is_contained(scatterDimsToOperandDims, inputWindowDim))
      updatePureWindowDims.push_back(updateWindowDim);
  }
  return updatePureWindowDims;
}

/// Specify the loop iterator types. Recall that the iteration space
/// is the space defined by the "scatter updates" tensor shape.
/// We treat all iterators as "parallel" unless doing so could
/// potentially cause overlapping updates.
////
/// If the dimension is a batch dimension, then it is parallel, no
/// need to check other conditions.
///
/// Otherwise, the dimension is not "batch", then it may be indexed by the
/// scatter start index vector. Now we need to check two things:
///
/// 1. Are the indices unique? If so, proceed to 2. If not, this
/// dimension is a reduction (since otherwise the result is not
/// deterministic).
/// 2. If the indices are unique, then this dimension
/// can be parallel *as long as there is no potential for overlapping
/// windows*. Overlapping windows can occur if this is a "windowed
/// scatter" (see `isWindowedScatter` above).
///
/// A special case is if there is only a single "update" slice (all dimensions
/// of update are "window dimensions"), then we can treat all dimensions as
/// parallel since there is no potential for overlapping updates.
SmallVector<utils::IteratorType> ScatterOp::getLoopIteratorTypes() {
  ShapedType updatesType = cast<ShapedType>(getUpdates().front().getType());

  // Start off by marking everything as "parallel".
  SmallVector<utils::IteratorType> iteratorTypes(updatesType.getRank(),
                                                 utils::IteratorType::parallel);

  ArrayRef<int64_t> scatterBatchDims = getScatterIndicesBatchingDims();
  SmallVector<int64_t, 4> updatePureWindowDims = getUpdatePureWindowDims(*this);
  const int64_t numUpdates = getNumUpdates(*this);
  const bool potentialOverlap = isWindowedScatter(*this);
  DBGV("updatePureWindowDims: {0}, numUpdates: {1}, potentialOverlap: {2}",
       llvm::iterator_range(updatePureWindowDims), numUpdates,
       potentialOverlap);
  if (numUpdates != 1) {
    for (int64_t d : llvm::seq<int64_t>(0, updatesType.getRank())) {
      if (llvm::is_contained(scatterBatchDims, d))
        continue;

      if (llvm::is_contained(updatePureWindowDims, d))
        continue;

      if (!getUniqueIndices() || potentialOverlap) {
        iteratorTypes[d] = utils::IteratorType::reduction;
        continue;
      }
    }
  }

  DBGV("iteratorTypes: {0}", llvm::iterator_range(iteratorTypes));

  return iteratorTypes;
}

/// The iteration domain is identical to the updates tensor shape.
SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &b) {
  ShapedType updatesType = cast<ShapedType>(getUpdates().front().getType());

  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<Range> loopRanges(updatesType.getRank(), {zero, one, one});
  for (auto [idx, dim] : llvm::enumerate(updatesType.getShape())) {
    if (!ShapedType::isDynamic(dim)) {
      loopRanges[idx].size = b.getIndexAttr(dim);
      continue;
    }
    loopRanges[idx].size =
        linalg::createFoldedDimOp(b, getLoc(), getUpdates().front(), idx);
  }

  return loopRanges;
}

FailureOr<TilingResult>
ScatterOp::getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  if (!hasPureTensorSemantics())
    return failure();
  OpFoldResult one = b.getI64IntegerAttr(1);
  SmallVector<Operation *> slices;

  // We defined the iteration space as as the space of the updates tensor,
  // so for the update tensor we just need to slice it directly.
  ShapedType updatesType = cast<ShapedType>(getUpdates().front().getType());
  SmallVector<OpFoldResult> updateStrides(updatesType.getRank(), one);

  assert(updatesType.getRank() == static_cast<int64_t>(offsets.size()));
  auto updateSlices =
      llvm::map_to_vector(getUpdates(), [&](Value update) -> Value {
        auto extractSliceOp = b.create<tensor::ExtractSliceOp>(
            getLoc(), update, offsets, sizes, updateStrides);
        slices.push_back(extractSliceOp);
        return extractSliceOp;
      });

  SmallSetVector<int64_t, 4> updateWindowDims(getUpdateWindowDims().begin(),
                                              getUpdateWindowDims().end());

  // Slice the indices.
  ShapedType indicesType = getIndices().getType();

  SmallVector<OpFoldResult> indicesOffsets;
  SmallVector<OpFoldResult> indicesSizes;
  SmallVector<OpFoldResult> indicesStrides(indicesType.getRank(), one);
  if (failed(getIndexTilePosition(b, *this, offsets, sizes, indicesOffsets,
                                  indicesSizes)))
    return failure();

  assert(indicesType.getRank() == static_cast<int64_t>(indicesSizes.size()) &&
         "mismatched indices sizes");
  TypedValue<RankedTensorType> indicesSlice = b.create<tensor::ExtractSliceOp>(
      getLoc(), getIndices(), indicesOffsets, indicesSizes, indicesStrides);
  slices.push_back(indicesSlice.getDefiningOp());

  // Slice the input. We can slice the batch dimensions
  auto inputType = cast<ShapedType>(getInits().front().getType());
  SmallVector<OpFoldResult> inputStrides(inputType.getRank(), one);
  SmallVector<OpFoldResult> inputOffsets;
  SmallVector<OpFoldResult> inputSizes;
  if (failed(getResultTilePosition(b, 0, offsets, sizes, inputOffsets,
                                   inputSizes)))
    return failure();
  SmallVector<Type> resultTypes;
  auto inputSlices = llvm::map_to_vector(getInits(), [&](Value input) -> Value {
    auto tileType = cast<RankedTensorType>(input.getType())
                        .clone(getCorrespondingShape(inputSizes));
    auto sliceOp = b.create<tensor::ExtractSliceOp>(
        getLoc(), tileType, input, inputOffsets, inputSizes, inputStrides);
    resultTypes.push_back(tileType);
    slices.push_back(sliceOp);
    return sliceOp;
  });

  // Create the tiled scatter op.
  auto tiledScatterOp = b.create<ScatterOp>(
      getLoc(), resultTypes, indicesSlice, updateSlices, inputSlices,
      getUpdateWindowDims(), getInsertedWindowDims(), getInputBatchingDims(),
      getScatterIndicesBatchingDims(), getScatterDimsToOperandDims(),
      getIndexVectorDim(), getIndicesAreSortedAttr() ? true : false,
      getUniqueIndicesAttr() ? true : false);

  IRMapping mapping;
  getUpdateComputation().cloneInto(&tiledScatterOp.getUpdateComputation(),
                                   mapping);

  return TilingResult{{tiledScatterOp},
                      SmallVector<Value>(tiledScatterOp.getResults()),
                      slices};
}

/// Method to return the position of the result tile computed by the
/// tiled operation.
///
/// For operations that return a value (typically a value of type
/// `ShapedType`), the generated tiled computation has to also
/// recompute a replacement for the results of the original operation.
/// The tiled implementation of the operation returns a tile of the
/// result(s). This methods returns information about what part of the
/// result tensor is computed by the tiled implementation. The manner in
/// which these tiles get put together to get the final result is upto
/// the surrounding loop construct. If an operation has no results, (for
/// example an operation that operates only on memrefs), then this method
/// need not be implemented by the operation.
/// - `resultNumber` is the result number of the original operation
///   being processed.
/// - `offsets` provides the offset of the tile in the coordinate system
///   of the original iteration space, i.e., if an iteration space
///   dimension had non-zero offset, it will be included in the offset
///   provided here (as opposed to zero-based offset "relative" to the
///   iteration space).
/// - `sizes` provides the size of the tile.
/// - `resultOffsets` is the offsets of the tile of the result generated
///   by the tiled implementation (returned by value).
/// - `resultSizes` is the size of the tile of the result generated
///   by the tiled implementation (returned by value).
///
/// Note: It is undefined behaviour if there is overlap between the
/// tiles of the result generated by the tiled implementation.
LogicalResult ScatterOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  assert(offsets.size() == sizes.size() && "mismatched offsets/sizes");
  // Note that the offset/sizes for the result tile are the same regardless of
  // the result tile position.
  (void)resultNumber;

  OpFoldResult zero = b.getIndexAttr(0);
  auto inputType = cast<ShapedType>(getInits().front().getType());

  // Compute the offset/size for the "start indices" tile. Some result tile
  // offset/size dimensions are more easily spelled in terms of the indices
  // offsets/sizes.
  SmallVector<OpFoldResult> indicesOffsets;
  SmallVector<OpFoldResult> indicesSizes;
  if (failed(getIndexTilePosition(b, *this, offsets, sizes, indicesOffsets,
                                  indicesSizes)))
    return failure();

  // Retrieve the "pure window dims".
  SmallVector<int64_t, 4> inputWindowDims = getInputWindowDims(*this);
  const bool windowedScatter = isWindowedScatter(*this);

  DBGV("inputWindowDims: {0}", llvm::iterator_range(inputWindowDims));

  for (int64_t d : llvm::seq<int64_t>(0, inputType.getRank())) {
    // Is it a "input/result batching dim"? If so, then the offset/size
    // are derived from the corresponding dimension of the update tensor.
    // We can then find the corresponding scatter indices batching dimension
    // and use it to determine the result tile offset/size in this dimennsion.
    if (auto batchingDimIt = llvm::find(getInputBatchingDims(), d);
        batchingDimIt != getInputBatchingDims().end()) {
      unsigned batchDimIdx =
          std::distance(getInputBatchingDims().begin(), batchingDimIt);
      assert(batchDimIdx < getScatterIndicesBatchingDims().size() &&
             "batchDimIdx is out of bounds");
      unsigned scatterIndicesBatchingDim =
          getScatterIndicesBatchingDims()[batchDimIdx];
      DBGV("resultTileOffset/Size[{0}] is input batch dim [{1}], which is "
           "scatter indices/updates dim = {2}",
           d, batchDimIdx, scatterIndicesBatchingDim);
      resultOffsets.push_back(indicesOffsets[scatterIndicesBatchingDim]);
      resultSizes.push_back(indicesSizes[scatterIndicesBatchingDim]);
      continue;
    }

    // Is this a "input window dim"? If so, then the offset/size is derived
    // from the corresponding dimension of the update tensor as long as this
    // is not indexed by the scatter indices in the case of a windowed scatter
    // operation.
    if (auto windowDimIt = llvm::find(inputWindowDims, d);
        windowDimIt != inputWindowDims.end()) {
      assert(inputWindowDims.size() == getUpdateWindowDims().size());
      unsigned windowDimIdx =
          std::distance(inputWindowDims.begin(), windowDimIt);

      if (!windowedScatter ||
          !llvm::is_contained(this->getScatterDimsToOperandDims(),
                              *windowDimIt)) {
        int64_t windowDim = getUpdateWindowDims()[windowDimIdx];
        DBGV("resultTileOffset/Size[{0}] is input window dim [{1}], which is "
             "update dim = {2}",
             d, windowDimIdx, windowDim);
        resultOffsets.push_back(offsets[windowDim]);
        resultSizes.push_back(sizes[windowDim]);
        continue;
      }
    }

    // Otherwise, we need the full result extent in this dimension.
    DBGV("resultTileOffset/Size[{0}] is not a batch or pure window dim", d);
    resultOffsets.push_back(zero);
    resultSizes.push_back(getTensorDimSize(b, getLoc(), getInits().front(), d));
  }

  return success();
}

FailureOr<TilingResult> ScatterOp::generateResultTileValue(
    OpBuilder &b, [[maybe_unused]] unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(b, offsets, sizes);
}

/// The compiler currently does not do very good integer range optimzations,
/// so this check is a greedy optimization to check whether a value is
/// in-bounds.
static bool isStaticallyInBounds(Value value, OpFoldResult lowerBound,
                                 OpFoldResult upperBound) {
  return ValueBoundsConstraintSet::compare(
             value, ValueBoundsConstraintSet::ComparisonOperator::LT,
             upperBound) &&
         ValueBoundsConstraintSet::compare(
             value, ValueBoundsConstraintSet::ComparisonOperator::GE,
             lowerBound);
}

/// Check that each of 'indices' is >= 0 and < the corresponding dimension size
/// of 'inputTensor'. Return i1 Value indicating whether the indices are
/// in-bounds or not.
static Value generateInBoundsCheck(OpBuilder &b, Location loc,
                                   ValueRange indices, Value inputTensor) {
  Value result =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), true));
  Value idxZero = b.create<arith::ConstantIndexOp>(loc, 0);
  for (auto [idx, coord] : llvm::enumerate(indices)) {
    OpFoldResult dimSize = getTensorDimSize(b, loc, inputTensor, idx);
    if (isStaticallyInBounds(coord, b.getIndexAttr(0), dimSize))
      continue;
    Value ltUB = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, coord,
        getValueOrCreateConstantIndexOp(b, loc, dimSize));
    Value gteZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, coord, idxZero);
    Value inBounds = b.create<arith::AndIOp>(loc, ltUB, gteZero);
    result = b.create<arith::AndIOp>(loc, inBounds, result);
  }
  return result;
}

SmallVector<Value>
ScatterOp::generateScalarImplementationOnTensors(RewriterBase &b, Location loc,
                                                 ValueRange updateIVs,
                                                 ValueRange inputBbArgs) {
  assert(hasPureTensorSemantics() && "expected pure tensor semantics");

  SmallVector<Value> updateScalars =
      llvm::map_to_vector(getUpdates(), [&](Value updateTensor) -> Value {
        return b.create<tensor::ExtractOp>(loc, updateTensor, updateIVs);
      });

  auto inputType = cast<ShapedType>(getInits().front().getType());
  auto updateType = cast<ShapedType>(getUpdates().front().getType());
  auto indexType = cast<ShapedType>(getIndices().getType());

  SmallVector<int64_t> updateScatterDims;
  SmallVector<Value> indexIndices;
  for (int64_t d : llvm::seq<int64_t>(0, updateType.getRank())) {
    if (llvm::is_contained(getUpdateWindowDims(), d))
      continue;
    updateScatterDims.push_back(d);
    indexIndices.push_back(updateIVs[d]);
  }

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  const int64_t indexVectorDim = getIndexVectorDim();
  int64_t indexVectorSize = 1;
  if (indexVectorDim < indexType.getRank()) {
    indexIndices.insert(indexIndices.begin() + indexVectorDim, zero);
    indexVectorSize = indexType.getDimSize(indexVectorDim);
  }

  // Get the index vector.
  SmallVector<Value> indexVectorElements;
  for (int64_t i = 0; i < indexVectorSize; ++i) {
    if (i > 0)
      indexIndices[indexVectorDim] = b.create<arith::ConstantIndexOp>(loc, i);
    indexVectorElements.push_back(b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<tensor::ExtractOp>(loc, getIndices(), indexIndices)));
  }

  SmallVector<Value> fullStartIndex(inputType.getRank(), zero);
  for (int64_t d : llvm::seq<int64_t>(0, inputType.getRank())) {
    auto dStartIt = llvm::find(getScatterDimsToOperandDims(), d);
    if (dStartIt == getScatterDimsToOperandDims().end())
      continue;
    auto dStart = dStartIt - getScatterDimsToOperandDims().begin();
    assert(dStart < static_cast<int64_t>(indexVectorElements.size()) &&
           "dStart is out of bounds");
    fullStartIndex[d] = indexVectorElements[dStart];
  }

  SmallVector<Value> fullBatchingIndex(inputType.getRank(), zero);
  for (int64_t d : llvm::seq<int64_t>(0, inputType.getRank())) {
    auto dBatchingIt = llvm::find(getInputBatchingDims(), d);
    if (dBatchingIt == getInputBatchingDims().end())
      continue;
    auto iBatching = dBatchingIt - getInputBatchingDims().begin();
    int64_t dStart = getScatterIndicesBatchingDims()[iBatching];

    assert(dStart < static_cast<int64_t>(indexIndices.size()) &&
           "dStart is out of bounds");
    fullBatchingIndex[d] = indexIndices[dStart];
  }

  SmallVector<Value> updateWindowIndex;
  for (int64_t d : getUpdateWindowDims())
    updateWindowIndex.push_back(updateIVs[d]);

  SmallVector<Value> fullWindowIndex(inputType.getRank(), zero);
  for (size_t i = 0, wi = 0; i < fullWindowIndex.size(); ++i) {
    if (llvm::is_contained(getInsertedWindowDims(), i) ||
        llvm::is_contained(getInputBatchingDims(), i))
      continue;

    assert(
        wi < updateWindowIndex.size() &&
        "updateWindowIndex is smaller than the number of update window dims");
    fullWindowIndex[i] = updateWindowIndex[wi++];
  }

  AffineExpr d0, d1, d2;
  bindDims(b.getContext(), d0, d1, d2);
  AffineMap sumMap = AffineMap::get(3, 0, d0 + d1 + d2);
  SmallVector<Value> resultIndex;
  for (auto [ai, bi, ci] :
       llvm::zip_equal(fullStartIndex, fullBatchingIndex, fullWindowIndex)) {
    resultIndex.push_back(
        b.create<affine::AffineApplyOp>(loc, sumMap, ValueRange{ai, bi, ci}));
  }

  // Check whether the indices are in-bounds.
  Value inBounds =
      generateInBoundsCheck(b, loc, resultIndex, inputBbArgs.front());
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange(inputBbArgs), inBounds,
                                  /*addThenBlock=*/true, /*addElseBlock=*/true);

  // in-bounds case
  {
    b.setInsertionPointToStart(ifOp.thenBlock());
    // Load original elements.
    SmallVector<Value> inputScalars =
        llvm::map_to_vector(inputBbArgs, [&](Value input) -> Value {
          return b.create<tensor::ExtractOp>(loc, input, resultIndex);
        });

    // Substitute into the update computation.
    IRMapping mapping;

    Region &computation = getUpdateComputation();
    for (auto [idx, arg] : llvm::enumerate(computation.getArguments())) {
      if (idx % 2 == 0) {
        mapping.map(arg, inputScalars[idx / 2]);
        continue;
      }
      mapping.map(arg, updateScalars[idx / 2]);
    }

    // Clone the computation.
    for (Operation &op : computation.front().without_terminator())
      b.clone(op, mapping);

    // Retrieve the scalar which should be inserted into the original
    // (destination) tensor.
    SmallVector<Value> yields;
    for (auto [v, dest] : llvm::zip_equal(
             computation.front().getTerminator()->getOperands(), inputBbArgs)) {
      Value scalar = mapping.lookup(v);
      yields.push_back(
          b.create<tensor::InsertOp>(loc, scalar, dest, resultIndex));
    }
    b.create<scf::YieldOp>(loc, yields);
  }

  // out-of-bounds case
  {
    b.setInsertionPointToStart(ifOp.elseBlock());
    b.create<scf::YieldOp>(loc, inputBbArgs);
  }

  return ifOp.getResults();
}

static LogicalResult createLoopNest(RewriterBase &rewriter, ScatterOp op,
                                    ArrayRef<Range> loopRanges,
                                    SmallVectorImpl<Value> &ivs,
                                    SmallVectorImpl<scf::ForOp> &loops) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = op.getLoc();
  ValueRange iterOperands = op.getInits();
  for (Range loopRange : loopRanges) {
    Value offset =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value size = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value stride =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.stride);
    scf::ForOp loop = rewriter.create<scf::ForOp>(
        loc, offset, size, stride, iterOperands,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          iterOperands = args;
          ivs.push_back(iv);
          b.create<scf::YieldOp>(loc, args);
        });
    loops.push_back(loop);
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return success();
}

FailureOr<LowerToLoopsResult> ScatterOp::lowerToLoops(RewriterBase &rewriter) {
  if (!hasPureTensorSemantics())
    return failure();
  SmallVector<Range> loopRanges = getIterationDomain(rewriter);
  auto updateType = cast<ShapedType>(getUpdates().front().getType());

  if (updateType.getRank() == 0) {
    rewriter.setInsertionPoint(*this);
    SmallVector<Value> scalarsToInsert = generateScalarImplementationOnTensors(
        rewriter, getLoc(), {}, getInits());
    return LowerToLoopsResult{{}, std::move(scalarsToInsert)};
  }

  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> loops;
  if (failed(createLoopNest(rewriter, *this, loopRanges, ivs, loops)))
    return failure();

  assert(static_cast<int64_t>(ivs.size()) == updateType.getRank() &&
         "mismatch in ivs and updateType");

  rewriter.setInsertionPoint(loops.back().getBody()->getTerminator());
  SmallVector<Value> scalarsToInsert = generateScalarImplementationOnTensors(
      rewriter, getLoc(), ivs, loops.back().getRegionIterArgs());
  rewriter.modifyOpInPlace(loops.back().getBody()->getTerminator(), [&]() {
    loops.back().getBody()->getTerminator()->setOperands(scalarsToInsert);
  });

  // Update all the yielded values in the outer loops.
  for (int64_t idx = static_cast<int64_t>(loops.size()) - 2; idx >= 0; idx--) {
    Operation *terminator = loops[idx].getBody()->getTerminator();
    rewriter.modifyOpInPlace(terminator, [&]() {
      terminator->setOperands(loops[idx + 1]->getResults());
    });
  }

  return LowerToLoopsResult{
      SmallVector<Operation *>(loops.begin(), loops.end()),
      loops.front()->getResults()};
}
