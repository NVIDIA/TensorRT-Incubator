//===- GatherScatterUtils.cpp  --------------------------------------------===//
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the stablehlo gather and scatter utils.
///
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/StableHloExt/Utils/GatherScatterUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::stablehlo_ext;

template <typename R>
static bool isSeq(R &&range, int64_t start, int64_t size) {
  return llvm::equal(std::forward<R>(range),
                     llvm::seq<int64_t>(start, start + size));
}

std::optional<int64_t>
stablehlo_ext::isSingleDimSimpleGatherWithImplicitIndexDim(GatherOp op) {
  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType startIndicesType = op.getStartIndices().getType();
  RankedTensorType resultType = op.getType();

  /// Sanity check the expected rank of the result.
  if (resultType.getRank() !=
      operandType.getRank() + startIndicesType.getRank() - 1)
    return {};

  const auto &dims = op.getDimensionNumbers();

  // (C3) Check for implicit size-1 index vector.
  if (dims.getIndexVectorDim() != startIndicesType.getRank())
    return {};

  // (C0) The dimension being gathered is the one that should be collapsed.
  if (dims.getStartIndexMap().size() != 1 ||
      dims.getStartIndexMap() != dims.getCollapsedSliceDims())
    return {};

  // (C1) The `slice_sizes` should equal the shape of the operand except
  // along the gather dimension, which is size 1.
  SmallVector<int64_t> expectedSliceSizes(op.getOperand().getType().getShape());
  expectedSliceSizes[dims.getStartIndexMap()[0]] = 1;
  if (!llvm::equal(expectedSliceSizes, op.getSliceSizes()))
    return {};

  // (C2) The offset dims of the result are the trailing dimensions after the
  // start index result dimensions.
  if (!isSeq(dims.getOffsetDims(), startIndicesType.getRank(),
             resultType.getRank() - startIndicesType.getRank()))
    return {};
  return dims.getStartIndexMap().front();
}

std::optional<int64_t>
stablehlo_ext::isSingleDimSimpleGatherWithExplicitIndexDim(GatherOp op) {
  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType startIndicesType = op.getStartIndices().getType();
  RankedTensorType resultType = op.getType();

  /// Sanity check the expected rank of the result.
  if (resultType.getRank() !=
      operandType.getRank() + startIndicesType.getRank() - 2)
    return {};

  const auto &dims = op.getDimensionNumbers();

  // (C3) Check for explicit size-1 index vector.
  if (dims.getIndexVectorDim() != startIndicesType.getRank() - 1 ||
      startIndicesType.getDimSize(dims.getIndexVectorDim()) != 1)
    return {};

  // (C0) The dimension being gathered is the one that should be collapsed.
  if (dims.getStartIndexMap().size() != 1 ||
      dims.getStartIndexMap() != dims.getCollapsedSliceDims())
    return {};

  // (C1) The `slice_sizes` should equal the shape of the operand except
  // along the gather dimension, which is size 1.
  SmallVector<int64_t> expectedSliceSizes(op.getOperand().getType().getShape());
  expectedSliceSizes[dims.getStartIndexMap()[0]] = 1;
  if (!llvm::equal(expectedSliceSizes, op.getSliceSizes()))
    return {};

  // (C2) The offset dims of the result are the trailing dimensions after the
  // start index result dimensions.
  if (!isSeq(dims.getOffsetDims(), startIndicesType.getRank() - 1,
             resultType.getRank() - startIndicesType.getRank() + 1))
    return {};
  return dims.getStartIndexMap().front();
}

bool stablehlo_ext::isSimpleLeadingMultiDimGather(stablehlo::GatherOp op) {
  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType indicesType = op.getStartIndices().getType();
  RankedTensorType resultType = op.getType();
  if (op.getDimensionNumbers().getIndexVectorDim() != indicesType.getRank() - 1)
    return false;

  int64_t indexVectorSize = indicesType.getShape().back();

  // Sanity check the expected rank.
  if (resultType.getRank() !=
      operandType.getRank() - indexVectorSize + indicesType.getRank() - 1)
    return false;

  // Check index vector size.
  if (indexVectorSize < 2)
    return false;

  // (C2) Check slice shape.
  ArrayRef<int64_t> sliceShape = op.getSliceSizes();
  auto slicePrefix = sliceShape.take_front(indexVectorSize);
  if (slicePrefix.front() != 1 || !llvm::all_equal(slicePrefix) ||
      !llvm::equal(sliceShape.drop_front(indexVectorSize),
                   operandType.getShape().drop_front(indexVectorSize)))
    return false;

  // (C0) Check the collapsed_slice_dims is equal to `range(0, indexVectorSize,
  // 1)`.
  if (!llvm::equal(op.getDimensionNumbers().getCollapsedSliceDims(),
                   llvm::seq<int64_t>(0, indexVectorSize)))
    return false;

  // (C1) Check collapsed slice dims is equal to `start_index_map`.
  if (!llvm::equal(op.getDimensionNumbers().getCollapsedSliceDims(),
                   op.getDimensionNumbers().getStartIndexMap()))
    return false;

  // (C3) Check the offset_dims is equal to
  // range(rank(indices)-1, rank(result)).
  if (!llvm::equal(
          op.getDimensionNumbers().getOffsetDims(),
          llvm::seq<int64_t>(indicesType.getRank() - 1, resultType.getRank())))
    return false;

  return true;
}

bool stablehlo_ext::isSimpleLeadingMultiDimGatherWithDegenerateDims(
    stablehlo::GatherOp op) {
  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType indicesType = op.getStartIndices().getType();
  RankedTensorType resultType = op.getType();
  if (op.getDimensionNumbers().getIndexVectorDim() != indicesType.getRank() - 1)
    return false;

  int64_t indexVectorSize = indicesType.getShape().back();

  // Sanity check the expected rank.
  if (resultType.getRank() != operandType.getRank() + indicesType.getRank() - 1)
    return false;

  // Check index vector size.
  if (indexVectorSize < 2)
    return false;

  // (C2) Check slice shape.
  ArrayRef<int64_t> sliceShape = op.getSliceSizes();
  auto slicePrefix = sliceShape.take_front(indexVectorSize);
  if (slicePrefix.front() != 1 || !llvm::all_equal(slicePrefix) ||
      !llvm::equal(sliceShape.drop_front(indexVectorSize),
                   operandType.getShape().drop_front(indexVectorSize)))
    return false;

  // (C0) Check the start_index_map is equal to `range(0, indexVectorSize,
  // 1)`.
  if (!llvm::equal(op.getDimensionNumbers().getStartIndexMap(),
                   llvm::seq<int64_t>(0, indexVectorSize)))
    return false;

  // (C1) Check collapsed slice dims is empty.
  if (!op.getDimensionNumbers().getCollapsedSliceDims().empty())
    return false;

  // (C3) Check the offset_dims is equal to
  // range(rank(result)-rank(operand), rank(result)).
  if (!llvm::equal(
          op.getDimensionNumbers().getOffsetDims(),
          llvm::seq<int64_t>(resultType.getRank() - operandType.getRank(),
                             resultType.getRank())))
    return false;

  return true;
}

bool stablehlo_ext::isCanonicalScatterNd(stablehlo::ScatterOp scatterOp) {
  if (llvm::any_of(scatterOp.getOperandTypes(), [](Type operandType) {
        return !isa<RankedTensorType>(operandType);
      }))
    return false;
  stablehlo::ScatterDimensionNumbersAttr dimsAttrs =
      scatterOp.getScatterDimensionNumbers();
  auto indicesType =
      cast<RankedTensorType>(scatterOp.getScatterIndices().getType());
  auto operandType =
      cast<RankedTensorType>(scatterOp.getInputs().front().getType());
  auto updateType =
      cast<RankedTensorType>(scatterOp.getUpdates().front().getType());
  auto isSeq = [](ArrayRef<int64_t> ar, int64_t start, int64_t end) {
    return llvm::equal(ar, llvm::seq<int64_t>(start, end));
  };
  int64_t indexDepth = indicesType.getDimSize(indicesType.getRank() - 1);
  return indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
         isSeq(dimsAttrs.getUpdateWindowDims(), 1, updateType.getRank()) &&
         isSeq(dimsAttrs.getScatterDimsToOperandDims(), 0,
               indicesType.getDimSize(1)) &&
         isSeq(dimsAttrs.getInsertedWindowDims(), 0, indexDepth) &&
         ((operandType.getRank() - indexDepth) + (indicesType.getRank() - 1)) ==
             updateType.getRank();
}

//===----------------------------------------------------------------------===//
// Code below this point was adapted from the MLIR-HLO project (part of OpenXLA
// project) `xla/mlir_hlo/mhlo/utils/mhlo_scatter_gather_utils.cc` and has the
// original license: Apache License v2.0. See
// https://github.com/openxla/xla/blob/main/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

static SmallVector<int64_t>
getInversePermutation(llvm::ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inversePermutation(permutation.size());
  for (size_t i = 0, e = permutation.size(); i < e; ++i)
    inversePermutation[permutation[i]] = i;
  return inversePermutation;
}

bool stablehlo_ext::isCanonicalScatter(ScatterOp scatterOp) {
  ScatterDimensionNumbersAttr dimsAttrs =
      scatterOp.getScatterDimensionNumbers();
  auto indicesType = scatterOp.getScatterIndices().getType();
  auto operandType =
      mlir::cast<RankedTensorType>(scatterOp.getOperands().front().getType());

  return indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
         dimsAttrs.getInsertedWindowDims().empty() &&
         isSeq(dimsAttrs.getUpdateWindowDims(), 1, operandType.getRank()) &&
         isSeq(dimsAttrs.getScatterDimsToOperandDims(), 0,
               indicesType.getDimSize(1));
}

bool stablehlo_ext::isCanonicalGather(GatherOp gatherOp) {
  const auto &startIndiceShape = gatherOp.getStartIndices().getType();
  const auto &dims = gatherOp.getDimensionNumbers();

  return startIndiceShape.getRank() == 2 && dims.getIndexVectorDim() == 1 &&
         isSeq(dims.getStartIndexMap(), 0, dims.getStartIndexMap().size()) &&
         dims.getCollapsedSliceDims().empty() &&
         isSeq(dims.getOffsetDims(), 1, dims.getOffsetDims().size());
}

// Creates a permutation that shuffles dimensions of `operands` to match the
// order in the index vector.

std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
stablehlo_ext::makeOperandStartIndexPermutations(ArrayRef<int64_t> dimMap,
                                                 int operandRank) {
  SmallVector<int64_t> permutation{dimMap};
  permutation.reserve(operandRank);
  for (int i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(dimMap, i))
      permutation.push_back(i);
  }
  return {permutation, getInversePermutation(permutation)};
}

Value stablehlo_ext::insertDegenerateDimensions(
    OpBuilder &b, Location loc, Value tensor, ArrayRef<int64_t> dimsToInsert) {
  assert(llvm::is_sorted(dimsToInsert) && "dimsToInsert must be sorted");
  if (dimsToInsert.empty())
    return tensor;
  TensorType type = mlir::cast<TensorType>(tensor.getType());
  SmallVector<int64_t> newShape{type.getShape()};
  for (int64_t dim : dimsToInsert)
    newShape.insert(newShape.begin() + dim, 1);
  auto newType = RankedTensorType::get(newShape, type.getElementType());

  return b
      .create<tensor::ExpandShapeOp>(
          loc, newType, tensor,
          *getReassociationIndicesForReshape(type, newType))
      .getResult();
}

// Checks if the indexVectorDim is equal to the rank of `indices`. In that
// case add the trailing 1 dimension. If indexVectorDim is not the innermost
// dimension, insert transpose to make it so.
static Value ensureIndexVectorDimPosition(OpBuilder &b, Location loc,
                                          Value indices,
                                          int64_t indexVectorDim) {
  int64_t indicesRank = mlir::cast<TensorType>(indices.getType()).getRank();
  if (indexVectorDim == indicesRank - 1)
    return indices;
  if (indexVectorDim == indicesRank)
    return insertDegenerateDimensions(b, loc, indices, {indicesRank});

  SmallVector<int64_t> permutation;
  for (int64_t i = 0; i < indicesRank; ++i)
    if (i != indexVectorDim)
      permutation.push_back(i);
  permutation.push_back(indexVectorDim);
  return b.create<TransposeOp>(loc, indices, permutation).getResult();
}

Value stablehlo_ext::canonicalizeStartIndices(OpBuilder &b, Location loc,
                                              Value indices,
                                              int64_t indexVectorDim) {
  indices = ensureIndexVectorDimPosition(b, loc, indices, indexVectorDim);

  int64_t indicesRank = mlir::cast<TensorType>(indices.getType()).getRank();

  if (indicesRank == 2)
    return indices;
  if (indicesRank == 1)
    return insertDegenerateDimensions(b, loc, indices, {0});

  // Insert reshape to collapse all outer dimensions of `Indices`.
  SmallVector<ReassociationIndices> reassociation{
      llvm::to_vector<2>(llvm::seq<int64_t>(0, indicesRank - 1)),
      {indicesRank - 1}};
  return b.create<tensor::CollapseShapeOp>(loc, indices, reassociation)
      .getResult();
}
