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
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::stablehlo_ext;

template <typename R>
static bool isSeq(R &&range, int64_t start, int64_t size) {
  return llvm::equal(std::forward<R>(range),
                     llvm::seq<int64_t>(start, start + size));
}

/// Common conditions shared by the static and dynamic versions of
/// "isSingleDimSimpleGatherWithImplicitIndexDim".
template <typename OpType>
static bool isSingleDimSimpleGatherWithImplicitIndexDimImpl(OpType op) {
  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType startIndicesType = op.getStartIndices().getType();
  RankedTensorType resultType = op.getType();

  /// Sanity check the expected rank of the result.
  if (resultType.getRank() !=
      operandType.getRank() + startIndicesType.getRank() - 1)
    return false;

  const auto &dims = op.getDimensionNumbers();

  // (C3) Check for implicit size-1 index vector.
  if (dims.getIndexVectorDim() != startIndicesType.getRank())
    return false;

  // (C0) The dimension being gathered is the one that should be collapsed.
  if (dims.getStartIndexMap().size() != 1 ||
      dims.getStartIndexMap() != dims.getCollapsedSliceDims())
    return false;

  // (C2) The offset dims of the result are the trailing dimensions after the
  // start index result dimensions.
  if (!isSeq(dims.getOffsetDims(), startIndicesType.getRank(),
             resultType.getRank() - startIndicesType.getRank()))
    return false;

  return true;
}

std::optional<int64_t>
stablehlo_ext::isSingleDimSimpleGatherWithImplicitIndexDim(GatherOp op) {
  if (!isSingleDimSimpleGatherWithImplicitIndexDimImpl(op))
    return {};

  // (C1) The `slice_sizes` should equal the shape of the operand except
  // along the gather dimension, which is size 1.
  const auto &dims = op.getDimensionNumbers();
  SmallVector<int64_t> expectedSliceSizes(op.getOperand().getType().getShape());
  expectedSliceSizes[dims.getStartIndexMap()[0]] = 1;
  if (!llvm::equal(expectedSliceSizes, op.getSliceSizes()))
    return {};

  if (!dims.getOperandBatchingDims().empty())
    return {};

  return dims.getStartIndexMap().front();
}

std::optional<int64_t>
stablehlo_ext::isSingleDimSimpleGatherWithImplicitIndexDim(
    DynamicGatherOp op, const ShapeInfoCallbacks &shapeInfoCallbacks) {

  // The dynamic gather 3rd parameter is the "slice sizes". We want to
  // ensure that "slice sizes" matches the operand shape in all dimensions
  // except for those dropped using "collapsed_dims".
  TypedValue<RankedTensorType> sliceSizes = op.getSliceSizes();
  RankedTensorType sliceSizesType = sliceSizes.getType();

  if (!isSingleDimSimpleGatherWithImplicitIndexDimImpl(op))
    return {};

  // (C1) The `slice_sizes` should equal the shape of the operand except
  // along the gather dimension, which is size 1.
  const auto &dims = op.getDimensionNumbers();
  if (!dims.getOperandBatchingDims().empty())
    return {};

  for (int64_t i = 0; i < sliceSizesType.getDimSize(0); i++) {
    if (i == dims.getStartIndexMap()[0]) {
      auto one =
          IntegerAttr::get(op.getSliceSizes().getType().getElementType(), 1);
      if (std::optional<bool> isEqualToOne =
              shapeInfoCallbacks.isElementValueEqualToConstant(
                  TensorElementValue(op.getSliceSizes(), i), one)) {
        if (!*isEqualToOne)
          return {};
        continue;
      }
      return {};
    }

    if (std::optional<bool> isEquivalent =
            shapeInfoCallbacks.isElementValueEqualToShapeDimExtent(
                TensorElementValue(op.getSliceSizes(), i),
                TensorShapeDimExtent(op.getOperand(), i))) {
      if (!*isEquivalent)
        return {};
      continue;
    }

    return {};
  }

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

  if (!dims.getOperandBatchingDims().empty())
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

  if (!op.getDimensionNumbers().getOperandBatchingDims().empty())
    return {};

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

  if (!op.getDimensionNumbers().getOperandBatchingDims().empty())
    return {};

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
  if (!scatterOp.getScatterDimensionNumbers()
           .getScatterIndicesBatchingDims()
           .empty())
    return {};

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

Value stablehlo_ext::createCollapsingReshape(
    OpBuilder &b, Location loc, Value input,
    ArrayRef<ReassociationIndices> reassociation) {
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());

  std::vector<int64_t> newShape(reassociation.size());
  for (auto [idx, re] : llvm::enumerate(reassociation)) {
    int64_t dim = 1;
    for (int64_t i : re) {
      if (inputType.isDynamicDim(i)) {
        dim = ShapedType::kDynamic;
        break;
      }
      dim *= inputType.getDimSize(i);
    }
    newShape[idx] = dim;
  }
  auto resultType = inputType.clone(newShape);

  assert(inputType.getRank() > resultType.getRank() &&
         "input rank should be > result rank");
  assert(static_cast<int64_t>(reassociation.size()) == resultType.getRank() &&
         "invalid reassociation indices");

  if (resultType.hasStaticShape())
    return b.create<stablehlo::ReshapeOp>(loc, resultType, input);

  // Calculate the shape.
  Type i32Type = b.getI32Type();
  RankedTensorType i32ScalarTensorType = RankedTensorType::get({}, i32Type);
  SmallVector<Value> components;
  for (const ReassociationIndices &indices : reassociation) {
    Value vol = b.create<stablehlo::ConstantOp>(
        loc,
        DenseElementsAttr::get(i32ScalarTensorType, static_cast<int32_t>(1)));
    for (int64_t index : indices) {
      Value extent = b.create<stablehlo::GetDimensionSizeOp>(
          loc, i32ScalarTensorType, input, index);
      vol = b.create<stablehlo::MulOp>(loc, vol, extent);
    }
    components.push_back(b.create<stablehlo::ReshapeOp>(
        loc, i32ScalarTensorType.clone({1}), vol));
  }
  Value shape = b.create<stablehlo::ConcatenateOp>(
      loc, i32ScalarTensorType.clone({resultType.getRank()}), components,
      /*dimension=*/0);
  return b.create<stablehlo::DynamicReshapeOp>(loc, resultType, input, shape);
}

Value stablehlo_ext::createExpandingReshape(
    OpBuilder &b, Location loc, RankedTensorType resultType, Value input,
    ArrayRef<ReassociationIndices> reassociation) {
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());

  assert(inputType.getRank() < resultType.getRank() &&
         "input rank should be > result rank");
  assert(static_cast<int64_t>(reassociation.size()) == inputType.getRank() &&
         "invalid reassociation indices");

  if (resultType.hasStaticShape())
    return b.create<stablehlo::ReshapeOp>(loc, resultType, input);

  // Calculate the shape.
  Type i32Type = b.getI32Type();
  RankedTensorType i32ScalarTensorType = RankedTensorType::get({}, i32Type);
  SmallVector<Value> components;
  for (auto [inputDim, resultIndices] : llvm::enumerate(reassociation)) {

    // Calculate how many dynamic dimensions are in the group. This function
    // only supports up to 1 dynamic dimension in each group, otherwise we can't
    // calculate the shape.
    int64_t numDynamicInGroup = 0;
    int64_t divisor = 1;
    for (int64_t resultDim : resultIndices) {
      if (resultType.isDynamicDim(resultDim)) {
        numDynamicInGroup += 1;
        continue;
      }
      divisor *= resultType.getDimSize(resultDim);
    }
    assert(numDynamicInGroup <= 1 && "invalid reshape configuration requested");

    for (int64_t resultDim : resultIndices) {
      if (!resultType.isDynamicDim(resultDim)) {
        components.push_back(b.create<stablehlo::ConstantOp>(
            loc, DenseElementsAttr::get(
                     i32ScalarTensorType.clone({1}),
                     static_cast<int32_t>(resultType.getDimSize(resultDim)))));
        continue;
      }

      Value extent = b.create<stablehlo::GetDimensionSizeOp>(
          loc, i32ScalarTensorType, input, inputDim);

      if (resultIndices.size() == 1 || divisor == 1) {
        components.push_back(b.create<stablehlo::ReshapeOp>(
            loc, i32ScalarTensorType.clone({1}), extent));
        continue;
      }

      // In the case where we are factoring out multiple constant dimensions, we
      // divide by the product of the other dimensions to get the expected
      // extent.
      extent = b.create<stablehlo::DivOp>(
          loc, extent,
          b.create<stablehlo::ConstantOp>(
              loc, DenseElementsAttr::get(i32ScalarTensorType,
                                          static_cast<int32_t>(divisor))));
      components.push_back(b.create<stablehlo::ReshapeOp>(
          loc, i32ScalarTensorType.clone({1}), extent));
    }
  }

  Value shape = b.create<stablehlo::ConcatenateOp>(
      loc, i32ScalarTensorType.clone({resultType.getRank()}), components,
      /*dimension=*/0);
  return b.create<stablehlo::DynamicReshapeOp>(loc, resultType, input, shape);
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

  if (!scatterOp.getScatterDimensionNumbers()
           .getScatterIndicesBatchingDims()
           .empty())
    return {};

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
  auto type = mlir::cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> newShape{type.getShape()};

  // Create an initial identity reassociation. We will update this as we insert
  // the degenerate dimensions.
  SmallVector<ReassociationIndices> reassociation;
  for (unsigned i = 0; i < newShape.size(); i++) {
    ReassociationIndices &back = reassociation.emplace_back();
    back.push_back(i);
  }

  for (int64_t dim : dimsToInsert) {
    newShape.insert(newShape.begin() + dim, 1);

    if (type.getRank() == 0)
      continue;

    /// Calculate which reassociation group this new degenerate dimension
    /// belongs to and where the degenerate dimension should be inserted.
    unsigned reassociationGroupIdx = 0;
    unsigned insertionPositionWithinGroup = 0;
    for (auto [idx, re] : llvm::enumerate(reassociation)) {
      if (reassociationGroupIdx + re.size() > static_cast<unsigned>(dim)) {
        insertionPositionWithinGroup = dim - reassociationGroupIdx;
        reassociationGroupIdx = idx;
        break;
      }
      reassociationGroupIdx += re.size();
    }
    reassociationGroupIdx =
        std::min<unsigned>(reassociationGroupIdx, reassociation.size() - 1);

    assert(reassociationGroupIdx < reassociation.size() &&
           "invalid reassociation group");

    reassociation[reassociationGroupIdx].insert(
        reassociation[reassociationGroupIdx].begin() +
            insertionPositionWithinGroup,
        reassociation[reassociationGroupIdx][insertionPositionWithinGroup]);
    // Update all indices to the right of where we inserted, for all groups.
    for (int64_t &d :
         llvm::MutableArrayRef<int64_t>(reassociation[reassociationGroupIdx])
             .drop_front(insertionPositionWithinGroup + 1))
      d += 1;

    for (ReassociationIndices &other :
         llvm::MutableArrayRef<ReassociationIndices>(reassociation)
             .drop_front(reassociationGroupIdx + 1)) {
      for (int64_t &d : other)
        d += 1;
    }
  }

  auto newType = RankedTensorType::get(newShape, type.getElementType());

  return createExpandingReshape(b, loc, newType, tensor, reassociation);
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
  auto indicesType = cast<RankedTensorType>(indices.getType());
  int64_t indicesRank = indicesType.getRank();

  if (indicesRank == 2)
    return indices;
  if (indicesRank == 1)
    return insertDegenerateDimensions(b, loc, indices, {0});

  // Insert reshape to collapse all outer dimensions of `Indices`.
  SmallVector<ReassociationIndices> reassociation{
      llvm::to_vector<2>(llvm::seq<int64_t>(0, indicesRank - 1)),
      {indicesRank - 1}};

  return createCollapsingReshape(b, loc, indices, reassociation);
}
