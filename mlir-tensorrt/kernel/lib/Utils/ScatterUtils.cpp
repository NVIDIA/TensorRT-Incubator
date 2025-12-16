//===- ScatterUtils.cpp ---------------------------------------------------===//
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
/// Utilities for dealing with `stablehlo.scatter` and `stablehlo.gather`
/// operations.
///
//===----------------------------------------------------------------------===//

#include "mlir-kernel/Utils/ScatterUtils.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

// Checks that `dim` vector is within range [0, `upperBound`) or
//  [0, `upperBound`] if `upperBoundInclusive` is true.
static LogicalResult checkDimInBounds(std::optional<Location> loc, int64_t dim,
                                      int64_t upperBound, StringRef dimName,
                                      StringRef upperBoundName,
                                      bool upperBoundInclusive) {
  StringRef rangeEnd = upperBoundInclusive ? "]" : ")";
  if (dim < 0 || dim >= upperBound + (upperBoundInclusive ? 1 : 0))
    return emitOptionalError(loc, "Expects ", dimName, " to be in range [0, ",
                             upperBoundName, rangeEnd, " i.e. [0, ", upperBound,
                             rangeEnd, ". got: ", dim, ".");
  return success();
}

// Checks that `dims` vector is within range [0, `upperBound`).
static LogicalResult checkDimsInBounds(std::optional<Location> loc,
                                       ArrayRef<int64_t> dims,
                                       int64_t upperBound, StringRef dimsName,
                                       StringRef upperBoundName) {
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= upperBound)
      return emitOptionalError(loc, "Expects each element of ", dimsName,
                               " to be in range [0, ", upperBoundName,
                               ") i.e. [0, ", upperBound, "). got: ", dim, ".");
  }
  return success();
}

static bool verifyCompatibleDims(int64_t dimSize1, int64_t dimSize2) {
  return ShapedType::isDynamic(dimSize1) || ShapedType::isDynamic(dimSize2) ||
         dimSize1 == dimSize2;
}

// Checks if the vector `nums` has duplicates.
static bool isUnique(ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> dimSet;
  dimSet.reserve(nums.size());
  for (auto dim : nums)
    if (!dimSet.insert(dim).second)
      return false;
  return true;
}

// Checks if the `llvm::concat(lhsDims, rhsDims)` has duplicates.
static LogicalResult checkDimsDistinct(std::optional<Location> loc,
                                       ArrayRef<int64_t> lhsDims,
                                       ArrayRef<int64_t> rhsDims,
                                       llvm::StringRef lhs,
                                       llvm::StringRef rhs) {
  llvm::SmallDenseSet<int64_t> dimSet;
  dimSet.reserve(lhsDims.size() + rhsDims.size());
  for (auto dim : llvm::concat<const int64_t>(lhsDims, rhsDims)) {
    if (!dimSet.insert(dim).second)
      return emitOptionalError(loc, "has duplicated dimension from ", lhs,
                               " and ", rhs, ": ", dim);
  }
  return success();
}

static LogicalResult validateScatterDimensionNumbers(
    ShapedType operandType, ArrayRef<int64_t> scatterIndicesShape,
    ShapedType updateType, ArrayRef<int64_t> updateWindowDims,
    ArrayRef<int64_t> insertedWindowDims, ArrayRef<int64_t> inputBatchingDims,
    ArrayRef<int64_t> scatterIndicesBatchingDims,
    ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim,
    std::optional<Location> loc) {
  // scatter_c2
  auto windowSize = updateWindowDims.size() + insertedWindowDims.size() +
                    inputBatchingDims.size();
  if (operandType.getRank() != static_cast<int64_t>(windowSize))
    return emitOptionalError(loc,
                             "Expects rank-of operand to match "
                             "size-of('update_window_dims') + "
                             "size-of('inserted_window_dims') + "
                             "size-of('input_batching_dims') i.e. ",
                             windowSize, " but got ", operandType.getRank(),
                             ".");

  // scatter_c7
  if (!llvm::is_sorted(updateWindowDims))
    return emitOptionalError(loc,
                             "Expects update_window_dims to be sorted; got: [",
                             updateWindowDims, "].");
  if (!isUnique(updateWindowDims))
    return emitOptionalError(loc,
                             "Expects update_window_dims to not repeat; got: [",
                             updateWindowDims, "].");

  // scatter_c8
  if (failed(checkDimsInBounds(loc, updateWindowDims, updateType.getRank(),
                               "update_window_dims", "rank-of('updates')")))
    return failure();

  // scatter_c9
  if (failed(checkDimsDistinct(loc, insertedWindowDims, inputBatchingDims,
                               "inserted_window_dims", "input_batching_dims")))
    return failure();

  // scatter_c10
  if (!llvm::is_sorted(insertedWindowDims))
    return emitOptionalError(
        loc, "Expects inserted_window_dims to be sorted; got: [",
        insertedWindowDims, "].");

  // scatter_c11
  if (failed(checkDimsInBounds(loc, insertedWindowDims, operandType.getRank(),
                               "inserted_window_dims", "rank-of('operand')")))
    return failure();

  // scatter_c12
  if (!llvm::is_sorted(inputBatchingDims))
    return emitOptionalError(loc,
                             "Expects input_batching_dims to be sorted; got: [",
                             inputBatchingDims, "].");

  // scatter_c13
  if (failed(checkDimsInBounds(loc, inputBatchingDims, operandType.getRank(),
                               "input_batching_dims", "rank-of('operand')")))
    return failure();

  // scatter_c14
  if (!isUnique(scatterIndicesBatchingDims))
    return emitOptionalError(
        loc, "Expects scatter_indices_batching_dims to not repeat; got: [",
        scatterIndicesBatchingDims, "].");

  // scatter_c15
  if (failed(checkDimsInBounds(
          loc, scatterIndicesBatchingDims, scatterIndicesShape.size(),
          "scatter_indices_batching_dims", "rank-of('scatter_indices')")))
    return failure();

  // scatter_c16
  if (llvm::is_contained(scatterIndicesBatchingDims, indexVectorDim))
    return emitOptionalError(loc,
                             "expects scatter_indices_batching_dims not to "
                             "include index_vector_dim ",
                             indexVectorDim, ".");

  // scatter_c17
  if (inputBatchingDims.size() != scatterIndicesBatchingDims.size()) {
    return emitOptionalError(
        loc, "input_batching_dims and scatter_indices_batching_dims "
             "should have the same size.");
  }

  // scatter_c18
  for (auto [index, dims] : llvm::enumerate(
           llvm::zip(inputBatchingDims, scatterIndicesBatchingDims))) {
    auto [inputDim, scatterIndicesDim] = dims;
    int64_t inputDimSize = operandType.getDimSize(inputDim);
    int64_t scatterIndicesDimSize = scatterIndicesShape[scatterIndicesDim];
    if (!verifyCompatibleDims(inputDimSize, scatterIndicesDimSize))
      return emitOptionalError(loc, "input_batching_dims[", index,
                               "] and scatter_indices_batching_dims[", index,
                               "] must have compatible sizes, but got ",
                               inputDimSize, " and ", scatterIndicesDimSize,
                               ".");
  }

  // scatter_c19
  if (indexVectorDim == static_cast<int64_t>(scatterIndicesShape.size()) &&
      scatterDimsToOperandDims.size() != 1)
    return emitOptionalError(
        loc, "Scatter op has ", scatterDimsToOperandDims.size(),
        " elements in scatter_dims_to_operand_dims and "
        "the bound of dimension index_vector_dim=",
        indexVectorDim,
        " of scatter_indices is 1. These two numbers must be equal.");

  if (!ShapedType::isDynamic(scatterIndicesShape[indexVectorDim]) &&
      static_cast<int64_t>(scatterDimsToOperandDims.size()) !=
          scatterIndicesShape[indexVectorDim])
    return emitOptionalError(loc, "Scatter op has ",
                             scatterDimsToOperandDims.size(),
                             " elements in scatter_dims_to_operand_dims and "
                             "the bound of dimension index_vector_dim=",
                             indexVectorDim, " of scatter_indices is ",
                             scatterIndicesShape[indexVectorDim],
                             ". These two numbers must be equal.");

  // scatter_c20
  if (failed(checkDimsDistinct(loc, scatterDimsToOperandDims, inputBatchingDims,
                               "scatter_dims_to_operand_dims",
                               "input_batching_dims")))
    return failure();

  // scatter_c21
  if (failed(checkDimsInBounds(
          loc, scatterDimsToOperandDims, operandType.getRank(),
          "scatter_dims_to_operand_dims", "rank-of('operand')")))
    return failure();

  return success();
}

template <typename T>
bool matchesType(Type a, Type b) {
  bool matches = isa<T>(a) && isa<T>(b);
  // Check that expressed type matches for quantized types
  if constexpr (std::is_same<T, quant::QuantizedType>::value) {
    return matches && cast<quant::QuantizedType>(a).getExpressedType() ==
                          cast<quant::QuantizedType>(b).getExpressedType();
  }
  return matches;
}

static LogicalResult verifyReducerShape(std::optional<Location> loc,
                                        Block &block,
                                        ArrayRef<Type> inputTypes) {
  int64_t numInputs = inputTypes.size();

  // all_reduce_c5, reduce_c6, reduce_scatter_c7, reduce_window_c13,
  // scatter_c23, select_and_scatter_c10
  if (static_cast<int64_t>(block.getArguments().size()) != numInputs * 2)
    return emitOptionalError(loc, "Reduction-region must take ", numInputs * 2,
                             " parameters, but takes ",
                             block.getArguments().size(), " parameter(s)");

  // all_reduce_c5, reduce_c6, reduce_scatter_c7, reduce_window_c13,
  // scatter_c23, select_and_scatter_c10
  if (block.getTerminator()->getOperands().empty())
    return emitOptionalError(
        loc, "The reduction-region expected to return some value(s)");

  // all_reduce_c5, reduce_c6, reduce_scatter_c7, reduce_window_c13,
  // scatter_c23, select_and_scatter_c10
  if (static_cast<int64_t>(block.getTerminator()->getOperands().size()) !=
      numInputs)
    return emitOptionalError(loc, "Reduction-region here must produce ",
                             numInputs, " values, but produces ",
                             block.getTerminator()->getOperands().size(),
                             " instead");

  // all_reduce_c5, reduce_c6, reduce_scatter_c7, reduce_window_c13,
  // scatter_c23, select_and_scatter_c10
  ValueRange yieldedValues = block.getTerminator()->getOperands();
  TypeRange accumulatorSubShapes = yieldedValues.getTypes();

  for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    // all_reduce_c5, reduce_c2, reduce_scatter_c7, reduce_window_c13,
    // scatter_c23, select_and_scatter_c10
    if (accumulatorSubShapes[inputIdx] != block.getArgument(inputIdx).getType())
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ", inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);

    // all_reduce_c5, reduce_c2, reduce_scatter_c7, reduce_window_c13,
    // scatter_c23, select_and_scatter_c3, select_and_scatter_c10
    if (accumulatorSubShapes[inputIdx] !=
        block.getArgument(numInputs + inputIdx).getType())
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ",
          numInputs + inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(numInputs + inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);
  }

  return success();
}

LogicalResult mlir::kernel::verifyStablehloLikeScatterOp(
    std::optional<Location> location, ValueRange inputs, Value scatterIndices,
    ValueRange updates, ArrayRef<int64_t> updateWindowDims,
    ArrayRef<int64_t> insertedWindowDims, ArrayRef<int64_t> inputBatchingDims,
    ArrayRef<int64_t> scatterIndicesBatchingDims,
    ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim,
    Region &updateComputation) {
  // Get the first operand and update, since variadic Scatter is not yet
  // implemented
  auto numOperands = inputs.size();
  auto scatterIndicesType = cast<ShapedType>(scatterIndices.getType());

  auto operandTypes = llvm::map_to_vector(
      inputs.getTypes(), [](Type type) { return cast<ShapedType>(type); });
  auto updatesTypes = llvm::map_to_vector(
      updates.getTypes(), [](Type type) { return cast<ShapedType>(type); });

  // scatter_c1
  for (auto operandType : operandTypes)
    if (failed(verifyCompatibleShape(operandTypes[0].getShape(),
                                     operandType.getShape())))
      return emitOptionalError(location,
                               "Not all inputs have compatible shapes.");

  // scatter_c3
  for (auto updateType : updatesTypes)
    if (failed(verifyCompatibleShape(updatesTypes[0].getShape(),
                                     updateType.getShape())))
      return emitOptionalError(location,
                               "Not all updates have compatible shapes.");

  // scatter_c22
  if (failed(checkDimInBounds(location, indexVectorDim,
                              scatterIndicesType.getRank(), "index_vector_dim",
                              "rank-of('scatter_indices')",
                              /*upperBoundInclusive=*/true)))
    return failure();

  SmallVector<Type> inputTypes, initValueTypes;
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    inputTypes.push_back(operandTypes[i].getElementType());
    initValueTypes.push_back(updatesTypes[i].getElementType());
  }
  // scatter_c6, scatter_c23
  if (failed(
          verifyReducerShape(location, updateComputation.front(), inputTypes)))
    return failure();

  // rank-of('updates[i]') == size-of('update_window_dims') +
  // rank-of('scatter_indices') - 1, where 'scatter_indices' is expanded by a
  // trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')
  // for all values of `i`.
  SmallVector<int64_t> expandedScatterIndicesShape =
      llvm::to_vector(scatterIndicesType.getShape());
  if (static_cast<int64_t>(expandedScatterIndicesShape.size()) ==
      indexVectorDim)
    expandedScatterIndicesShape.push_back(1);

  // scatter_c4
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    int64_t expectedUpdatesRank =
        expandedScatterIndicesShape.size() - 1 + updateWindowDims.size();
    if (updatesTypes[i].getRank() != expectedUpdatesRank)
      return emitOptionalError(
          location, "expects updates tensor must be of rank ",
          expectedUpdatesRank,
          " ( == rank-of('scatter_indices') - 1 + "
          "size-of('update_window_dims'), where 'scatter_indices' is "
          "expanded by a trailing 1 dimension if 'index_vector_dim' == "
          "rank-of('scatter_indices')), but got ",
          updatesTypes[i].getRank(), ".");
  }

  // scatter_c2, scatter_c7...scatter_c21
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (failed(validateScatterDimensionNumbers(
            operandTypes[i], expandedScatterIndicesShape, updatesTypes[i],
            updateWindowDims, insertedWindowDims, inputBatchingDims,
            scatterIndicesBatchingDims, scatterDimsToOperandDims,
            indexVectorDim, location)))
      return failure();
  }

  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    auto updatesShape = updatesTypes[i].getShape();
    auto operandShape = operandTypes[i].getShape();

    int64_t insertedDimsSeen = 0;
    int64_t batchingDimsSeen = 0;
    SmallVector<int64_t> maxUpdateSliceSizes;
    const auto dimensionsSize = operandTypes[i].getRank();
    maxUpdateSliceSizes.reserve(dimensionsSize);
    for (int i = 0; i < dimensionsSize; ++i) {
      if (insertedDimsSeen < static_cast<int64_t>(insertedWindowDims.size()) &&
          insertedWindowDims[insertedDimsSeen] == i)
        ++insertedDimsSeen;
      else if (batchingDimsSeen <
                   static_cast<int64_t>(inputBatchingDims.size()) &&
               inputBatchingDims[batchingDimsSeen] == i)
        ++batchingDimsSeen;
      else
        maxUpdateSliceSizes.push_back(operandShape[i]);
    }

    for (int64_t i = 0; i < static_cast<int64_t>(updateWindowDims.size());
         ++i) {
      auto updateWindowDim = updateWindowDims[i];

      if (ShapedType::isDynamic(updatesShape[updateWindowDim]) ||
          ShapedType::isDynamic(maxUpdateSliceSizes[i]))
        continue;

      // scatter_c4
      if (updatesShape[updateWindowDim] > maxUpdateSliceSizes[i]) {
        return emitOptionalError(
            location,
            "expects bounds of the window dimensions of updates to not "
            "exceed the bounds of the corresponding dimensions of operand. "
            "For dimension ",
            updateWindowDim, ", updates bound is ",
            updatesShape[updateWindowDim], ", operand bound is ",
            maxUpdateSliceSizes[i], ".");
      }
    }

    int64_t scatterDimsSeen = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(updatesShape.size()); ++i) {
      bool isUpdateWindowDim = std::binary_search(updateWindowDims.begin(),
                                                  updateWindowDims.end(), i);

      if (isUpdateWindowDim)
        continue;
      if (scatterDimsSeen == indexVectorDim)
        ++scatterDimsSeen;

      // scatter_c4
      if (failed(verifyCompatibleShape(
              updatesShape[i], expandedScatterIndicesShape[scatterDimsSeen])))
        return emitOptionalError(
            location,
            "expects bounds of the scatter dimensions of updates to be "
            "same as the bounds of the corresponding dimensions of scatter "
            "indices. For scatter dimension ",
            i, ", updates bound is ", updatesShape[i],
            " , scatter_indices bound is ",
            expandedScatterIndicesShape[scatterDimsSeen], ".");

      ++scatterDimsSeen;
    }
  }

  return success();
}
