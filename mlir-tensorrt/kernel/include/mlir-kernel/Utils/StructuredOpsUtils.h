//===- StructuredOpsUtils.h -----------------------------------------------===//
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
/// Utilities for dealing with linalg structured ops.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_STRUCTUREDOPSUTILS_H
#define MLIR_TENSORRT_UTILS_STRUCTUREDOPSUTILS_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

namespace detail {
/// Binds to a specific value and matches it or the result of it going through
/// some number of unary elementwise ops.
struct PatternMatcherElwiseValue {
  PatternMatcherElwiseValue(Value val) : value(val) {}
  bool match(Value val) const {
    while (true) {
      if (val == value)
        return true;
      Operation *producer = val.getDefiningOp();
      if (!producer)
        return false;
      if (!producer->hasTrait<OpTrait::Elementwise>() ||
          producer->getNumOperands() != 1 || producer->getNumResults() != 1)
        return false;
      val = producer->getOperand(0);
    }
    return false;
  }
  Value value;
};

inline auto m_ElwiseValue(Value v) { return PatternMatcherElwiseValue(v); }

} // namespace detail

/// Type of a function that checks if an AffineMap passes some condition.
using IndexingMapPredicateFn = std::function<bool(AffineMap)>;

/// Type of a function that checks if an AffineMap passes some condition. This
/// version also includes the LinalgOp incase the predicate needs additional
/// context.
using IndexingMapWithContextPredicateFn =
    std::function<bool(linalg::LinalgOp, AffineMap)>;

/// Type of a function that checks if a LinalgOp body passes some condition.
/// The function receives a yielded value and the block arguments
/// corresponding to the loaded scalars from the inputs/outputs as
/// `bodyIns`/`bodyOuts` respectively.
using BodyPredicateFn =
    std::function<bool(Value yieldedValue, Block::BlockArgListType bodyIns,
                       Block::BlockArgListType bodyOuts)>;

/// A utility for matching linalg operations against some criteria.
struct LinalgOpMatcher {
private:
  std::optional<std::pair<unsigned, unsigned>> numLoopsBounds{};
  std::optional<unsigned> numReductions{};
  std::function<bool(Region &)> regionPredicate{nullptr};
  std::optional<unsigned> numDpsInputs{};
  std::optional<unsigned> numDpsInits{};
  SmallVector<std::function<bool(linalg::LinalgOp)>> opPredicates{};

public:
  struct predicates {
    /// Return AffineMap predicate which matches permutation map.
    static IndexingMapPredicateFn IsPermutation();

    /// Return AffineMap predicate which matches permutation map.
    static IndexingMapPredicateFn
    IsProjectedPermutation(bool allowBroadcasting);

    /// Return AffineMap predicate which matches map that projects out a single
    /// given dimension. The given dimension can be specified as negative with
    /// wrap-around semantics.
    static IndexingMapWithContextPredicateFn
    ProjectsOutSingleDimension(int64_t projectedDimension);

    /// Return AffineMap predicate which matches permutation map.
    static IndexingMapPredicateFn IsIdentity();

    /// Return AffineMap predicate which matches minor identity map.
    static IndexingMapPredicateFn IsMinorIdentity();

    /// Return AffineMap predicate which matches (d0, d1, d2) -> (d0, d1)
    static IndexingMapPredicateFn IsMnkToMn();

    /// Return AffineMap predicate which matches (d0, d1, d2)->(d0, d2).
    static IndexingMapPredicateFn IsMnkToMk();

    /// Return AffineMap predicate which matches (d0, d1, d2)->(d1, d2).
    static IndexingMapPredicateFn IsMnkToNk();

    /// Return a body predicate which matches a yielded value that is
    // identical to an input block arg
    static BodyPredicateFn YieldedValueIsInput(unsigned inputIdx);

    /// Return a body predicate which matches a yielded value that is
    /// a constant scalar
    static BodyPredicateFn YieldedValueIsConstant(Attribute *constValue);

    /// Return a body predicate which matches a yielded value that is
    /// a constant scalar of value zero.
    static BodyPredicateFn YieldedValueIsZero();

    struct AnyOpTypeTag {};

    /// Return a body predicate which matches a yielded value that is
    /// a unary operation of the given input.
    static BodyPredicateFn
    YieldedValueIsUnaryOfInput(unsigned inputIdx, StringRef operationName = "");

    /// Return a body predicate which matches a yielded value that is a
    /// unary operation of the input.
    template <typename OpType>
    static BodyPredicateFn YieldedValueIsUnaryOfInput(unsigned inputIdx) {
      return YieldedValueIsUnaryOfInput(inputIdx, OpType::getOperationName());
    }

    /// Return a body predicate which matches a yielded value that is a binary
    /// operation of the input scalars at the given indices.
    static BodyPredicateFn
    YieldedValueIsBinaryOfInputs(unsigned inputIdxLhs, unsigned inputIdxRhs,
                                 StringRef operationName = "");

    /// Return a body predicate which matches a yielded value that is a binary
    /// operation of the input scalars at the given indices.
    template <typename OpType>
    static BodyPredicateFn YieldedValueIsBinaryOfInputs(unsigned inputIdxLhs,
                                                        unsigned inputIdxRhs) {
      return YieldedValueIsBinaryOfInputs(inputIdxLhs, inputIdxRhs,
                                          OpType::getOperationName());
    }

    /// Return a body predicate which matches a yielded value that is a binary
    /// of one of the inputs and one of the outputs/loop carries.
    static BodyPredicateFn
    YieldedValueIsBinaryOfInputAndOutput(unsigned inputIdx, unsigned outputIdx,
                                         bool isCommutative,
                                         StringRef operationName = "");

    /// Return a body predicate which matches a yielded value that is a binary
    /// of one of the inputs and one of the outputs/loop carries.
    template <typename OpType>
    static BodyPredicateFn
    YieldedValueIsBinaryOfInputAndOutput(unsigned inputIdx, unsigned outputIdx,
                                         bool isCommutative) {
      return YieldedValueIsBinaryOfInputAndOutput(
          inputIdx, outputIdx, isCommutative, OpType::getOperationName());
    }

    /// Return a body predicate which matches a matmul operation.
    /// TODO: add support for other scalar type variations.
    static BodyPredicateFn IsMatMul();
  };

  /// Return a matcher instance which identifies ops equivalent to
  /// `linalg.transpose`.
  static LinalgOpMatcher getTransposeMatcher() {
    return LinalgOpMatcher()
        .setNumReductionLoops(0)
        .setNumDpsInits(1)
        .setNumDpsInputs(1)
        .addRegionMatchRootedAtOutput(
            0, LinalgOpMatcher::predicates::YieldedValueIsInput(0))
        .matchDpsInitIndexingMap(0,
                                 LinalgOpMatcher::predicates::IsPermutation())
        .matchDpsInputIndexingMap(0, LinalgOpMatcher::predicates::IsIdentity());
  }

  /// Return a matcher instance which identifies ops equivalent to
  /// `linalg.transpose`, except the transpose is on the input.
  static LinalgOpMatcher getInputTransposeMatcher() {
    return LinalgOpMatcher()
        .setNumReductionLoops(0)
        .setNumDpsInits(1)
        .setNumDpsInputs(1)
        .addRegionMatchRootedAtOutput(
            0, LinalgOpMatcher::predicates::YieldedValueIsInput(0))
        .matchDpsInitIndexingMap(0, LinalgOpMatcher::predicates::IsIdentity())
        .matchDpsInputIndexingMap(0,
                                  LinalgOpMatcher::predicates::IsPermutation());
  }

  /// Return a matcher instance which identifies an op that is equivalent to a
  /// `linalg.map`-like operation over a single input. The body operations all
  /// trace back to operating on a point in the input operand.
  static LinalgOpMatcher getUnaryLikeMatcher() {
    return LinalgOpMatcher()
        .setNumReductionLoops(0)
        .setNumDpsInits(1)
        .setNumDpsInputs(1)
        .addRegionMatchRootedAtOutput(
            0, LinalgOpMatcher::predicates::YieldedValueIsUnaryOfInput(0))
        .matchDpsInitIndexingMap(0, LinalgOpMatcher::predicates::IsIdentity())
        .matchDpsInputIndexingMap(0,
                                  LinalgOpMatcher::predicates::IsPermutation());
  }

  /// Return a matcher instance which identifies ops equivalent to
  /// `linalg.fill`.
  static LinalgOpMatcher getFillMatcher() {
    return LinalgOpMatcher()
        .setNumDpsInits(1)
        .setNumDpsInputs(1)
        .setNumReductionLoops(0)
        .matchDpsInits(
            0, [](OpOperand &, AffineMap map) { return map.isIdentity(); })
        .matchDpsInputs(0,
                        [](OpOperand &operand, AffineMap map) {
                          return map.isProjectedPermutation() &&
                                 operand.get().getType().isIntOrFloat();
                        })
        .addRegionMatchRootedAtOutput(0, predicates::YieldedValueIsInput(0));
  }

  /// Set the number of min/max loops required by this matcher.
  LinalgOpMatcher &setNumLoopsBounds(unsigned minNumLoops,
                                     unsigned maxNumLoops) {
    this->numLoopsBounds = std::make_pair(minNumLoops, maxNumLoops);
    return *this;
  }
  /// Set the number of reduction loops required by this matcher.
  LinalgOpMatcher &setNumReductionLoops(unsigned numLoops) {
    this->numReductions = numLoops;
    return *this;
  }

  LinalgOpMatcher &setNumDpsInits(unsigned numDpsInits) {
    this->numDpsInits = this->numDpsInits
                            ? std::max(*this->numDpsInits, numDpsInits)
                            : numDpsInits;
    return *this;
  }
  LinalgOpMatcher &setNumDpsInputs(unsigned numDpsInputs) {
    this->numDpsInputs = this->numDpsInputs
                             ? std::max(*this->numDpsInputs, numDpsInputs)
                             : numDpsInputs;
    return *this;
  }

  /// Adds a requirement that the linalg op output operand
  /// at position `outputIdx` is a reduction variable.
  LinalgOpMatcher &iteratorIsReduction(int64_t outputIdx);

  LinalgOpMatcher &addRegionMatchRootedAtOutput(
      int64_t outputIdx,
      std::function<bool(Value yieldedValue, Block::BlockArgListType inputArgs,
                         Block::BlockArgListType outputArgs)>
          predicate);

  LinalgOpMatcher &
  matchDpsInputs(unsigned operandIdx,
                 std::function<bool(OpOperand &, AffineMap map)> predicate);

  LinalgOpMatcher &
  matchDpsInits(unsigned operandIdx,
                std::function<bool(OpOperand &, AffineMap map)> predicate);

  /// Adds a predicate for the indexing map of the specified `ins` argument.
  LinalgOpMatcher &
  matchDpsInputIndexingMap(unsigned operandIdx,
                           std::function<bool(AffineMap map)> predicate);

  /// Adds a predicate for the indexing map of the specified `ins` argument.
  LinalgOpMatcher &matchDpsInputIndexingMap(
      unsigned operandIdx,
      std::function<bool(linalg::LinalgOp, AffineMap map)> predicate);

  /// Adds a predicate for the indexing map of the specified `outs` argument.
  LinalgOpMatcher &
  matchDpsInitIndexingMap(unsigned initIdx,
                          std::function<bool(AffineMap map)> predicate);

  /// Adds a predicate for the indexing map of the specified `outs` argument.
  LinalgOpMatcher &
  matchDpsInitIndexingMap(unsigned initIdx,
                          IndexingMapWithContextPredicateFn predicate);

  /// Adds a predicate for the indexing map of all `outs` arguments.
  LinalgOpMatcher &
  matchAllDpsInitIndexingMaps(IndexingMapWithContextPredicateFn predicate);

  /// Adds a predicate for all indexing maps.
  LinalgOpMatcher &
  allIndexingMapsMatch(std::function<bool(AffineMap map)> predicate);

  /// Returns true if the matcher matches the given op.
  bool match(Operation *op) const;
};

namespace linalg_ext {
/// Rewrite conv2d  to img2col + matmul.
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcFhwcOp convOp);

/// Rewrite a convolution in "nhwc/hwcf" form to "nhwc/fhwc" form by
/// materializing a transpose on the filter.
FailureOr<linalg::Conv2DNhwcFhwcOp>
rewriteConv2dFilterToHwcf(RewriterBase &rewriter,
                          linalg::Conv2DNhwcHwcfOp convOp);

/// Check whether any indexing map constains negative multiplication
/// coefficients. This is the case in situations such as a reverse iteration
/// over a dimension. Currently, the upstream TilingInterface does not support
/// tiling linalg operations with such reversed indexing maps.
bool hasNegativeMultiplicationCoefficients(ArrayRef<AffineMap> indexingMaps);

} // namespace linalg_ext

//===----------------------------------------------------------------------===//
// Contraction Metadata Helpers
//===----------------------------------------------------------------------===//

namespace linalg_ext {

/// Find all the dimensions numbers that appear as one of the results of
/// `operand`s indexing map and do not appear in another other result of
/// `operand`'s indexing map.
llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(linalg::LinalgOp op, OpOperand *operand,
                                utils::IteratorType iterType);

/// Find all the dimensions numbers that appear as one of the results of
/// the `indexingMap` and match the iterator type of `iterType`.
llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(ArrayRef<utils::IteratorType> iteratorTypes,
                                AffineMap indexingMap,
                                utils::IteratorType iterType);

/// Returns the batch dimensions for a linalg operation.
/// TODO: define batch dim?
llvm::SmallDenseSet<int64_t> getBatchDimensions(linalg::LinalgOp op);

/// Clone `op`'s body using the given values as replacements for block
/// arguments. Returns the value(s) that would be yielded by the terminator
/// of the body. The terminator is not cloned. The `getIndexValue` function
/// should be a callback that returns the value of `linalg.index [dim number]`.
SmallVector<Value> cloneLinalgBodyWithArgReplacements(
    RewriterBase &rewriter, linalg::LinalgOp op,
    ValueRange bodyBlockArgReplacements,
    llvm::function_ref<Value(OpBuilder &builder, Location loc, unsigned dim)>
        getIndexValue);

enum class DpsOperandKind { Input, Init };

struct DpsOperandInfo {
  DpsOperandKind kind;
  unsigned operandNumber;
  ShapedType shapedType{};
  AffineMap indexingMap;
};

inline auto getShapedDpsOperandInfoRange(linalg::LinalgOp op) {
  auto range = llvm::map_range(
      op->getOpOperands(), [op_ = op.getOperation()](OpOperand &operand) {
        linalg::LinalgOp op = cast<linalg::LinalgOp>(op_);
        DpsOperandInfo info;
        info.kind = op.isDpsInit(&operand) ? DpsOperandKind::Init
                                           : DpsOperandKind::Input;
        info.operandNumber = operand.getOperandNumber();
        info.shapedType = dyn_cast<ShapedType>(operand.get().getType());
        info.indexingMap = op.getMatchingIndexingMap(&operand);
        return info;
      });

  return llvm::make_filter_range(range, [](const DpsOperandInfo &info) -> bool {
    return info.shapedType != nullptr;
  });
}

//===----------------------------------------------------------------------===//
// Helper functions for matching Linalg body operations to the MMA operation
//===----------------------------------------------------------------------===//

BodyPredicateFn isFloatMixedTypeMatMul();

template <typename ExtIType>
BodyPredicateFn isIntMixedTypeMatMul();

BodyPredicateFn isIntSatfiniteMixedTypeMatMul();

template <typename BitOpType>
BodyPredicateFn isB1Int32MixedTypeMatMul();

} // namespace linalg_ext
} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_STRUCTUREDOPSUTILS_H
