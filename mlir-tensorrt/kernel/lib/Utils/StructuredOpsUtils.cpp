//===- StructuredOpsUtils.cpp ---------------------------------------------===//
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
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <numeric>

using namespace mlir;
#define DEBUG_TYPE "structured-op-utils"
#define DBGS(x)                                                                \
  llvm::dbgs() << "[" DEBUG_TYPE "] " __FILE__ << ":" << __LINE__ << " "

//===----------------------------------------------------------------------===//
// LinalgOpMatcher
//===----------------------------------------------------------------------===//

/// Return AffineMap predicate which matches permutation map.
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsPermutation() {
  return [](AffineMap map) { return map.isPermutation(); };
}

/// Return AffineMap predicate which matches permutation map.
IndexingMapPredicateFn
LinalgOpMatcher::predicates::IsProjectedPermutation(bool allowBroadcasting) {
  return [=](AffineMap map) {
    return map.isProjectedPermutation(allowBroadcasting);
  };
}

IndexingMapWithContextPredicateFn
LinalgOpMatcher::predicates::ProjectsOutSingleDimension(
    int64_t projectedDimension) {
  return [=](linalg::LinalgOp op, AffineMap map) {
    int64_t dim = projectedDimension < 0 ? projectedDimension + op.getNumLoops()
                                         : projectedDimension;
    if (!map.isProjectedPermutation(/*allowZeroInResults=*/false))
      return false;

    // We further restrict to projection which does not permute and lacks only
    // the given `dim`.
    unsigned expectedDimValue = 0;
    for (AffineExpr result : map.getResults()) {
      if (expectedDimValue == dim)
        expectedDimValue++;
      AffineDimExpr dimExpr = llvm::dyn_cast<AffineDimExpr>(result);
      if (!dimExpr)
        return false;
      if (dimExpr.getPosition() != expectedDimValue++)
        return false;
    }
    return true;
  };
}

/// Return AffineMap predicate which matches permutation map.
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsIdentity() {
  return [](AffineMap map) { return map.isIdentity(); };
}
/// Return AffineMap predicate which matches minor identity map.
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsMinorIdentity() {
  return [](AffineMap map) { return map.isMinorIdentity(); };
}
/// Return AffineMap predicate which matches (d0, d1, d2) -> (d0, d1)
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsMnkToMn() {
  return [](AffineMap map) {
    AffineExpr d0, d1, d2;
    bindDims(map.getContext(), d0, d1, d2);
    AffineMap expected = AffineMap::get(3, 0, {d0, d1}, map.getContext());
    return map == expected;
  };
}
/// Return AffineMap predicate which matches (d0, d1, d2)->(d0, d2).
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsMnkToMk() {
  return [](AffineMap map) {
    AffineExpr d0, d1, d2;
    bindDims(map.getContext(), d0, d1, d2);
    AffineMap expected = AffineMap::get(3, 0, {d0, d2}, map.getContext());
    return map == expected;
  };
}
/// Return AffineMap predicate which matches (d0, d1, d2)->(d1, d2).
IndexingMapPredicateFn LinalgOpMatcher::predicates::IsMnkToNk() {
  return [](AffineMap map) {
    AffineExpr d0, d1, d2;
    bindDims(map.getContext(), d0, d1, d2);
    AffineMap expected = AffineMap::get(3, 0, {d1, d2}, map.getContext());
    return map == expected;
  };
}
/// Return a body predicate which matches a yielded value that is
// identical to an input block arg
BodyPredicateFn
LinalgOpMatcher::predicates::YieldedValueIsInput(unsigned inputIdx) {
  return [=](Value yieldedValue, Block::BlockArgListType bodyIns,
             Block::BlockArgListType) {
    assert(inputIdx < bodyIns.size() && "inputIdx is out of bounds");
    return yieldedValue == bodyIns[inputIdx];
  };
}

/// Return a body predicate which matches a yielded value is a constant zero.
BodyPredicateFn LinalgOpMatcher::predicates::YieldedValueIsZero() {
  return [=](Value yieldedValue, Block::BlockArgListType,
             Block::BlockArgListType) {
    return matchPattern(yieldedValue, m_Zero());
  };
}

/// Return a body predicate which matches a yielded value is a constant scalar.
BodyPredicateFn
LinalgOpMatcher::predicates::YieldedValueIsConstant(Attribute *constantValue) {
  return [=](Value yieldedValue, Block::BlockArgListType,
             Block::BlockArgListType) {
    return matchPattern(yieldedValue, m_Constant(constantValue));
  };
}

/// Return a body predicate which matches a yielded value that is
// identical to an input block arg
BodyPredicateFn LinalgOpMatcher::predicates::YieldedValueIsUnaryOfInput(
    unsigned inputIdx, StringRef operationName) {
  return [=](Value yieldedValue, Block::BlockArgListType bodyIns,
             Block::BlockArgListType) {
    assert(inputIdx < bodyIns.size() && "inputIdx is out of bounds");
    Value current = yieldedValue;
    while (Operation *producer = current.getDefiningOp()) {
      if (!operationName.empty() &&
          operationName != producer->getName().getStringRef())
        return false;
      if (!producer->hasTrait<OpTrait::Elementwise>())
        return false;
      if ((producer->getNumOperands() == 1) ||
          (producer->getNumOperands() == 2 &&
           producer->getOperand(0) == producer->getOperand(1))) {
        current = producer->getOperand(0);
        continue;
      }
      return false;
    }
    return current == bodyIns[inputIdx];
  };
}

/// Return a body predicate which matches a yielded value that is a unary
/// operation of the input.
BodyPredicateFn LinalgOpMatcher::predicates::YieldedValueIsBinaryOfInputs(
    unsigned inputIdxLhs, unsigned inputIdxRhs, StringRef operationName) {
  return [=](Value yieldedValue, Block::BlockArgListType bodyIns,
             Block::BlockArgListType) {
    assert(inputIdxLhs < bodyIns.size() && inputIdxRhs < bodyIns.size() &&
           "input indices are out of bounds");
    Value current = yieldedValue;
    Operation *producer = current.getDefiningOp();
    if (!producer || producer->getNumOperands() != 2 ||
        producer->getNumResults() != 1)
      return false;
    if (!operationName.empty() &&
        operationName != producer->getName().getStringRef())
      return false;
    return producer->getOperand(0) == bodyIns[inputIdxLhs] &&
           producer->getOperand(1) == bodyIns[inputIdxRhs];
  };
}

/// Return a body predicate which matches a yielded value that is a binary
/// of one of the inputs and one of the outputs/loop carries.
BodyPredicateFn
LinalgOpMatcher::predicates::YieldedValueIsBinaryOfInputAndOutput(
    unsigned inputIdx, unsigned outputIdx, bool isCommutative,
    StringRef operationName) {
  return [=](Value yieldedValue, Block::BlockArgListType bodyIns,
             Block::BlockArgListType bodyOuts) {
    assert(inputIdx < bodyIns.size() && outputIdx < bodyOuts.size() &&
           "block argument indices are out of bounds");
    Value current = yieldedValue;
    Operation *producer = current.getDefiningOp();
    if (!producer || producer->getNumOperands() != 2 ||
        producer->getNumResults() != 1)
      return false;
    if (!operationName.empty() &&
        operationName != producer->getName().getStringRef())
      return false;
    return (producer->getOperand(0) == bodyIns[inputIdx] &&
            producer->getOperand(1) == bodyOuts[outputIdx]) ||
           (isCommutative && producer->getOperand(1) == bodyIns[inputIdx] &&
            producer->getOperand(0) == bodyOuts[outputIdx]);
  };
}

/// Return a body predicate which matches a matmul operation.
/// TODO: add support for other scalar type variations.
BodyPredicateFn LinalgOpMatcher::predicates::IsMatMul() {
  using ::mlir::matchers::m_Val;
  return [](Value yieldedValue, Block::BlockArgListType inputArgs,
            Block::BlockArgListType outputArgs) -> bool {
    if (inputArgs.size() != 2)
      return false;
    return matchPattern(
               yieldedValue,
               m_Op<arith::AddFOp>(
                   m_Op<arith::MulFOp>(detail::m_ElwiseValue(inputArgs[0]),
                                       detail::m_ElwiseValue(inputArgs[1])),
                   m_Val(outputArgs[0]))) ||
           matchPattern(
               yieldedValue,
               m_Op<arith::AddFOp>(
                   m_Val(outputArgs[0]),
                   m_Op<arith::MulFOp>(detail::m_ElwiseValue(inputArgs[0]),
                                       detail::m_ElwiseValue(inputArgs[1]))));
  };
}

/// Adds a requirement that the linalg op output operand
/// at position `outputIdx` is a reduction variable.
LinalgOpMatcher &LinalgOpMatcher::iteratorIsReduction(int64_t outputIdx) {
  opPredicates.push_back([outputIdx](linalg::LinalgOp op) {
    unsigned idx = outputIdx < 0 ? op.getNumLoops() + outputIdx : outputIdx;
    return op.getIteratorTypesArray()[idx] == utils::IteratorType::reduction;
  });
  return *this;
}

LinalgOpMatcher &LinalgOpMatcher::addRegionMatchRootedAtOutput(
    int64_t outputIdx,
    std::function<bool(Value yieldedValue, Block::BlockArgListType inputArgs,
                       Block::BlockArgListType outputArgs)>
        predicate) {
  opPredicates.push_back([=](linalg::LinalgOp op) -> bool {
    unsigned idx = outputIdx < 0 ? op.getNumDpsInits() + outputIdx : outputIdx;
    Value yieldedVal =
        op->getRegion(0).front().getTerminator()->getOperands()[idx];
    return predicate(yieldedVal, op.getRegionInputArgs(),
                     op.getRegionOutputArgs());
  });
  return *this;
}
LinalgOpMatcher &LinalgOpMatcher::matchDpsInputs(
    unsigned operandIdx,
    std::function<bool(OpOperand &, AffineMap map)> predicate) {
  opPredicates.push_back([operandIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInputOperand(operandIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(*operand, map);
  });
  return setNumDpsInputs(operandIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchDpsInits(
    unsigned initIdx,
    std::function<bool(OpOperand &, AffineMap map)> predicate) {
  opPredicates.push_back([initIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInitOperand(initIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(*operand, map);
  });
  return setNumDpsInits(initIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchDpsInputIndexingMap(
    unsigned operandIdx, std::function<bool(AffineMap map)> predicate) {
  opPredicates.push_back([operandIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInputOperand(operandIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(map);
  });
  return setNumDpsInputs(operandIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchDpsInputIndexingMap(
    unsigned operandIdx, IndexingMapWithContextPredicateFn predicate) {
  opPredicates.push_back([operandIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInputOperand(operandIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(op, map);
  });
  return setNumDpsInputs(operandIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchDpsInitIndexingMap(
    unsigned initIdx, std::function<bool(AffineMap map)> predicate) {
  opPredicates.push_back([initIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInitOperand(initIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(map);
  });
  return setNumDpsInits(initIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchDpsInitIndexingMap(
    unsigned initIdx, IndexingMapWithContextPredicateFn predicate) {
  opPredicates.push_back([initIdx, predicate](linalg::LinalgOp op) {
    OpOperand *operand = op.getDpsInitOperand(initIdx);
    AffineMap map = op.getMatchingIndexingMap(operand);
    return predicate(op, map);
  });
  return setNumDpsInits(initIdx + 1);
}

LinalgOpMatcher &LinalgOpMatcher::matchAllDpsInitIndexingMaps(
    IndexingMapWithContextPredicateFn predicate) {

  opPredicates.push_back([predicate](linalg::LinalgOp op) {
    return llvm::all_of(op.getDpsInitsMutable(), [&](OpOperand &operand) {
      AffineMap map = op.getMatchingIndexingMap(&operand);
      return predicate(op, map);
    });
  });

  return *this;
}

LinalgOpMatcher &LinalgOpMatcher::allIndexingMapsMatch(
    std::function<bool(AffineMap map)> predicate) {
  opPredicates.push_back([predicate](linalg::LinalgOp op) {
    return llvm::all_of(op.getIndexingMapsArray(), predicate);
  });
  return *this;
}

bool LinalgOpMatcher::match(Operation *op) const {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  if (numDpsInputs && linalgOp.getNumDpsInputs() != *numDpsInputs)
    return false;
  if (numDpsInits && linalgOp.getNumDpsInits() != *numDpsInits)
    return false;
  if (numLoopsBounds) {
    auto [minLoops, maxLoops] = *numLoopsBounds;
    unsigned numLoops = linalgOp.getNumLoops();
    if (numLoops < minLoops || numLoops > maxLoops)
      return false;
  }
  if (numReductions && linalgOp.getNumReductionLoops() != *numReductions)
    return false;
  for (auto predFunc : opPredicates) {
    if (!predFunc(linalgOp))
      return false;
  }
  if (regionPredicate && linalgOp->getNumRegions() != 1)
    return false;
  if (regionPredicate && !regionPredicate(linalgOp->getRegion(0)))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Indexing Map Utilities
//===----------------------------------------------------------------------===//

namespace {
/// This helper assists with verifying that all linalg indexing maps do not have
/// negative coefficients. Such maps are poorly supported by core tiling
/// transformations.
struct NegMulCoefficientDetector
    : public AffineExprVisitor<NegMulCoefficientDetector> {
  void checkConstantNegCoefficient(AffineExpr e) {
    auto constExpr = dyn_cast<AffineConstantExpr>(e);
    if (constExpr && constExpr.getValue() < 0)
      detected = true;
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul) {
      checkConstantNegCoefficient(expr.getLHS());
      checkConstantNegCoefficient(expr.getRHS());
    }
  }
  bool detected = false;
};

} // namespace

bool linalg_ext::hasNegativeMultiplicationCoefficients(
    ArrayRef<AffineMap> indexingMaps) {
  for (AffineMap map : indexingMaps) {
    NegMulCoefficientDetector detector;
    for (AffineExpr expr : map.getResults()) {
      detector.visit(expr);
      if (detector.detected) {
        LLVM_DEBUG(
            DBGS()
            << "detected negative multiplication coefficient in indexing map: "
            << map << "\n");
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Code below this point is modified from ConvertConv2DToImg2Col.cpp, which does
// not have NHWC/FHWC support. This is a staging point for moving this code
// upstream. The original code is part of the LLVM Project and retains LLVM
// project license.
//
// Additions/modifications are Copyright (c) 2023, NVIDIA CORPORATION. All
// rights reserved.
//===----------------------------------------------------------------------===//

FailureOr<linalg::Conv2DNhwcFhwcOp>
linalg_ext::rewriteConv2dFilterToHwcf(RewriterBase &rewriter,
                                      linalg::Conv2DNhwcHwcfOp convOp) {
  TensorType filterType = cast<TensorType>(convOp.filter().getType());
  if (!filterType.hasStaticShape())
    return failure();
  SmallVector<int64_t> permutation{3, 0, 1, 2};
  SmallVector<int64_t> filterShape(filterType.getShape());
  applyPermutationToVector(filterShape, permutation);
  Value empty = rewriter.create<tensor::EmptyOp>(convOp.getLoc(), filterShape,
                                                 filterType.getElementType());
  Value transposeFilter =
      rewriter
          .create<linalg::TransposeOp>(convOp.getLoc(),
                                       convOp.getDpsInputOperand(1)->get(),
                                       empty, permutation)
          ->getResult(0);
  auto newOp = rewriter.create<linalg::Conv2DNhwcFhwcOp>(
      convOp.getLoc(), convOp->getResultTypes()[0],
      ValueRange{convOp.getOperand(0), transposeFilter},
      ValueRange{convOp.getDpsInitOperand(0)->get()});
  if (convOp.getStridesAttr())
    newOp.setStridesAttr(convOp.getStridesAttr());
  if (convOp.getDilationsAttr())
    newOp.setDilationsAttr(convOp.getDilationsAttr());
  rewriter.replaceOp(convOp, newOp->getResults());
  return newOp;
}

/// Create the appropriate add based on the value types.
/// This logic is from
/// `third_party/llvm-project/mlir/lib/Dialect/Linalg/Transforms/ConvertConv2DToImg2Col.cpp`,
/// reproduced here until this part can be upstreamed.
static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = isa<IntegerType>(x.getType());
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

/// Create the appropriate mul based on the scalar types.
/// This logic is from
/// `third_party/llvm-project/mlir/lib/Dialect/Linalg/Transforms/ConvertConv2DToImg2Col.cpp`,
/// reproduced here until this part can be upstreamed.
static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  Value xConvert = mlir::convertScalarToDtype(builder, loc, x, accType,
                                              /*isUnsignedCast=*/false);
  Value yConvert = mlir::convertScalarToDtype(builder, loc, y, accType,
                                              /*isUnsignedCast=*/false);
  if (isa<IntegerType>(accType))
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

/// Create a staticly shaped `tensor.collapse_shape` operation and infer the
/// result type.
static TypedValue<TensorType>
createStaticCollapse(RewriterBase &rewriter, Location loc,
                     TypedValue<TensorType> input,
                     ArrayRef<ReassociationIndices> reassociation) {
  auto product = [](ArrayRef<int64_t> in) {
    return std::accumulate(in.begin(), in.end(), 1, std::multiplies<int64_t>());
  };
  auto gather = [](ArrayRef<int64_t> in, ArrayRef<int64_t> indices) {
    return llvm::map_to_vector(indices, [&](int64_t i) { return in[i]; });
  };

  SmallVector<int64_t> newShape;
  ArrayRef<int64_t> inputShape = input.getType().getShape();
  for (const auto &re : reassociation)
    newShape.push_back(product(gather(inputShape, re)));
  auto reshapedType =
      RankedTensorType::get(newShape, input.getType().getElementType());
  return rewriter
      .create<tensor::CollapseShapeOp>(loc, reshapedType, input, reassociation)
      .getResult();
}

/// Create the matrix multiplication between im2col matrix and filter matrix.
static Value createIm2ColFilterMatMul(RewriterBase &rewriter, Location loc,
                                      Value lhs, Value rhs, Value out) {
  AffineExpr bDim, mDim, nDim, kDim;
  MLIRContext *ctx = rewriter.getContext();
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  bindDims(ctx, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, ctx);
  auto rhsMap = AffineMap::get(4, 0, {nDim, kDim}, ctx);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, ctx);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, out.getType(),
      /*inputs=*/ValueRange{lhs, rhs},
      /*outputs=*/ValueRange{out},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });
  return genericOp.getResults().front();
}

/// Create the im2col matrix if input is in format NHWC and filter is in format
/// FHWC.
static TypedValue<TensorType> createIm2ColTensorNhwcFhwc(
    RewriterBase &rewriter, Location loc, TypedValue<TensorType> input,
    ArrayRef<int64_t> outputShape, ArrayRef<int64_t> filterShape,
    DenseIntElementsAttr strides) {
  TensorType inputType = input.getType();
  const int64_t n = outputShape[0];
  const int64_t oh = outputShape[1];
  const int64_t ow = outputShape[2];
  const int64_t fh = filterShape[1];
  const int64_t fw = filterShape[2];
  const int64_t ic = filterShape[3];
  SmallVector<int64_t> colTensorShape = {n, oh, ow, fh, fw, ic};
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());
  AffineExpr nDim, ohDim, owDim, khDim, kwDim, icDim;
  bindDims(rewriter.getContext(), nDim, ohDim, owDim, khDim, kwDim, icDim);
  auto shSym = rewriter.getAffineConstantExpr(strides.getValues<int64_t>()[0]);
  auto swSym = rewriter.getAffineConstantExpr(strides.getValues<int64_t>()[1]);
  SmallVector<AffineExpr> inputExprs = {nDim, ohDim * shSym + khDim,
                                        owDim * swSym + kwDim, icDim};
  auto nloops = colTensorShape.size();
  SmallVector<utils::IteratorType, 3> img2colIterators(
      nloops, utils::IteratorType::parallel);
  SmallVector<AffineMap> img2colIndexingMaps = {
      AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });
  return cast<TypedValue<TensorType>>(genericOp.getResults().front());
}

static bool allUnitValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

FailureOr<std::pair<Operation *, Operation *>>
linalg_ext::rewriteInIm2Col(RewriterBase &rewriter,
                            linalg::Conv2DNhwcFhwcOp convOp) {
  auto inputType = cast<ShapedType>(convOp.getInputs()[0].getType());
  auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
  auto outputType = cast<ShapedType>(convOp.getOutputs()[0].getType());
  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");
  // TODO: support dynamic shaped input (in spatial dims)
  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");
  // TODO: support non-unit dilation
  if (!allUnitValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  auto input = cast<TypedValue<TensorType>>(convOp.getInputs()[0]);
  auto filter = cast<TypedValue<TensorType>>(convOp.getInputs()[1]);
  auto output = cast<TypedValue<TensorType>>(convOp.getOutputs()[0]);

  ArrayRef<int64_t> filterShape = filterType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  Location loc = convOp.getLoc();

  // Collapse the filter and the output
  TypedValue<TensorType> reshapedFilter =
      createStaticCollapse(rewriter, filter.getLoc(), filter, {{0}, {1, 2, 3}});
  const SmallVector<ReassociationIndices> outputReassociation{{0}, {1, 2}, {3}};
  TypedValue<TensorType> reshapedOutput = createStaticCollapse(
      rewriter, output.getLoc(), output, outputReassociation);
  // Create the im2col matrix and reshape it.
  TypedValue<TensorType> im2Col = createIm2ColTensorNhwcFhwc(
      rewriter, loc, input, outputShape, filterShape, convOp.getStrides());
  TypedValue<TensorType> img2ColTensor = createStaticCollapse(
      rewriter, im2Col.getLoc(), im2Col, {{0}, {1, 2}, {3, 4, 5}});
  // Create the matmul-like operation.
  Value matmulResult = createIm2ColFilterMatMul(
      rewriter, convOp.getLoc(), img2ColTensor, reshapedFilter, reshapedOutput);
  // Expand shape to match original.
  Value reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, matmulResult, outputReassociation);
  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});
  return std::make_pair(img2ColTensor.getDefiningOp(),
                        reshapedResult.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// Contraction Metadata Helpers
//
// Code below this point is modified from Linalg utils. This is a staging point
// for moving this code upstream. The original code is part of the LLVM Project
// and retains LLVM project license.
//
// Additions/modifications are Copyright (c) 2023, NVIDIA CORPORATION. All
// rights reserved.
//===----------------------------------------------------------------------===//

llvm::SmallDenseSet<int64_t> mlir::linalg_ext::findPermutationsIndexingOperand(
    ArrayRef<utils::IteratorType> iteratorTypes, AffineMap indexingMap,
    utils::IteratorType iterType) {
  llvm::SmallDenseSet<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    auto d = llvm::dyn_cast<AffineDimExpr>(e);
    if (!d)
      continue;
    unsigned pos = d.getPosition();
    if (iteratorTypes[pos] != iterType)
      continue;
    if (llvm::count_if(indexingMap.getResults(), [=](AffineExpr e) {
          return e.isFunctionOfDim(pos);
        }) != 1)
      continue;
    res.insert(pos);
  }
  return res;
}

llvm::SmallDenseSet<int64_t> mlir::linalg_ext::findPermutationsIndexingOperand(
    linalg::LinalgOp op, OpOperand *operand, utils::IteratorType iterType) {
  return findPermutationsIndexingOperand(
      op.getIteratorTypesArray(), op.getMatchingIndexingMap(operand), iterType);
}

llvm::SmallDenseSet<int64_t>
linalg_ext::getBatchDimensions(linalg::LinalgOp op) {
  auto operands = op->getOpOperands();
  llvm::SmallDenseSet<int64_t> batchDims = findPermutationsIndexingOperand(
      op, &operands.front(), utils::IteratorType::parallel);
  for (OpOperand &operand : operands.drop_front())
    llvm::set_intersect(batchDims,
                        findPermutationsIndexingOperand(
                            op, &operand, utils::IteratorType::parallel));
  return batchDims;
}

SmallVector<Value> mlir::linalg_ext::cloneLinalgBodyWithArgReplacements(
    RewriterBase &rewriter, linalg::LinalgOp op,
    ValueRange bodyBlockArgReplacements,
    llvm::function_ref<Value(OpBuilder &, Location, unsigned)> getIndexValue) {
  Operation *term = op.getBlock()->getTerminator();
  IRMapping mapping;
  mapping.map(op.getBlock()->getArguments(), bodyBlockArgReplacements);
  for (Operation &bodyOp : op.getBlock()->without_terminator()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(bodyOp)) {
      Value index = getIndexValue(rewriter, indexOp.getLoc(), indexOp.getDim());
      mapping.map(indexOp.getResult(), index);
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }
  return llvm::map_to_vector(term->getOperands(), [&](Value v) {
    Region *parent = v.getParentBlock()->getParent();
    if (parent->isProperAncestor(op.getBlock()->getParent()))
      return v;
    return mapping.lookup(v);
  });
}

BodyPredicateFn mlir::linalg_ext::isFloatMixedTypeMatMul() {
  using ::mlir::matchers::m_Val;
  return [](Value yieldedValue, Block::BlockArgListType inputArgs,
            Block::BlockArgListType outputArgs) -> bool {
    if (inputArgs.size() != 2)
      return false;
    auto outputVal = m_Val(outputArgs[0]);
    auto mulVal = m_Op<arith::ExtFOp>(
        m_Op<arith::MulFOp>(mlir::detail::m_ElwiseValue(inputArgs[0]),
                            mlir::detail::m_ElwiseValue(inputArgs[1])));
    return matchPattern(yieldedValue, m_Op<arith::AddFOp>(outputVal, mulVal)) ||
           matchPattern(yieldedValue, m_Op<arith::AddFOp>(mulVal, outputVal));
  };
}

template BodyPredicateFn linalg_ext::isIntMixedTypeMatMul<arith::ExtSIOp>();
template BodyPredicateFn linalg_ext::isIntMixedTypeMatMul<arith::ExtUIOp>();
template <typename ExtIType>
BodyPredicateFn mlir::linalg_ext::isIntMixedTypeMatMul() {
  using ::mlir::matchers::m_Val;
  return [](Value yieldedValue, Block::BlockArgListType inputArgs,
            Block::BlockArgListType outputArgs) -> bool {
    if (inputArgs.size() != 2)
      return false;
    auto outputVal = m_Op<ExtIType>(m_Val(outputArgs[0]));
    auto mulVal = m_Op<ExtIType>(m_Op<arith::MulIOp>(
        m_Op<ExtIType>(mlir::detail::m_ElwiseValue(inputArgs[0])),
        m_Op<ExtIType>(mlir::detail::m_ElwiseValue(inputArgs[1]))));
    return matchPattern(yieldedValue, m_Op<arith::TruncIOp>(m_Op<arith::AddIOp>(
                                          outputVal, mulVal))) ||
           matchPattern(yieldedValue, m_Op<arith::TruncIOp>(m_Op<arith::AddIOp>(
                                          mulVal, outputVal)));
  };
}

BodyPredicateFn mlir::linalg_ext::isIntSatfiniteMixedTypeMatMul() {
  using ::mlir::matchers::m_Val;
  return [](Value yieldedValue, Block::BlockArgListType inputArgs,
            Block::BlockArgListType outputArgs) -> bool {
    if (inputArgs.size() != 2 || outputArgs.size() != 1)
      return false;
    auto outputVal = m_Op<arith::ExtSIOp>(m_Val(outputArgs[0]));
    auto mulVal = m_Op<arith::ExtSIOp>(m_Op<arith::MulIOp>(
        m_Op<arith::ExtSIOp>(mlir::detail::m_ElwiseValue(inputArgs[0])),
        m_Op<arith::ExtSIOp>(mlir::detail::m_ElwiseValue(inputArgs[1]))));
    auto addPattern1 = m_Op<arith::AddIOp>(outputVal, mulVal);
    auto addPattern2 = m_Op<arith::AddIOp>(mulVal, outputVal);

    return matchPattern(
               yieldedValue,
               m_Op<arith::TruncIOp>(m_Op<arith::MaxSIOp>(
                   m_Op<arith::MinSIOp>(addPattern1, m_Op<arith::ConstantOp>()),
                   m_Op<arith::ConstantOp>()))) ||
           matchPattern(
               yieldedValue,
               m_Op<arith::TruncIOp>(m_Op<arith::MinSIOp>(
                   m_Op<arith::MaxSIOp>(addPattern1, m_Op<arith::ConstantOp>()),
                   m_Op<arith::ConstantOp>()))) ||
           matchPattern(
               yieldedValue,
               m_Op<arith::TruncIOp>(m_Op<arith::MinSIOp>(
                   m_Op<arith::MaxSIOp>(addPattern2, m_Op<arith::ConstantOp>()),
                   m_Op<arith::ConstantOp>()))) ||
           matchPattern(
               yieldedValue,
               m_Op<arith::TruncIOp>(m_Op<arith::MinSIOp>(
                   m_Op<arith::MaxSIOp>(addPattern2, m_Op<arith::ConstantOp>()),
                   m_Op<arith::ConstantOp>())));
  };
}

template BodyPredicateFn linalg_ext::isB1Int32MixedTypeMatMul<arith::XOrIOp>();
template BodyPredicateFn linalg_ext::isB1Int32MixedTypeMatMul<arith::AndIOp>();
template <typename BitOpType>
BodyPredicateFn mlir::linalg_ext::isB1Int32MixedTypeMatMul() {
  using ::mlir::matchers::m_Val;
  return [](Value yieldedValue, Block::BlockArgListType inputArgs,
            Block::BlockArgListType outputArgs) -> bool {
    if (inputArgs.size() != 2)
      return false;
    auto outputVal = m_Val(outputArgs[0]);
    auto ctPopVal = m_Op<math::CtPopOp>(
        m_Op<BitOpType>(mlir::detail::m_ElwiseValue(inputArgs[0]),
                        mlir::detail::m_ElwiseValue(inputArgs[1])));
    return matchPattern(yieldedValue,
                        m_Op<arith::AddIOp>(outputVal, ctPopVal)) ||
           matchPattern(yieldedValue, m_Op<arith::AddIOp>(ctPopVal, outputVal));
  };
}
