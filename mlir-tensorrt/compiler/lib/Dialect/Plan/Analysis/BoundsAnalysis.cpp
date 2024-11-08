//===- BoundsAnalysis.cpp -------------------------------------------------===//
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
/// Implementation of Plan dialect dataflow analyses for shape/value bounds.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/Support/Debug.h"
#include <limits>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::plan;

#define DEBUG_TYPE "plan-bounds-analysis"
#define DBGS(x) llvm::dbgs() << " [" DEBUG_TYPE "][" x "] "

static std::optional<tensorrt::ShapeProfileAttr>
maybeGetFunctionArgBound(Value value, StringRef attrName) {
  BlockArgument source = dyn_cast<BlockArgument>(value);
  if (!source)
    return {};
  func::FuncOp func = dyn_cast<func::FuncOp>(source.getOwner()->getParentOp());
  if (!func)
    return {};
  auto shapeProfile = func.getArgAttrOfType<tensorrt::ShapeProfileAttr>(
      source.getArgNumber(), attrName);
  if (!shapeProfile)
    return {};
  return shapeProfile;
}

static bool hasShapeFuncMarker(Value value, StringRef attrName) {
  BlockArgument source = dyn_cast<BlockArgument>(value);
  if (!source)
    return false;
  func::FuncOp func = dyn_cast<func::FuncOp>(source.getOwner()->getParentOp());
  if (!func)
    return false;
  Attribute attr = func->getAttr(attrName);
  if (attr)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// BoundsValue
//===----------------------------------------------------------------------===//

ConstantIntRanges BoundsArray::getMaxDimRange() {
  APInt smin = APInt(IndexType::kInternalStorageBitWidth, 0);
  APInt smax = APInt(IndexType::kInternalStorageBitWidth,
                     std::numeric_limits<int32_t>::max());
  return ConstantIntRanges::fromSigned(smin, smax);
}

BoundsArray BoundsArray::getMaxRangeForShapeBounds(Value v) {
  auto type = cast<ShapedType>(v.getType());
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(type.getRank());
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      ranges.push_back(getMaxDimRange());
      continue;
    }
    ranges.push_back(ConstantIntRanges::constant(APInt(64, dim)));
  }
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::getMaxRangeForValueBounds(Value v) {
  assert(TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(v) &&
         "value is unsuitable for analysis");
  Type elementType = mlir::getElementTypeOrSelf(v);
  unsigned numBits = ConstantIntRanges::getStorageBitwidth(elementType);
  APInt smin = APInt::getSignedMinValue(numBits);
  APInt smax = APInt::getSignedMaxValue(numBits);
  SmallVector<ConstantIntRanges> ranges(
      cast<ShapedType>(v.getType()).getNumElements(),
      ConstantIntRanges::fromSigned(smin, smax));
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::getFromConstantValue(DenseIntElementsAttr v) {
  assert(TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(v.getType()) &&
         "attribute type is unsuitable for creating value bound state");
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(cast<ShapedType>(v.getType()).getNumElements());
  for (const APInt &element : v.getValues<APInt>())
    ranges.push_back(ConstantIntRanges::constant(element));
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::fromShapeBounds(ArrayRef<int64_t> min,
                                         ArrayRef<int64_t> max) {
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(min, max))
    res.push_back(ConstantIntRanges::fromSigned(APInt(64, l), APInt(64, r)));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::fromIntegerValueBounds(unsigned bitWidth,
                                                ArrayRef<int64_t> min,
                                                ArrayRef<int64_t> max) {
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(min, max))
    res.push_back(
        ConstantIntRanges::fromSigned(APInt(64, l).sextOrTrunc(bitWidth),
                                      APInt(64, r).sextOrTrunc(bitWidth)));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::join(const BoundsArray &lhs, const BoundsArray &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(lhs.getValue(), rhs.getValue()))
    res.push_back(l.rangeUnion(r));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::meet(const BoundsArray &lhs, const BoundsArray &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(lhs.getValue(), rhs.getValue()))
    res.push_back(l.intersection(r));
  return BoundsArray(std::move(res));
}

void BoundsArray::print(raw_ostream &os) const {
  if (!value) {
    os << "<<uninitialized>>";
    return;
  }
  os << "<";
  llvm::interleaveComma(*value, os, [&](const ConstantIntRanges &r) {
    os << "[" << r.smin() << ", " << r.smax() << "]";
  });
  os << ">";
}

llvm::raw_ostream &plan::operator<<(llvm::raw_ostream &os,
                                    const BoundsArray &v) {
  v.print(os);
  return os;
}

std::pair<DenseElementsAttr, DenseElementsAttr>
BoundsArray::getAsElementsAttr(RankedTensorType type) const {
  assert(!isUninitialized() && "expected initialized value");
  assert(type.getNumElements() == static_cast<int64_t>(value->size()) &&
         "specified tensor type's volume does not match lattice value volume");
  SmallVector<APInt> lbs;
  lbs.reserve(type.getNumElements());
  SmallVector<APInt> ubs;
  ubs.reserve(type.getNumElements());
  for (const ConstantIntRanges &r : *value) {
    lbs.push_back(r.smin());
    ubs.push_back(r.smax());
  }
  return std::make_pair(DenseElementsAttr::get(type, lbs),
                        DenseElementsAttr::get(type, ubs));
}

/// Returns true if the element ranges are constant (single-value) ranges.
std::optional<DenseElementsAttr>
BoundsArray::getConstantValues(RankedTensorType type) const {
  assert(!isUninitialized() && "expected initialized value");
  assert(type.getNumElements() == static_cast<int64_t>(value->size()) &&
         "specified tensor type's volume does not match lattice value volume");
  SmallVector<APInt> lbs;
  lbs.reserve(type.getNumElements());
  for (const ConstantIntRanges &r : *value) {
    if (r.smin() != r.smax())
      return {};
    lbs.push_back(r.smin());
  }
  return DenseElementsAttr::get(type, lbs);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<ConstantIntRanges>>
intersectDimBoundsWithScalarBounds(
    ArrayRef<const IntegerValueRangeLattice *> scalarBounds,
    const ShapeBoundsLattice *dimBounds) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(scalarBounds.size());

  for (unsigned i = 0, e = scalarBounds.size(); i < e; i++) {
    const IntegerValueRangeLattice *scalar = scalarBounds[i];

    bool scalarIsInvalidOrMaxRange =
        !scalar || scalar->getValue().isUninitialized();

    if (dimBounds && !dimBounds->getValue().isUninitialized()) {
      if (!scalarIsInvalidOrMaxRange) {
        ranges.push_back(scalar->getValue().getValue().intersection(
            dimBounds->getValue().getValue()[i]));
        continue;
      }
      ranges.push_back(dimBounds->getValue().getValue()[i]);
      continue;
    }

    if (!scalarIsInvalidOrMaxRange) {
      // If we only have the scalar to go off of, then we intersect the scalar
      // with positive range. This is only necessary due to issues in upstream
      // op's shape materialization functions. It is necessary
      // because the bounds calculation may be operating on values that can only
      // give course bounds. For, example, function argument host tensor inputs
      // for a slice "offset" and "limit" are given with individual ranges. the
      // calculation "limit - offset" to get the size of the slice can have a
      // negative lower bound. If the lower bound is negative, this just means
      // that it is possible for the func caller to give values within the input
      // limits where a zero-size slice is expected.
      unsigned numBits = scalar->getValue().getValue().smin().getBitWidth();
      ranges.push_back(scalar->getValue().getValue().intersection(
          ConstantIntRanges::fromSigned(APInt::getZero(numBits),
                                        APInt::getSignedMaxValue(numBits))));
      continue;
    }

    return failure();
  }
  return ranges;
}

//===----------------------------------------------------------------------===//
// ShapeBoundsForwardAnalysis
//===----------------------------------------------------------------------===//

void ShapeBoundsForwardAnalysis::setToEntryState(ShapeBoundsLattice *lattice) {
  ShapedType shapedType = dyn_cast<ShapedType>(lattice->getAnchor().getType());
  if (!shapedType)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  if (shapedType.hasStaticShape())
    return propagateIfChanged(
        lattice, lattice->join(BoundsArray::getMaxRangeForShapeBounds(
                     lattice->getAnchor())));

  std::optional<tensorrt::ShapeProfileAttr> shapeProfile =
      maybeGetFunctionArgBound(
          lattice->getAnchor(),
          tensorrt::TensorRTDialect::getShapeProfileArgAttrName());
  if (!shapeProfile)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  return propagateIfChanged(
      lattice, lattice->join(BoundsArray::fromShapeBounds(
                   shapeProfile->getMin(), shapeProfile->getMax())));
}

LogicalResult ShapeBoundsForwardAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeBoundsLattice *> operands,
    ArrayRef<ShapeBoundsLattice *> results) {

  LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis") << "visiting " << *op << "\n");

  auto joinCallback = [&](Value v, ArrayRef<ConstantIntRanges> attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));
    ShapeBoundsLattice *lattice = results[result.getResultNumber()];
    const BoundsArray &oldRanges = lattice->getValue();
    BoundsArray newRange{llvm::to_vector(attrs)};

    LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
               << "inferred " << newRange << " for\n\t" << v << "\n");

    ChangeResult changed = lattice->join(newRange);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRanges.isUninitialized() &&
        !(lattice->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(BoundsArray::getMaxRangeForShapeBounds(v));
    }
    propagateIfChanged(lattice, changed);
  };

  for (ShapeBoundsLattice *lattice : results) {
    if (auto rtt = dyn_cast<RankedTensorType>(lattice->getAnchor().getType())) {
      if (rtt.hasStaticShape())
        propagateIfChanged(lattice,
                           lattice->join(BoundsArray::getMaxRangeForShapeBounds(
                               lattice->getAnchor())));
    }
  }

  auto getScalarLatticeValues = [&](ValueRange scalars) {
    SmallVector<const IntegerValueRangeLattice *> result;
    result.reserve(scalars.size());
    for (Value v : scalars)
      result.emplace_back(
          this->getOrCreateFor<IntegerValueRangeLattice>(op, v));
    return result;
  };

  if (auto withOp = dyn_cast<plan::WithShapeOp>(op)) {
    FailureOr<SmallVector<ConstantIntRanges>> ranges =
        intersectDimBoundsWithScalarBounds(
            getScalarLatticeValues(withOp.getShape()), operands[0]);
    if (failed(ranges))
      return success();
    joinCallback(withOp.getResult(), *ranges);
    return success();
  }
  return success();
}

void ShapeBoundsForwardAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<ShapeBoundsLattice *> argLattices, unsigned firstIndex) {
  LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
                 << "visiting non-control-flow arguments for ";
             op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}

//===----------------------------------------------------------------------===//
// ShapeBoundsBackwardsAnalysis
//===----------------------------------------------------------------------===//

void ShapeBoundsBackwardsAnalysis::setToExitState(ShapeBoundsLattice *lattice) {
  LLVM_DEBUG(
      DBGS("ShapeBoundsBackwardsAnalysis") << "setting to exit state: ";
      lattice->getAnchor().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n");

  auto shapedType = dyn_cast<ShapedType>(lattice->getAnchor().getType());
  if (!shapedType)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  if (shapedType.hasStaticShape())
    return propagateIfChanged(
        lattice, lattice->meet(BoundsArray::getMaxRangeForShapeBounds(
                     lattice->getAnchor())));

  return propagateIfChanged(lattice, lattice->join(BoundsArray()));
}

LogicalResult ShapeBoundsBackwardsAnalysis::visitOperation(
    Operation *op, ArrayRef<ShapeBoundsLattice *> operands,
    ArrayRef<const ShapeBoundsLattice *> results) {
  LLVM_DEBUG({
    DBGS("ShapeBoundsBackwardsAnalysis") << "visiting ";
    op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });

  auto getScalarLatticeValues = [&](ValueRange scalars) {
    SmallVector<const IntegerValueRangeLattice *> result;
    result.reserve(scalars.size());
    for (Value v : scalars)
      result.emplace_back(
          this->getOrCreateFor<IntegerValueRangeLattice>(op, v));
    return result;
  };

  if (auto withOp = dyn_cast<plan::WithShapeOp>(op)) {
    BoundsArray oldRanges = operands[0]->getValue();

    FailureOr<SmallVector<ConstantIntRanges>> ranges =
        intersectDimBoundsWithScalarBounds(
            getScalarLatticeValues(withOp.getShape()), results[0]);
    if (failed(ranges)) {
      propagateIfChanged(operands[0], operands[0]->meet(BoundsArray()));
      return success();
    }
    LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
               << "inferred " << BoundsArray(*ranges) << " for\n\t" << withOp
               << "\n");

    ChangeResult changed = operands[0]->meet(BoundsArray(std::move(*ranges)));

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(withOp->getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>() &&
             isa<LoopLikeOpInterface>(op->getParentOp());
    });
    if (isYieldedResult && !operands[0]->getValue().isUninitialized() &&
        !(operands[0]->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= operands[0]->join(
          BoundsArray::getMaxRangeForShapeBounds(withOp.getOperand()));
    }
    propagateIfChanged(operands[0], changed);
  }
  return success();
}

void ShapeBoundsBackwardsAnalysis::visitBranchOperand(OpOperand &operand) {
  LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "visiting non-forwarded branch operand ";
             operand.get().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}
void ShapeBoundsBackwardsAnalysis::visitCallOperand(OpOperand &operand) {
  LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "visiting non-forwarded call operand ";
             operand.get().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}

//===----------------------------------------------------------------------===//
// ShapeIntegerRangeAnalysis
//===----------------------------------------------------------------------===//

static std::optional<ConstantIntRanges>
maybeGetValueBounds(Value value, std::optional<int64_t> linearIndex) {
  Type elType = cast<RankedTensorType>(value.getType()).getElementType();
  assert(elType.isSignlessIntOrIndex() && "expected integer or index type");
  unsigned bitWidth = elType.isIndex() ? IndexType::kInternalStorageBitWidth
                                       : elType.getIntOrFloatBitWidth();
  std::optional<tensorrt::ShapeProfileAttr> bound = maybeGetFunctionArgBound(
      value, tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName());
  if (!bound)
    return {};
  ArrayRef<int64_t> min = bound->getMin();
  ArrayRef<int64_t> max = bound->getMax();
  if (!linearIndex)
    return ConstantIntRanges::fromSigned(
        APInt(bitWidth, *std::min_element(min.begin(), min.end())),
        APInt(bitWidth, *std::max_element(max.begin(), max.end())));
  return ConstantIntRanges::fromSigned(APInt(bitWidth, min[*linearIndex]),
                                       APInt(bitWidth, max[*linearIndex]));
}

static ConstantIntRanges
getMaxDimRange(unsigned bitWidth = IndexType::kInternalStorageBitWidth) {
  return ConstantIntRanges::fromSigned(
      APInt(bitWidth, 0), APInt(bitWidth, std::numeric_limits<int32_t>::max()));
}
static ConstantIntRanges getMaxValueRange(unsigned bitWidth) {
  return ConstantIntRanges::fromSigned(APInt::getSignedMinValue(bitWidth),
                                       APInt::getSignedMaxValue(bitWidth));
}

static ConstantIntRanges truncateToNonNegative(const ConstantIntRanges &lhs) {
  APInt zero(lhs.smin().getBitWidth(), 0);
  const APInt &smin = lhs.smin().sgt(zero) ? lhs.smin() : zero;
  return ConstantIntRanges::fromSigned(smin, lhs.smax());
}

/// If the `extractOp` indices can be trivially folded to constants, then
/// return the equivalent static linear index.
static std::optional<int64_t> getLinearIndex(tensor::ExtractOp extractOp) {
  if (!extractOp.getTensor().getType().hasStaticShape())
    return {};

  int64_t linearIndex = 0;
  SmallVector<int64_t> indices;
  for (Value v : extractOp.getIndices()) {
    APInt index;
    if (!matchPattern(v, m_ConstantInt(&index)))
      return {};
    indices.push_back(index.getSExtValue());
  }

  if (!indices.empty()) {
    SmallVector<int64_t> basis =
        mlir::computeSuffixProduct(extractOp.getTensor().getType().getShape());
    linearIndex = mlir::linearize(indices, basis);
  }
  return linearIndex;
}

static void inferResultRanges(tensor::DimOp dimOp,
                              ArrayRef<ConstantIntRanges> ranges,
                              SetIntRangeFn setResultRanges) {
  TensorType tensorType = dimOp.getSource().getType();

  std::optional<int64_t> staticDimNum =
      getConstantIntValue(dimOp.getDimension());
  if (!staticDimNum) {
    if (!tensorType.hasStaticShape())
      return setResultRanges(dimOp.getResult(), getMaxDimRange());
    auto shape = tensorType.getShape();
    int64_t minVal = *std::min_element(shape.begin(), shape.end());
    int64_t maxVal = *std::max_element(shape.begin(), shape.end());
    return setResultRanges(
        dimOp.getResult(),
        ConstantIntRanges::fromSigned(
            APInt(IndexType::kInternalStorageBitWidth, minVal),
            APInt(IndexType::kInternalStorageBitWidth, maxVal)));
  }

  if (!tensorType.isDynamicDim(*staticDimNum)) {
    APInt intStatic(IndexType::kInternalStorageBitWidth,
                    tensorType.getDimSize(*staticDimNum));
    return setResultRanges(dimOp.getResult(),
                           ConstantIntRanges::fromSigned(intStatic, intStatic));
  }

  std::optional<tensorrt::ShapeProfileAttr> shapeProfile =
      maybeGetFunctionArgBound(
          dimOp.getSource(),
          tensorrt::TensorRTDialect::getShapeProfileArgAttrName());
  if (!shapeProfile)
    return setResultRanges(dimOp.getResult(), getMaxDimRange());

  setResultRanges(dimOp.getResult(),
                  ConstantIntRanges::fromSigned(
                      APInt(IndexType::kInternalStorageBitWidth,
                            shapeProfile->getMin()[*staticDimNum]),
                      APInt(IndexType::kInternalStorageBitWidth,
                            shapeProfile->getMax()[*staticDimNum])));
}

static void inferResultRanges(tensor::ExtractOp extractOp,
                              ArrayRef<ConstantIntRanges> ranges,
                              SetIntRangeFn setResultRanges) {

  assert(extractOp.getResult().getType().isIntOrIndex() &&
         "expected index or integer type result");
  TensorType operandType = extractOp.getTensor().getType();
  Type elType = extractOp.getType();
  if (!elType.isSignlessIntOrIndex())
    return;
  unsigned bitWidth = elType.isIndex() ? IndexType::kInternalStorageBitWidth
                                       : elType.getIntOrFloatBitWidth();
  if (!operandType.hasStaticShape())
    return setResultRanges(extractOp.getResult(), getMaxValueRange(bitWidth));

  std::optional<int64_t> linearIndex = getLinearIndex(extractOp);
  std::optional<ConstantIntRanges> intRange =
      maybeGetValueBounds(extractOp.getTensor(), linearIndex);
  if (!intRange)
    return setResultRanges(extractOp.getResult(), getMaxValueRange(bitWidth));

  return setResultRanges(extractOp.getResult(), *intRange);
}

/// At an entry point, we cannot reason about integer value ranges except if
/// we are in a function where the bounds are encoded into the arguments.
void ShapeIntegerRangeAnalysis::setToEntryState(
    IntegerValueRangeLattice *lattice) {
  bool isShapeFunc = hasShapeFuncMarker(lattice->getAnchor(),
                                        PlanDialect::kShapeFuncMarkerAttrName);
  // No need to perform integer range analysis for shape func
  if (isShapeFunc) {
    LLVM_DEBUG(DBGS(
        "Skipping shape integer analysis for a shape function. "
        "We can not reason about integer value range for a shape function"));
    return propagateIfChanged(
        lattice,
        lattice->join(IntegerValueRange::getMaxRange(lattice->getAnchor())));
  }

  bool hasShapeUser =
      llvm::any_of(lattice->getAnchor().getUsers(), [&](Operation *user) {
        return isa<plan::WithShapeOp>(user);
      });

  if (!lattice->getAnchor().getType().isIntOrIndex())
    return propagateIfChanged(lattice, lattice->join(IntegerValueRange()));

  std::optional<tensorrt::ShapeProfileAttr> shapeProfile =
      maybeGetFunctionArgBound(
          lattice->getAnchor(),
          tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName());
  if (!shapeProfile) {
    IntegerValueRange range =
        IntegerValueRange::getMaxRange(lattice->getAnchor());
    if (hasShapeUser)
      range = IntegerValueRange(truncateToNonNegative(range.getValue()));
    return propagateIfChanged(lattice, lattice->join(range));
  }
  assert(shapeProfile->getMax().size() == 1 &&
         "expected one element for scalar value bounds");
  Type intType = lattice->getAnchor().getType();
  unsigned bitWidth = intType.isIndex() ? IndexType::kInternalStorageBitWidth
                                        : intType.getIntOrFloatBitWidth();
  return propagateIfChanged(
      lattice, lattice->join(IntegerValueRange(ConstantIntRanges::fromSigned(
                   APInt(bitWidth, shapeProfile->getMin()[0]),
                   APInt(bitWidth, shapeProfile->getMax()[0])))));
}

/// Visit an operation. Invoke the transfer function on each operation that
/// implements `InferIntRangeInterface`.
LogicalResult ShapeIntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {
  for (const IntegerValueRangeLattice *lattice : operands) {
    if (lattice->getAnchor().getType().isIntOrIndex() &&
        lattice->getValue().isUninitialized())
      return success();
  }

  SmallVector<ConstantIntRanges> argRanges(
      llvm::map_range(operands, [](const IntegerValueRangeLattice *val) {
        if (!val->getAnchor().getType().isIntOrIndex())
          return ConstantIntRanges::maxRange(64);
        return val->getValue().getValue();
      }));

  /// The join callback, reproduced from
  /// third_party/llvm-project/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp.
  auto setResultRanges = [&](Value v, const ConstantIntRanges &range) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();
    ConstantIntRanges baseRange = range;
    if (llvm::any_of(v.getUsers(), [](Operation *user) {
          return isa<plan::WithShapeOp>(user);
        }))
      baseRange = truncateToNonNegative(baseRange);

    LLVM_DEBUG(DBGS("ShapeIntegerRange")
               << " inferred range " << baseRange << " for:\n\t" << v << "\n");

    ChangeResult changed = lattice->join(IntegerValueRange{baseRange});

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(DBGS("ShapeIntegerRange")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerValueRange::getMaxRange(v));
    }
    propagateIfChanged(lattice, changed);
  };

  if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (!extractOp.getType().isIntOrIndex()) {
      propagateIfChanged(results[0], results[0]->join(IntegerValueRange()));
      return success();
    }
    inferResultRanges(extractOp, argRanges, setResultRanges);
    return success();
  }
  if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
    inferResultRanges(dimOp, argRanges, setResultRanges);
    return success();
  }

  return IntegerRangeAnalysis::visitOperation(op, operands, results);
}

//===----------------------------------------------------------------------===//
// TensorValueBoundsAnalysis
//===----------------------------------------------------------------------===//

bool TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(Type type) {
  if (auto rtt = dyn_cast<RankedTensorType>(type))
    return rtt.getElementType().isSignlessIntOrIndex() &&
           rtt.hasStaticShape() && rtt.getNumElements() <= kMaxVolumeThreshold;
  return false;
}
bool TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(Value value) {
  return shouldAnalyzeValueBounds(value.getType());
}

void TensorValueBoundsAnalysis::setToEntryState(
    TensorValueBoundsLattice *lattice) {
  Value point = lattice->getAnchor();
  if (!shouldAnalyzeValueBounds(point))
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  std::optional<tensorrt::ShapeProfileAttr> shapeProfile =
      maybeGetFunctionArgBound(
          point,
          tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName());
  if (!shapeProfile)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  return propagateIfChanged(
      lattice, lattice->join(BoundsArray::fromIntegerValueBounds(
                   ConstantIntRanges::getStorageBitwidth(
                       mlir::getElementTypeOrSelf(point.getType())),
                   shapeProfile->getMin(), shapeProfile->getMax())));
}

static void maybePopulateConstantValueBounds(
    Value point,
    llvm::function_ref<void(Value, ArrayRef<ConstantIntRanges>)> joinCallback) {
  if (!TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(point))
    return;
  DenseIntElementsAttr attr;
  if (!matchPattern(point, m_Constant(&attr)))
    return;
  BoundsArray val = BoundsArray::getFromConstantValue(attr);
  joinCallback(point, val.getValue());
}

static FailureOr<SmallVector<ConstantIntRanges>>
intersectTensorValueBoundsAndScalarBounds(
    ArrayRef<const IntegerValueRangeLattice *> scalarBounds,
    const TensorValueBoundsLattice *tensorBounds) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(scalarBounds.size());
  for (unsigned i = 0, e = scalarBounds.size(); i < e; i++) {
    const IntegerValueRangeLattice *scalar = scalarBounds[i];
    bool scalarIsInvalid = !scalar || scalar->getValue().isUninitialized();
    if (tensorBounds && !tensorBounds->getValue().isUninitialized()) {
      if (!scalarIsInvalid) {
        ranges.push_back(scalar->getValue().getValue().intersection(
            tensorBounds->getValue().getValue()[i]));
        continue;
      }
      ranges.push_back(tensorBounds->getValue().getValue()[i]);
      continue;
    }

    if (!scalarIsInvalid) {
      ranges.push_back(scalarBounds[i]->getValue().getValue());
      continue;
    }

    return failure();
  }
  return ranges;
}

LogicalResult TensorValueBoundsAnalysis::visitOperation(
    Operation *op, ArrayRef<const TensorValueBoundsLattice *> operands,
    ArrayRef<TensorValueBoundsLattice *> results) {

  LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis") << "visiting " << *op << "\n");

  auto joinCallback = [&](Value v, ArrayRef<ConstantIntRanges> attrs) {
    assert(shouldAnalyzeValueBounds(v) && "value is unsuitable for analysis");
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    TensorValueBoundsLattice *lattice = results[result.getResultNumber()];
    const BoundsArray &oldRanges = lattice->getValue();
    BoundsArray newRange{llvm::to_vector(attrs)};

    LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis")
               << "inferred " << newRange << " for\n\t" << v << "\n");

    ChangeResult changed = lattice->join(newRange);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRanges.isUninitialized() &&
        !(lattice->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(BoundsArray::getMaxRangeForValueBounds(v));
    }
    propagateIfChanged(lattice, changed);
  };

  // If the value is produced by constant op, populate ranges appropriately.
  // NOTE: we should instead use the mechanism from ConstantIntRanges lattice?
  for (TensorValueBoundsLattice *lattice : results) {
    Value point = lattice->getAnchor();
    if (!shouldAnalyzeValueBounds(point))
      continue;
    maybePopulateConstantValueBounds(point, joinCallback);
  }

  auto getScalarLatticeValues = [&](ValueRange scalars) {
    SmallVector<const IntegerValueRangeLattice *> result;
    result.reserve(scalars.size());
    for (Value v : scalars)
      result.emplace_back(
          this->getOrCreateFor<IntegerValueRangeLattice>(op, v));
    return result;
  };

  if (auto withOp = dyn_cast<plan::WithValuesOp>(op)) {
    FailureOr<SmallVector<ConstantIntRanges>> ranges =
        intersectTensorValueBoundsAndScalarBounds(
            getScalarLatticeValues(withOp.getElements()), operands[0]);
    if (failed(ranges))
      return success();
    joinCallback(withOp.getResult(), *ranges);
    return success();
  }
  return success();
}

void TensorValueBoundsAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<TensorValueBoundsLattice *> argLattices, unsigned firstIndex) {}
