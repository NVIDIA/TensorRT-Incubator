//===- StablehloToTensorRT.cpp --------------------------------------------===//
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
/// Implementation of pass to convert StableHLO ops to TensorRT dialect ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "ControlFlowOps.h"
#include "Matchers.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Conversion/Patterns.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Utils/GatherScatterUtils.h"
#include "mlir-tensorrt/Transforms/StablehloInputPreprocessing/StablehloPrepareScatter.h"
#include "mlir-tensorrt/Transforms/StablehloMatchers/StablehloMatchers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/StringExtras.h"
#include <functional>
#include <numeric>
#include <regex>

#define DEBUG_TYPE "stablehlo-to-tensorrt"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTENSORRTPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using mlir::tensorrt::TensorValue;

/// Convert an DenseIntElementsAttr of into i32 integers. Emit an error at the
/// given location and return failure if the indices can't be safely truncated
/// into i32.
static FailureOr<SmallVector<int32_t>>
truncateI64ToI32(Location loc, ArrayRef<int64_t> i64Indices) {
  SmallVector<int32_t> result;
  result.reserve(i64Indices.size());
  for (int64_t el : i64Indices) {
    auto trunc = static_cast<int32_t>(el);
    if (static_cast<int64_t>(trunc) != el)
      return emitError(loc)
             << "could not safely truncate 64-bit integer to 32-bit: "
             << IntegerAttr::get(IntegerType::get(loc->getContext(), 64), el);
    result.push_back(trunc);
  }
  return result;
}

/// Return an ArrayAttr assumed to hold IntegerAttr values as a vector of
/// integers.
template <typename T = int64_t,
          typename std::enable_if_t<std::is_integral_v<T>, T *> = nullptr>
SmallVector<T> getElementsAttrAsIntVector(DenseIntElementsAttr attr) {
  assert(attr.getType().getRank() == 1);
  return llvm::to_vector(
      llvm::map_range(attr.getValues<APInt>(), [](APInt intAttr) -> T {
        return static_cast<T>(intAttr.getLimitedValue());
      }));
}

/// Return an ArrayAttr assumed to hold IntegerAttr values as a vector of
/// integers (same as above, but accepts optional vector).
template <typename T = int64_t,
          typename std::enable_if_t<std::is_integral_v<T>, T *> = nullptr>
SmallVector<T>
getElementsAttrAsIntVector(std::optional<DenseIntElementsAttr> attr) {
  if (!attr.has_value())
    return SmallVector<T>{};
  assert(attr->getType().getRank() == 1);
  return llvm::to_vector(
      llvm::map_range(attr->getValues<APInt>(), [](APInt intAttr) -> T {
        return static_cast<T>(intAttr.getLimitedValue());
      }));
}

/// Return true if the given integer elements are splatted from the given value.
static bool isSplat(Attribute genericAttr, int64_t value) {
  if (auto attr = dyn_cast<DenseIntElementsAttr>(genericAttr))
    return attr.isSplat() && attr.getSplatValue<int64_t>() == value;
  if (auto attr = dyn_cast<DenseI64ArrayAttr>(genericAttr))
    return llvm::all_of(attr.asArrayRef(),
                        [&](int64_t element) { return element == value; });
  llvm_unreachable("unexpected kind of attribute encountered");
}

// Convert a Nx2 dense int64 padding attribute into a list for pre-padding and a
// list for post-padding. Adapted from a similar hlo helper op.
static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
convertPaddingAttribute(Location loc,
                        std::optional<DenseIntElementsAttr> optionalAttr) {
  if (!optionalAttr.has_value())
    return std::make_pair(SmallVector<int64_t>{}, SmallVector<int64_t>{});
  DenseIntElementsAttr attr = *optionalAttr;
  auto attrType = cast<RankedTensorType>(attr.getType());
  if (attrType.getRank() != 2 || attrType.getShape()[1] != 2)
    return emitOptionalError(
        loc, "expects the shape of padding-attribute to be {N, 2}, but got {",
        attrType.getShape(), "}.");
  auto it = attr.getValues<int64_t>().begin();
  SmallVector<int64_t> prePadding(attr.getNumElements() / 2);
  SmallVector<int64_t> postPadding(attr.getNumElements() / 2);
  for (int64_t i : llvm::seq<int64_t>(0, attr.getNumElements() / 2)) {
    prePadding[i] = *it;
    ++it;
    postPadding[i] = *it;
    ++it;
  }
  return std::make_pair(prePadding, postPadding);
}

/// If `v` is produced by a broadcast operation (`stablehlo.broadcast_in_dim`),
/// return the input to the broadcast and the broadcast dimensions attribute.
/// Otherwise, return failure.
static FailureOr<TensorValue>
matchBroadcastedValue(Value v, SmallVector<int64_t> &broadcastDims) {
  Operation *broadcastOp = v.getDefiningOp();
  if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(broadcastOp)) {
    broadcastDims = llvm::to_vector(bcastOp.getBroadcastDimensions());
    return cast<TensorValue>(bcastOp.getOperand());
  }
  if (auto bcastOp = dyn_cast<tensorrt::BroadcastOp>(broadcastOp)) {
    broadcastDims = llvm::to_vector(bcastOp.getBroadcastDims());
    return cast<TensorValue>(bcastOp.getInput());
  }
  return failure();
}

/// If `v` is produced by a broadcast operation (`stablehlo.broadcast_in_dim`),
/// and the source of the broadcast is an `iota` operation, then return the
/// dimension of `v` that corresponds to the `iota` result.
static FailureOr<int64_t> matchBroadcastedIota1d(Value v) {
  SmallVector<int64_t> broadcastDims;
  FailureOr<TensorValue> broadcastedValue =
      matchBroadcastedValue(v, broadcastDims);
  if (failed(broadcastedValue))
    return failure();
  if (broadcastDims.size() != 1)
    return failure();
  if (!matchPattern(*broadcastedValue, m_Op<stablehlo::IotaOp>()) &&
      !matchPattern(*broadcastedValue, m_Op<tensorrt::LinspaceOp>()))
    return failure();
  return broadcastDims.front();
}

// Checks whether second operand passed to the `stablehlo.sort` is indices
// generated by either iota or iota+broadcast_in_dim.
static LogicalResult doesOperandRepresentIndices(stablehlo::SortOp op,
                                                 Value indexOperand) {
  if (dyn_cast<RankedTensorType>(op->getOperands()[0].getType()).getRank() ==
      1) {
    // Check that indices are formed from iota
    Operation *maybeIotaOp = indexOperand.getDefiningOp();
    if (!matchPattern(maybeIotaOp, m_Op<stablehlo::IotaOp>()) &&
        !matchPattern(maybeIotaOp, m_Op<tensorrt::LinspaceOp>()))
      return failure();
  } else {
    // Check that indices are formed from iota->broadcast.
    FailureOr<int64_t> broadcastDim = matchBroadcastedIota1d(indexOperand);
    if (failed(broadcastDim))
      return failure();

    // Check that the iota dimension corresponds to the dimension we are
    // sorting.
    if (*broadcastDim != static_cast<int64_t>(op.getDimension()))
      return failure();
  }
  return success();
}

namespace {
// Matches a suitable `stablehlo.sort` operation and transforms it into a
// `tensorrt.top_k` operation. The following two case are supported,
// A. The `stablehlo.sort` operation has a single input operand: indices output
// of `tensorrt.top_k` is ignored in this case.
// B. The `stablehlo.sort` operation has two input operands: the values of the
// operand and the indices of the operand to sorted. The TF to stablehlo
// conversion will ensure the indices are produced by a combination of
// `stablehlo.iota` and `stablehlo.broadcast_in_dim`.
struct SortToTopK : public ConvertHloOpToTensorRTPattern<stablehlo::SortOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::SortOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We support `stablehlo::SortOp` op with one or two (conditional) operands.
    if (op->getOperands().size() > 2)
      return failure();

    // In the comparator block, we should have two operations (compare and
    // return).
    Block *comparatorBlock = &op.getComparator().front();
    if (comparatorBlock->getOperations().size() != 2)
      return failure();
    auto compareOp = dyn_cast<stablehlo::CompareOp>(
        comparatorBlock->getOperations().front());
    if (!compareOp)
      return failure();

    tensorrt::TopKOperation topKOpType;
    stablehlo::ComparisonDirection dir = compareOp.getComparisonDirection();
    if (dir == stablehlo::ComparisonDirection::GE ||
        dir == stablehlo::ComparisonDirection::GT)
      topKOpType = tensorrt::TopKOperation::kMAX;
    else if (dir == stablehlo::ComparisonDirection::LE ||
             dir == stablehlo::ComparisonDirection::LT)
      topKOpType = tensorrt::TopKOperation::kMIN;
    else
      return failure();

    // TODO: Support the dynamic sort case when more extension operations
    // corresponding to `tensor.dim` are added.
    TensorValue valueOperand = cast<TensorValue>(adaptor.getOperands()[0]);
    TensorType inputType = valueOperand.getType();
    if (inputType.isDynamicDim(op.getDimension()))
      return failure();

    // I32 is unsupported, so cast to fp32.
    if (inputType.getElementType().isInteger(32)) {
      inputType = RankedTensorType::Builder(cast<RankedTensorType>(inputType))
                      .setElementType(rewriter.getF32Type());
      valueOperand = castTensor(rewriter, inputType, valueOperand);
    }

    // 1D input is not supported, so we expand it by appending 1
    if (inputType.getRank() == 1) {
      SmallVector<int64_t> expandedShape{inputType.getShape().front(), 1};
      inputType =
          RankedTensorType::get(expandedShape, inputType.getElementType());
      valueOperand = rewriter.create<tensorrt::ExpandRankOp>(
          op->getLoc(), inputType, valueOperand);
    }

    // Max value of `k` can be 3840, as of TRT 8.6.1.
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_top_k_layer.html#a315ea9c13f1fb0affc93361fb5cc876a
    if (inputType.getDimSize(op.getDimension()) > 3840)
      return failure();

    // We have both values and indices as input
    if (op->getOperands().size() == 2) {
      Value indexOperand = adaptor.getOperands()[1];
      if (failed(doesOperandRepresentIndices(op, indexOperand)))
        return failure();
      auto topKOp = rewriter.create<tensorrt::TopKOp>(
          op.getLoc(), valueOperand,
          /*k=*/inputType.getDimSize(op.getDimension()),
          /*axis=*/static_cast<int64_t>(op.getDimension()), topKOpType);
      auto topKOpValues = topKOp.getValues();
      auto topKOpIndices = topKOp.getIndices();

      if (topKOp.getType(0) != op.getType(0))
        topKOpValues = castTensor(rewriter, op.getType(0), topKOpValues);

      if (topKOpValues.getType().getRank() !=
          dyn_cast<RankedTensorType>(op.getType(0)).getRank()) {
        topKOpValues = rewriter.create<tensorrt ::CollapseRankOp>(
            op->getLoc(), dyn_cast<RankedTensorType>(op.getType(0)),
            topKOpValues);
        topKOpIndices = rewriter.create<tensorrt ::CollapseRankOp>(
            op->getLoc(), dyn_cast<RankedTensorType>(op.getType(1)),
            topKOpIndices);
      }

      rewriter.replaceOp(op, {topKOpValues, topKOpIndices});
      return success();
    }

    // We have only values as input
    auto topKOpValues =
        rewriter
            .create<tensorrt::TopKOp>(
                op.getLoc(), valueOperand,
                /*k=*/inputType.getDimSize(op.getDimension()),
                /*axis=*/static_cast<int64_t>(op.getDimension()), topKOpType)
            .getValues();

    if (topKOpValues.getType() != op.getType(0))
      topKOpValues = castTensor(rewriter, op.getType(0), topKOpValues);

    if (topKOpValues.getType().getRank() !=
        dyn_cast<RankedTensorType>(op.getType(0)).getRank())
      topKOpValues =
          rewriter
              .create<tensorrt ::CollapseRankOp>(
                  op->getLoc(), dyn_cast<RankedTensorType>(op.getType(0)),
                  topKOpValues)
              .getResult();

    rewriter.replaceOp(op, topKOpValues);
    return success();
  }
};
} // namespace

/// Given a stablehlo reduction operation, convert to a `tensorrt.reduce`
/// operation if it is a simple reduction (e.g. sum, mul, max/min) that be
/// converted 1-1. Caller must do the replacement, this just creates the new
/// operation and returns the new value.
static FailureOr<Value> convertSimpleReductions(RewriterBase &rewriter,
                                                stablehlo::ReduceOp op,
                                                int64_t reductionDim,
                                                Value input, Value init) {
  // TODO: verify the init is the neutral value based on the op below.
  if (!matchPattern(init, m_Constant()))
    return failure();

  Block *reduceBody = &op.getBody().front();
  auto termOp = cast<stablehlo::ReturnOp>(reduceBody->getTerminator());
  if (termOp->getNumOperands() != 1 || reduceBody->getNumArguments() != 2)
    return failure();

  Location loc = op.getLoc();
  Value retValue = termOp.getOperands()[0];
  auto bbLhs = matchers::m_Val(reduceBody->getArgument(0));
  auto bbRhs = matchers::m_Val(reduceBody->getArgument(1));

  tensorrt::ReduceOperation reductionOp;
  if (matchPattern(retValue, m_Op<stablehlo::AddOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kSUM;
  else if (matchPattern(retValue, m_Op<stablehlo::MulOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kPROD;
  else if (matchPattern(retValue, m_Op<stablehlo::MinOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kMIN;
  else if (matchPattern(retValue, m_Op<stablehlo::MaxOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kMAX;
  else
    return failure();

  return rewriter
      .create<tensorrt::ReduceOp>(loc, op.getType(0), input,
                                  /*reduceDims=*/
                                  SmallVector<int64_t>{reductionDim},
                                  /*keepdims=*/false, reductionOp)
      .getResult();
}

static FailureOr<Value> convertBooleanReductions(RewriterBase &rewriter,
                                                 stablehlo::ReduceOp op,
                                                 int64_t reductionDim,
                                                 Value input, Value init) {
  Location loc = op.getLoc();
  // Create an int32 tensor types equivalent to the boolean tensor types.
  auto originalInputType = cast<RankedTensorType>(input.getType());
  auto originalResultType = cast<RankedTensorType>(op->getResultTypes()[0]);
  if (!originalResultType.getElementType().isInteger(1) ||
      !originalInputType.getElementType().isInteger(1))
    return failure();

  RankedTensorType integerInputType =
      RankedTensorType::Builder(originalInputType)
          .setElementType(rewriter.getI32Type());
  RankedTensorType integerResultType =
      RankedTensorType::Builder(originalResultType)
          .setElementType(rewriter.getI32Type());

  // Create the new reduction type.
  Block *reduceBody = &op.getBody().front();
  auto termOp = cast<stablehlo::ReturnOp>(reduceBody->getTerminator());
  if (termOp->getNumOperands() != 1 || reduceBody->getNumArguments() != 2)
    return failure();
  Value retValue = termOp.getOperands()[0];
  auto bbLhs = matchers::m_Val(reduceBody->getArgument(0));
  auto bbRhs = matchers::m_Val(reduceBody->getArgument(1));
  tensorrt::ReduceOperation reductionOpType;
  if (matchPattern(retValue, m_Op<stablehlo::OrOp>(bbLhs, bbRhs)))
    reductionOpType = tensorrt::ReduceOperation::kSUM;
  else if (matchPattern(retValue, m_Op<stablehlo::AndOp>(bbLhs, bbRhs)))
    reductionOpType = tensorrt::ReduceOperation::kPROD;
  else
    return failure();

  // Cast i1 to i32.
  Value i32Input =
      rewriter.create<tensorrt::IdentityOp>(loc, integerInputType, input);

  auto reduceOp = rewriter.create<tensorrt::ReduceOp>(
      loc, integerResultType, i32Input,
      /*reduceDims=*/SmallVector<int64_t>{reductionDim},
      /*keepdims=*/false, reductionOpType);
  // Cast i32 to i1.
  return rewriter
      .create<tensorrt::IdentityOp>(loc, originalResultType,
                                    reduceOp.getResult())
      .getResult();
}

/// Return true if `x` is a sequence starting from `x[0]` and incrementing by 1.
static bool isContiguousSequence(ArrayRef<int64_t> x) {
  return llvm::equal(x, llvm::seq<int64_t>(x.front(), x.front() + x.size()));
}

/// Given a tensor `t` and a contiguous set of dimensions `[firstDimToCollapse,
/// lastDimToCollapse]` (inclusive), return a new type where the specified
/// dimensions have been flatttend into a single dimension.
static RankedTensorType getCollapsedShape(RankedTensorType t,
                                          unsigned firstDimToCollapse,
                                          unsigned numDimsToCollapse) {
  assert(numDimsToCollapse >= 1 &&
         "expected at least one dimension to collapse");

  // Prepend non-collapsed dimensions.
  SmallVector<int64_t> newShape(t.getShape().take_front(firstDimToCollapse));
  // Flatten the reduced dimensions.
  ArrayRef<int64_t> view =
      t.getShape().slice(firstDimToCollapse, numDimsToCollapse);
  newShape.push_back(
      std::accumulate(view.begin(), view.end(), 1, std::multiplies<>()));

  // Append the trailing non-collapsed dims.
  if (firstDimToCollapse + numDimsToCollapse != t.getRank())
    llvm::append_range(
        newShape, t.getShape().take_back(
                      t.getRank() - (firstDimToCollapse + numDimsToCollapse)));
  return t.clone(newShape);
}

/// Drop the unit dimension at `dimToDrop` from each of `values`.
static SmallVector<Value> createRankReducedResults(RewriterBase &rewriter,
                                                   Location loc,
                                                   ResultRange values,
                                                   int64_t dimToDrop) {
  assert(!values.empty());
  SmallVector<Value> result;
  result.reserve(values.size());
  for (Value v : values) {
    auto inputType = dyn_cast<RankedTensorType>(v.getType());
    assert((!inputType || inputType.getDimSize(dimToDrop) == 1) &&
           "expected value to have unit dim to drop");
    auto rtt = RankedTensorType::Builder(inputType);
    rtt.dropDim(dimToDrop);
    Value collapsed =
        rewriter.create<tensorrt::CollapseRankOp>(loc, Type(rtt), v);
    result.push_back(collapsed);
  }
  return result;
}

template <typename TensorRTOpType, stablehlo::ComparisonDirection dir>
static LogicalResult
matchAndReplaceStablehloArgMinMax(stablehlo::ReduceOp op,
                                  RewriterBase &rewriter, Value operand,
                                  ArrayRef<int64_t> reductionDims) {
  if (!matchPattern(op,
                    matchers::detail::StablehloArgMinMaxReduceMatcher<dir>()))
    return failure();
  auto argMaxOp = rewriter.create<TensorRTOpType>(
      op.getLoc(),
      /*input=*/operand, /*axis=*/reductionDims.front());
  // Rank reduce the results.
  SmallVector<Value> replacements = createRankReducedResults(
      rewriter, op.getLoc(), argMaxOp->getResults(), reductionDims.front());
  rewriter.replaceOp(op, replacements);
  return success();
}

namespace {
// Converts a `stablehlo.reduce` operation to a `tensorrt.reduce` operation.
struct ConvertReduceOp
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReduceOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value operand = adaptor.getInputs().front();
    auto inputType = cast<RankedTensorType>(operand.getType());
    SmallVector<int64_t> reductionDims = llvm::to_vector(op.getDimensions());

    // Try to match and handle the ArgMin/ArgMax cases.
    if (succeeded(matchAndReplaceStablehloArgMinMax<
                  tensorrt::ArgMaxOp, stablehlo::ComparisonDirection::GE>(
            op, rewriter, operand, reductionDims)))
      return success();
    if (succeeded(matchAndReplaceStablehloArgMinMax<
                  tensorrt::ArgMinOp, stablehlo::ComparisonDirection::LE>(
            op, rewriter, operand, reductionDims)))
      return success();

    // TRT can only handle single reduction dims right now. We can support
    // multiple contiguous reduction dims by collapsing them.
    if (reductionDims.size() != 1) {
      llvm::sort(reductionDims);
      if (!isContiguousSequence(reductionDims))
        return failure();
      operand = rewriter.create<tensorrt::ReshapeOp>(
          op.getLoc(),
          getCollapsedShape(inputType, reductionDims.front(),
                            reductionDims.size()),
          operand);
      reductionDims = SmallVector<int64_t>{reductionDims.front()};
    }

    // If not in above special cases, try to match the simpler reductions across
    // a single input.
    if (op.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "number of reduction inputs not 1");
    Value init = adaptor.getInitValues().front();

    FailureOr<Value> replacement = convertBooleanReductions(
        rewriter, op, reductionDims.front(), operand, init);
    if (succeeded(replacement)) {
      rewriter.replaceOp(op, *replacement);
      return success();
    }

    replacement = convertSimpleReductions(rewriter, op, reductionDims.front(),
                                          operand, init);
    if (failed(replacement))
      return rewriter.notifyMatchFailure(
          op, "could not do simple reduction transform");
    rewriter.replaceOp(op, *replacement);
    return success();
  }
};

/// Convert `stablehlo.dot` to `tensorrt.matrix_multiply`.
/// TODO: clean since `dot` op is removed from stable hlo in the favor of
/// `dot_general`.
struct ConvertDot : public ConvertHloOpToTensorRTPattern<stablehlo::DotOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::DotOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType resultType = op.getType();
    tensorrt::MatrixOperation qualifierLhs = tensorrt::MatrixOperation::kNONE;
    tensorrt::MatrixOperation qualifierRhs = tensorrt::MatrixOperation::kNONE;
    auto lhsType = cast<TensorType>(adaptor.getLhs().getType());
    auto rhsType = cast<TensorType>(adaptor.getRhs().getType());
    if (lhsType.getRank() == 1)
      qualifierLhs = tensorrt::MatrixOperation::kVECTOR;
    if (rhsType.getRank() == 1)
      qualifierRhs = tensorrt::MatrixOperation::kVECTOR;

    auto replaceWithMaybeCast = [&](TensorValue replacement) {
      if (replacement.getType() != op.getType())
        replacement = castTensor(rewriter, op.getType(), replacement);
      rewriter.replaceOp(op, replacement);
    };

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    // For now, only handle I32 input case.
    if (lhsType.getElementType().isInteger(32)) {
      resultType =
          cast<RankedTensorType>(op.getType()).clone(rewriter.getF32Type());
      lhs = castTensor(rewriter, rewriter.getF32Type(), cast<TensorValue>(lhs));
      rhs = castTensor(rewriter, rewriter.getF32Type(), cast<TensorValue>(rhs));
    }

    auto replacement =
        rewriter
            .create<tensorrt::MatrixMultiplyOp>(op->getLoc(), resultType, lhs,
                                                rhs, qualifierLhs, qualifierRhs)
            .getResult();
    replaceWithMaybeCast(replacement);
    return success();
  }
};

/// Convert `stablehlo.dot_general` to `tensorrt.matrix_multiply`.
struct ConvertDotGeneral
    : public ConvertHloOpToTensorRTPattern<stablehlo::DotGeneralOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::DotGeneralOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();
    // Only convert if batch nums agree.
    ArrayRef<int64_t> lhsBatchDims = dimNums.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchDims = dimNums.getRhsBatchingDimensions();
    if (lhsBatchDims.size() != rhsBatchDims.size())
      return failure();

    TensorType resultType = op.getType();
    // Determine the TRT equivalent qualifier.
    tensorrt::MatrixOperation qualifierLhs = tensorrt::MatrixOperation::kNONE;
    tensorrt::MatrixOperation qualifierRhs = tensorrt::MatrixOperation::kNONE;
    TensorType lhsType = op.getLhs().getType();
    TensorType rhsType = op.getRhs().getType();

    auto replaceWithMaybeCast = [&](TensorValue replacement) {
      if (replacement.getType() != op.getType())
        replacement = castTensor(rewriter, op.getType(), replacement);
      rewriter.replaceOp(op, replacement);
    };

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    // For now, only handle I32 input case.
    if (lhsType.getElementType().isInteger(32)) {
      resultType =
          cast<RankedTensorType>(op.getType()).clone(rewriter.getF32Type());
      lhs = castTensor(rewriter, rewriter.getF32Type(), cast<TensorValue>(lhs));
      rhs = castTensor(rewriter, rewriter.getF32Type(), cast<TensorValue>(rhs));
    }

    // When both operand ranks are same and rank equals batch dimensions for
    // each operand, this is an elementwise multiplication op.
    if ((lhsType.getRank() == rhsType.getRank()) &&
        (lhsType.getRank() == static_cast<int64_t>(lhsBatchDims.size())) &&
        (rhsType.getRank() == static_cast<int64_t>(rhsBatchDims.size()))) {
      auto replacement = rewriter
                             .create<tensorrt::ElementWiseOp>(
                                 op->getLoc(), resultType, lhs, rhs,
                                 tensorrt::ElementWiseOperation::kPROD)
                             .getResult();
      replaceWithMaybeCast(replacement);
      return success();
    }

    // Allow reshaping to conform to MatMul requirements.
    if (rhsType.getRank() == static_cast<int64_t>(rhsBatchDims.size()) + 3) {
      if ((lhsType.getRank() + 1 == rhsType.getRank()) &&
          (dimNums.getLhsContractingDimensions().front() ==
           lhsType.getRank() - 1) &&
          (dimNums.getRhsContractingDimensions().front() ==
           rhsType.getRank() - 2)) {

        SmallVector<int64_t> newLhsShape(lhsType.getShape());
        newLhsShape.insert(newLhsShape.end() - 2, 1, 1);
        Value newLhs = rewriter.create<tensorrt::ExpandRankOp>(
            op->getLoc(),
            RankedTensorType::get(newLhsShape, lhsType.getElementType()), lhs);

        auto matMulOp = rewriter.create<tensorrt::MatrixMultiplyOp>(
            op->getLoc(), newLhs, rhs, qualifierLhs, qualifierRhs);

        // Convert the output shape to match the dot_general op's original
        // output shape.
        int64_t rank = resultType.getRank();
        SmallVector<unsigned> perm =
            llvm::to_vector(llvm::seq<unsigned>(0, rank));
        std::swap(perm[rank - 3], perm[rank - 2]);
        auto affineMap = AffineMap::getPermutationMap(perm, op->getContext());
        auto transposeOp = rewriter.create<tensorrt::TransposeOp>(
            op->getLoc(), matMulOp, affineMap);
        rewriter.replaceOp(op, transposeOp.getResult());
        return success();
      }
    }

    if (lhsType.getRank() == static_cast<int64_t>(lhsBatchDims.size()) + 2) {
      if (dimNums.getLhsContractingDimensions().front() ==
          lhsType.getRank() - 1)
        qualifierLhs = tensorrt::MatrixOperation::kNONE;
      else if (dimNums.getLhsContractingDimensions().front() ==
               lhsType.getRank() - 2)
        qualifierLhs = tensorrt::MatrixOperation::kTRANSPOSE;
      else
        return failure();
    } else if (lhsType.getRank() ==
               static_cast<int64_t>(lhsBatchDims.size()) + 1) {
      qualifierLhs = tensorrt::MatrixOperation::kVECTOR;
    } else {
      // TODO: should this be an assert?
      return failure();
    }

    if (rhsType.getRank() == static_cast<int64_t>(rhsBatchDims.size()) + 2) {
      if (dimNums.getRhsContractingDimensions().front() ==
          rhsType.getRank() - 1)
        qualifierRhs = tensorrt::MatrixOperation::kTRANSPOSE;
      else if (dimNums.getRhsContractingDimensions().front() ==
               rhsType.getRank() - 2)
        qualifierRhs = tensorrt::MatrixOperation::kNONE;
      else
        return failure();
    } else if (rhsType.getRank() ==
               static_cast<int64_t>(rhsBatchDims.size()) + 1) {
      qualifierRhs = tensorrt::MatrixOperation::kVECTOR;
    } else {
      // TODO: should this be an assert?
      return failure();
    }

    auto replacement =
        rewriter
            .create<tensorrt::MatrixMultiplyOp>(op->getLoc(), resultType, lhs,
                                                rhs, qualifierLhs, qualifierRhs)
            .getResult();
    replaceWithMaybeCast(replacement);
    return success();
  }
};
} // namespace

/// Given an expression try to find a single-character string from `termPool`
/// that is not used in `expression`. Returns the index of the unused character.
static FailureOr<unsigned> getUnusedTerm(StringRef expression,
                                         StringRef termPool) {
  for (unsigned i = 0; i < termPool.size(); i++) {
    StringRef term = termPool.substr(i, 1);
    if (!expression.contains(term))
      return i;
  }
  return failure();
}

/// Replace capital letters with other lowercase symbols.
static FailureOr<std::string> replaceCapitalSymbols(StringRef equation) {
  constexpr StringRef kTermPool = "abcdefghijklmnopqrstuvwxyz";
  unsigned lastUnused = 0;
  std::string result = equation.str();
  auto replaceInResult = [&](char from, char to) {
    for (char &i : result) {
      if (i == from)
        i = to;
    }
  };
  for (unsigned i = 0; i < result.size(); i++) {
    if (std::isupper(result[i]) == 0)
      continue;
    StringRef termPool = kTermPool.substr(lastUnused);
    FailureOr<unsigned> unusedTermIdx = getUnusedTerm(equation, termPool);
    if (failed(unusedTermIdx))
      return failure();
    lastUnused = lastUnused + *unusedTermIdx + 1;
    replaceInResult(equation[i], termPool[*unusedTermIdx]);
  }
  return result;
}

/// Given the parameters and types of an einsum operation, try to replace the
/// use of the ellipses ("...")  in the result and one of the operands with a
/// fixed set of letters. Note: currently this function only supports ellipses
/// in the lhs operand + the result, but it should be generalized to include
/// other cases.
static FailureOr<std::string>
tryReplaceEllipsis(StringRef input0, TensorType input0Type, StringRef input1,
                   TensorType input1Type, StringRef result,
                   TensorType resultType, StringRef fullEquation) {
  // The ellipsis only appears on the LHS.
  constexpr int64_t kEllipsisSize = 3;
  constexpr StringRef kTermPool = "abcdefghijklmnopqrstuvwxyz";
  if (input0.contains("...") && !input1.contains("...") &&
      result.contains("...")) {
    // Find the number of terms represented by the ellipsis. It should be equal
    // on the lhs and for the result.
    int64_t numTerms = input0Type.getRank() - (input0.size() - kEllipsisSize);
    if (numTerms != resultType.getRank() -
                        (static_cast<int64_t>(result.size()) - kEllipsisSize))
      return failure();
    std::string ellipsisSubst = "";
    unsigned lastUnused = 0;
    for (int64_t i = 0; i < numTerms; i++) {
      StringRef termPool = kTermPool.substr(lastUnused);
      FailureOr<unsigned> unusedTermIdx = getUnusedTerm(fullEquation, termPool);
      if (failed(unusedTermIdx))
        return failure();
      ellipsisSubst += termPool[*unusedTermIdx];
      lastUnused = lastUnused + *unusedTermIdx + 1;
    }

    std::string newLhs = std::regex_replace(
        input0.str(), std::regex(R"(\.\.\.)"), ellipsisSubst);
    std::string newResult = std::regex_replace(
        result.str(), std::regex(R"(\.\.\.)"), ellipsisSubst);
    return llvm::join(
        SmallVector<std::string>{newLhs, ",", input1.str(), "->", newResult},
        "");
  }
  return failure();
}

/// Given information for a `stablehlo.einsum` operation, try to match some
/// common special cases that are currently unsupported by TensorRT. These can
/// be manually.
static FailureOr<Value> handleSpecialEinsum(RewriterBase &rewriter,
                                            Location loc, StringRef equation,
                                            TensorType resultType,
                                            TensorValue lhs, TensorValue rhs) {
  SmallVector<StringRef> frags;
  llvm::SplitString(equation, frags, "->");
  assert(frags.size() == 2 && "expected an input and result fragment");
  SmallVector<StringRef> inputFrags;
  llvm::SplitString(frags[0], inputFrags, ",");
  if (inputFrags.size() != 2)
    return failure();
  StringRef input0 = inputFrags[0];
  StringRef input1 = inputFrags[1];
  StringRef res = frags[1];

  // Try to deduce static variables for the ellipses.
  if (equation.contains("...")) {
    FailureOr<std::string> ellipsisReplaced =
        tryReplaceEllipsis(input0, lhs.getType(), input1, rhs.getType(), res,
                           resultType, equation);
    if (succeeded(ellipsisReplaced))
      return rewriter
          .create<tensorrt::EinsumOp>(loc, resultType, ValueRange{lhs, rhs},
                                      rewriter.getStringAttr(*ellipsisReplaced))
          .getResult();
  }
  return failure();
}

namespace {
struct ConvertEinsum
    : public ConvertHloOpToTensorRTPattern<stablehlo::EinsumOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::EinsumOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::EinsumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        RankedTensorType::Builder(cast<RankedTensorType>(op.getType()))
            .setElementType(rewriter.getF32Type());

    // The result will always be a f32 tensor, so add an identity to convert if
    // required.
    auto replaceWithMaybeCast = [&](TensorValue replacement) {
      if (replacement.getType() != op.getType())
        replacement = castTensor(rewriter, op.getType(), replacement);
      rewriter.replaceOp(op, replacement);
    };

    FailureOr<std::string> equation =
        replaceCapitalSymbols(op.getEinsumConfig());
    if (failed(equation))
      return failure();

    // Try to handle special cases, e.g. replace ellipses if possible.
    FailureOr<Value> specialCase = handleSpecialEinsum(
        rewriter, op.getLoc(), *equation, cast<TensorType>(resultType),
        cast<TensorValue>(adaptor.getLhs()),
        cast<TensorValue>(adaptor.getRhs()));
    if (succeeded(specialCase)) {
      replaceWithMaybeCast(cast<TensorValue>(*specialCase));
      return success();
    }
    // Otherwise, proceed with a 1-1 swap.
    if (op.getEinsumConfig().contains("..."))
      return rewriter.notifyMatchFailure(
          op, "cannot convert einsum equations with ellipsis");
    TensorValue replacement =
        rewriter
            .create<tensorrt::EinsumOp>(
                op.getLoc(), resultType,
                ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                rewriter.getStringAttr(*equation))
            .getResult();
    replaceWithMaybeCast(replacement);
    return success();
  }
};

/// Convert `stablehlo` dialect binary operations to tensorrt
/// elementwise operation in a 1-1 manner.
template <typename HloOpType, tensorrt::ElementWiseOperation EwiseOpEnumValue>
struct HloBinaryOpConverter : public ConvertHloOpToTensorRTPattern<HloOpType> {
  using ConvertHloOpToTensorRTPattern<HloOpType>::ConvertHloOpToTensorRTPattern;

  LogicalResult matchAndRewrite(
      HloOpType op,
      typename ConvertHloOpToTensorRTPattern<HloOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (EwiseOpEnumValue == tensorrt::ElementWiseOperation::kXOR ||
        EwiseOpEnumValue == tensorrt::ElementWiseOperation::kOR ||
        EwiseOpEnumValue == tensorrt::ElementWiseOperation::kAND) {
      Value lhs = adaptor.getLhs();
      Value rhs = adaptor.getRhs();
      auto lhsType = cast<RankedTensorType>(lhs.getType());
      auto rhsType = cast<RankedTensorType>(rhs.getType());
      if (!lhsType.getElementType().isInteger(1) ||
          !rhsType.getElementType().isInteger(1))
        return failure();
    }
    Location loc = op.getLoc();
    auto elementwiseOp = rewriter.create<tensorrt::ElementWiseOp>(
        loc, /*result=*/op.getType(),
        /*input1=*/adaptor.getLhs(),
        /*input2=*/adaptor.getRhs(),
        /*elementwiseOperation=*/EwiseOpEnumValue);
    rewriter.replaceOp(op, elementwiseOp.getResult());
    return success();
  }
};

/// Convert `stablehlo` dialect unary operations to tensorrt
/// unary operation in a 1-1 manner.
template <typename HloOpType, tensorrt::UnaryOperation UnaryOpEnumValue>
struct HloUnaryOpConverter : public ConvertHloOpToTensorRTPattern<HloOpType> {
  using ConvertHloOpToTensorRTPattern<HloOpType>::ConvertHloOpToTensorRTPattern;

  LogicalResult matchAndRewrite(
      HloOpType op,
      typename ConvertHloOpToTensorRTPattern<HloOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto operand = cast<TensorValue>(adaptor.getOperand());
    auto operandType = operand.getType();
    // TensorRT Unary ops need at least 1D tensor
    if (operandType.getRank() == 0) {
      RankedTensorType newShape =
          RankedTensorType::get({1}, operandType.getElementType());
      operand = rewriter.create<tensorrt::ExpandRankOp>(loc, newShape, operand);
    }

    // kNOT only supports integer types, while hlo is bitwise.
    if (!operandType.getElementType().isInteger(1) &&
        UnaryOpEnumValue == tensorrt::UnaryOperation::kNOT)
      return failure();

    // I32 is unsupported, so cast to f32 first.
    if (operandType.getElementType().isInteger(32) &&
        UnaryOpEnumValue != tensorrt::UnaryOperation::kSIGN) {
      operand = HloUnaryOpConverter::castTensor(rewriter, rewriter.getF32Type(),
                                                operand);
    }
    TensorValue result =
        rewriter
            .create<tensorrt::UnaryOp>(loc, /*result=*/operand.getType(),
                                       /*input=*/operand,
                                       /*unaryOperation=*/UnaryOpEnumValue)
            .getResult();
    // Cast back if required.
    if (result.getType().getElementType() != op.getType().getElementType())
      result = HloUnaryOpConverter::castTensor(
          rewriter, cast<RankedTensorType>(op.getType()).getElementType(),
          result);

    if (result.getType().getRank() != op.getType().getRank())
      result =
          rewriter.create<tensorrt::CollapseRankOp>(loc, op.getType(), result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert `stablehlo` dialect unary operations to tensorrt
/// activation operation in a 1-1 manner.
template <typename HloOpType, tensorrt::ActivationType activationType>
struct HloUnaryOpToActivationConverter
    : public ConvertHloOpToTensorRTPattern<HloOpType> {
  using ConvertHloOpToTensorRTPattern<HloOpType>::ConvertHloOpToTensorRTPattern;
  LogicalResult matchAndRewrite(
      HloOpType op,
      typename ConvertHloOpToTensorRTPattern<HloOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto elementwiseOp = rewriter.create<tensorrt::ActivationOp>(
        loc, /*result=*/op.getResult().getType(),
        /*input=*/adaptor.getOperand(),
        /*activationType=*/activationType,
        tensorrt::ActivationOp::requiresAlphaAttribute(activationType)
            ? rewriter.getF32FloatAttr(1.0)
            : FloatAttr(),
        tensorrt::ActivationOp::requiresBetaAttribute(activationType)
            ? rewriter.getF32FloatAttr(0.0)
            : FloatAttr());
    rewriter.replaceOp(op, elementwiseOp->getResults());
    return success();
  }
};

/// Convert `stablehlo.rsqrt` to `tensorrt.unary` reciprocal + sqrt.
struct RsqrtConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::RsqrtOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::RsqrtOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getType());
    auto recipOp = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(), type, adaptor.getOperand(),
        tensorrt::UnaryOperation::kRECIP);
    rewriter.replaceOpWithNewOp<tensorrt::UnaryOp>(
        op, type, recipOp.getResult(), tensorrt::UnaryOperation::kSQRT);
    return success();
  }
};

/// Convert `stablehlo.rem(lhs, rhs)` to `tensorrt.element_wise` that calculate
/// "lhs - div(lhs, rhs) * rhs".
///
/// The "div" is interpreted as rounding to integer for floats. This could
/// technically be accomplished two ways:
///   1. "round nearest, half to even": using TensorRT unary kROUND op. This
///     produces behavior equivalent to IEEE-754 'remainder' specification.
///   2. "round toward zero": by either casting result of the floating point div
///     to int and back OR using a combination of Sign, Floor, and Ceil ops.
///
/// Alignment with precise semantics of `stablehlo.remainder`
/// should do #2. For now, we just using casting to int32 to do the "round half
/// to zero".
struct ConvertRemainder
    : public ConvertHloOpToTensorRTPattern<stablehlo::RemOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::RemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TensorType resultType = op.getType();
    Type convertedResultType =
        this->getTypeConverter()->convertType(resultType);
    // Do "lhs - div(lhs, rhs) * rhs".
    TensorValue floorDiv =
        rewriter
            .create<tensorrt::ElementWiseOp>(
                loc, convertedResultType, adaptor.getLhs(), adaptor.getRhs(),
                tensorrt::ElementWiseOperation::kDIV)
            .getResult();
    if (isa<FloatType>(resultType.getElementType())) {
      floorDiv = ConvertHloOpToTensorRTPattern::castTensor(
          rewriter, rewriter.getI32Type(), floorDiv);
      floorDiv = ConvertHloOpToTensorRTPattern::castTensor(
          rewriter, convertedResultType, floorDiv);
    }
    Value product = rewriter.create<tensorrt::ElementWiseOp>(
        loc, convertedResultType, floorDiv, adaptor.getRhs(),
        tensorrt::ElementWiseOperation::kPROD);
    rewriter.replaceOpWithNewOp<tensorrt::ElementWiseOp>(
        op, convertedResultType, adaptor.getLhs(), product,
        tensorrt::ElementWiseOperation::kSUB);
    return success();
  }
};

/// Convert `stablehlo.compare` to `tensorrt.element_wise`.
struct CompareConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::CompareOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::CompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType resultType = op.getType();
    Location loc = op.getLoc();

    auto convertI1ToI32 = [&](Value v) {
      if (!cast<RankedTensorType>(v.getType()).getElementType().isInteger(1))
        return v;
      Value newV = rewriter.create<tensorrt::IdentityOp>(
          v.getLoc(),
          cast<RankedTensorType>(v.getType()).clone(rewriter.getI32Type()), v);
      return newV;
    };
    Value lhs = convertI1ToI32(adaptor.getLhs());
    Value rhs = convertI1ToI32(adaptor.getRhs());
    assert(lhs.getType() == rhs.getType() &&
           "lhs type should be equal to rhs type.");

    auto replaceWithEwise = [&](tensorrt::ElementWiseOperation ewiseOp) {
      rewriter.replaceOpWithNewOp<tensorrt::ElementWiseOp>(op, resultType, lhs,
                                                           rhs, ewiseOp);
    };

    // Handle the simple cases with simple replacement.
    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::EQ) {
      replaceWithEwise(tensorrt::ElementWiseOperation::kEQUAL);
      return success();
    }
    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::LT) {
      replaceWithEwise(tensorrt::ElementWiseOperation::kLESS);
      return success();
    }
    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::GT) {
      replaceWithEwise(tensorrt::ElementWiseOperation::kGREATER);
      return success();
    }

    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::LE) {
      // There is no "LE" for TensorRT, so replace with OR(LT,EQ).
      Value eq = rewriter.create<tensorrt::ElementWiseOp>(
          loc, resultType, lhs, rhs, tensorrt::ElementWiseOperation::kEQUAL);
      Value lt = rewriter.create<tensorrt::ElementWiseOp>(
          loc, resultType, lhs, rhs, tensorrt::ElementWiseOperation::kLESS);
      rewriter.replaceOpWithNewOp<tensorrt::ElementWiseOp>(
          op, resultType, eq, lt, tensorrt::ElementWiseOperation::kOR);
      return success();
    }

    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::GE) {
      // There is no "GE" for TensorRT, so replace with OR(GT,EQ).
      Value eq = rewriter.create<tensorrt::ElementWiseOp>(
          loc, resultType, lhs, rhs, tensorrt::ElementWiseOperation::kEQUAL);
      Value gt = rewriter.create<tensorrt::ElementWiseOp>(
          loc, resultType, lhs, rhs, tensorrt::ElementWiseOperation::kGREATER);
      rewriter.replaceOpWithNewOp<tensorrt::ElementWiseOp>(
          op, resultType, eq, gt, tensorrt::ElementWiseOperation::kOR);
      return success();
    }

    if (op.getComparisonDirection() == stablehlo::ComparisonDirection::NE) {
      // Convert NE to NOT(EQ).
      Value eq = rewriter.create<tensorrt::ElementWiseOp>(
          loc, resultType, lhs, rhs, tensorrt::ElementWiseOperation::kEQUAL);
      // TensorRT unary ops need at least 1D tensor
      if (cast<RankedTensorType>(eq.getType()).getRank() == 0) {
        RankedTensorType newType =
            RankedTensorType::get({1}, resultType.getElementType());
        eq = rewriter.create<tensorrt::ExpandRankOp>(loc, newType, eq);
      }
      Value ne = rewriter.create<tensorrt::UnaryOp>(
          loc, eq.getType(), eq, tensorrt::UnaryOperation::kNOT);
      if (cast<RankedTensorType>(ne.getType()).getRank() !=
          resultType.getRank())
        ne = rewriter.create<tensorrt::CollapseRankOp>(loc, resultType, ne);
      rewriter.replaceOp(op, ne);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported comparison type");
  }
};

/// Convert `stablehlo.clamp(%min, %operand, %max) to `tensorrt.element_wise`
struct ClampConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::ClampOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto originalType = cast<RankedTensorType>(adaptor.getOperand().getType());

    auto valueOperand = cast<TensorValue>(adaptor.getOperand());
    auto valueMin = cast<TensorValue>(adaptor.getMin());
    auto valueMax = cast<TensorValue>(adaptor.getMax());

    // The clamp min/max values are either rank-0 or equal to the rank of the
    // operand.
    auto maybeExpandBoundsRank = [&](TensorValue v) -> TensorValue {
      if (v.getType().getRank() == originalType.getRank())
        return v;
      SmallVector<int64_t> newShape(originalType.getRank(), 1);
      Type expandedType =
          RankedTensorType::Builder(cast<RankedTensorType>(v.getType()))
              .setShape(newShape);
      return rewriter
          .create<tensorrt::ExpandRankOp>(op.getLoc(), expandedType, v)
          .getResult();
    };

    // I32 is unsupported, so cast to fp32.
    if (originalType.getElementType().isInteger(32)) {
      Type f32Type = rewriter.getF32Type();
      valueOperand = ConvertHloOpToTensorRTPattern::castTensor(
          rewriter, f32Type, valueOperand);
      valueMin = ConvertHloOpToTensorRTPattern::castTensor(rewriter, f32Type,
                                                           valueMin);
      valueMax = ConvertHloOpToTensorRTPattern::castTensor(rewriter, f32Type,
                                                           valueMax);
    }

    Value lowerBound = rewriter.create<tensorrt::ElementWiseOp>(
        op->getLoc(), valueOperand.getType(), valueOperand,
        maybeExpandBoundsRank(valueMin), tensorrt::ElementWiseOperation::kMAX);

    TensorValue result = rewriter
                             .create<tensorrt::ElementWiseOp>(
                                 op->getLoc(), lowerBound.getType(), lowerBound,
                                 maybeExpandBoundsRank(valueMax),
                                 tensorrt::ElementWiseOperation::kMIN)
                             .getResult();

    // Cast back if required.
    if (result.getType() != originalType) {
      rewriter.replaceOp(op, ConvertHloOpToTensorRTPattern::castTensor(
                                 rewriter, originalType, result));
      return success();
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

/// Returns the reduction op applied for `resultIndex`th index result.
/// Implementation is taken from
/// https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/IR/hlo_ops.cc#L3351
static Operation *getWindowReductionOp(stablehlo::ReduceWindowOp op,
                                       int64_t resultIndex) {
  auto returnOp =
      cast<stablehlo::ReturnOp>(op.getBody().front().getTerminator());
  Operation *computeOp = returnOp.getResults()[resultIndex].getDefiningOp();
  if (computeOp->getNumOperands() != 2)
    return nullptr;
  auto arg0 = dyn_cast<BlockArgument>(computeOp->getOperand(0));
  auto arg1 = dyn_cast<BlockArgument>(computeOp->getOperand(1));
  if (!arg0 || !arg1)
    return nullptr;
  int64_t arg0Num = arg0.getArgNumber();
  int64_t arg1Num = arg1.getArgNumber();
  int64_t otherArgIndex = resultIndex + op.getInputs().size();
  if (arg0Num == resultIndex && arg1Num == otherArgIndex)
    return computeOp;
  if (arg0Num == otherArgIndex && arg1Num == resultIndex &&
      computeOp->hasTrait<mlir::OpTrait::IsCommutative>())
    return computeOp;
  return nullptr;
}

static FailureOr<tensorrt::PoolingType>
getPoolingOpType(stablehlo::ReduceWindowOp op) {
  if (auto sumOp =
          dyn_cast_or_null<stablehlo::AddOp>(getWindowReductionOp(op, 0)))
    return tensorrt::PoolingType::kAVERAGE;
  if (auto maxOp =
          dyn_cast_or_null<stablehlo::MaxOp>(getWindowReductionOp(op, 0)))
    return tensorrt::PoolingType::kMAX;
  return failure();
}

template <typename T>
static Attribute getConstantAttrOf(Type type, T value) {
  if (isa<FloatType>(type))
    return FloatAttr::get(type, static_cast<double>(value));
  if (isa<IndexType>(type))
    return IntegerAttr::get(type, APInt(64, value));
  if (auto integerType = dyn_cast<IntegerType>(type))
    return IntegerAttr::get(type,
                            APInt(cast<IntegerType>(type).getWidth(), value));
  if (isa<RankedTensorType, VectorType>(type)) {
    auto vtType = cast<ShapedType>(type);
    auto element = getConstantAttrOf(vtType.getElementType(), value);
    if (!element)
      return {};
    return DenseElementsAttr::get(vtType, element);
  }
  llvm_unreachable("unhandled constant attr type");
  return {};
}

// Returns whether the window dimensions can be interpreted as applying to an
// "NCHW" format tensor, e.g. the pooling window only applies to the innermost
// dimensions.
static bool isNCHWPoolingWindow(ArrayRef<int64_t> windowDims,
                                ArrayRef<int64_t> windowStrides) {
  // This condition should be enforced by ConvertReduceWindow. TRT can only
  // handle 4D or 5D tensor.
  assert((windowDims.size() == 4 || windowDims.size() == 5) &&
         "expected size 4 or 5 vector");
  unsigned numBatchDims = windowDims.size() == 4 ? 2 : 3;
  for (unsigned i = 0; i < numBatchDims; i++) {
    if (windowDims[i] != 1 || windowStrides[i] != 1)
      return false;
  }
  return true;
}

/// Returns the required transpose for putting the pooling dimensions on inner
/// most dims.
static FailureOr<SmallVector<unsigned>>
getTransposeForReduceWindow(ArrayRef<int64_t> windowDims) {
  SmallVector<unsigned> batchDims;
  SmallVector<unsigned> reduceDims;
  if (windowDims.size() != 4 && windowDims.size() != 5)
    return failure();
  for (unsigned i = 0; i < windowDims.size(); i++) {
    if (windowDims[i] == 1) {
      batchDims.push_back(i);
      continue;
    }
    reduceDims.push_back(i);
  }
  // Permutation is all batch dims followed by reduce dims.
  llvm::append_range(batchDims, reduceDims);
  return batchDims;
}

namespace {
/// Convert `jnp.cumsum` expressed as `stablehlo.reduce_window<add>` to
/// `tensorrt.convolution`.
/// This pass only supports cases where the input of jnp.cumsum is <= 3D,
/// and the input precision is F16, F32, or I32.
struct ConvertCumsum
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReduceWindowOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto addOp =
        dyn_cast_or_null<stablehlo::AddOp>(getWindowReductionOp(op, 0));
    if (!addOp)
      return failure();

    auto isOnlyOneGreaterOne = [](ArrayRef<int64_t> seq) {
      auto it =
          std::find_if(seq.begin(), seq.end(), [](int num) { return num > 1; });
      // Check if only one element is greater than 1
      bool onlyOneGreaterOne =
          it != seq.end() && std::count_if(seq.begin(), seq.end(), [](int num) {
                               return num > 1;
                             }) == 1;
      if (onlyOneGreaterOne)
        return static_cast<int>(std::distance(seq.begin(), it));
      return -1;
    };
    auto isAllOne = [](ArrayRef<int64_t> seq) {
      return llvm::all_of(seq, [](int num) { return num == 1; });
    };

    auto inputType =
        cast<RankedTensorType>(adaptor.getInputs().front().getType());
    auto inputShape = inputType.getShape();
    ArrayRef<int64_t> windowDims = op.getWindowDimensions();
    int idx = isOnlyOneGreaterOne(windowDims);
    SmallVector<int64_t> windowStrides =
        op.getWindowStrides().has_value()
            ? llvm::to_vector(*op.getWindowStrides())
            : SmallVector<int64_t>(windowDims.size(), 1);
    SmallVector<int64_t> windowDilations =
        op.getWindowDilations().has_value()
            ? llvm::to_vector(*op.getWindowDilations())
            : SmallVector<int64_t>(windowDims.size(), 1);

    // Check if not Cumsum
    if (idx < 0 || inputShape[idx] != windowDims[idx] ||
        inputType.getRank() != static_cast<int64_t>(windowDims.size()) ||
        !isAllOne(windowStrides) || !isAllOne(windowDilations))
      return failure();
    FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> paddings =
        convertPaddingAttribute(op.getLoc(), op.getPadding());
    if (failed(paddings) || paddings->first[idx] != windowDims[idx] - 1)
      return failure();

    // windowDims is later rank-expanded adding 2D to be the kernel of
    // tensorrt::ConvolutionOp that can be up to 5D.
    if (windowDims.size() > 3)
      return failure();

    auto getKernelShape =
        [](SmallVector<int64_t> &kernelShape, SmallVector<int64_t> &strides,
           SmallVector<int64_t> &dilations, SmallVector<int64_t> &prePaddings,
           SmallVector<int64_t> &postPaddings) {
          int gap = 2 - kernelShape.size();
          if (gap > 0) {
            // To use Conv2D, referring to
            // https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Convolution.html#shape-information
            kernelShape.insert(kernelShape.begin(), gap, 1);
            strides.insert(strides.begin(), gap, 1);
            dilations.insert(dilations.begin(), gap, 1);
            prePaddings.insert(prePaddings.begin(), gap, 0);
            postPaddings.insert(postPaddings.begin(), gap, 0);
          }
          // Add Ch_{out} = 1 and Ch_{in} = 1
          kernelShape.insert(kernelShape.begin(), 2, 1);
        };
    SmallVector<int64_t> convKernelShape{windowDims};
    SmallVector<int64_t> prePaddings{paddings->first};
    SmallVector<int64_t> postPaddings{paddings->second};
    getKernelShape(convKernelShape, windowStrides, windowDilations, prePaddings,
                   postPaddings);
    SmallVector<int64_t> convInputShape{inputShape};
    convInputShape.insert(convInputShape.begin(),
                          convKernelShape.size() - inputShape.size(), 1);
    assert(convInputShape.size() == convKernelShape.size());

    // tensorrt::ConvolutionOp does not accept the i32 type for both input and
    // kernel.
    auto checkI32 = [&](Value v) {
      return mlir::getElementTypeOrSelf(v.getType()).isInteger(32);
    };
    auto convertToF32IfI32 = [&](Value v) {
      if (checkI32(v)) {
        Value newV = rewriter.create<tensorrt::IdentityOp>(
            v.getLoc(),
            cast<RankedTensorType>(v.getType()).clone(rewriter.getF32Type()),
            v);
        return newV;
      }
      return v;
    };
    auto convertToI32IfItWas = [&](Value v) {
      if (checkI32(adaptor.getInputs().front())) {
        Value newV = rewriter.create<tensorrt::IdentityOp>(
            v.getLoc(),
            cast<RankedTensorType>(v.getType()).clone(rewriter.getI32Type()),
            v);
        return newV;
      }
      return v;
    };

    Value input = convertToF32IfI32(adaptor.getInputs().front());
    Value convInput = rewriter.create<tensorrt::ExpandRankOp>(
        op->getLoc(),
        RankedTensorType::get(
            convInputShape,
            cast<RankedTensorType>(input.getType()).getElementType()),
        input);
    auto convKernelType =
        cast<RankedTensorType>(convInput.getType()).clone(convKernelShape);
    Value convKernel = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(
            convKernelType,
            getConstantAttrOf(convKernelType.getElementType(), 1.)));

    auto conv = rewriter.create<tensorrt::ConvolutionOp>(
        op->getLoc(), convInput.getType(),
        /*input=*/convInput,
        /*kernel=*/convKernel,
        /*bias=*/Value(),
        /*stride=*/windowStrides,
        /*pre_padding=*/prePaddings,
        /*post_padding=*/postPaddings,
        /*num_groups=*/1,
        /*dilation=*/windowDilations);

    // Restore the original shape and element type
    auto reshape = rewriter.create<tensorrt::ReshapeOp>(
        op->getLoc(), inputType, convertToI32IfItWas(conv));

    rewriter.replaceOp(op, reshape);

    return success();
  }
};

/// Convert `stablehlo.reduce_window` using `tensorrt.pooling`. Note that
/// TensorRT has some restrictions which make this nuanced. Note that a common
/// HLO operation is `stablehlo.reduce_window<add>`, which is equivalent to
/// convolution with a weight vector of 1's. Often, however, this will followed
/// by a division (to achieve "average pooling"). So instead of using
/// convolution, we just use average pooling and multiply by the window volume.
/// The elementwise canonicalizer should eliminate the repetitive mul/div ops,
/// leaving just "average pool". Otherwise, we translate
/// `stablehlo.reduce_window <max>` to `tensorrt.pooling <kMAX>`.
/// TODO: support `stablehlo.reduce_window <min>` multiplying input/result by
/// -1.
/// TODO: changing this to use convolution for `reduce_window<add>` is probably
/// better in the long run.
struct ConvertReduceWindow
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReduceWindowOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected one input to windowOp");
    auto inputType =
        cast<RankedTensorType>(adaptor.getInputs().front().getType());

    FailureOr<tensorrt::PoolingType> poolingType = getPoolingOpType(op);
    if (failed(poolingType))
      return rewriter.notifyMatchFailure(op, "unsupported pooling type");

    // TensorRT allows greater ranks, but this is most common.
    if (inputType.getRank() != 4 && inputType.getRank() != 5)
      return rewriter.notifyMatchFailure(
          op, "only rank-4 or rank-5 inputs are supported");

    // TODO: handle windows with dilation.
    if (op.getWindowDilations() && !isSplat(op.getWindowDilationsAttr(), 1))
      return rewriter.notifyMatchFailure(
          op, "non-unit window dilations are not supported");
    if (op.getBaseDilations() && !isSplat(op.getBaseDilationsAttr(), 1))
      return rewriter.notifyMatchFailure(
          op, "non-unit base dilations are not supported");
    // Get padding.
    std::pair<SmallVector<int64_t>, SmallVector<int64_t>> padding = {
        SmallVector<int64_t>(inputType.getRank(), 0),
        SmallVector<int64_t>(inputType.getRank(), 0)};
    if (op.getPadding() && !isSplat(op.getPaddingAttr(), 0)) {
      // Separate Nx2 padding attribute into pre/post padding.
      FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> res =
          convertPaddingAttribute(op.getLoc(), op.getPadding());
      if (failed(res))
        return rewriter.notifyMatchFailure(op, "could not parse padding");
      padding = std::move(*res);
    }

    SmallVector<int64_t> windowDims = llvm::to_vector(op.getWindowDimensions());
    SmallVector<int64_t> windowStrides =
        op.getWindowStrides().has_value()
            ? llvm::to_vector(*op.getWindowStrides())
            : SmallVector<int64_t>(windowDims.size(), 1);

    // Calculate transpose if required.
    Value input = adaptor.getInputs().front();
    AffineMap permMap = AffineMap::getMultiDimIdentityMap(
        inputType.getRank(), rewriter.getContext());
    if (!isNCHWPoolingWindow(windowDims, windowStrides)) {
      FailureOr<SmallVector<unsigned>> perm =
          getTransposeForReduceWindow(windowDims);
      if (failed(perm))
        return rewriter.notifyMatchFailure(
            op, "failed to calculate transpose permutation");

      // Transpose the input and all the attributes.
      permMap = AffineMap::getPermutationMap(*perm, rewriter.getContext());
      input =
          rewriter.create<tensorrt::TransposeOp>(op.getLoc(), input, permMap);
      padding.first = permMap.compose(padding.first);
      padding.second = permMap.compose(padding.second);
      windowStrides = permMap.compose(windowStrides);
      windowDims = permMap.compose(windowDims);
    }

    // Final check: only trailing dims (reduction dims) have non-unit stride or
    // non-zero padding/window size. This should be worked out above.
    auto isNonZero = [](int64_t val) { return val != 0; };
    auto isNonUnit = [](int64_t val) { return val != 1; };
    unsigned numBatchDims =
        inputType.getRank() - llvm::count_if(windowDims, isNonUnit);
    if (llvm::any_of(ArrayRef(windowDims).take_front(numBatchDims),
                     isNonUnit) ||
        llvm::any_of(ArrayRef(windowStrides).take_front(numBatchDims),
                     isNonUnit) ||
        llvm::any_of(ArrayRef(padding.first).take_front(numBatchDims),
                     isNonZero) ||
        llvm::any_of(ArrayRef(padding.second).take_front(numBatchDims),
                     isNonZero)) {
      return rewriter.notifyMatchFailure(op.getLoc(), "failed preconditions");
    }

    auto replaceOp = [&](TypedValue<RankedTensorType> result) {
      if (!permMap.isIdentity()) {
        rewriter.replaceOpWithNewOp<tensorrt::TransposeOp>(
            op, result, mlir::inversePermutation(permMap));
        return success();
      }
      rewriter.replaceOp(op, result);
      return success();
    };

    // Create the pooling op. TensorRT uses sparse description of window
    // attributes.
    SmallVector<int64_t> zeros(windowDims.size(), 0);
    auto resultType = cast<RankedTensorType>(op.getType(0));
    resultType = resultType.clone(permMap.compose(resultType.getShape()));
    auto poolOp = rewriter.create<tensorrt::PoolingOp>(
        op.getLoc(), resultType, input,
        /*windowSize=*/ArrayRef(windowDims).drop_front(numBatchDims),
        /*stride=*/ArrayRef(windowStrides).drop_front(numBatchDims),
        /*prePadding=*/ArrayRef(padding.first).drop_front(numBatchDims),
        /*postPadding=*/ArrayRef(padding.second).drop_front(numBatchDims),
        /*poolingType=*/*poolingType, FloatAttr(),
        /*averageCountExcludesPadding=*/
        *poolingType == tensorrt::PoolingType::kMAX
            ? nullptr
            : rewriter.getBoolAttr(true));

    if (*poolingType == tensorrt::PoolingType::kMAX)
      return replaceOp(poolOp.getResult());

    // Insert a multiplication operation to balance the fact that we're
    // replacing `reduce_window<sum>` with an "average pool" op.
    auto constShape =
        RankedTensorType::get(SmallVector<int64_t>(inputType.getRank(), 1),
                              inputType.getElementType());
    int64_t windowVolume = std::accumulate(windowDims.begin(), windowDims.end(),
                                           1, std::multiplies<>());
    Value windowVolConst = rewriter.create<tensorrt::ConstantOp>(
        op.getLoc(), constShape,
        DenseElementsAttr::get(
            constShape,
            getConstantAttrOf(constShape.getElementType(), windowVolume)));
    auto elwiseOp = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(), poolOp.getType(), poolOp.getResult(), windowVolConst,
        tensorrt::ElementWiseOperation::kPROD);
    return replaceOp(elwiseOp.getResult());
  }
};
} // namespace

static LogicalResult convertConstant(ConversionPatternRewriter &rewriter,
                                     Operation *op,
                                     ElementsAttr &constValueAttr,
                                     RankedTensorType originalTensorType,
                                     RankedTensorType convertedTensorType) {
  Type targetElementType = convertedTensorType.getElementType();
  if (originalTensorType == convertedTensorType)
    return success();

  // Convert the underlying values.
  // If the attribute is a "DenseElementsAttr", it is never elided.
  if (auto denseValue = dyn_cast<DenseElementsAttr>(constValueAttr)) {
    SmallVector<int32_t> newValues;
    newValues.reserve(denseValue.size());
    if (denseValue.getElementType().isInteger(1) &&
        targetElementType.isInteger(32)) {
      for (bool x : denseValue.template getValues<bool>())
        newValues.push_back(static_cast<int32_t>(x));
    } else if (denseValue.getElementType().isInteger(64) &&
               targetElementType.isInteger(32)) {
      for (int64_t x : denseValue.template getValues<int64_t>())
        newValues.push_back(static_cast<int32_t>(x));
    } else {
      return failure();
    }
    constValueAttr =
        DenseElementsAttr::get(convertedTensorType, llvm::ArrayRef(newValues));
  }

  // If the attribute is a "DenseResourceElementsAttr", it may be
  // `dense_resource<__elided__>`, and we want to handle it gracefully for
  // testing.
  else if (auto denseResourceAttr =
               dyn_cast<DenseResourceElementsAttr>(constValueAttr)) {
    DenseResourceElementsHandle handle = denseResourceAttr.getRawHandle();
    if (handle.getKey() == "__elided__") {
      // When elided, just update the type.
      constValueAttr =
          DenseResourceElementsAttr::get(convertedTensorType, handle);
    } else {
      // TODO: handle resource elements attrs. We need a TensorRT dialect
      // resource manager.
      return rewriter.notifyMatchFailure(
          op, "unhandled DenseResourceElementsAttr case");
    }
  } else {
    return rewriter.notifyMatchFailure(
        op, "unhandled HLO ConstantOp \"value\" attribute type");
  }
  return success();
}

/// Given the desired type `t`, return the type that should be used for creating
/// a `tensorrt.constant`, the result of which should be casted (using
/// `tensorrt.identity`) back to type `t`.
static RankedTensorType convertToLegalConstantType(RankedTensorType t) {
  if (t.getElementType().isInteger(1))
    return cast<RankedTensorType>(
        t.clone(IntegerType::get(t.getContext(), 32)));
  return t;
}

namespace {
/// Convert `stablehlo` constant operation to tensorrt constant op.
/// TODO: Should we remove template?
template <typename HloOpType>
struct HloConstantConverter : public ConvertHloOpToTensorRTPattern<HloOpType> {

  using ConvertHloOpToTensorRTPattern<HloOpType>::ConvertHloOpToTensorRTPattern;

  LogicalResult matchAndRewrite(
      HloOpType op,
      typename ConvertHloOpToTensorRTPattern<HloOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto originalType = cast<RankedTensorType>(op.getType());
    auto convertedType = dyn_cast_or_null<RankedTensorType>(
        this->getTypeConverter()->convertType(originalType));
    if (!convertedType)
      return failure();
    RankedTensorType supportedConstantType =
        convertToLegalConstantType(convertedType);

    ElementsAttr constValueAttr = op.getValue();
    if (failed(convertConstant(rewriter, op, constValueAttr, originalType,
                               supportedConstantType)))
      return rewriter.notifyMatchFailure(op, "could not convert i32 type");
    assert(!cast<RankedTensorType>(constValueAttr.getType())
                .getElementType()
                .isInteger(1));

    auto replaceRoot = [&](TypedValue<RankedTensorType> replacement) {
      // We may need an identity operation to convert back to boolean.
      rewriter.replaceOp(op, HloConstantConverter::castTensor(
                                 rewriter, convertedType, replacement));
      return success();
    };

    // If the constant is a splat that creates a 'big' tensor, then we convert
    // to constant+broadcast.
    bool isBigSplat =
        constValueAttr.isSplat() &&
        constValueAttr.getNumElements() *
                constValueAttr.getElementType().getIntOrFloatBitWidth() >=
            1024 * 1024 &&
        isa<DenseElementsAttr>(constValueAttr);

    if (!isBigSplat)
      return replaceRoot(rewriter.create<mlir::tensorrt::ConstantOp>(
          op.getLoc(), constValueAttr));

    auto constOp = rewriter.create<mlir::tensorrt::ConstantOp>(
        op.getLoc(), cast<DenseElementsAttr>(constValueAttr)
                         .resizeSplat(constValueAttr.getShapedType().clone(
                             SmallVector<int64_t>(originalType.getRank(), 1))));
    auto broadcastOp = rewriter.create<tensorrt::BroadcastOp>(
        op.getLoc(), supportedConstantType, constOp.getResult(),
        llvm::to_vector(llvm::seq<int64_t>(0, originalType.getRank())));
    return replaceRoot(broadcastOp);
  }
};
} // namespace

namespace {
/// Convert `stablehlo.slice` op to `tensorrt.slice` operation.
template <typename HloOpType>
struct HloSliceConverter : public ConvertHloOpToTensorRTPattern<HloOpType> {
  using ConvertHloOpToTensorRTPattern<HloOpType>::ConvertHloOpToTensorRTPattern;
  LogicalResult matchAndRewrite(
      HloOpType op,
      typename ConvertHloOpToTensorRTPattern<HloOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Get the values for the start, size, and stop indices. These must be
    // converted to i32 in a safe manner.
    Location loc = op.getLoc();
    FailureOr<SmallVector<int32_t>> startIndices =
        truncateI64ToI32(loc, op.getStartIndices());
    if (failed(startIndices))
      return rewriter.notifyMatchFailure(
          op, "could not convert i64 offsets to i32");
    FailureOr<SmallVector<int32_t>> strides =
        truncateI64ToI32(loc, op.getStrides());
    if (failed(strides))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert i64 stride to i32");
    FailureOr<SmallVector<int32_t>> i32Shape =
        truncateI64ToI32(loc, op.getType().getShape());
    if (failed(i32Shape))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert i64 shape to i32");
    auto sliceOp = rewriter.create<mlir::tensorrt::SliceOp>(
        op.getLoc(), adaptor.getOperand(), *startIndices, *i32Shape, *strides);

    // This should always be true since the semantics of the HLO slice op and
    // the TensorRT slice op align.
    assert(sliceOp.getType().getShape() == op.getType().getShape());
    rewriter.replaceOp(op, sliceOp.getResult());
    return success();
  }
};
} // namespace

/// Computes an i32 tensor representing size of all dimensions of a slice
/// result. A slice is parameterized by (offset, limit, stride) tensors. The
/// size_tensor = ceil((limit-offset)/stride) = (limit-offset+stride-1)/stride.
/// There is no ceil divide in TRT. We don't take advantage of static dimension
/// information in the result because TRT requires the size for all dimensions
/// to be packaged into a single tensor.
static Value calculateSliceSize(OpBuilder &b, Location loc, Value offset,
                                Value limit, Value stride) {
  Value limitMinusOffset =
      b.create<tensorrt::ElementWiseOp>(loc, offset.getType(), limit, offset,
                                        tensorrt::ElementWiseOperation::kSUB);
  Value one = b.create<tensorrt::ConstantOp>(
      loc, stride.getType(),
      DenseElementsAttr::get(
          cast<ShapedType>(stride.getType()),
          b.getIntegerAttr(
              cast<RankedTensorType>(stride.getType()).getElementType(), 1)));
  Value strideMinusOne = b.create<tensorrt::ElementWiseOp>(
      loc, offset.getType(), stride, one, tensorrt::ElementWiseOperation::kSUB);
  Value numerator = b.create<tensorrt::ElementWiseOp>(
      loc, offset.getType(), limitMinusOffset, strideMinusOne,
      tensorrt::ElementWiseOperation::kSUM);
  return b.create<tensorrt::ElementWiseOp>(
      loc, offset.getType(), numerator, stride,
      tensorrt::ElementWiseOperation::kDIV);
}

namespace {
/// Convert `stablehlo.real_dynamic_slice` to `tensorrt.slice`.
struct RealDynamicSliceConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::RealDynamicSliceOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::RealDynamicSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value sliceSize =
        calculateSliceSize(rewriter, loc, adaptor.getStartIndices(),
                           adaptor.getLimitIndices(), adaptor.getStrides());
    rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getOperand(),
        /*offset=*/adaptor.getStartIndices(), /*size=*/sliceSize,
        /*stride=*/adaptor.getStrides(), tensorrt::SliceMode::kDEFAULT);
    return success();
  }
};

/// Convert `stablehlo.dynamic_slice` to `tensorrt.slice`.
/// `stablehlo.dynamic_slice` isn't an actual dynamic slice -- see the
/// `stablehlo.real_dynamic_slice` converter above. Instead, only the slice
/// offsets (which are given as individual 0-D scalars) are dynamic, unit stride
/// is assumed, and the size is static.
struct DynamicSliceConverter
    : public ConvertOpToTensorRTPattern<stablehlo::DynamicSliceOp> {
  using ConvertOpToTensorRTPattern::ConvertOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DynamicSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    Location loc = op.getLoc();

    // Start indices is an array of `tensor<i32>` values. We need to convert
    // them into a single tensor. First reshape all of scalars to tensor<1xi32>.
    Type i32Type = rewriter.getI32Type();
    RankedTensorType scalar1DType = RankedTensorType::get({1}, i32Type);
    SmallVector<Value> reshapedIndices;
    reshapedIndices.reserve(adaptor.getStartIndices().size());
    for (Value offsetValue : adaptor.getStartIndices()) {
      // If we couldn't convert these to i32 integers, then we can't proceed.
      if (cast<TensorType>(offsetValue.getType()).getElementType() != i32Type)
        return rewriter.notifyMatchFailure(
            op, "offset tensors must have i32 element type");
      reshapedIndices.push_back(
          rewriter.create<tensorrt::ReshapeOp>(loc, scalar1DType, offsetValue));
    }

    // Then concatenate them together.
    const int64_t resultRank = cast<RankedTensorType>(resultType).getRank();
    Value dynamicOffsets =
        reshapedIndices.size() == 1
            ? reshapedIndices.front()
            : rewriter.create<tensorrt::ConcatenationOp>(loc, reshapedIndices,
                                                         /*axis=*/0);
    MLIRContext *ctx = getContext();
    SmallVector<int32_t> size = llvm::to_vector(
        llvm::map_range(op.getSliceSizes(), [](int64_t element) {
          return static_cast<int32_t>(element);
        }));
    rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
        op, resultType, adaptor.getOperand(),
        /*start=*/dynamicOffsets,
        /*size=*/
        DenseI32ArrayAttr::get(ctx, size),
        /*stride=*/
        DenseI32ArrayAttr::get(ctx, SmallVector<int32_t>(resultRank, 1)),
        /*mode=*/tensorrt::SliceMode::kDEFAULT);
    return success();
  }
};

/// Convert `stablehlo.broadcast_in_dim` to `tensorrt.broadcast` extension
/// operation.
struct ConvertBroadcastInDim
    : public ConvertHloOpToTensorRTPattern<stablehlo::BroadcastInDimOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensorrt::BroadcastOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperand(),
        DenseI64ArrayAttr::get(rewriter.getContext(),
                               op.getBroadcastDimensions()));
    return success();
  }
};

/// Convert `stablehlo.dynamic_broadcast_in_dim` to `tensorrt.broadcast`.
struct ConvertDynamicBroadcastInDim
    : public ConvertHloOpToTensorRTPattern<stablehlo::DynamicBroadcastInDimOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = this->getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();
    rewriter.replaceOpWithNewOp<tensorrt::BroadcastOp>(
        op, newType, adaptor.getOperand(), adaptor.getOutputDimensions(),
        rewriter.getDenseI64ArrayAttr(op.getBroadcastDimensions()));
    return success();
  }
};

/// Convert `stablehlo.broadcast` to `tensorrt.broadcast` extension operation.
struct ConvertBroadcast
    : public ConvertHloOpToTensorRTPattern<stablehlo::BroadcastOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t numLeadingDimensions = op.getBroadcastSizes().size();
    if (numLeadingDimensions == 0)
      return failure();

    // Lowering is easier if we first expand rank and then broadcast. This is
    // equal to a tile.
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    SmallVector<int64_t> resultShape(resultType.getShape());
    for (int64_t i = 0; i < numLeadingDimensions; i++)
      resultShape[i] = 1;
    RankedTensorType reshapeType =
        RankedTensorType::Builder(resultType).setShape(resultShape);
    Value reshapeResult = rewriter.create<tensorrt::ExpandRankOp>(
        op.getLoc(), reshapeType, adaptor.getOperand());

    auto broadcastDims =
        llvm::to_vector(llvm::seq<int64_t>(0, op.getType().getRank()));
    rewriter.replaceOpWithNewOp<tensorrt::BroadcastOp>(
        op, this->getTypeConverter()->convertType(op.getType()), reshapeResult,
        DenseI64ArrayAttr::get(rewriter.getContext(), broadcastDims));
    return success();
  }
};

/// Convert `stablehlo.reshape` to `tensorrt.reshape`. `stablehlo.reshape` can
/// only represent static reshapes/static result tensors.
struct ReshapeConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReshapeOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::ReshapeOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (succeeded(tensorrt::isUnitDimRankExpanding(op.getOperand().getType(),
                                                   op.getType()))) {
      rewriter.replaceOpWithNewOp<tensorrt::ExpandRankOp>(
          op, this->getTypeConverter()->convertType(op.getType()),
          adaptor.getOperand());
      return success();
    }
    if (succeeded(tensorrt::isUnitDimRankReducing(op.getOperand().getType(),
                                                  op.getType()))) {
      rewriter.replaceOpWithNewOp<tensorrt::CollapseRankOp>(
          op, this->getTypeConverter()->convertType(op.getType()),
          adaptor.getOperand());
      return success();
    }
    rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperand());
    return success();
  }
};

/// Convert `stablehlo.dynamic_reshape` to `tensorrt.reshape`.
struct DynamicReshapeConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::DynamicReshapeOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::DynamicReshapeOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DynamicReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    Value inputTensor = adaptor.getOperand();
    rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(
        op, resultType, inputTensor,
        resultType.hasStaticShape() ? Value() : adaptor.getOutputShape());
    return success();
  }
};

/// Convert `stablehlo.convert` to `tensorrt.identity`.
struct ConvertConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::ConvertOp> {
  using SourceOp = stablehlo::ConvertOp;
  using ConvertHloOpToTensorRTPattern<SourceOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = this->getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();
    rewriter.replaceOpWithNewOp<tensorrt::IdentityOp>(op, newType,
                                                      adaptor.getOperand());
    return success();
  }
};

/// Convert `stablehlo.concatenate` to `tensorrt.concatenate`
struct ConvertConcatenate
    : public ConvertHloOpToTensorRTPattern<stablehlo::ConcatenateOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::ConcatenateOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensorrt::ConcatenationOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperands(), op.getDimension());
    return success();
  }
};

/// Convert `stablehlo.select` to `tensorrt.select`.
struct ConvertSelect
    : public ConvertHloOpToTensorRTPattern<stablehlo::SelectOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TensorValue trueOperand = cast<TensorValue>(adaptor.getOnTrue());
    TensorValue falseOperand = cast<TensorValue>(adaptor.getOnFalse());

    RankedTensorType trueType = trueOperand.getType();
    RankedTensorType falseType = falseOperand.getType();
    if (trueType != falseType)
      return rewriter.notifyMatchFailure(op, "expected equal types");

    // i1 operands are not supported by tensorrt::SelectOp, so cast to fp32.
    if (trueType.getElementType().isSignlessInteger(1)) {
      RankedTensorType newType =
          RankedTensorType::Builder(cast<RankedTensorType>(trueType))
              .setElementType(rewriter.getF32Type());
      trueOperand = castTensor(rewriter, newType, trueOperand);
      falseOperand = castTensor(rewriter, newType, falseOperand);
    }

    auto selectOp = rewriter.create<tensorrt::SelectOp>(
        op.getLoc(), adaptor.getPred(), trueOperand, falseOperand);
    auto selectOpResult = selectOp.getResult();

    // Cast the result back to the original type.
    if (selectOpResult.getType() != op.getResult().getType())
      selectOpResult =
          castTensor(rewriter, op.getResult().getType(), selectOpResult);

    rewriter.replaceOp(op, selectOpResult);
    return success();
  }
};

/// Convert `stablehlo.transpose` to `tensorrt.transpose`.
struct ConvertTranspose
    : public ConvertHloOpToTensorRTPattern<stablehlo::TransposeOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int64_t> permutation = op.getPermutation();
    AffineMap permutationMap =
        AffineMap::getPermutationMap(permutation, rewriter.getContext());
    rewriter.replaceOpWithNewOp<tensorrt::TransposeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperand(), permutationMap);
    return success();
  }
};

/// Convert `stablehlo.iota` to `tensorrt.linspace`.
struct ConvertIota : public ConvertHloOpToTensorRTPattern<stablehlo::IotaOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getType());
    // Iota produces a tensor from no inputs, so we can hit this converter even
    // if the type converter rejects i64->i32 conversions.
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");
    if (cast<RankedTensorType>(resultType).getRank() != 1 ||
        op.getIotaDimension() != 0)
      return failure();
    rewriter.replaceOpWithNewOp<tensorrt::LinspaceOp>(
        op, resultType,
        /*shape=*/Value(), /*start=*/Value(), /*step=*/Value(),
        /*static_start=*/rewriter.getF64FloatAttr(0.0),
        /*static_step=*/rewriter.getF64FloatAttr(1.0));
    return success();
  }
};

/// Convert `stablehlo.dynamic_iota` to `tensorrt.linspace`.
struct ConvertDynamicIota
    : public ConvertHloOpToTensorRTPattern<stablehlo::DynamicIotaOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DynamicIotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType resultType = op.getType();
    if (resultType.getRank() != 1 ||
        !this->getTypeConverter()->convertType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<tensorrt::LinspaceOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        /*shape=*/adaptor.getOutputShape(), /*start=*/Value(), /*step=*/Value(),
        /*static_start=*/rewriter.getF64FloatAttr(0.0),
        /*static_step=*/rewriter.getF64FloatAttr(1.0));
    return success();
  }
};

/// Convert `stablehlo.atan2` to `tensorrt.element_wise`.
struct ConvertAtan2 : public ConvertOpToTensorRTPattern<stablehlo::Atan2Op> {
  using ConvertOpToTensorRTPattern::ConvertOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::Atan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!cast<RankedTensorType>(lhs.getType()).getElementType().isF32() ||
        !cast<RankedTensorType>(rhs.getType()).getElementType().isF32())
      return failure();
    RankedTensorType resultType = dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return failure();

    // Element-wise divide input Tensors, apply atan unary, apply quadrant
    // correction
    Value intermediateDiv = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/lhs,
        /*input2=*/rhs,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kDIV);
    Value atan2Intermediate = rewriter.create<tensorrt::UnaryOp>(
        op->getLoc(),
        /*input=*/intermediateDiv,
        /*unaryOperation=*/tensorrt::UnaryOperation::kATAN);

    // Constant tensors used for quadrant correction
    RankedTensorType constType =
        RankedTensorType::Builder(resultType)
            .setShape(SmallVector<int64_t>(resultType.getRank(), 1));

    Value constZero = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(constType, rewriter.getF32FloatAttr(0.0)));

    Value constOne = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(constType, rewriter.getF32FloatAttr(1.0)));
    Value constTwo = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(constType, rewriter.getF32FloatAttr(2.0)));

    Value constPi = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(constType, rewriter.getF32FloatAttr(M_PI)));
    // Quadrant correction is only needed when (other < 0) (elementwise)
    // In this scenario, the correction is +/- pi, depending on the sign of self
    // (elementwise)

    // Full atan2 Formula is given by:
    // atan2(self, other) = atan(self / other) - (other < 0) * (2 * (self < 0) -
    // 1) * pi

    // Mask of (lhs < 0)
    auto lhsMaskValue =
        rewriter
            .create<tensorrt::ElementWiseOp>(
                op.getLoc(),
                /*input1=*/lhs,
                /*input2=*/constZero,
                /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kLESS)
            .getResult();
    // Mask of (rhs < 0)
    auto rhsMaskValue =
        rewriter
            .create<tensorrt::ElementWiseOp>(
                op.getLoc(),
                /*input1=*/rhs,
                /*input2=*/constZero,
                /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kLESS)
            .getResult();

    Type f32Type = rewriter.getF32Type();
    auto lhsMask =
        ConvertOpToTensorRTPattern::castTensor(rewriter, f32Type, lhsMaskValue);
    auto rhsMask =
        ConvertOpToTensorRTPattern::castTensor(rewriter, f32Type, rhsMaskValue);
    // Apply 2 * x - 1 to translate mask from {0, 1} to {-1, 1}
    lhsMask = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/lhsMask,
        /*input2=*/constTwo,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);
    lhsMask = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/lhsMask,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUB);
    // Multiply mask by pi
    lhsMask = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/lhsMask,
        /*input2=*/constPi,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);
    // Take product of masks to generate correction term
    Value correctionTerm = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/lhsMask,
        /*input2=*/rhsMask,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);
    // Add correction term to atan(lhs/rhs) to obtain atan2(lhs, rhs)
    rewriter.replaceOpWithNewOp<tensorrt::ElementWiseOp>(
        op, atan2Intermediate, correctionTerm,
        tensorrt::ElementWiseOperation::kSUB);
    return success();
  }
};
} // namespace

/// Given the tensor `v`, reshape the tensor to drop the `dimToDrop`-th
/// dimension. This assumes `unitDimToDrop` is a unit-sized dimension, otherwise
/// the behavior is undefined.
static Value getRankReducedTensor(RewriterBase &rewriter, Location loc, Value v,
                                  int64_t unitDimToDrop) {
  auto inputType = dyn_cast<RankedTensorType>(v.getType());
  assert((!inputType || inputType.getDimSize(unitDimToDrop) == 1) &&
         "expected value to have unit dim to drop");
  Type newType = RankedTensorType::Builder(inputType).dropDim(unitDimToDrop);
  return rewriter.create<tensorrt::CollapseRankOp>(loc, newType, v);
}

namespace {
/// Convert `stablehlo.torch_index_select` to `tensorrt.gather`. Note that
/// `stablehlo.torch_index_select` is equivalent to `tf.gather`
/// (https://www.tensorflow.org/api_docs/python/tf/gather) AKA `tf.GatherV2` in
/// TF's MLIR dialects.
///
/// This op is equivalent to `tensorrt.gather` except for
/// the fact that `tensorrt.gather` limits the number of batch dims to zero or
/// 1, while TF allows any number. In practice, this op usually appears with
/// `batch_dims = 0` or `batch_dims = 1`.
///
/// TODO: For `batch_dims = -1`, lower using `tensorrt.gather_elements`.
struct TorchIndexSelectConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::TorchIndexSelectOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::TorchIndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The "batch_dims" attribute refers to the number of batch dimensions
    Value indices = adaptor.getIndex();
    TensorType indicesType = op.getIndex().getType();
    Value operand = adaptor.getOperand();
    TensorType operandType = op.getOperand().getType();
    auto resultType = cast<RankedTensorType>(op.getType());

    // TODO: support this case with simple rank expansion.
    if (indicesType.getRank() < 1)
      return failure();

    // The gather axis is allowed to wrap wrap around.
    auto axis = static_cast<int64_t>(op.getDim());
    auto numBatchDims = static_cast<int64_t>(op.getBatchDims());
    if (axis < 0)
      axis += operandType.getRank();
    if (numBatchDims == -1) {
      assert(indicesType.getRank() == operandType.getRank() &&
             "expected equal ranks");
      numBatchDims = indicesType.getRank() - 1;
    }

    // Try to handle `numBatchDims > 1` by folding away unit dims. Make sure we
    // have at least a 1D indices tensor remaining.
    while (numBatchDims > 1 && indicesType.getDimSize(0) == 1 &&
           operandType.getDimSize(0) == 1 && indicesType.getRank() > 1) {
      assert(resultType.getDimSize(0) == 1 && "expected result_shape[0] == 1");
      // Fold leading unit dim.
      operand = getRankReducedTensor(rewriter, op.getLoc(), operand, 0);
      indices = getRankReducedTensor(rewriter, op.getLoc(), indices, 0);
      // Adjust the effective gather parameters.
      operandType = cast<TensorType>(operand.getType());
      indicesType = cast<TensorType>(indices.getType());
      numBatchDims -= 1;
      axis -= 1;
      resultType = RankedTensorType::Builder(resultType).dropDim(0);
    }

    // Check if we can convert directly to `tensorrt.gather`. Otherwise, abort.
    if (numBatchDims > 1)
      return failure();

// Sanity check: all batch dims should be same size for indices and operand.
// Stable HLO doesn't have implicit broadcasting semantics.
#ifndef NDEBUG
    for (int64_t i = 0; i < numBatchDims; i++)
      assert(operandType.getDimSize(i) == indicesType.getDimSize(i) &&
             "expected batch dims to be the same size for indices and operand");
#endif // NDEBUG

    TensorValue result =
        rewriter
            .create<tensorrt::GatherOp>(loc, resultType, /*data=*/operand,
                                        /*indices=*/indices,
                                        /*axis=*/axis,
                                        /*numElementWiseDims=*/numBatchDims)
            .getResult();

    // Check if we need to expand the shape back up.
    if (result.getType().getRank() != op.getType().getRank())
      result =
          rewriter.create<tensorrt::ExpandRankOp>(loc, op.getType(), result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert `stablehlo.reverse` using `tensorrt.slice`.
struct ReverseConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReverseOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: does this op even get used in dynamic shape situations? That edge
    // case could be handled, but it doesn't look like a common use-case for
    // this op, so abort on unknown shapes for now.
    if (!op.getType().hasStaticShape())
      return failure();
    TensorType operandType = op.getOperand().getType();
    int64_t rank = operandType.getRank();
    SmallVector<int32_t> offsets(rank, 0);
    FailureOr<SmallVector<int32_t>> sizes =
        truncateI64ToI32(op.getLoc(), operandType.getShape());
    if (failed(sizes))
      return failure();
    SmallVector<int32_t> strides(rank, 1);
    assert(offsets.size() == sizes->size() &&
           offsets.size() == strides.size() && "invalid offsets/sizes/strides");
    FailureOr<SmallVector<int32_t>> dims =
        truncateI64ToI32(op.getLoc(), op.getDimensions());
    if (failed(dims))
      return failure();
    for (int32_t dimIdx : *dims) {
      // Sanity check: this should b e handled in verifier of target op.
      assert(dimIdx >= 0 && static_cast<unsigned>(dimIdx) < offsets.size() &&
             "dimension index out-of-bounds");
      // Adjust offset to end and stride to -1.
      strides[dimIdx] = -1;
      offsets[dimIdx] =
          static_cast<int32_t>(operandType.getDimSize(dimIdx)) - 1;
    }
    rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(op, adaptor.getOperand(),
                                                   offsets, *sizes, strides);
    return success();
  }
};

/// Convert `stablehlo.pad` to `tensorrt.slice`.
struct PadConverter : public ConvertHloOpToTensorRTPattern<stablehlo::PadOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(adaptor.getOperand().getType());
    Location loc = op.getLoc();
    // We don't handle non-zero interior padding (padding between elements).
    FailureOr<SmallVector<int32_t>> interiorPadding =
        truncateI64ToI32(loc, op.getInteriorPadding());
    if (failed(interiorPadding) ||
        !llvm::all_of(*interiorPadding, [](int32_t x) { return x == 0; }))
      return failure();
    FailureOr<SmallVector<int32_t>> edgePaddingHigh =
        truncateI64ToI32(loc, op.getEdgePaddingHigh());
    FailureOr<SmallVector<int32_t>> edgePaddingLow =
        truncateI64ToI32(loc, op.getEdgePaddingLow());
    if (failed(edgePaddingHigh) || failed(edgePaddingLow))
      return failure();

    // Low padding becomes the slice offset.
    SmallVector<int32_t> sliceOffset(edgePaddingLow->size());
    for (unsigned i = 0; i < edgePaddingLow->size(); i++) {
      // There is a `stablehlo.dynamic_pad` for when dynamic dims are padded.
      if ((*edgePaddingLow)[i] != 0 && (*edgePaddingHigh)[i] != 0 &&
          inputType.isDynamicDim(i))
        return failure();
      // Convert positive to negative for the pre-padding. Negative padding
      // means positive slice offset.
      sliceOffset[i] = -1 * (*edgePaddingLow)[i];
    }

    // Proceed along a simplified path if the in the shape is fully static.
    if (inputType.hasStaticShape()) {
      SmallVector<int32_t> size(op.getType().getShape());
      SmallVector<int32_t> stride(op.getType().getRank(), 1);
      assert(size.size() == edgePaddingLow->size() &&
             size.size() == edgePaddingHigh->size());
      rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
          op,
          /*input=*/adaptor.getOperand(),
          /*offset=*/*edgePaddingLow,
          /*size=*/size,
          /*stride=*/stride,
          /*slice_mode=*/tensorrt::SliceMode::kFILL,
          /*fill=*/adaptor.getPaddingValue());
      return success();
    }

    // Otherwise, calculate size = shape(input) + pad_low + pad_high.
    Value shape = rewriter.create<tensorrt::ShapeOp>(loc, adaptor.getOperand());
    auto shapeTensorType =
        RankedTensorType::get({inputType.getRank()}, rewriter.getI32Type());
    Value padHighConst = rewriter.create<tensorrt::ConstantOp>(
        loc, shapeTensorType,
        DenseIntElementsAttr::get(shapeTensorType, *edgePaddingHigh));
    Value padLowConst = rewriter.create<tensorrt::ConstantOp>(
        loc, shapeTensorType,
        DenseIntElementsAttr::get(shapeTensorType, *edgePaddingLow));
    Value size = rewriter.create<tensorrt::ElementWiseOp>(
        loc, shapeTensorType, padLowConst, padHighConst,
        tensorrt::ElementWiseOperation::kSUM);
    size = rewriter.create<tensorrt::ElementWiseOp>(
        loc, shapeTensorType, size, shape,
        tensorrt::ElementWiseOperation::kSUM);

    SmallVector<int32_t> stride(inputType.getRank(), 1);
    rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
        op, op.getType(),
        /*input=*/adaptor.getOperand(),
        /*offset=*/DenseI32ArrayAttr::get(rewriter.getContext(), sliceOffset),
        /*size=*/size,
        /*stride=*/DenseI32ArrayAttr::get(rewriter.getContext(), stride),
        /*mode=*/tensorrt::SliceMode::kFILL,
        /*fill=*/adaptor.getPaddingValue());
    return success();
  }
};

/// Convert `stablehlo.dynamic_pad` to `tensorrt.slice`
struct DynamicPadConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::DynamicPadOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::DynamicPadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // We don't handle non-zero interior padding (padding between elements).
    Value interiorPadding = adaptor.getInteriorPadding();
    if (!mlir::matchPattern(interiorPadding, m_Zero()))
      return failure();

    auto inputType = cast<RankedTensorType>(adaptor.getOperand().getType());

    auto loc = op->getLoc();

    auto edgePaddingLow = cast<TensorValue>(adaptor.getEdgePaddingLow());
    auto edgePaddingLowType = cast<RankedTensorType>(edgePaddingLow.getType());
    if (edgePaddingLowType.getElementType().isInteger(32)) {
      edgePaddingLowType = RankedTensorType::Builder(edgePaddingLowType)
                               .setElementType(rewriter.getF32Type());
      edgePaddingLow = DynamicPadConverter::castTensor(
          rewriter, edgePaddingLowType, edgePaddingLow);
    }
    TensorValue sliceOffset =
        rewriter
            .create<tensorrt::UnaryOp>(loc, edgePaddingLowType, edgePaddingLow,
                                       tensorrt::UnaryOperation::kNEG)
            .getResult();
    if (edgePaddingLowType != adaptor.getEdgePaddingLow().getType())
      sliceOffset = DynamicPadConverter::castTensor(
          rewriter, adaptor.getEdgePaddingLow().getType(), sliceOffset);

    // calculate size = shape(input) + pad_low + pad_high.
    Value shape = rewriter.create<tensorrt::ShapeOp>(loc, adaptor.getOperand());
    auto shapeTensorType =
        RankedTensorType::get({inputType.getRank()}, rewriter.getI32Type());
    Value size = rewriter.create<tensorrt::ElementWiseOp>(
        loc, shapeTensorType, adaptor.getEdgePaddingLow(),
        adaptor.getEdgePaddingHigh(), tensorrt::ElementWiseOperation::kSUM);
    size = rewriter.create<tensorrt::ElementWiseOp>(
        loc, shapeTensorType, size, shape,
        tensorrt::ElementWiseOperation::kSUM);

    SmallVector<int32_t> stride(inputType.getRank(), 1);

    rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
        op, op.getType(),
        /*input=*/adaptor.getOperand(),
        /*offset=*/sliceOffset,
        /*size=*/size,
        /*static_stride*/ DenseI32ArrayAttr::get(rewriter.getContext(), stride),
        /*mode*/ tensorrt::SliceMode::kFILL,
        /*fill=*/adaptor.getPaddingValue());

    return success();
  }
};

/// Convert `stablehlo.get_dimension_size` to `tensorrt.constant`
struct GetDimensionSizeConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::GetDimensionSizeOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::GetDimensionSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value shape = rewriter.create<tensorrt::ShapeOp>(loc, adaptor.getOperand());
    auto axis = static_cast<int32_t>(op.getDimension());

    SmallVector<int32_t> strides{1};
    SmallVector<int32_t> offsets{axis};
    SmallVector<int32_t> sizes{1};

    Value slice =
        rewriter.create<tensorrt::SliceOp>(loc, shape, offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), slice);

    return success();
  }
};

/// Convert 'stablehlo.log_plus_one' op to TensorRT
/// Uses an approximation from
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp#L745
/// Approximate log(1+x)
/// u = x + 1.0;
/// if (u == 1.0 || u == inf) return x;
/// return x * log(u) / (u - 1.0);
struct Log1pConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::Log1pOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value x = adaptor.getOperand();
    TensorType inputType = cast<TensorType>(x.getType());

    auto replaceWithMaybeCast = [&](TensorValue replacement) {
      if (replacement.getType() != op.getType())
        replacement = castTensor(rewriter, op.getType(), replacement);
      rewriter.replaceOp(op, replacement);
    };

    // Approximations work for only f32 operands
    // Stable HLO passes I32 after catsing into F32. F16 is casted explicitly.
    if (inputType.getElementType().isF16()) {
      inputType = inputType.clone(rewriter.getF32Type());
      x = rewriter.create<tensorrt::IdentityOp>(op->getLoc(), inputType, x);
    }
    RankedTensorType constOneType = RankedTensorType::get(
        SmallVector<int64_t>(cast<RankedTensorType>(inputType).getRank(), 1),
        inputType.getElementType());
    Value constOne = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*type=*/constOneType,
        /*weights=*/
        DenseElementsAttr::get(constOneType, rewriter.getF32FloatAttr(1.0)));
    Value u = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/x,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUM);
    RankedTensorType compareType = RankedTensorType::get(
        cast<RankedTensorType>(inputType).getShape(), rewriter.getI1Type());
    Value uEqualConstOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type*/ compareType,
        /*input1=*/u,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);
    Value logU = rewriter.create<tensorrt::UnaryOp>(
        op->getLoc(),
        /*type*/ inputType,
        /*input=*/u,
        /*unaryOperation=*/tensorrt::UnaryOperation::kLOG);
    Value uEqualInf = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type*/ compareType,
        /*input1=*/u,
        /*input2=*/logU,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);
    Value condition = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type*/ compareType,
        /*input1=*/uEqualConstOne,
        /*input2=*/uEqualInf,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kOR);
    Value uSubConstOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/u,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUB);
    Value logUDivUSubConstOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/logU,
        /*input2=*/uSubConstOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kDIV);
    Value largeLog = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/x,
        /*input2=*/logUDivUSubConstOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);

    auto approximation =
        rewriter
            .create<tensorrt::SelectOp>(op.getLoc(),
                                        /*type=*/inputType,
                                        /*condition=*/condition,
                                        /*thenInput=*/x,
                                        /*elseInput=*/largeLog)
            .getResult();

    replaceWithMaybeCast(approximation);
    return success();
  }
};

/// Convert 'stablehlo.exponential_minus_one' op to TensorRT.
/// Approximation from MLIR math dialect is used
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp#L1035
/// Following approximation equation is implemented
/// expm1(x) = exp(x) - 1 = u - 1
/// Special cases
/// a. when x is near 0 i.e. u~= 1
/// b. when x is ~= -inf i.e. u - 1 ~= -1 (note: e^(-inf) = 0)
///
/// u = exp(x)
/// logU = log(u) ~= x
/// expm1 = (u-1) * (x / ~x)
/// isUInf = (exp(x) == +inf) = (logU == u)
/// expm1 = isUInf ? u : expm1
/// if (u == 1)
///   return x
/// else
///   if ((u-1) == -1)
///     return -1
///   expm1
struct Expm1OpConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::Expm1Op> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::Expm1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value x = adaptor.getOperand();
    TensorType inputType = cast<TensorType>(x.getType());

    auto replaceWithMaybeCast = [&](TensorValue replacement) {
      if (replacement.getType() != op.getType())
        replacement = castTensor(rewriter, op.getType(), replacement);
      rewriter.replaceOp(op, replacement);
    };

    // Approximations work for only f32 operands
    // Stable HLO passes I32 after catsing into F32. F16 is casted explicitly.
    if (inputType.getElementType().isF16()) {
      inputType = inputType.clone(rewriter.getF32Type());
      x = rewriter.create<tensorrt::IdentityOp>(op->getLoc(), inputType, x);
    }
    RankedTensorType constOneType = RankedTensorType::get(
        SmallVector<int64_t>(cast<RankedTensorType>(inputType).getRank(), 1),
        inputType.getElementType());
    Value constOne = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*type=*/constOneType,
        /*weights=*/
        DenseElementsAttr::get(constOneType, rewriter.getF32FloatAttr(1.0)));
    Value constNegOne = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*type=*/constOneType,
        /*weights=*/
        DenseElementsAttr::get(constOneType, rewriter.getF32FloatAttr(-1.0)));
    Value u = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input=*/x,
        /*unaryOperation=*/tensorrt::UnaryOperation::kEXP);
    RankedTensorType compareType = RankedTensorType::get(
        cast<RankedTensorType>(inputType).getShape(), rewriter.getI1Type());

    // TensorRT doesn't have "unordered equal" operator that can check equality
    // as well as NaN values. Thus we implement (u==1 || u==NaN) in multiple
    // steps as follows,
    // a. Compute (u==1)
    // b. When element is NaN, (u==u) is False. Thus, NOT(u==u) is similar to
    // (u==NaN) c. (u==1) || NOT(u==u)
    Value uEqualConstOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input1=*/u,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);
    Value uEqualU = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input1=*/u,
        /*input2=*/u,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);
    Value uEqualNaN = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input=*/uEqualU,
        /*unaryOperation=*/tensorrt::UnaryOperation::kNOT);
    Value uEqualConstOneOrNaN = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input1=*/uEqualConstOne,
        /*input2=*/uEqualNaN,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kOR);
    Value uMinusOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/u,
        /*input2=*/constOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUB);
    Value uMinusOneEqualConstNegOne = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input1=*/uMinusOne,
        /*input2=*/constNegOne,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);
    Value logU = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input=*/u,
        /*unaryOperation=*/tensorrt::UnaryOperation::kLOG);
    Value isUInf = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/compareType,
        /*input1=*/logU,
        /*input2=*/u,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kEQUAL);

    // (x / ~x)
    Value xDivLogU = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/x,
        /*input2=*/logU,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kDIV);

    // expm1 = (u-1) * (x / ~x)
    Value expm1 = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*input1=*/uMinusOne,
        /*input2=*/xDivLogU,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);

    // expm1 = isUInf ? u : expm1
    expm1 = rewriter.create<tensorrt::SelectOp>(op.getLoc(),
                                                /*type=*/inputType,
                                                /*condition=*/isUInf,
                                                /*thenInput=*/u,
                                                /*elseInput=*/expm1);
    Value checkUMinusOneEqualConstNegOne = rewriter.create<tensorrt::SelectOp>(
        op.getLoc(),
        /*type=*/inputType,
        /*condition=*/uMinusOneEqualConstNegOne,
        /*thenInput=*/constNegOne,
        /*elseInput=*/expm1);

    auto approximation = rewriter
                             .create<tensorrt::SelectOp>(
                                 op.getLoc(),
                                 /*type=*/inputType,
                                 /*condition=*/uEqualConstOneOrNaN,
                                 /*thenInput=*/x,
                                 /*elseInput=*/checkUMinusOneEqualConstNegOne)
                             .getResult();

    replaceWithMaybeCast(approximation);
    return success();
  }
};

/// Convert 'stablehlo.batch_norm_inference' to TensorRT.
struct BatchNormInferenceOpConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::BatchNormInferenceOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::BatchNormInferenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    if (resultType.isDynamicDim(op.getFeatureIndex()))
      return rewriter.notifyMatchFailure(
          op, "TensorRT does not support dynamic feature dim");
    // scale(s), offset(o), mean(m) and variance(v) are 1D tensor of size equal
    // to feature dimension. We expand their ranks to be broadcastable in
    // TensorRT elementwise ops. We find a single broadcasted shape for all
    // above tensors based on feature_index and result shape.
    SmallVector<int64_t> broadcastedShape1DTensors(resultType.getShape());
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      if (i != static_cast<int64_t>(op.getFeatureIndex()))
        broadcastedShape1DTensors[i] = 1;
    }
    RankedTensorType broadcastedShape1DTensorsType =
        RankedTensorType::Builder(resultType)
            .setShape(broadcastedShape1DTensors);

    auto doBroadcast = [&](Value v) {
      return rewriter.create<tensorrt::ExpandRankOp>(
          op.getLoc(),
          /*type=*/broadcastedShape1DTensorsType,
          /*input=*/v);
    };
    Value broadcastedScale = doBroadcast(adaptor.getScale());
    Value broadcastedOffset = doBroadcast(adaptor.getOffset());
    Value broadcastedMean = doBroadcast(adaptor.getMean());
    Value broadcastedVariance = doBroadcast(adaptor.getVariance());

    // Both input tensor x type and thus result type is either FP16 or FP32.
    // Default `epsilon`(e) type is FP32. In case, input is not FP32, `epsilon`
    // needs to be casted to the input type.
    FloatAttr epsAttr = op.getEpsilonAttr();
    FloatType inpCompatibleEpsFloatType =
        cast<FloatType>(resultType.getElementType());
    if (inpCompatibleEpsFloatType != epsAttr.getType()) {
      // Cast
      APFloat epsAPFloat = epsAttr.getValue();
      bool isInfoLost;
      auto status =
          epsAPFloat.convert(inpCompatibleEpsFloatType.getFloatSemantics(),
                             APFloat::rmNearestTiesToEven, &isInfoLost);
      if (((status & APFloat::opInexact) == APFloat::opInexact) ||
          ((status & (~APFloat::opInexact)) == APFloat::opOK)) {
        // Allow inexact conversion since conversion of 1e-5 from FP32 to FP16
        // gives an underflow op status for APFloat conversion.
        epsAttr = rewriter.getFloatAttr(inpCompatibleEpsFloatType, epsAPFloat);
      } else {
        if (isInfoLost)
          return rewriter.notifyMatchFailure(
              op, "information is lost in the epsilon conversion.");
        return rewriter.notifyMatchFailure(
            op, "could not convert epsilon type to the input type.");
      }
    }
    RankedTensorType epsilonType =
        RankedTensorType::Builder(resultType)
            .setShape(SmallVector<int64_t>(resultType.getRank(), 1));
    Value epsilon = rewriter.create<tensorrt::ConstantOp>(
        op.getLoc(),
        /*type=*/epsilonType,
        /*weight=*/
        DenseElementsAttr::get(epsilonType, epsAttr.getValue()));

    // Input tensor x is normalized at inference time as follows.
    // x = x - m
    Value centeredOperand = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/adaptor.getOperand(),
        /*input2=*/broadcastedMean,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUB);
    // stddev = sqrt(v + e)
    Value updatedVariance = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/broadcastedVariance,
        /*input2=*/epsilon,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUM);
    Value stddev = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(),
        /*type=*/broadcastedShape1DTensorsType,
        /*input=*/updatedVariance,
        /*unaryOperation=*/tensorrt::UnaryOperation::kSQRT);
    // x = (x-m) / stedev
    Value normalizedOperand = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/centeredOperand,
        /*input2=*/stddev,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kDIV);
    // x = (x-m)/stddev * scale
    Value scaledOperand = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/normalizedOperand,
        /*input2=*/broadcastedScale,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kPROD);
    // x = (x-m)/stddev * scale + offset
    Value shiftedOperand = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(),
        /*input1=*/scaledOperand,
        /*input2=*/broadcastedOffset,
        /*elementwiseOperation=*/tensorrt::ElementWiseOperation::kSUM);

    rewriter.replaceOp(op, shiftedOperand);
    return success();
  };
};

/// Convert `stablehlo.uniform_quantize` to `tensorrt.quantize`
struct UniformQuantizeConverter
    : public ConvertOpToTensorRTPattern<stablehlo::UniformQuantizeOp> {
  using ConvertOpToTensorRTPattern::ConvertOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::UniformQuantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto quantizedType = cast<mlir::quant::UniformQuantizedType>(
        cast<ShapedType>(op.getType()).getElementType());

    if (quantizedType.getStorageType().isUnsignedInteger(8) ||
        quantizedType.getZeroPoint() != 0)
      return failure();

    // Construct constants from UniformQuantizedType
    SmallVector<int64_t> scaleShape(0, 0);
    RankedTensorType scaleType =
        RankedTensorType::get(scaleShape, rewriter.getF32Type());
    Value scaleValue = rewriter.create<tensorrt::ConstantOp>(
        loc,
        DenseElementsAttr::get(scaleType, float(quantizedType.getScale())));

    rewriter.replaceOpWithNewOp<tensorrt::QuantizeOp>(
        op, op.getType(), adaptor.getOperand(), scaleValue, IntegerAttr());
    return success();
  }
};

/// Convert `stablehlo.uniform_dequantize` to `tensorrt.dequantize`
struct UniformDequantizeConverter
    : public ConvertOpToTensorRTPattern<stablehlo::UniformDequantizeOp> {
  using ConvertOpToTensorRTPattern::ConvertOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::UniformDequantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Parse scale and zero_point info from UniformQuantizedType first
    auto quantizedType = cast<mlir::quant::UniformQuantizedType>(
        cast<ShapedType>(adaptor.getOperand().getType()).getElementType());
    if (quantizedType.getStorageType().isUnsignedInteger(8) ||
        quantizedType.getZeroPoint() != 0)
      return failure();

    SmallVector<int64_t> scaleShape(0, 0);

    RankedTensorType scaleType =
        RankedTensorType::get(scaleShape, rewriter.getF32Type());
    Value scaleValue = rewriter.create<tensorrt::ConstantOp>(
        loc, scaleType,
        DenseElementsAttr::get(scaleType, float(quantizedType.getScale())));

    rewriter.replaceOpWithNewOp<tensorrt::DequantizeOp>(
        op, op.getType(), adaptor.getOperand(), scaleValue, IntegerAttr());

    return success();
  }
};

/// Computes effective padding added by the TensorRT deconv op.
/// For the spatial dimension `i`, if StableHLO conv padding is `p[i]`,
/// effective padding added by the the TensorRT is `tensorrtPad[i] = dilation *
/// (kernelSize - 1) - p[i]`
/// (https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html).
/// NOTE: Kernel spatial dimensions must be static.
static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getEffectiveTensorRTPaddings(ArrayRef<int64_t> prePadding,
                             ArrayRef<int64_t> postPadding,
                             ArrayRef<int64_t> dilation,
                             RankedTensorType kernelType,
                             int64_t numSpatialDims) {
  SmallVector<int64_t> trtPrePadding(numSpatialDims);
  SmallVector<int64_t> trtPostPadding(numSpatialDims);
  for (int64_t i = 0; i < numSpatialDims; i++) {
    if (kernelType.isDynamicDim(2 + i))
      return failure();
    trtPrePadding[i] =
        dilation[i] * (kernelType.getDimSize(2 + i) - 1) - prePadding[i];
    // Stable HLO supports negative padding. If padding is negative, zeros (if
    // padding mode if `zero`) are padded to the output, even in the case of
    // deconvolution. For example, If original output is 1x1x6x6xf32 for 0
    // effective padding. Given everything else is constant, effective padding
    // of 1 will result in 1x1x4x4xf32 output (output is trimmed, as in deconv)
    // and effective padding of -1 will result in 1x1x8x8xf32 (i.e. zeros are
    // padded to the output). TensorRT doesn't suppport negative effective
    // padding i.e. it strictly remove elements if padding is provided.
    if (trtPrePadding[i] < 0)
      return failure();
    trtPostPadding[i] =
        dilation[i] * (kernelType.getDimSize(2 + i) - 1) - postPadding[i];
    if (trtPostPadding[i] < 0)
      return failure();
  }
  return std::make_pair(trtPrePadding, trtPostPadding);
}

// Modifies Stable HLO convolution kernel when convolution is representing
// deconvolution operation. Stable HLO convolution weights are in the form
// [c_out, c_in/num_groups,...] after input preprocessing pass. However,
// TensorRT deconv op expects weights in the shape [c_in, c_out/num_groups,
// ...]. This transformation happens in four stages, 0. Flip the spatial
// dimensions of the kernel
// 1. [c_out, c_in/num_groups, ...] -> [num_groups, c_out/num_groups,
// c_in/num_groups, ....] (ReshapeOp)
// 2. [num_groups, c_out/num_groups, c_in/num_groups, ....] ->
// [num_groups, c_in/num_groups, c_out/num_groups, ....] (TransposeOp)
// 3. [num_groups, c_in/num_groups, c_out/num_groups, ....] -> [c_in,
// c_out/num_groups, ....] (ReshapeOp)
// If num_groups=1, simple transpose works.
static Value prepareHloConvKernelForDeconvCase(RewriterBase &rewriter,
                                               Location loc, Value kernel,
                                               int64_t numGroups) {
  RankedTensorType kernelType = cast<RankedTensorType>(kernel.getType());
  // Flip the spatial dimensions
  SmallVector<int32_t> startIdx(kernelType.getRank(), 0);
  SmallVector<int32_t> strides(kernelType.getRank(), 1);
  SmallVector<int32_t> size(kernelType.getRank());
  assert(kernelType.getRank() >= 3 && "Minimum kernel rank expected is 3");
  // This is safe since kernel will have at least rank 3 (1d deconv case).
  size[0] = static_cast<int32_t>(kernelType.getDimSize(0));
  size[1] = static_cast<int32_t>(kernelType.getDimSize(1));
  for (int i = kernelType.getRank() - 1; i > 1; i--) {
    startIdx[i] = kernelType.getDimSize(i) - 1;
    strides[i] *= -1;
    size[i] = static_cast<int32_t>(kernelType.getDimSize(i));
  }
  Value flipped = rewriter.create<tensorrt::SliceOp>(loc, kernel,
                                                     /*offsets=*/startIdx,
                                                     /*sizes=*/size,
                                                     /*strides=*/strides);

  if (numGroups == 1) {
    SmallVector<unsigned> perm =
        llvm::to_vector(llvm::seq<unsigned>(0, kernelType.getRank()));
    std::swap(perm[0], perm[1]);
    Value transpose =
        rewriter
            .create<tensorrt::TransposeOp>(
                loc, flipped,
                AffineMap::getPermutationMap(perm, rewriter.getContext()))
            .getResult();
    return transpose;
  }
  SmallVector<int64_t> firstReshapeShape(kernelType.getShape());
  firstReshapeShape[0] /= numGroups;
  // Vector will be very small, so insert in the beginning is okay.
  firstReshapeShape.insert(firstReshapeShape.begin(), numGroups);
  Value firstReshape =
      rewriter
          .create<tensorrt::ReshapeOp>(
              loc,
              RankedTensorType::get(firstReshapeShape,
                                    kernelType.getElementType()),
              flipped)
          .getResult();

  SmallVector<unsigned> perm =
      llvm::to_vector(llvm::seq<unsigned>(0, (kernelType.getRank() + 1)));
  std::swap(perm[1], perm[2]);
  Value transpose =
      rewriter
          .create<tensorrt::TransposeOp>(
              loc, firstReshape,
              AffineMap::getPermutationMap(perm, rewriter.getContext()))
          .getResult();

  std::swap(firstReshapeShape[1], firstReshapeShape[2]);
  firstReshapeShape.erase(firstReshapeShape.begin());
  firstReshapeShape[0] *= numGroups;

  Value secondReshape =
      rewriter
          .create<tensorrt::ReshapeOp>(
              loc,
              RankedTensorType::get(firstReshapeShape,
                                    kernelType.getElementType()),
              transpose)
          .getResult();
  return secondReshape;
}

/// The `stablehlo.conv` can represent both convolution and transpose
/// convolution (aka de-convolution). Generally, if LHS dilation is present,
/// `stablehlo.conv` represents transpose convolution. This LHS dilation is same
/// as (and thus maps to) the `stride` of transpose convolution layer of
/// frameworks (e.g. TensorRT) which has transpose convolution and convolution
/// as a separate layer. However, JAX (LAX) convolution (which is convolution
/// operation represented by `stablehlo.conv`) operation doesn't drop LHS
/// dilation but instead sets it to all unit dimensions. This creates an issue
/// on how to distinguish between the following cases,
/// 1. LHS dilation is all ones, IR is coming from JAX so its convolution.
/// 2. LHS dilation is all ones, IR is coming from other frontend framwork
/// which maps unit `stride` of its transpose convolution operation to LHS
/// dilation.
/// From above, the 1st case is mapped to `tensorrt.convolution`. One way to
/// detect second case is checking if paddings are negative. Convolution
/// operation doesn't support negative padding. Thus, if paddings is negative
/// and all LHS dilation values are unit dimensions, we map such
/// `stablehlo.conv` to `tensorrt.deconvolution`. If any of LHS dilation value
/// is non unit, it is simple case of transpose convolution so is mapped to
/// `tensorrt.deconvolution`.
static bool isCanonicalDeconvolution(ArrayRef<int64_t> lhsDilation,
                                     ArrayRef<int64_t> prePadding,
                                     ArrayRef<int64_t> postPadding) {
  return llvm::any_of(lhsDilation, [](const int64_t &v) { return v != 1; }) ||
         (llvm::all_of(lhsDilation, [](const int64_t &v) { return v == 1; }) &&
          llvm::all_of(prePadding, [](const int64_t &v) { return v < 0; }) &&
          llvm::all_of(postPadding, [](const int64_t &v) { return v < 0; }));
}

static SmallVector<int64_t>
canonicalizeDilation(std::optional<ArrayRef<int64_t>> dilation,
                     unsigned numSpatialDims) {
  if (!dilation || dilation->empty())
    return SmallVector<int64_t>(numSpatialDims, 1);
  return llvm::to_vector(*dilation);
}

/// Convert `stablehlo.convolution` to `tensorrt.convolution` or
/// `tensorrt.deconvolution`.
struct ConvolutionConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::ConvolutionOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "could not convert result type");

    // MHLO allows representing convolution
    // ops that has 0 elements in the
    // output. Such operations can't be
    // handled by TensorRT.
    if (cast<RankedTensorType>(resultType).hasStaticShape() &&
        cast<RankedTensorType>(resultType).getNumElements() == 0)
      return rewriter.notifyMatchFailure(op, "result has 0 elements");

    // Stable HLO uses an attribute to specify what each dimension means.
    stablehlo::ConvDimensionNumbersAttr dimNumbers = op.getDimensionNumbers();

    const int64_t numSpatialDims =
        static_cast<int64_t>(dimNumbers.getInputSpatialDimensions().size());

    if (numSpatialDims != 2 && numSpatialDims != 3)
      return rewriter.notifyMatchFailure(op, "spatial dims should be 2 or 3");
    if (dimNumbers.getInputBatchDimension() != 0 ||
        dimNumbers.getOutputBatchDimension() != 0 ||
        dimNumbers.getInputFeatureDimension() != 1 ||
        dimNumbers.getOutputFeatureDimension() != 1)
      return rewriter.notifyMatchFailure(op, "batch dim should be in "
                                             "position 0");
    if (op.getBatchGroupCount() != 1)
      return rewriter.notifyMatchFailure(op, "groups not supported");
    const uint32_t numGroups = static_cast<uint32_t>(op.getFeatureGroupCount());

    // stablehlo makes strides/padding optional and infers some default value,
    // which makes us do some extra work.

    // Default value: one for each of the
    // spatial dimension.
    SmallVector<int64_t> windowStrides =
        op.getWindowStrides().has_value()
            ? llvm::to_vector(*op.getWindowStrides())
            : SmallVector<int64_t>(numSpatialDims, 1);

    // Default value: one for each of the
    // spatial dimension.
    SmallVector<int64_t> rhsDilation =
        canonicalizeDilation(op.getRhsDilation(), numSpatialDims);
    SmallVector<int64_t> lhsDilation =
        canonicalizeDilation(op.getLhsDilation(), numSpatialDims);

    // TODO: we could support "all true" or
    // "all false", but right now just
    // support "all false".
    ArrayRef<bool> reversal;
    if (op.getWindowReversal().has_value())
      reversal = *op.getWindowReversal();
    if (llvm::count(reversal, true) > 0)
      return failure();

    // Separate Nx2 padding attribute into
    // pre/post padding.
    FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> padding =
        convertPaddingAttribute(loc, op.getPadding());
    if (failed(padding))
      return failure();
    auto [prePadding, postPadding] = *padding;
    if (prePadding.empty())
      prePadding = SmallVector<int64_t>(numSpatialDims, 0);
    if (postPadding.empty())
      postPadding = SmallVector<int64_t>(numSpatialDims, 0);

    if (isCanonicalDeconvolution(lhsDilation, prePadding, postPadding)) {
      RankedTensorType kernelType = op.getRhs().getType();
      // LHS dilation is passed as stride.
      // Get paddings from flattened stable hlo conv op paddings.
      FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
          trtPadding = getEffectiveTensorRTPaddings(
              prePadding, postPadding, rhsDilation, kernelType, numSpatialDims);
      if (failed(trtPadding))
        return rewriter.notifyMatchFailure(
            op,
            "could not convert stablehlo convolution with `lhs_dilation` to "
            "TensorRT deconvolution op. This might be because kernel "
            "spatial dimension is not static or effective padding is "
            "negative.");

      Value updatedKernel = prepareHloConvKernelForDeconvCase(
          rewriter, op->getLoc(), op.getRhs(), numGroups);

      rewriter.replaceOpWithNewOp<tensorrt::DeconvolutionOp>(
          op, op.getType(),
          /*input=*/op.getLhs(),
          /*kernelWeights=*/updatedKernel,
          /*biasWeights=*/Value(),
          /*stride=*/lhsDilation,
          /*pre_padding=*/(*trtPadding).first,
          /*post_padding=*/(*trtPadding).second,
          /*num_groups=*/numGroups,
          /*dilation=*/rhsDilation);

      return success();
    }

    rewriter.replaceOpWithNewOp<tensorrt::ConvolutionOp>(
        op, op.getType(),
        /*input=*/op.getLhs(),
        /*kernel=*/op.getRhs(),
        /*bias=*/Value(),
        /*stride=*/windowStrides,
        /*pre_padding=*/prePadding,
        /*post_padding=*/postPadding,
        /*num_groups=*/numGroups,
        /*dilation=*/rhsDilation);
    return success();
  }
};

/// Convert `stablehlo.scatter` to `tensorrt.scatter`. We expect
/// `stablehlo.scatter` to appear in a canonical form corresponding to the "nD"
/// scatter mode.
struct ConvertScatterToTensorRT
    : public ConvertHloOpToTensorRTPattern<stablehlo::ScatterOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!tensorrt::isCanonicalScatterNd(op))
      return rewriter.notifyMatchFailure(
          op, "can only convert ops that are in ScatterNd format");
    auto indicesType =
        cast<RankedTensorType>(adaptor.getScatterIndices().getType());
    auto updatesType =
        cast<RankedTensorType>(adaptor.getUpdates().front().getType());
    assert(indicesType.getDimSize(0) == updatesType.getDimSize(0) &&
           "expected first dim of indices/updates to be identical");
    SmallVector<Value> replacements;
    replacements.reserve(adaptor.getInputs().size());
    for (auto [input, update] :
         llvm::zip(adaptor.getInputs(), adaptor.getUpdates()))
      replacements.push_back(rewriter.create<tensorrt::ScatterOp>(
          op.getLoc(), input, adaptor.getScatterIndices(), update));
    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

namespace {
/// Convert `stablehlo.scatter` that conducts the slice update to
/// `tensorrt.scatter_elements`.
struct ConvertScatterToTensorRTScatterElements
    : public ConvertHloOpToTensorRTPattern<stablehlo::ScatterOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType inputsType =
        cast<RankedTensorType>(adaptor.getInputs().getType().front());
    RankedTensorType indicesType =
        cast<RankedTensorType>(adaptor.getScatterIndices().getType());
    RankedTensorType updatesType =
        cast<RankedTensorType>(adaptor.getUpdates().front().getType());
    stablehlo::ScatterDimensionNumbersAttr dimsAttrs =
        op.getScatterDimensionNumbers();
    int64_t inputsTypeRank = inputsType.getRank();
    int64_t updatesTypeRank = updatesType.getRank();
    auto isSeq = [](ArrayRef<int64_t> ar, int64_t start, int64_t end) {
      return llvm::equal(ar, llvm::seq<int64_t>(start, end));
    };

    bool expectedForm =
        indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
        isSeq(dimsAttrs.getUpdateWindowDims(), 1, updatesTypeRank) &&
        isSeq(dimsAttrs.getScatterDimsToOperandDims(), 0,
              indicesType.getDimSize(1)) &&
        adaptor.getInputs().size() == 1 && indicesType.getDimSize(0) == 1 &&
        inputsType.getDimSize(0) == 1;
    bool sliceUpdate = updatesType.getDimSize(updatesTypeRank - 1) <
                       inputsType.getDimSize(inputsTypeRank - 1);

    if (!expectedForm || !sliceUpdate)
      return failure();

    TypedValue<RankedTensorType> startIndexTensor =
        rewriter.create<tensorrt::SliceOp>(op.getLoc(),
                                           adaptor.getScatterIndices(),
                                           /*offsets=*/ArrayRef<int32_t>{0, 1},
                                           /*sizes=*/ArrayRef<int32_t>{1, 1},
                                           /*strides=*/ArrayRef<int32_t>{1, 1});
    auto startIndex = rewriter.create<tensorrt::CollapseRankOp>(
        op->getLoc(), RankedTensorType::get({}, rewriter.getI32Type()),
        startIndexTensor);

    SmallVector<int64_t> newUpdateShape(updatesType.getShape().take_back(2));
    RankedTensorType newUpdateType = updatesType.clone(newUpdateShape);
    auto newUpdates = rewriter.create<tensorrt::CollapseRankOp>(
        op->getLoc(), newUpdateType, adaptor.getUpdates().front());

    Value constOneTuple = rewriter.create<tensorrt::ConstantOp>(
        op->getLoc(),
        /*weights=*/
        DenseElementsAttr::get(
            startIndex.getType().clone(SmallVector<int64_t>{2}),
            rewriter.getI32IntegerAttr(1)));

    Value newIndices = rewriter.create<tensorrt::LinspaceOp>(
        op->getLoc(), newUpdateType.clone(rewriter.getI32Type()), Value(),
        startIndex, constOneTuple, FloatAttr(), FloatAttr());

    auto checkI1 = [&](Value v) {
      return (
          cast<RankedTensorType>(v.getType()).getElementType().isInteger(1));
    };
    auto convertToI32 = [&](Value v) {
      return rewriter.create<tensorrt::IdentityOp>(
          v.getLoc(),
          cast<RankedTensorType>(v.getType()).clone(rewriter.getI32Type()), v);
    };
    auto convertToI1 = [&](Value v) {
      return rewriter.create<tensorrt::IdentityOp>(
          v.getLoc(),
          cast<RankedTensorType>(v.getType()).clone(rewriter.getI1Type()), v);
    };

    if (checkI1(adaptor.getInputs().front())) {
      auto newOp = rewriter.create<tensorrt::ScatterElementsOp>(
          op->getLoc(), /*data*/ convertToI32(adaptor.getInputs().front()),
          /*indices*/ newIndices, /*updates*/ convertToI32(newUpdates),
          /*axis*/ rewriter.getI64IntegerAttr(1));
      rewriter.replaceOp(op, convertToI1(newOp)->getResults());
    } else {
      auto newOp = rewriter.create<tensorrt::ScatterElementsOp>(
          op->getLoc(), /*data*/ adaptor.getInputs().front(),
          /*indices*/ newIndices, /*updates*/ newUpdates,
          /*axis*/ rewriter.getI64IntegerAttr(1));
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};
} // namespace

// Conversion of `stablehlo.gather` where the gathered slices are not "full"
// slices of the data operand cannot be lowered to `tensorrt.gather`. In order
// to convert `stablehlo.gather` that is gathering partial slices, we need to
// lower it to a series of slices and concatenations. Return true if the slices
// are partial.
static bool isCanonicalGatherWithPartialSlices(stablehlo::GatherOp op) {
  if (!stablehlo::isCanonicalGather(op))
    return false;
  TensorType operandType = op.getOperand().getType();
  ArrayRef<int64_t> sliceShape = op.getSliceSizes();
  ArrayRef<int64_t> operandTrailingShape =
      operandType.getShape().take_back(sliceShape.size());
  for (auto [i, dimSize] : llvm::enumerate(sliceShape)) {
    if ((i == 0 && dimSize != 1) ||
        (i > 0 && dimSize != operandTrailingShape[i]))
      return true;
  }
  return false;
}

namespace {

/// Convert 'Simple, Single Dimension Gather' that has implicit or explicit
/// index dimension to `tensorrt.gather`.
struct SingleDimSimpleGatherToTensorRTGatherPattern
    : public ConvertHloOpToTensorRTPattern<stablehlo::GatherOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast_or_null<RankedTensorType>(
        typeConverter->convertType(op.getType()));
    if (!resultType)
      return failure();

    if (std::optional<int64_t> gatherDim =
            stablehlo::isSingleDimSimpleGatherWithImplicitIndexDim(op)) {
      rewriter.replaceOpWithNewOp<tensorrt::GatherOp>(
          op, resultType, adaptor.getOperand(), adaptor.getStartIndices(),
          *gatherDim);
      return success();
    }

    if (std::optional<int64_t> gatherDim =
            stablehlo::isSingleDimSimpleGatherWithExplicitIndexDim(op)) {
      // We just need to insert a reshape to remove the explicit index
      // dimension.
      auto indicesType =
          cast<RankedTensorType>(adaptor.getStartIndices().getType());
      indicesType = indicesType.clone(indicesType.getShape().drop_back());
      auto reshapeOp = rewriter.create<tensorrt::CollapseRankOp>(
          op.getLoc(), indicesType, adaptor.getStartIndices());
      rewriter.replaceOpWithNewOp<tensorrt::GatherOp>(
          op, resultType, adaptor.getOperand(), reshapeOp.getResult(),
          *gatherDim);
      return success();
    }
    return failure();
  }
};

struct ConvertGatherToTensorRT
    : public ConvertHloOpToTensorRTPattern<stablehlo::GatherOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A canonical gather has empty "collapsed slice dims", a start_indices
    // operand rank of 2, index_vector_dim=1, start_index_map = [0, 1, ...],
    // offset_dims = [1, 2, ...].
    // Check to make sure we are not in "simple, single dim explicit index"
    // gather since that is a different pattern.
    if (!stablehlo::isCanonicalGather(op) ||
        stablehlo::isSingleDimSimpleGatherWithExplicitIndexDim(op))
      return rewriter.notifyMatchFailure(op, "not stablehlo canonical gather");

    // Check that the indices shape at the "index vector dim" is 1. TensorRT's
    // default gather operation is akin to what one might call a "pure batch
    // gather", i.e. the only overlap between "full start index" and "full
    // offset index" would be a unit dimension, see here:
    // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather.
    if (op.getStartIndices().getType().getDimSize(1) != 1)
      return rewriter.notifyMatchFailure(
          op, "the start indices shape at index_vector_dim should be 1");

    if (isCanonicalGatherWithPartialSlices(op))
      return rewriter.notifyMatchFailure(op, "gather has partial slices");

    rewriter.replaceOpWithNewOp<tensorrt::GatherOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getOperand(),
        adaptor.getStartIndices(), /*axis=*/0);
    return success();
  }
};

struct ConvertGatherToTensorRTGatherNd
    : public ConvertHloOpToTensorRTPattern<stablehlo::GatherOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return failure();

    if (stablehlo::isSimpleLeadingMultiDimGather(op)) {
      rewriter.replaceOpWithNewOp<tensorrt::GatherNdOp>(
          op, resultType,
          /*data=*/adaptor.getOperand(), /*indices=*/adaptor.getStartIndices());
      return success();
    }

    if (stablehlo::isSimpleLeadingMultiDimGatherWithDegenerateDims(op)) {
      RankedTensorType indicesType = op.getStartIndices().getType();
      SmallVector<int64_t> resultShape(
          resultType.getShape().take_front(indicesType.getRank() - 1));
      llvm::append_range(resultShape, resultType.getShape().drop_front(
                                          indicesType.getRank() - 1 +
                                          indicesType.getShape().back()));
      Value replacement = rewriter.create<tensorrt::GatherNdOp>(
          op.getLoc(), resultType.clone(resultShape),
          /*data=*/adaptor.getOperand(), /*indices=*/adaptor.getStartIndices());
      rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(op, resultType,
                                                       replacement);
      return success();
    }

    return failure();
  }
};

/// Convert canonical `stablehlo.gather` where `isGatherWithPartialSlices` is
/// true.
struct ConvertGatherWithPartialSlicesToTensorRT
    : public ConvertHloOpToTensorRTPattern<stablehlo::GatherOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // See notes above in the `ConvertGatherWithPartialSlices`.
    // Note that these conditions enforce that `startIndices` operand is
    // rank 2.
    // TODO: indexVectorSize constraint can probably be removed.
    if (op.getStartIndices().getType().getRank() != 2)
      return failure();
    int64_t indexVectorSize = op.getStartIndices().getType().getDimSize(1);
    if (!isCanonicalGatherWithPartialSlices(op) ||
        stablehlo::isSimpleLeadingMultiDimGather(op) || indexVectorSize != 1)
      return rewriter.notifyMatchFailure(
          op, "not stablehlo canonical gather with partial slices and "
              "index_vector_dim size of 1");

    // We have to lower to a set of slices and concatenations. This is going to
    // cause a code explosion for huge gathers, so put an upper bound.
    TensorType startIndicesType = op.getStartIndices().getType();
    constexpr int64_t kMaxNumSlices = 4;
    int64_t numSlices = startIndicesType.getDimSize(0);
    if (numSlices > kMaxNumSlices)
      return rewriter.notifyMatchFailure(
          op, "more than MaxNumSlices required to lower");

    // Assemble the slices.
    ArrayRef<int64_t> sliceShape = op.getSliceSizes();

    SmallVector<Value> slices;
    slices.reserve(op.getType().getDimSize(0));
    Location loc = op.getLoc();
    int64_t operandRank = op.getOperand().getType().getRank();
    for (int64_t i = 0; i < numSlices; i++) {
      // First, create the start indices tensor for the slice.
      // It is composed of the requisite slice from `startIndices`
      // concatenated with zeros.
      TypedValue<RankedTensorType> indexSlice = rewriter.create<
          tensorrt::SliceOp>(
          loc, adaptor.getStartIndices(),
          /*offsets=*/ArrayRef<int32_t>{static_cast<int32_t>(i), 0},
          /*sizes=*/ArrayRef<int32_t>{1, static_cast<int32_t>(indexVectorSize)},
          /*strides=*/ArrayRef<int32_t>{1, 1});

      // Flatten the index slice.
      indexSlice = rewriter.create<tensorrt::CollapseRankOp>(
          loc, indexSlice.getType().clone(ArrayRef<int64_t>{indexVectorSize}),
          indexSlice);

      // Concatenate with the right number of `0`s.
      int64_t numZeros = operandRank - indexVectorSize;
      assert(numZeros >= 0 &&
             "expected non-negative number of zeros to append");
      auto constZeros = rewriter.create<tensorrt::ConstantOp>(
          loc, cast<ElementsAttr>(rewriter.getZeroAttr(
                   RankedTensorType::get({numZeros}, rewriter.getI32Type()))));

      Value operandSliceIndices = rewriter.create<tensorrt::ConcatenationOp>(
          loc, ValueRange{indexSlice, constZeros},
          /*axis=*/static_cast<int32_t>(0));

      // Slice the operand.
      auto operandSlice = rewriter.create<tensorrt::SliceOp>(
          loc, adaptor.getOperand(),
          /*offsets=*/operandSliceIndices,
          /*size=*/
          rewriter.getDenseI32ArrayAttr(llvm::map_to_vector(
              sliceShape, [](int64_t x) -> int32_t { return x; })),
          /*strides=*/
          rewriter.getDenseI32ArrayAttr(SmallVector<int32_t>(operandRank, 1)));

      // Expand rank by prepending a 1 for the indexStart batch dimension.
      SmallVector<int64_t> newShape(operandSlice.getType().getShape());
      newShape.insert(newShape.begin(), 1);
      slices.push_back(rewriter.create<tensorrt::ExpandRankOp>(
          loc, operandSlice.getType().clone(newShape), operandSlice));
    }

    // Concatenate the slices.
    rewriter.replaceOpWithNewOp<tensorrt::ConcatenationOp>(
        op, typeConverter->convertType(op.getType()), slices,
        /*axis=*/0);

    return success();
  }
};

/// Look for Stablehlo and stablehlo softmax patterns and fold them to
/// `tensorrt.softmax` when matched. The OpRewrite is rooted at
/// `stablehlo.DivOp`.
struct ConvertHLOSoftmax : public OpRewritePattern<stablehlo::DivOp> {
  using OpRewritePattern<stablehlo::DivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {
    // tensorrt.softmax takes one input and produces one output.
    mlir::Value softmaxInputOperand;
    int64_t deducedAxis = -1;
    if (!matchPattern(op.getOperation(),
                      mlir::matchers::m_StableHLOSoftmaxMatcher(
                          softmaxInputOperand, deducedAxis)))
      return failure();

    // get input to softmax op, i.e, input to reduce max.
    // also set the deducedAxis
    Value softmaxOp = rewriter.create<tensorrt::SoftMaxOp>(
        op->getLoc(), softmaxInputOperand, deducedAxis);
    rewriter.replaceOp(op, softmaxOp);
    return success();
  }
};

// clang-format off
/// Converts some `stablehlo.dynamic_update_slice` using `tensorrt.concatenation`.
/// TensorRT does not have a generic "slice insertion" operation. However, we
/// can use an equivalent formulation in terms of a concatenation when at most
/// one of the offset indices is zero and the update tensor's shape is equal to
/// the result shape in all dimensions except for the non-zero offset dimension.
///
/// ### Example (pseudo-IR):
///
/// ```
/// %1 = < insert %update into %base[0, %offset, 0, 0] >
///    :  tensor<1x1x12x64xf32> into tensor<1x20x12x64>
/// ```
///
/// becomes
///
/// ```
/// %begin_size = <create shape ([1, %offset, 12, 64])> : tensor<4xi32>
/// %begin = tensorrt.slice %arg0[0, 0, 0, 0][%begin_size][1, 1, 1, 1]
/// %end_offset = <create shape [0, %offset+1, 0, 0]
/// %end_size = <create shape [1, 20 - %offset +1, 12, 64]
/// %end = tensorrt.slice %arg0[%end_offset][%end_size][1, 1, 1, 1]
/// %1_replacement = <concat %begin, %update, %end, axis=1>
/// ```
// clang-format on
struct DynamicUpdateSliceToConcatConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::DynamicUpdateSliceOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that the update has at most 1 dimension not equal in size to the
    // corresponding dimension of the result shape and exactly one of the
    // offsets is non-zero.
    Location loc = op.getLoc();
    TensorType resultType = op.getType();
    TensorType updateType = op.getUpdate().getType();
    TypedValue<RankedTensorType> updateStartOffset{};
    std::optional<int32_t> concatAxis;
    for (auto [idx, dimSizes] : llvm::enumerate(
             llvm::zip(resultType.getShape(), updateType.getShape()))) {
      auto [resultDimSize, updateDimSize] = dimSizes;
      if (updateDimSize < resultDimSize) {
        if (concatAxis)
          return failure();
        concatAxis = idx;
        updateStartOffset = llvm::cast<TypedValue<RankedTensorType>>(
            adaptor.getStartIndices()[idx]);
        continue;
      }
      assert(updateDimSize == resultDimSize &&
             "expected update dim size to equal result dim size");
      if (!matchPattern(adaptor.getStartIndices()[idx], m_Zero()))
        return failure();
    }

    // The slice is a full over-write. This situation should be folded away
    // automatically.
    if (!updateStartOffset || !concatAxis)
      return failure();

    // Reshape the scalar offset to tensor<1xi32>.
    updateStartOffset = rewriter.create<tensorrt::ExpandRankOp>(
        loc,
        RankedTensorType::get(
            {1}, mlir::getElementTypeOrSelf(updateStartOffset.getType())),
        updateStartOffset);

    // Calculate the beginning part.
    const int64_t rank = resultType.getRank();
    SmallVector<Value> parts;
    TypedValue<RankedTensorType> sliceSize = tensorrt::scatterShapeTensor(
        rewriter, loc, resultType.getShape(), *concatAxis, updateStartOffset);
    SmallVector<int64_t> firstPartShape(resultType.getShape());
    firstPartShape[*concatAxis] = ShapedType::kDynamic;
    parts.push_back(rewriter.create<tensorrt::SliceOp>(
        loc, resultType.clone(firstPartShape), adaptor.getOperand(),
        /*start=*/rewriter.getDenseI32ArrayAttr(SmallVector<int32_t>(rank, 0)),
        /*size=*/sliceSize,
        /*strides=*/
        rewriter.getDenseI32ArrayAttr(SmallVector<int32_t>(rank, 1))));

    // Add the middle part (the update).
    parts.push_back(adaptor.getUpdate());

    // Create the end part.
    // We must calculate the start and shape of a tensor to slice from the
    // input. This will form the final element of the concatenation. Set the
    // start and shape to be the values appropriate for !hasNonZeroUpdateStart
    // (static case). We will update them in the condition block.
    // Calculate the slice start = update offset + update size.
    TypedValue<RankedTensorType> concatDimOffset =
        rewriter.create<tensorrt::ElementWiseOp>(
            loc, updateStartOffset,
            tensorrt::createConstShapeTensor(
                rewriter, loc,
                {static_cast<int32_t>(updateType.getDimSize(*concatAxis))}),
            tensorrt::ElementWiseOperation::kSUM);
    TypedValue<RankedTensorType> endOffset = tensorrt::scatterShapeTensor(
        rewriter, loc, SmallVector<int64_t>(updateType.getRank(), 0),
        *concatAxis, concatDimOffset);
    // Calculate the slice size = result shape - update offset.
    TypedValue<RankedTensorType> finalPartDimSize =
        rewriter.create<tensorrt::ElementWiseOp>(
            loc,
            tensorrt::createConstShapeTensor(
                rewriter, loc,
                {static_cast<int32_t>(resultType.getDimSize(*concatAxis))}),
            concatDimOffset, tensorrt::ElementWiseOperation::kSUB);
    TypedValue<RankedTensorType> endShape = tensorrt::scatterShapeTensor(
        rewriter, loc, resultType.getShape(), *concatAxis, finalPartDimSize);

    // In this case, we know the shape will be same as result shape except the
    // concat dim will be unknown.
    SmallVector<int64_t> sliceShape(resultType.getShape());
    sliceShape[*concatAxis] = ShapedType::kDynamic;
    parts.push_back(rewriter.create<tensorrt::SliceOp>(
        loc, resultType.clone(sliceShape), adaptor.getOperand(),
        /*start=*/endOffset,
        /*size=*/endShape,
        /*stride=*/
        rewriter.getDenseI32ArrayAttr(
            SmallVector<int32_t>(updateType.getRank(), 1))));

    rewriter.replaceOpWithNewOp<tensorrt::ConcatenationOp>(
        op, op.getType(), parts,
        /*axis=*/*concatAxis);
    return success();
  }
};

/// Creates `tensorrt.constant` from the Q/DQ scale in the form of ElementsAttr.
/// Q/DQ scale is saved in the form of ElementsAttr as the value of `scale` key
/// in `stablehlo.composite` op `attr` dictionary by `stablehlo-raise-qdq` pass.
static FailureOr<Value> getScaleConstant(RewriterBase &rewriter, Location loc,
                                         DictionaryAttr attr,
                                         StringRef qdqMode) {
  if (qdqMode != "tensorrt.pt_q" && qdqMode != "tensorrt.pt_dq") {
    auto scaleAttr = dyn_cast<ElementsAttr>(attr.get("scale"));
    if (!scaleAttr)
      return failure();
    Value scaleConstant = rewriter.create<tensorrt::ConstantOp>(loc, scaleAttr);
    return scaleConstant;
  }
  // For `kPerTensorQuantize` and `kPerTensorDequantize` case, we need to
  // extract a single element from the splat elements attribute and create
  // tensor.
  auto scaleAttr = dyn_cast<ElementsAttr>(attr.get("scale"));
  if (!scaleAttr || !scaleAttr.isSplat())
    return failure();
  RankedTensorType scaleConstantType =
      RankedTensorType::get({}, scaleAttr.getElementType());
  return rewriter
      .create<tensorrt::ConstantOp>(
          loc, DenseElementsAttr::get(scaleConstantType,
                                      scaleAttr.getSplatValue<APFloat>()))
      .getResult();
}

template <typename OpTy>
static LogicalResult addQOrDQ(RewriterBase &rewriter, stablehlo::CompositeOp op,
                              DictionaryAttr &attr, StringRef qdqMode,
                              IntegerAttr axis) {
  FailureOr<Value> scaleConstant =
      getScaleConstant(rewriter, op->getLoc(), attr, qdqMode);
  if (failed(scaleConstant))
    return failure();
  rewriter.replaceOpWithNewOp<OpTy>(op, op->getResultTypes(),
                                    op->getOperands().front(), *scaleConstant,
                                    axis);
  return success();
}

/// Converts `stablehlo.composite` op to TensorRT Q or DQ, based on attribute
/// values. Name of the composite op tells about Q/DQ mode, in this case.
struct CompositeToQDQConverter
    : public ConvertHloOpToTensorRTPattern<stablehlo::CompositeOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::CompositeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DictionaryAttr attr = op.getCompositeAttributes();
    if (!attr.contains("axis") || !attr.contains("scale") ||
        !attr.contains("is_pointwise"))
      return failure();

    // Name of the composite op encodes TensorRT Q/DQ mode.
    StringRef qdqMode = op.getName();

    // Add Q or DQ based on Q/DQ mode
    if (qdqMode ==
        tensorrt::TensorRTDialect::kTensorRTPerTensorQuantizationMarker)
      return addQOrDQ<tensorrt::QuantizeOp>(rewriter, op, attr, qdqMode,
                                            IntegerAttr());
    if (qdqMode ==
        tensorrt::TensorRTDialect::kTensorRTPerChannelQuantizationMarker) {
      Attribute axisAttr = attr.get("axis");
      auto axis = cast<IntegerAttr>(axisAttr);
      return addQOrDQ<tensorrt::QuantizeOp>(rewriter, op, attr, qdqMode, axis);
    }
    if (qdqMode == tensorrt::TensorRTDialect::kTensorRTBlockQuantizationMarker)
      return addQOrDQ<tensorrt::QuantizeOp>(rewriter, op, attr, qdqMode,
                                            IntegerAttr());
    if (qdqMode ==
        tensorrt::TensorRTDialect::kTensorRTPerTensorDequantizationMarker)
      return addQOrDQ<tensorrt::DequantizeOp>(rewriter, op, attr, qdqMode,
                                              IntegerAttr());
    if (qdqMode ==
        tensorrt::TensorRTDialect::kTensorRTPerChannelDequantizationMarker) {
      Attribute axisAttr = attr.get("axis");
      auto axis = cast<IntegerAttr>(axisAttr);
      return addQOrDQ<tensorrt::DequantizeOp>(rewriter, op, attr, qdqMode,
                                              axis);
    }
    if (qdqMode ==
        tensorrt::TensorRTDialect::kTensorRTBlockDequantizationMarker)
      return addQOrDQ<tensorrt::DequantizeOp>(rewriter, op, attr, qdqMode,
                                              IntegerAttr());
    return failure();
  }
};

/// Populate tensorrt.softmax patterns.
static void populateTensorRTSoftmaxPatterns(RewritePatternSet &patterns) {
  patterns.insert<ConvertHLOSoftmax>(patterns.getContext());
}

/// Pass that converts Stablehlo to TensorRT dialect ops.
class ConvertStablehloToTensorRtPass
    : public mlir::impl::ConvertStablehloToTensorRTPassBase<
          ConvertStablehloToTensorRtPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    // Create the conversion target.
    MLIRContext *ctx = &getContext();

    // Patterns like softmax must be applied before other hlo-to-tensorrt
    // conversions run because matcher for softmax matches with base HLO ops.
    {
      RewritePatternSet patterns(ctx);
      populateTensorRTSoftmaxPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns))))
        return signalPassFailure();
    }

    // Apply the remaining op conversions from hlo to tensorrt
    {
      RewritePatternSet patterns(ctx);
      LowerToTensorRTOptions loweringOptions;
      if (allowI64ToI32Conversion)
        loweringOptions.setI64Lowering(
            LowerToTensorRTOptions::I64Lowering::CastI64ToI32);
      TensorRTTypeConverter typeConverter(ctx, loweringOptions);
      TensorRTConversionTarget target(*ctx, typeConverter);

      // Conversion of `stablehlo.composite` should not fail.
      target.addIllegalOp<stablehlo::CompositeOp>();

      // Private functions with `tensorrt.qdq` attributes are associated with
      // composite ops. Once composite op is successfully converted to TensorRT,
      // these functions are removed so we don't perform `to TensorRT`
      // conversion on operations nested inside it.
      target.markOpRecursivelyLegal<func::FuncOp>([&](func::FuncOp op) {
        return op.isPrivate() && op->hasAttr("plan.decomposition");
      });

      populateStablehloToTensorRtConversionPattern(typeConverter, patterns);
      populateStablehloControlFlowToTensorRtPatterns(
          typeConverter, patterns, convertLoops, convertConditionals);
      populateChloToTensorRtLegalityAndPatterns(typeConverter, target,
                                                patterns);
      mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateStablehloToTensorRtConversionPattern(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // Add larger patterns with a higher
  // benefit so that they run first.
  patterns.add<SortToTopK, ConvertReduceWindow>(
      typeConverter, patterns.getContext(), PatternBenefit(100));
  patterns.add<
      // Contraction Operations
      ConvertDot, ConvertDotGeneral, ConvertEinsum,
      // Shape related operations
      ReshapeConverter, ConvertBroadcastInDim, ConvertDynamicBroadcastInDim,
      ConvertBroadcast, ConvertIota, ConvertDynamicIota,
      DynamicReshapeConverter,
      // Cast-like operations
      ConvertConverter,
      // Other
      ConvertTranspose, ConvertSelect, ConvertConcatenate, Log1pConverter,
      ConvertAtan2, Expm1OpConverter, BatchNormInferenceOpConverter,
      DynamicUpdateSliceToConcatConverter, ConvertCumsum,

  // StableHLO Binary operations
#define MAKE_BINARY_OP_CONVERTER(x, y)                                         \
  HloBinaryOpConverter<stablehlo::x, tensorrt::ElementWiseOperation::y>

      MAKE_BINARY_OP_CONVERTER(AddOp, kSUM),
      MAKE_BINARY_OP_CONVERTER(SubtractOp, kSUB),
      MAKE_BINARY_OP_CONVERTER(MulOp, kPROD),
      MAKE_BINARY_OP_CONVERTER(DivOp, kDIV),
      MAKE_BINARY_OP_CONVERTER(OrOp, kOR),
      MAKE_BINARY_OP_CONVERTER(XorOp, kXOR),
      MAKE_BINARY_OP_CONVERTER(AndOp, kAND),
      MAKE_BINARY_OP_CONVERTER(MaxOp, kMAX),
      MAKE_BINARY_OP_CONVERTER(MinOp, kMIN),
      MAKE_BINARY_OP_CONVERTER(PowOp, kPOW),

#undef MAKE_BINARY_OP_CONVERTER

  // StableHLO unary operations
#define MAKE_UNARY_OP_CONVERTER(x, y)                                          \
  HloUnaryOpConverter<stablehlo::x, tensorrt::UnaryOperation::y>

      MAKE_UNARY_OP_CONVERTER(CeilOp, kCEIL),
      MAKE_UNARY_OP_CONVERTER(FloorOp, kFLOOR),
      MAKE_UNARY_OP_CONVERTER(AbsOp, kABS),
      MAKE_UNARY_OP_CONVERTER(ExpOp, kEXP),
      MAKE_UNARY_OP_CONVERTER(CosineOp, kCOS),
      MAKE_UNARY_OP_CONVERTER(SineOp, kSIN),
      MAKE_UNARY_OP_CONVERTER(SqrtOp, kSQRT),
      MAKE_UNARY_OP_CONVERTER(NegOp, kNEG),
      MAKE_UNARY_OP_CONVERTER(LogOp, kLOG),
      MAKE_UNARY_OP_CONVERTER(NotOp, kNOT),
      MAKE_UNARY_OP_CONVERTER(SignOp, kSIGN),
      MAKE_UNARY_OP_CONVERTER(RoundNearestEvenOp, kROUND),

#undef MAKE_UNARY_OP_CONVERTER
      ConvertRemainder, ConvertReduceOp, TorchIndexSelectConverter,
      ReverseConverter, PadConverter, DynamicPadConverter, CompareConverter,
      ClampConverter, GetDimensionSizeConverter, UniformQuantizeConverter,
      UniformDequantizeConverter,
      HloUnaryOpToActivationConverter<stablehlo::LogisticOp,
                                      tensorrt::ActivationType::kSIGMOID>,
      HloUnaryOpToActivationConverter<stablehlo::TanhOp,
                                      tensorrt::ActivationType::kTANH>,
      RsqrtConverter, HloConstantConverter<stablehlo::ConstantOp>,
      RealDynamicSliceConverter, HloSliceConverter<stablehlo::SliceOp>,
      DynamicSliceConverter, ConvolutionConverter, ConvertScatterToTensorRT,
      // clang-format off
      SingleDimSimpleGatherToTensorRTGatherPattern,
      ConvertScatterToTensorRTScatterElements,
      ConvertGatherToTensorRT,
      ConvertGatherToTensorRTGatherNd,
      ConvertGatherWithPartialSlicesToTensorRT,
      CompositeToQDQConverter
    >(typeConverter, patterns.getContext(), PatternBenefit(1));
  // clang-format on
}
