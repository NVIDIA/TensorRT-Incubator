//===- TensorRT.cpp  ------------------------------------------------------===//
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
/// Implementation of the TensorRT dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensorrt;

// Set the max size of tensors which can be constant-folded to 131072 (0.5 MB
// for f32 constants).
constexpr int64_t kFoldOpEltLimit = 1 << 17;

//===----------------------------------------------------------------------===//
// Custom Assembly Format Directives
//===----------------------------------------------------------------------===//

/// Parse an optional integer. We add this helper because the plain API is
/// counter-intuitive.
template <typename T>
static FailureOr<std::optional<T>>
parseOptionalIntegerWrapper(AsmParser &parser) {
  std::optional<T> value;
  value.emplace();
  OptionalParseResult result = parser.parseOptionalInteger<T>(*value);
  // Having a value means we have an integer parse result: there was an integer,
  // but we may have failed to parse it.
  if (result.has_value()) {
    if (succeeded(*result))
      return value;
    return failure();
  }
  // Otherwise, there was no integer.
  value.reset();
  return value;
}

template <typename T>
static FailureOr<SmallVector<T>>
parseCommaSeparatedIntegers(AsmParser &parser) {
  SmallVector<T> integers;
  if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
        FailureOr<std::optional<T>> element =
            parseOptionalIntegerWrapper<T>(parser);
        if (failed(element))
          return failure();
        // Allow for parsing integer to fail on the first integer (empty
        // list).
        if (!element->has_value())
          return success(integers.empty());
        integers.push_back(**element);
        return success();
      })))
    return failure();
  return integers;
}

/// Parsae a DenseI32ArrayAttr as a list of integers rather than the default
/// form.
static ParseResult parseStaticIndexI32Array(OpAsmParser &parser,
                                            DenseI32ArrayAttr &staticValues) {
  FailureOr<SmallVector<int32_t>> integers =
      parseCommaSeparatedIntegers<int32_t>(parser);
  if (failed(integers))
    return failure();
  staticValues = DenseI32ArrayAttr::get(parser.getContext(), *integers);
  return success();
}

/// Print a DenseI32ArrayAttr as a list of integers rather than the default
/// form.
static void printStaticIndexI32Array(OpAsmPrinter &printer, Operation *op,
                                     DenseI32ArrayAttr staticValues) {

  llvm::interleaveComma(staticValues.asArrayRef(), printer);
}

/// Parsae a DenseI64ArrayAttr as a list of integers rather than the default
/// form.
static ParseResult parseStaticIndexI64Array(OpAsmParser &parser,
                                            DenseI64ArrayAttr &staticValues) {
  FailureOr<SmallVector<int64_t>> integers =
      parseCommaSeparatedIntegers<int64_t>(parser);
  if (failed(integers))
    return failure();
  staticValues = DenseI64ArrayAttr::get(parser.getContext(), *integers);
  return success();
}

/// Print a DenseI64ArrayAttr as a list of integers rather than the default
/// form.
static void printStaticIndexI64Array(OpAsmPrinter &printer, Operation *op,
                                     DenseI64ArrayAttr staticValues) {
  llvm::interleaveComma(staticValues.asArrayRef(), printer);
}

//===----------------------------------------------------------------------===//
// TensorRTModuleOp
//===----------------------------------------------------------------------===//

void TensorRTModuleOp::build(OpBuilder &builder, OperationState &state,
                             StringRef name) {
  state.addRegion()->emplaceBlock();
  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

TensorRTModuleOp TensorRTModuleOp::create(Location loc, StringRef name) {
  OpBuilder b(loc.getContext());
  return b.create<TensorRTModuleOp>(loc, name);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

func::FuncOp CallOp::getFuncCallee(SymbolTableCollection &symbolTable) {
  Operation *module = (*this)->getParentWithTrait<OpTrait::SymbolTable>();
  assert(module && "expected call to be nested within symbol table");
  return dyn_cast_or_null<func::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(module, getCallee()));
}

/// Returns true if the arrays of types are equivalent to up unknown dims.
static bool areTensorTypesCompatible(TypeRange lhs, ArrayRef<Type> rhs) {
  for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
    if (l == r)
      return true;

    auto lType = dyn_cast<ShapedType>(l);
    auto rType = dyn_cast<ShapedType>(r);
    if (!lType || !rType)
      return false;
    if (!areShapesEquivalentUpToDynamicDims(lType, rType))
      return false;
  }
  return true;
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  func::FuncOp kernel = getFuncCallee(symbolTable);
  if (!kernel)
    return emitOpError() << "no valid kernel found with symbol name "
                         << getCallee();
  FunctionType funcType = kernel.getFunctionType();

  // TODO: For dynamic shapes, we allow passing a larger linear bufer.
  // Even if we keep this, we should still validate element type. We can
  // remove this change if we do the "slice+reshape" on DPS arg instead of on
  // cluster results.
  if (funcType.getNumInputs() != getInputs().size() ||
      funcType.getNumResults() != getResultTypes().size() ||
      !areTensorTypesCompatible(TypeRange(getInputs()), funcType.getInputs()))
    return emitOpError()
           << "callee has function type " << funcType
           << " which is not compatible with input/result types of call";

  return success();
}

//===----------------------------------------------------------------------===//
// CallAllocOp
//===----------------------------------------------------------------------===//

func::FuncOp CallAllocOp::getFuncCallee(SymbolTableCollection &symbolTable) {
  Operation *module = (*this)->getParentWithTrait<OpTrait::SymbolTable>();
  assert(module && "expected call to be nested within symbol table");
  return dyn_cast_or_null<func::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(module, getCallee()));
}

LogicalResult
CallAllocOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  func::FuncOp kernel = getFuncCallee(symbolTable);
  if (!kernel)
    return emitOpError() << "no valid kernel found with symbol name "
                         << getCallee();
  FunctionType funcType = kernel.getFunctionType();

  if (funcType.getNumInputs() != getInputs().size() ||
      funcType.getNumResults() != getResultTypes().size() ||
      !areTensorTypesCompatible(TypeRange(getInputs()), funcType.getInputs()))
    return emitOpError()
           << "callee has function type " << funcType
           << " which is not compatible with input/result types of call";

  return success();
}

//===----------------------------------------------------------------------===//
// ElementwiseOp
//===----------------------------------------------------------------------===//

namespace {
struct MaxToReluRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kMAX)
      return failure();
    Value newInput{};
    if (mlir::matchPattern(op.getInput1(), mlir::m_AnyZeroFloat()))
      newInput = op.getInput2();
    else if (mlir::matchPattern(op.getInput2(), mlir::m_AnyZeroFloat()))
      newInput = op.getInput1();
    else
      return failure();
    if (newInput.getType() != op.getType())
      return failure();
    rewriter.replaceOpWithNewOp<ActivationOp>(
        op, newInput, ActivationType::kRELU, FloatAttr(), FloatAttr());
    return success();
  }
};
/// Rewrite the following so that the elementwise operation is a `kSUM`.
/// TensorRT won't recognize that `kSUB` is a "bias add".
/// ```
/// %0 = tensorrt.convolution .... : tensor<10x3x128x128>
/// %1 = tensorrt.elementwise <kSUB> ins(%0, %cst : tensor<10x3x128x128>,
/// tensor<1x3x1x1>>) -> ...
/// ```
/// The rewrite also occurs if the RHS is a constant.
struct SubToSumRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kSUB)
      return failure();
    // Canonicalize sub-to-sum if (a) producer is convolution and (b) RHS is a
    // constant.
    if (!isa_and_nonnull<ConvolutionOp>(op.getInput1().getDefiningOp()) ||
        !matchPattern(op.getInput2(), m_Constant()))
      return failure();
    Value negWeights = rewriter.create<UnaryOp>(op.getLoc(), op.getInput2(),
                                                UnaryOperation::kNEG);
    rewriter.replaceOpWithNewOp<ElementWiseOp>(op, op.getInput1(), negWeights,
                                               ElementWiseOperation::kSUM);
    return success();
  }
};

/// This rewrites `sum(x,neg(y))` if `y` is not a constant to `sub(x, y)`. This
/// pattern and the above won't ping-pong because `SubToSum` only applies if `y`
/// is constant and follows a convolution.
struct SumToSubRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kSUM)
      return failure();
    auto negOp = op.getInput2().getDefiningOp<UnaryOp>();
    if (!negOp || negOp.getUnaryOperation() != UnaryOperation::kNEG)
      return failure();
    if (matchPattern(negOp.getInput(), m_Constant()))
      return failure();
    rewriter.replaceOpWithNewOp<ElementWiseOp>(
        op, op.getInput1(), negOp.getInput(), ElementWiseOperation::kSUB);
    return success();
  }
};
} // namespace

/// Return true if `root` is the result of a convolution or the result of one or
/// more elementwise sub/prod/sum ops on the result of a convolution.
static bool isEwiseToConvChain(Value root) {
  Operation *producer = root.getDefiningOp();
  while (producer) {
    if (isa<ConvolutionOp>(producer))
      return true;
    auto ewiseOp = dyn_cast<ElementWiseOp>(producer);
    if (!ewiseOp)
      return false;
    // We basically want to restrict to thse operations that are channel-wise
    // bias/scaling factors.
    auto ewiseType = ewiseOp.getElementwiseOperation();
    if (ewiseType != ElementWiseOperation::kSUB &&
        ewiseType != ElementWiseOperation::kPROD &&
        ewiseType != ElementWiseOperation::kSUM)
      return false;
    producer = ewiseOp.getInput1().getDefiningOp();
  }
  return false;
}

namespace {
/// TensorRT's precision options are underspecified from a users perspective.
/// Setting the output of a 'f16' convolution does not let you specify the
/// accumulation value, for example. What we have learned is that in order for
/// TRT to generate a fused conv16 kernel (e.g. fused bias add), we must have
/// the convolution and all following elementwise ops (e.g. bias add) be in the
/// same precision (f16). However, often times programs that result from "mixed
/// precision training" in frameworks will result in IR that looks like the
/// below:
/// ```
/// %0_f16 = tensorrt.convolution
/// %1_f32 = tensorrt.identity %0_f16     (cast to f32)
/// %2_f32 = tensorrt.element_wise <kADD> (bias add)
/// ```
/// Therefore, to achieve good f16 performance, we need to rotate the add above
/// the cast. To be conservative, we only do transformation for this particular
/// pattern (conv,cast,sum/prod/sub) and when the RHS of the element-wise
/// operation is a constant. Even in this case, this is not a
/// semantics-preserving transformation, but TRT does not give us precise
/// control over the accumulation element type anyway.
///
/// TODO: here we require casting the constants to fp16 since our elementwise
/// operation all operands to have the same result type. However, a better
/// solution in the long run may be to allow elementwise operations to have
/// mixed element type operands and results. This will allow encoding f16 conv +
/// f32 bias with f32 result, which may be fused by TRT (unconfirmed here)
/// without requiring us to make a lossy cast.
struct EwiseMixedPrecisionRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    auto ewiseType = op.getElementwiseOperation();
    if (ewiseType != ElementWiseOperation::kSUB &&
        ewiseType != ElementWiseOperation::kPROD &&
        ewiseType != ElementWiseOperation::kSUM)
      return failure();
    auto constOp = op.getInput2().getDefiningOp<ConstantOp>();
    auto identity = op.getInput1().getDefiningOp<IdentityOp>();
    if (!identity || !constOp || !op.getType().getElementType().isF32() ||
        !identity.getInput().getType().getElementType().isF16())
      return failure();

    Operation *producer = identity.getInput().getDefiningOp();
    if (!producer || !isEwiseToConvChain(identity.getInput()))
      return failure();
    auto lowerPrecisionType =
        cast<RankedTensorType>(producer->getResult(0).getType());

    Value castedRhs = rewriter.create<IdentityOp>(
        op.getLoc(),
        op.getInput2().getType().clone(lowerPrecisionType.getElementType()),
        op.getInput2());

    OperationState state(
        op.getLoc(), op->getName(), {producer->getResult(0), castedRhs},
        op.getType().clone(lowerPrecisionType.getElementType()),
        op->getAttrs());
    Operation *newEwiseOp = rewriter.create(state);
    rewriter.replaceOpWithNewOp<IdentityOp>(op, op.getType(),
                                            newEwiseOp->getResult(0));
    return success();
  }
};
} // namespace

static std::optional<FloatAttr> getFloatSplatValue(Value v) {
  DenseFPElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  if (!attr.isSplat())
    return std::nullopt;
  return attr.getSplatValue<FloatAttr>();
}

static std::optional<IntegerAttr> getIntegerSplatValue(Value v) {
  DenseIntElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  if (!attr.isSplat())
    return std::nullopt;
  return attr.getSplatValue<IntegerAttr>();
}

/// Check whether `x / x == 1` in the precision of the given float value.
static bool isXDivXUnity(FloatAttr attr) {
  APFloat v = attr.getValue();
  bool losesInfo{false};
  APFloat unity(1.0);
  APFloat::opStatus status =
      unity.convert(v.getSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  if (status != APFloat::opStatus::opOK)
    return false;
  assert(!losesInfo && "1.0 should be exactly represented");
  return v / v == unity;
}

static bool isFloatSplatValueSame(Value lhs, Value rhs) {
  std::optional<FloatAttr> lhsAttr = getFloatSplatValue(lhs);
  if (!lhsAttr)
    return false;
  std::optional<FloatAttr> rhsAttr = getFloatSplatValue(rhs);
  if (!rhsAttr)
    return false;
  if (*lhsAttr != *rhsAttr || !isXDivXUnity(*lhsAttr) ||
      !isXDivXUnity(*rhsAttr))
    return false;
  return true;
}

static bool isIntegerSplatValueEqualOne(Value lhs, Value rhs) {
  std::optional<IntegerAttr> lhsAttr = getIntegerSplatValue(lhs);
  if (!lhsAttr)
    return false;
  std::optional<IntegerAttr> rhsAttr = getIntegerSplatValue(rhs);
  if (!rhsAttr)
    return false;
  return (*lhsAttr).getValue().isOne() && (*rhsAttr).getValue().isOne();
}

namespace {
/// Remove elementwise `prod` followed by `div` pair, which results in no
/// change (i.e. multiply and divide by the same number). Such
/// pair is added while converting `mhlo.reduce_window` to `tensorrt.pooling`.
/// We also check that splat value is not a nan/inf/zero(for division). The
/// check for inf/nan/zero will rule out the obvious cases where this is not
/// correct, but it won't rule out cases of overflow or effects of floating
/// point precision, which cannot be guarded against at compile time using the
/// strategy implemented here.
/// TODO: To make the transformation more conservative, we need to to simplify
/// div( mul(x, a), a) into mul(x, a / a)) then implement the relevant
/// constant-folders so that a/a can be folded to 1 (or another value). Then we
/// have a folder that simplifies mul(x, 1) into just x, which should always be
/// legal.
struct RemoveProdDivPair : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kDIV)
      return failure();

    auto producer = op.getInput1().getDefiningOp<ElementWiseOp>();
    if (!producer ||
        producer.getElementwiseOperation() != ElementWiseOperation::kPROD)
      return failure();
    // Check if splat value of multiplier and divisor is same (and its not
    // nan/inf/zero in case of float).
    if (!isFloatSplatValueSame(producer.getInput2(), op.getInput2()) &&
        !isIntegerSplatValueEqualOne(producer.getInput2(), op.getInput2()))
      return failure();

    // Replace all uses of `div` result with the input1 of `prod`.
    rewriter.replaceAllUsesWith(op.getResult(), producer.getInput1());
    return success();
  }
};

/// Match min(max(x, splat_const0), splat_const1) and rewrite to
/// tensorrt.activation<kCLIP>.
struct MaxMinToClipRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kMIN)
      return failure();
    Value lhs = op.getInput1();
    auto producer = lhs.getDefiningOp<ElementWiseOp>();
    if (!producer ||
        producer.getElementwiseOperation() != ElementWiseOperation::kMAX ||
        producer.getInput1().getType() != op.getType())
      return failure();

    std::optional<FloatAttr> minVal = getFloatSplatValue(op.getInput2());
    std::optional<FloatAttr> maxVal = getFloatSplatValue(producer.getInput2());

    // Check that the RHS of min and max are both splats. The min must be >=
    // max. This is important since TRT clip is defined as max(alpha, min(beta,
    // x)), but here we are pattern matching min(beta, max(alpha, x)).
    if (!maxVal || !minVal ||
        minVal->getValueAsDouble() < maxVal->getValueAsDouble())
      return failure();

    // Min and max value must be 32-bit floats. Convert to float is safe for
    // lower precision types (IEEE half, bfloat16).
    auto getAsSingle = [](FloatAttr attr) -> std::optional<APFloat> {
      APFloat tmp = attr.getValue();
      bool losesInfo = false;
      if (APFloat::opStatus::opOK != tmp.convert(APFloat::IEEEsingle(),
                                                 APFloat::rmNearestTiesToEven,
                                                 &losesInfo) ||
          losesInfo)
        return {};
      return tmp;
    };

    std::optional<APFloat> maxVal32 = getAsSingle(*maxVal);
    std::optional<APFloat> minVal32 = getAsSingle(*minVal);
    if (!maxVal32 || !minVal32)
      return failure();

    Type f32Type = rewriter.getF32Type();
    rewriter.replaceOpWithNewOp<ActivationOp>(
        op, producer.getInput1(), ActivationType::kCLIP,
        rewriter.getFloatAttr(f32Type, *maxVal32),
        rewriter.getFloatAttr(f32Type, *minVal32));
    return success();
  }
};

/// Rewrites a sequence of elementwise and reduction operations to
/// `tensorrt.softmax`.
struct SoftmaxRewriter : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElementwiseOperation() != ElementWiseOperation::kDIV)
      return failure();

    auto expOp = op.getInput1().getDefiningOp<UnaryOp>();
    if (!expOp || expOp.getUnaryOperation() != UnaryOperation::kEXP)
      return failure();

    // If data type is F16, reduction happens in F32 to avoid overflow.
    ReduceOp redExpOp = nullptr;
    bool isDtypeF16 = expOp.getResult().getType().getElementType().isF16();
    if (isDtypeF16) {
      auto idF32ToF16 = op.getInput2().getDefiningOp<IdentityOp>();
      if (!idF32ToF16)
        return failure();
      redExpOp = idF32ToF16.getInput().getDefiningOp<ReduceOp>();
      if (!redExpOp || redExpOp.getReduceOperation() != ReduceOperation::kSUM ||
          redExpOp.getReduceAxes().size() != 1 || !redExpOp.getKeepDimensions())
        return failure();
      auto idF16ToF32 = redExpOp.getInput().getDefiningOp<IdentityOp>();
      if (!idF16ToF32)
        return failure();
    } else {
      redExpOp = op.getInput2().getDefiningOp<ReduceOp>();
      if (!redExpOp || redExpOp.getReduceOperation() != ReduceOperation::kSUM ||
          redExpOp.getInput() != expOp.getResult() ||
          redExpOp.getReduceAxes().size() != 1 || !redExpOp.getKeepDimensions())
        return failure();
    }

    auto subMaxOp = expOp.getInput().getDefiningOp<ElementWiseOp>();
    if (subMaxOp.getElementwiseOperation() != ElementWiseOperation::kSUB)
      return failure();

    Value smInput = subMaxOp.getInput1();
    auto maxOp = subMaxOp.getInput2().getDefiningOp<ReduceOp>();
    if (!maxOp || maxOp.getReduceOperation() != ReduceOperation::kMAX ||
        maxOp.getInput() != smInput ||
        maxOp.getReduceAxes() != redExpOp.getReduceAxes() ||
        !maxOp.getKeepDimensions())
      return failure();

    int64_t smAxis = redExpOp.getReduceAxes().front();
    rewriter.replaceOpWithNewOp<SoftMaxOp>(op, smInput, smAxis);
    return success();
  }
};

/// Tries to match layer normalization op implemented as a set of lower level
/// TensorRT ops. layer norm is implemented as follows, y = ((x -
/// E[x])/(sqrt(VAR[x] + eps))) * scale + bias where, VAR(x) i.e. variance is
/// computed as E[x**2] - (E[x])**2 and E[x] is mean along a specific axes.
/// Normalization op ends with an elementwise add of bias. We do bottom-up
/// check, starting with bias add op and going all the way upto first
/// elementwise product to compute `x^^2` where `x` is an input. Each operation
/// and corresponding values are mentioned in comments.
struct matchAndRewriteNormalizationOp : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {

    auto isComingFromConstantOpOrArgument = [](Value v) {
      // Layer normalization scale and bias has shape of [1, 1, D2, D3, D4 ...
      // DN] where, normalization happens over [2, .. N] axis. Thus, scale and
      // bias should either come from a BlockArgument OR a `ConstantOp` OR a
      // `ExtendRankOp` used to add 1's in the beginning. We check these two
      // possibilities here.
      if (isa<BlockArgument>(v) || v.getDefiningOp<ConstantOp>())
        return true;
      // Output of `expand_rank`. Input to `expand_rank` can be either an
      // argument or a constant op result. If input is arg, defining op is NULL.
      auto expandRankOp = v.getDefiningOp<ExpandRankOp>();
      return expandRankOp &&
             (expandRankOp.getInput().getDefiningOp<ConstantOp>() ||
              isa<BlockArgument>(expandRankOp.getInput()));
    };

    // This elementwise op is considered to be the last step in normalization,
    // which is `bias` add.
    // op: v15 = v14 + bias.
    if (op.getElementwiseOperation() != ElementWiseOperation::kSUM)
      return failure();
    auto preNorm = op.getInput1().getDefiningOp<ElementWiseOp>();
    Value bias = op.getInput2();
    if (!isComingFromConstantOpOrArgument(bias) || !preNorm ||
        preNorm.getElementwiseOperation() != ElementWiseOperation::kPROD)
      return failure();
    // preNorm: v14 = v9 * v13
    auto subMeanFromInput = preNorm.getInput1().getDefiningOp<ElementWiseOp>();
    auto mulByScale = preNorm.getInput2().getDefiningOp<ElementWiseOp>();
    if (!mulByScale ||
        mulByScale.getElementwiseOperation() != ElementWiseOperation::kPROD ||
        !subMeanFromInput ||
        subMeanFromInput.getElementwiseOperation() !=
            ElementWiseOperation::kSUB)
      return failure();
    // mulByScale: = v12 * scale
    auto sqrtOfVar = mulByScale.getInput1().getDefiningOp<UnaryOp>();
    Value scale = mulByScale.getInput2();
    if (!sqrtOfVar || sqrtOfVar.getUnaryOperation() != UnaryOperation::kSQRT ||
        !isComingFromConstantOpOrArgument(scale))
      return failure();
    // Check if scale and bias has the same shape
    if (!cast<RankedTensorType>(scale.getType())
             .getShape()
             .equals(cast<RankedTensorType>(bias.getType()).getShape()))
      return failure();
    // sqrtOfVar: v12 = 1/sqrt(v11)
    auto recipOfVar = sqrtOfVar.getInput().getDefiningOp<UnaryOp>();
    if (!recipOfVar || recipOfVar.getUnaryOperation() != UnaryOperation::kRECIP)
      return failure();
    // recipOfVar: v11 = 1/v10
    auto addEpsToVar = recipOfVar.getInput().getDefiningOp<ElementWiseOp>();
    if (!addEpsToVar ||
        addEpsToVar.getElementwiseOperation() != ElementWiseOperation::kSUM)
      return failure();
    // addEpsToVar: v10 = v8 + eps
    auto expandVarNegElimination =
        addEpsToVar.getInput1().getDefiningOp<ExpandRankOp>();
    auto eps = addEpsToVar.getInput2().getDefiningOp<ConstantOp>();
    if (!eps || !expandVarNegElimination)
      return failure();
    // subMeanFromInput: v9 = x - v7 (x-E[x])
    auto expandMeanInput =
        subMeanFromInput.getInput2().getDefiningOp<ExpandRankOp>();
    if (!expandMeanInput)
      return failure();
    // expandVarNegElimination: v8 = expand_rank(v6)
    auto varNegElimination =
        expandVarNegElimination.getInput().getDefiningOp<ActivationOp>();
    if (!varNegElimination ||
        varNegElimination.getActivationType() != ActivationType::kRELU)
      return failure();
    // expandMeanInput: v7 = expand_rank(v2)
    // op2 is checked as an input to op4.
    // varNegElimination: v6 = ReLU(v5)
    auto varInitial =
        varNegElimination.getInput().getDefiningOp<ElementWiseOp>();
    if (!varInitial ||
        varInitial.getElementwiseOperation() != ElementWiseOperation::kSUB)
      return failure();
    // varInitial: v5 = v3 - v4 (E[x^2] - (E[x])^2)
    auto meanSquareInput = varInitial.getInput1().getDefiningOp<ReduceOp>();
    auto squareMeanInput =
        varInitial.getInput2().getDefiningOp<ElementWiseOp>();
    if (!meanSquareInput ||
        meanSquareInput.getReduceOperation() != ReduceOperation::kAVG ||
        !squareMeanInput ||
        squareMeanInput.getElementwiseOperation() !=
            ElementWiseOperation::kPROD ||
        squareMeanInput.getInput1() != squareMeanInput.getInput2())
      return failure();
    // squareMeanInput: v4 = v2^2 ((E[x])^2)
    auto meanInput = squareMeanInput.getInput1().getDefiningOp<ReduceOp>();
    if (!meanInput || meanInput.getReduceOperation() != ReduceOperation::kAVG ||
        meanSquareInput.getReduceAxesAttr() != meanInput.getReduceAxesAttr())
      return failure();
    // meanSquareInput: v3 = reduce(v1) (E[x^2])
    auto squareInput =
        meanSquareInput.getInput().getDefiningOp<ElementWiseOp>();
    if (!squareInput ||
        squareInput.getElementwiseOperation() != ElementWiseOperation::kPROD ||
        squareInput.getInput1() != squareInput.getInput2())
      return failure();
    // meanInput: v2 = reduce(x) (E[x])
    // squareInput: v1 = x^2
    // check if all uses of x come from the same SSA value.
    if (meanInput.getInput() != squareInput.getInput1() ||
        squareInput.getInput1() != subMeanFromInput.getInput1())
      return failure();
    // Normalization op supports only F16 or F32 input.
    RankedTensorType inputType =
        cast<RankedTensorType>(squareInput.getInput1().getType());
    if (!inputType.getElementType().isF16() &&
        !inputType.getElementType().isF32())
      return failure();
    std::optional<FloatAttr> epsFloatAttr = getFloatSplatValue(eps);
    if (!epsFloatAttr)
      return failure();
    auto normalizationOp = rewriter.create<NormalizationOp>(
        op->getLoc(),
        /*input=*/squareInput.getInput1(),
        /*scale=*/scale,
        /*bias=*/bias,
        /*axis=*/meanSquareInput.getReduceAxes(),
        /*eps=*/*epsFloatAttr,
        /*num_groups=*/nullptr);
    rewriter.replaceAllUsesWith(op.getResult(), normalizationOp.getResult());
    return success();
  }
};

/// Rewrites `tensorrt.element_wise <..>(%lhs,%constant)` where the constant
/// is a SplatElementsAttr and has non-trivial shape. In this case, we can often
/// simplify the splat constant to a trivial shape (all ones) if the
/// broadcasting rules allow.
struct RewriteEwiseSplatConstant : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    SplatElementsAttr attr{};
    int64_t constPos = 0;
    if (matchPattern(op.getInput2(), m_Constant(&attr)))
      constPos = 1;
    else if (matchPattern(op.getInput1(), m_Constant(&attr)))
      constPos = 0;
    else
      return failure();

    // Myelin appears to have a bug where scalar constants can be truncated to
    // finite values. That can break correctness of the program if a float
    // constant purposely encodes a non-finite value. Therefore, don't do this
    // transformation for such values.
    if (isa<FloatType>(attr.getElementType())) {
      APFloat val = attr.getSplatValue<APFloat>();
      if (!val.isFinite())
        return failure();
    }

    std::array<RankedTensorType, 2> inputTypes = {op.getInput1().getType(),
                                                  op.getInput2().getType()};
    RankedTensorType constType = inputTypes[constPos];
    RankedTensorType otherType = inputTypes[(constPos + 1) % 2];
    SmallVector<int64_t> constantShape(inputTypes[constPos].getShape());
    int64_t rank = op.getType().getRank();
    for (int64_t i = 0; i < rank; i++) {
      if (constType.getDimSize(i) > 1 &&
          constType.getDimSize(i) == otherType.getDimSize(i))
        constantShape[i] = 1;
    }
    if (constantShape == constType.getShape())
      return failure();
    auto constOp = rewriter.create<ConstantOp>(
        op.getOperand(constPos).getLoc(),
        DenseElementsAttr::get(constType.clone(constantShape),
                               attr.getSplatValue<Attribute>()));
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(constPos).assign(constOp); });
    return success();
  }
};
} // namespace

void ElementWiseOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<matchAndRewriteNormalizationOp, MaxMinToClipRewriter,
              EwiseMixedPrecisionRewriter, RewriteEwiseSplatConstant>(context);
  results.add<MaxToReluRewriter, SubToSumRewriter, SumToSubRewriter,
              SoftmaxRewriter, RemoveProdDivPair>(context);
}

OpFoldResult tensorrt::ElementWiseOp::fold(FoldAdaptor adaptor) {
  RankedTensorType resultType = getType();
  // We currently only support folding elementwise operations for common
  // simple calculations on shape tensors.
  // TODO: add support for broadcasting case
  if (resultType.getRank() != 1 || !resultType.getElementType().isInteger(32) ||
      getInput1().getType() != getInput2().getType() ||
      getInput1().getType() != resultType)
    return {};
  auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput1());
  auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput2());
  if (!lhs || !rhs)
    return {};

  // Don't do very large expensive folds here.
  if (!lhs.isSplat() && !rhs.isSplat() &&
      rhs.getNumElements() > kFoldOpEltLimit)
    return {};

  ElementWiseOperation mode = getElementwiseOperation();
  return constFoldBinaryOp<IntegerAttr, APInt, void>(
      {lhs, rhs}, getType(),
      [mode](const APInt &l, const APInt &r) -> std::optional<APInt> {
        if (mode == ElementWiseOperation::kSUM)
          return l + r;
        if (mode == ElementWiseOperation::kSUB)
          return l - r;
        return {};
      });
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

OpFoldResult tensorrt::UnaryOp::fold(FoldAdaptor adaptor) {
  Attribute input = adaptor.getInput();
  if (!input)
    return {};

  if (auto floatEls = dyn_cast<DenseFPElementsAttr>(input)) {
    // Don't do very large expensive folds here.
    if (!floatEls.isSplat() && floatEls.getNumElements() > kFoldOpEltLimit)
      return {};
    if (getUnaryOperation() == UnaryOperation::kNEG)
      return constFoldUnaryOp<FloatAttr, FloatAttr::ValueType, void>(
          adaptor.getOperands(), [](const APFloat &a) { return -a; });
  }
  if (auto intEls = dyn_cast<DenseIntElementsAttr>(input)) {
    // Don't do very large expensive folds here.
    if (!intEls.isSplat() && intEls.getNumElements() > kFoldOpEltLimit)
      return {};
    if (getUnaryOperation() == UnaryOperation::kNEG)
      return constFoldUnaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
          adaptor.getOperands(), [](const APInt &a) { return -a; });
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void tensorrt::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  llvm::SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "cst_" << getType().getElementType();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult tensorrt::ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() &&
         "tensorrt.constant should have no operands");
  return getWeights();
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//

OpFoldResult tensorrt::ConcatenationOp::fold(FoldAdaptor adaptor) {
  // Concatenation of a single item is a no-op.
  if (getInputs().size() == 1)
    return getInputs().front();

  // We can remove any tensor from the list to be concatenated if it has
  // 0-extent.
  auto hasNonZeroConcatExtent = [&](Value v) {
    auto rtt = cast<RankedTensorType>(v.getType());
    int64_t extent = rtt.getDimSize(getAxis());
    return extent != 0;
  };
  auto filteredInputs = llvm::to_vector(
      llvm::make_filter_range(getInputs(), hasNonZeroConcatExtent));
  if (filteredInputs.size() < getInputs().size()) {
    getInputsMutable().assign(filteredInputs);
    return getResult();
  }

  // Currently we only support full constant folding for 1D i32 tensors (e.g.
  // shape tensors).
  if (getType().getRank() != 1 || !getType().getElementType().isInteger(32))
    return {};

  SmallVector<int32_t> result;
  for (unsigned i = 0, e = getInputs().size(); i < e; i++) {
    auto attr =
        llvm::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInputs()[i]);
    if (!attr)
      return {};
    llvm::append_range(result, attr.getValues<int32_t>());
  }
  return DenseIntElementsAttr::get(getType(), result);
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

void tensorrt::ConvolutionOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, Type type, Value input,
    OpFoldResult kernel, OpFoldResult bias, ArrayRef<int64_t> stride,
    ArrayRef<int64_t> prePadding, ArrayRef<int64_t> postPadding,
    uint32_t numGroups, std::optional<ArrayRef<int64_t>> dilation) {

  Value biasValue = nullptr;
  ElementsAttr biasStatic = nullptr;
  if (isa<Value>(bias)) {
    biasValue = cast<Value>(bias);
  } else if (isa<Attribute>(bias)) {
    biasStatic = dyn_cast<ElementsAttr>(cast<Attribute>(bias));
  }
  Value kernelValue = nullptr;
  ElementsAttr kernelStatic = nullptr;
  if (isa<Value>(kernel)) {
    kernelValue = cast<Value>(kernel);
  } else if (isa<Attribute>(kernel)) {
    kernelStatic = dyn_cast<ElementsAttr>(cast<Attribute>(kernel));
  }
  ConvolutionOp::build(
      odsBuilder, odsState, type, input, kernelValue, biasValue, kernelStatic,
      biasStatic, stride, prePadding, postPadding, numGroups,
      dilation.has_value()
          ? DenseI64ArrayAttr::get(odsBuilder.getContext(), *dilation)
          : DenseI64ArrayAttr());
}

namespace {
/// If the convolution is given the `kernel` as an SSA value produced from a
/// `tensorrt.constant`, fold the constant into the `kernelStatic` attribute of
/// the convolution op.
struct ConvKernelToStaticKernelRewriter : OpRewritePattern<ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getKernel())
      return failure();

    ConstantOp constOp = op.getKernel().getDefiningOp<ConstantOp>();
    if (!constOp)
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      op.setKernelStaticAttr(constOp.getWeights());
      op.getKernelMutable().clear();
    });
    return success();
  }
};
} // namespace

void ConvolutionOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<ConvKernelToStaticKernelRewriter>(context);
}

//===----------------------------------------------------------------------===//
// DeconvolutionOp
//===----------------------------------------------------------------------===//

void tensorrt::DeconvolutionOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, Type type, Value input,
    OpFoldResult kernelWeights, OpFoldResult biasWeights,
    ArrayRef<int64_t> stride, ArrayRef<int64_t> prePadding,
    ArrayRef<int64_t> postPadding, uint32_t numGroups,
    std::optional<ArrayRef<int64_t>> dilation) {

  Value biasValue = nullptr;
  ElementsAttr biasStatic = nullptr;
  if (isa<Value>(biasWeights)) {
    biasValue = cast<Value>(biasWeights);
  } else if (isa<Attribute>(biasWeights)) {
    biasStatic = dyn_cast<ElementsAttr>(cast<Attribute>(biasWeights));
  }
  Value kernelValue = nullptr;
  ElementsAttr kernelStatic = nullptr;
  if (isa<Value>(kernelWeights)) {
    kernelValue = cast<Value>(kernelWeights);
  } else if (isa<Attribute>(kernelWeights)) {
    kernelStatic = dyn_cast<ElementsAttr>(cast<Attribute>(kernelWeights));
  }
  DeconvolutionOp::build(
      odsBuilder, odsState, type, input, kernelValue, biasValue, kernelStatic,
      biasStatic, stride, prePadding, postPadding, numGroups,
      dilation.has_value()
          ? DenseI64ArrayAttr::get(odsBuilder.getContext(), *dilation)
          : DenseI64ArrayAttr());
}

//===----------------------------------------------------------------------===//
// MatrixMultiplyOp
//===----------------------------------------------------------------------===//

// Returns an identity permutation with the last two dims swapped.
// For example, if rank is 4, (d0, d1, d2, d3) -> (d0, d1, d3, d2) is returned.
static AffineMap getIdentityPermWithLastTwoDimsSwapped(int64_t rank,
                                                       MLIRContext *ctx) {
  SmallVector<uint32_t> perm = llvm::to_vector(llvm::seq<uint32_t>(0, rank));
  std::swap(perm.back(), perm[perm.size() - 2]);
  AffineMap transposeAffineMap = AffineMap::getPermutationMap(perm, ctx);
  return transposeAffineMap;
}

/// Canonicalizer for matmul(transpose(x), transpose(y)).
/// Tries to keep `MatrixMultiply` op0 to `k<NONE>`. Tried to pull transpose in
/// if input to the RHS is coming from transpose op and op1 is `k<NONE>`.
namespace {
struct MatMulTransposeHandling : OpRewritePattern<MatrixMultiplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getOp0() != MatrixOperation::kTRANSPOSE &&
        op.getOp1() != MatrixOperation::kNONE)
      return failure();

    bool didChange{false};

    // Try to keep op0 (LHS input transpose) to `k<NONE>`.
    // If op0 is not `kNONE`, TensorRT won't convert such MatMul to Conv which
    // is not optimal.
    if (op.getOp0() == MatrixOperation::kTRANSPOSE) {
      RankedTensorType input0Type = op.getInput0().getType();
      if (input0Type.getRank() < 2)
        return failure();
      // Simply add a transpose
      AffineMap transposeAffineMap = getIdentityPermWithLastTwoDimsSwapped(
          input0Type.getRank(), op->getContext());
      TransposeOp newTranspose = rewriter.create<TransposeOp>(
          op->getLoc(), op.getInput0(), transposeAffineMap);
      rewriter.replaceUsesWithIf(
          op.getInput0(), newTranspose.getResult(),
          [&](OpOperand &user) { return user.getOwner() == op; });
      rewriter.modifyOpInPlace(op,
                               [&]() { op.setOp0(MatrixOperation::kNONE); });
      didChange = true;
    }

    // Try to pull in outside transpose into RHS transpose
    if (op.getOp1() == MatrixOperation::kNONE) {
      // If input is coming from transpose op, try to pull that transpose op in
      // matmul.
      auto transpose =
          dyn_cast_or_null<TransposeOp>(op.getInput1().getDefiningOp());
      if (!transpose || !transpose->hasOneUse())
        return success(didChange);
      RankedTensorType input1Type = op.getInput1().getType();
      if (input1Type.getRank() < 2)
        return success(didChange);
      // Check whether transpose is transposing only last two dims
      if (transpose.getPermutation() !=
          getIdentityPermWithLastTwoDimsSwapped(input1Type.getRank(),
                                                op->getContext()))
        return success(didChange);
      rewriter.replaceOpWithNewOp<MatrixMultiplyOp>(
          op, op.getInput0(), transpose.getInput(), op.getOp0(),
          MatrixOperation::kTRANSPOSE);
      didChange = true;
    }
    return success(didChange);
  }
};

/// It is observed that, TensorRT kernel fusions are more aggressive if subgraph
/// is in the form of `matmul(softmax(*), *)`, instead of `matmul(*,
/// softmax(*))`. Tensor operations for outer matmul are kNONE and kTRANSPOSE
/// respectively. This rewrite pattern swaps the matmul operands, if necessary.
struct SwapMatMulOperands : public OpRewritePattern<MatrixMultiplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOp0() != MatrixOperation::kNONE ||
        op.getOp1() != MatrixOperation::kTRANSPOSE)
      return failure();

    auto softmax = op.getInput1().getDefiningOp<SoftMaxOp>();
    // Asserting rank to 4 is okay since this is specifically geared towards MHA
    // pattern.
    if (!softmax || softmax.getResult().getType().getRank() != 4)
      return failure();

    // Swap matmul operands.
    auto updatedMatmul =
        rewriter.create<MatrixMultiplyOp>(op->getLoc(),
                                          /*input0=*/softmax.getResult(),
                                          /*input1=*/op.getInput0(),
                                          /*op0=*/op.getOp0(),
                                          /*op1=*/op.getOp1());

    // Result of updated matmul with swapped operands will have last two dims
    // swapped compared to the original matmul. We create a transpose to swap
    // these last two dims back.
    SmallVector<unsigned> perm = llvm::to_vector(
        llvm::seq<unsigned>(0, updatedMatmul.getResult().getType().getRank()));
    std::swap(perm[3], perm[2]);
    auto affineMap = AffineMap::getPermutationMap(perm, op->getContext());
    auto transposeOp =
        rewriter.create<TransposeOp>(op->getLoc(),
                                     /*input=*/updatedMatmul.getResult(),
                                     /*permutation=*/affineMap);
    rewriter.replaceOp(op, transposeOp.getResult());
    return success();
  }
};
} // namespace

void MatrixMultiplyOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<SwapMatMulOperands, MatMulTransposeHandling>(context);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

/// Given an OpFoldResult value that is assumed to be either a Value or a
/// DenseI32Arrayattr, return a pair containing each value. One one will be
/// valid, the other will be nullptr.
static std::pair<Value, DenseI32ArrayAttr>
decomposeSliceOpFoldResultParam(OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr))
    return std::make_pair(v, nullptr);
  return std::make_pair(Value(), cast<DenseI32ArrayAttr>(cast<Attribute>(ofr)));
}

void tensorrt::SliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              Value input, OpFoldResult start,
                              OpFoldResult size, OpFoldResult stride,
                              SliceMode sliceMode, Value fill) {
  auto [startVal, startArr] = decomposeSliceOpFoldResultParam(start);
  auto [sizeVal, sizeArr] = decomposeSliceOpFoldResultParam(size);
  auto [strideVal, strideArr] = decomposeSliceOpFoldResultParam(stride);
  SliceOp::build(odsBuilder, odsState, input, /*fill=*/fill,
                 /*start=*/startVal,
                 /*size=*/sizeVal,
                 /*stride=*/strideVal,
                 /*static_start=*/startArr,
                 /*static_size=*/sizeArr,
                 /*static_stride=*/strideArr,
                 /*mode=*/sliceMode);
}

void tensorrt::SliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              Type result, Value input, OpFoldResult start,
                              OpFoldResult size, OpFoldResult stride,
                              SliceMode sliceMode, Value fill) {
  auto [startVal, startArr] = decomposeSliceOpFoldResultParam(start);
  auto [sizeVal, sizeArr] = decomposeSliceOpFoldResultParam(size);
  auto [strideVal, strideArr] = decomposeSliceOpFoldResultParam(stride);
  SliceOp::build(odsBuilder, odsState, result, input, /*fill=*/fill,
                 /*start=*/startVal,
                 /*size=*/sizeVal,
                 /*stride=*/strideVal,
                 /*static_start=*/startArr,
                 /*static_size=*/sizeArr,
                 /*static_stride=*/strideArr,
                 /*mode=*/sliceMode);
}

void tensorrt::SliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              Value input, ArrayRef<int32_t> start,
                              ArrayRef<int32_t> size, ArrayRef<int32_t> stride,
                              SliceMode sliceMode, Value fill) {
  auto toArrayAttr = [&](ArrayRef<int32_t> arr) {
    return OpFoldResult(DenseI32ArrayAttr::get(odsBuilder.getContext(), arr));
  };
  SliceOp::build(odsBuilder, odsState, input, toArrayAttr(start),
                 toArrayAttr(size), toArrayAttr(stride), sliceMode, fill);
}

void tensorrt::SliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              Type result, Value input, ArrayRef<int32_t> start,
                              ArrayRef<int32_t> size, ArrayRef<int32_t> stride,
                              SliceMode sliceMode, Value fill) {
  auto toArrayAttr = [&](ArrayRef<int32_t> arr) {
    return OpFoldResult(DenseI32ArrayAttr::get(odsBuilder.getContext(), arr));
  };
  SliceOp::build(odsBuilder, odsState, result, input, toArrayAttr(start),
                 toArrayAttr(size), toArrayAttr(stride), sliceMode, fill);
}

void tensorrt::SliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              Value input, ArrayRef<int32_t> start, Value size,
                              ArrayRef<int32_t> stride, SliceMode sliceMode,
                              Value fill) {
  auto toArrayAttr = [&](ArrayRef<int32_t> arr) {
    return OpFoldResult(DenseI32ArrayAttr::get(odsBuilder.getContext(), arr));
  };
  SliceOp::build(odsBuilder, odsState, input, toArrayAttr(start), size,
                 toArrayAttr(stride), sliceMode, fill);
}

namespace {
/// Move size|start dynamic arguments to static attributes if possible.
struct SliceDynamicParameterToStaticPattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    // If the dynamic size parameter is foldable, fold to static parameter.
    DenseIntElementsAttr value;
    if (op.getSize() && matchPattern(op.getSize(), m_Constant(&value))) {
      rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
          op, op.getType(), op.getInput(), op.getStartAsOpFoldResult(),
          rewriter.getDenseI32ArrayAttr(
              llvm::to_vector(value.getValues<int32_t>())),
          op.getStrideAsOpFoldResult(), op.getMode(), op.getFill());
      return success();
    }

    // If the dynamic start parameter is foldable, fold to static parameter.
    if (op.getStart() && matchPattern(op.getStart(), m_Constant(&value))) {
      rewriter.replaceOpWithNewOp<tensorrt::SliceOp>(
          op, op.getType(), op.getInput(),
          rewriter.getDenseI32ArrayAttr(
              llvm::to_vector(value.getValues<int32_t>())),
          op.getSizeAsOpFoldResult(), op.getStrideAsOpFoldResult(),
          op.getMode(), op.getFill());
      return success();
    }

    return failure();
  }
};
} // namespace

void tensorrt::SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<SliceDynamicParameterToStaticPattern>(context);
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

void tensorrt::ShuffleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TensorType resultType, TensorValue input,
                                bool zeroIsPlaceholder) {
  SmallVector<int64_t> reshapeShape =
      llvm::map_to_vector(resultType.getShape(), [](int64_t dim) {
        return ShapedType::isDynamic(dim) ? -1 : dim;
      });
  ShuffleOp::build(
      odsBuilder, odsState, resultType,
      /*input=*/input,
      /*dynamic_reshape=*/Value(),
      /*first_transpose=*/
      llvm::to_vector(llvm::seq<int64_t>(0, input.getType().getRank())),
      /*reshape=*/
      DenseI64ArrayAttr::get(odsBuilder.getContext(), reshapeShape),
      /*second_transpose=*/
      llvm::to_vector(llvm::seq<int64_t>(0, resultType.getRank())),
      /*zero_is_placeholder=*/zeroIsPlaceholder);
}

void tensorrt::ShuffleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TensorType resultType, TensorValue input,
                                TensorValue reshape, bool zeroIsPlaceholder) {
  ShuffleOp::build(
      odsBuilder, odsState, resultType,
      /*input=*/input,
      /*dynamic_reshape=*/reshape,
      /*first_transpose=*/
      llvm::to_vector(llvm::seq<int64_t>(0, input.getType().getRank())),
      /*reshape=*/
      DenseI64ArrayAttr(),
      /*second_transpose=*/
      llvm::to_vector(llvm::seq<int64_t>(0, resultType.getRank())),
      /*zero_is_placeholder=*/zeroIsPlaceholder);
}

AffineMap ShuffleOp::getFirstTransposeMap() {
  return getAsPermutationMap(getContext(), getFirstTranspose());
}

AffineMap ShuffleOp::getSecondTransposeMap() {
  return getAsPermutationMap(getContext(), getSecondTranspose());
}

RankedTensorType ShuffleOp::getIntermediateType() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  if (inputType.getRank() == 0)
    return cast<RankedTensorType>(getInput().getType());
  AffineMap firstTranspose = getFirstTransposeMap();
  return RankedTensorType::Builder(inputType).setShape(
      applyPermutationMap(firstTranspose, inputType.getShape()));
}

namespace {
/// Pattern to simplify a dynamic reshape value into a static one when the
/// dynamic reshape value is a constant. Note that the result type could still
/// be dynamic if `zero_is_placeholder` is true (see
/// `SimplifyZeroIsPlaceholderShuffle` below).
struct SimplifyDynamicShuffleReshapeToStaticReshape
    : public OpRewritePattern<ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the op has a dynamic reshape, and if so, whether the shape is a
    // constant.
    if (!op.getDynamicReshape())
      return rewriter.notifyMatchFailure(op, "not a dynamic reshape");

    DenseIntElementsAttr shapeAttr;
    if (!matchPattern(op.getDynamicReshape(), m_Constant(&shapeAttr)))
      return rewriter.notifyMatchFailure(op, "reshape value is not a constant");

    SmallVector<int64_t> shape;
    for (const APInt &dim : shapeAttr.getValues<APInt>())
      shape.push_back(static_cast<int64_t>(dim.getSExtValue()));

    rewriter.replaceOpWithNewOp<ShuffleOp>(
        op, op.getType(), /*input=*/op.getInput(),
        /*dynamic_reshape=*/Value(),
        /*first_transpose=*/op.getFirstTransposeAttr(),
        /*reshape=*/rewriter.getDenseI64ArrayAttr(shape),
        /*second_transpose=*/op.getSecondTransposeAttr(),
        /*zero_is_placeholder=*/op.getZeroIsPlaceholderAttr());
    return success();
  }
};

/// Pattern to simplify `tensorrt.shuffle` where `zero_is_placeholder` is set
/// but the result shape is static.
struct SimplifyZeroIsPlaceholderShuffle : public OpRewritePattern<ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getZeroIsPlaceholder())
      return rewriter.notifyMatchFailure(op, "zero is not placeholder");

    std::optional<ArrayRef<int64_t>> reshape = !op.getReshape();
    TensorType resultType = op.getType();
    // Static reshape value does not mean that the result type is static. The
    // input could have unknown dims, which are referenced by the 0's in the
    // reshape value.
    if (!reshape || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "not a static reshape");

    // Just update the attribute in place.
    rewriter.modifyOpInPlace(op, [&]() {
      op.setZeroIsPlaceholder(false);
      op.setReshapeAttr(rewriter.getDenseI64ArrayAttr(resultType.getShape()));
    });
    return success();
  }
};
} // namespace

/// Returns true if the transpose permutations (which must of equal rank) are
/// equal to the identity when composed or are both empty.
static bool isCompositionOfPermutationsIdentity(AffineMap t0, AffineMap t1) {
  if (t0.getNumResults() != t1.getNumResults() ||
      t0.getNumDims() != t1.getNumDims())
    return false;
  // If they are empty, it is for scalars and true.
  if (t0.getNumDims() == 0)
    return true;
  return t0.compose(t1).isIdentity();
}

namespace {
// Rewrite pattern to simplify sequential shuffle ops.
// Sequential shuffles
// "(t0 r0 t1) -> (t2 r1 t3)"
// can be simplified if (t1 compose t2) is the identity map. In that case we can
// replace them with "(t0 r1 t3)".
struct SimplifySequentialShuffle : public OpRewritePattern<ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<ShuffleOp>();
    if (!producer)
      return rewriter.notifyMatchFailure(op, "input producer is not a shuffle");
    if (!isCompositionOfPermutationsIdentity(producer.getSecondTransposeMap(),
                                             op.getFirstTransposeMap()))
      return rewriter.notifyMatchFailure(
          op, "inner transpose ops do not simplify to identity");
    rewriter.replaceOpWithNewOp<ShuffleOp>(
        op, op.getType(), producer.getInput(),
        /*dynamic_reshape=*/op.getDynamicReshape(),
        /*first_transpose=*/producer.getFirstTranspose(),
        /*reshape=*/op.getReshapeAttr(),
        /*second_transpose=*/op.getSecondTranspose(),
        /*zero_is_placeholder=*/op.getZeroIsPlaceholder());
    return success();
  }
};

/// Sequential shuffles "(t0 r0 t1) -> (t2 t3)" can be simplified into
/// "t0 r0 (t1 compose t2 compose t3)" if second transpose doesn't do reshape.
struct SimplifySequentialShuffleCase2 : public OpRewritePattern<ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<ShuffleOp>();
    if (!producer)
      return failure();
    if (op.getReshape().has_value() || op.getDynamicReshape())
      return failure();
    rewriter.replaceOpWithNewOp<ShuffleOp>(
        op, op.getType(), producer.getInput(),
        /*dynamic_reshape=*/producer.getDynamicReshape(),
        /*first_transpose=*/producer.getFirstTranspose(),
        /*reshape=*/producer.getReshapeAttr(),
        /*second_transpose=*/
        op.getSecondTransposeMap()
            .compose(op.getFirstTransposeMap())
            .compose(producer.getSecondTranspose()),
        /*zero_is_placeholder=*/producer.getZeroIsPlaceholder());
    return success();
  }
};
} // namespace

OpFoldResult ShuffleOp::fold(FoldAdaptor adaptor) {
  MLIRContext *ctx = getContext();
  auto isIdentityMap = [&](ArrayRef<int64_t> x) {
    return x.empty() || getAsPermutationMap(ctx, x).isIdentity();
  };
  // Don't fold anything for dynamic reshape or we are in "zero is placeholder"
  // mode. To be conservative, don't fold if any dims are unknown.
  std::optional<ArrayRef<int64_t>> reshapeShape = getReshape();
  TensorType resultType = getType();
  TensorType inputType = getInput().getType();
  if (!reshapeShape || getZeroIsPlaceholder() || !resultType.hasStaticShape() ||
      !inputType.hasStaticShape())
    return {};

  // The op can be folded if it is the identity. This can occur in the following
  // cases when the inputType is the same as the result type:
  if (inputType != resultType)
    return {};
  // - both transposes are the identity and the reshape is the same as the
  //   input/result shapes.
  if (reshapeShape == inputType.getShape() &&
      isIdentityMap(getFirstTranspose()) && isIdentityMap(getSecondTranspose()))
    return getInput();
  // - or the reshape is an identity operation (equal to shape after first
  // transpose) and the first and second transposes are inverse.
  if (isCompositionOfPermutationsIdentity(getFirstTransposeMap(),
                                          getSecondTransposeMap()) &&
      getIntermediateType().getShape() == *reshapeShape)
    return getInput();

  // If all except one dims are unity in both input and result AND
  // if input and result types are same, fold irrespective of permutation.
  auto areAllExceptOneDimsUnity = [](ArrayRef<int64_t> x) {
    return !x.empty() &&
           llvm::count(x, 1) == static_cast<int64_t>(x.size()) - 1;
  };

  if (inputType == resultType &&
      areAllExceptOneDimsUnity(inputType.getShape()) &&
      areAllExceptOneDimsUnity(resultType.getShape()))
    return getInput();

  return nullptr;
}

void ShuffleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<SimplifyDynamicShuffleReshapeToStaticReshape,
              SimplifyZeroIsPlaceholderShuffle, SimplifySequentialShuffle,
              SimplifySequentialShuffleCase2>(context);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  if (getPermutation().isIdentity())
    return getInput();
  return {};
}

namespace {
/// Combine sequantial transpose operations.
struct CombineTransposePattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<TransposeOp>();
    if (!producer)
      return failure();
    auto newMap = op.getPermutation().compose(producer.getPermutation());
    rewriter.replaceOpWithNewOp<tensorrt::TransposeOp>(op, producer.getInput(),
                                                       newMap);
    return success();
  }
};
} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<CombineTransposePattern>(context);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void tensorrt::YieldOp::build(OpBuilder &builder, OperationState &state) {
  return YieldOp::build(builder, state, ValueRange{});
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

void tensorrt::TopKOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             Value input, int64_t k, int64_t axis,
                             tensorrt::TopKOperation topKOperation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto rttBuilder = RankedTensorType::Builder(inputType);
  SmallVector<int64_t> shape(inputType.getShape());
  shape[axis] = k;
  rttBuilder.setShape(shape);
  RankedTensorType valuesType = rttBuilder;
  rttBuilder.setElementType(odsBuilder.getI32Type());
  RankedTensorType indicesType = rttBuilder;
  TopKOp::build(odsBuilder, odsState, TypeRange{valuesType, indicesType}, input,
                k, axis, topKOperation);
}

//===----------------------------------------------------------------------===//
// CollapseRankOp
//===----------------------------------------------------------------------===//

namespace {
/// Fold reshape-like operations into `tensorrt.constant`.
template <typename OpType>
struct ConstFoldReshapePattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().hasStaticShape())
      return failure();
    auto producer = op->getOperand(0).template getDefiningOp<ConstantOp>();
    if (!producer)
      return failure();

    // If the weights were elided, we can still notionally do this
    // transformation.
    if (std::optional<DenseResourceElementsHandle> elidedHandle =
            getElidedResourceElementsAttr(producer.getWeights())) {
      rewriter.replaceOpWithNewOp<tensorrt::ConstantOp>(
          op, DenseResourceElementsAttr::get(op.getType(), *elidedHandle));
      return success();
    }

    if (auto constAttr =
            dyn_cast<DenseElementsAttr>(producer.getWeightsAttr())) {
      rewriter.replaceOpWithNewOp<tensorrt::ConstantOp>(
          op, constAttr.reshape(op.getType()));
      return success();
    }
    return failure();
  }
};

/// Pattern to collapse producer/consumer reshape ops that are both collapsing
/// dimensions or are both expanding dimensions.
template <typename ReshapeOpTy>
struct ComposeTensorRTReassociativeReshapes
    : public OpRewritePattern<ReshapeOpTy> {
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcReshapeOp =
        reshapeOp.getSrc().template getDefiningOp<ReshapeOpTy>();
    if (!srcReshapeOp)
      return failure();
    RankedTensorType resultType = reshapeOp.getType();
    std::optional<SmallVector<ReassociationIndices>> reassociationIndices =
        composeReassociationIndices(srcReshapeOp.getReassociationIndices(),
                                    reshapeOp.getReassociationIndices(),
                                    rewriter.getContext());
    if (!reassociationIndices)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOpTy>(
        reshapeOp, resultType, srcReshapeOp.getSrc(), *reassociationIndices);
    return success();
  }
};

struct TensorRTComposeCollapseOfExpandOp
    : public OpRewritePattern<CollapseRankOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CollapseRankOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = collapseOp.getSrc().getDefiningOp<ExpandRankOp>();
    if (!expandOp)
      return failure();

    RankedTensorType srcType = expandOp.getInput().getType();
    RankedTensorType resultType = collapseOp.getType();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    SmallVector<ReassociationIndices, 4> higherRankReassociation,
        lowerRankReassociation;

    if (srcRank > resultRank) {
      higherRankReassociation = expandOp.getReassociationIndices();
      lowerRankReassociation = collapseOp.getReassociationIndices();
    } else {
      higherRankReassociation = collapseOp.getReassociationIndices();
      lowerRankReassociation = expandOp.getReassociationIndices();
    }

    size_t higherRankIndicesID = 0;
    SmallVector<ReassociationIndices, 4> composedReassociation;
    for (const auto &lowerRankIndices : lowerRankReassociation) {
      ReassociationIndices composedIndices;
      while (higherRankIndicesID < higherRankReassociation.size()) {
        auto rightmostIndex =
            higherRankReassociation[higherRankIndicesID].back();
        if (rightmostIndex > lowerRankIndices.back())
          return failure();
        composedIndices.push_back(higherRankIndicesID++);
        if (rightmostIndex == lowerRankIndices.back())
          break;
      }
      composedReassociation.push_back(composedIndices);
    }
    if (srcRank > resultRank) {
      rewriter.replaceOpWithNewOp<CollapseRankOp>(
          collapseOp, resultType, expandOp.getSrc(), composedReassociation);
      return success();
    }
    if (srcRank < resultRank) {
      rewriter.replaceOpWithNewOp<ExpandRankOp>(
          collapseOp, resultType, expandOp.getSrc(), composedReassociation);
      return success();
    }
    return failure();
  }
};

} // namespace

SmallVector<int64_t>
tensorrt::CollapseRankOp::getInputShapeDimIndicesOfRemovedDims() {
  // Collapse rank can only remove unit dimensions. Find the reassociation.
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(getType(), getInput().getType());
  assert(reassociation.has_value() &&
         "expected to be able to deduce reassociation");

  TypedValue<RankedTensorType> input = getInput();

  SmallVector<int64_t> result;
  for (ReassociationIndices &indices : *reassociation) {
    // bool wasBroadcastAdded = llvm::find(bcastOp.getBroadcastDims(),
    if (indices.size() < 2)
      continue;

    // This segment is a collapse. Find which dims are unit dims (and thus
    // removed)
    unsigned numAddedThisGroup = 0;
    for (int64_t dimInGroup : indices) {
      // Make sure to protect against saying all dims in the group are dropped
      // when the entire segment is 1's
      if (input.getType().getDimSize(dimInGroup) == 1 &&
          numAddedThisGroup < reassociation->size() - 1) {
        result.push_back(dimInGroup);
        numAddedThisGroup++;
      }
    }
  }
  return result;
}

void tensorrt::CollapseRankOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, Type result, Value input,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  (void)reassociationIndices;
  CollapseRankOp::build(odsBuilder, odsState, result, input);
}

SmallVector<ReassociationIndices>
tensorrt::CollapseRankOp::getReassociationIndices() {
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(getInput().getType(),
                                        /*  */ getResult().getType());
  assert(reassociation.has_value());
  return *reassociation;
}

void tensorrt::CollapseRankOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ComposeTensorRTReassociativeReshapes<CollapseRankOp>,
                 ConstFoldReshapePattern<CollapseRankOp>,
                 TensorRTComposeCollapseOfExpandOp>(context);
}

OpFoldResult tensorrt::CollapseRankOp::fold(FoldAdaptor adaptor) {
  return foldReshapeOp<CollapseRankOp, ExpandRankOp>(*this,
                                                     adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// ExpandRankOp
//===----------------------------------------------------------------------===//

void tensorrt::ExpandRankOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, Type result, Value input,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  (void)reassociationIndices;
  ExpandRankOp::build(odsBuilder, odsState, result, input);
}

SmallVector<ReassociationIndices>
tensorrt::ExpandRankOp::getReassociationIndices() {
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(getInput().getType(),
                                        /*  */ getResult().getType());
  assert(reassociation.has_value());
  return *reassociation;
}

struct TensorRTComposeExpandOfCollapseOp
    : public OpRewritePattern<ExpandRankOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandRankOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp = expandOp.getSrc().getDefiningOp<CollapseRankOp>();
    if (!collapseOp)
      return failure();

    RankedTensorType srcType = collapseOp.getInput().getType();
    RankedTensorType resultType = expandOp.getType();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    auto srcReassociation = collapseOp.getReassociationIndices();
    auto resultReassociation = expandOp.getReassociationIndices();
    if (srcRank > resultRank) {
      auto composedReassociation = getReassociationIndicesForCollapse(
          srcType.getShape(), resultType.getShape());
      if (!composedReassociation)
        return failure();
      rewriter.replaceOpWithNewOp<CollapseRankOp>(
          expandOp, resultType, collapseOp.getSrc(), *composedReassociation);
      return success();
    }

    auto composedReassociation = getReassociationIndicesForCollapse(
        resultType.getShape(), srcType.getShape());
    if (!composedReassociation)
      return failure();
    rewriter.replaceOpWithNewOp<ExpandRankOp>(
        expandOp, resultType, collapseOp.getSrc(), *composedReassociation);
    return success();
  }
};

void tensorrt::ExpandRankOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ComposeTensorRTReassociativeReshapes<ExpandRankOp>,
                 TensorRTComposeExpandOfCollapseOp,
                 ConstFoldReshapePattern<ExpandRankOp>>(context);
}

OpFoldResult tensorrt::ExpandRankOp::fold(FoldAdaptor adaptor) {
  if (!llvm::isa_and_nonnull<ElementsAttr>(adaptor.getInput()))
    return foldReshapeOp<ExpandRankOp, CollapseRankOp>(*this,
                                                       adaptor.getOperands());
  if (auto elAttr =
          llvm::dyn_cast_or_null<DenseIntOrFPElementsAttr>(adaptor.getInput()))
    return elAttr.reshape(getType());
  auto attr = cast<ElementsAttr>(adaptor.getInput());
  if (std::optional<DenseResourceElementsHandle> handle =
          getElidedResourceElementsAttr(attr))
    return DenseResourceElementsAttr::get(
        attr.getShapedType().clone(getType().getShape()), *handle);
  return {};
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace {
/// Replace sequential reshape operations with a single reshape.
struct SimplifyReshapeReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeDefOp = op.getInput().getDefiningOp<ReshapeOp>();
    if (!reshapeDefOp)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, op.getType(), reshapeDefOp.getInput(), op.getShape());
    return success();
  }
};

/// Canonicalize `tensorrt.reshape` into
/// `tensorrt.expand_rank`/`tensorrt.collapse_rank` if possible.
struct SimplifyReshapeToRankExpandCollapse
    : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Not valid for dynamic reshapes.
    if (op.getShape())
      return failure();
    if (succeeded(tensorrt::isUnitDimRankExpanding(op.getInput().getType(),
                                                   op.getType()))) {
      rewriter.replaceOpWithNewOp<tensorrt::ExpandRankOp>(op, op.getType(),
                                                          op.getInput());
      return success();
    }
    if (succeeded(tensorrt::isUnitDimRankReducing(op.getInput().getType(),
                                                  op.getType()))) {
      rewriter.replaceOpWithNewOp<tensorrt::CollapseRankOp>(op, op.getType(),
                                                            op.getInput());
      return success();
    }
    return failure();
  }
};
} // namespace

void tensorrt::ReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConstFoldReshapePattern<ReshapeOp>, SimplifyReshapeReshape,
                 SimplifyReshapeToRankExpandCollapse>(context);
}

void tensorrt::ReshapeOp::build(OpBuilder &builder, OperationState &state,
                                Type result, Value input) {
  ReshapeOp::build(builder, state, result, input, Value());
}

OpFoldResult tensorrt::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (getType().hasStaticShape() && getType() == getInput().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

AffineMap tensorrt::BroadcastOp::getBroadcastDimsPermutation() {
  ArrayRef<int64_t> broadcastDims = getBroadcastDims();
  if (broadcastDims.empty())
    return AffineMap::get(getContext());
  auto indices = llvm::to_vector(llvm::seq<unsigned>(0, broadcastDims.size()));
  llvm::sort(indices, [&](unsigned i1, unsigned i2) {
    return broadcastDims[i1] < broadcastDims[i2];
  });
  return AffineMap::getPermutationMap(indices, getContext());
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  TensorType inputType = getInput().getType();
  TensorType resultType = getResult().getType();
  // Be as conservative as possible. We require input/result type equality and
  // all static shapes. Broadcast dims must be the identity mapping.
  if (inputType.hasStaticShape() && inputType == resultType &&
      llvm::equal(getBroadcastDims(),
                  llvm::seq<int64_t>(0, inputType.getRank())))
    return getInput();
  return nullptr;
}

void BroadcastOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, Type result,
                        Value input, ArrayRef<int64_t> broadcast_dims) {
  BroadcastOp::build(odsBuilder, odsState, result, input, /*shape=*/Value(),
                     broadcast_dims);
}

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//

void tensorrt::ArgMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value input, int64_t axis) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto rttBuilder = RankedTensorType::Builder(inputType);
  SmallVector<int64_t> shape(inputType.getShape());
  shape[axis] = 1;
  rttBuilder.setShape(shape);
  RankedTensorType valuesType = rttBuilder;
  rttBuilder.setElementType(odsBuilder.getI32Type());
  RankedTensorType indicesType = rttBuilder;
  ArgMinOp::build(odsBuilder, odsState, TypeRange{valuesType, indicesType},
                  input, axis);
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

void tensorrt::ArgMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Value input, int64_t axis) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto rttBuilder = RankedTensorType::Builder(inputType);
  SmallVector<int64_t> shape(inputType.getShape());
  shape[axis] = 1;
  rttBuilder.setShape(shape);
  RankedTensorType valuesType = rttBuilder;
  rttBuilder.setElementType(odsBuilder.getI32Type());
  RankedTensorType indicesType = rttBuilder;
  ArgMaxOp::build(odsBuilder, odsState, TypeRange{valuesType, indicesType},
                  input, axis);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::GatherOp::verify() {
  // Verify `numBroadcastDims`.
  int64_t batchDims = this->getNumBroadcastDims();
  if (batchDims != 0 && batchDims != 1)
    return emitOpError("in kDEFAULT mode, numElementWiseDims must be 0 or 1");

  // Verify axis.
  int64_t axis = getAxis();
  if (axis < 0 || axis > getData().getType().getRank() - 1)
    return emitOpError("expected \"axis\" to must be in the range [")
           << 0 << ", " << getData().getType().getRank() << ")";

  // Verify `numBroadcastDims` <= `axis`
  if (batchDims > axis)
    return emitOpError("expected \"numBroadcastDims\" <= \"axis\"");

  // When `numBroadcastDims` is 1, verify first dim is broadcastable
  ArrayRef<int64_t> dataShape = getData().getType().getShape();
  ArrayRef<int64_t> indicesShape = getIndices().getType().getShape();
  if ((this->getNumBroadcastDims() == 1) &&
      (failed(checkShapesBroadcastable(dataShape.take_front(1),
                                       indicesShape.take_front(1)))))
    return emitOpError("when numBroadcastDims = 1, first dimension of data and "
                       "indices must be broadcastable");

  return success();
}

//===----------------------------------------------------------------------===//
// GatherElementsOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::GatherElementsOp::verify() {
  // Verify axis.
  int64_t axis = getAxis();
  if (axis < 0 || axis > getType().getRank() - 1)
    return emitOpError("expected \"axis\" to must be in the range [")
           << 0 << ", " << getType().getRank() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void tensorrt::IfOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    Value condition,
    function_ref<void(OpBuilder &, Location)> trueBranchBuilder,
    function_ref<void(OpBuilder &, Location)> falseBranchBuilder) {
  result.addOperands(condition);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard g(builder);
  Region *trueRegion = result.addRegion();
  builder.createBlock(trueRegion);
  trueBranchBuilder(builder, result.location);

  Region *falseRegion = result.addRegion();
  builder.createBlock(falseRegion);
  falseBranchBuilder(builder, result.location);
}

/// Return false if the op cannot be used in a conditional layer body, otherwise
/// return true.
/// TODO: this function currently partitions according to
/// https://docs.nvidia.com/deeplearning/tensorrt/operators/index.html#layers-flow-control-constructs
/// However, it is missing some checks on convolution/activation/fill/unary ops
/// and therefore may give false positives.
bool isOperationSupportedInControlFlowBranchRegion(TensorRTOpInterface op) {
  return !isa<PaddingOp, DeconvolutionOp, ParametricReLUOp, PoolingOp,
              RaggedSoftMaxOp, ResizeNearestOp, ResizeLinearOp, ResizeCubicOp>(
      op);
}

LogicalResult tensorrt::IfOp::verifyRegions() {
  TensorRTOpInterface errorOp = nullptr;
  for (Region *region : {&getTrueRegion(), &getFalseRegion()}) {
    WalkResult result = region->walk([&](TensorRTOpInterface trtOp) {
      if (!isOperationSupportedInControlFlowBranchRegion(trtOp)) {
        errorOp = trtOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      errorOp->emitOpError() << "not supported in control flow region body";
      return emitOpError("contains an operation that is not supported in a "
                         "control flow branch region");
    }
  }
  return success();
}

static std::optional<bool> getConditionalAttribute(Attribute condition) {
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(condition);
  if (!elementsAttr)
    return std::nullopt;
  auto rtt = dyn_cast<RankedTensorType>(elementsAttr.getType());
  if (!rtt || !rtt.getElementType().isInteger(1) || rtt.getRank() != 0)
    return std::nullopt;
  return elementsAttr.getValues<bool>()[0];
}

void tensorrt::IfOp::getRegionInvocationBounds(
    ::llvm::ArrayRef<::mlir::Attribute> operands,
    ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) {
  assert(operands.size() == 1 && "expected one operand");
  std::optional<bool> condFold = getConditionalAttribute(operands.front());
  if (condFold) {
    invocationBounds.emplace_back(0, *condFold ? 1 : 0);
    invocationBounds.emplace_back(0, *condFold ? 0 : 1);
    return;
  }

  invocationBounds.emplace_back(0, 1);
  invocationBounds.emplace_back(0, 1);
}

void tensorrt::IfOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.emplace_back(getResults());
    return;
  }
  regions.emplace_back(&getTrueRegion());
  regions.emplace_back(&getFalseRegion());
}

void tensorrt::IfOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands, *this);
  std::optional<bool> boolValue =
      getConditionalAttribute(adaptor.getCondition());

  // If the condition is not staticly determined or it is true, then else region
  // is possible.
  if (!boolValue || *boolValue)
    regions.emplace_back(&getTrueRegion());
  if (!boolValue || !*boolValue)
    regions.emplace_back(&getFalseRegion());
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  assert((point.isParent() || point == getParentOp().getBodyRegion()) &&
         "condition op can only exit the loop or branch to the body"
         "region");
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

void ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands, *this);
  tensorrt::WhileOp whileOp = getParentOp();
  // Condition can either lead to the after region or back to the parent op
  // depending on whether the condition is true or not.
  regions.emplace_back(&whileOp.getBodyRegion(),
                       whileOp.getBodyRegion().getArguments());
  regions.emplace_back(whileOp.getResults());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void tensorrt::WhileOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getCondRegion(), getCondRegion().getArguments());
    return;
  }

  // WhileOp has only two regions, so throw an error if index > 1
  assert((point.getRegionOrNull() == &getCondRegion() ||
          point.getRegionOrNull() == &getBodyRegion()) &&
         "there are only two regions in WhileOp");
  // The bodyRegion, i.e., Region#1 always branches back to the condition region
  if (point.getRegionOrNull() == &getBodyRegion()) {
    regions.emplace_back(&getCondRegion(), getCondRegion().getArguments());
    return;
  }

  // From cond region we can branch out or to the body.
  regions.emplace_back(getResults());
  regions.emplace_back(&getBodyRegion(), getBodyRegion().getArguments());
}

LogicalResult tensorrt::WhileOp::verifyRegions() {
  TensorRTOpInterface errorOp = nullptr;
  for (auto region : getRegions()) {
    WalkResult result = region->walk([&](TensorRTOpInterface trtOp) {
      if (!isOperationSupportedInControlFlowBranchRegion(trtOp)) {
        errorOp = trtOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      errorOp->emitOpError() << "not supported in control flow region body";
      return emitOpError("contains an operation that is not supported in a "
                         "control flow branch region");
    }
  }
  return success();
}

OperandRange
tensorrt::WhileOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
  assert(point.getRegionOrNull() &&
         point.getRegionOrNull() == &getCondRegion() &&
         "expected to branch only to the first region from entry");
  return getOperands();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  // The induction variable type (type of start, step, stop, and first block
  // arg) is `tensor<i32>`.
  Type inductionVariableType = RankedTensorType::get({}, builder.getI32Type());

  // The first block argument of the body region is the induction variable.
  OpAsmParser::Argument inductionVariable;
  inductionVariable.type = inductionVariableType;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse the induction variable followed by " `=` $iv_lower_bound `to`
  // $iv_upper_bound `step` $iv_step".
  if (parser.parseArgument(inductionVariable) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) ||
      parser.resolveOperand(lb, inductionVariableType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, inductionVariableType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, inductionVariableType, result.operands))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> initOperands;
  regionArgs.push_back(inductionVariable);

  if (parser.parseKeyword("init") ||
      // Assignment list is "`(` ($blockArg = $operand)* `)`"
      parser.parseAssignmentList(regionArgs, initOperands) ||
      parser.parseArrowTypeList(result.types))
    return failure();
  // Block args types (except IV), init operand types, and result types must
  // be identical.
  for (auto argOperandType :
       llvm::zip(llvm::drop_begin(regionArgs), initOperands, result.types)) {
    Type type = std::get<2>(argOperandType);
    std::get<0>(argOperandType).type = type;
    if (parser.resolveOperand(std::get<1>(argOperandType), type,
                              result.operands))
      return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  // Add the implicit terminator. This should never happen really, but
  // "SingleBlockImplicitTerminator" has other benefits.
  ForOp::ensureTerminator(*body, builder, result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLb() << " to " << getUb()
    << " step " << getStep();

  p << " init(";
  llvm::interleaveComma(
      llvm::zip(getBlockArgsForLoopCarriedDeps(), getInit()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ")";
  p << " -> (" << getInit().getTypes() << ')';
  p << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult tensorrt::ForOp::verifyRegions() {
  TensorRTOpInterface errorOp = nullptr;
  WalkResult result = getBodyRegion().walk([&](TensorRTOpInterface trtOp) {
    if (!isOperationSupportedInControlFlowBranchRegion(trtOp)) {
      errorOp = trtOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    errorOp->emitOpError() << "not supported in control flow region body";
    return emitOpError("contains an operation that is not supported in a "
                       "control flow branch region");
  }
  return success();
}

void tensorrt::ForOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  regions.push_back(
      RegionSuccessor(&getBodyRegion(), getBlockArgsForLoopCarriedDeps()));
  regions.push_back(RegionSuccessor(getResults()));
}

OperandRange ForOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getInit();
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

static OpFoldResult foldShapeOp(ShapeOp op) {
  // If the input is not statically shaped, just return the original result.
  TensorType inputType = op.getInput().getType();
  if (!inputType.hasStaticShape())
    return op.getResult();

  // Otherwise, return the constant value of the shape as DenseElementsAttr of
  // i32 tensor type.
  auto i32Type = IntegerType::get(op->getContext(), 32);
  auto shapeTensorType = RankedTensorType::get({inputType.getRank()}, i32Type);
  // Convert the shape to int32.
  auto shapeI32 =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [](int64_t dim) {
        assert(dim >= 0 && "expected non-negative dimensions");
        assert(static_cast<int64_t>(static_cast<int32_t>(dim)) == dim &&
               "expected shape dim can be represented in int32");
        return APInt(/*numBits=*/32, dim);
      }));
  return DenseElementsAttr::get(shapeTensorType, shapeI32);
}

namespace {
/// Pattern to simplify `tensorrt.shape` when the input is a staticly-shaped
/// tensor. In that case, we can replace `tensorrt.shape` with a constant i32
/// tensor containing the shape.
struct SimplifyShapeOfStaticInput : public OpRewritePattern<tensorrt::ShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensorrt::ShapeOp op,
                                PatternRewriter &rewriter) const override {
    OpFoldResult shape = foldShapeOp(op);
    if (isa<Value>(shape))
      return failure();
    auto shapeConst = cast<DenseElementsAttr>(cast<Attribute>(shape));
    rewriter.replaceOpWithNewOp<tensorrt::ConstantOp>(op, shapeConst.getType(),
                                                      shapeConst);
    return success();
  }
};
} // namespace

void ShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<SimplifyShapeOfStaticInput>(context);
}

OpFoldResult ShapeOp::fold(FoldAdaptor adaptor) {
  if (!getInput().getType().hasStaticShape())
    return nullptr;
  return foldShapeOp(*this);
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

static Attribute foldIdentityFloatToFloatCast(RankedTensorType resultType,
                                              Type newElementType,
                                              ElementsAttr attr) {
  if (auto handle = getElidedResourceElementsAttr(attr))
    return DenseResourceElementsAttr::get(resultType, *handle);
  DenseElementsAttr els = dyn_cast<DenseElementsAttr>(attr);
  if (!els)
    return {};

  // Don't do very large, expensive folds here.
  if (!els.isSplat() && els.getNumElements() > kFoldOpEltLimit)
    return {};

  // Float -> Float
  return els.mapValues(newElementType, [&](const APFloat &floatVal) -> APInt {
    APFloat convertedFloat = floatVal;
    bool losesInfo = false;
    convertedFloat.convert(cast<FloatType>(newElementType).getFloatSemantics(),
                           APFloat::rmNearestTiesToEven, &losesInfo);
    return convertedFloat.bitcastToAPInt();
  });
}

template <typename FoldAdaptor>
static OpFoldResult foldIdentity(RankedTensorType resultType,
                                 TypedValue<RankedTensorType> input,
                                 FoldAdaptor adaptor) {
  if (input.getType() == resultType)
    return input;
  if (auto idenProducer = input.getDefiningOp<IdentityOp>()) {
    // We can only fold if the effect is a noop (e.g. no truncation).
    if (idenProducer.getType().getElementTypeBitWidth() >
            resultType.getElementTypeBitWidth() &&
        resultType == idenProducer.getInput().getType())
      return idenProducer.getInput();
  }

  auto attr = dyn_cast_or_null<ElementsAttr>(adaptor.getInput());
  if (!attr)
    return {};

  Type oldElementType = mlir::getElementTypeOrSelf(input.getType());
  Type newElementType = mlir::getElementTypeOrSelf(resultType);

  if (isa<FloatType>(oldElementType) && isa<FloatType>(newElementType))
    return foldIdentityFloatToFloatCast(resultType, newElementType, attr);

  return {};
}

OpFoldResult IdentityOp::fold(FoldAdaptor adaptor) {
  return foldIdentity(getType(), getInput(), adaptor);
}

OpFoldResult Identity84Op::fold(FoldAdaptor adaptor) {
  return foldIdentity(getType(), getInput(), adaptor);
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

namespace {
/// A `tensorrt.reduce` that is followed by a `tensorrt.expand_rank` can be
/// simplified by making the reduce operation retain the reduction dimension in
/// the result as a size-1 dimension. To avoid duplicating the reduction
/// operation, we only do this if the only user of the reduction is the
/// `tensorrt.expand_rank` op.
struct ReduceExpandRankRewriter : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKeepDimensions() || !op->hasOneUse() ||
        op.getReduceAxes().size() != 1)
      return failure();
    int64_t reducedDim = op.getReduceAxes().front();
    auto expandOp = dyn_cast<ExpandRankOp>(*op->user_begin());
    if (!expandOp)
      return failure();
    RankedTensorType inputType = op.getInput().getType();
    RankedTensorType expandedType = expandOp.getType();
    if (expandedType.getRank() != inputType.getRank() ||
        expandedType.getDimSize(reducedDim) != 1)
      return failure();

    rewriter.replaceOpWithNewOp<ReduceOp>(expandOp, op.getInput(),
                                          op.getReduceAxes(), true,
                                          op.getReduceOperation());
    return success();
  }
};

/// A `tensorrt.reduce` that is followed by a `tensorrt.element_wise <kDIV>`
/// where the divisor is a constant of size equal to the dimension length being
/// reduced, then we can replace the `div(reduce)` with an equivalent
/// mean-reduction.
struct ReduceSumDivToReduceMeanRewriter : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getReduceAxes().size() != 1 ||
        op.getReduceOperation() != ReduceOperation::kSUM || !op->hasOneUse())
      return failure();
    auto ewiseOp = dyn_cast<ElementWiseOp>(*op->user_begin());
    if (!ewiseOp)
      return failure();
    auto constOp = ewiseOp.getInput2().getDefiningOp<ConstantOp>();
    if (ewiseOp.getElementwiseOperation() != ElementWiseOperation::kDIV ||
        !constOp)
      return failure();
    int64_t reducedDim = op.getReduceAxes().front();
    RankedTensorType reductionInputType = op.getInput().getType();
    ElementsAttr divisor = constOp.getWeights();
    if (!divisor.isSplat() ||
        !isa<FloatType>(mlir::getElementTypeOrSelf(divisor.getType())) ||
        divisor.getSplatValue<APFloat>().convertToDouble() !=
            static_cast<double>(reductionInputType.getDimSize(reducedDim)))
      return failure();

    rewriter.replaceOpWithNewOp<ReduceOp>(
        ewiseOp, op.getInput(), op.getReduceAxes(), op.getKeepDimensions(),
        ReduceOperation::kAVG);
    return success();
  }
};
} // namespace

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ReduceExpandRankRewriter, ReduceSumDivToReduceMeanRewriter>(
      context);
}

//===----------------------------------------------------------------------===//
// OpaquePluginOp
//===----------------------------------------------------------------------===//

void OpaquePluginOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {}

std::optional<SmallVector<ShapedTypeComponents>>
OpaquePluginOp::inferShapeComponentsFromShapesRegion() {
  Region &shapeRegion = getShapesRegion();
  if (shapeRegion.empty())
    return std::nullopt;

  SmallVector<ShapedTypeComponents> shapes;
  shapes.reserve(getNumResults());

  auto termOp = cast<tensorrt::YieldOp>(shapeRegion.front().getTerminator());
  unsigned yieldIdx = 0;
  for (Value resultTensor : getResults()) {
    auto tensorType = cast<RankedTensorType>(resultTensor.getType());
    SmallVector<int64_t> dims;
    for (int64_t i = 0, e = tensorType.getRank(); i < e; i++) {
      Value yieldedScalar = termOp->getOperand(yieldIdx++);
      IntegerAttr intAttr{};
      if (matchPattern(yieldedScalar, m_Constant(&intAttr))) {
        dims.push_back(intAttr.getInt());
        continue;
      }
      dims.push_back(ShapedType::kDynamic);
    }
    shapes.push_back(ShapedTypeComponents(dims, tensorType.getElementType()));
  }

  return shapes;
}

LogicalResult OpaquePluginOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getShapesRegion().empty())
    return failure();

  // Clone the scalar shape IR at the insertion point, remapping block args to
  // dimensions of inputs.
  Region &shapeRegion = getShapesRegion();
  SmallVector<Value> blockArgReplacements;
  blockArgReplacements.reserve(shapeRegion.getNumArguments());
  IRMapping mapping;
  unsigned blockArgIdx = 0;
  for (Value inputTensor : getInputs()) {
    auto tensorType = cast<RankedTensorType>(inputTensor.getType());
    for (int64_t i = 0, e = tensorType.getRank(); i < e; i++) {
      // If it's dynamic, create the tensor.dim op.
      if (tensorType.isDynamicDim(i)) {
        auto dimOp =
            builder.create<tensor::DimOp>(this->getLoc(), inputTensor, i);
        // Cast from index to i64.
        auto castOp = builder.create<arith::IndexCastOp>(
            this->getLoc(), builder.getI64Type(), dimOp);
        mapping.map(shapeRegion.getArgument(blockArgIdx++), castOp);
        continue;
      }
      // Otherwise, create the constant scalar.
      auto constOp = builder.create<arith::ConstantOp>(
          this->getLoc(), builder.getI64IntegerAttr(tensorType.getDimSize(i)));
      mapping.map(shapeRegion.getArgument(blockArgIdx++), constOp);
    }
  }

  // Clone the body (except the terminator) to materialize the shape
  // calculation.
  for (Operation &op : shapeRegion.front().without_terminator())
    builder.clone(op, mapping);

  // Lookup the cloned values that would have been yielded and organize them
  // into the output vector `reifiedReturnShapes`.
  auto termOp = cast<tensorrt::YieldOp>(shapeRegion.front().getTerminator());
  unsigned yieldIdx = 0;
  reifiedReturnShapes.reserve(getNumResults());
  for (Value resultTensor : getResults()) {
    auto tensorType = cast<RankedTensorType>(resultTensor.getType());
    SmallVector<OpFoldResult> &shape = reifiedReturnShapes.emplace_back();
    shape.reserve(tensorType.getRank());
    for (int64_t i = 0, e = tensorType.getRank(); i < e; i++) {
      Value yielded = mapping.lookup(termOp.getOperand(yieldIdx++));
      shape.push_back(builder.createOrFold<arith::IndexCastOp>(
          this->getLoc(), builder.getIndexType(), yielded));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ResizeOp
//===----------------------------------------------------------------------===//

/// Fold `tensorrt.op(..., tensor.cast(x)... )` to `tensorrt.op(..., x, ...)`
/// if the cast is a generalizing cast (it is removing some static dims of the
/// type of  `x` and replacing them with dynamic dimensions).
template <typename OpType>
struct AbsorbTensorCastOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (auto castOp = op.getInput().template getDefiningOp<tensor::CastOp>()) {
      RankedTensorType castType = cast<RankedTensorType>(castOp.getType());
      RankedTensorType sourceType =
          cast<RankedTensorType>(castOp.getSource().getType());
      if (castType.getEncoding() != sourceType.getEncoding() ||
          !isTargetRefinementOfSource(castType.getShape(),
                                      sourceType.getShape()))
        return failure();
      rewriter.modifyOpInPlace(
          op, [&]() { op.getInputMutable().assign(castOp.getSource()); });
      return success();
    }
    return failure();
  }
};

void ResizeCubicOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<AbsorbTensorCastOp<ResizeCubicOp>>(context);
}

void ResizeLinearOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.insert<AbsorbTensorCastOp<ResizeLinearOp>>(context);
}

void ResizeNearestOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.insert<AbsorbTensorCastOp<ResizeNearestOp>>(context);
}

//===----------------------------------------------------------------------===//
// DimList Attribute Parameter
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<int64_t>> tensorrt::parseDimList(AsmParser &parser) {
  SmallVector<int64_t> dims;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        dims.emplace_back();
        return parser.parseInteger(dims.back());
      })))
    return failure();
  return dims;
}

void tensorrt::printDimList(AsmPrinter &printer, ArrayRef<int64_t> dims) {
  printer << "[";
  llvm::interleaveComma(dims, printer);
  printer << "]";
}

//===----------------------------------------------------------------------===//
// ShapeProfileAttr
//===----------------------------------------------------------------------===//

ShapeProfileAttr
tensorrt::ShapeProfileAttr::get(MLIRContext *context,
                                DynamicDimensionBounds batchSizeBounds,
                                ArrayRef<int64_t> nonBatchDims) {
  SmallVector<int64_t> shapeMin, shapeOpt, shapeMax;
  // info.type = argType;
  shapeMin = {batchSizeBounds.min};
  llvm::append_range(shapeMin, nonBatchDims);
  shapeOpt = {batchSizeBounds.opt};
  llvm::append_range(shapeOpt, nonBatchDims);
  shapeMax = {batchSizeBounds.max};
  llvm::append_range(shapeMax, nonBatchDims);
  return ShapeProfileAttr::get(context, shapeMin, shapeOpt, shapeMax);
}

LogicalResult tensorrt::ShapeProfileAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> min, ::llvm::ArrayRef<int64_t> opt,
    ::llvm::ArrayRef<int64_t> max) {
  unsigned idx = 0;
  if (min.size() != opt.size() || opt.size() != max.size())
    return emitError() << "shape min/opt/max arrays should have equal size";
  for (auto [minVal, optVal, maxVal] : llvm::zip(min, opt, max)) {
    if (minVal > optVal)
      return emitError() << "profile dimension " << idx << " min=" << minVal
                         << " should be less than or equal to opt=" << optVal;
    if (optVal > maxVal)
      return emitError() << "profile dimension " << idx << " opt=" << optVal
                         << " should be less than or equal to max=" << maxVal;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Op Trait Helper Implementations
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::detail::verifyInferredTensorTypesWithPartialInfo(
    Operation *op,
    function_ref<LogicalResult(
        MLIRContext *, std::optional<Location>, ValueShapeRange, DictionaryAttr,
        OpaqueProperties, RegionRange, SmallVectorImpl<ShapedTypeComponents> &)>
        componentTypeFn,
    bool shapesEqualUpToDynamicAmbiguity) {

  SmallVector<ShapedTypeComponents> components;
  if (failed(componentTypeFn(op->getContext(), op->getLoc(), op->getOperands(),
                             op->getAttrDictionary(),
                             op->getPropertiesStorage(), op->getRegions(),
                             components)))
    return failure();

  if (components.size() != op->getNumResults())
    return op->emitOpError() << "inferred " << components.size()
                             << " results but expected " << components.size();

  auto compareShapes = [shapesEqualUpToDynamicAmbiguity](
                           ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    if (shapesEqualUpToDynamicAmbiguity)
      return areShapesEquivalentUpToDynamicDims(lhs, rhs);
    return lhs == rhs;
  };

  for (auto [idx, component, resultType] :
       llvm::zip(llvm::seq<unsigned>(0, components.size()), components,
                 op->getResultTypes())) {
    // TODO: fix upstream so that we don't have to use 'ranked' as a flag
    // here. Default for ShapedTypeComponents is to set 'ranked' to false.
    // Since we don't allow unranked shapes, we use this to mean the shape is
    // not inferrable.
    auto resultRtt = dyn_cast<RankedTensorType>(resultType);

    // This is true because TensorRT ops are only allowed to return
    // RankedTensorType'd values.
    if (!resultRtt)
      return op->emitOpError()
             << "expected a RankedTensorType but got " << resultType;

    if (component.hasRank() &&
        !compareShapes(resultRtt.getShape(), component.getDims())) {
      std::string errorMsg;
      llvm::raw_string_ostream ss(errorMsg);
      ss << "result " << idx << " has type " << resultRtt
         << " but inferred tensor of shape <";
      llvm::interleave(
          component.getDims(),
          [&](int64_t dim) {
            if (dim == ShapedType::kDynamic)
              ss << "?";
            else
              ss << dim;
          },
          [&]() { ss << "x"; });
      ss << ">";
      ss.flush();
      return op->emitOpError() << errorMsg;
    }

    if (component.getElementType() != nullptr &&
        component.getElementType() != resultRtt.getElementType())
      return op->emitOpError()
             << "result " << idx << " has element type "
             << resultRtt.getElementType() << " but inferred element type"
             << component.getElementType();
  }
  return success();
}

bool tensorrt::detail::isCompatibleReturnTypesShapes(
    TypeRange lhs, TypeRange rhs, bool shapesEqualUpToDynamicAmbiguity) {
  if (!shapesEqualUpToDynamicAmbiguity)
    return lhs == rhs;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [l, r] = pair;
    return areShapesEquivalentUpToDynamicDims(cast<RankedTensorType>(l),
                                              cast<RankedTensorType>(r));
  });
}

//===----------------------------------------------------------------------===//
// TensorRTDialectOpAsmInterface
//===----------------------------------------------------------------------===//

namespace {
class TensorRTDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  /// Tells MLIR assembly printer/parser that the ShapeProfileAttr can be
  /// aliased using #profile[num]. This make the IR more readable.
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<ShapeProfileAttr>(attr)) {
      os << "profile";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

Operation *TensorRTDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  if (!elementsAttr)
    return nullptr;
  if (type != elementsAttr.getType())
    return nullptr;
  return builder.create<ConstantOp>(loc, type, elementsAttr);
}

void TensorRTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.cpp.inc"
      >();

  addInterface<TensorRTDialectOpAsmInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttrInterfaces.cpp.inc"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.cpp.inc"
