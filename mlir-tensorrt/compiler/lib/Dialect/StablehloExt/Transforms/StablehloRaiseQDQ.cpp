//===- StablehloRaiseQDQ.cpp ----------------------------------------------===//
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
/// Implementation of pass that matches Q(quantize) and DQ(dequantize) patterns
/// in the StableHLO IR and raises these patterns to `stablehlo.composite` op
/// where decomposition is a private function implementing Q or DQ.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <string>

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_STABLEHLORAISEQDQPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;

/// Returns `true` if `type` is quantized type. Quantized type is one of `int8`,
/// `fp8`, and `int4`.
static bool isQuantizedType(RankedTensorType type) {
  Type elementType = type.getElementType();
  return isa<Float8E4M3FNType>(elementType) || elementType.isInteger(8) ||
         elementType.isInteger(4);
}

/// Returns `true` if `type` is quantizable type. Quantizable type is one of
/// `fp32`, `fp16`, and `bf16`.
static bool isQuantizableType(RankedTensorType type) {
  Type elementType = type.getElementType();
  return elementType.isF16() || elementType.isF32() || elementType.isBF16();
}

/// Outlines ops in `patternOps` to a private function and returns a composite
/// op with decomposition equal to the newly created private function. Value
/// `funcArgument` is input to the composite op (and thus to newly created
/// function) and Value `funcReturn` is return value from composite (and thus
/// from newly created function). Returned composite op has three attributes.
/// The `scale` attribute is DenseElementsAttr with value equals to the constant
/// scale used in Q/DQ, `qdq_type` attribute holds type of quantization or
/// dequantization, and `axis` attribute represents Q/DQ axis for per-channel
/// quantization, otherwise its -1. Values in `mapToFuncArgument` are mapped to
/// the argument of newly created function.
static FailureOr<Value> outlineQDQPatternToPrivateFuncAndAddComposite(
    RewriterBase &rewriter, SymbolTable &symTable,
    ArrayRef<Operation *> patternOps, Value funcArgument, Value funcReturn,
    ElementsAttr scaleAttr, ArrayRef<Value> mapToFuncArgument,
    StringRef qdqTypeName, int64_t qdqAxis) {
  OpBuilder b(rewriter.getContext());

  // Create outline function type and op
  FunctionType funcType =
      rewriter.getFunctionType({cast<RankedTensorType>(funcArgument.getType())},
                               {cast<RankedTensorType>(funcReturn.getType())});
  func::FuncOp funcOp =
      func::FuncOp::create(funcReturn.getLoc(), qdqTypeName, funcType);
  // Private visibility and `plan.decomposition` attribute is used to recognize
  // decomposition functions in other passes.
  funcOp.setPrivate();
  funcOp->setAttr("plan.decomposition", rewriter.getUnitAttr());
  symTable.insert(funcOp);

  // Add block and copy operations
  Block *funcEntryBlock = funcOp.addEntryBlock();
  IRMapping blockAndValueMap;
  for (auto v : mapToFuncArgument)
    blockAndValueMap.map(v, funcEntryBlock->getArguments().front());

  b.setInsertionPointToStart(funcEntryBlock);

  for (int i = patternOps.size() - 1; i >= 0; i--)
    b.clone(*patternOps[i], blockAndValueMap);
  b.create<func::ReturnOp>(funcOp->getLoc(),
                           blockAndValueMap.lookup(funcReturn));

  // Create composite op
  std::string compositeName = "tensorrt." + std::string(qdqTypeName);
  auto compositeAttrs = DictionaryAttr::get(
      rewriter.getContext(),
      {NamedAttribute(rewriter.getStringAttr("scale"), scaleAttr),
       NamedAttribute(rewriter.getStringAttr("axis"),
                      rewriter.getI32IntegerAttr(qdqAxis))});
  return rewriter
      .create<stablehlo::CompositeOp>(
          funcReturn.getLoc(),
          funcReturn.getDefiningOp()->getResult(0).getType(), funcArgument,
          compositeName, compositeAttrs, funcOp.getSymName())
      .getResults()[0];
}

/// Gets a producer of `OpTy` and adds it to `ops`, otherwise return nullptr.
template <typename OpTy>
static OpTy getProducer(Value v, SmallVectorImpl<Operation *> &ops) {
  if (OpTy producer = v.getDefiningOp<OpTy>()) {
    ops.push_back(producer);
    return producer;
  }
  return nullptr;
}

/// In quantization, `stablehlo.div` op divides the input(LHS) with scale(RHS)
/// where as in dequantization `stablehlo.mul` op multiplies the input(LHS) with
/// scale(RHS). Given the scale SSA value, `matchQDQScaleConstant` function
/// matches pattern to find the root `stablehlo.constant` op representing scale.
/// Each matched operation is pushed into `quantizePatternOps` for outlining to
/// a private `FuncOp` later. Value that needs to be mapped to the function
/// argument after outlining is pushed to `mapToFuncArgument`. This function
/// returns `MatchScaleReturnType` which holds index of the scale constant (i.e.
/// `stablehlo.constant` op) and QDQ axis (actual axis for per-channel Q/DQ mode
/// or -1 for per-tensor and block Q/DQ mode).
struct MatchScaleReturnType {
  int64_t scaleConstantOpIdx;
  int64_t axisForPerChannelMode;
};

static FailureOr<MatchScaleReturnType>
matchQDQScaleConstant(Value mulOrDivOpRhs,
                      SmallVectorImpl<Operation *> &quantizePatternOps,
                      SmallVectorImpl<Value> &mapToFuncArgument) {
  if (auto scaleConstOp = getProducer<stablehlo::ConstantOp>(
          mulOrDivOpRhs, quantizePatternOps)) {
    return MatchScaleReturnType{
        static_cast<int64_t>(quantizePatternOps.size() - 1), -1};
  }
  if (auto dynBroadcastInDimOp =
          getProducer<stablehlo::DynamicBroadcastInDimOp>(mulOrDivOpRhs,
                                                          quantizePatternOps)) {
    MatchScaleReturnType r{-1, -1};
    auto scaleConstOp = getProducer<stablehlo::ConstantOp>(
        dynBroadcastInDimOp.getOperand(), quantizePatternOps);
    if (!scaleConstOp)
      return failure();
    r.scaleConstantOpIdx = quantizePatternOps.size() - 1;
    // For per-channel quantization, scale value is broadcasted along a single
    // quantization axis to get scale of type same as input.
    // clang-format off
    // For example,
    // %arg0: tensor<?x3x?x?xf32>
    // %scale = stablehlo.constant dense_resource<__elided__> : tensor<3xf32>
    // .....
    // %inp_shape = stablehlo.concatenate .... -> tensor<4xi32>
    // %scale_broadcasted = stablehlo.dynamic_broadcast_in_dim %scale,
    // %inp_shape, dims = [1] : (tensor<3xf32>, tensor<4xi32>) ->
    // tensor<?x3x?x?xf32>
    // ....
    // clang-format on
    // Thus, if dynamic broadcast op has broadcasting dims, its size will be one
    // and value equal to the q/dq axis. For per-tensor quantization, a scalar
    // scale is broadcasted to the input shape so it doesn't have broadcast
    // dims.
    if (!dynBroadcastInDimOp.getBroadcastDimensions().empty())
      r.axisForPerChannelMode =
          dynBroadcastInDimOp.getBroadcastDimensions().front();
    auto concatOp = getProducer<stablehlo::ConcatenateOp>(
        dynBroadcastInDimOp.getOutputDimensions(), quantizePatternOps);
    if (!concatOp)
      return failure();
    for (Value inp : concatOp.getInputs()) {
      auto constOp =
          getProducer<stablehlo::ConstantOp>(inp, quantizePatternOps);
      if (constOp)
        continue;
      auto reshapeOp =
          getProducer<stablehlo::ReshapeOp>(inp, quantizePatternOps);
      if (!reshapeOp)
        return failure();
      auto getDimSizeOp = getProducer<stablehlo::GetDimensionSizeOp>(
          reshapeOp.getOperand(), quantizePatternOps);
      if (!getDimSizeOp)
        return failure();
      mapToFuncArgument.push_back(getDimSizeOp.getOperand());
    }
    return r;
  }
  if (auto broadcastInDimOp = getProducer<stablehlo::BroadcastInDimOp>(
          mulOrDivOpRhs, quantizePatternOps)) {
    MatchScaleReturnType r{-1, -1};
    if (!broadcastInDimOp.getBroadcastDimensions().empty())
      r.axisForPerChannelMode =
          broadcastInDimOp.getBroadcastDimensions().front();
    auto scaleConstOp = getProducer<stablehlo::ConstantOp>(
        broadcastInDimOp.getOperand(), quantizePatternOps);
    if (!scaleConstOp)
      return failure();
    r.scaleConstantOpIdx = quantizePatternOps.size() - 1;
    return r;
  }
  return failure();
}

/// Match min and max clamp values. These values can come from a constant or
/// `constant -> convert` op sequence for types such as bf16.
static LogicalResult
matchClampMinMax(Value clampOpMinOrMax,
                 SmallVectorImpl<Operation *> &quantizePatternOps) {
  auto minClampValue =
      getProducer<stablehlo::ConstantOp>(clampOpMinOrMax, quantizePatternOps);
  if (!minClampValue) {
    auto minClampConvert =
        getProducer<stablehlo::ConvertOp>(clampOpMinOrMax, quantizePatternOps);
    if (!minClampConvert)
      return failure();
    auto minClampConstant = getProducer<stablehlo::ConstantOp>(
        minClampConvert.getOperand(), quantizePatternOps);
    if (!minClampConstant)
      return failure();
  }
  return success();
}

/// Matches core quantize pattern shown below and adds matched ops to
/// `quantizePatternOps` for outlining to a private `FuncOp` later.
/// %div_r = div(%inp, %scale)
/// %rounded_r = round_nearest_even(%div_r)
/// %out = clamp(%min, %rounded_r, %max)
static LogicalResult
matchCoreQuantizePattern(stablehlo::ConvertOp op,
                         SmallVectorImpl<Operation *> &quantizePatternOps) {
  // 1. Parent of convert is `clamp`.
  auto clampOp =
      getProducer<stablehlo::ClampOp>(op.getOperand(), quantizePatternOps);
  if (!clampOp)
    return failure();
  // Get min and max clamp constants
  if (failed(matchClampMinMax(clampOp.getMin(), quantizePatternOps)))
    return failure();
  if (failed(matchClampMinMax(clampOp.getMax(), quantizePatternOps)))
    return failure();

  // 2. Parent of `clamp` is `round_nearest_even`.
  auto roundNearestEvenOp = getProducer<stablehlo::RoundNearestEvenOp>(
      clampOp.getOperand(), quantizePatternOps);
  if (!roundNearestEvenOp)
    return failure();

  // 3. Parent of `round_nearest_even` is `divide` op which divides input with
  // scale.
  auto divOp = getProducer<stablehlo::DivOp>(roundNearestEvenOp.getOperand(),
                                             quantizePatternOps);
  if (!divOp)
    return failure();
  return success();
}

namespace {
/// Match per-tensor and per-channel quantization pattern and raise it to a
/// `stablehlo.composite` op.
struct RaisePerTensorAndPerChannelQuantize
    : public OpRewritePattern<stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // Private func with `plan.decomposition` attribute is created by this pass.
    auto funcParentOp = op->getParentOfType<func::FuncOp>();
    if (funcParentOp && funcParentOp.isPrivate() &&
        funcParentOp->hasAttr("plan.decomposition"))
      return failure();
    // Conversion target is quantizableType -> quantizedType
    if (!isQuantizedType(op.getType()) ||
        !isQuantizableType(op.getOperand().getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "convert op does not convert to quantized type.");

    // Get module symbol table.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    SymbolTable symTable(module);

    // Populate op pattern that creates Q or DQ op.
    // A. Match core quantize patterns.
    SmallVector<Operation *> quantizePatternOps{op};
    if (failed(matchCoreQuantizePattern(op, quantizePatternOps)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match quantize pattern.");
    auto divOp = cast<stablehlo::DivOp>(quantizePatternOps.back());
    SmallVector<Value> mapToFuncArgument{divOp.getLhs()};
    // B. Match patterns to find Q/DQ scale.
    FailureOr<MatchScaleReturnType> scaleMatchResult = matchQDQScaleConstant(
        divOp.getRhs(), quantizePatternOps, mapToFuncArgument);
    if (failed(scaleMatchResult))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match constant scale.");

    // Get ElementsAttr from scale constant.
    Value scaleConstant =
        cast<stablehlo::ConstantOp>(
            quantizePatternOps[scaleMatchResult->scaleConstantOpIdx])
            .getResult();
    ElementsAttr scaleAttr{};
    if (!matchPattern(scaleConstant, m_Constant(&scaleAttr)))
      return failure();

    // For per-tensor quantization, axis is not valid input.
    if (scaleMatchResult->axisForPerChannelMode == -1) {
      std::string qdqTypeName = "pt_q";
      FailureOr<Value> compositeOp =
          outlineQDQPatternToPrivateFuncAndAddComposite(
              rewriter, symTable, quantizePatternOps, divOp.getLhs(),
              op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName,
              scaleMatchResult->axisForPerChannelMode);
      if (failed(compositeOp))
        return failure();
      rewriter.replaceOp(op, *compositeOp);
      return success();
    }

    // For per-channel quantization, `axisForPerChannelMode` returned by
    // `matchQDQScaleConstant` function represents quantization axis. Operation
    // `stablehlo.dyn/broadcast_in_dim` uses quantization axis to broadcast 1D
    // scale tensor to the shape of quantization input.
    std::string qdqTypeName = "pc_q";
    FailureOr<Value> compositeOp =
        outlineQDQPatternToPrivateFuncAndAddComposite(
            rewriter, symTable, quantizePatternOps, divOp.getLhs(),
            op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName,
            scaleMatchResult->axisForPerChannelMode);
    if (failed(compositeOp))
      return failure();
    rewriter.replaceOp(op, *compositeOp);
    return success();
  }
};

/// Match block quantization pattern and raise it to a `stablehlo.composite` op.
struct RaiseBlockQuantize : public OpRewritePattern<stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // Private func with `plan.decomposition` attribute is created by this pass.
    auto funcParentOp = op->getParentOfType<func::FuncOp>();
    if (funcParentOp && funcParentOp.isPrivate() &&
        funcParentOp->hasAttr("plan.decomposition"))
      return failure();
    // Conversion target is quantizableType -> quantizedType
    if (!isQuantizedType(op.getType()) ||
        !isQuantizableType(op.getOperand().getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "convert op does not convert to quantized type.");

    // Get module symbol table.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    SymbolTable symTable(module);

    // Match pattern
    SmallVector<Operation *> quantizePatternOps{op};
    if (failed(matchCoreQuantizePattern(op, quantizePatternOps)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match quantize pattern.");
    auto divOp = cast<stablehlo::DivOp>(quantizePatternOps.back());
    auto reshapeOp =
        getProducer<stablehlo::ReshapeOp>(divOp.getRhs(), quantizePatternOps);
    if (!reshapeOp)
      return failure();
    auto broadcastInDimOp = getProducer<stablehlo::BroadcastInDimOp>(
        reshapeOp.getOperand(), quantizePatternOps);
    if (!broadcastInDimOp)
      return failure();
    auto scaleConstant = getProducer<stablehlo::ConstantOp>(
        broadcastInDimOp.getOperand(), quantizePatternOps);
    if (!scaleConstant || scaleConstant.getType().getRank() != 2)
      return failure();

    // Get ElementsAttr from scale constant.
    ElementsAttr scaleAttr{};
    if (!matchPattern(scaleConstant.getResult(), m_Constant(&scaleAttr)))
      return failure();

    std::string qdqTypeName = "block_q";
    SmallVector<Value> mapToFuncArgument{divOp.getLhs()};
    FailureOr<Value> compositeOp =
        outlineQDQPatternToPrivateFuncAndAddComposite(
            rewriter, symTable, quantizePatternOps, divOp.getLhs(),
            op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName, -1);
    if (failed(compositeOp))
      return failure();
    rewriter.replaceOp(op, *compositeOp);
    return success();
  }
};
} // namespace

/// Matches core quantize pattern `i -> convert -> matmul -> o`.
static LogicalResult
matchCoreDequantizePattern(stablehlo::MulOp op,
                           SmallVectorImpl<Operation *> &dequantizePatternOps) {
  auto convertOp =
      getProducer<stablehlo::ConvertOp>(op.getLhs(), dequantizePatternOps);
  if (!convertOp)
    return failure();
  if (convertOp.getType() != op.getType())
    return failure();
  return success();
}

namespace {
/// Match per-tensor and per-channel dequantization pattern and raise it to a
/// `stablehlo.composite` op.
struct RaisePerTensorAndPerChannelDequantize
    : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    // Private func with `plan.decomposition` attribute is created by this pass.
    auto funcParentOp = op->getParentOfType<func::FuncOp>();
    if (funcParentOp && funcParentOp.isPrivate() &&
        funcParentOp->hasAttr("plan.decomposition"))
      return failure();

    // Get module symbol table.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    SymbolTable symTable(module);

    // Match Q/DQ pattern
    SmallVector<Operation *> dequantizePatternOps{op};
    if (failed(matchCoreDequantizePattern(op, dequantizePatternOps)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match dequantize pattern.");
    auto convertOp = cast<stablehlo::ConvertOp>(dequantizePatternOps.back());
    // Conversion target is quantizedType -> quantizableType
    if (!isQuantizedType(convertOp.getOperand().getType()) ||
        !isQuantizableType(convertOp.getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "convert op does not produce quantizable type.");
    SmallVector<Value> mapToFuncArgument{convertOp.getOperand()};
    FailureOr<MatchScaleReturnType> scaleMatchResult = matchQDQScaleConstant(
        op.getRhs(), dequantizePatternOps, mapToFuncArgument);
    if (failed(scaleMatchResult))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match constant scale.");

    // Get ElementsAttr from scale constant.
    Value scaleConstant =
        cast<stablehlo::ConstantOp>(
            dequantizePatternOps[scaleMatchResult->scaleConstantOpIdx])
            .getResult();
    ElementsAttr scaleAttr{};
    if (!matchPattern(scaleConstant, m_Constant(&scaleAttr)))
      return failure();

    // For per-tensor quantization, axis is not valid input.
    if (scaleMatchResult->axisForPerChannelMode == -1) {
      std::string qdqTypeName = "pt_dq";
      FailureOr<Value> compositeOp =
          outlineQDQPatternToPrivateFuncAndAddComposite(
              rewriter, symTable, dequantizePatternOps, convertOp.getOperand(),
              op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName,
              scaleMatchResult->axisForPerChannelMode);
      if (failed(compositeOp))
        return failure();
      rewriter.replaceOp(convertOp, *compositeOp);
      rewriter.replaceAllOpUsesWith(op, *compositeOp);
      return success();
    }
    // For per-channel dequantization, `axisForPerChannelMode` returned by
    // `matchQDQScaleConstant` function represents quantization axis. Operation
    // `stablehlo.dyn/broadcast_in_dim` uses dequantization axis to broadcast 1D
    // scale tensor to the shape of quantization input.
    std::string qdqTypeName = "pc_dq";
    FailureOr<Value> compositeOp =
        outlineQDQPatternToPrivateFuncAndAddComposite(
            rewriter, symTable, dequantizePatternOps, convertOp.getOperand(),
            op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName,
            scaleMatchResult->axisForPerChannelMode);
    if (failed(compositeOp))
      return failure();
    rewriter.replaceOp(convertOp, *compositeOp);
    rewriter.replaceAllOpUsesWith(op, *compositeOp);
    return success();
  }
};

/// Match block dequantization pattern and raise it to a
/// `stablehlo.composite` op.
struct RaiseBlockDequantize : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    // Private func with `plan.decomposition` attribute is created by this pass.
    auto funcParentOp = op->getParentOfType<func::FuncOp>();
    if (funcParentOp && funcParentOp.isPrivate() &&
        funcParentOp->hasAttr("plan.decomposition"))
      return failure();

    // Get module symbol table.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    SymbolTable symTable(module);

    // Match pattern.
    SmallVector<Operation *> dequantizePatternOps{op};
    if (failed(matchCoreDequantizePattern(op, dequantizePatternOps)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "failed to match dequantize pattern.");
    auto convertOp = cast<stablehlo::ConvertOp>(dequantizePatternOps.back());
    // Conversion target is quantizedType -> quantizableType
    if (!isQuantizedType(convertOp.getOperand().getType()) ||
        !isQuantizableType(convertOp.getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "convert op does not produce quantizable type.");

    SmallVector<Value> mapToFuncArgument{convertOp.getOperand()};

    auto reshapeOp =
        getProducer<stablehlo::ReshapeOp>(op.getRhs(), dequantizePatternOps);
    if (!reshapeOp)
      return failure();
    auto broadcastInDimOp = getProducer<stablehlo::BroadcastInDimOp>(
        reshapeOp.getOperand(), dequantizePatternOps);
    if (!broadcastInDimOp)
      return failure();
    auto scaleConstant = getProducer<stablehlo::ConstantOp>(
        broadcastInDimOp.getOperand(), dequantizePatternOps);
    if (!scaleConstant)
      return failure();

    // Get ElementsAttr from scale constant.
    ElementsAttr scaleAttr{};
    if (!matchPattern(scaleConstant.getResult(), m_Constant(&scaleAttr)))
      return failure();

    std::string qdqTypeName = "block_dq";
    FailureOr<Value> compositeOp =
        outlineQDQPatternToPrivateFuncAndAddComposite(
            rewriter, symTable, dequantizePatternOps, convertOp.getOperand(),
            op.getResult(), scaleAttr, mapToFuncArgument, qdqTypeName, -1);
    if (failed(compositeOp))
      return failure();
    rewriter.replaceOp(convertOp, *compositeOp);
    rewriter.replaceAllOpUsesWith(op, *compositeOp);
    return success();
  }
};

class StablehloRaiseQDQPass
    : public mlir::stablehlo_ext::impl::StablehloRaiseQDQPassBase<
          StablehloRaiseQDQPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns
        .insert<RaisePerTensorAndPerChannelQuantize, RaiseBlockQuantize,
                RaisePerTensorAndPerChannelDequantize, RaiseBlockDequantize>(
            ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc()) << "failed to run patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
