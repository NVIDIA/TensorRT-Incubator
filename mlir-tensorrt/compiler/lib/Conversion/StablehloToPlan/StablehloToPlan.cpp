//===- StablehloToPlan.cpp ------------------------------------------------===//
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
/// Implementation of the `convert-stablehlo-to-plan` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "stablehlo-to-plan"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOPLANPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir
using namespace mlir;

static constexpr StringRef kStablehloDonationArgumentAttr =
    "tf.aliasing_output";

/// If function argument has attribute `tf.aliasing_output = N`, replace it with
/// `plan.aliasing_output = N`. Such attribute represents argument donation hint
/// in stablehlo IR. This function also checks if N is within bound i.e. `N <
/// func.getNumResults()`. When Stablehlo IR is coming from JAX bounded N is
/// guaranteed.
static LogicalResult checkAndUpdateFunction(func::FuncOp func) {
  FunctionType funcType = func.getFunctionType();
  for (unsigned i = 0; i < funcType.getNumInputs(); i++) {
    if (auto N = func.getArgAttrOfType<IntegerAttr>(
            i, kStablehloDonationArgumentAttr)) {
      if (N.getInt() >= funcType.getNumResults())
        return failure();
      func.setArgAttr(i, plan::PlanDialect::kDonationArgAttrName, N);
      func.removeArgAttr(i, kStablehloDonationArgumentAttr);
    }
  }
  return success();
}

namespace {
struct TVMFFIPluginConfig {
  llvm::StringRef pluginName;
  llvm::StringRef functionName;
  llvm::SmallVector<llvm::StringRef> argSpec;
  llvm::SmallVector<int32_t> ioAliasing;
  FunctionType functionType;
  DictionaryAttr immediateArgs;
};
} // namespace

static std::optional<FailureOr<TVMFFIPluginConfig>>
tryGetTVMFFICustomCallConfig(stablehlo::CustomCallOp op) {

  static constexpr llvm::StringRef kMTRTFFIBackendAttrName = "mtrt_ffi_backend";
  static constexpr llvm::StringRef kPluginAttrName = "plugin";
  static constexpr llvm::StringRef kFuncAttrName = "func";
  static constexpr llvm::StringRef kArgSpecAttrName = "arg_spec";
  static constexpr llvm::StringRef kMhloBackendConfigAttrName =
      "mhlo.backend_config";

  auto config = op->getAttrOfType<DictionaryAttr>(kMhloBackendConfigAttrName);
  if (!config) {
    LLVM_DEBUG(DBGS() << "ignoring " << op << " because it has no "
                      << kMhloBackendConfigAttrName << " attribute\n");
    return std::nullopt;
  }
  auto ffiBackend = config.getAs<StringAttr>(kMTRTFFIBackendAttrName);
  if (!ffiBackend) {
    LLVM_DEBUG(DBGS() << "ignoring " << op << " because it has no attribute "
                      << kMTRTFFIBackendAttrName << "\n");
    return std::nullopt;
  }

  std::optional<executor::FFIBackend> ffiBackendEnum =
      executor::symbolizeFFIBackend(ffiBackend.getValue());
  if (!ffiBackendEnum || *ffiBackendEnum != executor::FFIBackend::tvm_ffi) {
    return op->emitError("mtrt_ffi_backend attribute has an invalid value \"")
           << ffiBackend.getValue() << "\"";
  }

  auto pluginName = config.getAs<StringAttr>(kPluginAttrName);
  if (!pluginName)
    return op->emitError("custom call config is missing ")
           << kPluginAttrName << " attribute";
  auto functionName = config.getAs<StringAttr>(kFuncAttrName);
  if (!functionName)
    return op->emitError("custom call config is missing ")
           << kFuncAttrName << " attribute";
  auto argSpec = config.getAs<StringAttr>(kArgSpecAttrName);
  if (!argSpec)
    return op->emitError("custom call config is missing ")
           << kArgSpecAttrName << " attribute";

  SmallVector<NamedAttribute> immediateArgs;
  SmallVector<llvm::StringRef> argSpecComponents;
  FailureOr<executor::abi::plugin::DecodeSpec> decodeSpec =
      executor::abi::plugin::ParseArgSpec(
          op, op->getNumOperands(), op->getNumResults(), argSpec.getValue(),
          config, argSpecComponents, immediateArgs);
  if (failed(decodeSpec))
    return failure();

  TVMFFIPluginConfig result;
  result.pluginName = pluginName.getValue();
  result.functionName = functionName.getValue();
  result.argSpec = argSpecComponents;
  result.immediateArgs = DictionaryAttr::get(op.getContext(), immediateArgs);

  result.ioAliasing = llvm::SmallVector<int32_t>(op->getNumOperands(), -1);
  if (auto aliasConfigArray = op.getOutputOperandAliases()) {
    for (auto aliasConfig :
         aliasConfigArray.getAsRange<stablehlo::OutputOperandAliasAttr>()) {
      if (!aliasConfig.getOperandTupleIndices().empty())
        return op->emitError("output operand alias with non-empty operand "
                             "tuple indices is not supported");
      if (aliasConfig.getOutputTupleIndices().size() != 1)
        return op->emitError("output operand alias should contain a "
                             "'output_tuple_indices' containing a single index "
                             "into the results (tuple results are not "
                             "supported currently for custom call operations)");
      result.ioAliasing[aliasConfig.getOperandIndex()] =
          aliasConfig.getOutputTupleIndices()[0];
    }
  }

  return result;
}

/// Convert `stablehlo.custom_call` to `executor.call_plugin`.
static LogicalResult replaceTVMFFICustomCall(RewriterBase &rewriter,
                                             stablehlo::CustomCallOp op,
                                             const TVMFFIPluginConfig &config) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  Location loc = op.getLoc();

  FunctionType functionType = FunctionType::get(
      op.getContext(), op->getOperandTypes(), op->getResultTypes());

  std::string pluginName =
      llvm::formatv("plugin_{0}", op.getCallTargetName()).str();
  SmallString<16> uniquePluginName =
      mlir::executor::getUniqueSymbolName(module, pluginName);

  executor::PluginOp pluginOp = [&] {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return rewriter.create<executor::PluginOp>(
        loc, uniquePluginName, config.pluginName, config.functionName,
        executor::FFIBackend::tvm_ffi, functionType, config.immediateArgs);
  }();

  SmallVector<Value> outputArguments(op->getResultTypes().size());
  for (auto [inputIdx, outputIdx] : llvm::enumerate(config.ioAliasing)) {
    if (outputIdx != -1)
      outputArguments[outputIdx] = op.getInputs()[inputIdx];
  }
  for (auto [idx, resultType] : llvm::enumerate(op->getResultTypes())) {
    if (outputArguments[idx])
      continue;
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape())
      return op->emitError(
                 "expected ranked tensor type with static shape for result ")
             << idx << " but got " << resultType;
    outputArguments[idx] =
        rewriter.create<tensor::EmptyOp>(loc, tensorType, ValueRange{});
  }

  executor::CallPluginOp callOp = rewriter.create<executor::CallPluginOp>(
      loc, pluginOp, /*stream=*/Value{}, op->getOperands(), outputArguments,
      config.immediateArgs, config.argSpec, config.ioAliasing);

  rewriter.replaceOp(op, callOp);

  return success();
}

namespace {

/// Convert `stablehlo.optimization_barrier` to `plan.optimization_barrier`.
/// TODO: Currently this does not support ops that have `!stablehlo.token`
/// typed operands.
struct OptimizationBarrierPattern
    : public OpRewritePattern<stablehlo::OptimizationBarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::OptimizationBarrierOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support stablehlo.token type conversions.
    if (!llvm::all_of(op.getOperandTypes(),
                      [](Type t) { return isa<RankedTensorType>(t); }))
      return failure();
    rewriter.replaceOpWithNewOp<plan::OptimizationBarrierOp>(
        op, op->getOperandTypes(), op.getOperands());
    return success();
  }
};

/// JAX emits `stablehlo.custom_call` for shape assertion statements like this:
///
/// ```
/// stablehlo.custom_call @shape_assertion(%6, %2, %0, %1) {api_version = 2
/// : i32, error_message = "Input shapes do not match the polymorphic shapes
/// specification. Found inconsistency between dimension size args[1].shape[0]
/// (= {0}) and the specification 'K' (= {1}). Using the following polymorphic
/// shapes specifications: args[0].shape = (K, n),args[1].shape =
/// (K,),args[2].shape = (n,). Obtained dimension variables: 'K' = {1} from
/// specification 'K' for dimension args[0].shape[0] (= {1}), 'n' = {2} from
/// specification 'n' for dimension args[0].shape[1] (= {2}), . Please see
/// https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors
/// for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>,
/// tensor<i32>, tensor<i32>) -> ()
/// ```
///
/// Lower this to `cf.assert`, but currently we can't support the substitutions
/// in the print formatting string.
///
/// just lower like
///
/// ```
/// %cond = tensor.extract %6[] : tensor<i1>
/// cf.assert %cond, "...the original unchanged message..."
/// ```
struct ShapeAssertionPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    // Only match shape_assertion custom calls.
    if (op.getCallTargetName() != "shape_assertion")
      return failure();

    // Verify the op has at least one operand (the condition).
    if (op.getInputs().empty())
      return rewriter.notifyMatchFailure(
          op, "shape_assertion requires at least one operand");

    // Verify the first operand is a tensor<i1>.
    Value condTensor = op.getInputs()[0];
    auto condType = dyn_cast<RankedTensorType>(condTensor.getType());
    if (!condType || condType.getRank() != 0 ||
        !condType.getElementType().isSignlessInteger(1))
      return rewriter.notifyMatchFailure(
          op, "shape_assertion condition must be a tensor<i1>");

    // Get the error message attribute.
    auto errorMessageAttr = op->getAttrOfType<StringAttr>("error_message");
    StringRef errorMessage = errorMessageAttr ? errorMessageAttr.getValue()
                                              : "shape assertion failed";

    // Extract the scalar condition from the tensor.
    Location loc = op.getLoc();
    Value cond = rewriter.create<tensor::ExtractOp>(loc, condTensor);

    // Create the cf.assert operation.
    rewriter.create<cf::AssertOp>(loc, cond, errorMessage);

    // Erase the original custom call (it has no results).
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertStablehloToPlanPass
    : public impl::ConvertStablehloToPlanPassBase<ConvertStablehloToPlanPass> {
  using Base::Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    patterns = std::make_shared<FrozenRewritePatternSet>([&] {
      RewritePatternSet patterns_(context);
      patterns_.add<OptimizationBarrierPattern, ShapeAssertionPattern>(context);
      return patterns_;
    }());
    return success();
  }

  void runOnOperation() override {
    IRRewriter rewriter(getOperation()->getContext());
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (failed(checkAndUpdateFunction(func)))
        return signalPassFailure();
      walkAndApplyPatterns(func, *patterns);

      auto walkResult =
          func.walk<WalkOrder::PostOrder>([&](stablehlo::CustomCallOp op) {
            if (auto config = tryGetTVMFFICustomCallConfig(op)) {
              if (failed(*config))
                return WalkResult::interrupt();
              rewriter.setInsertionPoint(op);
              if (failed(replaceTVMFFICustomCall(rewriter, op, **config)))
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
      if (walkResult.wasInterrupted())
        return signalPassFailure();
    }
  }
};
} // namespace
