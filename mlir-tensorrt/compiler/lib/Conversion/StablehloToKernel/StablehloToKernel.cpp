//===- StablehloToKernel.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// This file implements a conversion pass that converts Stablehlo dialect to
/// Kernel dialect operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_STABLEHLOTOKERNELPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

/// Converts a Stablehlo operation inside of a 'stablehlo.scatter' computation
/// body into a scalar MLIR operation using Arith/Math dialect.
static FailureOr<Value> mapOpToScalar(RewriterBase &rewriter,
                                      Operation *producer, IRMapping &map) {
  SmallVector<Value> operands;
  for (Value v : producer->getOperands()) {
    Value remapped = map.lookupOrDefault(v);
    if (!remapped)
      return failure();
    operands.push_back(remapped);
  }

  // clang-format off
  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(producer)
      .Case<
        stablehlo::AbsOp,
        stablehlo::AddOp,
        stablehlo::AndOp,
        stablehlo::Atan2Op,
        stablehlo::BitcastConvertOp,
        stablehlo::CbrtOp,
        stablehlo::CeilOp,
        stablehlo::ClampOp,
        stablehlo::ClzOp,
        stablehlo::CompareOp,
        stablehlo::ComplexOp,
        stablehlo::ConvertOp,
        stablehlo::CosineOp,
        stablehlo::DivOp,
        stablehlo::ExpOp,
        stablehlo::Expm1Op,
        stablehlo::FloorOp,
        stablehlo::ImagOp,
        stablehlo::IsFiniteOp,
        stablehlo::Log1pOp,
        stablehlo::LogOp,
        stablehlo::LogisticOp,
        stablehlo::MaxOp,
        stablehlo::MinOp,
        stablehlo::MulOp,
        stablehlo::NegOp,
        stablehlo::NotOp,
        stablehlo::OrOp,
        stablehlo::PopulationCountOp,
        stablehlo::PowOp,
        stablehlo::RealOp,
        stablehlo::ReducePrecisionOp,
        stablehlo::RemOp,
        stablehlo::RoundNearestEvenOp,
        stablehlo::RoundOp,
        stablehlo::RsqrtOp,
        stablehlo::SelectOp,
        stablehlo::ShiftLeftOp,
        stablehlo::ShiftRightArithmeticOp,
        stablehlo::ShiftRightLogicalOp,
        stablehlo::SignOp,
        stablehlo::SineOp,
        stablehlo::SqrtOp,
        stablehlo::SubtractOp,
        stablehlo::TanhOp,
        stablehlo::XorOp
      >([&](auto op){
        return stablehlo::StablehloOpToStdScalarOp::mapOp(
        op, mlir::getElementTypeOrSelf(op.getType()), operands, &rewriter);

      })
      .Default([&](Operation *op) { return failure(); });
  // clang-format on
}

/// Converts the body of a 'stablehlo.scatter' operation into a kernel.scatter
/// body. The original 'stablehlo.scatter' body block is 'body' and the
/// rewritten operations are built in 'newBody'.
static LogicalResult convertScatterRegion(RewriterBase &rewriter, Block &body,
                                          Block &newBody) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&newBody);
  IRMapping map;
  for (BlockArgument arg : body.getArguments()) {
    BlockArgument newArg = newBody.addArgument(
        cast<RankedTensorType>(arg.getType()).getElementType(), arg.getLoc());
    map.map(arg, newArg);
  }
  for (auto &op : body.without_terminator()) {
    FailureOr<Value> scalar = mapOpToScalar(rewriter, &op, map);
    if (failed(scalar))
      return failure();
    map.map(op.getResult(0), *scalar);
  }

  SmallVector<Value> yielded;
  for (Value v : body.getTerminator()->getOperands()) {
    Value remapped = map.lookupOrDefault(v);
    if (!remapped)
      return failure();
    yielded.push_back(remapped);
  }
  rewriter.create<kernel::YieldOp>(body.getTerminator()->getLoc(), yielded);
  return success();
}

/// Returns true if the given type is supported by the kernel.scatter operation.
static bool isElementTypeSupported(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return isElementTypeSupported(tensorType.getElementType());
  if (auto complexType = dyn_cast<ComplexType>(type))
    return isElementTypeSupported(complexType.getElementType());
  return isa<FloatType, IntegerType, IndexType>(type);
}

/// Check that the result/input types of a 'stablehlo.scatter' operation are
/// suitable for conversion to `kernel.scatter'. They may differ only in the
/// shape static information, not in encoding or element type.
static bool areResultAndInputTypesCastCompatible(stablehlo::ScatterOp op) {
  auto areCastCompatible = [](const auto &it) {
    auto inputTensorType = cast<RankedTensorType>(std::get<0>(it));
    auto resultTensorType = cast<RankedTensorType>(std::get<1>(it));
    // We don't allow type promotion.
    if (inputTensorType.getElementType() != resultTensorType.getElementType())
      return false;
    // We don't allow changing the encoding of the tensor.
    if (inputTensorType.getEncoding() != resultTensorType.getEncoding())
      return false;
    return tensor::CastOp::areCastCompatible(inputTensorType, resultTensorType);
  };
  return llvm::all_of(
      llvm::zip_equal(op->getResultTypes(), op.getInputs().getTypes()),
      areCastCompatible);
}

namespace {

/// Converts a 'stablehlo.scatter' operation into a 'kernel.scatter' operation.
struct StablehloScatterToKernelPattern
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(op.getInputs().getTypes(), isElementTypeSupported) ||
        !isElementTypeSupported(op.getScatterIndices().getType()) ||
        !llvm::all_of(op.getUpdates().getTypes(), isElementTypeSupported))
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    // 'stablehlo.scatter' supports type promotion of result to higher
    // bitwidths. We do not support this.
    // Stablehlo operations also allow the result to erase or add add static
    // information to the result/input shapes. We allow this, but we need to
    // insert `tensor.cast`.
    if (!areResultAndInputTypesCastCompatible(op))
      return rewriter.notifyMatchFailure(
          op, "result and input shapes are not cast compatible");

    stablehlo::ScatterDimensionNumbersAttr dims =
        op.getScatterDimensionNumbers();
    auto scatterOp = rewriter.create<kernel::ScatterOp>(
        op.getLoc(), op.getInputs().getTypes(), op.getScatterIndices(),
        op.getUpdates(), op.getInputs(), dims.getUpdateWindowDims(),
        dims.getInsertedWindowDims(), dims.getInputBatchingDims(),
        dims.getScatterIndicesBatchingDims(),
        dims.getScatterDimsToOperandDims(), dims.getIndexVectorDim(),
        op.getIndicesAreSorted(), op.getUniqueIndices());

    Block &body = scatterOp.getUpdateComputation().emplaceBlock();
    if (failed(convertScatterRegion(rewriter, op.getUpdateComputation().front(),
                                    body))) {
      rewriter.eraseOp(scatterOp);
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert scatter region");
    }

    SmallVector<Value> replacements(scatterOp.getResults());
    for (auto [replacement, original] :
         llvm::zip_equal(replacements, op.getResults())) {
      if (original.getType() != replacement.getType())
        replacement = rewriter.create<tensor::CastOp>(
            original.getLoc(), original.getType(), replacement);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Rewrites a 'stablehlo.sort' operation into a 'kernel.sort' operation.
/// This is currently only supported for a sort of a 1D tensor or a key-value
/// sort of a par of 1D tensors of integer or float type with '<' comparator
/// on the keys only.
struct SortToKernelPattern : public OpRewritePattern<stablehlo::SortOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::SortOp op,
                                PatternRewriter &rewriter) const override {

    auto isSupportedType = [&](Type type) {
      auto rtt = dyn_cast<RankedTensorType>(type);
      return rtt && isa<FloatType, IntegerType>(rtt.getElementType()) &&
             rtt.getRank() == 1;
    };

    if (op.getDimension() != 0 ||
        !llvm::all_of(op.getInputs().getTypes(), isSupportedType) ||
        op.getInputs().size() > 2)
      return rewriter.notifyMatchFailure(op, "not supported");

    Block &comparator = op.getComparator().front();
    Operation *returnOp = cast<stablehlo::ReturnOp>(comparator.getTerminator());
    auto compareOp =
        returnOp->getOperand(0).getDefiningOp<stablehlo::CompareOp>();
    if (!compareOp ||
        compareOp.getComparisonDirection() !=
            stablehlo::ComparisonDirection::LT ||
        compareOp->getOperand(0) != comparator.getArgument(0) ||
        compareOp->getOperand(1) != comparator.getArgument(1))
      return rewriter.notifyMatchFailure(op, "custom comparator not supported");

    auto kernelSort = rewriter.create<kernel::SortOp>(
        op.getLoc(), op.getInputs(), /*block_threads=*/256,
        /*items_per_thread=*/17);
    rewriter.replaceOp(op, kernelSort.getResults());
    return success();
  }
};

/// Implementation of the 'convert-stablehlo-to-kernel' pass.
class StablehloToKernelPass
    : public impl::StablehloToKernelPassBase<StablehloToKernelPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<
      StablehloScatterToKernelPattern,
      SortToKernelPattern
    >(&getContext());
    // clang-format on
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc(), "failed to run patterns in ") << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
