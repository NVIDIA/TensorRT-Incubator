//===- LegalizeInt8.cpp ---------------------------------------------------===//
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
/// Definitions of transforms related to int8 QDQ/explicit precision.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_LEGALIZEINT8PASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

template <typename T>
static bool hasUserOfType(mlir::Value v) {
  return llvm::any_of(v.getUsers(), [](Operation *op) { return isa<T>(op); });
}

template <typename T>
static bool isProducedBy(mlir::Value v) {
  return v.getDefiningOp<T>() != nullptr;
}

/// Given an int8 tensor value, add DQ->Q ops with identity scaling and return
/// the DQ and Q ops.
static std::pair<DequantizeOp, QuantizeOp> addIdentityQDQ(OpBuilder &rewriter,
                                                          Value int8Value) {
  Location loc = int8Value.getLoc();
  Value scale = rewriter.create<ConstantOp>(
      loc, DenseElementsAttr::get(
               RankedTensorType::get({1}, rewriter.getF32Type()), 1.0f));
  RankedTensorType dqType =
      RankedTensorType::Builder(int8Value.getType().cast<RankedTensorType>())
          .setElementType(rewriter.getF32Type());
  auto dequantizeOp = rewriter.create<DequantizeOp>(loc, dqType, int8Value,
                                                    scale, IntegerAttr());
  return {dequantizeOp, rewriter.create<QuantizeOp>(loc, int8Value.getType(),
                                                    dequantizeOp.getResult(),
                                                    scale, IntegerAttr())};
}

/// For any int8 input without a DQ user, insert a DQ/Q between the entry block
/// argument and all other users. To be more stringent, you could look at the
/// entire forward slice to find Q/Q nodes, but this is simpler. Do the same for
/// all returned int8 tensors.
static mlir::LogicalResult addDummyQDQNodes(RewriterBase &rewriter,
                                            func::FuncOp op) {
  if (op.getFunctionBody().empty())
    return failure();

  // Set insertion to function start.
  Block *body = &op.getFunctionBody().front();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(body);

  auto hasInt8ElType = [](Type t) {
    return isTensorRTInt8Type(t.cast<TensorType>().getElementType());
  };
  bool hasInt8Args = llvm::any_of(op.getArgumentTypes(), hasInt8ElType);
  bool hasInt8Results = llvm::any_of(op.getResultTypes(), hasInt8ElType);
  if (!hasInt8Args && !hasInt8Results)
    return failure();

  bool change = false;
  if (hasInt8Args) {
    for (BlockArgument arg : op.getArguments()) {
      if (!hasInt8ElType(arg.getType()) || hasUserOfType<DequantizeOp>(arg))
        continue;
      auto [dequantizeOp, quantizeOp] = addIdentityQDQ(rewriter, arg);
      rewriter.replaceAllUsesExcept(arg, quantizeOp.getResult(), dequantizeOp);
      change = true;
    }
  }

  if (hasInt8Results) {
    rewriter.setInsertionPoint(body->getTerminator());
    for (Value arg : body->getTerminator()->getOperands()) {
      if (!hasInt8ElType(arg.getType()) || hasUserOfType<DequantizeOp>(arg) ||
          isProducedBy<QuantizeOp>(arg))
        continue;

      auto [dequantizeOp, quantizeOp] = addIdentityQDQ(rewriter, arg);
      rewriter.replaceAllUsesExcept(arg, quantizeOp.getResult(), dequantizeOp);
      change = true;
    }
  }
  return success(change);
}

/// Convert i8 el attr to i32 attr.
static FailureOr<ElementsAttr> convertI8ElAttr(RewriterBase &rewriter,
                                               ElementsAttr attr) {
  TensorType newType =
      cast<RankedTensorType>(attr.getType()).clone(rewriter.getI32Type());

  if (std::optional<DenseResourceElementsHandle> elidedHandle =
          getElidedResourceElementsAttr(attr))
    return cast<ElementsAttr>(
        DenseResourceElementsAttr::get(newType, *elidedHandle));

  // The attribute is a "DenseElementsAttr", it is never elided.
  if (auto denseValue = dyn_cast<DenseElementsAttr>(attr)) {
    SmallVector<int32_t> newValues;
    newValues.reserve(denseValue.size());
    for (int64_t x : denseValue.getValues<int8_t>())
      newValues.push_back(static_cast<int32_t>(x));
    return cast<ElementsAttr>(
        DenseElementsAttr::get(newType, llvm::ArrayRef(newValues)));
  }
  return failure();
}

/// Rewrite Int8 instants to int32 constants. This is used if no QDQ nodes are
/// present since int8 constants are not allowed in "dynamic range mode". The
/// alternative is inserting QDQ nodes, but this also does not work out well
/// typically.
static LogicalResult convertInt8Constants(RewriterBase &rewriter,
                                          func::FuncOp funcOp) {
  SmallVector<ConstantOp> constOps;
  funcOp->walk([&](ConstantOp constOp) {
    if (constOp.getType().getElementType().isInteger(8))
      constOps.push_back(constOp);
  });
  for (ConstantOp constOp : constOps) {
    FailureOr<ElementsAttr> convertedAttr =
        convertI8ElAttr(rewriter, constOp.getWeights());
    if (failed(convertedAttr))
      return constOp->emitOpError("failed to convert value to i32 tensor");
    rewriter.setInsertionPoint(constOp);
    Value newConst =
        rewriter.create<ConstantOp>(constOp.getLoc(), *convertedAttr);
    rewriter.replaceOpWithNewOp<IdentityOp>(constOp, constOp.getType(),
                                            newConst);
  }
  return success();
}

namespace {
/// A simple pattern rewriter.
struct TrivialPatternRewriter : public PatternRewriter {
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};

/// Implementation of `tensorrt-legalize-int8` pass.
class LegalizeInt8Pass
    : public tensorrt::impl::LegalizeInt8PassBase<LegalizeInt8Pass> {
public:
  using Base::Base;
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    TrivialPatternRewriter rewriter(&getContext());

    bool isQDQMode = func->walk([](Operation *op) {
                           if (isa<QuantizeOp, DequantizeOp>(op))
                             return WalkResult::interrupt();
                           return WalkResult::advance();
                         })
                         .wasInterrupted();

    if (isQDQMode) {
      // This function returns failure when no change is required, which is not
      // an error. We use this convention in case the transform should be
      // outlined to a pattern later.
      (void)addDummyQDQNodes(rewriter, func);
      return;
    }

    if (failed(convertInt8Constants(rewriter, func)))
      return signalPassFailure();
  }
};
} // namespace
