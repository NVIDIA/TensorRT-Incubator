//===- StablehloToArith.cpp
//------------------------------------------------===//
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
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOARITHPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
//  ConvertStablehloToArithPass
//===----------------------------------------------------------------------===//

static FailureOr<ElementsAttr>
handleStablehloConstantAttr(Location loc, ElementsAttr elAttr) {
  Type elementType = elAttr.getElementType();
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    if (integerType.isSignless())
      return elAttr;
    Type signlessType =
        IntegerType::get(elAttr.getContext(), integerType.getWidth());
    if (auto denseElementsAttr = dyn_cast<DenseElementsAttr>(elAttr))
      return ElementsAttr(denseElementsAttr.bitcast(signlessType));
    if (auto denseResourceElementsAttr =
            dyn_cast<DenseResourceElementsAttr>(elAttr)) {
      auto handle = denseResourceElementsAttr.getRawHandle();
      return ElementsAttr(DenseResourceElementsAttr::get(
          elAttr.getShapedType().clone(signlessType), handle));
    }
    return emitError(loc, "unsupported constant attribute kind");
  }
  return elAttr;
}

class ConvertStablehloToArithPass
    : public impl::ConvertStablehloToArithPassBase<
          ConvertStablehloToArithPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func->getContext());
    auto walkResult =
        func.walk<WalkOrder::PostOrder>([&](stablehlo::ConstantOp constOp) {
          FailureOr<ElementsAttr> elAttr =
              handleStablehloConstantAttr(constOp.getLoc(), constOp.getValue());
          if (failed(elAttr))
            return WalkResult::interrupt();
          Type newType = elAttr->getType();
          rewriter.setInsertionPoint(constOp);
          auto newConstOp = rewriter.create<arith::ConstantOp>(
              constOp.getLoc(), newType, *elAttr);
          if (newType == constOp.getType()) {
            rewriter.replaceOp(constOp, newConstOp);
          } else {
            rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
                constOp, constOp.getType(), newConstOp.getResult());
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace

#endif // MLIR_TRT_ENABLE_HLO
