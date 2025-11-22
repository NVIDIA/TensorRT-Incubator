//===- StablehloInferTensorValueRangeImpl.cpp -----------------------------===//
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
/// Implementation of InferTensorValueRangeInterface for specific StableHlo
/// ops.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/IR/StableHloExt.h"
#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mtrt::compiler;

static BoundsArray extUIRanges(ArrayRef<ConstantIntRanges> ranges,
                               unsigned destWidth) {
  SmallVector<ConstantIntRanges> result;
  for (const auto &l : ranges)
    result.push_back(intrange::extUIRange(l, destWidth));
  return BoundsArray(result);
}

static BoundsArray truncRanges(ArrayRef<ConstantIntRanges> ranges,
                               unsigned destWidth) {
  SmallVector<ConstantIntRanges> result;
  for (const auto &l : ranges)
    result.push_back(intrange::truncRange(l, destWidth));
  return BoundsArray(result);
}

static BoundsArray extRanges(ArrayRef<ConstantIntRanges> ranges,
                             unsigned destWidth) {
  SmallVector<ConstantIntRanges> result;
  for (const auto &l : ranges)
    result.push_back(intrange::extRange(l, destWidth));
  return BoundsArray(result);
}

namespace {
class ConvertOpImpl : public InferTensorValueRangeInterface::ExternalModel<
                          ConvertOpImpl, stablehlo::ConvertOp> {
public:
  void
  inferResultRangesFromOptional(Operation *op_,
                                ArrayRef<IntOrTensorValueRange> argRanges,
                                SetTensorValueLatticeFn setResultRanges) const {
    auto op = cast<stablehlo::ConvertOp>(op_);
    Type sourceElementType = op.getOperand().getType().getElementType();
    Type resultElementType = op.getType().getElementType();

    if (!isa<IntegerType, IndexType>(sourceElementType) ||
        !isa<IntegerType, IndexType>(resultElementType) ||
        !BoundsArray::shouldAnalyzeValueBounds(op.getResult())) {
      setResultRanges(op.getResult(), BoundsArray());
      return;
    }

    unsigned sourceWidth =
        ConstantIntRanges::getStorageBitwidth(sourceElementType);
    unsigned destWidth =
        ConstantIntRanges::getStorageBitwidth(resultElementType);

    const auto *argRange0 = argRanges[0].dyn_cast<const BoundsArray *>();
    if (!argRange0 || argRange0->isUninitialized()) {
      setResultRanges(op.getResult(), BoundsArray());
      return;
    }

    // Per Stablehlo spec:
    // "For boolean-to-any-supported-type conversions, the value false is
    // converted to zero, and the value true is converted to one. For
    // any-supported-type-to-boolean conversions, a zero value is converted to
    // false, and non-zero values are converted to true."
    // See https://openxla.org/stablehlo/spec#convert.
    if (sourceWidth == 1 && destWidth > 1) {
      setResultRanges(op.getResult(),
                      extUIRanges(argRange0->getValue(), destWidth));
      return;
    }
    if (destWidth == 1 && sourceWidth > 1) {
      setResultRanges(op.getResult(),
                      truncRanges(argRange0->getValue(), destWidth));
      return;
    }

    if (sourceWidth < destWidth) {
      setResultRanges(op.getResult(),
                      extRanges(argRange0->getValue(), destWidth));
      return;
    }

    if (sourceWidth > destWidth) {
      setResultRanges(op.getResult(),
                      truncRanges(argRange0->getValue(), destWidth));
      return;
    }

    setResultRanges(op.getResult(), BoundsArray());
  }
};

} // namespace

void mlir::stablehlo::registerInferTensorValueRangeInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, stablehlo::StablehloDialect *dialect) {
        stablehlo::ConvertOp::attachInterface<ConvertOpImpl>(*ctx);
      });
}
