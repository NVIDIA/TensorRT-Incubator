//===- ConversionTarget.cpp  ----------------------------------------------===//
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
/// Implementation of common TensorRT dialect conversion infrastructure.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::tensorrt;

//===----------------------------------------------------------------------===//
// TensorRT Conversion Target
//===----------------------------------------------------------------------===//

TensorRTConversionTarget::TensorRTConversionTarget(
    MLIRContext &ctx, TensorRTTypeConverter &typeConverter)
    : ConversionTarget(ctx) {
  addLegalDialect<tensorrt::TensorRTDialect>();
  addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op->getOperandTypes());
  });
}

//===----------------------------------------------------------------------===//
// TensorRT Type Converter
//===----------------------------------------------------------------------===//

TensorRTTypeConverter::TensorRTTypeConverter(
    MLIRContext *ctx, const LowerToTensorRTOptions &options)
    : options(options) {

  addConversion([](Type t) -> std::optional<Type> { return t; });

  addConversion(
      [this](Type type,
             SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        auto tensorType = dyn_cast<TensorType>(type);
        if (!tensorType)
          return std::nullopt;
        if (std::optional<Type> converted = convertTensorType(tensorType)) {
          result.push_back(*converted);
          return success();
        }
        return failure();
      });

  // Add generic source and target materializations to handle cases where
  // non-TensorRT-valid types persist afterconversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    // TODO: revise this for complex number support.
    if (inputs.size() != 1)
      return std::nullopt;
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    // TODO: revise this for complex number support.
    if (inputs.size() != 1)
      return std::nullopt;
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

bool TensorRTTypeConverter::isLegalTensorType(TensorType type) {
  if (!type.hasRank())
    return false;
  Type elType = type.getElementType();
  if (elType.isUnsignedInteger(8))
    return true;
  if (isTensorRTInt8Type(elType))
    return true;
  return elType.isF16() || elType.isF32() || elType.isSignlessInteger(32) ||
         elType.isSignlessInteger(1) || elType.isFloat8E4M3FN() ||
         elType.isBF16() || elType.isInteger(4);
}

std::optional<Type> TensorRTTypeConverter::convertTensorType(TensorType type) {
  if (isLegalTensorType(type))
    return type;

  auto rtt = type.cast<RankedTensorType>();
  Type i32Type = IntegerType::get(type.getContext(), 32);

  // Handle i64 depending on options.
  if (type.getElementType().isInteger(64)) {
    if (options.allowsI64ToI32Conversion())
      return RankedTensorType::Builder(rtt).setElementType(i32Type);
    return std::nullopt;
  }

  // Handle index type.
  if (type.getElementType().isIndex())
    return RankedTensorType::Builder(rtt).setElementType(i32Type);

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// TensorRT Derived Conversion Pattern Rewriters
//===----------------------------------------------------------------------===//

TypedValue<RankedTensorType>
ConvertToTensorRTPattern::castTensor(RewriterBase &rewriter,
                                     Type newTypeOrElementType,
                                     TypedValue<RankedTensorType> src) {
  Type newElementType = mlir::getElementTypeOrSelf(newTypeOrElementType);
  if (newElementType == src.getType().getElementType())
    return src;
  Type newType =
      RankedTensorType::Builder(src.getType().cast<RankedTensorType>())
          .setElementType(newElementType);
  return rewriter.create<tensorrt::IdentityOp>(src.getLoc(), newType, src)
      .getResult();
}
