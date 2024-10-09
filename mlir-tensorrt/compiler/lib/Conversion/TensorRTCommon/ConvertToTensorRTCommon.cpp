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
         elType.isBF16() || elType.isInteger(4) || elType.isSignlessInteger(64);
}

std::optional<Type> TensorRTTypeConverter::convertTensorType(TensorType type) {
  if (isLegalTensorType(type))
    return type;

  auto rtt = cast<RankedTensorType>(type);
  Type i32Type = IntegerType::get(type.getContext(), 32);

  // Handle index type.
  if (type.getElementType().isIndex())
    return RankedTensorType::Builder(rtt).setElementType(i32Type);

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// TensorRT Derived Conversion Pattern Rewriters
//===----------------------------------------------------------------------===//

FailureOr<TypedValue<RankedTensorType>> ConvertToTensorRTPattern::castTensor(
    TensorRTConversionPatternRewriter &rewriter, int64_t trtMajorVersion,
    Type newTypeOrElementType, TypedValue<RankedTensorType> src) {
  Type newElementType = mlir::getElementTypeOrSelf(newTypeOrElementType);
  if (newElementType == src.getType().getElementType())
    return src;
  Type newType =
      RankedTensorType::Builder(cast<RankedTensorType>(src.getType()))
          .setElementType(newElementType);
  auto identityOp = rewriter.checkAndCreate<tensorrt::IdentityOp>(
      src.getLoc(), trtMajorVersion, newType, src);
  if (!identityOp)
    return failure();
  return identityOp.getResult();
}
