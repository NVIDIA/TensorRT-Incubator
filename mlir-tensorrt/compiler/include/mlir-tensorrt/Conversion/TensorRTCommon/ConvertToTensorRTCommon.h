//===- ConvertToTensorRTCommon.h --------------------------------*- C++ -*-===//
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
/// Declares a dialect conversion target, type converter, and pattern rewrite
/// classes for converting **to** the TensorRT dialect from other high-level
/// dialects (e.g. MHLO, StableHLO, etc).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_TENSORRTCOMMON_CONVERTTOTENSORRTCOMMON_H
#define MLIR_TENSORRT_CONVERSION_TENSORRTCOMMON_CONVERTTOTENSORRTCOMMON_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// TensorRT Conversion Options
//===----------------------------------------------------------------------===//

/// Encapsulates options that can be chosen by the caller when lowering to
/// TensorRT from another high-level dialect (e.g. StableHlo, etc).
struct LowerToTensorRTOptions {
  LowerToTensorRTOptions() = default;

  /// Specifies what to do when encountering an `i64` type.
  enum class I64Lowering {
    /// Convert all operations producing/consuming `i64` to tensors to act on
    /// `i32` tensors. Essentially this results in a function-wide replacement
    /// of i64 with i32. Operations such as `func.func` also have their `i64`
    /// tensor argmuments converted to i32.
    /// TODO: allow for special handling of function args (e.g. retain a special
    /// cast), since functions are ABI boundaries.
    CastI64ToI32 = 0,
    /// Fail when encountering an i64 tensor type. This is the default and
    /// should be used when the user is not sure of whether i64 to i32
    /// conversion would be ok.
    FailOnI64
  };

  LowerToTensorRTOptions &setI64Lowering(I64Lowering howToLowerI64) {
    this->i64Lowering = howToLowerI64;
    return *this;
  }

  /// Returns true if the options allow for global i64 to i32 replacement.
  bool allowsI64ToI32Conversion() const {
    return i64Lowering == I64Lowering::CastI64ToI32;
  }

private:
  I64Lowering i64Lowering = I64Lowering::FailOnI64;
};

//===----------------------------------------------------------------------===//
// TensorRT Conversion Target
//===----------------------------------------------------------------------===//

class TensorRTTypeConverter;

/// A conversion target that populates functions and other information useful
/// for converting from different sources (e.g. MHLO, or another
/// high-level dialect) to the TensorRT dialect.
class TensorRTConversionTarget : public ConversionTarget {
public:
  explicit TensorRTConversionTarget(MLIRContext &ctx,
                                    TensorRTTypeConverter &converter);
};

//===----------------------------------------------------------------------===//
// TensorRT Type Converter
//===----------------------------------------------------------------------===//

/// Supports type conversion during conversion to the TensorRT dialect from
/// another high-level dialect (e.g. MHLO, etc).
class TensorRTTypeConverter : public TypeConverter {
public:
  /// Create an LLVMTypeConverter using custom LowerToLLVMOptions. Optionally
  /// takes a data layout analysis to use in conversions.
  TensorRTTypeConverter(MLIRContext *ctx,
                        const LowerToTensorRTOptions &options);

  using TypeConverter::convertType;

  const LowerToTensorRTOptions &getOptions() const { return options; }

  bool isLegalTensorType(TensorType type);

private:
  /// For a given tensor type, return an equivalent legal TensorRT type, if
  /// possible. Otherwise returns nullptr.
  std::optional<Type> convertTensorType(TensorType type);

  LowerToTensorRTOptions options;
};

//===----------------------------------------------------------------------===//
// TensorRT Derived Conversion Pattern Rewriters
//===----------------------------------------------------------------------===//

/// A derived ConversionPattern that also allows a variety of helper methods to
/// be accessed from within `matchAndRewrite` functions.
class ConvertToTensorRTPattern : public ConversionPattern {
protected:
  /// Construct a conversion pattern with the given TensorRT-specific type
  /// converter, and forward the remaining arguments to ConversionPattern.
  template <typename... Args>
  explicit ConvertToTensorRTPattern(TensorRTTypeConverter &typeConverter,
                                    Args &&...args)
      : ConversionPattern(typeConverter, std::forward<Args>(args)...) {}

  // The below methods are for convenience use within the `matchAndRewrite`
  // function of derived patterns.

  /// Create a `tensorrt.identity` to cast tensor-typed value from one element
  /// type to another. If `newTypeOrElementType` is a tensor type, then only the
  /// element type is used.
  static TypedValue<RankedTensorType>
  castTensor(RewriterBase &rewriter, Type newTypeOrElementType,
             TypedValue<RankedTensorType> src);

  /// Overides base class templated method to get TensorRT type converter.
  const TensorRTTypeConverter *getTypeConverter() const {
    return ConversionPattern::getTypeConverter<TensorRTTypeConverter>();
  }
};

/// Wrapper that allows declaring a specific source operation in the template
/// parameter. Same as `OpConversionPattern` but specific to TensorRT dialect.
/// When instantiated, a TensorRT type converter must be passed.
template <typename SourceOp>
class ConvertOpToTensorRTPattern : public ConvertToTensorRTPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  ConvertOpToTensorRTPattern(TensorRTTypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : ConvertToTensorRTPattern(typeConverter, SourceOp::getOperationName(),
                                 benefit, context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    if constexpr (SourceOp::hasProperties())
      return rewrite(cast<SourceOp>(op),
                     OpAdaptor(operands, op->getAttrDictionary(),
                               cast<SourceOp>(op).getProperties()),
                     rewriter);
    rewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()),
            rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if constexpr (SourceOp::hasProperties())
      return matchAndRewrite(cast<SourceOp>(op),
                             OpAdaptor(operands, op->getAttrDictionary(),
                                       cast<SourceOp>(op).getProperties()),
                             rewriter);
    return matchAndRewrite(cast<SourceOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual LogicalResult match(SourceOp op) const {
    (void)op;
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual void rewrite(SourceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    (void)op;
    (void)adaptor;
    (void)rewriter;
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, adaptor, rewriter);
    return success();
  }

private:
  using ConversionPattern::matchAndRewrite;
};
} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_TENSORRTCOMMON_CONVERTTOTENSORRTCOMMON_H
