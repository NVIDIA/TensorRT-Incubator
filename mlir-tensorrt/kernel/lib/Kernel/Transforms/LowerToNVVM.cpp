//===- LowerToNVVM.cpp ----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Utils/CUDAUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-lower-to-nvvm"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_LOWERTONVVMPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

static void configureNVVMTarget(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();

  target.addDynamicallyLegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op,
                               LLVM::FAbsOp, LLVM::FCeilOp, LLVM::FFloorOp,
                               LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op,
                               LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>(
      [](Operation *op) -> bool {
        auto vecType = dyn_cast<VectorType>(op->getResultTypes().front());
        return vecType && vecType.getRank() == 1;
      });
}

/// Map gpu address space to the corresponding NVVM enum value as an integer.
static unsigned gpuAddressSpaceToNVVMMemorySpace(gpu::AddressSpace space) {
  switch (space) {
  case gpu::AddressSpace::Global:
    return static_cast<unsigned>(NVVM::NVVMMemorySpace::Global);
  case gpu::AddressSpace::Workgroup:
    return static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared);
  case gpu::AddressSpace::Private:
    return 0;
  }
  llvm_unreachable("unknown address space enum value");
}

namespace {

struct ComplexBitcast : public OpRewritePattern<complex::BitcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    // Cast to complex.
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    if (operand.getType().isSignlessInteger() &&
        isa<ComplexType>(op.getType())) {

      Type integerPartType =
          rewriter.getIntegerType(cast<ComplexType>(op.getType())
                                      .getElementType()
                                      .getIntOrFloatBitWidth());

      Value realPart =
          rewriter.create<arith::TruncIOp>(loc, integerPartType, operand);
      Value shift = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(
                   operand.getType(), integerPartType.getIntOrFloatBitWidth()));
      Value imagPart = rewriter.create<arith::ShRUIOp>(loc, operand, shift);
      imagPart =
          rewriter.create<arith::TruncIOp>(loc, integerPartType, imagPart);

      auto castToElementType = [&](Value v) -> Value {
        return rewriter.create<arith::BitcastOp>(
            loc, cast<ComplexType>(op.getType()).getElementType(), v);
      };

      rewriter.replaceOpWithNewOp<complex::CreateOp>(
          op, op.getType(), castToElementType(realPart),
          castToElementType(imagPart));
      return success();
    }

    // Cast from complex to int
    if (op.getType().isSignlessInteger() &&
        isa<ComplexType>(operand.getType())) {

      Type integerPartType =
          rewriter.getIntegerType(cast<ComplexType>(operand.getType())
                                      .getElementType()
                                      .getIntOrFloatBitWidth());

      Value realPart = rewriter.create<complex::ReOp>(op.getLoc(), operand);
      Value imagPart = rewriter.create<complex::ImOp>(op.getLoc(), operand);

      realPart =
          rewriter.create<arith::BitcastOp>(loc, integerPartType, realPart);
      imagPart =
          rewriter.create<arith::BitcastOp>(loc, integerPartType, imagPart);

      realPart = rewriter.create<arith::ExtUIOp>(loc, op.getType(), realPart);
      imagPart = rewriter.create<arith::ExtUIOp>(loc, op.getType(), imagPart);

      Value shiftSize = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(
                   op.getType(), integerPartType.getIntOrFloatBitWidth()));
      imagPart = rewriter.create<arith::ShLIOp>(loc, imagPart, shiftSize);
      rewriter.replaceOpWithNewOp<arith::OrIOp>(op, realPart, imagPart);
      return success();
    }

    return failure();
  }
};

template <typename T>
struct MTRTArithToLLVMConverter : public ConvertOpToLLVMPattern<T> {
  MTRTArithToLLVMConverter(const std::optional<StringRef> smVersion,
                           const int32_t ptxVersion,
                           const LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = 10)
      : ConvertOpToLLVMPattern<T>(typeConverter, benefit), smVersion(smVersion),
        ptxVersion(ptxVersion) {}

  /// Creates bitcast op to cast `source` to `targetType`.
  /// Returns failure if number of bits doesn't match.
  FailureOr<Value> createBitcastOp(RewriterBase &rewriter, Location loc,
                                   Value source, Type targetType) const {
    auto sourceType = getElementTypeOrSelf(source.getType());
    // Make sure bitwidth matches.
    if (sourceType.getIntOrFloatBitWidth() !=
        targetType.getIntOrFloatBitWidth())
      return failure();
    return rewriter.create<LLVM::BitcastOp>(loc, targetType, source).getRes();
  }

  /// Creates zext op to extend `source` to `targetType`, by prepending 0 bits
  /// (sign of source is not considered). Returns failure if target bitwidth is
  /// smaller than source.
  FailureOr<Value> createZextOp(RewriterBase &rewriter, Location loc,
                                Value source, Type targetType) const {
    auto sourceType = getElementTypeOrSelf(source.getType());
    // Target type bitwidth must be greater than or equal to source type
    // bitwidth.
    if (targetType.getIntOrFloatBitWidth() < sourceType.getIntOrFloatBitWidth())
      return failure();
    return rewriter.create<LLVM::ZExtOp>(loc, targetType, source).getRes();
  }

  /// Creates trunc op to truncate `source` to `targetType`, by removing MSBs.
  /// Returns failure if target bitwidth is greater than source.
  FailureOr<Value> createTruncOp(RewriterBase &rewriter, Location loc,
                                 Value source, Type targetType) const {
    auto sourceType = getElementTypeOrSelf(source.getType());
    // Target type bitwidth must be less than or equal to source type
    // bitwidth.
    if (targetType.getIntOrFloatBitWidth() > sourceType.getIntOrFloatBitWidth())
      return failure();
    return rewriter.create<LLVM::TruncOp>(loc, targetType, source).getRes();
  }

  /// Creates inline ASM op for given `ptxInstr`.
  Value createInlineAsmOp(RewriterBase &rewriter, Location loc,
                          StringRef ptxInstr, ArrayRef<Value> inputs,
                          Type resultType, StringRef constraints) const {
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);
    return LLVM::InlineAsmOp::create(rewriter, loc, resultType, inputs,
                                     ptxInstr, constraints,
                                     /*has_side_effects=*/false,
                                     /*is_align_stack=*/false,
                                     LLVM::TailCallKind::None, asmDialectAttr,
                                     /*operand_attrs=*/mlir::ArrayAttr())
        .getRes();
  }

  /// Converts `sm_x` SM version string to integer SM number `x` and returns it.
  /// Returns `failure()` otherwise.
  FailureOr<int32_t> getIntegerSMVersion() const {
    if (!smVersion)
      return failure();
    int32_t intSMVersion;
    if ((*smVersion).substr(3).getAsInteger(10, intSMVersion))
      return failure();
    return intSMVersion;
  }

  /// Returns `success()` if PTX cvt.* instruction with e4m3x2 is valid for
  /// given SM and PTX version, else returns `failure()`.
  LogicalResult isCvtWithFp8x2Valid() const {
    auto intSMVersion = getIntegerSMVersion();
    if (failed(intSMVersion))
      return failure();
    int32_t sm = *intSMVersion;
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    // Rule 1: For sm_90 or higher
    if (sm >= 90)
      return ptxVersion >= 78 ? success() : failure();
    // Rule 2: For sm_89
    if (sm == 89)
      return ptxVersion >= 81 ? success() : failure();
    return failure();
  }

  /// Returns `success()` if PTX cvt.* instruction with e2m1x2 is valid for
  /// given SM and PTX version, else returns `failure()`.
  LogicalResult isCvtWithFp4x2Valid() const {
    auto intSMVersion = getIntegerSMVersion();
    if (failed(intSMVersion))
      return failure();
    int32_t sm = *intSMVersion;
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    if (sm >= 100)
      return ptxVersion >= 86 ? success() : failure();
    return failure();
  }

protected:
  std::optional<StringRef> smVersion;
  int32_t ptxVersion;
};

//===----------------------------------------------------------------------===//
// Converters for `arith.truncf` - f16 -> fp4/fp8 only
//===----------------------------------------------------------------------===//

/// Convert `arith.truncf` to `llvm.inline_asm` op with PTX ASM, if source
/// type is `f16` and destination type is `f8E4M3FN`.
///
/// PTX instruction used is: cvt.rn.satfinite.e4m3x2.f16x2 d, a;
///
/// At this point, we are dealing with a single `arith.truncf` op thus lower
/// 16 bits of `a` contains source. Output register `d` has type `.b32` and
/// lower 16 bits contains result.
struct ArithTruncfF16ToF8E4M3FNConverter
    : public MTRTArithToLLVMConverter<arith::TruncFOp> {
  ArithTruncfF16ToF8E4M3FNConverter(const std::optional<StringRef> smVersion,
                                    const int32_t ptxVersion,
                                    const LLVMTypeConverter &typeConverter)
      : MTRTArithToLLVMConverter<arith::TruncFOp>(smVersion, ptxVersion,
                                                  typeConverter) {}

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(isCvtWithFp8x2Valid()))
      return rewriter.notifyMatchFailure(
          op, " SM and/or PTX version is not valid for this pattern to apply.");

    Type operandType = op.getIn().getType();
    Type resultType = op.getOut().getType();
    if (!isa<Float16Type>(operandType) || !isa<Float8E4M3FNType>(resultType))
      return rewriter.notifyMatchFailure(
          op, " only f16 -> f8E4M3FN case is handled.");

    // Bitcast f16 to i16 and zero-extend to i32.
    auto bitcastToI16 = createBitcastOp(rewriter, op->getLoc(), adaptor.getIn(),
                                        rewriter.getIntegerType(16));
    if (failed(bitcastToI16))
      return rewriter.notifyMatchFailure(
          op, " failed to create f16 -> i16 bitcast");

    auto ptxIn = createZextOp(rewriter, op->getLoc(), *bitcastToI16,
                              rewriter.getIntegerType(32));
    if (failed(ptxIn))
      return rewriter.notifyMatchFailure(op,
                                         " failed to create i16 -> i32 zext");

    // Call PTX instruction.
    Value ptxOut = createInlineAsmOp(
        rewriter, op->getLoc(), "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1;",
        {*ptxIn}, rewriter.getIntegerType(16), "=h,r");

    // Truncate to final size.
    auto truncated = createTruncOp(rewriter, op->getLoc(), ptxOut,
                                   rewriter.getIntegerType(8));
    if (failed(truncated))
      return rewriter.notifyMatchFailure(
          op, " failed to create truncation for quantized output");

    rewriter.replaceOp(op, *truncated);
    return success();
  }
};

/// Convert `arith.truncf` to `llvm.inline_asm` op with PTX ASM, if source
/// type is `f16` and destination type is `f4E2M1FN`.
///
/// PTX instruction used is: cvt.rn.satfinite.e2m1x2.f32 d, a, b;
///
/// NOTE: PTX doesn't have direct f16->fp4 conversion, so we need a workaround.
/// We'll use the f32 instruction with f16 extended to f32.
/// At this point, we are dealing with a single `arith.truncf` op thus `a` is
/// zero at `b` holds actual value to be converted to `e2m1`.
struct ArithTruncfF16ToF4E2M1FNConverter
    : public MTRTArithToLLVMConverter<arith::TruncFOp> {
  ArithTruncfF16ToF4E2M1FNConverter(const std::optional<StringRef> smVersion,
                                    const int32_t ptxVersion,
                                    const LLVMTypeConverter &typeConverter)
      : MTRTArithToLLVMConverter<arith::TruncFOp>(smVersion, ptxVersion,
                                                  typeConverter) {}

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(isCvtWithFp4x2Valid()))
      return rewriter.notifyMatchFailure(
          op, " SM and/or PTX version is not valid for this pattern to apply.");

    Type operandType = op.getIn().getType();
    Type resultType = op.getOut().getType();
    if (!isa<Float16Type>(operandType) || !isa<Float4E2M1FNType>(resultType))
      return rewriter.notifyMatchFailure(
          op, " only f16 -> f4E2M1FN case is handled.");

    // Extend f16 to f32
    Value f32Val = rewriter.create<LLVM::FPExtOp>(
        op->getLoc(), rewriter.getF32Type(), adaptor.getIn());

    // Create zero for first operand
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getF32Type(),
        APFloat::getZero(APFloat::IEEEsingle()));

    // Use explicit .b8 register to hold e2m1 result.
    // `mov.b16` packs .b8 register into lower 8 bits.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mov-2
    Value ptxOut = createInlineAsmOp(
        rewriter, op->getLoc(),
        "{\n"
        ".reg .b8 byte_result;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte_result, $1, $2;\n"
        "mov.b16 $0, {byte_result, 0};\n"
        "}",
        {zero, f32Val}, rewriter.getIntegerType(16), "=h,r,r");

    // Truncate to i4
    auto truncated = createTruncOp(rewriter, op->getLoc(), ptxOut,
                                   rewriter.getIntegerType(4));
    if (failed(truncated))
      return rewriter.notifyMatchFailure(
          op, " failed to create truncation for quantized output");

    rewriter.replaceOp(op, *truncated);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Converters for `arith.extf` - fp4/fp8 -> f16 only
//===----------------------------------------------------------------------===//

/// Convert `arith.extf` to `llvm.inline_asm` op with PTX ASM, if source
/// type is `f8E4M3FN` and destination type is `f16`.
///
/// PTX instruction used is: `cvt.rn.f16x2.e4m3x2 d,a;`
///
/// Source register `a` is of type `.b16`. At this point, we are dealing with
/// a single `arith.extf` op thus we always store 0 in upper 8 bits of source.
/// Output register `d` is of type `.b32` and lower 16 bit contains result.
struct ArithExtF8E4M3FNToF16Converter
    : public MTRTArithToLLVMConverter<arith::ExtFOp> {
  ArithExtF8E4M3FNToF16Converter(const std::optional<StringRef> smVersion,
                                 const int32_t ptxVersion,
                                 const LLVMTypeConverter &typeConverter)
      : MTRTArithToLLVMConverter<arith::ExtFOp>(smVersion, ptxVersion,
                                                typeConverter) {}

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(isCvtWithFp8x2Valid()))
      return rewriter.notifyMatchFailure(
          op, " SM and/or PTX version is not valid for this pattern to apply.");

    Type operandType = op.getIn().getType();
    Type resultType = op.getOut().getType();
    if (!isa<Float8E4M3FNType>(operandType) || !isa<Float16Type>(resultType))
      return rewriter.notifyMatchFailure(
          op, " only f8E4M3FN -> f16 case is handled.");

    // Zero-extend input to required size.
    auto ptxIn = createZextOp(rewriter, op->getLoc(), adaptor.getIn(),
                              rewriter.getIntegerType(16));
    if (failed(ptxIn))
      return rewriter.notifyMatchFailure(
          op, " failed to create zero extension for PTX input");

    // Call PTX instruction.
    Value ptxOut =
        createInlineAsmOp(rewriter, op->getLoc(), "cvt.rn.f16x2.e4m3x2 $0, $1;",
                          {*ptxIn}, rewriter.getIntegerType(32), "=r,h");

    // Truncate to i16 and bitcast to f16.
    auto truncated = createTruncOp(rewriter, op->getLoc(), ptxOut,
                                   rewriter.getIntegerType(16));
    if (failed(truncated))
      return rewriter.notifyMatchFailure(op,
                                         " failed to create i32 -> i16 trunc");

    auto bitcastOut = createBitcastOp(rewriter, op->getLoc(), *truncated,
                                      rewriter.getF16Type());
    if (failed(bitcastOut))
      return rewriter.notifyMatchFailure(
          op, " failed to create i16 -> f16 bitcast");

    rewriter.replaceOp(op, *bitcastOut);
    return success();
  }
};

/// Convert `arith.extf` to `llvm.inline_asm` op with PTX ASM, if source
/// type is `f4E2M1FN` and destination type is `f16`.
///
/// PTX instruction used is: `cvt.rn.f16x2.e2m1x2  d, a;`
///
/// Source register `a` is of type `.b8`. At this point, we are dealing with
/// a single `arith.extf` op thus we always store 0 in upper 4 bits of source.
/// Output register `d` is of type `.b32` and lower 16 bit contains result.
struct ArithExtF4E2M1FNToF16Converter
    : public MTRTArithToLLVMConverter<arith::ExtFOp> {
  ArithExtF4E2M1FNToF16Converter(const std::optional<StringRef> smVersion,
                                 const int32_t ptxVersion,
                                 const LLVMTypeConverter &typeConverter)
      : MTRTArithToLLVMConverter<arith::ExtFOp>(smVersion, ptxVersion,
                                                typeConverter) {}

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(isCvtWithFp4x2Valid()))
      return rewriter.notifyMatchFailure(
          op, " SM and/or PTX version is not valid for this pattern to apply.");

    Type operandType = op.getIn().getType();
    Type resultType = op.getOut().getType();
    if (!isa<Float4E2M1FNType>(operandType) || !isa<Float16Type>(resultType))
      return rewriter.notifyMatchFailure(
          op, " only f4E2M1FN -> f16 case is handled.");

    // Zero-extend input to i16 for the constraint system
    auto ptxIn = createZextOp(rewriter, op->getLoc(), adaptor.getIn(),
                              rewriter.getIntegerType(16));
    if (failed(ptxIn))
      return rewriter.notifyMatchFailure(
          op, " failed to create zero extension for PTX input");

    // Call PTX instruction with explicit .b8 register.
    Value ptxOut =
        createInlineAsmOp(rewriter, op->getLoc(),
                          "{\n"
                          ".reg .b8 byte_input;\n"
                          "cvt.u8.u16 byte_input, $1;\n"
                          "cvt.rn.f16x2.e2m1x2 $0, byte_input;\n"
                          "}",
                          {*ptxIn}, rewriter.getIntegerType(32), "=r,h");

    // Truncate to i16 and bitcast to f16
    auto truncated = createTruncOp(rewriter, op->getLoc(), ptxOut,
                                   rewriter.getIntegerType(16));
    if (failed(truncated))
      return rewriter.notifyMatchFailure(op,
                                         " failed to create i32 -> i16 trunc");

    auto bitcastOut = createBitcastOp(rewriter, op->getLoc(), *truncated,
                                      rewriter.getF16Type());
    if (failed(bitcastOut))
      return rewriter.notifyMatchFailure(
          op, " failed to create i16 -> f16 bitcast");

    rewriter.replaceOp(op, *bitcastOut);
    return success();
  }
};

class LowerToNVVMPass
    : public kernel::impl::LowerToNVVMPassBase<LowerToNVVMPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Lower SCF to ControlFlow.
    if (!preserveStructuredControlFlow) {
      RewritePatternSet patterns(ctx);
      populateSCFToControlFlowConversionPatterns(patterns);
      ConversionTarget target(getContext());
      target.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                          scf::IndexSwitchOp, scf::ParallelOp, scf::WhileOp,
                          scf::ExecuteRegionOp>();
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns)))) {
        emitError(getOperation()->getLoc())
            << "failed to lower SCF to ControlFlow in " << getArgument();
        return signalPassFailure();
      }
    }

    // First remove vector leading dims.
    {
      RewritePatternSet patterns(ctx);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      patterns.add<ComplexBitcast>(ctx);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }

    // Second, a set of rewrites are executed to prepare the IR for lowering
    // to LLVM. Many dialects such as Affine, Math, and Arith define
    // "expansion" passes that decompose more sophisticated or auxiliary
    // operations into simpler operations that can be lowered to LLVM.
    // Therefore, we combine these rewrites together and execute them here
    // to avoid having to put a separate pass into the pipeline.
    {
      RewritePatternSet patterns(ctx);
      populateAffineToStdConversionPatterns(patterns);
      arith::populateArithExpandOpsPatterns(patterns);
      affine::populateAffineExpandIndexOpsPatterns(patterns);

      vector::VectorTransformsOptions vectorTransformsOptions;
      vectorTransformsOptions.setVectorTransposeLowering(
          vector::VectorTransposeLowering::EltWise);

      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBitCastLoweringPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns, vectorTransformsOptions.vectorContractLowering);
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      vector::populateVectorInterleaveLoweringPatterns(patterns);
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vectorTransformsOptions.vectorTransposeLowering);
      // Vector transfer ops with rank > 1 should be lowered with
      // VectorToSCF.
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     /*maxTransferRank=*/1);
      vector::populateScalarVectorTransferLoweringPatterns(
          patterns,
          /*benefit=*/1,
          /*allowMultipleUses=*/
          false);
      vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
      vector::populateVectorStepLoweringPatterns(patterns);
      vector::populateVectorRankReducingFMAPattern(patterns);
      mlir::populateGpuShufflePatterns(patterns);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }

    LLVM_DEBUG(
        DBGS() << "before conversion to NVVM, after preparatory rewrites:\n"
               << *getOperation() << "\n");

    // Second, the various conversion patterns from each of the allowed
    // dialects' `X-to-llvm` conversion passes defined upstream are combined
    // into a conversion rewrite set. This conversion is a "full" conversion
    // and will fail if any ops or types in the input IR are not fully
    // legalized.
    {
      LLVMConversionTarget target(getContext());
      target.addLegalOp<gpu::GPUModuleOp>();
      configureNVVMTarget(target);
      LowerToLLVMOptions llvmOptions(&getContext());
      llvmOptions.useBarePtrCallConv = true;
      LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
      typeConverter.addConversion([&](FloatType type) -> std::optional<Type> {
        if (isa<Float8E4M3FNType>(type))
          return IntegerType::get(type.getContext(),
                                  type.getIntOrFloatBitWidth());
        if (isa<Float4E2M1FNType>(type))
          return IntegerType::get(type.getContext(),
                                  type.getIntOrFloatBitWidth());
        return std::nullopt;
      });

      // Populate the memref address space conversion function.
      typeConverter.addTypeAttributeConversion(
          [](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
            gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
            unsigned addressSpace =
                gpuAddressSpaceToNVVMMemorySpace(memorySpace);
            return IntegerAttr::get(
                IntegerType::get(memorySpaceAttr.getContext(), 64),
                addressSpace);
          });

      // Populate patterns
      RewritePatternSet patterns(&getContext());

      // Populate MLIR-TensorRT written arith->LLVM patterns.
      gpu::GPUModuleOp gpuModuleOp = cast<gpu::GPUModuleOp>(getOperation());
      int32_t ptxVersion = getHighestPTXVersion();
      patterns.add<
          ArithExtF8E4M3FNToF16Converter, ArithExtF4E2M1FNToF16Converter,
          ArithTruncfF16ToF8E4M3FNConverter, ArithTruncfF16ToF4E2M1FNConverter>(
          getUniqueTargetChip(gpuModuleOp), ptxVersion, typeConverter);

      arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
      // Prefer GPU-to-NVVM patterns over Math-to-LLVM patterns since the
      // GPU-to-NVVM patterns map to libdevice calls.
      populateGpuToNVVMConversionPatterns(typeConverter, patterns,
                                          /*benefit=*/10);
      populateMathToLLVMConversionPatterns(typeConverter, patterns,
                                           /*approximateLog1p=*/true,
                                           /*benefit=*/1);
      populateNVGPUToNVVMConversionPatterns(typeConverter, patterns);
      cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
      populateVectorToLLVMConversionPatterns(typeConverter, patterns,
                                             /*reassociateFPReductions=*/true,
                                             /*force32BitVectorIndices=*/true);
      populateComplexToLLVMConversionPatterns(typeConverter, patterns);
      ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

      // Populate structural patterns for preserving structured control
      // flow.
      if (preserveStructuredControlFlow)
        scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        emitError(getOperation()->getLoc())
            << "failed to apply patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // Since we use `func.func` to represent kernel entrypoints, we need to
    // manually translate the GPU kernel entrypoint annotation to NVVM. Upstream
    // patterns only cover `gpu.func`.
    getOperation()->walk([](LLVM::LLVMFuncOp func) {
      if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
        func->removeAttr(gpu::GPUDialect::getKernelFuncAttrName());
        func->setAttr(NVVM::NVVMDialect::getKernelFuncAttrName(),
                      UnitAttr::get(func->getContext()));
      }
    });
  }
};
} // namespace
