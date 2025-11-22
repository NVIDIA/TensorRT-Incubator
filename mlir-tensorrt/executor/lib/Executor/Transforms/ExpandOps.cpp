//===- ExpandOps.cpp ------------------------------------------------------===//
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
/// Implementation of the `executor-expand-ops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTOREXPANDOPSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

/// Return the value for `ofr` or create a constant value if required. If `ofr`
/// is a value, its type is checked against `intType`. If it is an attribute,
/// then it is checked that it can be losslessly converted to `intType`.
static FailureOr<Value> getOrCreateAndCheckIndexValue(RewriterBase &rewriter,
                                                      GetOffsetOp op,
                                                      IntegerType intType,
                                                      OpFoldResult ofr) {
  Location loc = op.getLoc();
  if (auto val = dyn_cast<Value>(ofr)) {
    if (val.getType() != intType)
      return failure();
    return val;
  }

  IntegerAttr srcAttr = cast<IntegerAttr>(cast<Attribute>(ofr));
  APInt srcInt = srcAttr.getValue();
  if (srcInt.getBitWidth() == intType.getWidth())
    return rewriter
        .create<ConstantOp>(loc, rewriter.getIntegerAttr(intType, srcInt))
        .getResult();

  if (srcInt.getBitWidth() < intType.getWidth())
    return rewriter
        .create<ConstantOp>(loc, rewriter.getIntegerAttr(
                                     intType, srcInt.zext(intType.getWidth())))
        .getResult();

  if (!srcInt.isIntN(intType.getWidth()))
    return failure();

  return rewriter
      .create<ConstantOp>(loc, rewriter.getIntegerAttr(
                                   intType, srcInt.trunc(intType.getWidth())))
      .getResult();
}

/// Lower the `executor.getoffset` operation into more primitive ops.
static FailureOr<Value> lowerGetOffset(RewriterBase &rewriter,
                                       const DataLayout &layout,
                                       GetOffsetOp op) {
  SmallVector<OpFoldResult> indices = op.getIndices();
  Location loc = op.getLoc();

  // The type we should use for index calculations.
  IntegerType computeType = rewriter.getIntegerType(
      layout.getTypeSizeInBits(rewriter.getIndexType()));

  if (computeType != op.getType())
    return rewriter.notifyMatchFailure(
        op, llvm::formatv("result type ({0}) does not match the width of the "
                          "IndexType ({1}) specified by the DataLayout",
                          op.getType(), computeType)
                .str());

  auto getIndexConst = [&](int64_t value) -> Value {
    return rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(computeType, value));
  };

  FailureOr<Value> indexValue =
      getOrCreateAndCheckIndexValue(rewriter, op, computeType, indices[0]);
  if (failed(indexValue))
    return rewriter.notifyMatchFailure(
        op,
        llvm::formatv(
            "index #0 ({0}) cannot be converted losslessly to IndexType ({1})",
            indices[0], computeType)
            .str());

  Value offset = rewriter.create<MulIOp>(
      loc, *indexValue, getIndexConst(layout.getTypeSize(op.getElemType())));

  Type currentType = op.getElemType();
  for (OpFoldResult index : llvm::drop_begin(indices)) {
    if (auto structType = dyn_cast<TableType>(currentType)) {
      ArrayRef<Type> body = structType.getBody();
      // This is a plain cast since the verifier checks that indices into
      // aggregates are constants.
      IntegerAttr indexStatic = llvm::cast<IntegerAttr>(cast<Attribute>(index));
      assert(static_cast<unsigned>(indexStatic.getInt()) < body.size() &&
             "getoffset index is out-of-bounds for indexed aggregate");
      for (int64_t i = 0, e = indexStatic.getInt(); i < e; i++) {
        llvm::TypeSize typeSize = layout.getTypeSize(body[i]);
        IntegerAttr alignment =
            rewriter.getUI32IntegerAttr(layout.getTypeABIAlignment(body[i]));
        offset = rewriter.create<AlignToOp>(loc, offset, alignment);
        offset = rewriter.create<AddIOp>(loc, offset, getIndexConst(typeSize));
      }
      IntegerAttr alignment = rewriter.getUI32IntegerAttr(
          layout.getTypeABIAlignment(body[indexStatic.getInt()]));
      offset = rewriter.create<AlignToOp>(loc, offset, alignment);
      currentType = body[indexStatic.getInt()];
      continue;
    }

    // This could also be an assertion. If this occurs then the the op should be
    // invalid.
    return op->emitOpError("failed to lower invalid executor.getoffset op");
  }
  return offset;
}

namespace {

/// Lowers `executor.getoffset` by creating more primitive arithmetic
/// operations. May also produce `executor.alignto`.
struct LowerGetOffsetPattern : public OpRewritePattern<GetOffsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerGetOffsetPattern(const DataLayout &dataLayout, MLIRContext *ctx,
                        PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(GetOffsetOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> offset = lowerGetOffset(rewriter, dataLayout, op);
    if (failed(offset)) {
      return failure();
    }

    rewriter.replaceOp(op, *offset);
    return success();
  }

  const DataLayout &dataLayout;
};

/// Lowers `executor.alloca` by replacing with a normal allocation and adding a
/// dealloc at the end of the block.
struct LowerAllocaPattern : public OpRewritePattern<AllocaOp> {

  LowerAllocaPattern(const DataLayout &dataLayout, MLIRContext *ctx,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(AllocaOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto indexType = rewriter.getIntegerType(
        dataLayout.getTypeSizeInBits(rewriter.getIndexType()));
    Value numBytes = rewriter.create<GetOffsetOp>(
        loc, indexType, op.getElementType(),
        ArrayRef<OpFoldResult>{op.getNumElements()});
    Value alignment = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, dataLayout.getTypeABIAlignment(
                                                    op.getElementType())));
    Value alloc = rewriter.create<AllocateOp>(
        loc, PointerType::get(rewriter.getContext(), MemoryType::host),
        numBytes, alignment);
    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    rewriter.create<executor::DeallocateOp>(loc, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }

  const DataLayout &dataLayout;
};

/// Base class for lowering operations with unsupported types (e.g., f4E2M1FN).
///
/// This class provides a collection of helper methods for generating primitive
/// executor operations when lowering complex operations that involve
/// unsupported types. The helpers facilitate bit manipulation, comparisons, and
/// control flow operations needed to implement type conversions and arithmetic
/// emulation.
///
/// The pattern is used primarily to support sub-byte types such as f4E2M1FN,
/// where operations must be decomposed into sequences of integer operations
/// since the runtime doesn't natively support 4-bit floating-point types.
template <typename T>
struct LowerUnsupportedTypeOpsPattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  Value createIntegerConstant(RewriterBase &rewriter, Location loc,
                              uint64_t bitwidth, int64_t value) const {
    return rewriter.create<executor::ConstantOp>(
        loc, rewriter.getIntegerType(bitwidth),
        IntegerAttr::get(rewriter.getIntegerType(bitwidth), value));
  }

  Value createBoolConstant(RewriterBase &rewriter, Location loc,
                           bool value) const {
    return rewriter.create<executor::ConstantOp>(
        loc, rewriter.getI1Type(),
        IntegerAttr::get(rewriter.getI1Type(), value ? 1 : 0));
  }

  /// Creates a chain of select operations to implement multi-way branching.
  /// The conditions are evaluated in reverse order (last to first).
  Value createSelectChain(RewriterBase &rewriter, Location loc,
                          ArrayRef<std::pair<Value, Value>> conditionsAndValues,
                          Value defaultValue) const {
    Value result = defaultValue;
    for (auto it = conditionsAndValues.rbegin();
         it != conditionsAndValues.rend(); ++it) {
      result = rewriter.create<executor::SelectOp>(loc, it->first, it->second,
                                                   result);
    }
    return result;
  }

  /// Extracts and tests the sign bit from a bit representation.
  /// Returns true if the sign bit is set (negative).
  Value extractSignBit(RewriterBase &rewriter, Location loc, Value bits,
                       Value shiftAmount) const {
    Value signBit =
        rewriter.create<executor::ShiftRightLogicalIOp>(loc, bits, shiftAmount);
    return rewriter.create<executor::ICmpOp>(
        loc, signBit,
        createIntegerConstant(rewriter, loc,
                              signBit.getType().getIntOrFloatBitWidth(), 0),
        executor::ICmpType::ne);
  }

  /// Extracts bits using a mask and optional right shift.
  /// If shiftAmount is null, no shift is performed.
  Value extractBits(RewriterBase &rewriter, Location loc, Value value,
                    Value mask, Value shiftAmount) const {
    Value masked = rewriter.create<executor::BitwiseAndIOp>(loc, value, mask);
    if (shiftAmount) {
      return rewriter.create<executor::ShiftRightLogicalIOp>(loc, masked,
                                                             shiftAmount);
    }
    return masked;
  }

  Value createEqualityCheck(
      RewriterBase &rewriter, Location loc, Value value, Value constant,
      executor::ICmpType cmpType = executor::ICmpType::eq) const {
    return rewriter.create<executor::ICmpOp>(loc, value, constant, cmpType);
  }

  /// Creates a select operation with a boolean constant as the false value.
  Value createConditionalBool(RewriterBase &rewriter, Location loc,
                              Value condition, Value ifTrue,
                              bool defaultValue) const {
    return rewriter.create<executor::SelectOp>(
        loc, condition, ifTrue,
        createBoolConstant(rewriter, loc, defaultValue));
  }

  Value bitcast(RewriterBase &rewriter, Location loc, Type targetType,
                Value value) const {
    return rewriter.create<executor::BitcastOp>(loc, targetType, value);
  }

  Value zeroExtend(RewriterBase &rewriter, Location loc, Type targetType,
                   Value value) const {
    return rewriter.create<executor::ZExtOp>(loc, targetType, value);
  }

  Value shiftLeft(RewriterBase &rewriter, Location loc, Value value,
                  Value shiftAmount) const {
    return rewriter.create<executor::ShiftLeftIOp>(loc, value, shiftAmount);
  }

  Value bitwiseOr(RewriterBase &rewriter, Location loc, Value lhs,
                  Value rhs) const {
    return rewriter.create<executor::BitwiseOrIOp>(loc, lhs, rhs);
  }

  Value select(RewriterBase &rewriter, Location loc, Value condition,
               Value trueValue, Value falseValue) const {
    return rewriter.create<executor::SelectOp>(loc, condition, trueValue,
                                               falseValue);
  }

  Value subtract(RewriterBase &rewriter, Location loc, Value lhs,
                 Value rhs) const {
    return rewriter.create<executor::SubIOp>(loc, lhs, rhs);
  }

  Value add(RewriterBase &rewriter, Location loc, Value lhs, Value rhs) const {
    return rewriter.create<executor::AddIOp>(loc, lhs, rhs);
  }
};

/// Executor runtime doesn't support `f4E2M1FN` type. This pattern
/// rewrites `executor.extf` op doing `f4E2M1FN->f16` conversion to
/// series of primitive executor ops running this conversion.
struct LowerF4E2M1FNToF16Extension
    : public LowerUnsupportedTypeOpsPattern<executor::ExtfOp> {
  using LowerUnsupportedTypeOpsPattern<
      executor::ExtfOp>::LowerUnsupportedTypeOpsPattern;

  LogicalResult matchAndRewrite(executor::ExtfOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<Float4E2M1FNType>(op.getOperand().getType()) ||
        !isa<Float16Type>(op.getResult().getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Pattern applies for f4E2M1FN -> f16 extension.");

    // Create necessary constants.
    Value c0I4 = createIntegerConstant(rewriter, op->getLoc(), 4, 0);
    Value c1I4 = createIntegerConstant(rewriter, op->getLoc(), 4, 1);
    Value c3I4 = createIntegerConstant(rewriter, op->getLoc(), 4, 3);
    Value c6I4 = createIntegerConstant(rewriter, op->getLoc(), 4,
                                       6); // Exponent mask: 0b0110

    Value c0I16 = createIntegerConstant(rewriter, op->getLoc(), 16, 0);
    Value c9I16 = createIntegerConstant(rewriter, op->getLoc(), 16,
                                        9); // Mantissa shift amount
    Value c10I16 = createIntegerConstant(rewriter, op->getLoc(), 16,
                                         10); // Exponent shift amount
    Value c14I16 =
        createIntegerConstant(rewriter, op->getLoc(), 16, 14); // f16_bias
    Value c15I16 = createIntegerConstant(rewriter, op->getLoc(), 16,
                                         15); // Sign shift amount
    // Bitcast f4 to i4
    Value i4 = bitcast(rewriter, op->getLoc(), rewriter.getIntegerType(4),
                       op.getOperand());

    // Extract sign bit (1 bit)
    Value shiftedI4 =
        rewriter.create<executor::ShiftRightLogicalIOp>(op->getLoc(), i4, c3I4);
    Value shiftedI16 = zeroExtend(rewriter, op->getLoc(),
                                  rewriter.getIntegerType(16), shiftedI4);
    Value I16Sign = shiftLeft(rewriter, op->getLoc(), shiftedI16, c15I16);

    // Extract exponent (2 bits)
    Value expI4 = extractBits(rewriter, op->getLoc(), i4, c6I4, c1I4);

    // Extract mantissa (1 bit)
    Value mantissaI4 = extractBits(rewriter, op->getLoc(), i4, c1I4, nullptr);
    Value mantissaI4I16 = zeroExtend(rewriter, op->getLoc(),
                                     rewriter.getIntegerType(16), mantissaI4);

    // Check if exponent is 0.
    // If exponent is 0, it is subnormal f4E2M1FN which is only 0.5.
    Value isExponentZero =
        createEqualityCheck(rewriter, op->getLoc(), expI4, c0I4);
    Value isMantissaZero =
        createEqualityCheck(rewriter, op->getLoc(), mantissaI4, c0I4);

    // Case 1: Normal Value
    // fp4 bias = 1, fp16 bias = 15
    // To get fp16 exponent, we first remove bias from fp4 exponent (exponent -
    // 1) and then add fp16 bias. fp16_exp = fp4_exp_unbiased + fp16_bias =
    // (fp4_exp - 1) + 15 = fp4_exp + 14 .
    Value expI4I16 =
        zeroExtend(rewriter, op->getLoc(), rewriter.getIntegerType(16), expI4);
    Value normalExpI16 = add(rewriter, op->getLoc(), expI4I16, c14I16);
    // f16 is 1 sign, 5 exponent and 10 mantissa bits.
    // Left shift exponent by 10 bits to put at correct position.
    Value normalExpI16Positioned =
        shiftLeft(rewriter, op->getLoc(), normalExpI16, c10I16);
    // Left shift fp4 mantissa (1 bit) by 9 bits so that it sits at MSB
    // position.
    Value normalManI16Positioned =
        shiftLeft(rewriter, op->getLoc(), mantissaI4I16, c9I16);

    // Case 2/3: Subnormal (exponent == 0 and mantissa != 0) and Zero (exponenet
    // == 0 and mantissa == 0). The only subnormal value of fp4 (0.5) falls in
    // fp16 normal range.
    // Normal representation in float is (-1)^sign × 2^(exponent - bias) × (1 +
    // mantissa). Considering normal representation, f16 for +/-0.5 is (-1)^sign
    // × 2^(-1) × (1 + 0)
    // For example,
    // 0.5 = 2^(-1) * (1 + 0)
    // i.e. exponenet - bias = -1
    // exponent - 15 (bias for f16) = -1
    // exponent = 14
    Value subnormalExpI16 =
        select(rewriter, op->getLoc(), isMantissaZero, c0I16, c14I16);
    Value subnormalExpI16Positioned =
        shiftLeft(rewriter, op->getLoc(), subnormalExpI16, c10I16);
    Value subnormalMantI16Positioned =
        createIntegerConstant(rewriter, op->getLoc(), 16, 0);

    // Final selection
    Value I16Exp = select(rewriter, op->getLoc(), isExponentZero,
                          subnormalExpI16Positioned, normalExpI16Positioned);
    Value I16Mant = select(rewriter, op->getLoc(), isExponentZero,
                           subnormalMantI16Positioned, normalManI16Positioned);
    Value signAddedExp = bitwiseOr(rewriter, op->getLoc(), I16Exp, I16Sign);
    Value resultI16 = bitwiseOr(rewriter, op->getLoc(), signAddedExp, I16Mant);
    rewriter.replaceOpWithNewOp<executor::BitcastOp>(op, rewriter.getF16Type(),
                                                     resultI16);
    return success();
  }
};

/// Executor runtime doesn't support `f4E2M1FN` type. This pattern
/// rewrites `executor.truncf` op doing `f16->f4E2M1FN` conversion to
/// series of primitive executor ops running this conversion.
struct LowerF16ToF4E2M1FNTruncation
    : public LowerUnsupportedTypeOpsPattern<executor::TruncfOp> {
  using LowerUnsupportedTypeOpsPattern<
      executor::TruncfOp>::LowerUnsupportedTypeOpsPattern;

  LogicalResult matchAndRewrite(executor::TruncfOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<Float16Type>(op.getOperand().getType()) ||
        !isa<Float4E2M1FNType>(op.getResult().getType()))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Pattern applies for f16 -> f4E2M1FN truncation.");

    Location loc = op->getLoc();
    Value fp16Val = op.getOperand();

    // I16 constants
    Value c0I16 = createIntegerConstant(rewriter, loc, 16, 0);
    Value c10I16 =
        createIntegerConstant(rewriter, loc, 16, 10); // Mantissa shift
    Value c13I16 = createIntegerConstant(rewriter, loc, 16, 13); // Exp for 0.25
    Value c14I16 = createIntegerConstant(rewriter, loc, 16, 14); // Exp for 0.5
    Value c15I16 =
        createIntegerConstant(rewriter, loc, 16, 15); // f16 bias & sign shift
    Value c16I16 = createIntegerConstant(rewriter, loc, 16, 16); // Exp for 2.0
    Value c17I16 = createIntegerConstant(rewriter, loc, 16, 17); // Exp for 4.0
    Value c31I16 =
        createIntegerConstant(rewriter, loc, 16, 31); // Max f16 exponent
    Value c256I16 = createIntegerConstant(rewriter, loc, 16, 256); // 2^8
    Value c512I16 = createIntegerConstant(rewriter, loc, 16, 512); // 2^9
    Value c768I16 = createIntegerConstant(rewriter, loc, 16, 768); // 3*256
    Value c1023I16 =
        createIntegerConstant(rewriter, loc, 16, 1023); // Mantissa mask
    Value c31744I16 =
        createIntegerConstant(rewriter, loc, 16, 31744); // Exp mask (0x7C00)

    // I4 constants for FP4 values
    Value c0I4 = createIntegerConstant(rewriter, loc, 4, 0); // 0
    Value c1I4 = createIntegerConstant(rewriter, loc, 4, 1); // 0.5 (subnormal)
    Value c2I4 = createIntegerConstant(rewriter, loc, 4, 2); // 1.0
    Value c3I4 = createIntegerConstant(rewriter, loc, 4, 3); // 1.5
    Value c4I4 = createIntegerConstant(rewriter, loc, 4, 4); // 2.0
    Value c5I4 = createIntegerConstant(rewriter, loc, 4, 5); // 3.0
    Value c6I4 = createIntegerConstant(rewriter, loc, 4, 6); // 4.0
    Value c7I4 =
        createIntegerConstant(rewriter, loc, 4, 7); // 6.0 (max positive)
    Value c15I4 = createIntegerConstant(
        rewriter, loc, 4, 15); // -6.0 (max negative in 2's complement)

    // --- Deconstruct FP16 ---
    // Bitcast f16 to i16
    Value fp16Bits =
        bitcast(rewriter, loc, rewriter.getIntegerType(16), fp16Val);

    // Extract sign bit
    Value signI1 = extractSignBit(rewriter, loc, fp16Bits, c15I16);

    // Extract exponent (5 bits)
    Value exp16 = extractBits(rewriter, loc, fp16Bits, c31744I16, c10I16);

    // Extract mantissa (10 bits)
    Value mant16 = extractBits(rewriter, loc, fp16Bits, c1023I16, nullptr);

    // --- Handle Special Values ---
    // Check for Inf/NaN (exponent all ones)
    Value isExpAllOnes = createEqualityCheck(rewriter, loc, exp16, c31I16);
    Value isMantZero = createEqualityCheck(rewriter, loc, mant16, c0I16);
    Value isInf =
        createConditionalBool(rewriter, loc, isExpAllOnes,
                              select(rewriter, loc, isMantZero,
                                     createBoolConstant(rewriter, loc, true),
                                     createBoolConstant(rewriter, loc, false)),
                              false);
    Value isNan =
        createConditionalBool(rewriter, loc, isExpAllOnes,
                              select(rewriter, loc, isMantZero,
                                     createBoolConstant(rewriter, loc, false),
                                     createBoolConstant(rewriter, loc, true)),
                              false);

    // Check for zero
    Value isExpZero = createEqualityCheck(rewriter, loc, exp16, c0I16);
    Value isZero =
        createConditionalBool(rewriter, loc, isExpZero,
                              select(rewriter, loc, isMantZero,
                                     createBoolConstant(rewriter, loc, true),
                                     createBoolConstant(rewriter, loc, false)),
                              false);

    // --- Map to FP4 values based on exponent and mantissa ---
    // For values < 0.25: round to 0
    Value isLt13 = createEqualityCheck(rewriter, loc, exp16, c13I16,
                                       executor::ICmpType::ult);

    // For exp=13 (0.25 to 0.5 range): round based on mantissa
    // 0.25 exactly (mant=0) rounds to 0.0 (tie-to-even)
    // Everything else in range (0.25, 0.5) rounds to 0.5
    Value isExp13 = createEqualityCheck(rewriter, loc, exp16, c13I16);
    Value mantEq0 = createEqualityCheck(rewriter, loc, mant16, c0I16);
    Value mantNe0 = createEqualityCheck(rewriter, loc, mant16, c0I16,
                                        executor::ICmpType::ne);
    Value round13ToZero =
        createConditionalBool(rewriter, loc, isExp13, mantEq0, false);
    Value round13ToHalf =
        createConditionalBool(rewriter, loc, isExp13, mantNe0, false);

    // For exp=14 (0.5 to 1.0 range)
    // 0.5 exactly: mant=0 -> FP4 0.5 (subnormal)
    // 0.75 (mant=512): rounds to 1.0
    Value isExp14 = createEqualityCheck(rewriter, loc, exp16, c14I16);
    Value mantLt512_exp14 = createEqualityCheck(rewriter, loc, mant16, c512I16,
                                                executor::ICmpType::ult);
    Value mantGe512_exp14 = select(rewriter, loc, mantLt512_exp14,
                                   createBoolConstant(rewriter, loc, false),
                                   createBoolConstant(rewriter, loc, true));
    Value round14ToHalf =
        createConditionalBool(rewriter, loc, isExp14, mantLt512_exp14, false);
    Value round14ToOne =
        createConditionalBool(rewriter, loc, isExp14, mantGe512_exp14, false);

    // For exp=15 (1.0 to 2.0 range)
    Value isExp15 = createEqualityCheck(rewriter, loc, exp16, c15I16);
    Value mantLt256 = createEqualityCheck(rewriter, loc, mant16, c256I16,
                                          executor::ICmpType::ult);
    Value mantEq256 = createEqualityCheck(rewriter, loc, mant16, c256I16);
    Value mantLt768 = createEqualityCheck(rewriter, loc, mant16, c768I16,
                                          executor::ICmpType::ult);
    Value mantEq768 = createEqualityCheck(rewriter, loc, mant16, c768I16);
    Value mantGt256 = createEqualityCheck(rewriter, loc, mant16, c256I16,
                                          executor::ICmpType::ugt);
    Value mantGt768 = createEqualityCheck(rewriter, loc, mant16, c768I16,
                                          executor::ICmpType::ugt);

    // Round to 1.0 if mant <= 256
    Value mantLe256 = rewriter.create<executor::SelectOp>(
        loc, mantLt256, createBoolConstant(rewriter, loc, true),
        rewriter.create<executor::SelectOp>(
            loc, mantEq256, createBoolConstant(rewriter, loc, true),
            createBoolConstant(rewriter, loc, false)));
    Value exp15To1p0 = rewriter.create<executor::SelectOp>(
        loc, isExp15, mantLe256, createBoolConstant(rewriter, loc, false));

    // Round to 1.5 if 256 < mant < 768
    Value mantBetween256And768 = rewriter.create<executor::SelectOp>(
        loc, mantGt256, mantLt768, createBoolConstant(rewriter, loc, false));
    Value exp15To1p5 = rewriter.create<executor::SelectOp>(
        loc, isExp15, mantBetween256And768,
        createBoolConstant(rewriter, loc, false));

    // Round to 2.0 if mant >= 768
    Value mantGe768 = rewriter.create<executor::SelectOp>(
        loc, mantGt768, createBoolConstant(rewriter, loc, true),
        rewriter.create<executor::SelectOp>(
            loc, mantEq768, createBoolConstant(rewriter, loc, true),
            createBoolConstant(rewriter, loc, false)));
    Value exp15To2p0 = rewriter.create<executor::SelectOp>(
        loc, isExp15, mantGe768, createBoolConstant(rewriter, loc, false));

    // For exp=16 (2.0 to 4.0 range)
    Value isExp16 = createEqualityCheck(rewriter, loc, exp16, c16I16);
    Value exp16To2p0 =
        createConditionalBool(rewriter, loc, isExp16, mantLe256, false);
    Value exp16To3p0 = createConditionalBool(rewriter, loc, isExp16,
                                             mantBetween256And768, false);
    Value mantLt768_2 = createEqualityCheck(rewriter, loc, mant16, c768I16,
                                            executor::ICmpType::ult);
    Value mantGe768_2 = select(rewriter, loc, mantLt768_2,
                               createBoolConstant(rewriter, loc, false),
                               createBoolConstant(rewriter, loc, true));
    Value exp16To4p0 =
        createConditionalBool(rewriter, loc, isExp16, mantGe768_2, false);

    // For exp=17 (4.0 to 8.0 range)
    Value isExp17 = createEqualityCheck(rewriter, loc, exp16, c17I16);
    Value exp17To4p0 =
        createConditionalBool(rewriter, loc, isExp17, mantLe256, false);
    Value exp17To6p0 =
        createConditionalBool(rewriter, loc, isExp17, mantGt256, false);

    // For exp>17: saturate to 6.0
    Value isGt17 = createEqualityCheck(rewriter, loc, exp16, c17I16,
                                       executor::ICmpType::ugt);

    // --- Determine unsigned FP4 value using a chain of selects ---
    std::vector<std::pair<Value, Value>> conditionsAndValues = {
        {rewriter.create<executor::SelectOp>(
             loc, isLt13, createBoolConstant(rewriter, loc, true),
             round13ToZero),
         c0I4},
        {rewriter.create<executor::SelectOp>(
             loc, round13ToHalf, createBoolConstant(rewriter, loc, true),
             round14ToHalf),
         c1I4},
        {rewriter.create<executor::SelectOp>(
             loc, round14ToOne, createBoolConstant(rewriter, loc, true),
             exp15To1p0),
         c2I4},
        {exp15To1p5, c3I4},
        {rewriter.create<executor::SelectOp>(
             loc, exp15To2p0, createBoolConstant(rewriter, loc, true),
             exp16To2p0),
         c4I4},
        {exp16To3p0, c5I4},
        {rewriter.create<executor::SelectOp>(
             loc, exp16To4p0, createBoolConstant(rewriter, loc, true),
             exp17To4p0),
         c6I4},
        {rewriter.create<executor::SelectOp>(
             loc, exp17To6p0, createBoolConstant(rewriter, loc, true), isGt17),
         c7I4}};

    Value unsignedVal =
        createSelectChain(rewriter, loc, conditionsAndValues, c0I4);

    // Apply sign bit if negative (f4E2M1FN uses sign-magnitude, not two's
    // complement)
    Value c8I4 =
        createIntegerConstant(rewriter, loc, 4, 8); // Sign bit mask 0b1000
    Value signedVal =
        select(rewriter, loc, signI1,
               bitwiseOr(rewriter, loc, unsignedVal, c8I4), unsignedVal);

    // Handle special cases
    Value resultNormal = select(rewriter, loc, isZero, c0I4, signedVal);
    Value satVal = select(rewriter, loc, signI1, c15I4, c7I4);
    Value resultInf = select(rewriter, loc, isInf, satVal, resultNormal);
    Value resultI4 = select(rewriter, loc, isNan, c0I4, resultInf);

    // Bitcast i4 result to f4E2M1FN
    rewriter.replaceOpWithNewOp<executor::BitcastOp>(
        op, Float4E2M1FNType::get(op->getContext()), resultI4);

    return success();
  }
};

class ExecutorExpandOpsPass
    : public executor::impl::ExecutorExpandOpsPassBase<ExecutorExpandOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout =
        dataLayoutAnalysis.getAtOrAbove(getOperation());

    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    target.addLegalDialect<executor::ExecutorDialect>();
    if (lowerGetOffset) {
      target.addIllegalOp<executor::GetOffsetOp>();
      patterns.add<LowerGetOffsetPattern>(dataLayout, ctx);
    }
    if (lowerAlloca) {
      target.addIllegalOp<executor::AllocaOp>();
      patterns.add<LowerAllocaPattern>(dataLayout, ctx);
    }

    patterns.add<LowerF4E2M1FNToF16Extension, LowerF16ToF4E2M1FNTruncation>(
        ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
