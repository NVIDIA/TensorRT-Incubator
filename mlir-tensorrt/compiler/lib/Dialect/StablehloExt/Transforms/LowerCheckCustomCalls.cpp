//===- LowerCheckCustomCalls.cpp ------------------------------------------===//
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
/// Lower custom_call operations used for testing (check.expect_close, etc.)
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"            // IWYU pragma: keep
#include "stablehlo/dialect/ChloOps.h" // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include <string>

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_STABLEHLOLOWERCHECKCUSTOMCALLSPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

namespace {

//===----------------------------------------------------------------------===//
// Type Helpers
//===----------------------------------------------------------------------===//

/// Get a fully dynamic shape for a given rank.
static SmallVector<int64_t> getDynShape(int64_t rank) {
  return SmallVector<int64_t>(rank, ShapedType::kDynamic);
}

/// Get unsigned integer type of specified bit width.
static Type getUnsignedIntOfWidth(MLIRContext *ctx, unsigned bitWidth) {
  return IntegerType::get(ctx, bitWidth,
                          IntegerType::SignednessSemantics::Unsigned);
}

//===----------------------------------------------------------------------===//
// Constant Creation Helpers
//===----------------------------------------------------------------------===//

static Value makeScalarConst(OpBuilder &b, Location loc, Type elemType,
                             Attribute valueAttr) {
  auto scalarTy = RankedTensorType::get({}, elemType);
  return b.create<stablehlo::ConstantOp>(
      loc, scalarTy, DenseElementsAttr::get(scalarTy, valueAttr));
}

static Value makeScalarBool(OpBuilder &b, Location loc, bool v) {
  return makeScalarConst(b, loc, b.getI1Type(), b.getBoolAttr(v));
}

//===----------------------------------------------------------------------===//
// Shape and Broadcast Helpers
//===----------------------------------------------------------------------===//

/// Build a 1-D tensor<rankxi64> containing the runtime shape of `tensor`.
static Value buildOutputDimsI64(OpBuilder &b, Location loc, Value tensor,
                                int64_t rank) {
  auto i32Ty = b.getI32Type();
  auto i64Ty = b.getI64Type();
  auto i32ScalarTy = RankedTensorType::get({}, i32Ty);
  auto i64ScalarTy = RankedTensorType::get({}, i64Ty);
  auto i64VecTy = RankedTensorType::get({rank}, i64Ty);

  if (rank == 0)
    return b.create<stablehlo::ConstantOp>(
        loc, i64VecTy, DenseElementsAttr::get(i64VecTy, ArrayRef<int64_t>{}));

  SmallVector<Value> dims;
  dims.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    Value dimI32 =
        b.create<stablehlo::GetDimensionSizeOp>(loc, i32ScalarTy, tensor, dim);
    Value dimI64 = b.create<stablehlo::ConvertOp>(loc, i64ScalarTy, dimI32);
    dims.push_back(b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({1}, i64Ty), dimI64));
  }

  if (dims.size() == 1)
    return b.create<stablehlo::ReshapeOp>(loc, i64VecTy, dims.front());

  return b.create<stablehlo::ConcatenateOp>(loc, i64VecTy, dims,
                                            b.getI64IntegerAttr(0));
}

/// Broadcast a scalar to match the shape of `ref`.
static Value broadcastScalar(OpBuilder &b, Location loc, Value scalar,
                             Value ref, RankedTensorType targetTy) {
  Value outDims = buildOutputDimsI64(b, loc, ref, targetTy.getRank());
  return b
      .create<stablehlo::DynamicBroadcastInDimOp>(
          loc, targetTy, scalar, outDims, b.getDenseI64ArrayAttr({}))
      .getResult();
}

//===----------------------------------------------------------------------===//
// Reduction Helpers
//===----------------------------------------------------------------------===//

/// Reduces an i1 predicate tensor by AND-ing across all dimensions.
static Value reduceAllAnd(OpBuilder &b, Location loc, Value pred,
                          int64_t rank) {
  if (rank == 0)
    return pred;

  auto scalarBoolTy = RankedTensorType::get({}, b.getI1Type());
  Value initTrue = makeScalarBool(b, loc, true);

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < rank; ++i)
    dims.push_back(i);

  auto reduceOp = b.create<stablehlo::ReduceOp>(
      loc, TypeRange{scalarBoolTy}, ValueRange{pred}, ValueRange{initTrue},
      b.getDenseI64ArrayAttr(dims));

  Region &body = reduceOp.getBody();
  body.push_back(new Block());
  Block &rb = body.back();
  rb.addArgument(scalarBoolTy, loc);
  rb.addArgument(scalarBoolTy, loc);

  OpBuilder::InsertionGuard rg(b);
  b.setInsertionPointToStart(&rb);
  Value andRes =
      b.create<stablehlo::AndOp>(loc, rb.getArgument(0), rb.getArgument(1));
  b.create<stablehlo::ReturnOp>(loc, andRes);

  return reduceOp.getResult(0);
}

/// Reduce to scalar and assert with given message.
static void reduceAndAssert(OpBuilder &b, Location loc, Value pred,
                            int64_t rank, StringRef msg) {
  Value reduced = reduceAllAnd(b, loc, pred, rank);
  Value extracted = b.create<tensor::ExtractOp>(loc, reduced, ValueRange{});
  b.create<cf::AssertOp>(loc, extracted, b.getStringAttr(msg));
}

//===----------------------------------------------------------------------===//
// Float Comparison Helpers
//===----------------------------------------------------------------------===//

/// Returns tensor<...xi1> true where x is NaN.
static Value buildIsNan(OpBuilder &b, Location loc, Value x) {
  return b.create<stablehlo::CompareOp>(loc, x, x,
                                        stablehlo::ComparisonDirection::NE);
}

/// Returns tensor<...xi1> signbit(x) (handles -0.0).
static Value buildSignBit(OpBuilder &b, Location loc, Value x,
                          RankedTensorType xType, Type unsignedIntElemTy) {
  auto unsignedTensorTy =
      RankedTensorType::get(xType.getShape(), unsignedIntElemTy);
  Value bits = b.create<stablehlo::BitcastConvertOp>(loc, unsignedTensorTy, x);

  unsigned bw = unsignedIntElemTy.getIntOrFloatBitWidth();
  APInt signMaskVal = APInt::getOneBitSet(bw, bw - 1);
  Value signMask =
      makeScalarConst(b, loc, unsignedIntElemTy,
                      IntegerAttr::get(unsignedIntElemTy, signMaskVal));
  Value maskBcast = broadcastScalar(b, loc, signMask, x, unsignedTensorTy);
  Value masked = b.create<stablehlo::AndOp>(loc, bits, maskBcast);

  Value zero =
      makeScalarConst(b, loc, unsignedIntElemTy,
                      IntegerAttr::get(unsignedIntElemTy, APInt(bw, 0)));
  Value zeroBcast = broadcastScalar(b, loc, zero, x, unsignedTensorTy);
  return b.create<stablehlo::CompareOp>(loc, masked, zeroBcast,
                                        stablehlo::ComparisonDirection::NE);
}

/// Build "almost equal" comparison for floats with tolerance.
static Value buildAlmostEqFloat(OpBuilder &b, Location loc, Value x, Value y,
                                RankedTensorType floatTensorTy,
                                Value tolScalar) {
  // Exact equality
  Value eq = b.create<stablehlo::CompareOp>(loc, x, y,
                                            stablehlo::ComparisonDirection::EQ);
  // Both NaN
  Value bothNan = b.create<stablehlo::AndOp>(loc, buildIsNan(b, loc, x),
                                             buildIsNan(b, loc, y));
  // Sign bits
  unsigned bw = floatTensorTy.getElementType().getIntOrFloatBitWidth();
  Type unsignedTy = getUnsignedIntOfWidth(b.getContext(), bw);
  Value xSign = buildSignBit(b, loc, x, floatTensorTy, unsignedTy);
  Value ySign = buildSignBit(b, loc, y, floatTensorTy, unsignedTy);
  Value signSame = b.create<stablehlo::CompareOp>(
      loc, xSign, ySign, stablehlo::ComparisonDirection::EQ);

  // |x-y| <= tol
  Value diff = b.create<stablehlo::SubtractOp>(loc, x, y);
  Value absDiff = b.create<stablehlo::AbsOp>(loc, diff);
  Value tolBcast = broadcastScalar(b, loc, tolScalar, x, floatTensorTy);
  Value withinTol = b.create<stablehlo::CompareOp>(
      loc, absDiff, tolBcast, stablehlo::ComparisonDirection::LE);

  Value signedOk = b.create<stablehlo::AndOp>(loc, signSame, withinTol);
  Value eqOrNan = b.create<stablehlo::OrOp>(loc, eq, bothNan);
  return b.create<stablehlo::OrOp>(loc, eqOrNan, signedOk);
}

/// Compute elementwise ULP difference for floats.
static Value buildUlpDiffFloat(OpBuilder &b, Location loc, Value f, Value g,
                               RankedTensorType floatTensorTy,
                               Type unsignedIntElemTy) {
  int64_t rank = floatTensorTy.getRank();
  auto ui64Ty = getUnsignedIntOfWidth(b.getContext(), 64);
  auto ui64TensorTy = RankedTensorType::get(floatTensorTy.getShape(), ui64Ty);
  auto unsignedTensorTy =
      RankedTensorType::get(floatTensorTy.getShape(), unsignedIntElemTy);

  Value outDims = buildOutputDimsI64(b, loc, f, rank);

  // Bitwise equal -> 0
  Value fBits = b.create<stablehlo::BitcastConvertOp>(loc, unsignedTensorTy, f);
  Value gBits = b.create<stablehlo::BitcastConvertOp>(loc, unsignedTensorTy, g);
  Value bitwiseEq = b.create<stablehlo::CompareOp>(
      loc, fBits, gBits, stablehlo::ComparisonDirection::EQ);

  // Both NaN -> 0
  Value bothNan = b.create<stablehlo::AndOp>(loc, buildIsNan(b, loc, f),
                                             buildIsNan(b, loc, g));

  // Both finite?
  Value fFinite = b.create<stablehlo::IsFiniteOp>(loc, f);
  Value gFinite = b.create<stablehlo::IsFiniteOp>(loc, g);
  Value bothFinite = b.create<stablehlo::AndOp>(loc, fFinite, gFinite);

  // Compute |f|, |g| as unsigned ints
  Value absF = b.create<stablehlo::AbsOp>(loc, f);
  Value absG = b.create<stablehlo::AbsOp>(loc, g);
  Value af = b.create<stablehlo::BitcastConvertOp>(loc, unsignedTensorTy, absF);
  Value ag = b.create<stablehlo::BitcastConvertOp>(loc, unsignedTensorTy, absG);

  // Different signs?
  Value fSign = buildSignBit(b, loc, f, floatTensorTy, unsignedIntElemTy);
  Value gSign = buildSignBit(b, loc, g, floatTensorTy, unsignedIntElemTy);
  Value diffSign = b.create<stablehlo::CompareOp>(
      loc, fSign, gSign, stablehlo::ComparisonDirection::NE);

  // Same-sign ULP = |af - ag|
  Value aGtB = b.create<stablehlo::CompareOp>(
      loc, af, ag, stablehlo::ComparisonDirection::GT);
  Value aMinusB = b.create<stablehlo::SubtractOp>(loc, af, ag);
  Value bMinusA = b.create<stablehlo::SubtractOp>(loc, ag, af);
  Value ulpSame = b.create<stablehlo::SelectOp>(loc, aGtB, aMinusB, bMinusA);

  // Diff-sign ULP = af + ag
  Value ulpDiff = b.create<stablehlo::AddOp>(loc, af, ag);
  Value ulpUnsigned =
      b.create<stablehlo::SelectOp>(loc, diffSign, ulpDiff, ulpSame);

  // Convert to ui64
  Value ulpUi64 =
      b.create<stablehlo::ConvertOp>(loc, ui64TensorTy, ulpUnsigned);

  // Constants for select
  APInt maxU64 = APInt::getAllOnes(64);
  Value maxScalar =
      makeScalarConst(b, loc, ui64Ty, IntegerAttr::get(ui64Ty, maxU64));
  Value zeroScalar =
      makeScalarConst(b, loc, ui64Ty, IntegerAttr::get(ui64Ty, APInt(64, 0)));
  Value maxBcast =
      b.create<stablehlo::DynamicBroadcastInDimOp>(
           loc, ui64TensorTy, maxScalar, outDims, b.getDenseI64ArrayAttr({}))
          .getResult();
  Value zeroBcast =
      b.create<stablehlo::DynamicBroadcastInDimOp>(
           loc, ui64TensorTy, zeroScalar, outDims, b.getDenseI64ArrayAttr({}))
          .getResult();

  // Select: bitwiseEq -> 0, bothFinite -> ulp, bothNan -> 0, else -> max
  Value nonFiniteCase =
      b.create<stablehlo::SelectOp>(loc, bothNan, zeroBcast, maxBcast);
  Value finiteOrNot =
      b.create<stablehlo::SelectOp>(loc, bothFinite, ulpUi64, nonFiniteCase);
  return b.create<stablehlo::SelectOp>(loc, bitwiseEq, zeroBcast, finiteOrNot);
}

//===----------------------------------------------------------------------===//
// Complex Type Helpers
//===----------------------------------------------------------------------===//

/// Apply a binary operation to real and imaginary parts of complex tensors.
template <typename OpFn>
static Value applyToComplexParts(OpBuilder &b, Location loc, Value lhs,
                                 Value rhs, RankedTensorType complexTensorTy,
                                 OpFn opFn) {
  auto complexTy = cast<ComplexType>(complexTensorTy.getElementType());
  Type floatTy = complexTy.getElementType();
  auto floatTensorTy =
      RankedTensorType::get(complexTensorTy.getShape(), floatTy);

  Value lhsRe = b.create<stablehlo::RealOp>(loc, floatTensorTy, lhs);
  Value rhsRe = b.create<stablehlo::RealOp>(loc, floatTensorTy, rhs);
  Value lhsIm = b.create<stablehlo::ImagOp>(loc, floatTensorTy, lhs);
  Value rhsIm = b.create<stablehlo::ImagOp>(loc, floatTensorTy, rhs);

  Value reResult = opFn(b, loc, lhsRe, rhsRe, floatTensorTy);
  Value imResult = opFn(b, loc, lhsIm, rhsIm, floatTensorTy);

  return b.create<stablehlo::AndOp>(loc, reResult, imResult);
}

//===----------------------------------------------------------------------===//
// Elementwise Comparison Builders
//===----------------------------------------------------------------------===//

/// Build element-wise equality comparison.
static Value buildElementwiseEq(OpBuilder &b, Location loc, Value lhs,
                                Value rhs, RankedTensorType dynType) {
  Type elemTy = dynType.getElementType();

  if (auto complexTy = dyn_cast<ComplexType>(elemTy)) {
    return applyToComplexParts(
        b, loc, lhs, rhs, dynType,
        [](OpBuilder &b, Location loc, Value l, Value r, RankedTensorType) {
          return b.create<stablehlo::CompareOp>(
              loc, l, r, stablehlo::ComparisonDirection::EQ);
        });
  }

  if (isa<FloatType>(elemTy)) {
    // For floats, check.eq uses "almost equal" with default tolerance 1e-4.
    auto scalarF64Ty = RankedTensorType::get({}, b.getF64Type());
    Value tol = b.create<stablehlo::ConstantOp>(
        loc, scalarF64Ty,
        DenseElementsAttr::get(scalarF64Ty, b.getF64FloatAttr(1e-4)));
    auto scalarFloatTy = RankedTensorType::get({}, elemTy);
    Value tolFloat = b.create<stablehlo::ConvertOp>(loc, scalarFloatTy, tol);
    return buildAlmostEqFloat(b, loc, lhs, rhs, dynType, tolFloat);
  }

  return b.create<stablehlo::CompareOp>(loc, lhs, rhs,
                                        stablehlo::ComparisonDirection::EQ);
}

/// Build exact element-wise equality (no tolerance for floats).
static Value buildExactEq(OpBuilder &b, Location loc, Value lhs, Value rhs,
                          RankedTensorType dynType) {
  Type elemTy = dynType.getElementType();

  if (auto complexTy = dyn_cast<ComplexType>(elemTy)) {
    return applyToComplexParts(
        b, loc, lhs, rhs, dynType,
        [](OpBuilder &b, Location loc, Value l, Value r, RankedTensorType) {
          return b.create<stablehlo::CompareOp>(
              loc, l, r, stablehlo::ComparisonDirection::EQ);
        });
  }

  return b.create<stablehlo::CompareOp>(loc, lhs, rhs,
                                        stablehlo::ComparisonDirection::EQ);
}

/// Build almost-equal comparison with tolerance.
static Value buildAlmostEq(OpBuilder &b, Location loc, Value lhs, Value rhs,
                           RankedTensorType dynType, Value tolF64) {
  Type elemTy = dynType.getElementType();
  auto scalarFloatTy = RankedTensorType::get({}, elemTy);

  if (auto complexTy = dyn_cast<ComplexType>(elemTy)) {
    Type floatTy = complexTy.getElementType();
    auto floatScalarTy = RankedTensorType::get({}, floatTy);
    Value tolFloat = b.create<stablehlo::ConvertOp>(loc, floatScalarTy, tolF64);
    return applyToComplexParts(
        b, loc, lhs, rhs, dynType,
        [tolFloat](OpBuilder &b, Location loc, Value l, Value r,
                   RankedTensorType floatTensorTy) {
          return buildAlmostEqFloat(b, loc, l, r, floatTensorTy, tolFloat);
        });
  }

  Value tolFloat = b.create<stablehlo::ConvertOp>(loc, scalarFloatTy, tolF64);
  return buildAlmostEqFloat(b, loc, lhs, rhs, dynType, tolFloat);
}

/// Build ULP-based close comparison.
static Value buildUlpClose(OpBuilder &b, Location loc, Value lhs, Value rhs,
                           RankedTensorType dynType, Value minUlp,
                           Value maxUlp) {
  Type elemTy = dynType.getElementType();
  int64_t rank = dynType.getRank();
  auto ui64Ty = getUnsignedIntOfWidth(b.getContext(), 64);
  auto ui64TensorTy = RankedTensorType::get(dynType.getShape(), ui64Ty);

  Value ulpDiff;
  if (auto complexTy = dyn_cast<ComplexType>(elemTy)) {
    Type floatTy = complexTy.getElementType();
    auto floatTensorTy = RankedTensorType::get(dynType.getShape(), floatTy);
    unsigned bw = floatTy.getIntOrFloatBitWidth();
    Type unsignedTy = getUnsignedIntOfWidth(b.getContext(), bw);

    Value aRe = b.create<stablehlo::RealOp>(loc, floatTensorTy, lhs);
    Value bRe = b.create<stablehlo::RealOp>(loc, floatTensorTy, rhs);
    Value aIm = b.create<stablehlo::ImagOp>(loc, floatTensorTy, lhs);
    Value bIm = b.create<stablehlo::ImagOp>(loc, floatTensorTy, rhs);

    Value ulpRe =
        buildUlpDiffFloat(b, loc, aRe, bRe, floatTensorTy, unsignedTy);
    Value ulpIm =
        buildUlpDiffFloat(b, loc, aIm, bIm, floatTensorTy, unsignedTy);
    ulpDiff = b.create<stablehlo::MaxOp>(loc, ulpRe, ulpIm);
  } else {
    unsigned bw = elemTy.getIntOrFloatBitWidth();
    Type unsignedTy = getUnsignedIntOfWidth(b.getContext(), bw);
    ulpDiff = buildUlpDiffFloat(b, loc, lhs, rhs, dynType, unsignedTy);
  }

  Value outDims = buildOutputDimsI64(b, loc, lhs, rank);
  Value minBcast =
      b.create<stablehlo::DynamicBroadcastInDimOp>(
           loc, ui64TensorTy, minUlp, outDims, b.getDenseI64ArrayAttr({}))
          .getResult();
  Value maxBcast =
      b.create<stablehlo::DynamicBroadcastInDimOp>(
           loc, ui64TensorTy, maxUlp, outDims, b.getDenseI64ArrayAttr({}))
          .getResult();

  Value geMin = b.create<stablehlo::CompareOp>(
      loc, ulpDiff, minBcast, stablehlo::ComparisonDirection::GE);
  Value leMax = b.create<stablehlo::CompareOp>(
      loc, ulpDiff, maxBcast, stablehlo::ComparisonDirection::LE);
  return b.create<stablehlo::AndOp>(loc, geMin, leMax);
}

//===----------------------------------------------------------------------===//
// Helper Function Generators
//===----------------------------------------------------------------------===//

/// Create a no-inline function with dynamic tensor arguments.
static func::FuncOp createHelperFunc(OpBuilder &b, Location loc, StringRef name,
                                     FunctionType funcType) {
  auto funcOp = b.create<func::FuncOp>(loc, name, funcType);
  funcOp.setNoInline(true);
  funcOp.setPrivate();
  return funcOp;
}

/// Generate check.eq function that returns tensor<i1> result (no assertion).
static func::FuncOp generateCheckEqFunc(OpBuilder &b, Location loc,
                                        RankedTensorType type, unsigned id) {
  int64_t rank = type.getRank();
  auto dynType =
      RankedTensorType::get(getDynShape(rank), type.getElementType());
  auto boolTy = RankedTensorType::get({}, b.getI1Type());
  auto funcType = b.getFunctionType({dynType, dynType}, {boolTy});

  auto funcOp =
      createHelperFunc(b, loc, "check_eq_" + std::to_string(id), funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entry);

  Value eq = buildElementwiseEq(b, loc, funcOp.getArgument(0),
                                funcOp.getArgument(1), dynType);
  Value reduced = reduceAllAnd(b, loc, eq, rank);
  b.create<func::ReturnOp>(loc, reduced);
  return funcOp;
}

/// Generate check.expect_eq assertion function.
static func::FuncOp generateExpectEqFunc(OpBuilder &b, Location loc,
                                         RankedTensorType type, unsigned id) {
  int64_t rank = type.getRank();
  auto dynType =
      RankedTensorType::get(getDynShape(rank), type.getElementType());
  auto funcType = b.getFunctionType({dynType, dynType}, {});

  auto funcOp = createHelperFunc(
      b, loc, "check_expect_eq_" + std::to_string(id), funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entry);

  Value eq = buildExactEq(b, loc, funcOp.getArgument(0), funcOp.getArgument(1),
                          dynType);
  reduceAndAssert(b, loc, eq, rank, "check_expect_eq failed");
  b.create<func::ReturnOp>(loc);
  return funcOp;
}

/// Generate check.expect_almost_eq assertion function.
static func::FuncOp generateExpectAlmostEqFunc(OpBuilder &b, Location loc,
                                               RankedTensorType type,
                                               unsigned id) {
  int64_t rank = type.getRank();
  auto dynType =
      RankedTensorType::get(getDynShape(rank), type.getElementType());
  auto scalarF64 = RankedTensorType::get({}, b.getF64Type());
  auto funcType = b.getFunctionType({dynType, dynType, scalarF64}, {});

  auto funcOp = createHelperFunc(
      b, loc, "check_expect_almost_eq_" + std::to_string(id), funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entry);

  Value pred =
      buildAlmostEq(b, loc, funcOp.getArgument(0), funcOp.getArgument(1),
                    dynType, funcOp.getArgument(2));
  reduceAndAssert(b, loc, pred, rank, "check_expect_almost_eq failed");
  b.create<func::ReturnOp>(loc);
  return funcOp;
}

/// Generate check.expect_close assertion function.
static func::FuncOp generateExpectCloseFunc(OpBuilder &b, Location loc,
                                            RankedTensorType type,
                                            unsigned id) {
  int64_t rank = type.getRank();
  auto dynType =
      RankedTensorType::get(getDynShape(rank), type.getElementType());
  auto ui64Ty = getUnsignedIntOfWidth(b.getContext(), 64);
  auto scalarUi64 = RankedTensorType::get({}, ui64Ty);
  auto funcType =
      b.getFunctionType({dynType, dynType, scalarUi64, scalarUi64}, {});

  auto funcOp = createHelperFunc(
      b, loc, "check_expect_close_" + std::to_string(id), funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entry);

  Value isClose =
      buildUlpClose(b, loc, funcOp.getArgument(0), funcOp.getArgument(1),
                    dynType, funcOp.getArgument(2), funcOp.getArgument(3));
  reduceAndAssert(b, loc, isClose, rank, "check_expect_close failed");
  b.create<func::ReturnOp>(loc);
  return funcOp;
}

/// Generate no-op function for check.expect_serialized_eq.
static func::FuncOp generateExpectSerializedEqNoopFunc(OpBuilder &b,
                                                       Location loc,
                                                       RankedTensorType type,
                                                       unsigned id) {
  int64_t rank = type.getRank();
  auto dynType =
      RankedTensorType::get(getDynShape(rank), type.getElementType());
  auto funcType = b.getFunctionType({dynType}, {});

  auto funcOp = createHelperFunc(
      b, loc, "check_expect_serialized_eq_noop_" + std::to_string(id),
      funcType);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entry);
  b.create<func::ReturnOp>(loc);
  return funcOp;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

/// Cast value to dynamic shape type if needed.
static Value castToDynamic(OpBuilder &b, Location loc, Value v, int64_t rank,
                           Type elemTy) {
  auto dynType = RankedTensorType::get(getDynShape(rank), elemTy);
  if (v.getType() != dynType)
    return b.create<tensor::CastOp>(loc, dynType, v);
  return v;
}

/// Helper function cache key.
struct HelperKey {
  StringRef kind;
  int64_t rank;
  Type elem;
};

struct HelperKeyInfo {
  static HelperKey getEmptyKey() { return {StringRef(), -1, Type()}; }
  static HelperKey getTombstoneKey() { return {StringRef("~"), -2, Type()}; }
  static unsigned getHashValue(const HelperKey &k) {
    return llvm::hash_combine(k.kind, k.rank, k.elem);
  }
  static bool isEqual(const HelperKey &a, const HelperKey &b) {
    return a.kind == b.kind && a.rank == b.rank && a.elem == b.elem;
  }
};

class StablehloLowerCheckCustomCallsPass
    : public stablehlo_ext::impl::StablehloLowerCheckCustomCallsPassBase<
          StablehloLowerCheckCustomCallsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto moduleOp = dyn_cast<ModuleOp>(root);
    if (!moduleOp)
      moduleOp = root->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      root->emitError()
          << "stablehlo-ext-lower-check-custom-calls requires a ModuleOp";
      return signalPassFailure();
    }

    // Collect relevant custom calls.
    SmallVector<stablehlo::CustomCallOp> customCalls;
    moduleOp.walk([&](stablehlo::CustomCallOp op) {
      StringRef t = op.getCallTargetName();
      if (t == "check.eq" || t == "check.expect_close" ||
          t == "check.expect_eq" || t == "check.expect_eq_const" ||
          t == "check.expect_almost_eq" ||
          t == "check.expect_almost_eq_const" ||
          t == "check.expect_serialized_eq")
        customCalls.push_back(op);
    });

    if (customCalls.empty())
      return;

    llvm::DenseMap<HelperKey, func::FuncOp, HelperKeyInfo> helperCache;
    unsigned funcCounter = 0;

    auto getOrCreateHelper = [&](StringRef kind, RankedTensorType type,
                                 Location loc,
                                 auto generateFn) -> func::FuncOp {
      HelperKey key{kind, type.getRank(), type.getElementType()};
      if (auto it = helperCache.find(key); it != helperCache.end())
        return it->second;

      OpBuilder mb(moduleOp.getBodyRegion());
      mb.setInsertionPointToEnd(moduleOp.getBody());
      auto helper = generateFn(mb, loc, type, funcCounter++);
      helperCache[key] = helper;
      return helper;
    };

    for (stablehlo::CustomCallOp op : customCalls) {
      Location loc = op.getLoc();
      StringRef target = op.getCallTargetName();

      auto arg0Type = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      if (!arg0Type) {
        op.emitError("check op operand must be a ranked tensor");
        return signalPassFailure();
      }

      int64_t rank = arg0Type.getRank();
      Type elemTy = arg0Type.getElementType();
      OpBuilder b(op);

      // Handle check.eq (returns tensor<i1> result, no internal assertion).
      if (target == "check.eq") {
        if (op.getNumOperands() != 2) {
          op.emitError("check.eq expects 2 operands");
          return signalPassFailure();
        }

        // Only process check.eq if it returns tensor<i1>.
        if (op.getNumResults() != 1) {
          op.emitError("check.eq must return exactly one result of type "
                       "tensor<i1>");
          return signalPassFailure();
        }

        auto resTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!resTy || resTy.getRank() != 0 ||
            !resTy.getElementType().isInteger(1)) {
          op.emitError("check.eq must return tensor<i1>");
          return signalPassFailure();
        }

        auto helper =
            getOrCreateHelper("check.eq", arg0Type, loc, generateCheckEqFunc);

        Value lhs = castToDynamic(b, loc, op.getOperand(0), rank, elemTy);
        Value rhs = castToDynamic(b, loc, op.getOperand(1), rank, elemTy);
        auto callOp = b.create<func::CallOp>(
            loc, helper.getSymName(), TypeRange{resTy}, ValueRange{lhs, rhs});
        op.getResult(0).replaceAllUsesWith(callOp.getResult(0));
        op.erase();
        continue;
      }

      // Remaining are side-effect checks.
      if (op.getNumResults() != 0) {
        op.emitError("check.expect_* custom calls are expected to return ()");
        return signalPassFailure();
      }

      // Handle check.expect_serialized_eq (no-op).
      if (target == "check.expect_serialized_eq") {
        if (op.getNumOperands() != 1) {
          op.emitError("check.expect_serialized_eq expects 1 operand");
          return signalPassFailure();
        }
        auto helper = getOrCreateHelper(target, arg0Type, loc,
                                        generateExpectSerializedEqNoopFunc);
        Value arg = castToDynamic(b, loc, op.getOperand(0), rank, elemTy);
        b.create<func::CallOp>(loc, helper.getSymName(), TypeRange{},
                               ValueRange{arg});
        op.erase();
        continue;
      }

      // Binary checks require 2 operands of same type.
      if (op.getNumOperands() != 2) {
        op.emitError("expected 2 operands for " + target);
        return signalPassFailure();
      }
      auto rhsType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
      if (!rhsType || rhsType != arg0Type) {
        op.emitError("operands must be ranked tensors of the same type");
        return signalPassFailure();
      }

      Value lhs = castToDynamic(b, loc, op.getOperand(0), rank, elemTy);
      Value rhs = castToDynamic(b, loc, op.getOperand(1), rank, elemTy);

      if (target == "check.expect_eq" || target == "check.expect_eq_const") {
        auto helper =
            getOrCreateHelper(target, arg0Type, loc, generateExpectEqFunc);
        b.create<func::CallOp>(loc, helper.getSymName(), TypeRange{},
                               ValueRange{lhs, rhs});
        op.erase();
        continue;
      }

      if (target == "check.expect_almost_eq" ||
          target == "check.expect_almost_eq_const") {
        double defaultTol = (target == "check.expect_almost_eq") ? 1e-3 : 1e-4;
        double tolVal = defaultTol;
        if (auto a = op->getAttrOfType<FloatAttr>("tolerance"))
          tolVal = a.getValueAsDouble();

        auto helper = getOrCreateHelper(target, arg0Type, loc,
                                        generateExpectAlmostEqFunc);
        auto scalarF64Ty = RankedTensorType::get({}, b.getF64Type());
        Value tol = b.create<stablehlo::ConstantOp>(
            loc, scalarF64Ty,
            DenseElementsAttr::get(scalarF64Ty, b.getF64FloatAttr(tolVal)));
        b.create<func::CallOp>(loc, helper.getSymName(), TypeRange{},
                               ValueRange{lhs, rhs, tol});
        op.erase();
        continue;
      }

      if (target == "check.expect_close") {
        uint64_t minUlpVal = 0, maxUlpVal = 3;
        if (auto attr = op->getAttrOfType<IntegerAttr>("min_ulp_difference"))
          minUlpVal = attr.getValue().getZExtValue();
        if (auto attr = op->getAttrOfType<IntegerAttr>("max_ulp_difference"))
          maxUlpVal = attr.getValue().getZExtValue();

        auto helper =
            getOrCreateHelper(target, arg0Type, loc, generateExpectCloseFunc);
        auto ui64Ty = getUnsignedIntOfWidth(&getContext(), 64);
        auto scalarUi64Ty = RankedTensorType::get({}, ui64Ty);
        Value minUlp = b.create<stablehlo::ConstantOp>(
            loc, scalarUi64Ty,
            DenseElementsAttr::get(scalarUi64Ty,
                                   b.getIntegerAttr(ui64Ty, minUlpVal)));
        Value maxUlp = b.create<stablehlo::ConstantOp>(
            loc, scalarUi64Ty,
            DenseElementsAttr::get(scalarUi64Ty,
                                   b.getIntegerAttr(ui64Ty, maxUlpVal)));
        b.create<func::CallOp>(loc, helper.getSymName(), TypeRange{},
                               ValueRange{lhs, rhs, minUlp, maxUlp});
        op.erase();
        continue;
      }

      op.emitError("unhandled check custom call: " + target);
      return signalPassFailure();
    }
  }
};

} // namespace
