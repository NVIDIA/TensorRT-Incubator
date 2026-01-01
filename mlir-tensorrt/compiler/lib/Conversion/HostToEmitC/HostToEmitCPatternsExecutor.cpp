//===- HostToEmitCPatternsExecutor.cpp ------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Executor + small misc op lowering patterns for `convert-host-to-emitc`.
///
/// This file mostly emits straightforward C/C++ "glue" around runtime APIs:
///   - `executor.print` -> `printf(...)`
///   - `cf.assert`      -> `assert(cond && "...")`
///   - `math.log`       -> `log/logf(...)`
///   - executor ABI ops -> `*ptr` loads / `ptr[0] = value` stores
//===----------------------------------------------------------------------===//

#include "HostToEmitCDetail.h"

#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::host_to_emitc;

namespace {

struct ExecutorPrintConverter
    : public EmitCConversionPattern<executor::PrintOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(executor::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `printf("<fmt>\\n", args...);`
    std::optional<StringRef> format = op.getFormat();
    Location loc = op.getLoc();
    if (!format)
      return failure();

    SmallVector<OpFoldResult> staticArgs = {emitc::OpaqueAttr::get(
        rewriter.getContext(), llvm::formatv("\"{0}\\n\"", *format).str())};
    llvm::append_range(staticArgs, adaptor.getArguments());
    createCallOpaque(rewriter, loc, {}, "printf", staticArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Math conversions missing from 'math-to-emitc' pass
//===----------------------------------------------------------------------===//

struct MathLogToEmitCPattern : public EmitCConversionPattern<math::LogOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(math::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `log(x)` / `logf(x)` depending on element type.
    Value input = adaptor.getOperand();
    Type inputType = input.getType();
    if (!inputType.isF32() && !inputType.isF64())
      return failure();
    llvm::StringRef funcName = inputType.isF32() ? "logf" : "log";
    auto callOp =
        createCallOpaque(rewriter, op.getLoc(), inputType, funcName, {input});
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CF Conversions
//===----------------------------------------------------------------------===//

struct CFAssertPattern : public EmitCConversionPattern<cf::AssertOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cf::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `assert(condition && "<msg>");`
    Location loc = op.getLoc();
    Value condition = adaptor.getArg();
    StringRef msg = op.getMsg();

    std::string escapedMsg;
    for (char c : msg) {
      if (c == '"')
        escapedMsg += "\\\"";
      else if (c == '\\')
        escapedMsg += "\\\\";
      else if (c == '{')
        escapedMsg += "{{";
      else
        escapedMsg += c;
    }
    auto verbatimStr = "assert({} && \"" + escapedMsg + "\");";
    rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(verbatimStr),
                                       condition);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Executor ABI Conversions
//===----------------------------------------------------------------------===//

struct ExecutorABIRecvPattern
    : public EmitCConversionPattern<executor::ABIRecvOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(executor::ABIRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `return *reinterpret_cast<T*>(ptr);`
    Type resultType = op.getType();
    Type convertedResultType = typeConverter->convertType(resultType);
    if (!convertedResultType)
      return failure();

    Type targetPtrType = getPointerType(convertedResultType);
    Value ptr = adaptor.getPtr();
    if (targetPtrType != ptr.getType())
      ptr = createCast(rewriter, targetPtrType, ptr);

    rewriter.replaceOpWithNewOp<emitc::ApplyOp>(op, convertedResultType, "*",
                                                ptr);
    return success();
  }
};

struct ExecutorABISendPattern
    : public EmitCConversionPattern<executor::ABISendOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(executor::ABISendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `reinterpret_cast<T*>(ptr)[0] = value;`
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(
          op, "ABISendOp must have no users before final lowering");
    Type valueType = op.getValue().getType();
    Type convertedValueType = typeConverter->convertType(valueType);
    if (!convertedValueType)
      return failure();

    auto lowerToStore = [&]() {
      Type targetPtrType = getPointerType(convertedValueType);
      Value ptr = adaptor.getPtr();
      if (targetPtrType != ptr.getType())
        ptr = createCast(rewriter, targetPtrType, ptr);

      Value zero = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value dest = rewriter.create<emitc::SubscriptOp>(
          op.getLoc(), getLValueType(convertedValueType), ptr, zero);
      rewriter.create<emitc::AssignOp>(op.getLoc(), dest, adaptor.getValue());
      rewriter.eraseOp(op);
      return success();
    };

    const bool isScalar =
        isa<FloatType, IndexType, IntegerType, ComplexType>(valueType);
    if (isScalar)
      return lowerToStore();

    if (auto memrefType = dyn_cast<MemRefType>(valueType)) {
      (void)memrefType;
      auto func = op->getParentOfType<FunctionOpInterface>();
      if (!func)
        return failure();

      auto blockArg = dyn_cast<BlockArgument>(op.getPtr());
      if (!blockArg)
        return failure();

      executor::ArgumentABIAttr abiAttr =
          executor::abi::getArgumentABIAttr(func, blockArg.getArgNumber());
      if (!abiAttr)
        return failure();

      if (!abiAttr.getUndef()) {
        rewriter.eraseOp(op);
        return success();
      }

      Value ownership = op.getOwnership();
      if (!ownership || !mlir::matchPattern(ownership, mlir::m_One()))
        return rewriter.notifyMatchFailure(
            op, "ownership must be statically true for final lowering of "
                "ABISendOp marked as `byref`");

      return lowerToStore();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unhandled value type for ABISendOp");
  }
};

} // namespace

namespace mlir::host_to_emitc {
void populateHostToEmitCExecutorPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         const DataLayout &dataLayout) {
  patterns.add<CFAssertPattern, ExecutorABIRecvPattern, ExecutorABISendPattern,
               ExecutorPrintConverter, MathLogToEmitCPattern>(
      typeConverter, dataLayout, patterns.getContext());
}
} // namespace mlir::host_to_emitc
