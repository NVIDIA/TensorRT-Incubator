//===- StdToExecutor.cpp  -------------------------------------------------===//
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
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::executor {
#define GEN_PASS_DEF_CONVERTSTDTOEXECUTORPASS
#include "mlir-executor/Conversion/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using executor::ConvertOpToExecutorPattern;
using executor::ExecutorConversionTarget;
using executor::ExecutorTypeConverter;
using executor::LowerToExecutorOptions;

#include "MathToExecutor.pdll.h.inc"

namespace {
/// Rewrite `arith.constant` to `executor.constant`.
struct RewriteConst : ConvertOpToExecutorPattern<arith::ConstantOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter();
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();
    // We need to ensure that index type attributes are converted correctly.
    auto convertAttr = [&](Attribute val) -> Attribute {
      if (auto intAttr = dyn_cast<IntegerAttr>(val)) {
        return rewriter.getIntegerAttr(
            typeConverter->convertType(intAttr.getType()),
            intAttr.getValue().getSExtValue());
      };
      return val;
    };
    rewriter.replaceOpWithNewOp<executor::ConstantOp>(
        op, cast<TypedAttr>(convertAttr(op.getValue())));

    return success();
  }
};

/// Convert basic binary arithmetic operations from `arith` dialect to
/// `executor` dialect.
template <typename ArithOpTy, typename ExecOpTy>
struct ConvertBinaryArithToExecute
    : public ConvertOpToExecutorPattern<ArithOpTy> {
  using ConvertOpToExecutorPattern<ArithOpTy>::ConvertOpToExecutorPattern;
  LogicalResult matchAndRewrite(
      ArithOpTy op,
      typename ConvertOpToExecutorPattern<ArithOpTy>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ExecOpTy>(op, adaptor.getLhs(),
                                          adaptor.getRhs());
    return success();
  }
};

/// Convert unary arithmetic ops from `math` and `arith` dialect to `executor`
/// dialect.
template <typename UnaryOpTy, typename ExecOpTy>
struct ConvertUnaryArithToExecute
    : public ConvertOpToExecutorPattern<UnaryOpTy> {
  using ConvertOpToExecutorPattern<UnaryOpTy>::ConvertOpToExecutorPattern;
  LogicalResult matchAndRewrite(
      UnaryOpTy op,
      typename ConvertOpToExecutorPattern<UnaryOpTy>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ExecOpTy>(op, op.getType(),
                                          adaptor.getOperand());
    return success();
  }
};

template <typename UnaryOpTy, typename ExecOpTy>
struct ConvertUnaryCastToExecute
    : public ConvertOpToExecutorPattern<UnaryOpTy> {
  using ConvertOpToExecutorPattern<UnaryOpTy>::ConvertOpToExecutorPattern;
  LogicalResult matchAndRewrite(
      UnaryOpTy op,
      typename ConvertOpToExecutorPattern<UnaryOpTy>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ExecOpTy>(op, op.getType(), adaptor.getIn());
    return success();
  }
};

/// Convert `arith.index_cast` by forwarding the operand.
struct ConvertArithIndexCastOp
    : public ConvertOpToExecutorPattern<arith::IndexCastOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we only need to convert this for equal bit types. It can be
    // expanded upon further use cases.
    Type resultType = op.getResult().getType();
    Type targetElementType = this->typeConverter->convertType(resultType);
    Type sourceElementType = adaptor.getIn().getType();
    unsigned targetBits = targetElementType.getIntOrFloatBitWidth();
    unsigned sourceBits = sourceElementType.getIntOrFloatBitWidth();
    if (!sourceElementType.isSignlessInteger() ||
        !targetElementType.isSignlessInteger())
      return rewriter.notifyMatchFailure(
          op, "unhandled index cast case for non-signless integers");
    if (targetBits == sourceBits) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }
    if (targetBits > sourceBits) {
      rewriter.replaceOpWithNewOp<executor::SIExtOp>(op, targetElementType,
                                                     adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<executor::TruncOp>(op, targetElementType,
                                                   adaptor.getIn());
    return success();
  }
};

/// Rewrite `arith.select` to `executor.select` op
struct ConvertArithSelect : public ConvertOpToExecutorPattern<arith::SelectOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<executor::SelectOp>(
        op, getTypeConverter()->convertType(op.getType()),
        adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};
} // namespace

static FailureOr<executor::ICmpType> convertPredicate(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return executor::ICmpType::eq;
  case arith::CmpIPredicate::ne:
    return executor::ICmpType::ne;
  case arith::CmpIPredicate::slt:
    return executor::ICmpType::slt;
  case arith::CmpIPredicate::sgt:
    return executor::ICmpType::sgt;
  case arith::CmpIPredicate::sle:
    return executor::ICmpType::sle;
  case arith::CmpIPredicate::sge:
    return executor::ICmpType::sge;
  case arith::CmpIPredicate::ult:
    return executor::ICmpType::ult;
  case arith::CmpIPredicate::ugt:
    return executor::ICmpType::ugt;
  default:
    return failure();
  }
}

namespace {
/// Convert `arith.cmpi` to `executor.icmp`.
struct ConvertCmpI : public ConvertOpToExecutorPattern<arith::CmpIOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<executor::ICmpType> predicate = convertPredicate(op);
    if (failed(predicate))
      return failure();
    rewriter.replaceOpWithNewOp<executor::ICmpOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs(), *predicate);
    return success();
  }
};
} // namespace

/// Convert the `arith` float comparison predicate to an executor float
/// comparison type. This is designed to be a direct cast.
static FailureOr<executor::FCmpType> convertFloatPredicate(arith::CmpFOp op) {
  return static_cast<executor::FCmpType>(op.getPredicate());
}

namespace {
/// Convert `arith.cmpf` to `executor.fcmp`.
struct ConvertCmpF : public ConvertOpToExecutorPattern<arith::CmpFOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<executor::FCmpType> predicate = convertFloatPredicate(op);
    if (failed(predicate))
      return failure();
    rewriter.replaceOpWithNewOp<executor::FCmpOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs(), *predicate);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Control Flow Ops
//===----------------------------------------------------------------------===//

/// Template for updating `cf.branch`/`cf.cond_branch` with type-converted
/// types.
template <typename OpType>
struct ConvertBranchBase : public ConvertOpToExecutorPattern<OpType> {
  using ConvertOpToExecutorPattern<OpType>::ConvertOpToExecutorPattern;
  using typename ConvertOpToExecutorPattern<OpType>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands() == op->getOperands())
      return failure();
    if (failed(verifyDestBlockTypes(op, adaptor, rewriter)))
      return failure();
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }

  static LogicalResult
  checkJumpValidity(MutableArrayRef<BlockArgument> blockArgs,
                    ValueRange operands, ConversionPatternRewriter &rewriter) {
    for (auto [blockArg, operand] : llvm::zip(blockArgs, operands)) {
      auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(
          rewriter.getRemappedValue(blockArg).getDefiningOp());
      // If there's no cast, then we're good for this operand.
      if (!castOp)
        continue;
      // If there's a non-trivial cast, then we shouldn't convert this branch
      // op.
      if (castOp->getOperandTypes().front() != operand.getType())
        return failure();
    }
    return success();
  };

  virtual LogicalResult
  verifyDestBlockTypes(OpType op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const = 0;
};

struct ConvertCfBranch : public ConvertBranchBase<cf::BranchOp> {
  using ConvertBranchBase::ConvertBranchBase;
  LogicalResult
  verifyDestBlockTypes(cf::BranchOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final {
    return checkJumpValidity(op.getDest()->getArguments(),
                             adaptor.getDestOperands(), rewriter);
  }
};

struct ConvertCfCondBranch : public ConvertBranchBase<cf::CondBranchOp> {
  using ConvertBranchBase::ConvertBranchBase;
  LogicalResult
  verifyDestBlockTypes(cf::CondBranchOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final {

    return success(
        succeeded(checkJumpValidity(op.getTrueDest()->getArguments(),
                                    adaptor.getTrueDestOperands(), rewriter)) &&
        succeeded(checkJumpValidity(op.getFalseDest()->getArguments(),
                                    adaptor.getFalseDestOperands(), rewriter)));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Func Dialect Conversions
//===----------------------------------------------------------------------===//

/// Choose which function attributes to carry over during conversion of
/// `func.func.` We really just want to avoid copying over attributes carying
/// type/symbol information.
static SmallVector<NamedAttribute>
getConvertedFunctionAttributes(func::FuncOp func) {
  return llvm::to_vector(llvm::make_filter_range(
      func->getAttrs(), [&](const NamedAttribute &attr) {
        return !(attr.getName() == SymbolTable::getSymbolAttrName() ||
                 attr.getName() == func.getFunctionTypeAttrName() ||
                 attr.getName() == func.getArgAttrsAttrName() ||
                 attr.getName() == func.getResAttrsAttrName());
      }));
}

static FailureOr<SmallVector<DictionaryAttr>>
getConvertedArgAttrs(func::FuncOp func, FunctionType newFuncType,
                     const TypeConverter::SignatureConversion &sigConverter) {
  SmallVector<DictionaryAttr> result(
      newFuncType.getNumInputs(), DictionaryAttr::get(func->getContext(), {}));
  if (ArrayAttr argAttrs = func.getAllArgAttrs()) {
    // Loop over original func arguments and assign.
    for (unsigned i = 0; i < func.getNumArguments(); i++) {
      std::optional<TypeConverter::SignatureConversion::InputMapping> mapping =
          sigConverter.getInputMapping(i);
      if (!mapping)
        return failure();
      // 1-to-many: drop the arg attrs. This is what LLVM conversion does as
      // well.
      if (mapping->size != 1)
        continue;
      result[mapping->inputNo] = cast<DictionaryAttr>(argAttrs[i]);
    }
  }
  // TODO: also handle result arguments.
  return result;
}

namespace {
/// Rewrite `func.func` with a converted signature. The biggest item that should
/// change is `memref` type arguments should be expanded.
struct RewriteFunc : ConvertOpToExecutorPattern<func::FuncOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType funcType = op.getFunctionType();
    TypeConverter::SignatureConversion sigConvert(funcType.getNumInputs());
    FunctionType newFuncType = dyn_cast_or_null<FunctionType>(
        getTypeConverter()->convertFunctionSignature(funcType, sigConvert));
    if (!newFuncType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");

    SmallVector<NamedAttribute> attributes = getConvertedFunctionAttributes(op);
    FailureOr<SmallVector<DictionaryAttr>> argAttrDicts =
        getConvertedArgAttrs(op, newFuncType, sigConvert);
    if (failed(argAttrDicts))
      return rewriter.notifyMatchFailure(op, "failed to convert arg attrs");

    auto replacement = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getName(), newFuncType, attributes, *argAttrDicts);
    rewriter.inlineRegionBefore(op.getBody(), replacement.getBody(),
                                replacement.end());
    if (failed(rewriter.convertRegionTypes(&replacement.getBody(),
                                           *getTypeConverter(), &sigConvert)))
      return rewriter.notifyMatchFailure(op, "failed to convert region types");

    // Finally, erase the op.
    rewriter.eraseOp(op);
    return success();
  }
};

// We must rewrite the return op to replace operands with equivalent operands
// from adaptor.
struct RewriteReturn : ConvertOpToExecutorPattern<func::ReturnOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
/// Convert `func.call`.
struct ConvertCall : public ConvertOpToExecutorPattern<func::CallOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return failure();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> operands = convertFuncCallOperands(
        rewriter, op.getLoc(), op.getOperands(), adaptor.getOperands(),
        getTypeConverter()->getOptions().memrefArgPassingConvention);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, types, op.getCallee(),
                                              operands);
    return success();
  }
};
} // namespace

void executor::populateFuncToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter) {
  patterns.add<RewriteFunc, ConvertCall, RewriteReturn>(typeConverter,
                                                        patterns.getContext());
}

void executor::populateControlFlowToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter) {
  patterns.add<ConvertCfBranch, ConvertCfCondBranch>(typeConverter,
                                                     patterns.getContext());
}

void executor::populateArithToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter) {
  patterns
      .add<RewriteConst,
           ConvertBinaryArithToExecute<arith::AddIOp, executor::AddIOp>,
           ConvertBinaryArithToExecute<arith::AddFOp, executor::AddFOp>,
           ConvertBinaryArithToExecute<arith::MulIOp, executor::MulIOp>,
           ConvertBinaryArithToExecute<arith::MulFOp, executor::MulFOp>,
           ConvertBinaryArithToExecute<arith::SubIOp, executor::SubIOp>,
           ConvertBinaryArithToExecute<arith::SubFOp, executor::SubFOp>,
           ConvertBinaryArithToExecute<arith::DivSIOp, executor::SDivIOp>,
           ConvertBinaryArithToExecute<arith::DivFOp, executor::DivFOp>,
           ConvertBinaryArithToExecute<arith::ShLIOp, executor::ShiftLeftIOp>,
           ConvertBinaryArithToExecute<arith::ShRUIOp,
                                       executor::ShiftRightLogicalIOp>,
           ConvertBinaryArithToExecute<arith::ShRSIOp,
                                       executor::ShiftRightArithmeticIOp>,
           ConvertBinaryArithToExecute<arith::AndIOp, executor::BitwiseAndIOp>,
           ConvertBinaryArithToExecute<arith::OrIOp, executor::BitwiseOrIOp>,
           ConvertBinaryArithToExecute<arith::XOrIOp, executor::BitwiseXOrIOp>,
           ConvertUnaryCastToExecute<arith::BitcastOp, executor::BitcastOp>,
           ConvertUnaryCastToExecute<arith::SIToFPOp, executor::SIToFPOp>,
           ConvertUnaryCastToExecute<arith::FPToSIOp, executor::FPToSIOp>,
           ConvertArithIndexCastOp, ConvertArithSelect, ConvertCmpI,
           ConvertCmpF>(typeConverter, patterns.getContext());
}

namespace {
class ConvertStdToExecutorPass
    : public mlir::executor::impl::ConvertStdToExecutorPassBase<
          ConvertStdToExecutorPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // We then eliminate all other illegal ops/types also making 'index' type
    // illegal.
    {
      LowerToExecutorOptions opts;
      opts.indexType = IntegerType::get(ctx, indexBitwidth);
      opts.memrefArgPassingConvention =
          usePackedMemRefCConv ? executor::MemRefArgPassingConvention::Packed
                               : executor::MemRefArgPassingConvention::Unpacked;
      Operation *op = getOperation();
      FailureOr<DataLayout> dataLayout =
          executor::setDataLayoutSpec(op, indexBitwidth, 64);
      if (failed(dataLayout)) {
        emitError(op->getLoc())
            << "failed to set DataLayout; op has DLTI spec that is "
               "inconsistent with provided options";
        return signalPassFailure();
      }
      ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));
      ExecutorConversionTarget target(getContext());
      target.addIllegalDialect<arith::ArithDialect>();
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
      target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
        return typeConverter.isLegal(op->getOperandTypes());
      });
      target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op.getResultTypes());
      });
      target.addDynamicallyLegalDialect<cf::ControlFlowDialect>(
          [&](Operation *op) {
            return typeConverter.isLegal(op->getOperandTypes());
          });

      RewritePatternSet patterns(&getContext());
      executor::populateFuncToExecutorPatterns(patterns, typeConverter);
      executor::populateArithToExecutorPatterns(patterns, typeConverter);
      executor::populateControlFlowToExecutorPatterns(patterns, typeConverter);

      // Add the math-to-executor patterns.
      registerConversionPDLFunctions(patterns);
      populateGeneratedPDLLPatterns(patterns,
                                    PDLConversionConfig(&typeConverter));

      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};
} // namespace
