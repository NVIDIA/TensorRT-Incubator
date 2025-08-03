//===- AssignMemorySpaces.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
///  Implementation of the `plan-assign-memory-spaces` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "plan-assign-memory-spaces"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANASSIGNMEMORYSPACESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {

// Generic pattern that rewrites any op by rewriting its operands and result
// types. Regions are also rewritten.
class GenericConvertSpace : public ConversionPattern {
public:
  GenericConvertSpace(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (isa<arith::ConstantOp, bufferization::AllocTensorOp>(op))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region &before = std::get<0>(regions);
      Region &parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Apply special conversion logic for `bufferization.alloc_tensor` operations.
/// It has a `memory_space` attribute that acts as a constraint.
/// memory space of the allocated tensor.
class ConvertAllocTensorPattern
    : public OpConversionPattern<bufferization::AllocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto originalType = dyn_cast<RankedTensorType>(op.getType());
    if (!originalType)
      return failure();

    auto resultConstraint =
        dyn_cast_or_null<MemorySpaceAttr>(originalType.getEncoding());
    auto opMemorySpaceConstraint =
        dyn_cast_if_present<MemorySpaceAttr>(op.getMemorySpaceAttr());

    auto expectedResultType = dyn_cast_if_present<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!expectedResultType)
      return failure();

    MemorySpaceAttr constraint =
        opMemorySpaceConstraint ? opMemorySpaceConstraint : resultConstraint;

    RankedTensorType constraintedType =
        constraint ? originalType.cloneWithEncoding(constraint)
                   : expectedResultType;

    if (adaptor.getCopy()) {
      auto castToConstraint = rewriter.create<TransferOp>(
          op.getLoc(), constraintedType, adaptor.getCopy());
      auto castOp = rewriter.create<TransferOp>(op.getLoc(), expectedResultType,
                                                castToConstraint);
      rewriter.replaceOp(op, castOp);
      return success();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getResult().setType(constraintedType);
      op.setMemorySpaceAttr(constraintedType.getEncoding());
      op.getCopyMutable().clear();
    });
    rewriter.setInsertionPointAfter(op);

    auto newAllocOp = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), constraintedType,
        /*dynamic_dimensions=*/adaptor.getDynamicSizes(),
        /*copy=*/Value{},
        /*size_hint=*/Value{},
        /*memory_space=*/constraintedType.getEncoding());
    auto castOp = rewriter.create<TransferOp>(op.getLoc(), expectedResultType,
                                              newAllocOp.getResult());
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

// A pattern that converts the type of the attribute used as an operand for
// arith.constant
class ConvertConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  ConvertConstantPattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = dyn_cast_if_present<ShapedType>(
        typeConverter->convertType(op.getType()));
    if (!newType)
      return failure();

    ElementsAttr newAttr{};
    if (auto elementsAttr = dyn_cast<DenseElementsAttr>(op.getValue()))
      newAttr = elementsAttr.reshape(newType);
    if (auto resourceAttr =
            dyn_cast<DenseResourceElementsAttr>(op.getValue())) {
      DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
      newAttr = DenseResourceElementsAttr::get(newType, handle);
    }
    if (!newAttr)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
    return success();
  }
};
} // namespace

namespace {

/// A type converter that adds a MemorySpaceAttr the the encoding of tensor
/// types. TensorTypes are legal only if they have the required encoding.
class TensorEncodingConverter : public TypeConverter {
public:
  TensorEncodingConverter(MLIRContext &context, plan::MemorySpace encoding)
      : requiredMemorySpace{plan::MemorySpaceAttr::get(&context, encoding)} {
    addConversion([&](Type type) -> std::optional<Type> { return type; });
    addConversion([&](RankedTensorType type) -> std::optional<Type> {
      return type.cloneWithEncoding(requiredMemorySpace);
    });
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<TransferOp>(loc, resultType, inputs.front());
    });
    addTargetMaterialization([&](OpBuilder &builder, TypeRange resultTypes,
                                 ValueRange inputs,
                                 Location loc) -> SmallVector<Value> {
      return {
          builder.create<TransferOp>(loc, resultTypes.front(), inputs.front())
              .getResult()};
    });
  }

  /// Convert a function signature, accounting for constraints specified in the
  /// arg/result attributes.
  FunctionType convertFuncSignature(func::FuncOp func) const {
    FunctionType funcType = func.getFunctionType();
    SmallVector<Type> newInputs, newResults;
    for (unsigned i = 0, e = funcType.getNumInputs(); i != e; ++i)
      newInputs.push_back(convertFuncSignatureElement(funcType.getInput(i),
                                                      func.getArgAttrDict(i)));
    for (unsigned i = 0, e = funcType.getNumResults(); i != e; ++i)
      newResults.push_back(convertFuncSignatureElement(
          funcType.getResult(i), func.getResultAttrDict(i)));
    return FunctionType::get(func.getContext(), newInputs, newResults);
  }

private:
  /// Convert a single element (arg or result type) of a function signature. The
  /// `dict` should contain the arg/result attributes or nullptr if not present.
  Type convertFuncSignatureElement(Type type, DictionaryAttr dict) const {
    if (auto rtt = dyn_cast<RankedTensorType>(type)) {
      if (auto constraint =
              dict ? dyn_cast_if_present<plan::MemorySpaceAttr>(dict.get(
                         PlanDialect::getMemorySpaceConstraintAttrName()))
                   : nullptr)
        return rtt.cloneWithEncoding(constraint);
      if (auto existing =
              dyn_cast_if_present<MemorySpaceAttr>(rtt.getEncoding()))
        return rtt;
    }
    return convertType(type);
  }

  plan::MemorySpaceAttr requiredMemorySpace;
};
} // namespace

/// Convert the block arguments for a single block where the RankedTensorTypes
/// may have received an updated encoding.
static void applySignatureConversion(RewriterBase &rewriter, Block *block,
                                     const TensorEncodingConverter &converter,
                                     TypeRange convertedTypes) {
  OpBuilder::InsertionGuard g(rewriter);
  assert(convertedTypes.size() == block->getNumArguments() &&
         "convertedTypes size mismatch");
  for (BlockArgument arg : block->getArguments()) {
    Type origType = arg.getType();
    if (origType == convertedTypes[arg.getArgNumber()])
      continue;
    auto castOp = rewriter.create<TransferOp>(
        arg.getLoc(), convertedTypes[arg.getArgNumber()], arg);
    rewriter.replaceAllUsesExcept(arg, castOp, castOp);
    arg.setType(convertedTypes[arg.getArgNumber()]);
  }
}

/// Convert the block arguments for all Blocks in a function body where the
/// RankedTensorTypes may have received an updated encoding.
static LogicalResult
convertFuncRegionTypes(RewriterBase &rewriter, func::FuncOp funcOp,
                       const TensorEncodingConverter &converter,
                       FunctionType newType) {
  if (funcOp.isDeclaration())
    return success();

  Region *region = &funcOp.getBody();

  // Convert the arguments of each non-entry block within the region.
  for (Block &block :
       llvm::make_early_inc_range(llvm::drop_begin(*region, 1))) {
    rewriter.setInsertionPointToStart(&block);
    // Compute the signature for the block with the provided converter.
    std::optional<TypeConverter::SignatureConversion> conversion =
        converter.convertBlockSignature(&block);
    if (!conversion)
      return failure();
    // Convert the block with the computed signature.
    applySignatureConversion(rewriter, &block, converter,
                             conversion->getConvertedTypes());
  }

  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  applySignatureConversion(rewriter, &funcOp.getBody().front(), converter,
                           newType.getInputs());

  return success();
}

/// Convert the operands and results of a function's callers after the `func`
/// has been updated to a new function signature type. The only types that can
/// change are RankedTensorTypes where the encoding has been updated. Therefore,
/// we only insert `tensor.cast` operations to cast the values back to their
/// original types.
struct LogicalResult convertFuncUsers(RewriterBase &rewriter, func::FuncOp func,
                                      const SymbolUserMap &userMap) {
  OpBuilder::InsertionGuard g(rewriter);
  FunctionType funcType = func.getFunctionType();
  auto handleValue = [&](Value value, Type desiredType) -> Value {
    if (value.getType() == desiredType)
      return value;
    return rewriter.create<TransferOp>(value.getLoc(), desiredType, value);
  };
  for (Operation *user : userMap.getUsers(func)) {
    auto call = dyn_cast<func::CallOp>(user);
    if (!call)
      continue;
    rewriter.setInsertionPoint(call);
    SmallVector<Value> newOperands;
    for (auto [newType, arg] :
         llvm::zip_equal(funcType.getInputs(), call.getOperands()))
      newOperands.push_back(handleValue(arg, newType));

    rewriter.setInsertionPointAfter(call);
    SmallVector<Value> replacements;
    for (auto [newType, result] :
         llvm::zip_equal(funcType.getResults(), call.getResults()))
      replacements.push_back(handleValue(result, newType));

    rewriter.modifyOpInPlace(call, [&]() {
      call.getOperandsMutable().assign(newOperands);
      for (auto [oldResult, replacement, newType] : llvm::zip_equal(
               call.getResults(), replacements, funcType.getResults())) {
        if (oldResult.getType() != newType) {
          oldResult.setType(newType);
          rewriter.replaceAllUsesExcept(oldResult, replacement,
                                        replacement.getDefiningOp());
        }
      }
    });
  }
  return success();
}

/// Convert the signature, block arguments, terminator operands, and caller
/// operands/results of a particular function by updating the types in place to
/// include the required memory space encodings. `tensor.cast` operations are
/// inserted to cast values back to their original types.
static LogicalResult
convertFuncOpTypes(func::FuncOp funcOp,
                   const TensorEncodingConverter &typeConverter,
                   RewriterBase &rewriter, const SymbolUserMap &userMap) {
  FunctionType type = funcOp.getFunctionType();
  FunctionType newType = typeConverter.convertFuncSignature(funcOp);
  if (type == newType)
    return success();
  if (failed(convertFuncRegionTypes(rewriter, funcOp, typeConverter, newType)))
    return failure();
  rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });

  if (!funcOp.isDeclaration()) {
    funcOp.walk([&](func::ReturnOp op) {
      rewriter.setInsertionPoint(op);
      SmallVector<Value> newTermOperands;
      bool changed = false;
      for (auto [newType, arg] :
           llvm::zip_equal(newType.getResults(), op.getOperands())) {
        if (arg.getType() == newType) {
          newTermOperands.push_back(arg);
          continue;
        }
        changed = true;
        auto cast = rewriter.create<TransferOp>(arg.getLoc(), newType, arg);
        newTermOperands.push_back(cast);
      }
      if (!changed)
        return;
      rewriter.modifyOpInPlace(
          op, [&]() { op.getOperandsMutable().assign(newTermOperands); });
    });
  }

  return convertFuncUsers(rewriter, funcOp, userMap);
}

/// Get the default memory space for a particular function.
static plan::MemorySpace getFuncitonDefaultEncoding(func::FuncOp func) {
  // The `plan.memory_space` attribute takes precedence over the cluster kind
  // default memory space.
  if (auto constraintOverride = func->getAttrOfType<plan::MemorySpaceAttr>(
          plan::PlanDialect::getMemorySpaceConstraintAttrName()))
    return constraintOverride.getValue();
  if (auto clusterKindAttr = func->getAttrOfType<CompilerBackendAttrInterface>(
          plan::PlanDialect::kFuncTargetKind))
    return clusterKindAttr.getDefaultMemorySpace();
  if (auto parentModule = func->getParentWithTrait<OpTrait::SymbolTable>())
    if (auto constraint = parentModule->getAttrOfType<plan::MemorySpaceAttr>(
            plan::PlanDialect::getMemorySpaceConstraintAttrName()))
      return constraint.getValue();
  return plan::MemorySpace::device;
}

/// Convert the signatures of functions and their callers by adding the
/// appropriate memory space attribute to all tensor types.
static LogicalResult
assignMemorySpacesToFunctionBoundaries(IRRewriter &rewriter, ModuleOp module) {
  SymbolTableCollection symbolTables;
  SymbolUserMap symbolUserMap(symbolTables, module);
  for (auto func : module.getOps<func::FuncOp>()) {
    plan::MemorySpace defaultEncoding = getFuncitonDefaultEncoding(func);
    TensorEncodingConverter converter(*func.getContext(), defaultEncoding);
    if (failed(convertFuncOpTypes(func, converter, rewriter, symbolUserMap)))
      return failure();
  }
  return success();
}

static bool hasMemorySpaceEncoding(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return false;
  return dyn_cast_if_present<MemorySpaceAttr>(tensorType.getEncoding()) !=
         nullptr;
}

static LogicalResult applyConversionToFunction(func::FuncOp func) {
  MLIRContext *context = func.getContext();
  plan::MemorySpace defaultEncoding = getFuncitonDefaultEncoding(func);
  TensorEncodingConverter converter(*context, defaultEncoding);

  // Ops are legal if they are in a nested module or if their operand and
  // result types are legal.
  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return converter.isLegal(op->getOperandTypes()) &&
           converter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
    return converter.isLegal(op.getType()) &&
           converter.isLegal(op.getValue().getType());
  });
  target.addDynamicallyLegalOp<TransferOp>([&](TransferOp op) {
    return hasMemorySpaceEncoding(op.getType()) &&
           hasMemorySpaceEncoding(op.getOperand().getType());
  });
  target.addDynamicallyLegalOp<tensor::CastOp>([&](tensor::CastOp op) {
    auto sourceType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto destType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceType || !destType ||
        !isa_and_present<plan::MemorySpaceAttr>(destType.getEncoding()) ||
        !isa_and_present<plan::MemorySpaceAttr>(sourceType.getEncoding()) ||
        destType.getEncoding() != sourceType.getEncoding())
      return false;
    return true;
  });
  target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
      [&](bufferization::AllocTensorOp op) {
        if (op.getCopy())
          return false;
        return hasMemorySpaceEncoding(op.getType());
      });
  target.addLegalDialect<func::FuncDialect>();

  RewritePatternSet patterns(context);
  patterns.add<GenericConvertSpace, ConvertConstantPattern,
               ConvertAllocTensorPattern>(converter, context);
  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  if (failed(applyFullConversion(func, target, std::move(patterns))))
    return emitError(func.getLoc(), "failed to assign memory spaces");
  return success();
}

namespace {
struct AssignMemorySpacesPass
    : public plan::impl::PlanAssignMemorySpacesPassBase<
          AssignMemorySpacesPass> {
  void runOnOperation() override {

    MLIRContext *context = &getContext();

    IRRewriter rewriter(context);

    /// Update all function signatures and their callers to include the required
    /// memory space encodings.
    if (failed(
            assignMemorySpacesToFunctionBoundaries(rewriter, getOperation())))
      return signalPassFailure();

    for (auto func :
         llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>())) {
      if (failed(applyConversionToFunction(func)))
        return signalPassFailure();
    }
  }
};
} // namespace
