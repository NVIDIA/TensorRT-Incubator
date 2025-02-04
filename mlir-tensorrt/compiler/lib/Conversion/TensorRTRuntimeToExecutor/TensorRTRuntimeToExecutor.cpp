//===- TensorRTRuntimeToExecutor.cpp --------------------------------------===//
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
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORRTRUNTIMETOEXECUTORPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;
using namespace mlir::cuda;

namespace {
struct TensorRTRuntimeBuiltinCallBuilders {
  Type indexType;
  MLIRContext *ctx = indexType.getContext();
  Type hostPointerType =
      executor::PointerType::get(ctx, executor::MemoryType::host);

  ExecutorCallBuilder createRuntime = {
      ctx, "_trtrt_create_runtime", hostPointerType, {}};

  ExecutorCallBuilder loadEngine = {ctx,
                                    "_trtrt_load",
                                    hostPointerType,
                                    {/*runtime*/ hostPointerType,
                                     /*serialized engine*/ hostPointerType,
                                     /*serialized engine size*/ indexType}};

  ExecutorCallBuilder createContext = {ctx, "_trtrt_create_context",
                                       hostPointerType, hostPointerType};
};
} // namespace

template <typename T>
struct ConvertTRTRTOpToExecutorPattern : public ConvertOpToExecutorPattern<T> {
  using ConvertOpToExecutorPattern<T>::ConvertOpToExecutorPattern;
  MLIRContext *ctx{this->getContext()};
  TensorRTRuntimeBuiltinCallBuilders builderUtils{
      this->getTypeConverter()->getIndexType()};
};

static GlobalOp getOrCreateRuntimeGlobalOp(
    RewriterBase &rewriter, ModuleOp op,
    const TensorRTRuntimeBuiltinCallBuilders &builderUtils) {
  return getOrCreateGlobalOp(
      rewriter, op.getLoc(), op, "tensorrt_runtime",
      builderUtils.hostPointerType, false,
      [&](OpBuilder &nested, Location loc) {
        Value runtime =
            builderUtils.createRuntime.create(nested, loc, op, {}).getResult(0);
        nested.create<ReturnOp>(loc, runtime);
      });
}

/// Create a `executor.global` to load the TensorRT engine/execution context.
static GlobalOp getOrCreateExecutionContextGlobal(
    RewriterBase &rewriter, trtrt::CompiledFuncOp trtFunc,
    ConstantResourceOp resourceOp, GlobalOp runtimeGlobal,
    const TensorRTRuntimeBuiltinCallBuilders &callBuilder) {
  std::string name = (trtFunc.getName() + "_exec_ctx").str();
  auto parentModule = trtFunc->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(parentModule);
  executor::PointerType hostPointerType = executor::PointerType::get(
      rewriter.getContext(), executor::MemoryType::host);
  return getOrCreateGlobalOp(
      rewriter, trtFunc.getLoc(), parentModule, name, hostPointerType, true,
      [&](OpBuilder &nested, Location loc) {
        ImplicitLocOpBuilder ib(loc, nested);
        Value data = ib.create<ConstantResourceLoadOp>(
            FlatSymbolRefAttr::get(resourceOp));
        // Use 'executor.getoffset' as a portable way of calculating the final
        // buffer size. The data type for TRT engines should always be 'i8', but
        // this is more fool-proof.
        ShapedType dataType = resourceOp.getValue().getShapedType();
        Value dataSize = ib.create<GetOffsetOp>(
            callBuilder.indexType, dataType.getElementType(),
            ArrayRef<OpFoldResult>{
                rewriter.getI64IntegerAttr(dataType.getNumElements())});
        Value runtime = ib.create<GetGlobalOp>(
            hostPointerType, FlatSymbolRefAttr::get(runtimeGlobal));
        Value engine = callBuilder.loadEngine
                           .create(ib, trtFunc.getLoc(), parentModule,
                                   {runtime, data, dataSize})
                           .getResult(0);
        Value context =
            callBuilder.createContext
                .create(ib, trtFunc.getLoc(), parentModule, {engine})
                .getResult(0);
        ib.create<ReturnOp>(context);
      });
}

namespace {

/// Convert `tensorrt.compile` to a set of globals declarations/global
/// retrievals representing the deserialization of the TensorRT engine and the
/// retrieval of the execution context. The primary prequisite for this pattern
/// is that the serialized engine data must be available on the reference
/// TensorRT engine.
struct ConvertCompile
    : public ConvertTRTRTOpToExecutorPattern<trtrt::GetFunctionOp> {
  using ConvertTRTRTOpToExecutorPattern::ConvertTRTRTOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(trtrt::GetFunctionOp getFunctionOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = getFunctionOp->getParentOfType<ModuleOp>();
    auto op =
        module.lookupSymbol<trtrt::CompiledFuncOp>(getFunctionOp.getModule());
    if (!op)
      return failure();

    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(op, module);
    SmallVector<trtrt::GetFunctionOp> users;
    if (uses) {
      for (auto use : *uses) {
        if (auto user = llvm::dyn_cast<trtrt::GetFunctionOp>(use.getUser())) {
          users.push_back(user);
          continue;
        }
        return failure();
      }
    }

    GlobalOp runtimeGlobal =
        getOrCreateRuntimeGlobalOp(rewriter, module, builderUtils);
    ConstantResourceOp resourceGlobal = getOrCreateConstantResourceDeclaration(
        rewriter, op.getLoc(), module, op.getSymName(), op.getValue());
    GlobalOp executionContextGlobal = getOrCreateExecutionContextGlobal(
        rewriter, op, resourceGlobal, runtimeGlobal, builderUtils);
    resourceGlobal->moveAfter(runtimeGlobal);
    executionContextGlobal->moveAfter(resourceGlobal);

    for (trtrt::GetFunctionOp user : users) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(user);
      Value context = rewriter.create<executor::GetGlobalOp>(
          op.getLoc(), builderUtils.hostPointerType,
          FlatSymbolRefAttr::get(executionContextGlobal));
      rewriter.replaceOp(user, context);
      continue;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert `tensorrt.enqueue` to `executor.call`.
struct ConvertEnqueueToCall
    : public ConvertTRTRTOpToExecutorPattern<trtrt::EnqueueOp> {
  using ConvertTRTRTOpToExecutorPattern::ConvertTRTRTOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // For now just do simple substitution to get the name.
    std::string funcName;
    funcName =
        "_" + llvm::join(llvm::split(op->getName().getStringRef(), "."), "_");
    if (op->getNumResults() > 0)
      return failure();

    SmallVector<Value> newOperands = {adaptor.getExecutionContext(),
                                      adaptor.getStream()};
    SmallVector<Value> argTablePack;

    auto createMemRefAndExractPtr = [&](Value oldVal, Value newVal) {
      auto memrefType = cast<MemRefType>(oldVal.getType());
      if (!memrefType)
        return failure();
      assert(isa<TableType>(newVal.getType()));
      executor::MemRefDescriptor memref(newVal, memrefType);
      Value offset =
          convertOffsetInElementsToBytes(b, memref.offset(b), memrefType);

      // Append the aligned pointer and offset.
      argTablePack.append({memref.alignedPtr(b), offset});

      // Append the rank followed by the shape integers.
      argTablePack.push_back(
          this->createIndexConstant(b, memrefType.getRank()));
      for (int64_t i = 0; i < memrefType.getRank(); i++)
        argTablePack.push_back(memref.size(b, i));

      return success();
    };

    for (auto [oldVal, newVal] :
         llvm::zip(op.getInputs(), adaptor.getInputs())) {
      if (failed(createMemRefAndExractPtr(oldVal, newVal)))
        return failure();
    }
    for (auto [oldVal, newVal] : llvm::zip(op.getOuts(), adaptor.getOuts())) {
      if (failed(createMemRefAndExractPtr(oldVal, newVal)))
        return failure();
    }

    // Create the table containing the pointer/offset args and append it to the
    // arguments for the call op.
    Value args = b.create<executor::CreateTableOp>(
        executor::TableType::get(rewriter.getContext(),
                                 llvm::to_vector(TypeRange(argTablePack))),
        argTablePack);
    newOperands.push_back(args);

    auto parentModule = op->getParentOfType<ModuleOp>();
    auto enqueueFunc = getOrInsertFuncDeclaration(
        rewriter, op.getLoc(), parentModule, funcName,
        ExecutorFunctionType::get(rewriter.getContext(),
                                  {adaptor.getExecutionContext().getType(),
                                   adaptor.getStream().getType()},
                                  {}, rewriter.getUnitAttr()));

    rewriter.replaceOpWithNewOp<CallOp>(
        op, TypeRange{}, enqueueFunc.getLeafReference(), newOperands);

    return success();
  }
};

/// Convert `tensorrt.enqueue_alloc` to `executor.call`.
struct ConvertEnqueueAllocToCall
    : public ConvertTRTRTOpToExecutorPattern<trtrt::EnqueueAllocOp> {
  using ConvertTRTRTOpToExecutorPattern::ConvertTRTRTOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Function name for the enqueue alloc operation
    StringRef funcName = "_trtrt_enqueue_alloc";

    // Create new operands for the call op
    SmallVector<Value> newOperands = {adaptor.getExecutionContext(),
                                      adaptor.getStream()};

    // The second operand is a descriptor which contains several values
    // stored as a flat list of integers:
    // - the number of results
    // - the rank, ptr, shape, and strides of each memref operand.
    const int64_t numResults = op->getNumResults();
    int64_t totalElements = 1;
    for (int i = 0; i < numResults; ++i) {
      int64_t rank = cast<MemRefType>(op.getResult(i).getType()).getRank();
      totalElements += 2 + rank * 2; // Increment for shape and stride
    }
    SmallVector<Type> outputTypes(totalElements, i64Type);
    auto hostPointerType = executor::PointerType::get(
        op->getContext(), executor::MemoryType::host);
    auto structType = executor::TableType::get(op->getContext(), outputTypes);
    Value one = createIndexConstant(b, 1);

    // Create output descriptors
    Value outputDescriptors = rewriter.create<executor::AllocaOp>(
        op->getLoc(), hostPointerType, one, /*alignment=*/IntegerAttr{},
        structType);

    // Store number of results. It is always a first value in output
    // descriptors
    Value numResultsOffset = b.create<executor::GetOffsetOp>(
        i64Type, structType,
        ArrayRef<OpFoldResult>{rewriter.getI64IntegerAttr(0),
                               rewriter.getI64IntegerAttr(0)});

    auto resultValue =
        b.create<executor::ConstantOp>(rewriter.getI64IntegerAttr(numResults));
    b.create<executor::StoreOp>(outputDescriptors, numResultsOffset,
                                resultValue);

    // Store rank per result
    for (int i = 0, descriptorOffset = 1; i < numResults; ++i) {
      Value rankOffset = b.create<executor::GetOffsetOp>(
          i64Type, structType,
          ArrayRef<OpFoldResult>{this->createIndexConstant(b, 0),
                                 rewriter.getI64IntegerAttr(descriptorOffset)});
      int64_t rank = cast<MemRefType>(op.getResult(i).getType()).getRank();
      auto rankValue =
          b.create<executor::ConstantOp>(rewriter.getI64IntegerAttr(rank));
      b.create<executor::StoreOp>(outputDescriptors, rankOffset, rankValue);
      descriptorOffset +=
          2 + rank * 2; // Skip the fields that are populated by the callee.
    }

    SmallVector<Value> inputMemrefValues;

    auto createMemRefAndExractPtr = [&](Value oldVal, Value newVal) {
      auto memrefType = cast<MemRefType>(oldVal.getType());
      if (!memrefType)
        return failure();
      assert(isa<TableType>(newVal.getType()));
      executor::MemRefDescriptor memref(newVal, memrefType);
      Value offset =
          convertOffsetInElementsToBytes(b, memref.offset(b), memrefType);

      // Append the aligned pointer and offset.
      inputMemrefValues.append({memref.alignedPtr(b), offset});

      // Append the rank followed by the shape integers.
      inputMemrefValues.push_back(
          this->createIndexConstant(b, memrefType.getRank()));
      for (int64_t i = 0; i < memrefType.getRank(); i++)
        inputMemrefValues.push_back(memref.size(b, i));

      return success();
    };

    // Insert output descriptors
    newOperands.push_back(outputDescriptors);

    // Create input memref table
    for (auto [oldVal, newVal] :
         llvm::zip(op.getInputs(), adaptor.getInputs())) {
      if (failed(createMemRefAndExractPtr(oldVal, newVal)))
        return failure();
    }

    // Create the table containing the pointer/offset args and append it to the
    // arguments for the call op.
    Value args = b.create<executor::CreateTableOp>(
        executor::TableType::get(rewriter.getContext(),
                                 llvm::to_vector(TypeRange(inputMemrefValues))),
        inputMemrefValues);
    newOperands.push_back(args);

    // Create and insert the function declaration
    auto parentModule = op->getParentOfType<ModuleOp>();
    auto enqueueAllocFunc = getOrInsertFuncDeclaration(
        rewriter, op.getLoc(), parentModule, funcName,
        ExecutorFunctionType::get(rewriter.getContext(),
                                  {adaptor.getExecutionContext().getType(),
                                   adaptor.getStream().getType(),
                                   outputDescriptors.getType()},
                                  {}, rewriter.getUnitAttr()));

    // Create the call op
    b.create<CallOp>(TypeRange{}, enqueueAllocFunc.getLeafReference(),
                     newOperands);

    // Create output memrefs from output descriptors
    SmallVector<Value> results;
    // Initialize output descriptor offset to skip number of results.
    // `outputDescOffset` is used to retrieve rank, ptr, shapes, and strides per
    // result.
    unsigned outputDescOffset = 1;
    Value zero = this->createIndexConstant(b, 0);
    for (auto [idx, result] : llvm::enumerate(op.getResults())) {
      MemRefType memrefType = cast<MemRefType>(result.getType());
      unsigned rank = memrefType.getRank();
      Value devicePtrOffset = b.create<executor::GetOffsetOp>(
          i64Type, structType,
          ArrayRef<OpFoldResult>{
              zero, rewriter.getI64IntegerAttr(outputDescOffset++)});

      std::optional<MemoryType> memSpace = this->getMemorySpace(memrefType);
      if (!memSpace)
        return failure();

      Type ptrType = executor::PointerType::get(getContext(), *memSpace);

      Value intPtr = b.create<executor::LoadOp>(i64Type, outputDescriptors,
                                                devicePtrOffset);
      Value alignedPtr = b.create<executor::IntToPtrOp>(ptrType, intPtr);

      SmallVector<Value, 4> shapes, strides;
      for (unsigned r = 0; r < rank; ++r) {
        Value shapeOffset = b.create<executor::GetOffsetOp>(
            i64Type, structType,
            ArrayRef<OpFoldResult>{
                zero, rewriter.getI64IntegerAttr(outputDescOffset++)});
        Value shape =
            b.create<executor::LoadOp>(i64Type, outputDescriptors, shapeOffset);
        shapes.push_back(shape);
      }

      for (unsigned r = 0; r < rank; ++r) {
        Value strideOffset = b.create<executor::GetOffsetOp>(
            i64Type, structType,
            ArrayRef<OpFoldResult>{
                zero, rewriter.getI64IntegerAttr(outputDescOffset++)});
        Value shape = b.create<executor::LoadOp>(i64Type, outputDescriptors,
                                                 strideOffset);
        shapes.push_back(shape);
      }

      results.push_back(MemRefDescriptor::fromComponents(
          b, *getTypeConverter(), memrefType, alignedPtr, alignedPtr, zero,
          shapes, strides));
    }

    rewriter.replaceOp(op, results);

    return success();
  }

  Type i64Type{IntegerType::get(getContext(), 64)};
};
} // namespace

namespace {
class TensorRTRuntimeToExecutorPass
    : public mlir::impl::ConvertTensorRTRuntimeToExecutorPassBase<
          TensorRTRuntimeToExecutorPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToExecutorOptions opts;
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    opts.memrefArgPassingConvention =
        usePackedMemRefCConv ? MemRefArgPassingConvention::Packed
                             : MemRefArgPassingConvention::Unpacked;
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
    executor::PointerType hostPointerType =
        PointerType::get(ctx, MemoryType::host);

    typeConverter.addConversion([&](Type t) -> std::optional<Type> {
      if (isa<trtrt::EngineType, cuda::StreamType, trtrt::ExecutionContextType>(
              t))
        return hostPointerType;
      return {};
    });

    // Convert `trtrt.enqueue|create_runtime|execution_context|load` to
    // `executor.call` and function declarations.
    {
      ConversionTarget target(*ctx);
      target.addIllegalDialect<trtrt::TensorRTRuntimeDialect>();
      target.addLegalDialect<ExecutorDialect, CUDADialect>();
      target.addLegalOp<UnrealizedConversionCastOp>();

      RewritePatternSet patterns(&getContext());
      patterns
          .add<ConvertEnqueueToCall, ConvertEnqueueAllocToCall, ConvertCompile>(
              typeConverter, ctx);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};
} // namespace
