//===- TensorRTRuntimeToLLVM.cpp -----------------------------------------===//
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
/// Implementation of the `convert-tensorrt-runtime-to-llvm` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/TensorRTRuntimeToLLVM/TensorRTRuntimeToLLVM.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/CUDAToLLVM/CUDAToLLVM.h"
#include "mlir-tensorrt/Conversion/LLVMCommon/LLVMCommon.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Conversion/PlanToLLVM/PlanToLLVM.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORRTRUNTIMETOLLVMPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TRTRTToLLVMBuilderUtils {
  TRTRTToLLVMBuilderUtils(MLIRContext *ctx) : ctx(ctx) {}

  MLIRContext *ctx;
  Type i64Type{IntegerType::get(ctx, 64)};
  Type i32Type{IntegerType::get(ctx, 32)};
  Type llvmPtrType{LLVM::LLVMPointerType::get(ctx)};
  Type llvmVoidType{LLVM::LLVMVoidType::get(ctx)};

  LLVMOpaqueCallBuilder trtRuntimeCreateCallBuilder = {
      "mtrt_tensorrt_runtime_create", llvmPtrType, {}};

  LLVMOpaqueCallBuilder trtRuntimeDestroyCallBuilder = {
      "mtrt_tensorrt_runtime_destroy", llvmVoidType, {llvmPtrType}};

  LLVMOpaqueCallBuilder enqueueCallBuilder = {
      "mtrt_tensorrt_enqueue",
      llvmVoidType,
      {/*executionContext*/ llvmPtrType, /*stream*/ llvmPtrType,
       /*numInputs*/ i32Type, llvmPtrType, /*numOutputs*/ i32Type,
       llvmPtrType}};

  LLVMOpaqueCallBuilder enqueueAllocCallBuilder = {
      "mtrt_tensorrt_enqueue_alloc",
      llvmVoidType,
      {llvmPtrType, llvmPtrType, i32Type, llvmPtrType, i32Type, llvmPtrType}};

  LLVMOpaqueCallBuilder loadEngineCallBuilder = {
      "mtrt_load_tensorrt_engine",
      llvmPtrType,
      {/*tensorrt runtime handle*/ llvmPtrType,
       /*serialized engine*/ llvmPtrType,
       /*serialized engine size*/ i64Type}};

  LLVMOpaqueCallBuilder loadEngineFromFileCallBuilder = {
      "mtrt_load_tensorrt_engine_from_file",
      llvmPtrType,
      {/*tensorrt runtime handle*/ llvmPtrType,
       /*serialized engine*/ llvmPtrType,
       /*serialized engine size*/ i64Type}};

  LLVMOpaqueCallBuilder destroyExecutionContextCallBuilder = {
      "mtrt_tensorrt_execution_context_destroy", llvmVoidType, {llvmPtrType}};
};
} // namespace

template <typename T>
struct ConvertTRTRTOpToLLVMPattern : public ConvertOpToLLVMPattern<T> {
  ConvertTRTRTOpToLLVMPattern(const LLVMTypeConverter &typeConverter,
                              SymbolTableCollection *symbolTable = nullptr,
                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<T>(typeConverter, benefit) {}

protected:
  SymbolTableCollection *symbolTable;
  MLIRContext *ctx{this->getContext()};
  const TRTRTToLLVMBuilderUtils builderUtils{ctx};
};

namespace {
struct TensorRTRuntimeToLLVMOneShotConverter {
  TensorRTRuntimeToLLVMOneShotConverter(ModuleOp module,
                                        StringRef artifactsDir);
  ModuleOp module;
  std::string artifactsDir;
  MLIRContext *ctx{module.getContext()};
  SymbolTableCollection symbolTables;
  SymbolTable &symbolTable{symbolTables.getSymbolTable(module)};
  SymbolUserMap userMap{symbolTables, module};
  IRRewriter rewriter{module->getContext()};
  TRTRTToLLVMBuilderUtils builderUtils{module->getContext()};

  LLVM::GlobalOp tensorrtRuntimeGlobal{nullptr};
  int32_t ctorPriority = 10;
  int32_t dtorPriority = 0;

  LogicalResult convert(trtrt::GetFunctionOp op,
                        LLVM::GlobalOp executionContextGlobal);
  LogicalResult convert(trtrt::CompiledFuncOp op);
  LogicalResult convert();
};
} // namespace

TensorRTRuntimeToLLVMOneShotConverter::TensorRTRuntimeToLLVMOneShotConverter(
    ModuleOp module, StringRef artifactsDir)
    : module(module), artifactsDir(artifactsDir.str()) {}

LogicalResult TensorRTRuntimeToLLVMOneShotConverter::convert(
    trtrt::GetFunctionOp op, LLVM::GlobalOp executionContextGlobal) {
  Location loc = op.getLoc();
  Value ptr = rewriter.create<LLVM::AddressOfOp>(loc, executionContextGlobal);
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(),
      Value(rewriter.create<LLVM::LoadOp>(loc, builderUtils.llvmPtrType, ptr)));
  return success();
}

/// Convert 'trtrt.compile' operations into ctors. The initialization of
/// a TensorRT execution context doesn't have prior dependencies, so we don't
/// necessarily need to worry about the ctor priority here.
LogicalResult
TensorRTRuntimeToLLVMOneShotConverter::convert(trtrt::CompiledFuncOp op) {
  auto engineData = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!engineData || !engineData.getElementType().isInteger(8))
    return emitError(op.getLoc()) << "unhandled engine data attribute";
  StringRef engineName = op.getName();
  MLIRContext *ctx = op->getContext();
  ArrayRef<char> values = engineData.getRawData();
  auto data = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(engineData.size())},
                      IntegerType::get(ctx, 8)),
      engineName,
      HeapAsmResourceBlob::allocateAndCopyWithAlign(values, alignof(char),
                                                    /*dataIsMutable=*/true));

  Location loc = op.getLoc();
  LLVM::GlobalOp dataGlobalOp;
  if (artifactsDir.empty()) {
    dataGlobalOp = insertLLVMGlobal(
        rewriter, loc, (engineName + ".data").str(), /*constant=*/true,
        LLVM::LLVMArrayType::get(rewriter.getI8Type(), data.size()),
        LLVM::Linkage::Private, data, &symbolTable);
  } else {
    if (!llvm::sys::fs::is_directory(artifactsDir))
      return module.emitError() << "artifacts-directory does not exist";
    FailureOr<std::unique_ptr<llvm::ToolOutputFile>> outputFile =
        serializeElementsAttrToFile(loc, op.getValueAttr(), artifactsDir,
                                    (op.getName() + ".trt_plan.bin").str());
    if (failed(outputFile))
      return failure();
    (*outputFile)->keep();
  }

  LLVM::GlobalOp contextGlobalOp =
      insertLLVMGlobal(rewriter, loc, (engineName + ".context").str(),
                       /*constant=*/false, builderUtils.llvmPtrType,
                       LLVM::Linkage::Internal, Attribute{}, &symbolTable);

  // Use ctor function to load the context from the engine data.
  // We don't need to change ctor priority.
  insertLLVMCtorFunction(
      rewriter, loc, symbolTable, (engineName + "_context_init").str(),
      ctorPriority, [&](OpBuilder &nested, Location loc) {
        Value trtRuntime =
            nested
                .create<LLVM::LoadOp>(loc, builderUtils.llvmPtrType,
                                      nested.create<LLVM::AddressOfOp>(
                                          loc, tensorrtRuntimeGlobal))
                .getResult();
        Value engineContext{};
        if (dataGlobalOp) {
          Value engineAddr =
              nested.create<LLVM::AddressOfOp>(loc, dataGlobalOp);
          Value sizeVal = nested.create<LLVM::ConstantOp>(
              loc, nested.getI64IntegerAttr(engineData.size()));
          engineContext =
              builderUtils.loadEngineCallBuilder
                  .create(loc, nested, {trtRuntime, engineAddr, sizeVal},
                          &symbolTable)
                  .getResult();
        } else {
          std::string filename = (op.getName() + ".trt_plan.bin").str();
          Value filenamePtr = insertLLVMStringLiteral(
              rewriter, loc, filename, (op.getName() + "_filename").str());
          Value filenameSize = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(filename.size()));
          engineContext =
              builderUtils.loadEngineFromFileCallBuilder
                  .create(loc, nested, {trtRuntime, filenamePtr, filenameSize},
                          &symbolTable)
                  .getResult();
        }
        nested.create<LLVM::StoreOp>(
            loc, engineContext,
            nested.create<LLVM::AddressOfOp>(loc, contextGlobalOp));
      });

  insertLLVMCtorFunction(
      rewriter, loc, symbolTable, (engineName + "_context_deinit").str(),
      dtorPriority, [&](OpBuilder &nested, Location loc) {
        builderUtils.destroyExecutionContextCallBuilder.create(
            loc, nested,
            {nested.create<LLVM::LoadOp>(
                loc, builderUtils.llvmPtrType,
                nested.create<LLVM::AddressOfOp>(loc, contextGlobalOp))});
      });

  for (Operation *user : userMap.getUsers(op)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto getFuncOp = dyn_cast<trtrt::GetFunctionOp>(user)) {
      if (failed(convert(getFuncOp, contextGlobalOp)))
        return failure();
      continue;
    }
    return emitError(user->getLoc())
           << "unexpected dangling reference to TensorRT engine symbol";
  }

  // There are no longer references to the original func, we can remove it.
  symbolTable.remove(op);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult TensorRTRuntimeToLLVMOneShotConverter::convert() {
  rewriter.setInsertionPointToEnd(module.getBody());

  // Populate the global TensorRT runtime handle.
  tensorrtRuntimeGlobal =
      insertLLVMGlobal(rewriter, module.getLoc(), "tensorrt_runtime",
                       /*constant=*/false, builderUtils.llvmPtrType,
                       LLVM::Linkage::Internal, Attribute{}, &symbolTable);

  // Initialize it in a ctor.
  insertLLVMCtorFunction(
      rewriter, module.getLoc(), symbolTable, "tensorrt_runtime_init",
      ctorPriority, [&](OpBuilder &nested, Location loc) {
        Value handle =
            builderUtils.trtRuntimeCreateCallBuilder.create(loc, nested, {})
                .getResult();
        nested.create<LLVM::StoreOp>(
            loc, handle,
            nested.create<LLVM::AddressOfOp>(loc, tensorrtRuntimeGlobal));
      });

  ctorPriority -= 1;

  // Destory it in a dtor.
  insertLLVMDtorFunction(
      rewriter, module.getLoc(), symbolTable, "tensorrt_runtime_deinit",
      dtorPriority, [&](OpBuilder &nested, Location loc) {
        builderUtils.trtRuntimeDestroyCallBuilder.create(
            loc, nested,
            {nested.create<LLVM::LoadOp>(
                loc, builderUtils.llvmPtrType,
                nested.create<LLVM::AddressOfOp>(loc, tensorrtRuntimeGlobal))});
      });

  dtorPriority += 1;

  for (trtrt::CompiledFuncOp op :
       llvm::make_early_inc_range(module.getOps<trtrt::CompiledFuncOp>())) {
    if (userMap.getUsers(op).empty()) {
      symbolTable.remove(op);
      rewriter.eraseOp(op);
      continue;
    }

    if (failed(convert(op)))
      return failure();
  }
  return success();
}

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
static Value createArgumentPointerPack(RewriterBase &rewriter, Location loc,
                                       ValueRange convertedOperands) {
  SmallVector<Type> structTypes(convertedOperands.size());
  for (auto [i, arg] : llvm::enumerate(convertedOperands))
    structTypes[i] = arg.getType();

  MLIRContext *ctx = rewriter.getContext();
  Type structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), structTypes);
  Type i32Type = rewriter.getI32Type();
  Value one = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 1);
  auto pointerTy = LLVM::LLVMPointerType::get(ctx);
  Value argStruct =
      rewriter.create<LLVM::AllocaOp>(loc, pointerTy, structTy, one);
  Value numArgs =
      rewriter.create<LLVM::ConstantOp>(loc, i32Type, structTypes.size());
  Value argArray =
      rewriter.create<LLVM::AllocaOp>(loc, pointerTy, pointerTy, numArgs);
  for (auto [i, arg] : llvm::enumerate(convertedOperands)) {
    Value structMember = rewriter.create<LLVM::GEPOp>(
        loc, pointerTy, structTy, argStruct, ArrayRef<LLVM::GEPArg>{0, i});
    rewriter.create<LLVM::StoreOp>(loc, arg, structMember);
    Value arrayMember = rewriter.create<LLVM::GEPOp>(
        loc, pointerTy, pointerTy, argArray, ArrayRef<LLVM::GEPArg>{i});
    rewriter.create<LLVM::StoreOp>(loc, structMember, arrayMember);
  }
  return argArray;
}

namespace {

/// Convert `tensorrt.enqueue` to `llvm.call`.
struct ConvertEnqueueToCall
    : public ConvertTRTRTOpToLLVMPattern<trtrt::EnqueueOp> {
  using ConvertTRTRTOpToLLVMPattern::ConvertTRTRTOpToLLVMPattern;

  Value convertOffsetToBytes(ConversionPatternRewriter &rewriter, Location loc,
                             Type elementType, Value in) const {
    // Get element size.
    auto sizeInBytes = getSizeInBytes(loc, elementType, rewriter);
    return rewriter.create<LLVM::MulOp>(loc, in, sizeInBytes);
  }

  LogicalResult
  matchAndRewrite(trtrt::EnqueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Ensure that the 'trtrt.enqueue' operation is in bufferized form. It
    // should not have any results. Only the 'trtrt.enqueue_alloc' form can
    // allocate results.
    if (!op.hasPureBufferSemantics() || op->getNumResults() > 0)
      return rewriter.notifyMatchFailure(
          op, "must have pure buffer semantics with no results");

    // Create new operands for the call op

    Location loc = op.getLoc();
    Value numInputs = createIndexAttrConstant(
        rewriter, loc, builderUtils.i32Type, adaptor.getInputs().size());
    Value numOutputs = createIndexAttrConstant(
        rewriter, loc, builderUtils.i32Type, op.getOuts().size());

    SmallVector<Value> updatedInputs = promoteLLVMMemRefDescriptorsToUnranked(
        rewriter, loc, *getTypeConverter(), op.getInputs(),
        adaptor.getInputs());

    SmallVector<Value> updatedOutputs = promoteLLVMMemRefDescriptorsToUnranked(
        rewriter, loc, *getTypeConverter(), op.getOuts(), adaptor.getOuts());

    Value inputsPtr = createArgumentPointerPack(rewriter, loc, updatedInputs);
    Value outsPtr = createArgumentPointerPack(rewriter, loc, updatedOutputs);

    SmallVector<Value> newOperands = {adaptor.getExecutionContext(),
                                      adaptor.getStream(),
                                      numInputs,
                                      inputsPtr,
                                      numOutputs,
                                      outsPtr};

    builderUtils.enqueueCallBuilder.create(loc, rewriter, newOperands);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert `tensorrt.enqueue_alloc` to `llvm.call`.
struct ConvertEnqueueAllocToCall
    : public ConvertTRTRTOpToLLVMPattern<trtrt::EnqueueAllocOp> {
  using ConvertTRTRTOpToLLVMPattern::ConvertTRTRTOpToLLVMPattern;

  Value convertOffsetToBytes(ConversionPatternRewriter &rewriter, Location loc,
                             Type elementType, Value in) const {
    // Get element size.
    auto sizeInBytes = getSizeInBytes(loc, elementType, rewriter);
    return rewriter.create<LLVM::MulOp>(loc, in, sizeInBytes);
  }

  LogicalResult
  matchAndRewrite(trtrt::EnqueueAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Create new operands for the call op
    Location loc = op.getLoc();
    Value numInputs = createIndexAttrConstant(
        rewriter, loc, builderUtils.i32Type, adaptor.getInputs().size());
    Value numOutputs = createIndexAttrConstant(
        rewriter, loc, builderUtils.i32Type, op->getNumResults());

    SmallVector<Value> updatedInputs = promoteLLVMMemRefDescriptorsToUnranked(
        rewriter, loc, *getTypeConverter(), op.getInputs(),
        adaptor.getInputs());

    Value inputsPtr = createArgumentPointerPack(rewriter, loc, updatedInputs);

    // We now create all the result descriptors. These descriptors are initially
    // undef values since the call to the `enqueue_alloc` actually performs the
    // allocation. We promote to unranked descriptors and then pass these
    // unranked descriptors to the call to `mtrt_tensorrt_enqueue_alloc`.
    const unsigned numResults = op.getNumResults();
    SmallVector<Value> resultDescriptors;
    resultDescriptors.reserve(numResults);
    SmallVector<Type> resultStructTypes;
    resultStructTypes.reserve(numResults);
    for (MemRefType t : make_cast_range<MemRefType>(op.getResultTypes())) {
      auto llvmTargetDescriptorTy =
          dyn_cast_or_null<LLVM::LLVMStructType>(typeConverter->convertType(t));
      if (!llvmTargetDescriptorTy)
        return failure();
      auto rankedDescriptor =
          MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);
      resultStructTypes.push_back(Value(rankedDescriptor).getType());
      Value unranked = getUnrankedLLVMMemRefDescriptor(
          rewriter, loc, *getTypeConverter(), rankedDescriptor, t);
      resultDescriptors.push_back(unranked);
    }

    Value outputsPtr =
        createArgumentPointerPack(rewriter, loc, resultDescriptors);

    SmallVector<Value> newOperands = {adaptor.getExecutionContext(),
                                      adaptor.getStream(),
                                      numInputs,
                                      inputsPtr,
                                      numOutputs,
                                      outputsPtr};

    builderUtils.enqueueAllocCallBuilder.create(loc, rewriter, newOperands);

    // The `enqueueAllocOp` returned ranked memref values. We must load the
    // result descriptors from the pointer in the unranked LLVM descriptor. The
    // call to `mtrt_tensorrt_enqueue_alloc` has a side effect -- it writes to
    // these descriptors to fill in their shape and stride information.
    SmallVector<Value> results;
    results.reserve(resultDescriptors.size());
    for (auto [v, structType] :
         llvm::zip_equal(resultDescriptors, resultStructTypes)) {
      UnrankedMemRefDescriptor desc(v);
      Value rankedDesc = rewriter.create<LLVM::LoadOp>(
          loc, structType, desc.memRefDescPtr(rewriter, loc));
      results.push_back(rankedDesc);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

/// Populate type conversions for TensorRTRuntime dialect types to LLVM types.
void mlir::populateTensorRTRuntimeToLLVMTypeConversions(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    if (isa<trtrt::EngineType, trtrt::ExecutionContextType>(t))
      return mlir::LLVM::LLVMPointerType::get(t.getContext());
    return {};
  });
}

/// Populate op conversion patterns for TensorRTRuntime dialect ops to LLVM ops.
void mlir::populateTensorRTRuntimeToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ConvertEnqueueAllocToCall, ConvertEnqueueToCall>(typeConverter);
}

namespace {
class TensorRTRuntimeToLLVMPass
    : public mlir::impl::ConvertTensorRTRuntimeToLLVMPassBase<
          TensorRTRuntimeToLLVMPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp rootOp = getOperation();
    TensorRTRuntimeToLLVMOneShotConverter converter(rootOp, artifactsDirectory);
    if (failed(converter.convert())) {
      emitError(rootOp.getLoc())
          << "failed to convert engine symbols in " << getArgument();
      return signalPassFailure();
    }

    // Use dialect conversion to convert remaining ops.
    LLVMTypeConverter typeConverter(&getContext());
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<trtrt::TensorRTRuntimeDialect>();
    RewritePatternSet patterns(&getContext());
    populateCUDAToLLVMTypeConversions(typeConverter);
    populatePlanToLLVMTypeConversions(typeConverter);
    populateTensorRTRuntimeToLLVMTypeConversions(typeConverter);
    populateTensorRTRuntimeToLLVMConversionPatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(rootOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};

/// Implement the interface to convert Func to LLVM.
struct TensorRTRuntimeToLLVMDialectInterface
    : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<plan::PlanDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    target.addIllegalDialect<trtrt::TensorRTRuntimeDialect>();
    populateTensorRTRuntimeToLLVMTypeConversions(typeConverter);
    populateTensorRTRuntimeToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertTensorRTRuntimeToLLVMPatternInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, trtrt::TensorRTRuntimeDialect *dialect) {
        dialect->addInterfaces<TensorRTRuntimeToLLVMDialectInterface>();
      });
}
