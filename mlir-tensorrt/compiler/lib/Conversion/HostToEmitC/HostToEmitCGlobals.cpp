//===- HostToEmitCGlobals.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Global-symbol conversion utilities for `convert-host-to-emitc`.
///
/// This file is responsible for emitting *module-scope* C++ state and helpers:
///   - `emitc.global` declarations for engines/modules/constants/streams
///   - initialization/destruction functions that populate those globals
///   - rewriting symbol uses (e.g. `trtrt.get_function`) to load from globals
///
/// In other words, this file creates the "globals + init/destroy" scaffolding
/// that the per-op conversion patterns assume exists.
//===----------------------------------------------------------------------===//

#include "HostToEmitCDetail.h"
#include "HostToEmitCDetailCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Support/ArtifactManager.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::host_to_emitc;

namespace {
/// A utility for converting global symbols and all their users to EmitC. To
/// use SymbolTable and be more efficient, this conversion step happens in a
/// single linear walk prior to running pattern-based dialect conversion
/// patterns.
struct HostToEmitCGlobalsConverter {
  HostToEmitCGlobalsConverter(ModuleOp module, bool emitAggregateInitDestroy);
  ModuleOp module;
  bool emitAggregateInitDestroy{true};
  MLIRContext *ctx{module.getContext()};
  SymbolTableCollection symbolTables;
  SymbolTable &symbolTable{symbolTables.getSymbolTable(module)};
  SymbolUserMap userMap{symbolTables, module};
  IRRewriter rewriter{module->getContext()};

  EmitCCallBuilders builders{ctx};

  emitc::GlobalOp streamGlobal{};

  /// Names of generated init/destroy helpers with signature:
  ///   - init:   `i32 ()`
  ///   - destroy:`void ()`
  /// These are safe to call from an aggregated init/destroy entrypoint.
  SmallVector<std::string> noArgInitFuncs;
  SmallVector<std::string> noArgDestroyFuncs;

  LogicalResult convert(trtrt::GetFunctionOp op,
                        emitc::GlobalOp executionContextGlobal);
  LogicalResult convert(trtrt::CompiledFuncOp op);
  LogicalResult convert(cuda::CompiledModuleOp op);
  LogicalResult
  convert(cuda::GetFunctionOp op, emitc::GlobalOp cuModuleGlobal,
          llvm::SmallDenseMap<StringAttr, emitc::GlobalOp> &cuFuncCache,
          emitc::FuncOp ctorFunc, Value cuModule);
  LogicalResult convert(cuda::GetGlobalStreamOp op);
  LogicalResult convert(memref::GlobalOp op);
  LogicalResult convert(memref::GetGlobalOp op, emitc::GlobalOp globalOp);
  LogicalResult convert();

private:
  Type i32Type{IntegerType::get(ctx, 32)};
  Type i64Type{IntegerType::get(ctx, 64)};

  Value getI32Val(OpBuilder &rewriter, Location loc, int32_t val) const {
    return rewriter.create<emitc::ConstantOp>(loc, i32Type,
                                              IntegerAttr::get(i32Type, val));
  }
};
} // namespace

HostToEmitCGlobalsConverter::HostToEmitCGlobalsConverter(
    ModuleOp module, bool emitAggregateInitDestroy)
    : module(module), emitAggregateInitDestroy(emitAggregateInitDestroy) {}

LogicalResult
HostToEmitCGlobalsConverter::convert(trtrt::GetFunctionOp op,
                                     emitc::GlobalOp executionContextGlobal) {
  // Intended C++:
  //   // load the global `nvinfer1::IExecutionContext*` and use it as the value
  //   nvinfer1::IExecutionContext *ctx = <global>;
  Location loc = op.getLoc();
  Value ptr = rewriter.create<emitc::GetGlobalOp>(
      loc, emitc::LValueType::get(executionContextGlobal.getType()),
      executionContextGlobal.getSymName());
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(),
      Value(rewriter.create<emitc::LoadOp>(
          loc, executionContextGlobal.getType(), ptr)));
  return success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(trtrt::CompiledFuncOp op) {
  auto engineData = dyn_cast<ElementsAttr>(op.getValue());
  if (!engineData || !engineData.getElementType().isInteger(8))
    return emitError(op.getLoc()) << "unhandled engine data attribute";
  MLIRContext *ctx = op->getContext();

  std::string relPath = mtrt::compiler::createCanonicalArtifactRelativePath(
      op, mtrt::compiler::ArtifactKind::TRTEngine);
  rewriter.setInsertionPoint(op);
  rewriter.create<executor::FileArtifactOp>(op.getLoc(), relPath, engineData);

  rewriter.setInsertionPoint(op);
  Type cuEngineType = emitc::OpaqueType::get(ctx, "nvinfer1::ICudaEngine");
  Type cuEnginePtrType = emitc::PointerType::get(cuEngineType);
  Type trtExecCtxType =
      emitc::OpaqueType::get(ctx, "nvinfer1::IExecutionContext");
  Type trtExecCtxPtrType = emitc::PointerType::get(trtExecCtxType);
  emitc::GlobalOp trtExecCtxGlobal = rewriter.create<emitc::GlobalOp>(
      op.getLoc(), op.getSymName(), trtExecCtxPtrType, Attribute{}, false, true,
      false);
  emitc::GlobalOp cuEngineGlobal = rewriter.create<emitc::GlobalOp>(
      op.getLoc(),
      (*module.getName() + "_" + op.getSymName() + "_trt_cuda_engine").str(),
      cuEnginePtrType, Attribute{}, false, true, false);

  // Intended C++ (schematic):
  //
  //   static nvinfer1::ICudaEngine *<engine_global> = nullptr;
  //   static nvinfer1::IExecutionContext *<ctx_global> = nullptr;
  //
  //   int32_t <op>_initialize(nvinfer1::IRuntime* runtime) {
  //     nvinfer1::ICudaEngine *tmp = nullptr;
  //     int32_t st0 = mtrt::tensorrt_engine_create_from_file(runtime, "path",
  //     &tmp); if (st0) return st0; <engine_global> = tmp;
  //     nvinfer1::IExecutionContext *tmpCtx = nullptr;
  //     int32_t st1 = mtrt::tensorrt_execution_context_create(tmp, &tmpCtx);
  //     if (st1) return st1;
  //     <ctx_global> = tmpCtx;
  //     return 0;
  //   }
  //
  //   void <op>_destroy() {
  //     mtrt::tensorrt_execution_context_destroy(<ctx_global>);
  //     mtrt::tensorrt_engine_destroy(<engine_global>);
  //   }
  emitc::FuncOp initFunc = insertEmitCFunction(
      rewriter, op.getLoc(), module,
      (*module.getName() + "_" + op.getName() + "_initialize").str(), i32Type,
      {builders.trtRuntimePtrType},
      [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
        Value cuEngineL = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(cuEnginePtrType),
            cuEngineGlobal.getName());
        Value filenameLiteral = builders.createStrLiteral(b, loc, relPath);
        Value cuEngineTmp = b.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(cuEnginePtrType),
            emitc::OpaqueAttr::get(ctx, "nullptr"));
        Value cuEngineTmpAddr = getAddrOfLValue(b, loc, cuEngineTmp);
        Value st0 = builders.createTensorRTEngine.create(
            b, loc, {args.front(), filenameLiteral, cuEngineTmpAddr});
        emitStatusCheckOrAbort(b, loc, st0);
        Value cuEnginePtrVal =
            b.create<emitc::LoadOp>(loc, cuEnginePtrType, cuEngineTmp);
        b.create<emitc::AssignOp>(loc, cuEngineL, cuEnginePtrVal);

        Value trtExecCtxL = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(trtExecCtxPtrType),
            trtExecCtxGlobal.getName());
        Value trtExecCtxTmp = b.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(trtExecCtxPtrType),
            emitc::OpaqueAttr::get(ctx, "nullptr"));
        Value trtExecCtxTmpAddr = getAddrOfLValue(b, loc, trtExecCtxTmp);
        Value st1 = builders.createExecutionContext.create(
            b, loc, {cuEnginePtrVal, trtExecCtxTmpAddr});
        emitStatusCheckOrAbort(b, loc, st1);
        Value trtExecCtxVal =
            b.create<emitc::LoadOp>(loc, trtExecCtxPtrType, trtExecCtxTmp);
        b.create<emitc::AssignOp>(loc, trtExecCtxL, trtExecCtxVal);
        return getI32Val(b, loc, 0);
      });

  emitc::FuncOp destroyFunc = insertEmitCFunction(
      rewriter, op.getLoc(), module,
      (*module.getName() + "_" + op.getName() + "_destroy").str(), {}, {},
      [&](OpBuilder &b, Location loc, ValueRange) -> Value {
        Value execCtxLVal = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(trtExecCtxPtrType),
            trtExecCtxGlobal.getName());
        Value execCtx =
            b.create<emitc::LoadOp>(loc, trtExecCtxPtrType, execCtxLVal);
        builders.trtExecutionContextDestroy.create(b, loc, execCtx);

        Value cuEngineLVal = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(cuEnginePtrType),
            cuEngineGlobal.getName());
        Value cuEngine =
            b.create<emitc::LoadOp>(loc, cuEnginePtrType, cuEngineLVal);
        builders.trtEngineDestroy.create(b, loc, cuEngine);
        return {};
      });

  // The initialization function requires arguments, so it cannot be added to
  // the noArgInitFuncs list.
  (void)initFunc;
  noArgDestroyFuncs.push_back(destroyFunc.getName().str());

  for (Operation *user : userMap.getUsers(op)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto getFuncOp = dyn_cast<trtrt::GetFunctionOp>(user)) {
      if (failed(convert(getFuncOp, trtExecCtxGlobal)))
        return failure();
      continue;
    }
    return emitError(user->getLoc())
           << "unexpected dangling reference to TensorRT engine symbol";
  }

  rewriter.eraseOp(op);
  return success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(cuda::GetGlobalStreamOp op) {
  // Intended C++: return/load a global `CUstream` (we model it as an EmitC
  // global and later wrap it into a Program class field).
  if (op.getIndex() != 0)
    return failure();
  Location loc = op.getLoc();
  Value ptr = rewriter.create<emitc::GetGlobalOp>(
      loc, emitc::LValueType::get(streamGlobal.getType()),
      streamGlobal.getSymName());
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(),
      Value(rewriter.create<emitc::LoadOp>(loc, streamGlobal.getType(), ptr)));
  return llvm::success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(memref::GetGlobalOp op,
                                                   emitc::GlobalOp globalOp) {
  // Intended C++: produce a `mtrt::RankedMemRef<Rank>` that points at the
  // global storage and uses a row-major stride layout.
  Location loc = op.getLoc();
  Value ptr{};
  if (isa<emitc::PointerType>(globalOp.getType())) {
    Value ptrLVal = rewriter.create<emitc::GetGlobalOp>(
        loc, emitc::LValueType::get(globalOp.getType()), globalOp.getSymName());
    ptr = rewriter.create<emitc::LoadOp>(loc, globalOp.getType(), ptrLVal);
  } else if (auto arrayType = dyn_cast<emitc::ArrayType>(globalOp.getType())) {
    ptr = rewriter.create<emitc::GetGlobalOp>(loc, arrayType,
                                              globalOp.getSymName());
  } else {
    return failure();
  }
  Value offset = getI32Val(rewriter, loc, 0);
  assert(op.getType().hasStaticShape() &&
         "expected memref.global to have static shape");
  SmallVector<int64_t> strides =
      mlir::computeSuffixProduct(op.getType().getShape());
  Value memref = makeMemRefDescriptor(rewriter, loc, ptr, ptr, offset,
                                      op.getType().getShape(), strides);
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                          memref);
  return llvm::success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(memref::GlobalOp op) {
  if (!op.getInitialValue())
    return emitError(op.getLoc(), "uninitialized global memref unsupported");
  std::optional<plan::MemorySpace> memSpace = getMemorySpace(op.getType());
  if (!memSpace)
    return emitError(op.getLoc(), "memref.global address space not specified");
  Type ptrType = builders.voidPtrType;
  const bool useFileExport = op.getType().getNumElements() > 128 ||
                             memSpace != plan::MemorySpace::host;
  std::string relPath = mtrt::compiler::createCanonicalArtifactRelativePath(
      op, mtrt::compiler::ArtifactKind::ConstantBlob);

  auto globalTypeAndInitialValue =
      [&]() -> FailureOr<std::pair<Type, Attribute>> {
    SmallVector<int64_t> arrayTypeShape(op.getType().getShape());

    const bool isZeroRank = arrayTypeShape.empty();
    if (isZeroRank)
      arrayTypeShape.push_back(1);
    auto arrayType =
        emitc::ArrayType::get(arrayTypeShape, op.getType().getElementType());

    if (Attribute initialValue = op.getInitialValueAttr()) {
      if (useFileExport) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(op);
        rewriter.create<executor::FileArtifactOp>(
            op.getLoc(), relPath, cast<ElementsAttr>(initialValue));
        return std::make_pair(ptrType, Attribute{});
      }

      if (isZeroRank) {
        if (auto elementsAttr = dyn_cast<DenseElementsAttr>(initialValue)) {
          initialValue = elementsAttr.reshape(
              elementsAttr.getType().clone(arrayTypeShape));
        } else {
          return emitError(op.getLoc(),
                           "failed to reshape global initial value");
        }
      }
      return std::make_pair(Type(arrayType), initialValue);
    }

    return std::make_pair(Type(arrayType), Attribute{});
  }();

  if (failed(globalTypeAndInitialValue))
    return failure();

  emitc::GlobalOp globalOp = rewriter.create<emitc::GlobalOp>(
      op.getLoc(), op.getSymName(), std::get<Type>(*globalTypeAndInitialValue),
      std::get<Attribute>(*globalTypeAndInitialValue), false, op.isPrivate(),
      false);
  if (useFileExport) {
    // Intended C++ (schematic):
    //
    //   static void *<global> = nullptr;
    //   int32_t <name>_initialize() {
    //     void *tmp = nullptr;
    //     int32_t st = mtrt::constant_load_from_file("path", align, space,
    //     &tmp); if (st) return st; <global> = tmp; return 0;
    //   }
    //   void <name>_destroy() {
    //     mtrt::constant_destroy(<global>, space);
    //   }
    emitc::FuncOp initFunc = insertEmitCFunction(
        rewriter, op.getLoc(), module,
        (*module.getName() + "_" + op.getName() + "_initialize").str(), i32Type,
        {}, [&](OpBuilder &b, Location loc, ValueRange) -> Value {
          Type globalLoadType = emitc::LValueType::get(ptrType);
          Value bufferPtrLVal = b.create<emitc::GetGlobalOp>(
              loc, globalLoadType, globalOp.getSymName());
          Value alignment =
              getI32Val(b, loc, op.getAlignment() ? *op.getAlignment() : 16);
          Value filenameLiteral = builders.createStrLiteral(b, loc, relPath);
          Value memorySpaceVal =
              getI32Val(b, loc, static_cast<int32_t>(*memSpace));
          Value tmp = b.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(ptrType),
              emitc::OpaqueAttr::get(ctx, "nullptr"));
          Value tmpAddr = getAddrOfLValue(b, loc, tmp);
          Value st = builders.constantLoadFromFile.create(
              b, loc, {filenameLiteral, alignment, memorySpaceVal, tmpAddr});
          emitStatusCheckOrAbort(b, loc, st);
          Value buffer = b.create<emitc::LoadOp>(loc, ptrType, tmp);
          b.create<emitc::AssignOp>(loc, bufferPtrLVal, buffer);
          return getI32Val(b, loc, 0);
        });
    emitc::FuncOp destroyFunc = insertEmitCFunction(
        rewriter, op.getLoc(), module,
        (*module.getName() + "_" + op.getName() + "_destroy").str(), {}, {},
        [&](OpBuilder &b, Location loc, ValueRange) -> Value {
          Value bufferPtrLVal = b.create<emitc::GetGlobalOp>(
              loc, emitc::LValueType::get(ptrType), globalOp.getSymName());
          Value bufferPtr =
              b.create<emitc::LoadOp>(loc, ptrType, bufferPtrLVal);
          Value memorySpaceVal =
              getI32Val(b, loc, static_cast<int32_t>(*memSpace));
          builders.destroyConstant.create(b, loc, {bufferPtr, memorySpaceVal});
          return {};
        });

    noArgInitFuncs.push_back(initFunc.getName().str());
    noArgDestroyFuncs.push_back(destroyFunc.getName().str());
  }

  for (Operation *user : userMap.getUsers(op)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(user)) {
      if (failed(convert(getGlobalOp, globalOp)))
        return failure();
      continue;
    }
    return emitError(user->getLoc())
           << "unexpected dangling reference to memref.global symbol";
  }
  rewriter.eraseOp(op);

  return llvm::success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(
    cuda::GetFunctionOp op, emitc::GlobalOp cuModuleGlobal,
    llvm::SmallDenseMap<StringAttr, emitc::GlobalOp> &cuFuncCache,
    emitc::FuncOp ctorFunc, Value cuModule) {
  Location loc = op.getLoc();

  emitc::GlobalOp funcGlobal = cuFuncCache.lookup(op.getKernelNameAttr());

  auto replace = [&](emitc::GlobalOp funcGlobal) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    Value cuFuncLVal = rewriter.create<emitc::GetGlobalOp>(
        loc, emitc::LValueType::get(funcGlobal.getType()),
        funcGlobal.getSymName());
    Value cuFuncVal =
        rewriter.create<emitc::LoadOp>(loc, funcGlobal.getType(), cuFuncLVal);
    rewriter.replaceOp(op, cuFuncVal);
    return success();
  };

  if (funcGlobal)
    return replace(funcGlobal);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(cuModuleGlobal);
  funcGlobal = rewriter.create<emitc::GlobalOp>(
      op.getLoc(),
      (cuModuleGlobal.getSymName() + "_" + op.getKernelName()).str(),
      builders.cuFuncType, Attribute{}, false, true, false);

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(ctorFunc.getBody().front().getTerminator());
    Type cuFuncLValType = emitc::LValueType::get(builders.cuFuncType);
    Value cuFuncLVal = rewriter.create<emitc::GetGlobalOp>(
        loc, cuFuncLValType, funcGlobal.getSymName());
    Value strLiteral =
        builders.createStrLiteral(rewriter, loc, op.getKernelName());
    Value cuFuncTmp = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(builders.cuFuncType),
        emitc::OpaqueAttr::get(ctx, "nullptr"));
    Value cuFuncTmpAddr = getAddrOfLValue(rewriter, loc, cuFuncTmp);
    Value st = builders.cudaModuleGetFunc.create(
        rewriter, loc, {cuModule, strLiteral, cuFuncTmpAddr});
    emitStatusCheckOrAbort(rewriter, loc, st);
    Value cuFunc =
        rewriter.create<emitc::LoadOp>(loc, builders.cuFuncType, cuFuncTmp);
    rewriter.create<emitc::AssignOp>(loc, cuFuncLVal, cuFunc);
  }

  cuFuncCache[op.getKernelNameAttr()] = funcGlobal;
  return replace(funcGlobal);
}

LogicalResult HostToEmitCGlobalsConverter::convert(cuda::CompiledModuleOp op) {
  std::string relPath = mtrt::compiler::createCanonicalArtifactRelativePath(
      op, mtrt::compiler::ArtifactKind::PTXModule);
  if (op.hasFileReference()) {
    FailureOr<ElementsAttr> resourceAttr =
        mtrt::compiler::createElementsAttrFromFile(
            op.getLoc(), (op.getName() + "_data").str(), op.getFilePath());
    if (failed(resourceAttr))
      return failure();
    rewriter.setInsertionPointAfter(op);
    rewriter.create<executor::FileArtifactOp>(op.getLoc(), relPath,
                                              *resourceAttr);
  } else if (auto cuModuleData =
                 dyn_cast_or_null<ElementsAttr>(op.getValueAttr())) {
    if (!cuModuleData.getElementType().isInteger(8))
      return emitError(op.getLoc())
             << "unhandled CUDA compiled module data attribute";
    rewriter.setInsertionPointAfter(op);
    rewriter.create<executor::FileArtifactOp>(op.getLoc(), relPath,
                                              cuModuleData);
  } else {
    return emitError(op.getLoc())
           << "cuda.compiled_module must specify either 'value' or 'file'";
  }

  MLIRContext *ctx = op->getContext();
  rewriter.setInsertionPoint(op);

  Type cuModuleType = emitc::OpaqueType::get(ctx, "CUmodule");
  emitc::GlobalOp cuModuleGlobal = rewriter.create<emitc::GlobalOp>(
      op.getLoc(),
      (*module.getName() + "_" + op.getSymName() + "_cumodule").str(),
      cuModuleType, Attribute{}, false, true, false);

  Value cuModuleVal{};
  emitc::FuncOp ctorFunc = insertEmitCFunction(
      rewriter, op.getLoc(), module,
      (*module.getName() + "_" + op.getName() + "_initialize").str(), i32Type,
      {}, [&](OpBuilder &b, Location loc, ValueRange) -> Value {
        // Intended C++: load PTX module from file:
        //   CUmodule tmp = nullptr;
        //   int32_t st = mtrt::cuda_module_create_from_ptx_file("path", &tmp);
        //   if (st) return st;
        //   <global_cumodule> = tmp;
        Value cuModule = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(cuModuleType),
            cuModuleGlobal.getSymName());
        Value filenameLiteral = builders.createStrLiteral(b, loc, relPath);
        Value tmp = b.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(cuModuleType),
            emitc::OpaqueAttr::get(ctx, "nullptr"));
        Value tmpAddr = getAddrOfLValue(b, loc, tmp);
        Value st = builders.cudaModuleCreateFromPtxFile.create(
            b, loc, {filenameLiteral, tmpAddr});
        emitStatusCheckOrAbort(b, loc, st);
        cuModuleVal = b.create<emitc::LoadOp>(loc, cuModuleType, tmp);
        b.create<emitc::AssignOp>(loc, cuModule, cuModuleVal);
        return getI32Val(b, loc, 0);
      });

  emitc::FuncOp dtorFunc = insertEmitCFunction(
      rewriter, op.getLoc(), module,
      (*module.getName() + "_" + op.getName() + "_destroy").str(), {}, {},
      [&](OpBuilder &b, Location loc, ValueRange) -> Value {
        // Intended C++:
        //   int32_t st = mtrt::cuda_module_destroy(<global_cumodule>);
        //   if (st) ...;
        Value cuModuleRef = b.create<emitc::GetGlobalOp>(
            loc, emitc::LValueType::get(cuModuleType),
            cuModuleGlobal.getSymName());
        Value cuModule =
            b.create<emitc::LoadOp>(loc, cuModuleType, cuModuleRef);
        Value st = builders.cudaModuleDestroy.create(b, loc, {cuModule});
        emitStatusCheckOrAbort(b, loc, st);
        return {};
      });

  noArgInitFuncs.push_back(ctorFunc.getName().str());
  noArgDestroyFuncs.push_back(dtorFunc.getName().str());

  llvm::SmallDenseMap<StringAttr, emitc::GlobalOp> cuFuncCache;
  for (Operation *user : userMap.getUsers(op)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto getFuncOp = dyn_cast<cuda::GetFunctionOp>(user)) {
      if (failed(convert(getFuncOp, cuModuleGlobal, cuFuncCache, ctorFunc,
                         cuModuleVal)))
        return failure();
      continue;
    }
    return emitError(user->getLoc())
           << "unexpected dangling reference to TensorRT engine symbol";
  }
  rewriter.eraseOp(op);
  return success();
}

LogicalResult HostToEmitCGlobalsConverter::convert() {
  SmallVector<Operation *> cudaStreamOps;
  module.walk([&](Operation *op) {
    if (isa<cuda::GetGlobalStreamOp>(op))
      cudaStreamOps.push_back(op);
  });

  if (!cudaStreamOps.empty()) {
    // Intended C++: `static CUstream <module>_cuda_stream;`
    rewriter.setInsertionPointToStart(module.getBody());
    std::string name =
        (module.getSymName() ? *module.getSymName() : "unnamed_module").str();
    streamGlobal = rewriter.create<emitc::GlobalOp>(
        module.getLoc(), name + "_cuda_stream", builders.cuStreamType,
        Attribute{}, false, false, false);
  }

  rewriter.setInsertionPointToEnd(module.getBody());

  for (Operation &op : llvm::make_early_inc_range(module.getOps())) {
    rewriter.setInsertionPoint(&op);
    if (auto compiledModuleOp = dyn_cast<trtrt::CompiledFuncOp>(op)) {
      if (userMap.getUsers(compiledModuleOp).empty()) {
        symbolTable.remove(compiledModuleOp);
        rewriter.eraseOp(compiledModuleOp);
        continue;
      }
      if (failed(convert(compiledModuleOp)))
        return failure();
      continue;
    }
    if (auto compiledModuleOp = dyn_cast<cuda::CompiledModuleOp>(op)) {
      if (userMap.getUsers(compiledModuleOp).empty()) {
        symbolTable.remove(compiledModuleOp);
        rewriter.eraseOp(compiledModuleOp);
        continue;
      }
      if (failed(convert(compiledModuleOp)))
        return failure();
      continue;
    }
    if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
      if (userMap.getUsers(globalOp).empty()) {
        symbolTable.remove(globalOp);
        rewriter.eraseOp(globalOp);
        continue;
      }
      if (failed(convert(globalOp)))
        return failure();
      continue;
    }
  }

  if (!cudaStreamOps.empty()) {
    for (Operation *op : cudaStreamOps) {
      auto cudaOp = dyn_cast<cuda::GetGlobalStreamOp>(op);
      if (!cudaOp)
        continue;
      rewriter.setInsertionPoint(cudaOp);
      if (failed(convert(cudaOp)))
        return failure();
    }
  }

  if (emitAggregateInitDestroy &&
      (!noArgInitFuncs.empty() || !noArgDestroyFuncs.empty())) {
    // Intended C++ convenience API:
    //   int32_t <module>_initialize_all();
    //   void <module>_destroy_all();
    rewriter.setInsertionPointToEnd(module.getBody());

    std::string modName =
        (module.getSymName() ? *module.getSymName() : "unnamed_module").str();
    std::string initAllName = modName + "_initialize_all";
    std::string destroyAllName = modName + "_destroy_all";

    insertEmitCFunction(
        rewriter, module.getLoc(), module, initAllName, i32Type, {},
        [&](OpBuilder &b, Location loc, ValueRange) -> Value {
          for (const std::string &fn : noArgInitFuncs) {
            Value st = createCallOpaque(b, loc, i32Type, fn, ValueRange{})
                           .getResult(0);
            emitStatusCheckOrAbort(b, loc, st);
          }
          return getI32Val(b, loc, 0);
        });

    insertEmitCFunction(rewriter, module.getLoc(), module, destroyAllName, {},
                        {},
                        [&](OpBuilder &b, Location loc, ValueRange) -> Value {
                          for (auto it = noArgDestroyFuncs.rbegin();
                               it != noArgDestroyFuncs.rend(); ++it) {
                            createCallOpaque(b, loc, {}, *it, ValueRange{});
                          }
                          return {};
                        });
  }

  return success();
}

namespace mlir::host_to_emitc {
LogicalResult convertHostToEmitCGlobals(ModuleOp module,
                                        bool emitAggregateInitDestroy) {
  HostToEmitCGlobalsConverter converter(module, emitAggregateInitDestroy);
  return converter.convert();
}
} // namespace mlir::host_to_emitc
