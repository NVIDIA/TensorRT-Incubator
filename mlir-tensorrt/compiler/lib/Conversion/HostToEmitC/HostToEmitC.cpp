//===- HostToEmitC.cpp ----------------------------------------------------===//
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
/// Implementation of the `convert-host-to-emitc` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/HostToEmitC/HostToEmitC.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Support/ArtifactManager.h"
#include "mlir-tensorrt/Conversion/LLVMCommon/LLVMCommon.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/MathToEmitC/MathToEmitCPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHOSTTOEMITCPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static emitc::OpaqueType getMemRefDescriptorType(MLIRContext *ctx,
                                                 int64_t rank) {
  return emitc::OpaqueType::get(
      ctx, llvm::formatv("mtrt::RankedMemRef<{0}>", rank).str());
}

static emitc::OpaqueType getPointerShapeDescriptorType(MLIRContext *ctx,
                                                       int64_t rank) {
  return emitc::OpaqueType::get(
      ctx, llvm::formatv("mtrt::PtrAndShape<{0}>", rank).str());
}

static emitc::CallOpaqueOp createCallOpaque(OpBuilder &rewriter, Location loc,
                                            Type result, StringRef name,
                                            ValueRange args);

namespace {
struct EmitCCallBuilder {
  StringRef name;
  Type resultType;
  SmallVector<Type> argTypes;

  Value create(OpBuilder &b, Location loc, ValueRange args) const {
    auto callOp = createCallOpaque(b, loc, resultType, name, args);
    return callOp->getNumResults() > 0 ? callOp->getResult(0) : Value{};
  }
};

struct EmitCCallBuilders {
  MLIRContext *ctx;
  Type voidPtrType{
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))};
  Type voidPtrPtrType{emitc::PointerType::get(voidPtrType)};

  Type cuEngineType = emitc::OpaqueType::get(ctx, "nvinfer1::ICudaEngine");
  Type trtRuntimeType = emitc::OpaqueType::get(ctx, "nvinfer1::IRuntime");
  Type cuEnginePtrType = emitc::PointerType::get(cuEngineType);
  Type trtRuntimePtrType = emitc::PointerType::get(trtRuntimeType);
  Type trtExecCtxType =
      emitc::OpaqueType::get(ctx, "nvinfer1::IExecutionContext");
  Type trtExecCtxPtrType = emitc::PointerType::get(trtExecCtxType);
  Type strLiteralType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const char"));
  Type i8Type = IntegerType::get(ctx, 8);
  Type i32Type = IntegerType::get(ctx, 32);
  Type i64Type = IntegerType::get(ctx, 64);
  Type cuModuleType = emitc::OpaqueType::get(ctx, "CUmodule");
  Type cuModulePtrType = emitc::PointerType::get(cuModuleType);
  Type cuFuncType = emitc::OpaqueType::get(ctx, "CUfunction");
  Type cuFuncPtrType = emitc::PointerType::get(cuFuncType);
  Type cuStreamType = emitc::OpaqueType::get(ctx, "CUstream");

  //===----------------------------------------------------------------------===//
  // TensorRT Runtime Functions
  //===----------------------------------------------------------------------===//

  EmitCCallBuilder createTensorRTEngine = {
      "mtrt::tensorrt_engine_create_from_file",
      i32Type,
      {trtRuntimePtrType, strLiteralType,
       emitc::PointerType::get(cuEnginePtrType)}};

  EmitCCallBuilder createExecutionContext = {
      "mtrt::tensorrt_execution_context_create",
      i32Type,
      {cuEnginePtrType, emitc::PointerType::get(trtExecCtxPtrType)}};

  EmitCCallBuilder trtEngineDestroy = {
      "mtrt::tensorrt_engine_destroy", {}, {cuEnginePtrType}};
  EmitCCallBuilder trtExecutionContextDestroy = {
      "mtrt::tensorrt_execution_context_destroy", {}, {trtExecCtxPtrType}};

  EmitCCallBuilder trtEnqueue = {"mtrt::tensorrt_enqueue",
                                 i32Type,
                                 {trtExecCtxPtrType, cuStreamType, i32Type,
                                  voidPtrPtrType, i32Type, voidPtrPtrType}};

  //===----------------------------------------------------------------------===//
  // Host Memory Management Runtime Functions
  //===----------------------------------------------------------------------===//

  EmitCCallBuilder hostAlloc = {
      "mtrt::host_aligned_alloc", i32Type, {i64Type, i32Type, voidPtrPtrType}};
  EmitCCallBuilder hostFree = {"mtrt::host_free", {}, {voidPtrType}};

  EmitCCallBuilder constantLoadFromFile = {
      "mtrt::constant_load_from_file",
      i32Type,
      {strLiteralType, /*alignment*/ i32Type, /*memorySpace*/ i32Type,
       voidPtrPtrType}};
  EmitCCallBuilder destroyConstant = {
      "mtrt::constant_destroy",
      {},
      {/*ptr*/ voidPtrType, /*memorySpace*/ i32Type}};

  //===----------------------------------------------------------------------===//
  // CUDA Runtime Functions
  //===----------------------------------------------------------------------===//

  EmitCCallBuilder cudaModuleCreateFromPtxFile = {
      "mtrt::cuda_module_create_from_ptx_file",
      i32Type,
      {/*filename*/ strLiteralType, /*outModule*/ cuModulePtrType}};
  EmitCCallBuilder cudaModuleGetFunc = {"mtrt::cuda_module_get_func",
                                        i32Type,
                                        {/*module*/ cuModuleType,
                                         /*name*/ strLiteralType,
                                         /*outFunc*/ cuFuncPtrType}};
  EmitCCallBuilder cudaModuleDestroy = {
      "mtrt::cuda_module_destroy", i32Type, {cuModuleType}};

  EmitCCallBuilder cudaGetCurrentDevice = {
      "mtrt::cuda_get_current_device",
      i32Type,
      {/*outDevice*/ emitc::PointerType::get(i32Type)}};

  EmitCCallBuilder cudaStreamSync = {
      "mtrt::cuda_stream_sync", i32Type, {cuStreamType}};

  EmitCCallBuilder cudaAlloc = {
      "mtrt::cuda_alloc",
      i32Type,
      {cuStreamType, i64Type, i8Type, i8Type, voidPtrPtrType}};

  EmitCCallBuilder cudaCopy = {
      "mtrt::cuda_copy",
      i32Type,
      {cuStreamType, voidPtrType, voidPtrType, i64Type}};

  EmitCCallBuilder cudaLaunchKernel = {"mtrt::cuda_launch_kernel",
                                       i32Type,
                                       {cuFuncType, i32Type, i32Type, i32Type,
                                        i32Type, i32Type, i32Type, i32Type,
                                        cuStreamType, voidPtrPtrType}};

  EmitCCallBuilder cudaFree = {"mtrt::cuda_free",
                               i32Type,
                               {cuStreamType, voidPtrType,
                                /*isHostPinned*/ i8Type, /*isManaged*/ i8Type}};

  Value createStrLiteral(OpBuilder &b, Location loc, StringRef literal) const {
    return b.create<emitc::ConstantOp>(
        loc, strLiteralType,
        emitc::OpaqueAttr::get(ctx, llvm::formatv("\"{0}\"", literal).str()));
  }
};

struct EmitCMemRefDescriptor {
  EmitCMemRefDescriptor(Value desc) : desc(desc) {}

  Value desc;

  operator Value() const { return desc; }

  Type voidPtrType{emitc::PointerType::get(
      emitc::OpaqueType::get(desc.getContext(), "void"))};

  Value getMemRefAllocatedPtr(OpBuilder &rewriter, Location loc) const {
    return createCallOpaque(rewriter, loc, voidPtrType,
                            "mtrt::memref_descriptor_get_allocated_ptr", {desc})
        .getResult(0);
  }

  Value getMemRefAlignedPtr(OpBuilder &rewriter, Location loc) const {
    return createCallOpaque(rewriter, loc, voidPtrType,
                            "mtrt::memref_descriptor_get_aligned_ptr", {desc})
        .getResult(0);
  }

  Value getMemRefDimSize(OpBuilder &rewriter, Location loc, Value dim) const {
    return createCallOpaque(rewriter, loc, rewriter.getI64Type(),
                            "mtrt::memref_descriptor_get_dim_size", {desc, dim})
        .getResult(0);
  }

  Value getMemRefDimSize(OpBuilder &rewriter, Location loc, int64_t dim) const {
    return createCallOpaque(rewriter, loc, rewriter.getI64Type(),
                            "mtrt::memref_descriptor_get_dim_size",
                            {desc, rewriter.create<emitc::ConstantOp>(
                                       loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(dim))})
        .getResult(0);
  }

  Value getMemRefOffset(OpBuilder &rewriter, Location loc) const {
    return createCallOpaque(rewriter, loc, rewriter.getI64Type(),
                            "mtrt::memref_descriptor_get_offset", {desc})
        .getResult(0);
  }

  Value getMemRefBufferStart(OpBuilder &rewriter, Location loc,
                             const DataLayout &dataLayout,
                             Type elementType) const {
    Value aligned = getMemRefAlignedPtr(rewriter, loc);
    Value alignedI8 = rewriter.create<emitc::CastOp>(
        loc, emitc::PointerType::get(rewriter.getI8Type()), aligned);
    Value offset = getMemRefOffset(rewriter, loc);
    Value byteSize = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(dataLayout.getTypeSize(elementType)));
    Value byteOffset =
        rewriter.create<emitc::MulOp>(loc, offset.getType(), offset, byteSize);
    return rewriter.create<emitc::CastOp>(
        loc, aligned.getType(),
        rewriter.create<emitc::AddOp>(loc, aligned.getType(), alignedI8,
                                      byteOffset));
  }

  Value getShapeVolumeInElements(OpBuilder &b, Location loc,
                                 MemRefType type) const {
    if (type.getRank() == 0)
      return b.create<emitc::ConstantOp>(loc, b.getI64Type(),
                                         b.getI64IntegerAttr(1));

    if (type.hasStaticShape())
      return b.create<emitc::ConstantOp>(
          loc, b.getI64Type(), b.getI64IntegerAttr(type.getNumElements()));

    // Compute dynamic product.
    Value numElements = b.create<emitc::ConstantOp>(loc, b.getI64Type(),
                                                    b.getI64IntegerAttr(1));
    for (int64_t pos = 0; pos < type.getRank(); pos++)
      numElements = b.create<emitc::MulOp>(loc, b.getI64Type(), numElements,
                                           this->getMemRefDimSize(b, loc, pos));
    return numElements;
  }

  Value getSizeInBytes(OpBuilder &b, Location loc, const DataLayout &dataLayout,
                       MemRefType type) const {
    Type elementType = type.getElementType();
    Value numElements = getShapeVolumeInElements(b, loc, type);
    return b.create<emitc::MulOp>(
        loc, b.getI64Type(), numElements,
        b.create<emitc::ConstantOp>(
            loc, b.getI64Type(),
            b.getI64IntegerAttr(dataLayout.getTypeSize(elementType))));
  }
};

} // namespace

static emitc::FuncOp insertEmitCFunction(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name,
    Type resultType, TypeRange args,
    std::function<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(module.getBody());
  auto func = b.create<emitc::FuncOp>(
      loc, name,
      FunctionType::get(b.getContext(), args,
                        resultType ? resultType : TypeRange{}));
  if (bodyBuilder) {
    Block *body = func.addEntryBlock();
    OpBuilder::InsertionGuard inner(b);
    b.setInsertionPointToStart(body);
    Value v = bodyBuilder(b, loc, body->getArguments());
    b.create<emitc::ReturnOp>(loc, v);
  }
  return func;
}

emitc::CallOpaqueOp createCallOpaque(OpBuilder &rewriter, Location loc,
                                     Type result, StringRef name,
                                     ValueRange args) {
  auto indices = llvm::map_to_vector(
      llvm::seq<unsigned>(args.size()),
      [&](unsigned x) -> Attribute { return rewriter.getIndexAttr(x); });
  return rewriter.create<emitc::CallOpaqueOp>(
      loc, result ? result : TypeRange{}, name, args,
      rewriter.getArrayAttr(indices));
}

static Value getAddrOfLValue(OpBuilder &b, Location loc, Value lvalue) {
  auto lv = cast<emitc::LValueType>(lvalue.getType());
  return b.create<emitc::ApplyOp>(
      loc, emitc::PointerType::get(lv.getValueType()), "&", lvalue);
}

static FunctionOpInterface getParentFunction(Operation *op) {
  assert(op && "expected operation");
  auto funcLike = dyn_cast<FunctionOpInterface>(op);
  return funcLike ? funcLike : op->getParentOfType<FunctionOpInterface>();
}

/// Emit a C++-level status check after a runtime call that returned an `i32`
/// status code (0 == success).
///
/// - If we're inside a function that returns `i32`, emit `return status;` on
///   failure so callers can propagate the error code.
/// - Otherwise, call `mtrt::abort_on_error(status)` (this preserves existing
///   non-status function signatures while avoiding silent failures).
static void emitStatusCheckOrAbort(OpBuilder &b, Location loc, Value status) {
  Operation *parentOp = b.getInsertionBlock()->getParentOp();
  assert(parentOp && "expected parent operation");
  // Try to find the parent function.
  if (auto funcLike = getParentFunction(parentOp)) {
    auto type = dyn_cast<FunctionType>(funcLike.getFunctionType());
    if (type && type.getNumResults() == 1 && type.getResult(0).isInteger(32)) {
      b.create<emitc::VerbatimOp>(
          loc, b.getStringAttr("if ({} != 0) {{ return {}; }"),
          ValueRange{status, status});
      return;
    }
  }

  b.create<emitc::VerbatimOp>(loc, b.getStringAttr("mtrt::abort_on_error({});"),
                              status);
}

static void getMemRefDescriptorSizes(
    const mlir::DataLayout &dataLayout, const TypeConverter &typeConverter,
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides, Value &size, bool sizeInBytes) {
  assert(count(memRefType.getShape(), ShapedType::kDynamic) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");
  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  Type indexType = IntegerType::get(memRefType.getContext(), 64);
  for (int64_t size : memRefType.getShape()) {
    sizes.push_back(
        size == ShapedType::kDynamic
            ? dynamicSizes[dynamicIndex++]
            : rewriter.create<emitc::ConstantOp>(
                  loc, indexType, rewriter.getI64IntegerAttr(size)));
  }

  // Strides: iterate sizes in reverse order and multiply.
  int64_t stride = 1;
  Value runningStride = rewriter.create<emitc::ConstantOp>(
      loc, indexType, rewriter.getI64IntegerAttr(1));
  strides.resize(memRefType.getRank());
  for (auto i = memRefType.getRank(); i-- > 0;) {
    strides[i] = runningStride;

    int64_t staticSize = memRefType.getShape()[i];
    bool useSizeAsStride = stride == 1;
    if (staticSize == ShapedType::kDynamic)
      stride = ShapedType::kDynamic;
    if (stride != ShapedType::kDynamic)
      stride *= staticSize;

    if (useSizeAsStride)
      runningStride = sizes[i];
    else if (stride == ShapedType::kDynamic)
      runningStride = rewriter.create<emitc::MulOp>(loc, indexType,
                                                    runningStride, sizes[i]);
    else
      runningStride = rewriter.create<emitc::ConstantOp>(
          loc, indexType, rewriter.getI64IntegerAttr(stride));
  }
  if (sizeInBytes) {
    // Buffer size in bytes.
    Type elementType = typeConverter.convertType(memRefType.getElementType());
    Value elementTypeSize = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(dataLayout.getTypeSize(elementType)));
    size = rewriter.create<emitc::MulOp>(loc, indexType, elementTypeSize,
                                         runningStride);
  } else {
    size = runningStride;
  }
}

static Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  int64_t rank, ValueRange args) {
  auto indices = llvm::map_to_vector(
      llvm::seq<unsigned>(args.size()),
      [&](unsigned x) -> Attribute { return rewriter.getIndexAttr(x); });
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, getMemRefDescriptorType(rewriter.getContext(), rank),
          "mtrt::make_memref_descriptor", args, rewriter.getArrayAttr(indices),
          rewriter.getArrayAttr(rewriter.getI32IntegerAttr(rank)))
      .getResult(0);
}

static Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  Value allocated, Value aligned, Value offset,
                                  ValueRange shape, ValueRange strides) {
  assert(shape.size() == strides.size() && "mismatched shape/stride ranks");
  SmallVector<Value, 8> args = {allocated, aligned, offset};
  llvm::append_range(args, shape);
  llvm::append_range(args, strides);
  return makeMemRefDescriptor(rewriter, loc, shape.size(), args);
}

static Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  Value allocated, Value aligned, Value offset,
                                  ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> strides) {
  assert(shape.size() == strides.size() && "mismatched shape/stride ranks");

  SmallVector<Attribute> indices =
      llvm::map_to_vector(llvm::seq<unsigned>(3), [&](unsigned x) -> Attribute {
        return rewriter.getIndexAttr(x);
      });
  auto makeAttr = [&](int64_t x) -> Attribute {
    return rewriter.getI64IntegerAttr(x);
  };
  llvm::append_range(indices, llvm::map_range(shape, makeAttr));
  llvm::append_range(indices, llvm::map_range(strides, makeAttr));
  SmallVector<Value, 3> args = {allocated, aligned, offset};
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, getMemRefDescriptorType(rewriter.getContext(), shape.size()),
          "mtrt::make_memref_descriptor", args, rewriter.getArrayAttr(indices),
          rewriter.getArrayAttr(rewriter.getI32IntegerAttr(shape.size())))
      .getResult(0);
}

static std::optional<plan::MemorySpace> getMemorySpace(MemRefType type) {
  auto srcMemoryTypeAttr =
      dyn_cast_or_null<plan::MemorySpaceAttr>(type.getMemorySpace());
  if (!srcMemoryTypeAttr)
    return plan::MemorySpace::host;
  return srcMemoryTypeAttr.getValue();
}

static Value getMemRefPtrShape(OpBuilder &rewriter, Location loc,
                               const DataLayout &dataLayout, MemRefType type,
                               Value sourceMemRef) {
  EmitCMemRefDescriptor sourceDesc(sourceMemRef);
  Value start = sourceDesc.getMemRefBufferStart(rewriter, loc, dataLayout,
                                                type.getElementType());
  auto makeAttr = [&](int64_t x) -> Attribute {
    return rewriter.getI64IntegerAttr(x);
  };
  if (type.hasStaticShape()) {
    SmallVector<Attribute> args = {rewriter.getIndexAttr(0)};
    llvm::append_range(args, llvm::map_range(type.getShape(), makeAttr));
    return rewriter
        .create<emitc::CallOpaqueOp>(
            loc,
            getPointerShapeDescriptorType(rewriter.getContext(),
                                          type.getRank()),
            "mtrt::make_ptr_shape_descriptor", start,
            rewriter.getArrayAttr(args),
            rewriter.getArrayAttr(rewriter.getI32IntegerAttr(type.getRank())))
        .getResult(0);
  }
  SmallVector<Value> args = {start};
  for (int64_t i = 0; i < type.getRank(); ++i)
    args.push_back(sourceDesc.getMemRefDimSize(rewriter, loc, i));
  auto indices = llvm::map_to_vector(
      llvm::seq<unsigned>(args.size()),
      [&](unsigned x) -> Attribute { return rewriter.getIndexAttr(x); });
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc,
          getPointerShapeDescriptorType(rewriter.getContext(), type.getRank()),
          "mtrt::make_ptr_shape_descriptor", args,
          rewriter.getArrayAttr(indices),
          rewriter.getArrayAttr(rewriter.getI32IntegerAttr(type.getRank())))
      .getResult(0);
}

static Value getStridedElementPtr(OpBuilder &rewriter, Location loc,
                                  const TypeConverter &typeConverter,
                                  MemRefType type,
                                  EmitCMemRefDescriptor memRefDesc,
                                  ValueRange indices) {
  auto [strides, offset] = type.getStridesAndOffset();
  // Use a canonical representation of the start address so that later
  // optimizations have a longer sequence of instructions to CSE.
  // If we don't do that we would sprinkle the memref.offset in various
  // position of the different address computations.
  Value index = memRefDesc.getMemRefOffset(rewriter, loc);
  auto getI64Val = [&](int64_t x) {
    return rewriter.create<emitc::ConstantOp>(loc, rewriter.getI64Type(),
                                              rewriter.getI64IntegerAttr(x));
  };
  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride = ShapedType::isDynamic(strides[i])
                         ? memRefDesc.getMemRefDimSize(rewriter, loc, i)
                         : getI64Val(strides[i]);
      increment = rewriter.create<emitc::MulOp>(loc, increment.getType(),
                                                increment, stride);
    }
    index = index ? rewriter.create<emitc::AddOp>(loc, index.getType(), index,
                                                  increment)
                  : increment;
  }
  return index;
}

//===----------------------------------------------------------------------===//
// Global Symbol Conversion Utility
//===----------------------------------------------------------------------===//

namespace {
/// A utility for converting global symbols and all their users to EmitC. To
/// use SymbolTable and be more efficient, this conversion step happens in a
/// single linear walk prior to running pattern-based dialect conversion
/// patterns.
struct HostToEmitCGlobalsConverter {
  HostToEmitCGlobalsConverter(ModuleOp module);
  ModuleOp module;
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
  Value getI64Val(OpBuilder &rewriter, Location loc, int32_t val) const {
    return rewriter.create<emitc::ConstantOp>(loc, i64Type,
                                              IntegerAttr::get(i64Type, val));
  }
};
} // namespace

HostToEmitCGlobalsConverter::HostToEmitCGlobalsConverter(ModuleOp module)
    : module(module) {}

LogicalResult
HostToEmitCGlobalsConverter::convert(trtrt::GetFunctionOp op,
                                     emitc::GlobalOp executionContextGlobal) {
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

  // Creat initialization and destroy functions for the TensorRT 'ICudaEngine'
  // and 'IExecutionContext'. We keep them as separate globals here in order
  // to make it easier to potentially use multiple execution contexts in the
  // future.

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

  // initFunc takes a runtime pointer argument, so it can't be called from a
  // no-arg aggregated initializer. The destroy helper is safe to call.
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

  // There are no longer references to the original func, we can remove it.
  rewriter.eraseOp(op);
  return success();
}

LogicalResult HostToEmitCGlobalsConverter::convert(cuda::GetGlobalStreamOp op) {
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

    // We require "0 rank arrays" to be represented as "size-1 arrays" in
    // EmitC.
    const bool isZeroRank = arrayTypeShape.empty();
    if (isZeroRank)
      arrayTypeShape.push_back(1);
    auto arrayType =
        emitc::ArrayType::get(arrayTypeShape, op.getType().getElementType());

    if (Attribute initialValue = op.getInitialValueAttr()) {
      if (useFileExport) {
        OpBuilder::InsertionGuard g(rewriter);
        // export data to file.
        rewriter.setInsertionPoint(op);
        rewriter.create<executor::FileArtifactOp>(
            op.getLoc(), relPath, cast<ElementsAttr>(initialValue));
        return std::make_pair(ptrType, Attribute{});
      }

      // Reshape the initial value if required.
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
              getI32Val(rewriter, loc, static_cast<int32_t>(*memSpace));
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
              getI32Val(rewriter, loc, static_cast<int32_t>(*memSpace));
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
  // Only materialize the CUDA stream global if the module actually uses CUDA
  // stream ops. This prevents non-CUDA EmitC outputs from requiring CUDA
  // types (e.g. `CUstream`) and headers at compile time.
  SmallVector<Operation *> cudaStreamOps;
  module.walk([&](Operation *op) {
    if (isa<cuda::GetGlobalStreamOp>(op))
      cudaStreamOps.push_back(op);
  });

  if (!cudaStreamOps.empty()) {
    rewriter.setInsertionPointToStart(module.getBody());
    std::string name =
        (module.getSymName() ? *module.getSymName() : "unamed_module").str();
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

  // Emit aggregated no-arg init/destroy helpers to simplify drivers.
  // - init returns i32 Status (0 == success)
  // - destroy is best-effort void
  if (!noArgInitFuncs.empty() || !noArgDestroyFuncs.empty()) {
    rewriter.setInsertionPointToEnd(module.getBody());

    std::string modName =
        (module.getSymName() ? *module.getSymName() : "unnamed_module").str();
    std::string initAllName = modName + "_initialize_all";
    std::string destroyAllName = modName + "_destroy_all";

    // int32_t <module>_initialize_all()
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

    // void <module>_destroy_all()
    insertEmitCFunction(
        rewriter, module.getLoc(), module, destroyAllName, {}, {},
        [&](OpBuilder &b, Location loc, ValueRange) -> Value {
          for (auto it = noArgDestroyFuncs.rbegin();
               it != noArgDestroyFuncs.rend(); ++it) {
            (void)createCallOpaque(b, loc, Type{}, *it, ValueRange{});
          }
          return {};
        });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmitCConversionPattern
//===----------------------------------------------------------------------===//

static emitc::PointerType getPointerType(Type elementType) {
  return emitc::PointerType::get(elementType);
}

namespace {
/// EmitCConversionPattern is the base for all EmitC conversion patterns
/// organized by dialect below.
template <typename OpType>
struct EmitCConversionPattern : OpConversionPattern<OpType> {

  EmitCConversionPattern(const TypeConverter &typeConverter,
                         const DataLayout &dataLayout, MLIRContext *ctx,
                         PatternBenefit benefit = PatternBenefit(10))
      : OpConversionPattern<OpType>(typeConverter, ctx, benefit),
        dataLayout(dataLayout) {}

  MLIRContext *ctx{this->getContext()};
  EmitCCallBuilders builders{ctx};
  Type voidPtrType{builders.voidPtrType};
  IntegerType i8Type{IntegerType::get(ctx, 8)};
  IntegerType i32Type{IntegerType::get(ctx, 32)};
  IntegerType i64Type{IntegerType::get(ctx, 64)};
  const mlir::DataLayout &dataLayout;

  emitc::OpaqueType getOpaqueType(StringRef str) const {
    return emitc::OpaqueType::get(ctx, str);
  }
  emitc::LValueType getLValueType(Type t) const {
    return emitc::LValueType::get(t);
  }
  emitc::OpaqueAttr getOpaqueAttr(StringRef str) const {
    return emitc::OpaqueAttr::get(ctx, str);
  }
  emitc::ArrayType getArrayType(ArrayRef<int64_t> shape, Type t) const {
    return emitc::ArrayType::get(shape, t);
  }

  Value createCast(OpBuilder &b, Type to, Value from) const {
    return b.create<emitc::CastOp>(from.getLoc(), to, from);
  }
  Value getI32Val(OpBuilder &rewriter, Location loc, int32_t val) const {
    return rewriter.create<emitc::ConstantOp>(loc, i32Type,
                                              IntegerAttr::get(i32Type, val));
  }
  Value getAddr(OpBuilder &b, Location loc, Value val) const {
    return b.create<emitc::ApplyOp>(
        loc,
        getPointerType(cast<emitc::LValueType>(val.getType()).getValueType()),
        "&", val);
  }
  Value getNullptr(OpBuilder &b, Location loc, Type t) const {
    return b.create<emitc::ConstantOp>(
        loc, t, emitc::OpaqueAttr::get(t.getContext(), "nullptr"));
  }

  emitc::OpaqueType unrankedDescriptorType{
      getOpaqueType("mtrt::UnrankedMemRef")};

  auto callOpaque(OpBuilder &b, Location loc, Type result, StringRef name,
                  ValueRange args) const {
    auto indices = llvm::map_to_vector(
        llvm::seq<unsigned>(args.size()),
        [&](unsigned x) -> Attribute { return b.getIndexAttr(x); });
    return b.create<emitc::CallOpaqueOp>(loc, result ? result : TypeRange{},
                                         name, args, b.getArrayAttr(indices));
  }

  Value getUnrankedDescriptor(OpBuilder &b, Location loc, int64_t rank,
                              Value rankedDesc) const {
    Value rankVal = getI32Val(b, loc, rank);
    return callOpaque(b, loc, unrankedDescriptorType,
                      "mtrt::make_unranked_descriptor", {rankVal, rankedDesc})
        .getResult(0);
  }
};

//===----------------------------------------------------------------------===//
// TensorRTRuntime-to-EmitC Patterns
//===----------------------------------------------------------------------===//

struct TRTEnqueueConverter : EmitCConversionPattern<trtrt::EnqueueOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value inputPtrs = rewriter.create<emitc::VariableOp>(
        loc,
        getArrayType({static_cast<int64_t>(adaptor.getInputs().size())},
                     unrankedDescriptorType),
        getOpaqueAttr(""));
    for (auto [idx, v, originalType] :
         llvm::enumerate(adaptor.getInputs(), TypeRange(op.getInputs()))) {
      Value arrayElement = rewriter.create<emitc::SubscriptOp>(
          loc, getLValueType(unrankedDescriptorType), inputPtrs,
          getI32Val(rewriter, loc, idx));
      getI32Val(rewriter, loc, cast<MemRefType>(originalType).getRank());
      Value rankVal =
          getI32Val(rewriter, loc, cast<MemRefType>(originalType).getRank());
      rewriter.create<emitc::AssignOp>(
          loc, arrayElement,
          callOpaque(
              rewriter, loc, unrankedDescriptorType,
              "mtrt::make_unranked_descriptor",
              {rankVal, getMemRefPtrShape(rewriter, loc, dataLayout,
                                          cast<MemRefType>(originalType), v)})
              .getResult(0));
    }

    Value outputPtrs = rewriter.create<emitc::VariableOp>(
        loc,
        getArrayType({static_cast<int64_t>(adaptor.getOuts().size())},
                     unrankedDescriptorType),
        getOpaqueAttr(""));
    for (auto [idx, v, originalType] :
         llvm::enumerate(adaptor.getOuts(), TypeRange(op.getOuts()))) {
      Value arrayElement = rewriter.create<emitc::SubscriptOp>(
          loc, getLValueType(unrankedDescriptorType), outputPtrs,
          getI32Val(rewriter, loc, idx));
      rewriter.create<emitc::AssignOp>(
          loc, arrayElement,
          getUnrankedDescriptor(
              rewriter, loc, cast<MemRefType>(originalType).getRank(),
              getMemRefPtrShape(rewriter, loc, dataLayout,
                                cast<MemRefType>(originalType), v)));
    }

    Value numInputs = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(adaptor.getInputs().size()));
    Value numOutputs = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(adaptor.getOuts().size()));

    Value st = builders.trtEnqueue.create(rewriter, loc,
                                          {adaptor.getExecutionContext(),
                                           adaptor.getStream(), numInputs,
                                           inputPtrs, numOutputs, outputPtrs});
    emitStatusCheckOrAbort(rewriter, loc, st);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CUDA-to-EmitC Patterns
//===----------------------------------------------------------------------===//

struct CUDALaunchConverter : public EmitCConversionPattern<cuda::LaunchOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // The CUDA driver API expects a `void**` array where each element points
    // to the storage for the corresponding by-value kernel parameter.
    //
    // For EmitC, we materialize a local variable for each argument, take its
    // address, cast to `void*`, and store into the argument pointer array.
    //
    // NOTE: We intentionally do not attempt to \"expand\" memrefs into
    // multiple kernel parameters here. The contract is that upstream lowering
    // has already flattened kernel arguments into a low-level ABI (e.g. base
    // buffer pointers + explicit offset/size/stride scalars as needed).
    const int64_t numArgs = adaptor.getArgs().size();
    auto argvType = getArrayType({numArgs}, voidPtrType);
    Value argv =
        rewriter.create<emitc::VariableOp>(loc, argvType, getOpaqueAttr(""));

    auto getZeroInitAttr = [&](Type t) -> Attribute {
      if (isa<emitc::PointerType>(t))
        return getOpaqueAttr("nullptr");
      if (auto it = dyn_cast<IntegerType>(t))
        return IntegerAttr::get(it, 0);
      if (auto ft = dyn_cast<FloatType>(t))
        return FloatAttr::get(ft, 0.0);
      if (isa<IndexType>(t))
        return rewriter.getIndexAttr(0);
      return {};
    };

    for (auto [idx, value, originalType] :
         llvm::enumerate(adaptor.getArgs(), TypeRange(op.getArgs()))) {
      Value rhs = value;
      Type storageType = value.getType();

      // Memrefs are passed as their buffer-start pointer (aligned + offset).
      if (auto memRefType = dyn_cast<MemRefType>(originalType)) {
        EmitCMemRefDescriptor desc(value);
        rhs = desc.getMemRefBufferStart(rewriter, loc, dataLayout,
                                        memRefType.getElementType());
        storageType = voidPtrType;
      } else if (isa<IndexType>(originalType)) {
        // Normalize `index` arguments to 64-bit integers for a stable kernel
        // ABI.
        rhs = createCast(rewriter, i64Type, value);
        storageType = i64Type;
      } else if (!originalType.isSignlessIntOrIndexOrFloat()) {
        return rewriter.notifyMatchFailure(
            op, "unsupported cuda.launch argument type; expected memref or "
                "signless int/float/index");
      }

      Attribute initAttr = getZeroInitAttr(storageType);
      if (!initAttr)
        return rewriter.notifyMatchFailure(
            op, "failed to form a suitable EmitC variable initializer for "
                "cuda.launch argument");

      Value local = rewriter.create<emitc::VariableOp>(
          loc, getLValueType(storageType), initAttr);
      rewriter.create<emitc::AssignOp>(loc, local, rhs);

      // argv[idx] = (void*)&local;
      Value addr = getAddr(rewriter, loc, local);
      Value addrVoid = createCast(rewriter, voidPtrType, addr);
      Value argvElem = rewriter.create<emitc::SubscriptOp>(
          loc, getLValueType(voidPtrType), argv, getI32Val(rewriter, loc, idx));
      rewriter.create<emitc::AssignOp>(loc, argvElem, addrVoid);
    }

    Value st = builders.cudaLaunchKernel.create(
        rewriter, loc,
        ValueRange{adaptor.getFunc(), adaptor.getGridX(), adaptor.getGridY(),
                   adaptor.getGridZ(), adaptor.getBlockX(), adaptor.getBlockY(),
                   adaptor.getBlockZ(), adaptor.getDynamicSharedMem(),
                   adaptor.getStream(), argv});
    emitStatusCheckOrAbort(rewriter, loc, st);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CUDAGetCurrentDeviceConverter
    : public EmitCConversionPattern<cuda::GetActiveDeviceOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::GetActiveDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value devVar = rewriter.create<emitc::VariableOp>(
        loc, getLValueType(i32Type), rewriter.getI32IntegerAttr(0));
    Value devAddr = getAddr(rewriter, loc, devVar);
    Value st = builders.cudaGetCurrentDevice.create(rewriter, loc, {devAddr});
    emitStatusCheckOrAbort(rewriter, loc, st);
    Value dev = rewriter.create<emitc::LoadOp>(loc, i32Type, devVar);
    rewriter.replaceOp(op, dev);
    return success();
  }
};

struct CUDAStreamSyncConverter
    : public EmitCConversionPattern<cuda::StreamSyncOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::StreamSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value st =
        builders.cudaStreamSync.create(rewriter, loc, {adaptor.getStream()});
    emitStatusCheckOrAbort(rewriter, loc, st);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

/// Returns `true` if a memref with shape `shape` and `strides` represents a
/// contiguous array of memory. This is equivalent to checking whether some
/// subview is contiguous. The idea here is that the shape and laout should
/// have a canonical row-major layout when removing the unit extents. For
/// example, `memref<8x1x4, strided<[4, 32, 1], offset: ?>>` should be
/// contiguous since we can ignore the middle unit extent dimension.
static bool isContiguousImpl(ArrayRef<int64_t> strides,
                             ArrayRef<int64_t> shape) {
  unsigned e = strides.size();
  if (shape.empty() || strides.empty())
    return true;

  auto findNextIndex = [&](unsigned start) -> std::optional<unsigned> {
    for (unsigned i = start; i < e; i++) {
      if (shape[i] != 1)
        return i;
    }
    return {};
  };

  // If no starting index, then this is a scalar shape.
  std::optional<unsigned> index = findNextIndex(0);
  if (!index)
    return true;

  while (*index < e) {
    std::optional<unsigned> next = findNextIndex(*index + 1);
    // If this is the last relevant index, it must be unit stride or unit
    // access.
    if (!next)
      return strides[*index] == 1 || shape[*index] == 1;
    if (ShapedType::isDynamic(strides[*index]) ||
        ShapedType::isDynamic(strides[*next]))
      return false;
    if (strides[*index] != strides[*next] * shape[*next])
      return false;
    index = *next;
  }
  return true;
}

static bool isContiguous(MemRefType t) {
  if (t.getLayout().isIdentity())
    return true;
  if (!t.hasStaticShape())
    return false;
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(t.getStridesAndOffset(strides, offset)))
    return false;
  return isContiguousImpl(strides, t.getShape());
}

static bool isCopyStrided(MemRefType srcMemRefType, MemRefType dstMemRefType) {
  return !isContiguous(srcMemRefType) || !isContiguous(dstMemRefType);
}

namespace {

struct CUDAAllocConverter : public EmitCConversionPattern<cuda::AllocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    MemRefType memrefType = op.getResult().getType();
    if (!memrefType.hasRank())
      return rewriter.notifyMatchFailure(op, "cannot convert unranked memref");
    if (!isContiguous(memrefType))
      return failure();

    std::optional<plan::MemorySpace> space = getMemorySpace(memrefType);
    if (!space ||
        !llvm::is_contained(
            ArrayRef<plan::MemorySpace>{plan::MemorySpace::device,
                                        plan::MemorySpace::host_pinned,
                                        plan::MemorySpace::unified},
            *space))
      return failure();

    Value isManaged = rewriter.create<emitc::ConstantOp>(
        loc, i8Type,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::unified));
    Value isPinned = rewriter.create<emitc::ConstantOp>(
        loc, i8Type,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::host_pinned));

    Value sizeBytes;
    SmallVector<Value> shape;
    SmallVector<Value> strides;
    getMemRefDescriptorSizes(dataLayout, *getTypeConverter(), loc, memrefType,
                             adaptor.getDynamicSizes(), rewriter, shape,
                             strides, sizeBytes, /*sizeInBytes=*/true);

    Value stream = adaptor.getStream()
                       ? adaptor.getStream()
                       : getNullptr(rewriter, loc, builders.cuStreamType);
    Value ptrVar = rewriter.create<emitc::VariableOp>(
        loc, getLValueType(voidPtrType), getOpaqueAttr("nullptr"));
    Value ptrAddr = getAddr(rewriter, loc, ptrVar);
    Value st = builders.cudaAlloc.create(
        rewriter, loc, {stream, sizeBytes, isPinned, isManaged, ptrAddr});
    emitStatusCheckOrAbort(rewriter, loc, st);
    Value ptr = rewriter.create<emitc::LoadOp>(loc, voidPtrType, ptrVar);

    Value replacement = makeMemRefDescriptor(
        rewriter, loc, ptr, ptr, getI32Val(rewriter, loc, 0), shape, strides);

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

struct CudaDeallocConverter : public EmitCConversionPattern<cuda::DeallocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(cuda::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    std::optional<plan::MemorySpace> space =
        getMemorySpace(op.getMemref().getType());
    if (!space ||
        !llvm::is_contained(
            ArrayRef<plan::MemorySpace>{plan::MemorySpace::device,
                                        plan::MemorySpace::host_pinned,
                                        plan::MemorySpace::unified},
            *space))
      return failure();
    Value isPinned = rewriter.create<emitc::ConstantOp>(
        loc, i8Type,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::host_pinned));
    Value isManaged = rewriter.create<emitc::ConstantOp>(
        loc, i8Type,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::unified));

    EmitCMemRefDescriptor desc(adaptor.getMemref());
    Value ptr = desc.getMemRefAllocatedPtr(rewriter, loc);
    Value st = builders.cudaFree.create(
        rewriter, loc, {adaptor.getStream(), ptr, isPinned, isManaged});
    emitStatusCheckOrAbort(rewriter, loc, st);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename CpyOpType>
struct CudaCopyConverter : public EmitCConversionPattern<CpyOpType> {
  using EmitCConversionPattern<CpyOpType>::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(CpyOpType op,
                  typename EmitCConversionPattern<CpyOpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType srcType = op.getSource().getType();
    MemRefType dstType = op.getTarget().getType();
    std::optional<plan::MemorySpace> srcSpace = getMemorySpace(srcType);
    std::optional<plan::MemorySpace> dstSpace = getMemorySpace(dstType);
    if (!srcSpace || !dstSpace)
      return failure();

    EmitCMemRefDescriptor src(adaptor.getSource());
    EmitCMemRefDescriptor dest(adaptor.getTarget());
    Location loc = op.getLoc();
    Value srcStart = src.getMemRefBufferStart(rewriter, loc, this->dataLayout,
                                              srcType.getElementType());
    Value destStart = dest.getMemRefBufferStart(rewriter, loc, this->dataLayout,
                                                dstType.getElementType());

    if (!isCopyStrided(srcType, dstType)) {
      Value totalSize =
          src.getSizeInBytes(rewriter, loc, this->dataLayout, srcType);
      Value st = this->builders.cudaCopy.create(
          rewriter, loc, {adaptor.getStream(), srcStart, destStart, totalSize});
      emitStatusCheckOrAbort(rewriter, loc, st);
      rewriter.eraseOp(op);
      return success();
    }

    Value rankVal =
        this->getI32Val(rewriter, loc, cast<MemRefType>(srcType).getRank());
    Value srcUnranked =
        this->callOpaque(rewriter, loc, this->unrankedDescriptorType,
                         "mtrt::make_unranked_descriptor", {rankVal, src})
            .getResult(0);

    Value dstRank =
        this->getI32Val(rewriter, loc, cast<MemRefType>(dstType).getRank());
    Value dstUnranked =
        this->callOpaque(rewriter, loc, this->unrankedDescriptorType,
                         "mtrt::make_unranked_descriptor", {dstRank, dest})
            .getResult(0);

    this->callOpaque(
        rewriter, loc, Type{}, "mtrt::cuda_copy_strided",
        {adaptor.getStream(), srcStart, srcUnranked, destStart, dstUnranked});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MemRef-to-EmitC Converters
//===----------------------------------------------------------------------===//

struct ExecutorPrintConverter
    : public EmitCConversionPattern<executor::PrintOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(executor::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<StringRef> format = op.getFormat();
    Location loc = op.getLoc();
    if (!format)
      return failure();

    SmallVector<Attribute> staticArgs = {emitc::OpaqueAttr::get(
        rewriter.getContext(), llvm::formatv("\"{0}\\n\"", *format).str())};
    llvm::append_range(
        staticArgs,
        llvm::map_range(llvm::seq<unsigned>(0, adaptor.getArguments().size()),
                        [&](unsigned x) { return rewriter.getIndexAttr(x); }));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "printf", adaptor.getOperands(),
        rewriter.getArrayAttr(staticArgs), ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts `memref.extract_strided_metadata` op to MTRT C++ API calls to
/// decompose the descriptor. Adapted from upstream memref-to-llvm pass.
class ExtractStridedMetadataOpLowering
    : public EmitCConversionPattern<memref::ExtractStridedMetadataOp> {
public:
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the descriptor.
    Value sourceMemRef(adaptor.getSource());
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    auto sourceMemRefType = cast<MemRefType>(source.getType());
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    // Base buffer.
    Value baseBuffer =
        callOpaque(rewriter, loc, voidPtrType,
                   "mtrt::memref_descriptor_get_allocated_ptr", {sourceMemRef})
            .getResult(0);
    Value alignedBuffer =
        callOpaque(rewriter, loc, voidPtrType,
                   "mtrt::memref_descriptor_get_aligned_ptr", {sourceMemRef})
            .getResult(0);
    Value offset =
        callOpaque(rewriter, loc, i64Type, "mtrt::memref_descriptor_get_offset",
                   {sourceMemRef})
            .getResult(0);

    Value dstMemRef = makeMemRefDescriptor(
        rewriter, loc, 0,
        {baseBuffer, alignedBuffer, getI32Val(rewriter, loc, 0)});

    results.push_back(dstMemRef);

    // Offset.
    results.push_back(offset);

    // Sizes.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(callOpaque(rewriter, loc, i64Type,
                                   "mtrt::memref_descriptor_get_dim_size",
                                   {sourceMemRef, getI32Val(rewriter, loc, i)})
                            .getResult(0));
    // Strides.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(callOpaque(rewriter, loc, i64Type,
                                   "mtrt::memref_descriptor_get_stride",
                                   {sourceMemRef, getI32Val(rewriter, loc, i)})
                            .getResult(0));

    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};

/// Converts `memref.reinterpret_cast` op to MTRT C++ API calls to
/// compose the descriptor. Adapted from upstream memref-to-llvm pass.
struct MemRefReinterpretCastOpLowering
    : public EmitCConversionPattern<memref::ReinterpretCastOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = castOp.getSource().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  LogicalResult convertSourceMemRefToDescriptor(
      ConversionPatternRewriter &rewriter, Type srcType,
      memref::ReinterpretCastOp castOp,
      memref::ReinterpretCastOp::Adaptor adaptor, Value *descriptor) const {
    MemRefType targetMemRefType =
        cast<MemRefType>(castOp.getResult().getType());

    // Create descriptor.
    Location loc = castOp.getLoc();

    // Set allocated and aligned pointers.
    EmitCMemRefDescriptor srcMemRef(adaptor.getSource());
    SmallVector<Value> args = {srcMemRef.getMemRefAlignedPtr(rewriter, loc),
                               srcMemRef.getMemRefAlignedPtr(rewriter, loc)};

    // Set offset.
    if (castOp.isDynamicOffset(0))
      args.push_back(adaptor.getOffsets()[0]);
    else
      args.push_back(getI32Val(rewriter, loc, 0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        args.push_back(adaptor.getSizes()[dynSizeId++]);
      else
        args.push_back(getI32Val(rewriter, loc, castOp.getStaticSize(i)));
    }
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicStride(i))
        args.push_back(adaptor.getStrides()[dynStrideId++]);
      else
        args.push_back(getI32Val(rewriter, loc, castOp.getStaticStride(i)));
    }
    *descriptor =
        makeMemRefDescriptor(rewriter, loc, targetMemRefType.getRank(), args);
    return success();
  }
};

struct MemRefAllocOpLowering : public EmitCConversionPattern<memref::AllocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isContiguous(op.getType()))
      return failure();
    std::optional<plan::MemorySpace> memoryType = getMemorySpace(op.getType());
    if (!memoryType || *memoryType != plan::MemorySpace::host)
      return rewriter.notifyMatchFailure(op, "unsupported memory space");

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();
    Location loc = op.getLoc();

    SmallVector<Value> sizes, strides;
    Value size;
    getMemRefDescriptorSizes(dataLayout, *getTypeConverter(), loc, op.getType(),
                             op.getDynamicSizes(), rewriter, sizes, strides,
                             size, /*sizeInBytes=*/true);

    int32_t alignemnt = adaptor.getAlignment() ? *adaptor.getAlignment() : 16;
    Value alignVal = getI32Val(rewriter, loc, alignemnt);
    Value allocVar = rewriter.create<emitc::VariableOp>(
        loc, getLValueType(voidPtrType), getOpaqueAttr("nullptr"));
    Value allocAddr = getAddr(rewriter, loc, allocVar);
    Value st =
        builders.hostAlloc.create(rewriter, loc, {size, alignVal, allocAddr});
    emitStatusCheckOrAbort(rewriter, loc, st);
    Value alloc = rewriter.create<emitc::LoadOp>(loc, voidPtrType, allocVar);
    Value desc =
        makeMemRefDescriptor(rewriter, loc, alloc, alloc,
                             getI32Val(rewriter, loc, 0), sizes, strides);
    rewriter.replaceOp(op, desc);
    return success();
  }
};

struct MemrefCastOpLowering : public EmitCConversionPattern<memref::CastOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType sourceType = dyn_cast<MemRefType>(op.getSource().getType());
    MemRefType targetType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!sourceType || !targetType)
      return failure();
    if (sourceType.getRank() != targetType.getRank())
      return failure();
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// Convert `memref.dealloc` to MTRT C++ API calls.
struct MemRefDeallocLowering
    : public EmitCConversionPattern<memref::DeallocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = dyn_cast<MemRefType>(op.getMemref().getType());
    if (!memRefType)
      return failure();
    std::optional<plan::MemorySpace> memoryType = getMemorySpace(memRefType);
    if (!memoryType || *memoryType != plan::MemorySpace::host)
      return rewriter.notifyMatchFailure(op, "unsupported memory space");
    Value ptr = EmitCMemRefDescriptor(adaptor.getMemref())
                    .getMemRefAllocatedPtr(rewriter, op.getLoc());
    builders.hostFree.create(rewriter, op.getLoc(), {ptr});
    rewriter.eraseOp(op);
    return success();
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct MemRefDimOpLowering : public EmitCConversionPattern<memref::DimOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp dimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = dimOp.getSource().getType();
    if (isa<UnrankedMemRefType>(operandType))
      return failure();
    rewriter.replaceOp(dimOp, EmitCMemRefDescriptor(adaptor.getSource())
                                  .getMemRefDimSize(rewriter, dimOp.getLoc(),
                                                    adaptor.getIndex()));
    return success();
  }
};

// Convert memref.load to C++.
struct MemRefLoadOpLowering : public EmitCConversionPattern<memref::LoadOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.getMemRefType();
    Type elementType = typeConverter->convertType(type.getElementType());
    if (!elementType)
      return failure();
    Value offset =
        getStridedElementPtr(rewriter, loc, *getTypeConverter(), type,
                             adaptor.getMemref(), adaptor.getIndices());
    Value alignedPtr = EmitCMemRefDescriptor(adaptor.getMemref())
                           .getMemRefAlignedPtr(rewriter, loc);
    Value typedPtr = rewriter.create<emitc::CastOp>(
        loc, getPointerType(elementType), alignedPtr);
    Value lval = rewriter.create<emitc::SubscriptOp>(
        loc, getLValueType(elementType), typedPtr, offset);
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, elementType, lval);
    return success();
  }
};

// Convert memref.load to C++.
struct MemRefStoreOpLowering : public EmitCConversionPattern<memref::StoreOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.getMemRefType();
    Type elementType = typeConverter->convertType(type.getElementType());
    if (!elementType)
      return failure();
    Value offset =
        getStridedElementPtr(rewriter, loc, *getTypeConverter(), type,
                             adaptor.getMemref(), adaptor.getIndices());
    Value alignedPtr = EmitCMemRefDescriptor(adaptor.getMemref())
                           .getMemRefAlignedPtr(rewriter, loc);
    Value typedPtr = rewriter.create<emitc::CastOp>(
        loc, getPointerType(elementType), alignedPtr);
    Value lval = rewriter.create<emitc::SubscriptOp>(
        loc, getLValueType(elementType), typedPtr, offset);
    rewriter.create<emitc::AssignOp>(loc, lval, adaptor.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Unpack the pointer returned by a memref.extract_aligned_pointer_as_index.
class MemRefExtractAlignedPointerAsIndexConverter
    : public EmitCConversionPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp extractOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BaseMemRefType sourceTy = extractOp.getSource().getType();

    if (!sourceTy.hasRank())
      return failure();

    Location loc = extractOp.getLoc();
    Value alignedPtr = EmitCMemRefDescriptor(adaptor.getSource())
                           .getMemRefAlignedPtr(rewriter, loc);
    rewriter.replaceOpWithNewOp<emitc::CastOp>(
        extractOp, getTypeConverter()->convertType(extractOp.getType()),
        alignedPtr);
    return success();
  }
};

/// Convert `memref.copy` to `std::memcpy` for host-visible contiguous
/// memrefs.
struct MemRefCopyOpLowering : public EmitCConversionPattern<memref::CopyOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstType = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcType || !dstType)
      return rewriter.notifyMatchFailure(op, "unranked memref not supported");

    // Check both are contiguous (non-strided layouts).
    if (!isContiguous(srcType) || !isContiguous(dstType))
      return rewriter.notifyMatchFailure(
          op, "source or destination is not contiguous");

    // Check both are in host-visible memory space.
    std::optional<plan::MemorySpace> srcSpace = getMemorySpace(srcType);
    std::optional<plan::MemorySpace> dstSpace = getMemorySpace(dstType);

    auto isHostVisible = [](std::optional<plan::MemorySpace> space) {
      return space && (*space == plan::MemorySpace::host ||
                       *space == plan::MemorySpace::host_pinned ||
                       *space == plan::MemorySpace::unified);
    };

    if (!isHostVisible(srcSpace) || !isHostVisible(dstSpace))
      return rewriter.notifyMatchFailure(
          op, "source or destination is not in host-visible memory");

    Location loc = op.getLoc();
    EmitCMemRefDescriptor src(adaptor.getSource());
    EmitCMemRefDescriptor dst(adaptor.getTarget());

    Value srcPtr = src.getMemRefBufferStart(rewriter, loc, dataLayout,
                                            srcType.getElementType());
    Value dstPtr = dst.getMemRefBufferStart(rewriter, loc, dataLayout,
                                            dstType.getElementType());
    Value size = src.getSizeInBytes(rewriter, loc, dataLayout, srcType);

    // Call std::memcpy(dst, src, size).
    callOpaque(rewriter, loc, Type{}, "std::memcpy", {dstPtr, srcPtr, size});

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Math conversions missing from 'math-to-emitc' pass
//===----------------------------------------------------------------------===//

namespace {
struct MathLogToEmitCPattern : public EmitCConversionPattern<math::LogOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(math::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getOperand();
    Type inputType = input.getType();
    if (!inputType.isF32() && !inputType.isF64())
      return failure();
    llvm::StringRef funcName = "log";
    if (inputType.isF32())
      funcName = "logf";
    auto callOp =
        createCallOpaque(rewriter, op.getLoc(), inputType, funcName, {input});
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// CF Conversions
//===----------------------------------------------------------------------===//

namespace {
struct CFAssertPattern : public EmitCConversionPattern<cf::AssertOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cf::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value condition = adaptor.getArg();
    StringRef msg = op.getMsg();

    Value assertCondition = condition;
    // Escape quotes, backslashes, and curly braces for C++ string literal.
    // Curly braces must be escaped because EmitC uses `{}` for SSA value
    // substitution.
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
                                       assertCondition);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Executor ABI Conversions
//===----------------------------------------------------------------------===//

namespace {
struct ExecutorABIRecvPattern
    : public EmitCConversionPattern<executor::ABIRecvOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(executor::ABIRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getType();
    Type convertedResultType = typeConverter->convertType(resultType);
    if (!convertedResultType)
      return failure();

    Type targetPtrType = getPointerType(convertedResultType);
    Value ptr = adaptor.getPtr();
    if (targetPtrType != ptr.getType()) {
      ptr = createCast(rewriter, targetPtrType, ptr);
    }

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
      if (targetPtrType != ptr.getType()) {
        ptr = createCast(rewriter, targetPtrType, ptr);
      }
      Value zero = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value dest = rewriter.create<emitc::SubscriptOp>(
          op.getLoc(), getLValueType(convertedValueType), ptr, zero);
      rewriter.create<emitc::AssignOp>(op.getLoc(), dest, adaptor.getValue());
      rewriter.eraseOp(op);
      return success();
    };

    // Check if value is a scalar type (FloatType, IndexType, IntegerType) or
    // ComplexType
    const bool isScalar =
        isa<FloatType, IndexType, IntegerType, ComplexType>(valueType);

    if (isScalar)
      return lowerToStore();

    // Check if value is a MemRef type
    if (auto memrefType = dyn_cast<MemRefType>(valueType)) {
      // For MemRef types, check additional conditions:
      // 1. The ABIArgumentAttr must be marked as undef
      // 2. The ownership value must be statically known to be true

      // Get the function containing this operation
      auto func = op->getParentOfType<FunctionOpInterface>();
      if (!func)
        return failure();

      // Get the block argument for the ptr operand
      auto blockArg = dyn_cast<BlockArgument>(op.getPtr());
      if (!blockArg)
        return failure();

      // Get the ABI attribute for this argument
      executor::ArgumentABIAttr abiAttr =
          executor::abi::getArgumentABIAttr(func, blockArg.getArgNumber());
      if (!abiAttr)
        return failure();

      // Check if the argument has 'undef' parameter set.
      // If not, then this operation is a no-op: it is functionally pure since
      // "abi.send" on a by-value argument has a copy-on-write semantic. we
      // already verified that it has no users. Therefore, we can just erase
      // the operation.
      if (!abiAttr.getUndef()) {
        rewriter.eraseOp(op);
        return success();
      }

      // Check if ownership is statically known to be true
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

//===----------------------------------------------------------------------===//
// HostToEmitC Pass/Patterns API
//===----------------------------------------------------------------------===//

static Value materializeConversion(OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) {
  if (inputs.size() == 1 &&
      emitc::CastOp::areCastCompatible(inputs.front().getType(), resultType))
    return builder.create<emitc::CastOp>(loc, resultType, inputs[0]);
  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      .getResult(0);
}

/// Populate EmitC type conversions and op conversion patterns.
static void populateEmitCConversionPatternsAndLegality(
    const DataLayout &dataLayout, TypeConverter &typeConverter,
    ConversionTarget &target, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  Type cuEngineType = emitc::OpaqueType::get(ctx, "nvinfer1::ICudaEngine");
  Type cuEnginePtrType = emitc::PointerType::get(cuEngineType);
  Type trtExecCtxType =
      emitc::OpaqueType::get(ctx, "nvinfer1::IExecutionContext");
  Type trtExecCtxPtrType = emitc::PointerType::get(trtExecCtxType);
  Type voidPtrType{
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))};
  typeConverter.addConversion([](Type t) { return t; });
  typeConverter.addConversion(
      [&](BaseMemRefType memrefType) -> std::optional<Type> {
        if (memrefType.hasRank())
          return getMemRefDescriptorType(memrefType.getContext(),
                                         memrefType.getRank());
        return {};
      });

  typeConverter.addConversion(
      [=](executor::PointerType pointerType) { return voidPtrType; });

  typeConverter.addConversion([=](Type t) -> std::optional<Type> {
    if (isa<trtrt::EngineType>(t))
      return cuEnginePtrType;
    if (isa<trtrt::ExecutionContextType>(t))
      return trtExecCtxPtrType;
    return {};
  });

  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    StringRef name =
        llvm::TypeSwitch<Type, StringRef>(t)
            .Case<cuda::ModuleType>(
                [](cuda::ModuleType) -> StringRef { return "CUmodule"; })
            .Case<cuda::FunctionType>(
                [](cuda::FunctionType) -> StringRef { return "CUfunction"; })
            .Case<cuda::StreamType>(
                [](cuda::StreamType) -> StringRef { return "CUstream"; })
            .Case<cuda::StreamType>(
                [](cuda::StreamType) -> StringRef { return "CUstream"; })
            .Default([](Type) -> StringRef { return ""; });
    if (name.empty())
      return {};
    return emitc::OpaqueType::get(t.getContext(), name);
  });

  // Setup conversion materialization functions.
  typeConverter.addSourceMaterialization(materializeConversion);
  typeConverter.addTargetMaterialization(materializeConversion);

  // Setup legality constraints.
  target.addLegalOp<executor::FileArtifactOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<trtrt::TensorRTRuntimeDialect, cuda::CUDADialect,
                           executor::ExecutorDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&typeConverter](func::CallOp op) {
    return typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op->getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&typeConverter](func::ReturnOp op) {
        return mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
      });

  // clang-format off
  patterns.add<
      CFAssertPattern,
      CUDAAllocConverter,
      CudaCopyConverter<cuda::CopyD2DOp>,
      CudaCopyConverter<cuda::CopyD2HOp>,
      CudaCopyConverter<cuda::CopyH2DOp>,
      CudaDeallocConverter,
      CUDAGetCurrentDeviceConverter,
      CUDALaunchConverter,
      CUDAStreamSyncConverter,
      ExecutorABIRecvPattern,
      ExecutorABISendPattern,
      ExecutorPrintConverter,
      ExtractStridedMetadataOpLowering,
      MathLogToEmitCPattern,
      MemRefAllocOpLowering,
      MemrefCastOpLowering,
      MemRefCopyOpLowering,
      MemRefDeallocLowering,
      MemRefDimOpLowering,
      MemRefExtractAlignedPointerAsIndexConverter,
      MemRefLoadOpLowering,
      MemRefReinterpretCastOpLowering,
      MemRefStoreOpLowering,
      TRTEnqueueConverter
    >(typeConverter, dataLayout, patterns.getContext());
  // clang-format on
  mlir::populateSCFToEmitCConversionPatterns(patterns, typeConverter);
  mlir::populateArithToEmitCPatterns(typeConverter, patterns);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
}

namespace {
class HostToEmitCPass
    : public mlir::impl::ConvertHostToEmitCPassBase<HostToEmitCPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Detect which runtime headers are required before conversion.
    // We prefer to only include TensorRT / CUDA headers when actually used so
    // that non-TRT EmitC outputs can compile in more environments.
    bool needsCudaRuntime = false;
    bool needsTensorRTRuntime = false;
    moduleOp.walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      needsCudaRuntime |= isa<cuda::CUDADialect>(dialect);
      needsTensorRTRuntime |= isa<trtrt::TensorRTRuntimeDialect>(dialect);
    });

    if (!moduleOp.getSymName())
      moduleOp.setSymName("unnamed_module");

    // Before running pattern-based conversion driver, handle ctor/dtor.
    IRRewriter rewriter(moduleOp->getContext());

    HostToEmitCGlobalsConverter converter(moduleOp);
    if (failed(converter.convert()))
      return signalPassFailure();

    const DataLayoutAnalysis &dataLayoutAnalysis =
        getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(moduleOp);

    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    populateEmitCConversionPatternsAndLegality(dataLayout, typeConverter,
                                               target, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply conversion in " << getArgument();
      return signalPassFailure();
    }

    //===----------------------------------------------------------------------===//
    // Handle unrealized casts
    //===----------------------------------------------------------------------===//

    //===----------------------------------------------------------------------===//
    // Insert include ops
    //===----------------------------------------------------------------------===//
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cstdio", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cstdint", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cstddef", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cstdlib", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cstring", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cmath", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "cassert", true);
    rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "MTRTRuntimeCore.h",
                                      false);
    if (needsCudaRuntime)
      rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(), "MTRTRuntimeCuda.h",
                                        false);
    if (needsTensorRTRuntime)
      rewriter.create<emitc::IncludeOp>(moduleOp->getLoc(),
                                        "MTRTRuntimeTensorRT.h", false);

    //===----------------------------------------------------------------------===//
    // cleanup
    //===----------------------------------------------------------------------===//
    RewritePatternSet cleanupPatterns(moduleOp->getContext());
    cleanupPatterns.add(
        +[](emitc::CastOp op, PatternRewriter &rewriter) -> LogicalResult {
          // Eliminate useless casts.
          if (op.getType() == op.getOperand().getType()) {
            rewriter.replaceOp(op, op.getOperand());
            return llvm::success();
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::CastOp>()) {
            if (isa<emitc::PointerType, IndexType, IntegerType, FloatType>(
                    parent.getType())) {
              rewriter.modifyOpInPlace(op, [&]() {
                op.getSourceMutable().assign(parent.getSource());
              });
              if (parent->use_empty())
                rewriter.eraseOp(parent);
              return llvm::success();
            }
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::VariableOp>()) {
            if (!parent->hasOneUse())
              return failure();
            rewriter.modifyOpInPlace(
                parent, [&]() { parent.getResult().setType(op.getType()); });
            rewriter.replaceOp(op, parent);
            return llvm::success();
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::ApplyOp>()) {
            if (!parent->hasOneUse())
              return failure();
            rewriter.modifyOpInPlace(
                parent, [&]() { parent.getResult().setType(op.getType()); });
            rewriter.replaceOp(op, parent);
            return llvm::success();
          }
          return failure();
        });

    if (failed(mlir::applyPatternsGreedily(moduleOp,
                                           std::move(cleanupPatterns)))) {
      emitError(moduleOp->getLoc())
          << "failed to apply cleanup patterns in " << getArgument();
      return signalPassFailure();
    }

    // For functions, we convert `executor.ptr` arguments to void-pointers in
    // the rewrite patterns. This is because the function argument value types
    // are in function ABI attributes and can't be used as part of the type
    // conversion. At this step, we look at all arguments which have ABI
    // information. If there is a more refiend ABI value type available for
    // poitner arguments, update the pointer type. This is always legal since
    // users of `void*` values are always `emitc.cast` operations.
    // In addition, remove any other attributes which are no longer
    // relevant and will fail verification after this type conversion.
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      if (executor::abi::isABIWrapperFunction(funcOp))
        funcOp->removeAttr(executor::ExecutorDialect::kFuncABIAttrName);

      FunctionType funcType = funcOp.getFunctionType();
      SmallVector<Type> updatedArgTypes(funcType.getInputs());
      SmallVector<Type> updatedResultTypes(funcType.getResults());

      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        if (auto abiAttr = funcOp.getArgAttrOfType<executor::ArgumentABIAttr>(
                i, executor::ExecutorDialect::kArgABIAttrName)) {
          if (Type convertedType =
                  typeConverter.convertType(abiAttr.getValueType())) {
            Type newPointerType = getPointerType(convertedType);
            updatedArgTypes[i] = newPointerType;
            funcOp.getArgument(i).setType(newPointerType);
          }
          funcOp.removeArgAttr(i, executor::ExecutorDialect::kArgABIAttrName);
        }
        if (funcOp.getArgAttr(i, plan::PlanDialect::kShapeBoundsAttrName))
          funcOp.removeArgAttr(i, plan::PlanDialect::kShapeBoundsAttrName);
        if (funcOp.getArgAttr(i, plan::PlanDialect::kValueBoundsAttrName))
          funcOp.removeArgAttr(i, plan::PlanDialect::kValueBoundsAttrName);
      }

      funcOp.setFunctionType(FunctionType::get(
          funcOp.getContext(), updatedArgTypes, updatedResultTypes));
    }
  }
};
} // namespace

void mtrt::compiler::applyEmitCLoweringPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToEmitC());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createConvertHostToEmitCPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}
