//===- HostToEmitCDetail.h --------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Internal helpers for the HostToEmitC conversion.
///
/// This header is intentionally private to the HostToEmitC implementation and
/// should not be included outside of `compiler/lib/Conversion/HostToEmitC/*`.
///
/// When reading conversion patterns, it helps to keep in mind the mapping:
///   - MLIR EmitC ops are "structured C++ AST-ish" IR.
///   - `emitc.call_opaque "foo"` is intended to become `foo(...)` in C++.
///   - `emitc.variable` is intended to become a local C++ variable.
///   - `emitc.get_global` is intended to become a reference to a C++ global.
///   - `emitc.assign` / `emitc.load` model C++ assignment / lvalue-to-rvalue.
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_DETAIL_H
#define MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_DETAIL_H

#include "HostToEmitCDetailCommon.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

namespace mlir::host_to_emitc {

inline constexpr int64_t kMaxSupportedRank = 8;

/// Convert global-symbol-like ops (`trtrt.compiled_func`,
/// `cuda.compiled_module`, `memref.global`, etc.) to EmitC globals and helper
/// functions. Implemented in `HostToEmitCGlobals.cpp`.
LogicalResult convertHostToEmitCGlobals(ModuleOp module,
                                        bool emitAggregateInitDestroy);

/// Populate pattern groups. Implemented in the corresponding `HostToEmitC*`
/// `.cpp` files.
void populateHostToEmitCTensorRTPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         const DataLayout &dataLayout);
void populateHostToEmitCCudaPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     const DataLayout &dataLayout);
void populateHostToEmitCMemRefPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       const DataLayout &dataLayout);
void populateHostToEmitCExecutorPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         const DataLayout &dataLayout);

inline emitc::OpaqueType getMemRefDescriptorType(MLIRContext *ctx,
                                                 int64_t rank) {
  // Intended C++ type: `mtrt::RankedMemRef<rank>`
  return emitc::OpaqueType::get(
      ctx, llvm::formatv("mtrt::RankedMemRef<{0}>", rank).str());
}

inline emitc::OpaqueType getPointerShapeDescriptorType(MLIRContext *ctx,
                                                       int64_t rank) {
  // Intended C++ type: `mtrt::PtrAndShape<rank>`
  return emitc::OpaqueType::get(
      ctx, llvm::formatv("mtrt::PtrAndShape<{0}>", rank).str());
}

struct EmitCCallBuilder {
  StringRef name;
  Type resultType;
  SmallVector<Type> argTypes;

  Value create(OpBuilder &b, Location loc, ValueRange args) const {
    // Intended C++: `name(args...)`
    auto callOp = createCallOpaque(
        b, loc, resultType ? TypeRange{resultType} : TypeRange{}, name, args);
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
  Type f32Type = Float32Type::get(ctx);
  Type cuModuleType = emitc::OpaqueType::get(ctx, "CUmodule");
  Type cuModulePtrType = emitc::PointerType::get(cuModuleType);
  Type cuFuncType = emitc::OpaqueType::get(ctx, "CUfunction");
  Type cuFuncPtrType = emitc::PointerType::get(cuFuncType);
  Type cuStreamType = emitc::OpaqueType::get(ctx, "CUstream");
  Type cudaEventType = emitc::OpaqueType::get(ctx, "cudaEvent_t");
  Type unrankedMemRefType = emitc::OpaqueType::get(ctx, "mtrt::UnrankedMemRef");
  Type unrankedMemRefPtrType = emitc::PointerType::get(unrankedMemRefType);
  Type unrankedMemRefMutType =
      emitc::OpaqueType::get(ctx, "mtrt::UnrankedMemRefMut");
  Type unrankedMemRefMutPtrType =
      emitc::PointerType::get(unrankedMemRefMutType);

  //===----------------------------------------------------------------------===//
  // TensorRT Runtime Functions
  //===----------------------------------------------------------------------===//
  // Each `EmitCCallBuilder` here corresponds to a C++ runtime function declared
  // in `executor/lib/Runtime/StandaloneCPP/MTRTRuntime*.h` (e.g.:
  // `mtrt::tensorrt_enqueue`, `mtrt::cuda_alloc`, ...).
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
                                  unrankedMemRefPtrType, i32Type,
                                  unrankedMemRefPtrType}};
  EmitCCallBuilder trtEnqueueAlloc = {"mtrt::tensorrt_enqueue_alloc",
                                      i32Type,
                                      {trtExecCtxPtrType, cuStreamType, i32Type,
                                       unrankedMemRefPtrType, i32Type,
                                       unrankedMemRefMutPtrType}};

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

  EmitCCallBuilder cudaGetProgramDevice = {
      "mtrt::cuda_get_program_device",
      i32Type,
      {/*logicalDeviceId*/ i32Type,
       /*outDevice*/ emitc::PointerType::get(i32Type)}};

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

  //===----------------------------------------------------------------------===//
  // CUDA Event Runtime Functions
  //===----------------------------------------------------------------------===//
  EmitCCallBuilder cudaEventCreate = {
      "mtrt::cuda_event_create",
      i32Type,
      {/*device*/ i32Type,
       /*outEvent*/ emitc::PointerType::get(cudaEventType)}};
  EmitCCallBuilder cudaEventRelease = {
      "mtrt::cuda_event_release", i32Type, {/*event*/ cudaEventType}};
  EmitCCallBuilder cudaStreamRecordEvent = {
      "mtrt::cuda_stream_record_event", i32Type, {cuStreamType, cudaEventType}};
  EmitCCallBuilder cudaStreamWaitEvent = {
      "mtrt::cuda_stream_wait_event", i32Type, {cuStreamType, cudaEventType}};
  EmitCCallBuilder cudaEventSync = {
      "mtrt::cuda_event_sync", i32Type, {/*event*/ cudaEventType}};
  EmitCCallBuilder cudaEventElapsedMsec = {
      "mtrt::cuda_event_elapsed_msec",
      i32Type,
      {/*start*/ cudaEventType, /*end*/ cudaEventType,
       /*outMs*/ emitc::PointerType::get(f32Type)}};

  Value createStrLiteral(OpBuilder &b, Location loc, StringRef literal) const {
    // Intended C++: `"literal"` (string literal constant)
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
    // Intended C++: `mtrt::memref_descriptor_get_allocated_ptr(desc)`
    return createCallOpaque(rewriter, loc, voidPtrType,
                            "mtrt::memref_descriptor_get_allocated_ptr", {desc})
        .getResult(0);
  }

  Value getMemRefAlignedPtr(OpBuilder &rewriter, Location loc) const {
    // Intended C++: `mtrt::memref_descriptor_get_aligned_ptr(desc)`
    return createCallOpaque(rewriter, loc, voidPtrType,
                            "mtrt::memref_descriptor_get_aligned_ptr", {desc})
        .getResult(0);
  }

  Value getMemRefDimSize(OpBuilder &rewriter, Location loc, Value dim) const {
    // Intended C++: `mtrt::memref_descriptor_get_dim_size(desc, dim)`
    return createCallOpaque(rewriter, loc, rewriter.getI64Type(),
                            "mtrt::memref_descriptor_get_dim_size", {desc, dim})
        .getResult(0);
  }

  Value getMemRefDimSize(OpBuilder &rewriter, Location loc, int64_t dim) const {
    return createCallOpaque(
               rewriter, loc, rewriter.getI64Type(),
               "mtrt::memref_descriptor_get_dim_size",
               SmallVector<Value>{desc, rewriter.create<emitc::ConstantOp>(
                                            loc, rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(dim))})
        .getResult(0);
  }

  Value getMemRefOffset(OpBuilder &rewriter, Location loc) const {
    // Intended C++: `mtrt::memref_descriptor_get_offset(desc)`
    return createCallOpaque(rewriter, loc, rewriter.getI64Type(),
                            "mtrt::memref_descriptor_get_offset", {desc})
        .getResult(0);
  }

  Value getMemRefBufferStart(OpBuilder &rewriter, Location loc,
                             const DataLayout &dataLayout,
                             Type elementType) const {
    // Intended C++: compute `(aligned_ptr + offset_in_bytes)`.
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

    Value numElements = b.create<emitc::ConstantOp>(loc, b.getI64Type(),
                                                    b.getI64IntegerAttr(1));
    for (int64_t pos = 0; pos < type.getRank(); pos++)
      numElements = b.create<emitc::MulOp>(loc, b.getI64Type(), numElements,
                                           getMemRefDimSize(b, loc, pos));
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

inline Value getAddrOfLValue(OpBuilder &b, Location loc, Value lvalue) {
  auto lv = cast<emitc::LValueType>(lvalue.getType());
  return b.create<emitc::ApplyOp>(
      loc, emitc::PointerType::get(lv.getValueType()), "&", lvalue);
}

inline void getMemRefDescriptorSizes(
    const mlir::DataLayout &dataLayout, const TypeConverter &typeConverter,
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides, Value &size, bool sizeInBytes) {
  assert(count(memRefType.getShape(), ShapedType::kDynamic) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");
  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  Type indexType = IntegerType::get(memRefType.getContext(), 64);
  for (int64_t dimSize : memRefType.getShape()) {
    sizes.push_back(
        dimSize == ShapedType::kDynamic
            ? dynamicSizes[dynamicIndex++]
            : rewriter.create<emitc::ConstantOp>(
                  loc, indexType, rewriter.getI64IntegerAttr(dimSize)));
  }

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
      runningStride = getValueOrCreateConstantIndexOp(rewriter, loc, sizes[i]);
    else if (stride == ShapedType::kDynamic)
      runningStride = rewriter.create<emitc::MulOp>(
          loc, indexType, runningStride,
          getValueOrCreateConstantIndexOp(rewriter, loc, sizes[i]));
    else
      runningStride = rewriter.create<emitc::ConstantOp>(
          loc, indexType, rewriter.getI64IntegerAttr(stride));
  }
  if (sizeInBytes) {
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

inline Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  int64_t rank, ArrayRef<OpFoldResult> args) {
  return createCallOpaque(rewriter, loc,
                          getMemRefDescriptorType(rewriter.getContext(), rank),
                          "mtrt::make_memref_descriptor", args,
                          rewriter.getI32IntegerAttr(rank))
      .getResult(0);
}

inline Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  OpFoldResult allocated, OpFoldResult aligned,
                                  OpFoldResult offset,
                                  ArrayRef<OpFoldResult> shape,
                                  ArrayRef<OpFoldResult> strides) {
  assert(shape.size() == strides.size() && "mismatched shape/stride ranks");
  SmallVector<OpFoldResult, 8> args = {allocated, aligned, offset};
  llvm::append_range(args, shape);
  llvm::append_range(args, strides);
  return makeMemRefDescriptor(rewriter, loc, shape.size(), args);
}

inline Value makeMemRefDescriptor(RewriterBase &rewriter, Location loc,
                                  Value allocated, Value aligned, Value offset,
                                  ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> strides) {
  assert(shape.size() == strides.size() && "mismatched shape/stride ranks");
  SmallVector<OpFoldResult> args = {allocated, aligned, offset};
  auto makeAttr = [&](int64_t x) -> Attribute {
    return rewriter.getI64IntegerAttr(x);
  };
  llvm::append_range(args, llvm::map_range(shape, makeAttr));
  llvm::append_range(args, llvm::map_range(strides, makeAttr));
  return createCallOpaque(
             rewriter, loc,
             getMemRefDescriptorType(rewriter.getContext(), shape.size()),
             "mtrt::make_memref_descriptor", args,
             rewriter.getI32IntegerAttr(shape.size()))
      .getResult(0);
}

inline std::optional<plan::MemorySpace> getMemorySpace(MemRefType type) {
  auto srcMemoryTypeAttr =
      dyn_cast_or_null<plan::MemorySpaceAttr>(type.getMemorySpace());
  if (!srcMemoryTypeAttr)
    return plan::MemorySpace::host;
  return srcMemoryTypeAttr.getValue();
}

inline Value getMemRefPtrShape(OpBuilder &rewriter, Location loc,
                               const DataLayout &dataLayout, MemRefType type,
                               Value sourceMemRef) {
  EmitCMemRefDescriptor sourceDesc(sourceMemRef);
  Value start = sourceDesc.getMemRefBufferStart(rewriter, loc, dataLayout,
                                                type.getElementType());
  auto makeAttr = [&](int64_t x) -> Attribute {
    return rewriter.getI64IntegerAttr(x);
  };
  if (type.hasStaticShape()) {
    SmallVector<OpFoldResult> args = {start};
    llvm::append_range(args, llvm::map_range(type.getShape(), makeAttr));
    return createCallOpaque(rewriter, loc,
                            getPointerShapeDescriptorType(rewriter.getContext(),
                                                          type.getRank()),
                            "mtrt::make_ptr_shape_descriptor", args,
                            rewriter.getI32IntegerAttr(type.getRank()))
        .getResult(0);
  }
  SmallVector<Value> args = {start};
  for (int64_t i = 0; i < type.getRank(); ++i)
    args.push_back(sourceDesc.getMemRefDimSize(rewriter, loc, i));
  return createCallOpaque(rewriter, loc,
                          getPointerShapeDescriptorType(rewriter.getContext(),
                                                        type.getRank()),
                          "mtrt::make_ptr_shape_descriptor", args,
                          rewriter.getI32IntegerAttr(type.getRank()))
      .getResult(0);
}

inline Value getStridedElementPtr(OpBuilder &rewriter, Location loc,
                                  const TypeConverter & /*typeConverter*/,
                                  MemRefType type,
                                  EmitCMemRefDescriptor memRefDesc,
                                  ValueRange indices) {
  auto [strides, offset] = type.getStridesAndOffset();
  (void)offset;
  Value index = memRefDesc.getMemRefOffset(rewriter, loc);
  auto getI64Val = [&](int64_t x) {
    return rewriter.create<emitc::ConstantOp>(loc, rewriter.getI64Type(),
                                              rewriter.getI64IntegerAttr(x));
  };
  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) {
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

inline emitc::PointerType getPointerType(Type elementType) {
  return emitc::PointerType::get(elementType);
}

//===----------------------------------------------------------------------===//
// Shared layout helpers (used by both CUDA and memref conversions).
//===----------------------------------------------------------------------===//

inline bool isContiguousImpl(ArrayRef<int64_t> strides,
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

  std::optional<unsigned> index = findNextIndex(0);
  if (!index)
    return true;

  while (*index < e) {
    std::optional<unsigned> next = findNextIndex(*index + 1);
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

inline bool isContiguous(MemRefType t) {
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

inline bool isCopyStrided(MemRefType srcMemRefType, MemRefType dstMemRefType) {
  return !isContiguous(srcMemRefType) || !isContiguous(dstMemRefType);
}

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
  const DataLayout &dataLayout;

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

  Value getUnrankedDescriptor(OpBuilder &b, Location loc, int64_t rank,
                              Value rankedDesc) const {
    Value rankVal = getI32Val(b, loc, rank);
    return createCallOpaque(b, loc, unrankedDescriptorType,
                            "mtrt::make_unranked_descriptor",
                            {rankVal, rankedDesc})
        .getResult(0);
  }
};

} // namespace mlir::host_to_emitc

#endif // MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_DETAIL_H
