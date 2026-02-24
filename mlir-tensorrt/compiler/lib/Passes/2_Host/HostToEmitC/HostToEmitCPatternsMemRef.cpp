//===- HostToEmitCPatternsMemRef.cpp --------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// MemRef op lowering patterns for `convert-host-to-emitc`.
///
/// These patterns lower MemRef ops to C++ that manipulates the StandaloneCPP
/// memref ABI (`mtrt::RankedMemRef<rank>`).
///
/// Think of `mtrt::RankedMemRef<rank>` as the C++ "descriptor" containing:
///   - allocated/aligned pointers
///   - offset
///   - sizes and strides
///
/// Loads/stores become pointer arithmetic + `ptr[offset]` subscripts.
//===----------------------------------------------------------------------===//

#include "HostToEmitCDetail.h"
#include "HostToEmitCDetailCommon.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
using namespace mlir::host_to_emitc;

namespace {

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
    // Intended C++ (schematic):
    //   void *allocated = mtrt::memref_descriptor_get_allocated_ptr(src);
    //   void *aligned   = mtrt::memref_descriptor_get_aligned_ptr(src);
    //   int64_t offset  = mtrt::memref_descriptor_get_offset(src);
    //   auto baseDesc   = mtrt::make_memref_descriptor<0>(allocated, aligned,
    //   0); return (baseDesc, offset, sizes..., strides...);
    Value sourceMemRef(adaptor.getSource());
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    auto sourceMemRefType = cast<MemRefType>(source.getType());
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    Value baseBuffer =
        createCallOpaque(rewriter, loc, voidPtrType,
                         "mtrt::memref_descriptor_get_allocated_ptr",
                         {sourceMemRef})
            .getResult(0);
    Value alignedBuffer =
        createCallOpaque(rewriter, loc, voidPtrType,
                         "mtrt::memref_descriptor_get_aligned_ptr",
                         {sourceMemRef})
            .getResult(0);
    Value offset =
        createCallOpaque(rewriter, loc, i64Type,
                         "mtrt::memref_descriptor_get_offset", {sourceMemRef})
            .getResult(0);

    Value dstMemRef = makeMemRefDescriptor(
        rewriter, loc, 0,
        {OpFoldResult(baseBuffer), OpFoldResult(alignedBuffer),
         OpFoldResult(getI32Val(rewriter, loc, 0))});

    results.push_back(dstMemRef);
    results.push_back(offset);

    for (unsigned i = 0; i < rank; ++i)
      results.push_back(
          createCallOpaque(rewriter, loc, i64Type,
                           "mtrt::memref_descriptor_get_dim_size",
                           {sourceMemRef, getI32Val(rewriter, loc, i)})
              .getResult(0));
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(
          createCallOpaque(rewriter, loc, i64Type,
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
    // Intended C++: rebuild a new `mtrt::RankedMemRef<rank>` with the desired
    // (possibly dynamic) sizes/strides, reusing the aligned pointer from
    // source.
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

    Location loc = castOp.getLoc();

    EmitCMemRefDescriptor srcMemRef(adaptor.getSource());
    SmallVector<OpFoldResult> args = {
        srcMemRef.getMemRefAllocatedPtr(rewriter, loc),
        srcMemRef.getMemRefAlignedPtr(rewriter, loc)};

    if (castOp.isDynamicOffset(0))
      args.push_back(adaptor.getOffsets()[0]);
    else
      args.push_back(getI32Val(rewriter, loc, 0));

    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;

    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        args.push_back(adaptor.getSizes()[dynSizeId++]);
      else
        args.push_back(rewriter.getI64IntegerAttr(castOp.getStaticSize(i)));
    }
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicStride(i))
        args.push_back(adaptor.getStrides()[dynStrideId++]);
      else
        args.push_back(rewriter.getI64IntegerAttr(castOp.getStaticStride(i)));
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
    // Intended C++ (host allocation):
    //   void *ptr = nullptr;
    //   int32_t st = mtrt::host_aligned_alloc(bytes, alignment, &ptr);
    //   mtrt::abort_on_error(st);
    //   return mtrt::make_memref_descriptor<rank>(ptr, ptr, 0, sizes...,
    //   strides...);
    if (!isContiguous(op.getType()))
      return failure();
    std::optional<plan::MemorySpace> memoryType = getMemorySpace(op.getType());
    if (!memoryType || *memoryType != plan::MemorySpace::host)
      return rewriter.notifyMatchFailure(op, "unsupported memory space");

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();
    Location loc = op.getLoc();

    SmallVector<OpFoldResult> sizes, strides;
    Value size;
    getMemRefDescriptorSizes(dataLayout, *getTypeConverter(), loc, op.getType(),
                             adaptor.getDynamicSizes(), rewriter, sizes,
                             strides, size, /*sizeInBytes=*/true);

    int32_t alignment = op.getAlignment() ? *op.getAlignment() : 16;
    Value alignVal = getI32Val(rewriter, loc, alignment);
    Value allocVar = rewriter.create<emitc::VariableOp>(
        loc, getLValueType(voidPtrType), getOpaqueAttr("nullptr"));
    Value allocAddr = getAddr(rewriter, loc, allocVar);
    Value st =
        builders.hostAlloc.create(rewriter, loc, {size, alignVal, allocAddr});
    emitStatusCheckOrAbort(rewriter, loc, st);
    Value alloc = rewriter.create<emitc::LoadOp>(loc, voidPtrType, allocVar);
    Value desc =
        makeMemRefDescriptor(rewriter, loc, alloc, alloc,
                             rewriter.getI64IntegerAttr(0), sizes, strides);
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
    // Intended C++:
    //   void *p = mtrt::memref_descriptor_get_allocated_ptr(desc);
    //   mtrt::host_free(p);
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

/// Convert `memref.dim` to descriptor queries or constants.
struct MemRefDimOpLowering : public EmitCConversionPattern<memref::DimOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp dimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `(size_t)mtrt::memref_descriptor_get_dim_size(desc, index)`
    Type operandType = dimOp.getSource().getType();
    if (isa<UnrankedMemRefType>(operandType))
      return failure();
    Location loc = dimOp.getLoc();
    Value dimSize = EmitCMemRefDescriptor(adaptor.getSource())
                        .getMemRefDimSize(rewriter, loc, adaptor.getIndex());
    // The runtime helper returns i64, but the MLIR index type converts to
    // size_t. Cast to the expected result type.
    Type resultType = getTypeConverter()->convertType(dimOp.getType());
    rewriter.replaceOpWithNewOp<emitc::CastOp>(dimOp, resultType, dimSize);
    return success();
  }
};

struct MemRefLoadOpLowering : public EmitCConversionPattern<memref::LoadOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   auto *typed = (T*)mtrt::memref_descriptor_get_aligned_ptr(desc);
    //   int64_t off = <linearized index including memref.offset>;
    //   return typed[off];
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

struct MemRefStoreOpLowering : public EmitCConversionPattern<memref::StoreOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   auto *typed = (T*)mtrt::memref_descriptor_get_aligned_ptr(desc);
    //   int64_t off = <linearized index including memref.offset>;
    //   typed[off] = value;
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
    // Intended C++: `(intptr_t)mtrt::memref_descriptor_get_aligned_ptr(desc)`
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

/// Convert `memref.copy` to `std::memcpy` for host-visible contiguous memrefs.
struct MemRefCopyOpLowering : public EmitCConversionPattern<memref::CopyOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++: `std::memcpy(dstStart, srcStart, bytes);`
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstType = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcType || !dstType)
      return rewriter.notifyMatchFailure(op, "unranked memref not supported");

    if (!isContiguous(srcType) || !isContiguous(dstType))
      return rewriter.notifyMatchFailure(
          op, "source or destination is not contiguous");

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

    createCallOpaque(rewriter, loc, {}, "std::memcpy", {dstPtr, srcPtr, size});

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::host_to_emitc {
void populateHostToEmitCMemRefPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       const DataLayout &dataLayout) {
  patterns
      .add<ExtractStridedMetadataOpLowering, MemRefReinterpretCastOpLowering,
           MemRefAllocOpLowering, MemrefCastOpLowering, MemRefCopyOpLowering,
           MemRefDeallocLowering, MemRefDimOpLowering,
           MemRefExtractAlignedPointerAsIndexConverter, MemRefLoadOpLowering,
           MemRefStoreOpLowering>(typeConverter, dataLayout,
                                  patterns.getContext());
}
} // namespace mlir::host_to_emitc
