//===- HostToEmitCPatternsCuda.cpp ----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// CUDA op lowering patterns for `convert-host-to-emitc`.
///
/// These patterns lower CUDA dialect ops to C++ calls into the StandaloneCPP
/// runtime (see `MTRTRuntimeCuda.h`). The intent is that the emitted C++ looks
/// like ordinary CUDA-driver-style host code, e.g.:
///   - `mtrt::cuda_launch_kernel(func, gridX, ..., stream, argv)`
///   - `mtrt::cuda_alloc(stream, bytes, isPinned, isManaged, &ptr)`
///   - `mtrt::cuda_copy(stream, src, dst, bytes)`
//===----------------------------------------------------------------------===//

#include "HostToEmitCDetail.h"
#include "HostToEmitCDetailCommon.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"

using namespace mlir;
using namespace mlir::host_to_emitc;

namespace {

struct CUDALaunchConverter : public EmitCConversionPattern<cuda::LaunchOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   void *argv[numArgs];
    //   // For each arg:
    //   auto tmp_i = <by-value arg>;
    //   argv[i] = (void*)&tmp_i;
    //   int32_t st = mtrt::cuda_launch_kernel(func, gridX, gridY, gridZ,
    //                                        blockX, blockY, blockZ,
    //                                        sharedMem, stream, argv);
    //   mtrt::abort_on_error(st);
    Location loc = op->getLoc();
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

      if (auto memRefType = dyn_cast<MemRefType>(originalType)) {
        EmitCMemRefDescriptor desc(value);
        rhs = desc.getMemRefBufferStart(rewriter, loc, dataLayout,
                                        memRefType.getElementType());
        storageType = voidPtrType;
      } else if (isa<IndexType>(originalType)) {
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
    // Intended C++:
    //   int32_t dev = 0;
    //   int32_t st = mtrt::cuda_get_current_device(&dev);
    //   mtrt::abort_on_error(st);
    //   return dev;
    (void)adaptor;
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
    // Intended C++:
    //   int32_t st = mtrt::cuda_stream_sync(stream);
    //   mtrt::abort_on_error(st);
    Location loc = op.getLoc();
    Value st =
        builders.cudaStreamSync.create(rewriter, loc, {adaptor.getStream()});
    emitStatusCheckOrAbort(rewriter, loc, st);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CUDAAllocConverter : public EmitCConversionPattern<cuda::AllocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(cuda::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   void *ptr = nullptr;
    //   int32_t st = mtrt::cuda_alloc(stream, sizeBytes, isPinned, isManaged,
    //   &ptr); mtrt::abort_on_error(st); return
    //   mtrt::make_memref_descriptor<rank>(ptr, ptr, /*offset*/0, sizes...,
    //   strides...);
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
    SmallVector<OpFoldResult> shape;
    SmallVector<OpFoldResult> strides;
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
    // Intended C++:
    //   int32_t st = mtrt::cuda_free(stream, allocatedPtr, isPinned,
    //   isManaged); mtrt::abort_on_error(st);
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
    // Intended C++:
    //   - contiguous case: `mtrt::cuda_copy(stream, srcStart, dstStart, bytes)`
    //   - strided case:    `mtrt::cuda_copy_strided(stream, srcStart,
    //   srcUnranked,
    //                                             dstStart, dstUnranked)`
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

    Value srcUnranked =
        createCallOpaque(rewriter, loc, this->unrankedDescriptorType,
                         "mtrt::make_unranked_descriptor",
                         {OpFoldResult(rewriter.getI32IntegerAttr(
                              cast<MemRefType>(srcType).getRank())),
                          OpFoldResult(src)})
            .getResult(0);

    Value dstUnranked =
        createCallOpaque(rewriter, loc, this->unrankedDescriptorType,
                         "mtrt::make_unranked_descriptor",
                         {OpFoldResult(rewriter.getI32IntegerAttr(
                              cast<MemRefType>(dstType).getRank())),
                          OpFoldResult(dest)})
            .getResult(0);

    createCallOpaque(
        rewriter, loc, {}, "mtrt::cuda_copy_strided",
        {adaptor.getStream(), srcStart, srcUnranked, destStart, dstUnranked});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::host_to_emitc {
void populateHostToEmitCCudaPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     const DataLayout &dataLayout) {
  patterns.add<CUDALaunchConverter, CUDAAllocConverter,
               CudaCopyConverter<cuda::CopyD2DOp>,
               CudaCopyConverter<cuda::CopyD2HOp>,
               CudaCopyConverter<cuda::CopyH2DOp>, CudaDeallocConverter,
               CUDAGetCurrentDeviceConverter, CUDAStreamSyncConverter>(
      typeConverter, dataLayout, patterns.getContext());
}
} // namespace mlir::host_to_emitc
