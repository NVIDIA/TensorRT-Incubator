//===- CUDAToExecutor.cpp -------------------------------------------------===//
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
/// Implementation of the `convert-cuda-to-executor` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/CUDAToExecutor/CUDAToExecutor.h"
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCUDATOEXECUTORPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

namespace {

template <typename OpType>
struct CUDAOpToExecutorCallLowering
    : public ConvertOpToExecutorPattern<OpType> {
public:
  using ConvertOpToExecutorPattern<OpType>::ConvertOpToExecutorPattern;

protected:
  MLIRContext *ctx = this->getTypeConverter()->getContext();
  Type indexType = this->getTypeConverter()->getIndexType();
  Type i32Type = IntegerType::get(ctx, 32);
  Type i64Type = IntegerType::get(ctx, 64);
  Type hostPointerType = executor::PointerType::get(ctx, MemoryType::host);
  Type pinnedPointerType =
      executor::PointerType::get(ctx, MemoryType::host_pinned);
  Type devicePointerType = executor::PointerType::get(ctx, MemoryType::device);
  Type strLiteralType = executor::StrLiteralType::get(ctx);

  ExecutorCallBuilder getActiveDeviceBuilder = {
      ctx, "__cuda_get_active_device", {i32Type}, {}};

  ExecutorCallBuilder setActiveDeviceBuilder = {
      ctx, "__cuda_set_active_device", {}, {i32Type}};

  ExecutorCallBuilder getDeviceCountBuilder = {
      ctx, "__cuda_num_devices", {i32Type}, {}};

  ExecutorCallBuilder getDeviceBuilder = {
      ctx, "__cuda_get_device", {i32Type}, {i32Type}};

  ExecutorCallBuilder hostToDeviceCopyBuilder = {
      ctx,
      "__cuda_memcpy_host2device",
      {},
      {hostPointerType, hostPointerType, indexType, devicePointerType,
       indexType, indexType}};
  ExecutorCallBuilder hostToHostPinnedCopyBuilder = {
      ctx,
      "__memcpy_host2host_pinned",
      {},
      {hostPointerType, indexType, pinnedPointerType, indexType, indexType}};
  ExecutorCallBuilder streamCreateBuilder = {
      ctx, "__cuda_stream_create", {hostPointerType}, {i32Type}};
  ExecutorCallBuilder streamSyncBuilder = {
      ctx, "__cuda_stream_sync", {}, {hostPointerType}};
  ExecutorCallBuilder streamDestroyBuilder = {
      ctx, "__cuda_stream_destroy", {}, {hostPointerType}};

  ExecutorCallBuilder deviceAllocBuilder = {
      ctx,
      "__cuda_alloc_device",
      {devicePointerType},
      {/*stream=*/hostPointerType, /*size=*/indexType, /*alignment=*/i32Type}};
  ExecutorCallBuilder hostPinnedAllocBuilder = {ctx,
                                                "__cuda_alloc_host_pinned",
                                                {pinnedPointerType},
                                                {indexType, i32Type}};

  ExecutorCallBuilder cudaGetFuncBuilder = {ctx,
                                            "__cuda_get_function",
                                            {hostPointerType},
                                            {hostPointerType, strLiteralType}};

  ExecutorCallBuilder cudaLaunchBuilder = {ctx,
                                           "__cuda_launch",
                                           {},
                                           {/*cuFunction*/ hostPointerType,
                                            /*grid*/ i32Type, i32Type, i32Type,
                                            /*block*/ i32Type, i32Type, i32Type,
                                            /*dsmem*/ i32Type,
                                            /*stream*/ hostPointerType,
                                            /*arg ptr array*/ hostPointerType}};
};

template <typename Derived, typename OpType>
struct SimpleCudaOpToExecutorCallLowering
    : public CUDAOpToExecutorCallLowering<OpType> {
  using CUDAOpToExecutorCallLowering<OpType>::CUDAOpToExecutorCallLowering;

  LogicalResult matchAndRewrite(
      OpType op,
      typename CUDAOpToExecutorCallLowering<OpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto callOp = static_cast<const Derived *>(this)->callBuilder.create(
        rewriter, op.getLoc(), op->template getParentOfType<ModuleOp>(),
        adaptor.getOperands());
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CuBlas op patterns
//===----------------------------------------------------------------------===//

// Converts `BlasRunGemmOp` to the executor. Based on whether problem is
// GEMM or MatMul, number of input arguments change.
// NOTE* This converter pattern must run after bufferization.
struct CudaBlasRunGemmOpConverter
    : public ConvertOpToExecutorPattern<cuda::BlasRunGemmOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(cuda::BlasRunGemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    std::string funcName =
        "_" + getTypeConverter()->convertOpNameToBackendBuiltinFuncName(
                  op->getName().getStringRef());
    SmallVector<Value> newOperands = {adaptor.getHandle(), adaptor.getStream()};
    newOperands.push_back(adaptor.getAlgo());
    auto createMemRefAndExractPtr = [&](Value oldVal, Value newVal) {
      auto memrefType = cast<MemRefType>(oldVal.getType());
      if (!memrefType)
        return failure();
      assert(isa<TableType>(newVal.getType()));
      executor::MemRefDescriptor memref(newVal, memrefType);
      newOperands.push_back(memref.alignedPtr(b));
      return success();
    };
    if (op.getAlpha() &&
        failed(createMemRefAndExractPtr(op.getAlpha(), adaptor.getAlpha())))
      return failure();

    if (failed(createMemRefAndExractPtr(op.getMatA(), adaptor.getMatA())))
      return failure();

    if (failed(createMemRefAndExractPtr(op.getMatB(), adaptor.getMatB())))
      return failure();

    if (op.getBeta() &&
        failed(createMemRefAndExractPtr(op.getBeta(), adaptor.getBeta())))
      return failure();

    if (failed(createMemRefAndExractPtr(op.getMatC(), adaptor.getMatC())))
      return failure();

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();

    ExecutorCallBuilder externalCallBuilder = {
        getContext(), funcName, newResultTypes,
        llvm::to_vector(TypeRange(newOperands))};
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.replaceOp(
        op, externalCallBuilder.create(b, op.getLoc(), module, newOperands));
    return success();
  }
};
} // namespace

// TypeAttr representing problem data type is converted to runtime
// ScalarTypeCode. It is converted back to cuBLAS and CUDA types during
// algorithm selection process at runtime.
static int64_t convertTypeAttrToScalarType(Type t) {
  if (t.isF16())
    return static_cast<int64_t>(mtrt::ScalarTypeCode::f16);
  if (t.isF32())
    return static_cast<int64_t>(mtrt::ScalarTypeCode::f32);
  if (t.isF64())
    return static_cast<int64_t>(mtrt::ScalarTypeCode::f64);
  if (t.isInteger(32))
    return static_cast<int64_t>(mtrt::ScalarTypeCode::i32);
  llvm_unreachable("unhandled or invalid data type to convert to cuBLAS");
}

namespace {

// Converts `BlasHeuristicAlgoSelectionOp` to the executor. Always TOP 1
// algorithm is selected.
struct CudaBlasHeuristicAlgoSelectionOpConverter
    : public ConvertOpToExecutorPattern<cuda::BlasHeuristicAlgoSelectionOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(cuda::BlasHeuristicAlgoSelectionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    std::string funcName =
        "_" + getTypeConverter()->convertOpNameToBackendBuiltinFuncName(
                  op->getName().getStringRef());

    SmallVector<Value> newOperands = {adaptor.getHandle()};
    int64_t dataTypeInt = convertTypeAttrToScalarType(adaptor.getDataType());
    newOperands.push_back(b.create<executor::ConstantOp>(
                               b.getI64Type(), b.getI64IntegerAttr(dataTypeInt))
                              .getResult());

    // Batch size is either 1 or N, where N is batch size in 3D input.
    if (op.getSizeAAttr().asArrayRef().size() == 2)
      newOperands.push_back(
          b.create<executor::ConstantOp>(b.getI64Type(), b.getI64IntegerAttr(1))
              .getResult());
    else
      newOperands.push_back(
          b.create<executor::ConstantOp>(
               b.getI64Type(),
               b.getI64IntegerAttr(op.getSizeAAttr().asArrayRef()[0]))
              .getResult());

    auto createConstantsForArrayAttr = [&](DenseI64ArrayAttr arrayAttr) {
      // Batch size is already added so pick only last two dimensions
      if (arrayAttr.asArrayRef().size() == 3) {
        for (auto e : arrayAttr.asArrayRef().drop_front(1)) {
          newOperands.push_back(b.create<executor::ConstantOp>(
                                     b.getI64Type(), b.getI64IntegerAttr(e))
                                    .getResult());
        }
        return success();
      } else {
        for (auto e : arrayAttr.asArrayRef()) {
          newOperands.push_back(b.create<executor::ConstantOp>(
                                     b.getI64Type(), b.getI64IntegerAttr(e))
                                    .getResult());
        }
      }
      return success();
    };

    Value one =
        b.create<executor::ConstantOp>(b.getI64Type(), b.getI64IntegerAttr(1))
            .getResult();
    Value zero =
        b.create<executor::ConstantOp>(b.getI64Type(), b.getI64IntegerAttr(0))
            .getResult();
    if (failed(createConstantsForArrayAttr(adaptor.getSizeAAttr())))
      return failure();
    if (failed(createConstantsForArrayAttr(adaptor.getStrideAAttr())))
      return failure();
    newOperands.push_back(op->hasAttr("transpose_a") ? one : zero);

    if (failed(createConstantsForArrayAttr(adaptor.getSizeBAttr())))
      return failure();
    if (failed(createConstantsForArrayAttr(adaptor.getStrideBAttr())))
      return failure();
    newOperands.push_back(op->hasAttr("transpose_b") ? one : zero);

    if (failed(createConstantsForArrayAttr(adaptor.getSizeCAttr())))
      return failure();
    if (failed(createConstantsForArrayAttr(adaptor.getStrideCAttr())))
      return failure();

    // Don't transpose C
    newOperands.push_back(zero);

    // Tile sizes is optional argument
    if (failed(createConstantsForArrayAttr(adaptor.getTileSizesAttr())))
      return failure();

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();

    ExecutorCallBuilder externalCallBuilder = {
        b.getContext(), funcName, newResultTypes,
        llvm::to_vector(TypeRange(newOperands))};
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.replaceOp(
        op, externalCallBuilder.create(b, op.getLoc(), module, newOperands));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CUDA Launch & Stream Conversions
//===----------------------------------------------------------------------===//

/// Convert `CudaLaunchKernelOp` to a variadic `executor.call` operation. The
/// variadic function is declared `executor_cuda_launch_kernel` at the top of
/// the module.
struct LowerCudaLaunchKernelToCall
    : public CUDAOpToExecutorCallLowering<cuda::LaunchOp> {
  using CUDAOpToExecutorCallLowering::CUDAOpToExecutorCallLowering;

  LogicalResult
  matchAndRewrite(cuda::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value one = rewriter.create<executor::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 1));

    SmallVector<Value> storagePtrs;
    SmallVector<Value> promotedArgs;
    if (failed(this->getTypeConverter()->promoteOperands(
            loc, op.getArgs(), adaptor.getArgs(), rewriter,
            /*useBarePointerCallConv=*/true, promotedArgs)))
      return failure();

    Value zero = rewriter.create<executor::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));

    for (Value toStoreVal : promotedArgs) {
      Value valuePtr = rewriter.create<executor::AllocaOp>(
          loc, hostPointerType, one, IntegerAttr{}, toStoreVal.getType());
      rewriter.create<executor::StoreOp>(loc, valuePtr, zero, toStoreVal);
      storagePtrs.push_back(valuePtr);
    }

    // Create and populate the array-of-pointers that is required by the
    // launch config.
    auto operandPtrStorageType = executor::TableType::get(
        ctx, SmallVector<Type>(storagePtrs.size(), hostPointerType));
    auto argPtrsPtr = rewriter.create<executor::AllocaOp>(
        loc, hostPointerType, one, IntegerAttr{}, operandPtrStorageType);
    for (auto [idx, value] : llvm::enumerate(storagePtrs)) {
      auto offsetOp = rewriter.create<executor::GetOffsetOp>(
          loc, indexType, operandPtrStorageType,
          ArrayRef<OpFoldResult>{rewriter.getI64IntegerAttr(0),
                                 rewriter.getI64IntegerAttr(idx)});
      rewriter.create<executor::StoreOp>(loc, argPtrsPtr, offsetOp, value);
    }

    cudaLaunchBuilder.create(
        rewriter, loc, op->getParentOfType<ModuleOp>(),
        /*functionInputs*/
        {adaptor.getFunc(), adaptor.getGridX(), adaptor.getGridY(),
         adaptor.getGridZ(), adaptor.getBlockX(), adaptor.getBlockY(),
         adaptor.getBlockZ(), adaptor.getDynamicSharedMem(),
         adaptor.getStream(), argPtrsPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

class CudaGetGlobalStreamConverter
    : public CUDAOpToExecutorCallLowering<cuda::GetGlobalStreamOp> {
  using CUDAOpToExecutorCallLowering::CUDAOpToExecutorCallLowering;

  LogicalResult
  matchAndRewrite(cuda::GetGlobalStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchPattern(op.getDevice(), m_Op<cuda::GetActiveDeviceOp>())) {
      return emitError(op.getLoc(), "currently only 'cuda.get_active_device' "
                                    "is supported as device for global streams")
                 .attachNote(op.getDevice().getLoc())
             << "see device here";
    }
    auto module = op->getParentOfType<ModuleOp>();
    std::string name = llvm::formatv("stream{0}", op.getIndex());
    MLIRContext *ctx = rewriter.getContext();
    auto hostPointerType = PointerType::get(ctx, MemoryType::host);
    GlobalOp global = executor::getOrCreateGlobalOp(
        rewriter, module.getLoc(), module, name, hostPointerType, true,
        [&](OpBuilder &b, Location loc) {
          Value device =
              this->getActiveDeviceBuilder
                  .create(b, op.getLoc(), op->getParentOfType<ModuleOp>(), {})
                  .getResult(0);
          Value stream = this->streamCreateBuilder
                             .create(b, op.getLoc(),
                                     op->getParentOfType<ModuleOp>(), {device})
                             .getResult(0);
          b.create<executor::ReturnOp>(loc, stream);
        });
    rewriter.replaceOpWithNewOp<executor::GetGlobalOp>(op, global);
    return success();
  }
};

struct CudaGetActiveDeviceConverter final
    : public SimpleCudaOpToExecutorCallLowering<CudaGetActiveDeviceConverter,
                                                cuda::GetActiveDeviceOp> {
  using SimpleCudaOpToExecutorCallLowering::SimpleCudaOpToExecutorCallLowering;

  ExecutorCallBuilder &callBuilder = this->getActiveDeviceBuilder;
};

struct CudaSetActiveDeviceConverter final
    : public SimpleCudaOpToExecutorCallLowering<CudaSetActiveDeviceConverter,
                                                cuda::SetActiveDeviceOp> {
  using SimpleCudaOpToExecutorCallLowering::SimpleCudaOpToExecutorCallLowering;

  ExecutorCallBuilder &callBuilder = this->setActiveDeviceBuilder;
};

struct CudaGetDeviceCountConverter final
    : public SimpleCudaOpToExecutorCallLowering<CudaGetDeviceCountConverter,
                                                cuda::DeviceCountOp> {
  using SimpleCudaOpToExecutorCallLowering::SimpleCudaOpToExecutorCallLowering;

  ExecutorCallBuilder &callBuilder = this->getDeviceCountBuilder;
};
} // namespace

// Constructs two temp buffers:
// - One buffer holds the actual argument values.
// - One buffer holds a pointer to each argument value in
//   the first buffer.
// The function returns the pointer to the second buffer and the
// number of arguments (number of pointers).
static Value
promoteShapeAndStridesToAlloca(ImplicitLocOpBuilder &rewriter,
                               const ExecutorTypeConverter &typeConverter,
                               MemRefDescriptor desc) {
  MLIRContext *context = rewriter.getContext();
  SmallVector<Type> bodyTypes(desc.getMemRefType().getRank() * 2,
                              typeConverter.getIndexType());
  auto structType = executor::TableType::get(context, bodyTypes);
  auto llvmPtrIntType = typeConverter.getIndexType();
  auto llvmPointerType =
      executor::PointerType::get(context, executor::MemoryType::host);
  auto one =
      rewriter.create<executor::ConstantOp>(rewriter.getI32IntegerAttr(1));
  auto structPtr = rewriter.create<executor::AllocaOp>(
      llvmPointerType, one, IntegerAttr{}, structType);

  for (int64_t i = 0; i < desc.getMemRefType().getRank() * 2; i++) {
    Value fieldValueOffset = rewriter.create<executor::GetOffsetOp>(
        llvmPtrIntType, structType,
        ArrayRef<OpFoldResult>{rewriter.getI64IntegerAttr(0),
                               rewriter.getI64IntegerAttr(i)});
    rewriter.create<executor::StoreOp>(
        structPtr, fieldValueOffset,
        i < desc.getMemRefType().getRank()
            ? desc.size(rewriter, i)
            : desc.stride(rewriter, i - desc.getMemRefType().getRank()));
  }
  return structPtr;
}

namespace {
/// Lowers `cuda.memcpy_*` to a call to externally defined function.
template <typename OpTy>
struct CudaMemCopyOpToBuiltinCallConverter
    : public CUDAOpToExecutorCallLowering<OpTy> {
  using CUDAOpToExecutorCallLowering<OpTy>::CUDAOpToExecutorCallLowering;

  LogicalResult matchAndRewrite(
      OpTy op, typename CUDAOpToExecutorCallLowering<OpTy>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Determine whether the memref is contiguous.
    MemRefType srcType = op.getSource().getType();
    MemRefType dstType = op.getTarget().getType();
    std::optional<MemoryType> srcSpace = this->getMemorySpace(srcType);
    std::optional<MemoryType> dstSpace = this->getMemorySpace(dstType);
    if (!srcSpace || !dstSpace)
      return failure();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefDescriptor src(adaptor.getSource(), srcType);
    MemRefDescriptor dest(adaptor.getTarget(), dstType);

    if (!this->isCopyStrided(srcType, dstType)) {
      Value srcOffset =
          this->convertOffsetInElementsToBytes(b, src.offset(b), srcType);
      Value dstOffset =
          this->convertOffsetInElementsToBytes(b, dest.offset(b), dstType);

      // By definition, contiguous copies are not strided and thus the copy
      // size is equivalent to the shape volume (stride can be disregarded).
      Value sizeBytes = this->convertOffsetInElementsToBytes(
          b, src.shapeVolumeInElements(b), srcType);

      SmallVector<Value> callOperands = {
          adaptor.getStream(), src.alignedPtr(b), srcOffset,
          dest.alignedPtr(b),  dstOffset,         sizeBytes};
      MLIRContext *ctx = rewriter.getContext();
      std::string name =
          llvm::formatv("__cuda_memcpy_{0}2{1}", stringifyMemoryType(*srcSpace),
                        stringifyMemoryType(*dstSpace))
              .str();
      ExecutorCallBuilder copyBuilder = {
          ctx,
          name,
          {},
          {this->hostPointerType, callOperands[1].getType(), this->indexType,
           callOperands[3].getType(), this->indexType, this->indexType}};
      copyBuilder.create(rewriter, op.getLoc(),
                         op->template getParentOfType<ModuleOp>(),
                         callOperands);
      rewriter.eraseOp(op);
      return success();
    }

    // Allocate space for the full descriptors.
    Value shapeAndStridesSrc =
        promoteShapeAndStridesToAlloca(b, *this->getTypeConverter(), src);
    Value shapeAndStridesDest =
        promoteShapeAndStridesToAlloca(b, *this->getTypeConverter(), dest);

    // Strided copy conversion.
    SmallVector<Value> operands = {
        adaptor.getStream(),
        this->createIndexConstant(b, srcType.getRank()),
        this->createIndexConstant(
            b, this->getTypeConverter()->getMemRefElementTypeByteSize(srcType)),
        src.alignedPtr(b),
        src.offset(b),
        shapeAndStridesSrc,
        dest.alignedPtr(b),
        dest.offset(b),
        shapeAndStridesDest};

    MLIRContext *ctx = rewriter.getContext();
    std::string name = llvm::formatv("__cuda_memcpy_strided_async_{0}2{1}",
                                     stringifyMemoryType(*srcSpace),
                                     stringifyMemoryType(*dstSpace))
                           .str();
    ExecutorCallBuilder copyBuilder = {
        ctx, name, {}, llvm::to_vector(TypeRange(operands))};
    copyBuilder.create(rewriter, op.getLoc(),
                       op->template getParentOfType<ModuleOp>(), operands);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert `memref.alloc` to`executor.allocate`, which has the semantics of
/// an aligned allocation. Replace the result with the descriptor.
class CudaAllocToBuiltinCallConverter
    : public ConvertOpToExecutorPattern<cuda::AllocOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(cuda::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefType memrefType = op.getResult().getType();
    if (!memrefType.hasRank())
      return rewriter.notifyMatchFailure(op, "cannot convert unranked memref");
    std::optional<MemoryType> space = getMemorySpace(memrefType);
    if (!space)
      return failure();
    Type resultType = getTypeConverter()->convertType(memrefType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "could not convert memref type");

    FailureOr<MemRefAllocationInformation> info =
        getMemRefAllocationInformation(b, memrefType,
                                       adaptor.getDynamicSizes());
    if (failed(info))
      return rewriter.notifyMatchFailure(op, "failed to get allocation info");

    // Get the alignement requirement and memory space.
    Value alignment = b.create<executor::ConstantOp>(
        b.getI32IntegerAttr(op.getAlignment() ? *op.getAlignment() : 8));
    SmallVector<Value> callOperands;
    if (*space != MemoryType::host_pinned)
      callOperands.append({adaptor.getStream()});
    callOperands.push_back(info->sizeBytes);
    callOperands.push_back(alignment);

    SmallVector<Type> operandTypes = llvm::to_vector(TypeRange(callOperands));
    Type pointerType = PointerType::get(rewriter.getContext(), *space);
    std::string funcName =
        llvm::formatv("__cuda_alloc_{0}", stringifyMemoryType(*space));

    ModuleOp module = op->getParentOfType<ModuleOp>();
    ExecutorCallBuilder callBuilder = {
        rewriter.getContext(), funcName, {pointerType}, operandTypes};
    Value alloc =
        callBuilder.create(rewriter, op->getLoc(), module, callOperands)
            .getResult(0);

    auto memref = MemRefDescriptor::fromComponents(
        b, *getTypeConverter(), memrefType, alloc, alloc,
        createIndexConstant(b, 0), info->sizes, info->strides);
    rewriter.replaceOp(op, {memref});

    return success();
  }

private:
  /// Default data layout.
  DataLayout defaultLayout;
};

/// Convert `memref.dealloc` to `executor.deallocate`.
struct CudaDeallocToBuiltinCallConverter
    : public ConvertOpToExecutorPattern<cuda::DeallocOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(cuda::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefDescriptor desc(adaptor.getMemref(), op.getMemref().getType());

    std::optional<MemoryType> space = getMemorySpace(op.getMemref().getType());
    if (!space)
      return failure();

    assert(*space != MemoryType::host && "expected device-visible space");
    SmallVector<Value> callOperands = {adaptor.getStream(),
                                       desc.allocatedPtr(b)};
    SmallVector<Type> operandTypes = llvm::to_vector(TypeRange(callOperands));

    std::string funcName =
        llvm::formatv("__cuda_free_{0}", stringifyMemoryType(*space));
    ModuleOp module = op->getParentOfType<ModuleOp>();
    ExecutorCallBuilder callBuilder = {
        rewriter.getContext(), funcName, {}, operandTypes};
    rewriter.replaceOp(
        op, callBuilder.create(rewriter, op->getLoc(), module, callOperands));

    return success();
  }
};

/// A catch-all pattern that lowers `cuda` dialect operations to calls to
/// externally defined functions.
struct CudaOpToRuntimeBuiltinCallConverter : public ConvertToExecutorPattern {
  CudaOpToRuntimeBuiltinCallConverter(ExecutorTypeConverter &typeConverter,
                                      MLIRContext *context,
                                      PatternBenefit benefit = 1)
      : ConvertToExecutorPattern(typeConverter, MatchAnyOpTypeTag(), benefit,
                                 context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<cuda::CUDADialect>(op->getDialect()) ||
        isa<cuda::LaunchOp, cuda::CopyD2DOp, cuda::CopyH2DOp, cuda::CopyD2HOp,
            cuda::AllocOp, cuda::DeallocOp, cuda::GetActiveDeviceOp,
            cuda::GetGlobalStreamOp, cuda::CompiledModuleOp,
            cuda::GetFunctionOp>(op))
      return failure();
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    SmallVector<Type> newArgTypes = llvm::to_vector(TypeRange(operands));
    std::string funcName =
        "_" + getTypeConverter()->convertOpNameToBackendBuiltinFuncName(
                  op->getName().getStringRef());
    auto module = op->getParentOfType<ModuleOp>();

    ExecutorCallBuilder callBuilder = {rewriter.getContext(), funcName,
                                       newResultTypes, newArgTypes};

    rewriter.replaceOp(
        op, callBuilder.create(rewriter, op->getLoc(), module, operands));

    return success();
  }
};

/// Converts `cuda.get_function` to a `executor.global` that initializes the
/// CUDA cuFunction object from the CUDA cuModule object. The map provides the
/// mapping from compiled_module symbol name to cuModule object.
struct GetFunctionToCallConverter
    : public CUDAOpToExecutorCallLowering<cuda::GetFunctionOp> {

  GetFunctionToCallConverter(
      const llvm::SmallDenseMap<StringAttr, executor::GlobalOp> &map,
      ExecutorTypeConverter &typeConverter, MLIRContext *ctx)
      : CUDAOpToExecutorCallLowering(typeConverter, ctx),
        compiledModuleToGlobalMap(map) {}

  LogicalResult
  matchAndRewrite(cuda::GetFunctionOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto pointerType =
        PointerType::get(rewriter.getContext(), MemoryType::host);
    auto module = op->getParentOfType<ModuleOp>();
    executor::GlobalOp cuModuleGlobalOp =
        compiledModuleToGlobalMap.lookup(op.getModuleAttr().getLeafReference());

    std::string funcGlobalSymName = llvm::formatv(
        "{0}_{1}_cuFunc", cuModuleGlobalOp.getSymName(), op.getKernelName());

    if (auto global =
            module.lookupSymbol<executor::GlobalOp>(funcGlobalSymName)) {
      rewriter.replaceOpWithNewOp<executor::GetGlobalOp>(
          op, pointerType, FlatSymbolRefAttr::get(global));
      return success();
    }

    executor::GlobalOp funcGlobalOp = [&]() {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(cuModuleGlobalOp);

      return rewriter.create<executor::GlobalOp>(
          op.getLoc(), funcGlobalSymName, pointerType,
          [&](OpBuilder &nested, Location loc) {
            Value cuModulePtr = nested.create<GetGlobalOp>(
                loc, pointerType, FlatSymbolRefAttr::get(cuModuleGlobalOp));
            Value nameLiteral = rewriter.create<executor::StrLiteralOp>(
                op.getLoc(), op.getKernelName());
            Value funcPtr =
                cudaGetFuncBuilder
                    .create(nested, loc, module, {cuModulePtr, nameLiteral})
                    .getResult(0);
            nested.create<executor::ReturnOp>(loc, funcPtr);
          },
          true);
    }();

    rewriter.replaceOpWithNewOp<executor::GetGlobalOp>(
        op, pointerType, FlatSymbolRefAttr::get(funcGlobalOp));

    return success();
  }

private:
  const llvm::SmallDenseMap<StringAttr, executor::GlobalOp>
      &compiledModuleToGlobalMap;
};

} // namespace

/// Replaces a `cuda.compiled_module` operation with a `executor.global`
/// containing a pointer to the CUDA driver cuModule object. Creates a
/// `executor.data_segment` object for the PTX data.
static executor::GlobalOp lowerCompiledModuleOp(RewriterBase &rewriter,
                                                Location loc,
                                                ModuleOp parentModule,
                                                StringRef cudaModuleName,
                                                ElementsAttr ptxData) {
  std::string ptxDataName = (cudaModuleName + "_ptx_data").str();
  std::string cuModuleGlobalName = (cudaModuleName + "_cuModule").str();
  auto resourceOp =
      rewriter.create<executor::DataSegmentOp>(loc, ptxDataName, ptxData,
                                               /*constant=*/true,
                                               /*uninitialized=*/false,
                                               /*alignment=*/IntegerAttr{});
  Type pointerType = PointerType::get(rewriter.getContext(), MemoryType::host);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  return rewriter.create<executor::GlobalOp>(
      loc, cuModuleGlobalName, pointerType,
      [&](OpBuilder &nested, Location loc) {
        Value cudaDevice = nested.create<cuda::GetActiveDeviceOp>(loc);
        Value ptx = nested.create<ConstantResourceLoadOp>(
            loc, FlatSymbolRefAttr::get(resourceOp));
        assert(ptxData.getElementType().isInteger(8) &&
               "expected i8/byte element type");
        Value numBytes = nested.create<ConstantOp>(
            loc, nested.getI64IntegerAttr(ptxData.getNumElements()));

        ExecutorCallBuilder loadModuleBuilder = {
            rewriter.getContext(),
            "__cuda_load_module",
            pointerType,
            {i32Type, pointerType, i64Type}};

        Value cuModulePtr =
            loadModuleBuilder
                .create(nested, loc, parentModule, {cudaDevice, ptx, numBytes})
                .getResult(0);
        nested.create<executor::ReturnOp>(loc, cuModulePtr);
      },
      true);
}

/// Replaces all `cuda.compiled_module` operations with `executor.global`
/// containing a pointer to the CUDA driver cuModule object.
static LogicalResult lowerCompiledModuleOps(
    RewriterBase &rewriter, ModuleOp op,
    llvm::SmallDenseMap<StringAttr, executor::GlobalOp> &map) {
  for (auto compiledModuleOp :
       llvm::make_early_inc_range(op.getOps<cuda::CompiledModuleOp>())) {
    rewriter.setInsertionPoint(compiledModuleOp);
    executor::GlobalOp cuModuleGlobal = lowerCompiledModuleOp(
        rewriter, compiledModuleOp.getLoc(), op, compiledModuleOp.getSymName(),
        compiledModuleOp.getValue());
    map[compiledModuleOp.getSymNameAttr()] = cuModuleGlobal;
    rewriter.eraseOp(compiledModuleOp);
  }
  return success();
}

void mlir::populateCUDAToExecutorTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    if (isa<cuda::StreamType, cuda::EventType, cuda::ModuleType,
            cuda::FunctionType, cuda::BlasHandleType,
            cuda::BlasGemmAlgorithmType>(t))
      return PointerType::get(t.getContext(), MemoryType::host);
    return {};
  });
}

namespace {
class CUDAToExecutorPass
    : public mlir::impl::ConvertCUDAToExecutorPassBase<CUDAToExecutorPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp rootOp = getOperation();
    MLIRContext *ctx = &getContext();
    LowerToExecutorOptions opts;
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    FailureOr<DataLayout> dataLayout =
        executor::setDataLayoutSpec(rootOp, indexBitwidth, 64);
    if (failed(dataLayout)) {
      emitError(rootOp->getLoc())
          << "failed to set DataLayout; op has DLTI spec that is "
             "inconsistent with provided options";
      return signalPassFailure();
    }
    ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));
    populateCUDAToExecutorTypeConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<cuda::CUDADialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<executor::ExecutorDialect>();
    auto isInNestedSymbolTable = [rootOp](Operation *op) {
      auto symbolTableParent = op->getParentWithTrait<OpTrait::SymbolTable>();
      return symbolTableParent != rootOp && op != rootOp;
    };

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) -> std::optional<bool> {
          if (isInNestedSymbolTable(op))
            return true;
          return {};
        });

    // For each CompiledModuleOp, convert them into globals for the cuModule,
    // cuFunction, and kernel binary data. We keep a map from symbol names of
    // `cuda.compiled_module` ops to the `executor.global` that replaced it.
    // The `cuda.compiled_modules` are erased here.
    IRRewriter rewriter(ctx);
    llvm::SmallDenseMap<StringAttr, executor::GlobalOp>
        compiledModuleToGlobalMap;
    if (failed(lowerCompiledModuleOps(rewriter, rootOp,
                                      compiledModuleToGlobalMap))) {
      emitError(rootOp.getLoc())
          << "failed to lower cuda.compiled_module ops in " << getArgument();
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<
        CudaAllocToBuiltinCallConverter,
        CudaBlasHeuristicAlgoSelectionOpConverter,
        CudaBlasRunGemmOpConverter,
        CudaDeallocToBuiltinCallConverter,
        CudaGetActiveDeviceConverter,
        CudaGetDeviceCountConverter,
        CudaGetGlobalStreamConverter,
        CudaMemCopyOpToBuiltinCallConverter<cuda::CopyD2DOp>,
        CudaMemCopyOpToBuiltinCallConverter<cuda::CopyD2HOp>,
        CudaMemCopyOpToBuiltinCallConverter<cuda::CopyH2DOp>,
        CudaOpToRuntimeBuiltinCallConverter,
        CudaSetActiveDeviceConverter,
        LowerCudaLaunchKernelToCall
      >(typeConverter, ctx);
    patterns.add<
        GetFunctionToCallConverter
      >(compiledModuleToGlobalMap,  typeConverter, ctx);
    // clang-format on

    if (failed(applyPartialConversion(rootOp, target, std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply conversion patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
