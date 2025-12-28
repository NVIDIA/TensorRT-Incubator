//===- CUDAToLLVM.cpp -----------------------------------------------------===//
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
/// Implementation of the `convert-cuda-to-llvm` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/CUDAToLLVM/CUDAToLLVM.h"
#include "mlir-tensorrt/Conversion/LLVMCommon/LLVMCommon.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Conversion/PlanToLLVM/PlanToLLVM.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCUDATOLLVMPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

struct CUDAExternalCallBuilders {
  CUDAExternalCallBuilders(MLIRContext *ctx) : ctx(ctx) {}

  MLIRContext *ctx;
  Type i8Type{IntegerType::get(ctx, 8)};
  Type i32Type{IntegerType::get(ctx, 32)};
  Type i64Type{IntegerType::get(ctx, 64)};
  Type llvmPtrType{LLVM::LLVMPointerType::get(ctx)};
  Type llvmVoidType{LLVM::LLVMVoidType::get(ctx)};

  LLVMOpaqueCallBuilder streamCreateBuilder = {
      "mtrt_cuda_stream_create", llvmPtrType, {/*device=*/i32Type}};
  LLVMOpaqueCallBuilder streamDestroyBuilder = {
      "mtrt_cuda_stream_destroy", llvmVoidType, {llvmPtrType}};
  LLVMOpaqueCallBuilder streamSyncBuilder = {
      "mtrt_cuda_stream_sync", llvmVoidType, {llvmPtrType}};

  LLVMOpaqueCallBuilder cuModuleGetFuncBuilder = {
      "mtrt_cuda_module_get_function",
      llvmPtrType,
      {/*cumodule*/ llvmPtrType, /*function literal ptr*/ llvmPtrType,
       /*name size*/ i64Type}};

  LLVMOpaqueCallBuilder cuModuleUnloadBuilder = {
      "mtrt_cuda_module_unload", llvmVoidType, {/*cumodule*/ llvmPtrType}};

  LLVMOpaqueCallBuilder cuModuleLoadFromPtxBuilder = {
      "mtrt_cuda_module_load_from_ptx",
      llvmPtrType,
      {/*ptx data ptr*/ llvmPtrType,
       /*ptx data size*/ i64Type}};

  LLVMOpaqueCallBuilder cuModuleLoadFromPtxFileBuilder = {
      "mtrt_cuda_module_load_from_ptx_file",
      llvmPtrType,
      {/*filename ptr*/ llvmPtrType,
       /*filename size*/ i64Type}};

  LLVMOpaqueCallBuilder launchKernelCallBuilder = {
      "mtrt_cuda_launch_kernel",
      llvmVoidType,
      {/*cuFunction*/ llvmPtrType,
       /*grid*/ i32Type, i32Type, i32Type,
       /*block*/ i32Type, i32Type, i32Type,
       /*dsmem*/ i32Type,
       /*stream*/ llvmPtrType,
       /*arg ptr array*/ llvmPtrType}};

  LLVMOpaqueCallBuilder cudaAllocAsyncBuilder = {"mtrt_cuda_alloc_async",
                                                 llvmPtrType,
                                                 {/*stream*/ llvmPtrType,
                                                  /*bytes*/ i64Type,
                                                  /*alignment*/ i32Type,
                                                  /*isHostPinned*/ i8Type,
                                                  /*isManaged*/ i8Type}};

  LLVMOpaqueCallBuilder cudaFreeBuilder = {
      "mtrt_cuda_free",
      llvmVoidType,
      {/*stream*/ llvmPtrType, /*data*/ llvmPtrType, /*isHostPinned*/ i8Type,
       /*isManaged*/ i8Type}};

  LLVMOpaqueCallBuilder cudaMemcpyAsyncBuilder = {"mtrt_cuda_memcpy_async",
                                                  llvmVoidType,
                                                  {/*stream*/ llvmPtrType,
                                                   /*src*/ llvmPtrType,
                                                   /*dst*/ llvmPtrType,
                                                   /*bytes*/ i64Type}};

  LLVMOpaqueCallBuilder cudaMemcpyStridedAsyncBuilder = {
      "mtrt_cuda_memcpy_strided_async",
      llvmVoidType,
      {/*stream*/ llvmPtrType,
       /*src rank*/ i64Type,
       /*src descriptor*/ llvmPtrType,
       /*dsta rank*/ i64Type,
       /*dst descriptor*/ llvmPtrType}};

  LLVMOpaqueCallBuilder cudaGetActiveDeviceBuilder = {
      "mtrt_cuda_get_active_device", i32Type, {}};
  LLVMOpaqueCallBuilder cudaSetActiveDeviceBuilder = {
      "mtrt_cuda_set_active_device", llvmVoidType, {/*device*/ i32Type}};
  LLVMOpaqueCallBuilder cudaGetDeviceCountBuilder = {
      "mtrt_cuda_get_device_count", i32Type, {}};
  LLVMOpaqueCallBuilder cudaGetDeviceBuilder = {
      "mtrt_cuda_get_device", i32Type, {/*device*/ i32Type}};
};

//===----------------------------------------------------------------------===//
// MemRef/Layout Helpers
//===----------------------------------------------------------------------===//

/// Returns `true` if a memref with shape `shape` and `strides` represents a
/// contiguous array of memory. This is equivalent to checking whether some
/// subview is contiguous. The idea here is that the shape and laout should have
/// a canonical row-major layout when removing the unit extents. For example,
/// `memref<8x1x4, strided<[4, 32, 1], offset: ?>>` should be contiguous since
/// we can ignore the middle unit extent dimension.
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

//===----------------------------------------------------------------------===//
// ConvertCUDAOpToLLVMPattern
//===----------------------------------------------------------------------===//

template <typename T>
struct ConvertCUDAOpToLLVMPattern : public ConvertOpToLLVMPattern<T> {
  ConvertCUDAOpToLLVMPattern(const LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<T>(typeConverter, benefit) {}

  std::optional<plan::MemorySpace> getMemorySpace(MemRefType type) const {
    auto srcMemoryTypeAttr =
        dyn_cast_or_null<plan::MemorySpaceAttr>(type.getMemorySpace());
    if (!srcMemoryTypeAttr)
      return plan::MemorySpace::host;
    return srcMemoryTypeAttr.getValue();
  }

  // Returns the number of elements in a packed (identity layout) memref.
  Value getNumElements(RewriterBase &rewriter, Location loc, MemRefType type,
                       MemRefDescriptor desc) const {
    Type indexType = ConvertToLLVMPattern::getIndexType();
    return type.hasStaticShape()
               ? ConvertToLLVMPattern::createIndexAttrConstant(
                     rewriter, loc, indexType, type.getNumElements())
               : rewriter.create<LLVM::MulOp>(loc,
                                              desc.stride(rewriter, loc, 0),
                                              desc.size(rewriter, loc, 0));
  }

protected:
  MLIRContext *ctx{this->getContext()};
  Type i32Type{IntegerType::get(ctx, 32)};
  Type i64Type{IntegerType::get(ctx, 64)};
  Type llvmPtrType{LLVM::LLVMPointerType::get(ctx)};
  Type llvmVoidType{LLVM::LLVMVoidType::get(ctx)};
  CUDAExternalCallBuilders cudaFuncs{ctx};
};

template <typename Derived, typename OpType>
struct SimpleCudaOpToLLVMCallLowering : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpType op,
                  typename ConvertOpToLLVMPattern<OpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto callOp = static_cast<const Derived *>(this)->callBuilder.create(
        op.getLoc(), rewriter, adaptor.getOperands());
    rewriter.replaceOp(op, callOp);
    return success();
  }

  CUDAExternalCallBuilders callBuilders{this->getContext()};
};

namespace {

struct CudaGetActiveDeviceConverter final
    : public SimpleCudaOpToLLVMCallLowering<CudaGetActiveDeviceConverter,
                                            cuda::GetActiveDeviceOp> {
  using SimpleCudaOpToLLVMCallLowering::SimpleCudaOpToLLVMCallLowering;
  LLVMOpaqueCallBuilder &callBuilder =
      this->callBuilders.cudaGetActiveDeviceBuilder;
};

struct CudaSetActiveDeviceConverter final
    : public SimpleCudaOpToLLVMCallLowering<CudaSetActiveDeviceConverter,
                                            cuda::SetActiveDeviceOp> {
  using SimpleCudaOpToLLVMCallLowering::SimpleCudaOpToLLVMCallLowering;
  LLVMOpaqueCallBuilder &callBuilder =
      this->callBuilders.cudaSetActiveDeviceBuilder;
};

struct CudaGetDeviceCountConverter final
    : public SimpleCudaOpToLLVMCallLowering<CudaGetDeviceCountConverter,
                                            cuda::DeviceCountOp> {
  using SimpleCudaOpToLLVMCallLowering::SimpleCudaOpToLLVMCallLowering;
  LLVMOpaqueCallBuilder &callBuilder =
      this->callBuilders.cudaGetDeviceCountBuilder;
};

struct CudaGetDeviceConverter final
    : public SimpleCudaOpToLLVMCallLowering<CudaGetDeviceConverter,
                                            cuda::GetDeviceOp> {
  using SimpleCudaOpToLLVMCallLowering::SimpleCudaOpToLLVMCallLowering;
  LLVMOpaqueCallBuilder &callBuilder = this->callBuilders.cudaGetDeviceBuilder;
};

struct CudaStreamSyncConverter final
    : public SimpleCudaOpToLLVMCallLowering<CudaStreamSyncConverter,
                                            cuda::StreamSyncOp> {
  using SimpleCudaOpToLLVMCallLowering::SimpleCudaOpToLLVMCallLowering;
  LLVMOpaqueCallBuilder &callBuilder = this->callBuilders.streamSyncBuilder;
};

struct CudaAllocConverter final
    : public ConvertCUDAOpToLLVMPattern<cuda::AllocOp> {
  using ConvertCUDAOpToLLVMPattern::ConvertCUDAOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(cuda::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memrefType = op.getResult().getType();
    if (!memrefType.hasRank())
      return rewriter.notifyMatchFailure(op, "cannot convert unranked memref");
    if (!isConvertibleAndHasIdentityMaps(memrefType))
      return failure();

    std::optional<plan::MemorySpace> space = getMemorySpace(memrefType);
    if (!space ||
        !llvm::is_contained(
            ArrayRef<plan::MemorySpace>{plan::MemorySpace::device,
                                        plan::MemorySpace::host_pinned,
                                        plan::MemorySpace::unified},
            *space))
      return failure();

    Location loc = op.getLoc();
    Value isManaged = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(*space == plan::MemorySpace::unified));
    Value isPinned = rewriter.create<LLVM::ConstantOp>(
        loc,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::host_pinned));

    Type resultType = getTypeConverter()->convertType(memrefType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "could not convert memref type");

    SmallVector<Value> shape;
    SmallVector<Value> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memrefType, adaptor.getDynamicSizes(),
                             rewriter, shape, strides, sizeBytes);

    Value stream = adaptor.getStream()
                       ? adaptor.getStream()
                       : rewriter.create<LLVM::ZeroOp>(loc, llvmPtrType);

    Value alignment = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(op.getAlignment() ? *op.getAlignment()
                                                          : 16));

    Value allocatedPtr =
        cudaFuncs.cudaAllocAsyncBuilder
            .create(loc, rewriter,
                    {stream, sizeBytes, alignment, isPinned, isManaged})
            .getResult();

    // Create the MemRef descriptor.
    MemRefDescriptor memRefDescriptor = this->createMemRefDescriptor(
        loc, memrefType, allocatedPtr, allocatedPtr, shape, strides, rewriter);

    rewriter.replaceOp(op, Value(memRefDescriptor));
    return success();
  }
};

struct CudaDeallocConverter
    : public ConvertCUDAOpToLLVMPattern<cuda::DeallocOp> {
  using ConvertCUDAOpToLLVMPattern::ConvertCUDAOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(cuda::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefDescriptor descriptor(adaptor.getMemref());
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
    Value isPinned = rewriter.create<LLVM::ConstantOp>(
        loc,
        rewriter.getI8IntegerAttr(*space == plan::MemorySpace::host_pinned));
    Value isManaged = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(*space == plan::MemorySpace::unified));
    cudaFuncs.cudaFreeBuilder.create(loc, rewriter,
                                     {adaptor.getStream(),
                                      descriptor.allocatedPtr(rewriter, loc),
                                      isPinned, isManaged});
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename CpyOpType>
struct CudaCopyConverter : public ConvertCUDAOpToLLVMPattern<CpyOpType> {
  using ConvertCUDAOpToLLVMPattern<CpyOpType>::ConvertCUDAOpToLLVMPattern;
  LogicalResult matchAndRewrite(
      CpyOpType op,
      typename ConvertCUDAOpToLLVMPattern<CpyOpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MemRefType srcType = op.getSource().getType();
    MemRefType dstType = op.getTarget().getType();
    std::optional<plan::MemorySpace> srcSpace = this->getMemorySpace(srcType);
    std::optional<plan::MemorySpace> dstSpace = this->getMemorySpace(dstType);
    if (!srcSpace || !dstSpace)
      return failure();

    MemRefDescriptor src(adaptor.getSource());
    MemRefDescriptor dest(adaptor.getTarget());
    Location loc = op.getLoc();

    if (!isCopyStrided(srcType, dstType)) {
      Value numElements = this->getNumElements(rewriter, loc, srcType, src);
      Type elementPtrType = src.getElementPtrType();
      Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, elementPtrType);
      Value gepPtr = rewriter.create<LLVM::GEPOp>(
          loc, elementPtrType,
          this->typeConverter->convertType(srcType.getElementType()), nullPtr,
          numElements);
      Value sizeBytes =
          rewriter.create<LLVM::PtrToIntOp>(loc, this->getIndexType(), gepPtr);
      this->cudaFuncs.cudaMemcpyAsyncBuilder.create(
          loc, rewriter,
          {adaptor.getStream(),
           src.bufferPtr(rewriter, loc, *this->getTypeConverter(), srcType),
           dest.bufferPtr(rewriter, loc, *this->getTypeConverter(), dstType),
           sizeBytes});
      rewriter.eraseOp(op);
      return success();
    }

    // Put descriptors on the stack.
    UnrankedMemRefDescriptor srcUnranked = getUnrankedLLVMMemRefDescriptor(
        rewriter, loc, *this->getTypeConverter(), src, srcType);
    UnrankedMemRefDescriptor destUnranked = getUnrankedLLVMMemRefDescriptor(
        rewriter, loc, *this->getTypeConverter(), dest, dstType);

    this->cudaFuncs.cudaMemcpyStridedAsyncBuilder.create(
        loc, rewriter,
        {adaptor.getStream(), srcUnranked.rank(rewriter, loc),
         srcUnranked.memRefDescPtr(rewriter, loc),
         destUnranked.rank(rewriter, loc),
         destUnranked.memRefDescPtr(rewriter, loc)});
    rewriter.eraseOp(op);
    return success();
  }
};

struct CudaLaunchOpToLLVMCallConverter
    : public ConvertCUDAOpToLLVMPattern<cuda::LaunchOp> {
  using ConvertCUDAOpToLLVMPattern::ConvertCUDAOpToLLVMPattern;

  MLIRContext *ctx = &this->getTypeConverter()->getContext();
  Type i32Type = mlir::IntegerType::get(ctx, 32);
  Type i64Type = mlir::IntegerType::get(ctx, 64);
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);

  LogicalResult
  matchAndRewrite(cuda::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                                  rewriter.getIndexAttr(1));

    SmallVector<Value> storagePtrs;
    SmallVector<Value> promotedArgs = this->getTypeConverter()->promoteOperands(
        loc, op.getArgs(), adaptor.getArgs(), rewriter,
        /*useBarePtrCallConv=*/true);
    for (Value toStoreVal : promotedArgs) {
      Value valuePtr = rewriter.create<LLVM::AllocaOp>(
          loc, llvmPtrType, toStoreVal.getType(), one);
      rewriter.create<LLVM::StoreOp>(loc, toStoreVal, valuePtr);
      storagePtrs.push_back(valuePtr);
    }

    // Create and populate the array-of-pointers that is required by the
    // launch config.
    auto operandPtrStorageType =
        LLVM::LLVMArrayType::get(llvmPtrType, storagePtrs.size());
    auto argPtrsPtr = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, operandPtrStorageType, one);
    for (auto [idx, value] : llvm::enumerate(storagePtrs)) {
      auto gepOp = rewriter.create<LLVM::GEPOp>(
          loc, llvmPtrType, operandPtrStorageType, argPtrsPtr,
          ArrayRef<LLVM::GEPArg>{0, idx});
      rewriter.create<LLVM::StoreOp>(loc, value, gepOp);
    }

    cudaFuncs.launchKernelCallBuilder.create(
        loc, rewriter,
        /*functionInputs*/
        {adaptor.getFunc(), adaptor.getGridX(), adaptor.getGridY(),
         adaptor.getGridZ(), adaptor.getBlockX(), adaptor.getBlockY(),
         adaptor.getBlockZ(), adaptor.getDynamicSharedMem(),
         adaptor.getStream(), argPtrsPtr});
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

/// Lower a `cuda.get_function` call by inserting a LLVM global, populating that
/// global with the loaded function from the CUmodule in the ctor function, and
/// then replacing the `cuda.get_function` with load of the CUfunction global.
///
/// Once the CUfunction global is created, it will be inserted in to the cache.
/// Future calls referencing the same function module and kernel name will then
/// just pull from the cache instead of creating a new global.
static LogicalResult convertCudaGetFunction(
    RewriterBase &rewriter, cuda::GetFunctionOp op,
    LLVM::GlobalOp cumoduleGlobal,
    llvm::SmallDenseMap<StringAttr, LLVM::GlobalOp> &cufuncCache,
    SymbolTable &symbolTable, const CUDAExternalCallBuilders &cudaFuncs,
    int globalPriority) {

  LLVM::GlobalOp cufuncGlobal = cufuncCache.lookup(op.getKernelNameAttr());
  MLIRContext *ctx = rewriter.getContext();
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto replace = [&](LLVM::GlobalOp cufuncGlobal) {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, cuda::FunctionType::get(ctx),
        rewriter
            .create<LLVM::LoadOp>(
                loc, llvmPtrType,
                rewriter.create<LLVM::AddressOfOp>(loc, cufuncGlobal))
            .getResult());
    return success();
  };
  if (cufuncGlobal)
    return replace(cufuncGlobal);

  // Construct the cumodule global.
  cufuncGlobal = insertLLVMGlobal(
      rewriter, loc,
      (cumoduleGlobal.getName() + "_" + op.getKernelName()).str(),
      /*constant=*/false, llvmPtrType, LLVM::Linkage::Internal, Attribute{},
      &symbolTable);

  // Populate the ctor.
  insertLLVMCtorFunction(
      rewriter, loc, symbolTable, (cufuncGlobal.getName() + "_init").str(),
      globalPriority, [&](OpBuilder &rewriter, Location loc) {
        Value kernelNameLiteral = insertLLVMStringLiteral(
            rewriter, loc, op.getKernelName(),
            (op.getKernelName() + "_name").str(), &symbolTable);
        Value kernelNameSize = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(op.getKernelName().size()));
        Value cumoduleAddr =
            rewriter.create<LLVM::AddressOfOp>(loc, cumoduleGlobal);
        Value cumodule =
            rewriter.create<LLVM::LoadOp>(loc, llvmPtrType, cumoduleAddr);
        Value cufunc =
            cudaFuncs.cuModuleGetFuncBuilder
                .create(loc, rewriter,
                        {cumodule, kernelNameLiteral, kernelNameSize})
                .getResult();
        rewriter.create<LLVM::StoreOp>(
            loc, cufunc, rewriter.create<LLVM::AddressOfOp>(loc, cufuncGlobal));
      });

  // Update the cache.
  cufuncCache[op.getKernelNameAttr()] = cufuncGlobal;
  return replace(cufuncGlobal);
}

/// This function removes `cuda.compiled_module` operations and creates two
/// `llvm.mlir.global` vars for the module name and module data.
/// It inserts llvm.calls to cuModuleLoad and cuModuleGetFunction into the
/// constructor function and inserts a llvm.call to cuModuleUnload into the
/// destructor function. It also creates a lookup table (param: map) from module
/// name strings to the addresses of the module data.
static LogicalResult lowerCuModuleOps(RewriterBase &rewriter, ModuleOp module,
                                      SymbolTable &symbolTable,
                                      SymbolUserMap &userMap,
                                      StringRef artifactsDir) {
  MLIRContext *ctx = rewriter.getContext();
  Type llvmPtrType = LLVM::LLVMPointerType::get(ctx);
  CUDAExternalCallBuilders cudaFuncs{ctx};

  for (auto compiledModuleOp :
       llvm::make_early_inc_range(module.getOps<cuda::CompiledModuleOp>())) {
    Location loc = compiledModuleOp->getLoc();

    // Construct the cumodule global.
    LLVM::GlobalOp cumoduleGlobal =
        insertLLVMGlobal(rewriter, loc, compiledModuleOp.getName(),
                         /*constant=*/false, llvmPtrType,
                         LLVM::Linkage::Internal, Attribute{}, &symbolTable);

    // Determine how the module will be loaded:
    // - If a file is provided, always load from file at runtime (optionally
    //   staging into `artifactsDir`).
    // - Otherwise, use embedded bytes, optionally serializing to a file.
    LLVM::GlobalOp ptxGlobal{};
    std::string filename;

    if (compiledModuleOp.hasFileReference()) {
      llvm::SmallString<256> srcPath(compiledModuleOp.getFilePath());
      if (srcPath.empty())
        return compiledModuleOp.emitOpError()
               << "cuda.compiled_module 'file' attribute must not be empty";

      llvm::SmallString<256> stagedRelPath;
      if (llvm::sys::path::is_absolute(srcPath))
        stagedRelPath = llvm::sys::path::filename(srcPath);
      else
        stagedRelPath = srcPath;

      filename = stagedRelPath.str().str();

      if (!artifactsDir.empty()) {
        if (!llvm::sys::fs::is_directory(artifactsDir))
          return module.emitError() << "artifacts-directory does not exist";

        llvm::SmallString<256> stagedAbsPath(artifactsDir);
        llvm::sys::path::append(stagedAbsPath, stagedRelPath);
        llvm::sys::path::remove_dots(stagedAbsPath, /*remove_dot_dot=*/true);

        if (std::error_code ec = llvm::sys::fs::create_directories(
                llvm::sys::path::parent_path(stagedAbsPath)))
          return compiledModuleOp.emitOpError()
                 << "failed to create directory for staged PTX file: "
                 << ec.message();

        if (!llvm::sys::fs::exists(stagedAbsPath)) {
          if (std::error_code ec =
                  llvm::sys::fs::copy_file(srcPath, stagedAbsPath))
            return compiledModuleOp.emitOpError()
                   << "failed to stage PTX file '" << srcPath << "' into '"
                   << stagedAbsPath << "': " << ec.message();
        }
      } else {
        // No artifacts dir: use provided path directly.
        filename = srcPath.str().str();
      }
    } else if (auto ptxData = dyn_cast_or_null<ElementsAttr>(
                   compiledModuleOp.getValueAttr())) {
      if (artifactsDir.empty()) {
        ptxGlobal = lookupOrInsertGlobal(
            rewriter, loc, (compiledModuleOp.getName() + "_ptx").str(),
            /*constant=*/true,
            LLVM::LLVMArrayType::get(rewriter.getI8Type(),
                                     ptxData.getNumElements()),
            LLVM::Linkage::Internal, compiledModuleOp.getValueAttr(),
            &symbolTable);
      } else {
        if (!llvm::sys::fs::is_directory(artifactsDir))
          return module.emitError() << "artifacts-directory does not exist";

        filename = (compiledModuleOp.getName() + ".ptx").str();
        FailureOr<std::unique_ptr<llvm::ToolOutputFile>> outputFile =
            serializeElementsAttrToFile(loc, compiledModuleOp.getValueAttr(),
                                        artifactsDir, filename);
        if (failed(outputFile))
          return failure();
        (*outputFile)->keep();
      }
    } else {
      return compiledModuleOp.emitOpError()
             << "cuda.compiled_module must specify either 'value' or 'file'";
    }

    // Insert initialization for the cumodule/cufunc into  the constructor
    // function.
    int globalPriority = 0;
    insertLLVMCtorFunction(
        rewriter, loc, symbolTable,
        (compiledModuleOp.getName() + "_init").str(), globalPriority,
        [&](OpBuilder &rewriter, Location loc) {
          Value cuModule;
          if (ptxGlobal) {
            Value ptxStr = rewriter.create<LLVM::AddressOfOp>(loc, ptxGlobal);
            Value ptxSize = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(
                         compiledModuleOp.getValueAttr().size()));
            cuModule = cudaFuncs.cuModuleLoadFromPtxBuilder
                           .create(loc, rewriter, {ptxStr, ptxSize})
                           .getResult();
          } else {
            Value nameStr = insertLLVMStringLiteral(
                rewriter, loc, filename,
                (compiledModuleOp.getName() + "_filename").str());
            Value nameSize = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(filename.size()));
            cuModule = cudaFuncs.cuModuleLoadFromPtxFileBuilder
                           .create(loc, rewriter, {nameStr, nameSize})
                           .getResult();
          }
          Value cumoduleGlobalAddr =
              rewriter.create<LLVM::AddressOfOp>(loc, cumoduleGlobal);
          rewriter.create<LLVM::StoreOp>(loc, cuModule, cumoduleGlobalAddr);
        });

    // In the dtor, unload the module.
    insertLLVMDtorFunction(
        rewriter, loc, symbolTable,
        (compiledModuleOp.getName() + "_deinit").str(), globalPriority,
        [&](OpBuilder &rewriter, Location loc) {
          Value cumodule = rewriter.create<LLVM::LoadOp>(
              loc, llvmPtrType,
              rewriter.create<LLVM::AddressOfOp>(loc, cumoduleGlobal));
          cudaFuncs.cuModuleUnloadBuilder.create(loc, rewriter, cumodule);
        });

    llvm::SmallDenseMap<StringAttr, LLVM::GlobalOp> cufuncCache;

    // Check all users of this symbol.
    for (Operation *user : userMap.getUsers(compiledModuleOp)) {
      // The only users should be `cuda.get_function` ops. Any other user is an
      // error because will end up with a dangling symbol reference after the op
      // is erased.
      if (!isa<cuda::GetFunctionOp>(user)) {
        return emitError(compiledModuleOp->getLoc(),
                         "unexpected user of 'cuda.compiled_module'")
                   .attachNote(user->getLoc())
               << "see user here";
      }

      if (failed(convertCudaGetFunction(
              rewriter, cast<cuda::GetFunctionOp>(user), cumoduleGlobal,
              cufuncCache, symbolTable, cudaFuncs, globalPriority + 1)))
        return failure();
    }

    // All users are now guarunteed erased, so we can erase the
    // 'cuda.compiled_module' op.
    rewriter.eraseOp(compiledModuleOp);
  }
  return success();
}

/// Retrieve or populate a `llvm.global` containing a CUDA stream keyed by the
/// `index` in `cudaFuncs`.
static LLVM::GlobalOp
insertOrCreateStreamGlobal(RewriterBase &rewriter, Location loc,
                           SymbolTable &symbolTable, unsigned index,
                           llvm::SmallDenseMap<unsigned, LLVM::GlobalOp> &cache,
                           const CUDAExternalCallBuilders &cudaFuncs) {
  if (LLVM::GlobalOp op = cache.lookup(index))
    return op;

  LLVM::GlobalOp global =
      insertLLVMGlobal(rewriter, loc, llvm::formatv("stream_{0}", index).str(),
                       /*constant=*/false, cudaFuncs.llvmPtrType,
                       LLVM::Linkage::Internal, Attribute{}, &symbolTable);
  cache[index] = global;

  // Populate ctor.
  insertLLVMCtorFunction(
      rewriter, loc, symbolTable, (global.getName() + "_init").str(), 0,
      [&](OpBuilder &rewriter, Location loc) {
        Value device =
            cudaFuncs.cudaGetActiveDeviceBuilder.create(loc, rewriter, {})
                .getResult();
        Value stream =
            cudaFuncs.streamCreateBuilder.create(loc, rewriter, {device})
                .getResult();
        rewriter.create<LLVM::StoreOp>(
            loc, stream, rewriter.create<LLVM::AddressOfOp>(loc, global));
      });

  // Populate dtor.
  insertLLVMDtorFunction(
      rewriter, loc, symbolTable, (global.getName() + "_deinit").str(), 0,
      [&](OpBuilder &rewriter, Location loc) {
        Value stream = rewriter.create<LLVM::LoadOp>(
            loc, cudaFuncs.llvmPtrType,
            rewriter.create<LLVM::AddressOfOp>(loc, global));
        cudaFuncs.streamDestroyBuilder.create(loc, rewriter, {stream});
      });

  return global;
}

/// Lower all `cuda.global_stream` operations.
static LogicalResult lowerGetGlobalStreamOps(RewriterBase &rewriter,
                                             ModuleOp module,
                                             SymbolTable &symbolTable) {
  MLIRContext *ctx = rewriter.getContext();
  CUDAExternalCallBuilders cudaFuncs{ctx};
  llvm::SmallDenseMap<unsigned, LLVM::GlobalOp> streamMap;
  SmallVector<cuda::GetGlobalStreamOp> getStreamOps;
  module->walk([&](cuda::GetGlobalStreamOp op) { getStreamOps.push_back(op); });
  for (cuda::GetGlobalStreamOp op : getStreamOps) {

    if (!matchPattern(op.getDevice(), m_Op<cuda::GetActiveDeviceOp>())) {
      return emitError(op.getLoc(), "currently only 'cuda.get_active_device' "
                                    "is supported as device for global streams")
                 .attachNote(op.getDevice().getLoc())
             << "see device here";
    }

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    LLVM::GlobalOp streamGlobal = insertOrCreateStreamGlobal(
        rewriter, loc, symbolTable, op.getIndex(), streamMap, cudaFuncs);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, cuda::StreamType::get(ctx),
        rewriter
            .create<LLVM::LoadOp>(
                loc, cudaFuncs.llvmPtrType,
                rewriter.create<LLVM::AddressOfOp>(loc, streamGlobal))
            .getResult());
  }
  return success();
}

/// Populate type conversions for CUDA dialect types to LLVM types.
void mlir::populateCUDAToLLVMTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    if (isa<cuda::EventType, cuda::ModuleType, cuda::FunctionType,
            cuda::StreamType>(t))
      return LLVM::LLVMPointerType::get(t.getContext());
    return {};
  });
}

/// Populate op conversion patterns for CUDA dialect ops to LLVM ops.
void mlir::populateCUDAToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    CudaAllocConverter,
    CudaCopyConverter<cuda::CopyD2DOp>,
    CudaCopyConverter<cuda::CopyD2HOp>,
    CudaCopyConverter<cuda::CopyH2DOp>,
    CudaDeallocConverter,
    CudaLaunchOpToLLVMCallConverter,
    CudaStreamSyncConverter
  >(typeConverter);
  patterns.add<
    CudaGetActiveDeviceConverter,
    CudaSetActiveDeviceConverter,
    CudaGetDeviceCountConverter,
    CudaGetDeviceConverter
  >(typeConverter, PatternBenefit(10));
  // clang-format on
}

LogicalResult mlir::lowerCUDAGlobalsToLLVM(IRRewriter &rewriter,
                                           ModuleOp rootOp,
                                           SymbolTableCollection &symbolTables,
                                           StringRef artifactsDir) {
  SymbolUserMap userMap(symbolTables, rootOp);
  SymbolTable &symbolTable = symbolTables.getSymbolTable(rootOp);

  // Remove `cuda.compiled_module` operations, and create a lookup table
  // (param: map) from module name strings to the addresses of the module
  // data.
  if (failed(lowerCuModuleOps(rewriter, rootOp, symbolTable, userMap,
                              artifactsDir)))
    return emitError(rootOp.getLoc())
           << "failed to lower 'cuda.compiled_module' ops";

  if (failed(lowerGetGlobalStreamOps(rewriter, rootOp, symbolTable)))
    return emitError(rootOp.getLoc()) << "failed to lower CUDA global streams";
  return success();
}

namespace {
class CUDAToLLVMPass
    : public mlir::impl::ConvertCUDAToLLVMPassBase<CUDAToLLVMPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp rootOp = getOperation();
    IRRewriter rewriter(IRRewriter::atBlockEnd(rootOp.getBody()));
    SymbolTableCollection symbolTables;
    if (failed(lowerCUDAGlobalsToLLVM(rewriter, rootOp, symbolTables,
                                      artifactsDirectory)))
      return signalPassFailure();

    // Convert all cuda ops in the program to llvm.call ops to the
    // corresponding cuda API functions
    LLVMTypeConverter typeConverter(&getContext());
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    target.addIllegalDialect<cuda::CUDADialect>();
    populateCUDAToLLVMConversionPatterns(typeConverter, patterns);
    populateCUDAToLLVMTypeConversions(typeConverter);
    populatePlanToLLVMTypeConversions(typeConverter);
    if (failed(applyPartialConversion(rootOp, target, std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply conversion patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace

namespace {
/// Implement the interface to convert CUDA to LLVM.
struct CUDAToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    target.addIllegalDialect<cuda::CUDADialect>();
    populateCUDAToLLVMConversionPatterns(typeConverter, patterns);
    populateCUDAToLLVMTypeConversions(typeConverter);
  }
};
} // namespace

void mlir::registerConvertCUDAToLLVMPatternInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cuda::CUDADialect *dialect) {
    dialect->addInterfaces<CUDAToLLVMDialectInterface>();
  });
}
