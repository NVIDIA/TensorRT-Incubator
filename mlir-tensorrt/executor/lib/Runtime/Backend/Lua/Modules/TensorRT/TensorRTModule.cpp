//===- TensorRTModule.cpp -------------------------------------------------===//
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
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Lua/SolAdaptor.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include <memory>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInfer.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

namespace {
/// A simple logger that implements TensorRT's logging interface. Errors and
/// warnings are reported through TensorRT's diagnostic system, everything else
/// is printed to stderr if the verbose flag is present.
class StdioLogger : public nvinfer1::ILogger {
public:
  StdioLogger(bool verbose) : verbose(verbose) {}

protected:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      fprintf(stderr, "%s\n", msg);
      return;
    }
    if (severity == Severity::kWARNING) {
      fprintf(stderr, "%s\n", msg);
      return;
    }
    if (!verbose)
      return;
    fprintf(stderr, "%s\n", msg);
  }
  bool verbose;
};

} // namespace

static StdioLogger logger(/*verbose=*/false);

//===----------------------------------------------------------------------===//
// ExecutionContextWrapper
//===----------------------------------------------------------------------===//
namespace {
struct Signature {
  unsigned numInputs;
  unsigned numOutputs;

  explicit Signature(const nvinfer1::ICudaEngine *e)
      : numInputs(0), numOutputs(0) {
    for (int32_t i = 0; i < e->getNbIOTensors(); i++) {
      const char *name = e->getIOTensorName(i);
      if (e->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        numInputs++;
      else
        numOutputs++;
    }
  }
};

class NvInferEngineWrapper {
public:
  explicit NvInferEngineWrapper(std::shared_ptr<nvinfer1::IRuntime> &runtime,
                                uintptr_t pointer, size_t size)
      : runtime(runtime) {
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(reinterpret_cast<void *>(pointer), size),
        [](nvinfer1::ICudaEngine *engine) {
          MTRT_DBGF("freeing cuda engine at %lu",
                    reinterpret_cast<uintptr_t>(engine));
          delete engine;
        });
  }

  nvinfer1::ICudaEngine *operator*() { return engine.get(); }
  nvinfer1::ICudaEngine *operator->() { return engine.get(); }

  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
};
} // namespace

static uint64_t roundUp(uint64_t m, uint64_t n) {
  return llvm::divideCeil(m, n) * n;
}

//===----------------------------------------------------------------------===//
// NvInferResultAllocator
//===----------------------------------------------------------------------===//

namespace {
/// The NvInferResultAllocator provides hooks for allocating a TensorRT
/// execution result. There should be one instance of this class for each result
/// tensor of a TensorRT engine. The allocator can be seeded with an initial
/// allocation so that it can be extended via reallocation when the final size
/// is known. This capability is currently unused, however.
/// TODO: to use reallocation capability, we need to represent the allocator in
/// the IR.
class NvInferResultAllocator : public nvinfer1::IOutputAllocator {
public:
  NvInferResultAllocator(AllocTracker *tracker, unsigned resultIdx)
      : mTracker(tracker) {
    // TensorRT tracks inputs/outputs by name literal rather than index. We use
    // a particular convention between compiler and runtime for the naming of
    // inputs and results.
    mTensorName = "result" + std::to_string(resultIdx);

    allocInfo = PointerInfo(0, 0);
  }

  /// It is the responsibilty of runtime client to release the buffer.
  ~NvInferResultAllocator() override = default;

  // Disable copy and move operations
  NvInferResultAllocator(const NvInferResultAllocator &) = delete;
  NvInferResultAllocator &operator=(const NvInferResultAllocator &) = delete;
  NvInferResultAllocator(NvInferResultAllocator &&) = delete;
  NvInferResultAllocator &operator=(NvInferResultAllocator &&) = delete;

  void *reallocateOutputAsyncImpl(const char *name, void *memory, uint64_t size,
                                  uint64_t alignment, cudaStream_t stream) {
    assert(name == mTensorName && "Tensor name mismatch");

    // Some memory allocators return nullptr when allocating zero bytes, but
    // TensorRT requires a non-null ptr
    // even for empty tensors, so allocate a dummy byte.
    size = std::max(size, static_cast<uint64_t>(1));
    size = roundUp(size, alignment);

    StatusOr<PointerInfo> alloc = mlirtrt::runtime::allocate(
        *mTracker, PointerType::device, size, alignment,
        stream ? std::optional<CudaStreamPtr>(stream) : std::nullopt);
    if (!alloc.isOk())
      return nullptr;

    allocInfo = std::move(*alloc);
    // Mark the output pointer for release after consumption
    // This is necessary because TensorRT-allocated pointers used in
    // device-device or device-host copies may not be wrapped in a memref
    // and tracked by the client. By marking it here, we ensure it will be
    // explicitly freed after it's consumed in copy operations, preventing
    // memory leaks.
    mTracker->markForReleaseAfterConsumption(allocInfo.ptr);
    MTRT_DBGF("tensorrt module output allocator allocating %lu bytes at 0x%lx",
              allocInfo.size, allocInfo.ptr);
    return reinterpret_cast<void *>(allocInfo.ptr);
  }

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  void *reallocateOutputAsync(const char *name, void *memory, uint64_t size,
                              uint64_t alignemnt,
                              cudaStream_t stream) override {
    return reallocateOutputAsyncImpl(name, memory, size, alignemnt, stream);
  }
#else
  void *reallocateOutput(const char *name, void *memory, uint64_t size,
                         uint64_t alignment) override {
    return reallocateOutputAsyncImpl(name, memory, size, alignment, nullptr);
  }
#endif

  void notifyShape(const char *name,
                   const nvinfer1::Dims &dims) noexcept override {
    assert(name == mTensorName && "Tensor name mismatch");
    mOutputDims = dims;
  }

  // Accessor methods
  const nvinfer1::Dims &getOutputDims() const { return mOutputDims; }
  uintptr_t getOutputPtr() const { return allocInfo.ptr; }
  uint64_t getOutputSize() const { return allocInfo.size; }
  const std::string &getTensorName() const { return mTensorName; }

private:
  AllocTracker *mTracker{nullptr};
  std::string mTensorName;
  PointerInfo allocInfo;
  nvinfer1::Dims mOutputDims;
};

//===----------------------------------------------------------------------===//
// NvInferResultAllocators
//===----------------------------------------------------------------------===//

/// NvInferResultAllocators wraps all result allocators for a single TensorRT
/// execution context.
class NvInferResultAllocators {
public:
  explicit NvInferResultAllocators(AllocTracker *tracker,
                                   nvinfer1::IExecutionContext *context,
                                   int64_t nbResults) {
    mAllocators.reserve(nbResults);
    for (int64_t i = 0; i < nbResults; ++i) {
      mAllocators.push_back(
          std::make_unique<NvInferResultAllocator>(tracker, i));
      context->setOutputAllocator(mAllocators.back()->getTensorName().c_str(),
                                  mAllocators.back().get());
    }
  }

  NvInferResultAllocator *getAllocator(size_t index) const {
    assert(index >= 0 && index < mAllocators.size() && "Index out of bounds");
    return mAllocators[index].get();
  }

private:
  std::vector<std::unique_ptr<NvInferResultAllocator>> mAllocators;
};

//===----------------------------------------------------------------------===//
// OutputDescriptor
//===----------------------------------------------------------------------===//

/// Manages output tensor descriptors for TensorRT execution.
class OutputDescriptor {
public:
  OutputDescriptor(uintptr_t ptr)
      : mData(reinterpret_cast<int64_t *>(ptr)),
        mSize(calculateTotalSize(ptr)) {}

  int64_t getNumberOfResults() const { return mData[0]; }

  unsigned getRank(int resultIndex) const {
    size_t index = getIndexForResult(resultIndex);
    return static_cast<unsigned>(mData[index]);
  }

  void setTensorDataPtr(int resultIndex, uintptr_t ptr) {
    size_t index = getIndexForResult(resultIndex);
    mData[index + 1] = static_cast<int64_t>(ptr);
  }

  void setShape(int resultIndex, const std::vector<int64_t> &shape) {
    size_t index = getIndexForResult(resultIndex);
    unsigned rank = static_cast<unsigned>(mData[index]);
    assert(shape.size() == rank && "Shape size doesn't match the rank");
    index += OUTPUT_DESC_FIXED_FIELDS;
    std::copy(shape.begin(), shape.end(), mData + index);
  }

  void setStride(int resultIndex, const std::vector<int64_t> &stride) {
    size_t index = getIndexForResult(resultIndex);
    unsigned rank = static_cast<unsigned>(mData[index]);
    assert(stride.size() == rank && "Stride size doesn't match the rank");
    index += OUTPUT_DESC_FIXED_FIELDS + rank;
    std::copy(stride.begin(), stride.end(), mData + index);
  }

private:
  int64_t *mData; // Pointer to the raw descriptor data.
  size_t mSize;   // Total size of the descriptor data.

  size_t getIndexForResult(int resultIndex) const {
    return calculateOffsetForResult(mData, resultIndex);
  }

  static size_t calculateTotalSize(uintptr_t ptr) {
    int64_t *desc = reinterpret_cast<int64_t *>(ptr);
    int64_t numResults = desc[0];
    return calculateOffsetForResult(desc, numResults);
  }

  static size_t calculateOffsetForResult(const int64_t *desc,
                                         int64_t resultIndex) {
    size_t offset = 1; // Start after number of results
    for (int64_t i = 0; i < resultIndex; ++i) {
      unsigned rank = static_cast<unsigned>(desc[offset]);
      offset += 2 + 2 * rank; // rank + dataPtr + shape + stride
    }
    return offset;
  }

  /// Fixed fields corresponding to rank, data ptr.
  static constexpr int OUTPUT_DESC_FIXED_FIELDS = 2;
};

//===----------------------------------------------------------------------===//
// NvInferExecContextWrapper
//===----------------------------------------------------------------------===//

/// Wraps an nvinfer1::ExecutionContext object.
class NvInferExecContextWrapper {
private:
  explicit NvInferExecContextWrapper(
      std::shared_ptr<NvInferEngineWrapper> engine,
      std::unique_ptr<nvinfer1::IExecutionContext,
                      void (*)(nvinfer1::IExecutionContext *)>
          context,
      std::vector<PinnedMemoryBlock> inputHostBuffers)
      : engine(std::move(engine)), context(std::move(context)),
        signature(**this->engine), hostIOBuffers(std::move(inputHostBuffers)) {}

public:
  static StatusOr<std::shared_ptr<NvInferExecContextWrapper>>
  create(std::shared_ptr<NvInferEngineWrapper> engine,
         PinnedMemoryAllocator *pinnedMemoryAllocator) {
    auto context = std::unique_ptr<nvinfer1::IExecutionContext,
                                   void (*)(nvinfer1::IExecutionContext *)>(
        (*engine)->createExecutionContext(),
        [](nvinfer1::IExecutionContext *ctx) {
          MTRT_DBGF("freeing execution context at %lu",
                    reinterpret_cast<uintptr_t>(ctx));
          delete ctx;
        });
    if (!context)
      return getInternalErrorStatus(
          "failed to create TensorRT ExecutionContext");

    std::vector<PinnedMemoryBlock> hostIOBuffers;
    const int32_t numIOTensors = (*engine)->getNbIOTensors();
    for (int32_t ioIdx = 0; ioIdx < numIOTensors; ++ioIdx) {
      auto const &name = (*engine)->getIOTensorName(ioIdx);
      nvinfer1::TensorLocation location = (*engine)->getTensorLocation(name);
      if (location != nvinfer1::TensorLocation::kHOST)
        continue;
      bool isShapeInferenceIO = (*engine)->isShapeInferenceIO(name);

      assert(isShapeInferenceIO && "expected host tensor to be shape tensor");

      nvinfer1::Dims dims = (*engine)->getTensorShape(name);
      nvinfer1::DataType dataType = (*engine)->getTensorDataType(name);

      // We expect the host shape tensor to be a 0-d or 1-d tensor.
      assert(dataType == nvinfer1::DataType::kINT32 &&
             "expected i32 data type for shape");
      assert(dims.nbDims <= 1 && dims.d[0] >= 0 &&
             "expected rank-0 or rank-1 shape of positive extent");

      // Create the pinned host buffer. Minimum allocation must be 16 bytes,
      // since device allocation may be 16 bytes. These allocations are
      // long-lived for the duration of the program. They are automatically
      // cleaned up when the allocator destructs at program exit.
      StatusOr<PinnedMemoryBlock> hostBuffer = pinnedMemoryAllocator->allocate(
          sizeof(int32_t) * std::max<int32_t>(dims.d[0], 16));
      if (!hostBuffer.isOk())
        return getStatusWithMsg(
            StatusCode::InternalError,
            "failed to allocate host buffer for TRT engine IO shape tensor: ",
            hostBuffer.getString());
      hostIOBuffers.push_back(*hostBuffer);
      assert(hostIOBuffers.back().ptr != 0 && hostIOBuffers.back().ptr > 0);
    }

    return std::shared_ptr<NvInferExecContextWrapper>(
        new NvInferExecContextWrapper(std::move(engine), std::move(context),
                                      std::move(hostIOBuffers)));
  }

  nvinfer1::IExecutionContext *operator*() { return context.get(); }
  nvinfer1::IExecutionContext *getExecutionContext() { return context.get(); }

  nvinfer1::IExecutionContext *operator->() { return context.get(); }
  const Signature &getSignature() const { return signature; }

  /// Returned the pre-allocated host staging buffers.
  std::vector<PinnedMemoryBlock> &getHostIOBuffers() { return hostIOBuffers; }

private:
  // We keep a reference to the cuda engine to keep it from going out of scope.
  // The standard TensorRTRuntime-to-Executor lowering only creates globals for
  // the ExecutionContext, not the engine.
  std::shared_ptr<NvInferEngineWrapper> engine;
  std::unique_ptr<nvinfer1::IExecutionContext,
                  void (*)(nvinfer1::IExecutionContext *)>
      context;
  Signature signature;

  /// A set of pinned host buffers one per input host buffer (shape tensor) to
  /// the TRT network.
  std::vector<PinnedMemoryBlock> hostIOBuffers;
};
} // namespace

static Status setTensorAddressesOrReport(
    NvInferExecContextWrapper &context,
    const std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>
        &buffers) {
  ADD_TENSORRT_MODULE_RANGE("set_tensor_addresses");
  unsigned idx = 0;
  for (auto &[name, ptr, dims] : buffers) {
    constexpr intptr_t kMinAlignmentBytes = 256;
    if (ptr % kMinAlignmentBytes != 0)
      MTRT_WARNV("TensorRT input {0} (ptr = {1}) does not meet minimum "
                 "alignment of {2} bytes",
                 name, ptr, kMinAlignmentBytes);

    bool result =
        context->setTensorAddress(name.c_str(), reinterpret_cast<void *>(ptr));

    if (!result) {
      std::stringstream ss;
      const nvinfer1::ICudaEngine &engine = context->getEngine();
      ss << "Failed to set tensor address for IO tensor: " << name
         << " at position " << idx << "; the IO tensors are:\n";
      for (int64_t i = 0; i < engine.getNbIOTensors(); i++) {
        const char *name = engine.getIOTensorName(i);
        ss << "[" << i << "] " << name << " : ";
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
          ss << "input";
        else
          ss << "output";
        ss << "\n";
      }
      return getStatusWithMsg(StatusCode::InternalError, ss.str());
    }

    if (idx < context.getSignature().numInputs) {
      result = context->setInputShape(name.c_str(), dims);
      if (!result)
        return getInternalErrorStatus("failed to set input shape");
    }

    MTRT_DBGF("Set tensor address [%d] = %lu", idx, ptr);
    idx++;
  }
  return getOkStatus();
}

/// Prepare buffer inputs for passing to a TensorRT engine asynchronous
/// execution call.
static StatusOr<std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>>
prepareBuffers(const AllocTracker &allocTracker,
               NvInferExecContextWrapper &context, CudaStreamPtr stream,
               sol::table &va, bool outputAsArg = true) {
  ADD_TENSORRT_MODULE_RANGE("prepare_buffers");
  std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>> result;
  const Signature &sig = context.getSignature();
  unsigned argumentBuffersIdx = 1;
  // The number of arguments should be equal to the number of results plus the
  // number of arguments of the TensorRT engine's functional signature.
  const unsigned numOperands =
      (outputAsArg ? sig.numOutputs : 0) + sig.numInputs;
  result.reserve(va.size() / 3);
  std::vector<PinnedMemoryBlock> &hostBuffers = context.getHostIOBuffers();
  unsigned hostBufferIdx = 0;
  for (unsigned i = 0; i < numOperands; i++) {
    // We have a fixed naming scheme for each operand that is obeyed by the
    // compiler.
    /// TODO: make this less hacky.
    std::string name =
        (i < sig.numInputs ? "arg" : "result") +
        std::to_string(i >= sig.numInputs ? i - sig.numInputs : i);

    // Parse the arguments: ptr, offset, rank, shape...
    uintptr_t ptr = va.get<uintptr_t>(argumentBuffersIdx++);
    PointerInfo buffer = allocTracker.lookupOrDefault(ptr);
    int64_t offset = va.get<int64_t>(argumentBuffersIdx++);
    int64_t rank = va.get<int64_t>(argumentBuffersIdx++);
    nvinfer1::Dims dims;
    dims.nbDims = rank;
    for (int64_t dimIdx = 0; dimIdx < rank; dimIdx++)
      dims.d[dimIdx] = va.get<int64_t>(argumentBuffersIdx++);

    uintptr_t pointer = buffer.ptr + offset;
    MTRT_DBGF("enqueue arg %u ptr=0x%lx offset=%ld", i, buffer.ptr, offset);

    // Determine whether the TensorRT engine expects the buffer in the host or
    // device address spaces.
    nvinfer1::TensorLocation location =
        context->getEngine().getTensorLocation(name.c_str());

    // If TRT expect's the device space but the buffer is not present on the
    // device, then fail hard with an error. By default, the compiler will put
    // all buffers that feed into TRT on the device, so this case should never
    // happen. In the future we can try to predict host/device placement then
    // use this case as a chance to make a correction and copy a buffer from
    // device-to-host.
    if (location == nvinfer1::TensorLocation::kDEVICE &&
        !buffer.isDeviceVisible())
      return getInvalidArgStatus(
          "attempted to invoke a TensorRT engine with a non-device-visible "
          "buffer where a device buffer was expected");

    // If TRT expect's the buffer to be on the host (e.g. shape tensor), then we
    // should copy the device buffer to a page-locked staging host buffer.
    if (location == nvinfer1::TensorLocation::kHOST &&
        !buffer.isHostVisible()) {
      // If the buffer has unknown size (i.e. it is an unkown size buffer view),
      // then this is an error.
      if (!buffer.hasKnownSize())
        return getStatusWithMsg(
            StatusCode::InternalError,
            "buffer must be copied to host, but it has unknown size");
      MTRT_DBGF("Input %s should be located on HOST, copying to temporary host "
                "buffer of size %lu",
                name.c_str(), buffer.size);

      // Asynchronously copy the buffer to host. We do not need to
      // synchronize here because the `stream` is the same one on which TRT
      // execution will be enqueued.
      unsigned indexInHostBuffers = hostBufferIdx++;
      assert(indexInHostBuffers < hostBuffers.size() &&
             "host buffer index out-of-bounds");
      PinnedMemoryBlock &hostBuffer = hostBuffers[indexInHostBuffers];
      MTRT_DBGF("copying %ld bytes from 0x%lx (device) to 0x%lx (host)",
                buffer.size, pointer, hostBuffer.ptr);
      cudaError_t cudaErr =
          cudaMemcpyAsync(reinterpret_cast<void *>(hostBuffer.ptr),
                          reinterpret_cast<void *>(pointer), buffer.size,
                          cudaMemcpyDeviceToHost, stream);
      if (cudaErr != cudaSuccess)
        return getStatusWithMsg(StatusCode::InternalError,
                                "failed to copy shape tensor to host: ",
                                cudaGetErrorString(cudaErr));
      result.emplace_back(name, hostBuffer.ptr, dims);
      continue;
    }

    result.emplace_back(name, pointer, dims);
  }
  return result;
}

static Status enqueueV3Wrapper(AllocTracker &tracker,
                               ResourceTracker &resourceTracker,
                               NvInferExecContextWrapper &context,
                               CudaStreamPtr stream, sol::table &va) {
  StatusOr<std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>>
      buffers = prepareBuffers(tracker, context, stream, va);
  if (!buffers.isOk())
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to prepare buffers: ", buffers.getString());
  MTRT_RETURN_IF_ERROR(setTensorAddressesOrReport(context, *buffers));
  // Create an event that we can wait on for releasing any host-pinned staging
  // allocations we made.
  MTRT_ASSIGN_OR_RETURN(CudaEventPtr inputConsumedEvent,
                        CudaEventPtr::create(resourceTracker));
  if (!context->setInputConsumedEvent(inputConsumedEvent))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to set input-consumed event");

  if (!context->enqueueV3(stream))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to enqueue engine execution on stream");
  cudaError_t waitResult = cudaStreamWaitEvent(stream, inputConsumedEvent);
  RETURN_ERROR_IF_CUDART_ERROR(waitResult);

  MTRT_DBGF("%s", "enqueueV3 successful and inputs are consumed");

  return getOkStatus();
}

static Status enqueueAllocV3Wrapper(AllocTracker &tracker,
                                    ResourceTracker &resourceTracker,
                                    NvInferResultAllocators *outputAllocator,
                                    NvInferExecContextWrapper &context,
                                    CudaStreamPtr stream, sol::table &va,
                                    OutputDescriptor outputDesc) {

  StatusOr<std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>>
      buffers =
          prepareBuffers(tracker, context, stream, va, /*outputAsArg=*/false);
  if (!buffers.isOk())
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to prepare buffers: ", buffers.getString());

  MTRT_RETURN_IF_ERROR(setTensorAddressesOrReport(context, *buffers));

  // Create an event that we can wait on for releasing any host-pinned staging
  // allocations we made.
  MTRT_ASSIGN_OR_RETURN(CudaEventPtr inputConsumedEvent,
                        CudaEventPtr::create(resourceTracker));

  if (!context->setInputConsumedEvent(inputConsumedEvent))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to set input-consumed event");

  // Number of results are known in advance.
  int64_t nbResults = outputDesc.getNumberOfResults();

  if (!context->enqueueV3(stream))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to enqueue engine execution on stream");

  cudaError_t waitResult = cudaStreamWaitEvent(stream, inputConsumedEvent);
  RETURN_ERROR_IF_CUDART_ERROR(waitResult);

  MTRT_DBGF("enqueueV3 successful and inputs are consumed on stream 0x%lx",
            stream.ptr);

  for (int64_t i = 0; i < nbResults; ++i) {
    NvInferResultAllocator *outputAllocatorImpl =
        outputAllocator->getAllocator(i);
    std::string name = "result" + std::to_string(i);

    int32_t rank = outputDesc.getRank(i);

    // Validate rank
    if (rank != outputAllocatorImpl->getOutputDims().nbDims) {
      return getStatusWithMsg(
          StatusCode::InternalError,
          "Result rank mismatch. Expected rank: ", std::to_string(rank),
          std::to_string(outputAllocatorImpl->getOutputDims().nbDims));
    }

    // Set output pointer
    outputDesc.setTensorDataPtr(i, outputAllocatorImpl->getOutputPtr());

    // Validate and set shape
    nvinfer1::Dims shape = context->getTensorShape(name.c_str());
    assert(std::equal(shape.d, shape.d + shape.nbDims,
                      outputAllocatorImpl->getOutputDims().d));
    assert(shape.nbDims == static_cast<int32_t>(rank));
    std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
    outputDesc.setShape(i, shapeVec);

    // Validate and set stride
    nvinfer1::Dims stride = context->getTensorStrides(name.c_str());
    assert(stride.nbDims == static_cast<int32_t>(rank));
    std::vector<int64_t> strideVec(stride.d, stride.d + stride.nbDims);
    outputDesc.setStride(i, strideVec);
  }

  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// Executor TensorRT Methods
//===----------------------------------------------------------------------===//

static void registerExecutorTensorRTModuleLuaRuntimeMethods(
    lua_State *luaState, PinnedMemoryAllocator *pinnedMemoryAllocator,
    AllocTracker *allocTracker, ResourceTracker *resourceTracker) {
  sol::state_view lua(luaState);

  lua["_trtrt_create_runtime"] = [](sol::this_state state) {
    ADD_TENSORRT_MODULE_RANGE("trtrt_create_runtime");
    MTRT_DBGF("%s", "creating nvinfer runtime");
    return std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger));
  };

  lua["_trtrt_load"] =
      [allocTracker](sol::this_state state,
                     std::shared_ptr<nvinfer1::IRuntime> &runtime,
                     uintptr_t pointer,
                     size_t size) -> std::shared_ptr<NvInferEngineWrapper> {
    ADD_TENSORRT_MODULE_RANGE("trtrt_load");
    const AllocTracker &tracker = *allocTracker;
    MTRT_DBGF("%s", "loading nvinfer cuda engine");
    assert(size <= tracker.get(pointer).size &&
           "specified engine size is smaller than loaded serialized "
           "engine buffer");
    return std::make_shared<NvInferEngineWrapper>(runtime, pointer,
                                                  tracker.get(pointer).size);
  };

  lua["_trtrt_create_context"] =
      [pinnedMemoryAllocator](sol::this_state state,
                              std::shared_ptr<NvInferEngineWrapper> &engine)
      -> std::shared_ptr<NvInferExecContextWrapper> {
    ADD_TENSORRT_MODULE_RANGE("trtrt_create_context");
    sol::state_view luaState(state);
    assert(engine != nullptr);
    StatusOr<std::shared_ptr<NvInferExecContextWrapper>> ctx =
        NvInferExecContextWrapper::create(engine, pinnedMemoryAllocator);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(ctx, state, nullptr);
    assert(*ctx && "expected valid context");
    return *ctx;
  };

  lua["_trtrt_enqueue"] =
      [allocTracker,
       resourceTracker](sol::this_state state,
                        std::shared_ptr<NvInferExecContextWrapper> context,
                        CudaStreamPtr stream, sol::table va) {
        ADD_TENSORRT_MODULE_RANGE("trtrt_enqueue");
        sol::state_view luaState(state);
        assert(context != nullptr);
        assert(stream != nullptr && "expected valid stream");
        Status result = enqueueV3Wrapper(*allocTracker, *resourceTracker,
                                         *context, stream, va);
        SET_LUA_ERROR_IF_ERROR(result, state);
      };

  lua["_trtrt_enqueue_alloc"] =
      [allocTracker, resourceTracker](
          sol::this_state state,
          std::shared_ptr<NvInferExecContextWrapper> context,
          CudaStreamPtr stream, uintptr_t outputDesc, sol::table va) {
        ADD_TENSORRT_MODULE_RANGE("trtrt_enqueue_alloc");
        sol::state_view luaState(state);
        assert(context != nullptr);
        assert(stream != nullptr && "expected valid stream");

        OutputDescriptor desc(outputDesc);

        auto allocator = std::make_unique<NvInferResultAllocators>(
            allocTracker, **context, desc.getNumberOfResults());
        Status result =
            enqueueAllocV3Wrapper(*allocTracker, *resourceTracker,
                                  allocator.get(), *context, stream, va, desc);
        SET_LUA_ERROR_IF_ERROR(result, state);
      };
}

namespace mlirtrt::runtime {
void registerLuaTensorRTRuntimeExtension() {
  registerLuaRuntimeExtension(
      "tensorrt",
      LuaRuntimeExtension{
          [](const RuntimeSessionOptions &options, lua_State *state,
             PinnedMemoryAllocator *pinnedMemoryAllocator,
             AllocTracker *allocTracker, ResourceTracker *resourceTracker) {
            registerExecutorTensorRTModuleLuaRuntimeMethods(
                state, pinnedMemoryAllocator, allocTracker, resourceTracker);
          }});
}
} // namespace mlirtrt::runtime
