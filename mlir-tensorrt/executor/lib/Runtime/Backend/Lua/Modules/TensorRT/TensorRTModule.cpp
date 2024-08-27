//===- TensorRTModule.cpp -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-executor/Runtime/Backend/Lua/Modules/TensorRT/TensorRTModule.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-executor/Support/Status.h"
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

//===----------------------------------------------------------------------===//
// TensorRTCallBackOutputAllocator
//===----------------------------------------------------------------------===//

static bool isSubByte(nvinfer1::DataType t) {
  return t == nvinfer1::DataType::kINT4;
}

static int32_t elementSizeInBits(nvinfer1::DataType t) {
  switch (t) {
  case nvinfer1::DataType::kINT64:
    return 64;
  case nvinfer1::DataType::kINT32:
    return 32;
  case nvinfer1::DataType::kFLOAT:
    return 32;
  case nvinfer1::DataType::kHALF:
    return 16;
  case nvinfer1::DataType::kBF16:
    return 16;
  case nvinfer1::DataType::kINT8:
    return 8;
  case nvinfer1::DataType::kBOOL:
    return 8;
  case nvinfer1::DataType::kUINT8:
    return 8;
  case nvinfer1::DataType::kFP8:
    return 8;
  case nvinfer1::DataType::kINT4:
    return 4;
  }
  return 0;
}

static int32_t elementeSizeInBytes(nvinfer1::DataType dtype) {
  if (!isSubByte(dtype)) {
    auto bits = elementSizeInBits(dtype);
    assert(bits % 8 == 0);
    return bits / 8;
  }
  if (dtype == nvinfer1::DataType::kINT4) {
    return 1;
  }
  return -1;
}

static int64_t volume(nvinfer1::Dims64 const& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
    {
        v *= d.d[i];
    }
    return v;
}

class TensorRTCallBackOutputAllocator final
    : public nvinfer1::IOutputAllocator {
public:
  TensorRTCallBackOutputAllocator(GpuAllocator* gpuAllocator, OutputAllocator *outputAllocator,
                                  const char *tensorName, void *currentMemory,
                                  nvinfer1::Dims64 dims,
                                  nvinfer1::DataType dtype)
      : nvinfer1::IOutputAllocator(),
        mOutputAllocatorCallBack(outputAllocator) {
    mOutputAllocatorCallBack->setGpuAllocator(gpuAllocator);
    mOutputAllocatorCallBack->setTensorName(tensorName);
    mOutputAllocatorCallBack->setCurrentMemory(currentMemory);
    mOutputAllocatorCallBack->setOutputSize(volume(dims) *
                                            elementeSizeInBytes(dtype));
  }

  void *reallocateOutput(char const *tensorName, void *currentMemory,
                         uint64_t size, uint64_t alignment) noexcept override {
    return mOutputAllocatorCallBack->reallocateOutputAsync(
        tensorName, currentMemory, size, alignment, nullptr);
  }

  //! IMirroredBuffer does not implement Async allocation, hence this is just a
  //! wrap around
  void *reallocateOutputAsync(char const *tensorName, void *currentMemory,
                              uint64_t size, uint64_t alignment,
                              cudaStream_t stream) noexcept override {

    return mOutputAllocatorCallBack->reallocateOutputAsync(
        tensorName, currentMemory, size, alignment, &stream);
  }

  void notifyShape(char const *tensorName,
                   nvinfer1::Dims const &dims) noexcept override {
    return mOutputAllocatorCallBack->notifyShape(tensorName, &dims.d[0], dims.nbDims);
  }

  ~TensorRTCallBackOutputAllocator() override {}

private:
  OutputAllocator *mOutputAllocatorCallBack;
};

//===----------------------------------------------------------------------===//
// TensorRTCallBackAllocator
//===----------------------------------------------------------------------===//

class TensorRTCallBackAllocator final : public nvinfer1::IGpuAsyncAllocator {
public:
  TensorRTCallBackAllocator(GpuAllocator *gpuAllocator)
      : nvinfer1::IGpuAsyncAllocator(), mGpuAllocatorCallBack(gpuAllocator) {}

  void *allocateAsync(uint64_t const size, uint64_t const alignment,
                      uint32_t flags, cudaStream_t stream) noexcept final {
    void *result =
        mGpuAllocatorCallBack->allocate(size, alignment, flags, &stream);
    return result;
  }

  bool deallocateAsync(void *const memory,
                       cudaStream_t stream) noexcept override {
    bool result = mGpuAllocatorCallBack->deallocate(memory, &stream);
    return result;
  }

private:
  GpuAllocator *mGpuAllocatorCallBack;
};

} // namespace

static StdioLogger logger(/*verbose=*/false);

//===----------------------------------------------------------------------===//
// ExecutionContextWrapper
//===----------------------------------------------------------------------===//
namespace {
struct Signature {
  unsigned numArguments;
  unsigned numResults;

  explicit Signature(const nvinfer1::ICudaEngine *e)
      : numArguments(0), numResults(0) {
    for (int32_t i = 0; i < e->getNbIOTensors(); i++) {
      const char *name = e->getIOTensorName(i);
      if (e->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        numArguments++;
      else
        numResults++;
    }
  }
};

class NvInferRuntimeWrapper {
public:
  explicit NvInferRuntimeWrapper(GpuAllocator* gpuAllocator) {
    runtime = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger), [](nvinfer1::IRuntime *runtime) {
          MTRT_DBGF("freeing tensorrt runtime at %lu",
                    reinterpret_cast<uintptr_t>(runtime));
          delete runtime;
        });
    // GpuAllocator is optional.
    if (gpuAllocator) {
      callbackAllocatorPair =
          std::make_pair(std::shared_ptr<nvinfer1::IGpuAsyncAllocator>(
                             new TensorRTCallBackAllocator(gpuAllocator)),
                         gpuAllocator);
      runtime->setGpuAllocator(callbackAllocatorPair.first.get());
    }
  }

  nvinfer1::IRuntime *operator*() { return runtime.get(); }
  nvinfer1::IRuntime *operator->() { return runtime.get(); }

  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::pair<std::shared_ptr<nvinfer1::IGpuAsyncAllocator>, GpuAllocator*> callbackAllocatorPair;
};

class NvInferEngineWrapper {
public:
  explicit NvInferEngineWrapper(std::shared_ptr<NvInferRuntimeWrapper> runtime,
                                uintptr_t pointer, size_t size)
      : runtime(runtime) {
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->runtime->deserializeCudaEngine(
            reinterpret_cast<void *>(pointer), size),
        [](nvinfer1::ICudaEngine *engine) {
          MTRT_DBGF("freeing cuda engine at %lu",
                    reinterpret_cast<uintptr_t>(engine));
          delete engine;
        });
  }

  nvinfer1::ICudaEngine *operator*() { return engine.get(); }
  nvinfer1::ICudaEngine *operator->() { return engine.get(); }

  std::shared_ptr<NvInferRuntimeWrapper> runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
};

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
  nvinfer1::IExecutionContext *operator->() { return context.get(); }
  const Signature &getSignature() const { return signature; }

  /// Returned the pre-allocated host staging buffers.
  std::vector<PinnedMemoryBlock> &getHostIOBuffers() { return hostIOBuffers; }

  /// Add a call back output allocator.
  void addCallBackAllocators(
      std::unique_ptr<TensorRTCallBackOutputAllocator> allocator) {
    outputAllocators.emplace_back(std::move(allocator));
  }

  /// Return the last call back output allocator pointer.
  TensorRTCallBackOutputAllocator *getLastCallBackAllocatorPtr() {
    return outputAllocators.back().get();
  }

  /// Return registered callback gpu allocator.
  GpuAllocator *getGpuAllocator() {
    return engine->runtime->callbackAllocatorPair.second;
  }

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
  std::vector<std::unique_ptr<TensorRTCallBackOutputAllocator>> outputAllocators;
};
} // namespace

static Status setTensorAddressesAndOutputAllocatorsOrReport(
    NvInferExecContextWrapper &context,
    const std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>
        &buffers, OutputAllocatorTracker &outputAllocatorTracker) {
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

    const nvinfer1::ICudaEngine &engine = context->getEngine();

    if (!result) {
      std::stringstream ss;
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

    if (idx < context.getSignature().numArguments) {
      result = context->setInputShape(name.c_str(), dims);
      if (!result)
        return getInternalErrorStatus("failed to set input shape");
    }

    // Set output allocators
    if (engine.getTensorIOMode(name.c_str()) ==
            nvinfer1::TensorIOMode::kOUTPUT and
        engine.getTensorLocation(name.c_str()) ==
            nvinfer1::TensorLocation::kDEVICE) {

      // Since setting output allocator is optional.
      if (outputAllocatorTracker.getAllocator(reinterpret_cast<void *>(ptr)) !=
          nullptr) {
        context.addCallBackAllocators(
            std::make_unique<TensorRTCallBackOutputAllocator>(
                context.getGpuAllocator(),
                outputAllocatorTracker.getAllocator(
                    reinterpret_cast<void *>(ptr)),
                name.c_str(), reinterpret_cast<void *>(ptr), dims,
                engine.getTensorDataType(name.c_str())));
        context->setOutputAllocator(name.c_str(),
                                    static_cast<nvinfer1::IOutputAllocator *>(
                                        context.getLastCallBackAllocatorPtr()));
      } else {
        // It is possible that previous call with same output name and different
        // memref pointer would have set output allocator. Due to "hacky" naming
        // scheme, outputs are always named as "result0", "result1", .... If not
        // tracker is found for a given pointer, let's unset the output
        // allocator.
        if (context->getOutputAllocator(name.c_str())) {
          context->setOutputAllocator(name.c_str(), nullptr);
        }
      }
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
               sol::table &va) {
  ADD_TENSORRT_MODULE_RANGE("prepare_buffers");
  std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>> result;
  const Signature &sig = context.getSignature();
  unsigned argumentBuffersIdx = 1;
  // The number of arguments should be equal to the number of results plus the
  // number of arguments of the TensorRT engine's functional signature.
  const unsigned numOperands = sig.numResults + sig.numArguments;
  result.reserve(va.size() / 3);
  std::vector<PinnedMemoryBlock> &hostBuffers = context.getHostIOBuffers();
  unsigned hostBufferIdx = 0;
  for (unsigned i = 0; i < numOperands; i++) {
    // We have a fixed naming scheme for each operand that is obeyed by the
    // compiler.
    /// TODO: make this less hacky.
    std::string name =
        (i < sig.numArguments ? "arg" : "result") +
        std::to_string(i >= sig.numArguments ? i - sig.numArguments : i);

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
        !buffer.isDeviceVisible()) {
      mlir_trt_unreachable("expected device buffer");
    }

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
                               OutputAllocatorTracker &outputAllocatorTracker,
                               NvInferExecContextWrapper &context,
                               CudaStreamPtr stream, sol::table &va) {
  StatusOr<std::vector<std::tuple<std::string, uintptr_t, nvinfer1::Dims>>>
      buffers = prepareBuffers(tracker, context, stream, va);
  if (!buffers.isOk())
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to prepare buffers: ", buffers.getString());


  MTRT_RETURN_IF_ERROR(setTensorAddressesAndOutputAllocatorsOrReport(context, *buffers, outputAllocatorTracker));
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

//===----------------------------------------------------------------------===//
// Executor TensorRT Methods
//===----------------------------------------------------------------------===//
void mlirtrt::runtime::registerExecutorTensorRTModuleLuaRuntimeMethods(
    lua_State *luaState, PinnedMemoryAllocator *pinnedMemoryAllocator,
    AllocTracker *allocTracker, ResourceTracker *resourceTracker,
    OutputAllocatorTracker *outputAllocatorTracker, GpuAllocator *allocator) {
  sol::state_view lua(luaState);

  lua["_trtrt_create_runtime"] =
      [allocator](sol::this_state state) -> std::shared_ptr<NvInferRuntimeWrapper> {
    ADD_TENSORRT_MODULE_RANGE("trtrt_create_runtime");
    MTRT_DBGF("%s", "creating nvinfer runtime");
    return std::make_shared<NvInferRuntimeWrapper>(allocator);
  };

  lua["_trtrt_load"] =
      [allocTracker](
          sol::this_state state,
          std::shared_ptr<NvInferRuntimeWrapper> &runtime,
          uintptr_t pointer) -> std::shared_ptr<NvInferEngineWrapper> {
    ADD_TENSORRT_MODULE_RANGE("trtrt_load");
    const AllocTracker &tracker = *allocTracker;
    MTRT_DBGF("%s", "loading nvinfer cuda engine");
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
      [allocTracker, resourceTracker, outputAllocatorTracker](
          sol::this_state state,
          std::shared_ptr<NvInferExecContextWrapper> context,
          CudaStreamPtr stream, sol::table va) {
        ADD_TENSORRT_MODULE_RANGE("trtrt_enqueue");
        sol::state_view luaState(state);
        assert(context != nullptr);
        assert(stream != nullptr && "expected valid stream");
        Status result =
            enqueueV3Wrapper(*allocTracker, *resourceTracker,
                             *outputAllocatorTracker, *context, stream, va);
        SET_LUA_ERROR_IF_ERROR(result, state);
      };
}
