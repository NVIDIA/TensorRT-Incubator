//===- TensorRTModule.cpp -------------------------------------------------===//
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
/// MTRT TensorRT C runtime module definitions.
///
//===----------------------------------------------------------------------===//
#include "TensorRTModule.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "NvInferRuntime.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define MTRT_DBGF(fmt, ...)                                                    \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      std::fprintf(stderr, "%s:%d:%s(): " fmt "\n", "TensorRTModule.cpp",      \
                   __LINE__, __func__, __VA_ARGS__);                           \
  } while (0)

static const char *kDebugEnvironmentVariable = "MTRT_TENSORRT_DEBUG";

/// Helper method that checks environment value for debugging.
static bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

static int readInputFile(const std::string &filename,
                         std::vector<char> &buffer) {
  // Open the binary file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  // Get the size of the file
  std::streamsize size = file.tellg();

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Create a vector to hold the file contents
  buffer.resize(size);

  // Read the entire file into the vector
  if (file.read(buffer.data(), size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

namespace {
/// Template class for tensor IO descriptors.
template <uint32_t Rank>
struct MemRefDescriptor {
  int64_t rank;
  void *data;
  int64_t shape[Rank];
};

/// Explicits specialization for 0D descriptors since
/// "0-length arrays" are only part of certain C/C++ language extensions
template <>
struct MemRefDescriptor<0> {
  int64_t rank;
  void *data;
};

/// A simple logger that implements TensorRT's logging interface. Errors and
/// warnings are reported through TensorRT's diagnostic system, everything else
/// is printed to stderr if the verbose flag is present.
class StdioLogger : public nvinfer1::ILogger {
protected:
  void log(Severity severity, const char *msg) noexcept override {
    fprintf(stderr, "%s\n", msg);
  }
};

//===----------------------------------------------------------------------===//
// NvInferEngine
//===----------------------------------------------------------------------===//

/// A simple RAII wrapper around a `nvinfer1::ICudaEngine`.
class NvInferEngineWrapper {
public:
  NvInferEngineWrapper(const NvInferEngineWrapper &) = delete;
  NvInferEngineWrapper &operator=(const NvInferEngineWrapper &) = delete;

  static std::unique_ptr<NvInferEngineWrapper>
  create(nvinfer1::IRuntime *runtime, void *data, size_t dataSize) {
    return std::unique_ptr<NvInferEngineWrapper>(
        new NvInferEngineWrapper(runtime, data, dataSize));
  }

  static std::unique_ptr<NvInferEngineWrapper>
  createFromFile(nvinfer1::IRuntime *runtime, const char *filename,
                 size_t filenameSize) {
    std::vector<char> fileData;
    if (readInputFile(std::string(filename, filenameSize), fileData))
      return nullptr;
    return std::unique_ptr<NvInferEngineWrapper>(
        new NvInferEngineWrapper(runtime, fileData.data(), fileData.size()));
  }

  ~NvInferEngineWrapper() {
    MTRT_DBGF("freeing cuda engine at 0x%lx",
              reinterpret_cast<uintptr_t>(engine));
    ::delete engine;
  }

  nvinfer1::ICudaEngine *operator*() { return engine; }
  nvinfer1::ICudaEngine *operator->() { return engine; }

private:
  NvInferEngineWrapper(nvinfer1::IRuntime *runtime, void *data,
                       size_t dataSize) {
    MTRT_DBGF("deserializing TensorRT engine from %p size %lu", data, dataSize);
    engine = runtime->deserializeCudaEngine(data, dataSize);
    MTRT_DBGF("constructed cuda engine at %p", static_cast<void *>(engine));
  }

  nvinfer1::ICudaEngine *engine{nullptr};
};

//===----------------------------------------------------------------------===//
// PinnedMemoryBuffer
//===----------------------------------------------------------------------===//

/// A simple RAII wrapper around allocating pinned host memory.
class PinnedMemoryBuffer {
public:
  PinnedMemoryBuffer(const PinnedMemoryBuffer &) = delete;
  PinnedMemoryBuffer &operator=(const PinnedMemoryBuffer &) = delete;

  static std::unique_ptr<PinnedMemoryBuffer> create(size_t size) {
    void *data{nullptr};
    cudaHostAlloc(&data, size, cudaHostAllocDefault);
    return std::unique_ptr<PinnedMemoryBuffer>(
        new PinnedMemoryBuffer(data, size));
  }

  size_t getSize() const { return size; }
  void *getData() const { return data; }

  ~PinnedMemoryBuffer() {
    if (data) {
      cudaFree(data);
      data = nullptr;
    }
  }

private:
  PinnedMemoryBuffer(void *data, size_t size) : data(data), size(size) {}

  void *data{nullptr};
  size_t size{0};
};

//===----------------------------------------------------------------------===//
// NvInferExecutionContext
//===----------------------------------------------------------------------===//

/// A simple wrapper around `nvinfer1::IExecutionContext`. This is the main
/// interface for execution. It rains a single loaded TRT engine as well as
/// an Execution context (although we could expand this to allow multiple
/// contexts all associated with a single engine).
class NvInferExecutionContext {
public:
  NvInferExecutionContext(const NvInferExecutionContext &) = delete;
  NvInferExecutionContext &operator=(const NvInferExecutionContext &) = delete;

  nvinfer1::IExecutionContext *operator*() { return context; }
  nvinfer1::IExecutionContext *operator->() { return context; }

  /// Construct a new execution context by deserializing the given TensorRT
  /// engine buffer.
  static std::unique_ptr<NvInferExecutionContext>
  create(nvinfer1::IRuntime *runtime, void *data, size_t dataSize) {
    MTRT_DBGF("creating TensorRT engine and execution context from %p", data);
    std::unique_ptr<NvInferEngineWrapper> engine =
        NvInferEngineWrapper::create(runtime, data, dataSize);
    if (!engine)
      return nullptr;

    nvinfer1::IExecutionContext *context = (*engine)->createExecutionContext();
    if (!context)
      return nullptr;

    MTRT_DBGF("created TensorRT engine context at %p", (void *)context);

    std::vector<std::unique_ptr<PinnedMemoryBuffer>> hostIOBuffers;
    std::vector<std::string> ioNames;
    std::vector<nvinfer1::Dims> ioShapes;
    std::vector<void *> ioAddresses;

    const int32_t numIOTensors = (*engine)->getNbIOTensors();
    for (int32_t ioIdx = 0; ioIdx < numIOTensors; ++ioIdx) {
      auto const &name = (*engine)->getIOTensorName(ioIdx);
      nvinfer1::TensorLocation location = (*engine)->getTensorLocation(name);
      nvinfer1::Dims dims = (*engine)->getTensorShape(name);
      ioNames.push_back(name);
      ioShapes.push_back(dims);
      ioAddresses.push_back(nullptr);

      if (location != nvinfer1::TensorLocation::kHOST)
        continue;

      assert((*engine)->isShapeInferenceIO(name) &&
             "expected host tensor to be shape tensor");
      // We expect the host shape tensor to be a 0-d or 1-d tensor.
      assert((*engine)->getTensorDataType(name) == nvinfer1::DataType::kINT32 &&
             "expected i32 data type for shape");

      // Create the pinned host buffer. Minimum allocation is 16 bytes. These
      // allocations are long-lived for the duration of the program to avoid
      // allocating host staging buffer during execution. They are automatically
      // cleaned up when the allocator destructs at program exit.
      int64_t volume = 1;
      for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0)
          return nullptr;
        volume *= dims.d[i];
      }

      std::unique_ptr<PinnedMemoryBuffer> hostBuffer =
          PinnedMemoryBuffer::create(sizeof(int32_t) *
                                     std::max<int32_t>(volume, 16));
      if (!hostBuffer)
        return nullptr;
      hostIOBuffers.push_back(std::move(hostBuffer));
      ioAddresses.back() = hostIOBuffers.back()->getData();
    }

    return std::unique_ptr<NvInferExecutionContext>(new NvInferExecutionContext(
        std::move(engine), context, std::move(hostIOBuffers),
        std::move(ioAddresses), std::move(ioShapes), std::move(ioNames)));
  }

  /// Construct a new execution context by reading and deserializing the given
  /// TensorRT engine file.
  static std::unique_ptr<NvInferExecutionContext>
  createFromFile(nvinfer1::IRuntime *runtime, const char *filename,
                 size_t filenameSize) {
    std::vector<char> fileData;
    if (readInputFile(std::string(filename, filenameSize), fileData))
      return nullptr;
    return create(runtime, fileData.data(), fileData.size());
  }

  ~NvInferExecutionContext() {
    MTRT_DBGF("freeing TensorRT execution context @ %p",
              static_cast<void *>(context));
    ::delete context;
  }

  /// Updates the internal argument shape (pointers, shape, etc).
  ///
  /// The generated code will call 'mtrt_tensorrt_enqueue' and pass a
  /// stack-allocated buffer of pointers to descriptors for each argument. There
  /// are two levels of indirection, so here is a diagram to visualize:
  ///
  /// ```
  /// args = [ptr1, ptr2, ptr3, ..., ptrN]
  ///
  /// *ptr1 = struct{ int64_t rank, void* dataPtr, shape... }
  /// ````
  ///
  /// So each pointer in `args` points to a descriptor whose first
  /// element is the rank, second the pointer to the buffer, and finally
  /// `rank` int64 values describing the shape.
  ///
  /// This is obviously required implement a simple variadic calling
  /// interface over the C boundary.
  int setTensorAddressesOrReport(void **inputArgs, unsigned numInputArgs,
                                 void **outputArgs, unsigned numOutputArgs);

private:
  NvInferExecutionContext(
      std::unique_ptr<NvInferEngineWrapper> engine,
      nvinfer1::IExecutionContext *ctx,
      std::vector<std::unique_ptr<PinnedMemoryBuffer>> hostStagingBuffers,
      std::vector<void *> ioAddresses, std::vector<nvinfer1::Dims> ioShapes,
      std::vector<std::string> ioNames)
      : engine(std::move(engine)), context(ctx),
        hostStagingBuffers(std::move(hostStagingBuffers)),
        ioAddresses(std::move(ioAddresses)), ioShapes(std::move(ioShapes)),
        ioNames(std::move(ioNames)) {}

  std::unique_ptr<NvInferEngineWrapper> engine;
  nvinfer1::IExecutionContext *context;

  /// Host staging buffers where required (for shape/host input tensors).
  std::vector<std::unique_ptr<PinnedMemoryBuffer>> hostStagingBuffers;

  /// Tensor address information which will be passed to the inference execution
  /// command. This is allocated up from and ptrs are updated each inference.
  std::vector<void *> ioAddresses;
  std::vector<nvinfer1::Dims> ioShapes;
  std::vector<std::string> ioNames;
};
} // namespace

/// Given a ranked descriptor, populate the ioAddress and ioShape fields.
template <uint32_t Rank>
void populateArgumentPack(const MemRefDescriptor<Rank> &desc, void *&ioAddress,
                          nvinfer1::Dims &ioShape) {
  ioAddress = desc.data;
  // 0-d descriptor doesn't have shape, so guard this with constexpr.
  if constexpr (Rank > 0) {
    for (uint32_t i = 0; i < Rank; ++i) {
      ioShape.d[i] = desc.shape[i];
      ioShape.nbDims = desc.rank;
    }
  }
}

/// Dynamically dispatch over the parsing of the ranked descriptor:
/// *descPtr = struct{ int64_t rank, void* dataPtr, shape... }
static int dispatchPopulateArgumentPack(const void *descPtr, void *&ioAddress,
                                        nvinfer1::Dims &ioShape) {
  int64_t rank = static_cast<const int64_t *>(descPtr)[0];
#define HANDLE_CASE(rank)                                                      \
  case rank: {                                                                 \
    const auto *desc = static_cast<const MemRefDescriptor<rank> *>(descPtr);   \
    populateArgumentPack(*desc, ioAddress, ioShape);                           \
    break;                                                                     \
  }
  switch (rank) {
    HANDLE_CASE(0)
    HANDLE_CASE(1)
    HANDLE_CASE(2)
    HANDLE_CASE(3)
    HANDLE_CASE(4)
    HANDLE_CASE(5)
    HANDLE_CASE(6)
    HANDLE_CASE(7)
    HANDLE_CASE(8)
  default:
    return 1;
  }
#undef HANDLE_CASE
  return 0;
}

int NvInferExecutionContext::setTensorAddressesOrReport(
    void **inputArgs, unsigned numInputArgs, void **outputArgs,
    unsigned numOutputArgs) {
  const unsigned numTotalArgs = numInputArgs + numOutputArgs;
  for (uint32_t i = 0; i < numTotalArgs; ++i) {
    void *descPtr =
        i < numInputArgs ? inputArgs[i] : outputArgs[i - numInputArgs];

    void *ioAddr{nullptr};
    nvinfer1::Dims ioShape{};
    dispatchPopulateArgumentPack(descPtr, ioAddr, ioShape);

    const char *ioName = ioNames[i].c_str();
    bool result = context->setTensorAddress(ioName, ioAddr);
    if (!result) {
      std::stringstream ss;
      const nvinfer1::ICudaEngine &engine = context->getEngine();
      ss << "Failed to set tensor address for IO tensor: " << ioName
         << " at position " << i << "; the IO tensors are:\n";
      for (int64_t i = 0; i < engine.getNbIOTensors(); i++) {
        const char *name = engine.getIOTensorName(i);
        ss << "[" << i << "] " << name << " : ";
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
          ss << "input";
        else
          ss << "output";
        ss << "\n";
      }
      std::cerr << ss.str();
      return 1;
    }

    if (i < numInputArgs) {
      bool result = context->setInputShape(ioName, ioShape);
      if (!result)
        return 1;
    }

    MTRT_DBGF("Set tensor address [%d] = 0x%lx", i,
              reinterpret_cast<uintptr_t>(ioAddr));
  }
  return 0;
}

static int enqueueV3Wrapper(NvInferExecutionContext &context, CUstream stream,
                            void **inputDescriptors, unsigned numInputs,
                            void **outputDescriptors, unsigned numOutputs) {
  int code = context.setTensorAddressesOrReport(inputDescriptors, numInputs,
                                                outputDescriptors, numOutputs);
  if (code != 0)
    return code;
  // Create an event that we can wait on for releasing any host-pinned staging
  // allocations we made.
  CUevent event;
  cudaEventCreate(&event);
  if (!context->setInputConsumedEvent(event))
    return 1;

  if (!context->enqueueV3(stream))
    return 1;
  cudaError_t waitResult = cudaStreamWaitEvent(stream, event);
  if (waitResult != cudaSuccess)
    return 1;

  return 0;
}

//===----------------------------------------------------------------------===//
// C API Method Implementations
//===----------------------------------------------------------------------===//

extern "C" {
void *mtrt_tensorrt_runtime_create() {
  static StdioLogger defaultLogger;
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(defaultLogger);
  MTRT_DBGF("created TensorRT Runtime at 0x%lx",
            reinterpret_cast<uintptr_t>(runtime));
  return runtime;
}

void mtrt_tensorrt_runtime_destroy(void *nvinferRuntime) {
  MTRT_DBGF("freeing TensorRT Runtime at 0x%lx",
            reinterpret_cast<uintptr_t>(nvinferRuntime));
  ::delete static_cast<nvinfer1::IRuntime *>(nvinferRuntime);
}

void mtrt_tensorrt_enqueue(void *executionContext, CUstream stream,
                           int32_t numInputs, void *inputDescriptors,
                           int32_t numOutputs, void *outputDescriptors) {
  enqueueV3Wrapper(*static_cast<NvInferExecutionContext *>(executionContext),
                   stream, static_cast<void **>(inputDescriptors), numInputs,
                   static_cast<void **>(outputDescriptors), numOutputs);
}

void mtrt_tensorrt_enqueue_alloc(void *executionContext, CUstream stream,
                                 int32_t numInputs, void *inputDescriptors,
                                 int32_t numOutputs, void *outputDescriptors) {}

void *mtrt_load_tensorrt_engine(void *runtime, void *data, size_t dataSize) {
  assert(runtime && "expected valid runtime");
  auto engine = NvInferExecutionContext::create(
      static_cast<nvinfer1::IRuntime *>(runtime), data, dataSize);
  return engine.release();
}

void *mtrt_load_tensorrt_engine_from_file(void *runtime, const char *filename,
                                          size_t filenameSize) {
  auto engine = NvInferExecutionContext::createFromFile(
      static_cast<nvinfer1::IRuntime *>(runtime), filename, filenameSize);
  return engine.release();
}

void mtrt_tensorrt_execution_context_destroy(void *executionContext) {
  ::delete static_cast<NvInferExecutionContext *>(executionContext);
}
}
