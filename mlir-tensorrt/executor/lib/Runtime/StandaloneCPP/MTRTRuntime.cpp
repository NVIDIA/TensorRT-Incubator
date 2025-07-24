//===- MTRTRuntime.cpp ----------------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This file contains an example implementation of C++ functions required
/// to interact with generated C++ host code.
///
//===----------------------------------------------------------------------===//
#include "MTRTRuntime.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mtrt;

#define MTRT_DBGF(fmt, ...)                                                    \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      std::fprintf(stderr, "%s:%d:%s(): " fmt "\n", "MTRTRuntime.cpp",         \
                   __LINE__, __func__, __VA_ARGS__);                           \
  } while (0)

#define HANDLE_CUDADRV_ERROR(x, ...)                                           \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      std::fprintf(stderr, "%s:%d:%s(): CUDA Driver Error: %s\n",              \
                   "MTRTRuntime.cpp", __LINE__, __func__, msg);                \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#define HANDLE_CUDART_ERROR(x, ...)                                            \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      const char *msg = "";                                                    \
      msg = cudaGetErrorString(err);                                           \
      std::fprintf(stderr, "%s:%d:%s(): CUDA Runtime Error: %s\n",             \
                   "MTRTRuntime.cpp", __LINE__, __func__, msg);                \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

static const char *kDebugEnvironmentVariable = "MTRT_DEBUG";

/// Helper method that checks environment value for debugging.
static bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static size_t getFileSize(const std::string &filename) {
  // Open the binary file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 0;
  }
  return file.tellg();
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

static int readInputFile(const std::string &filename, char *buffer) {
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

  // Read the entire file into the vector
  if (file.read(buffer, size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

//===----------------------------------------------------------------------===//
// CUDA Wrappers
//===----------------------------------------------------------------------===//

CUmodule mtrt::cuda_module_create_from_ptx_file(const char *filename) {
  CUmodule module;
  std::vector<char> buffer;
  if (readInputFile(filename, buffer))
    return nullptr;
  buffer.push_back('\0');
  HANDLE_CUDADRV_ERROR(
      cuModuleLoadDataEx(&module, buffer.data(), 0, nullptr, nullptr), nullptr);
  return module;
}

void mtrt::cuda_module_destroy(CUmodule module) {
  HANDLE_CUDADRV_ERROR(cuModuleUnload(module), );
}

CUfunction mtrt::cuda_module_get_func(CUmodule module, const char *name) {
  CUfunction func;
  HANDLE_CUDADRV_ERROR(cuModuleGetFunction(&func, module, name), nullptr);
  return func;
}

void mtrt::cuda_launch_kernel(CUfunction func, int32_t gridX, int32_t gridY,
                              int32_t gridZ, int32_t blockX, int32_t blockY,
                              int32_t blockZ, int32_t dynamicSharedMemoryBytes,
                              CUstream stream, void **arguments) {
  HANDLE_CUDADRV_ERROR(cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY,
                                      blockZ, dynamicSharedMemoryBytes, stream,
                                      arguments, nullptr), );
}

void mtrt::cuda_stream_sync(CUstream stream) {
  HANDLE_CUDART_ERROR(cudaStreamSynchronize(stream), );
}

/// Return the current CUDA device.
int32_t mtrt::cuda_get_current_device() {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), 0);
  return device;
}

/// Perform a CUDA allocation.
void *mtrt::cuda_alloc(CUstream stream, int64_t size, bool isHostPinned,
                       bool isManaged) {
  void *result{nullptr};
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaMallocManaged(&result, size), nullptr);
    return result;
  }
  HANDLE_CUDART_ERROR(cudaMallocAsync(&result, size, stream), nullptr);
  return result;
}

void mtrt::cuda_free(CUstream stream, void *ptr, int8_t isHostPinned,
                     int8_t isManaged) {
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaFree(ptr), );
    return;
  }
  HANDLE_CUDART_ERROR(cudaFreeAsync(ptr, stream), );
}

void mtrt::cuda_copy(CUstream stream, void *src, void *dest, int64_t size) {
  HANDLE_CUDART_ERROR(
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream), );
}

//===----------------------------------------------------------------------===//
// TensorRT Wrappers
//===----------------------------------------------------------------------===//

/// Given a ranked descriptor, populate the ioAddress and ioShape fields.
template <uint32_t Rank>
void populateArgumentPack(const PtrAndShape<Rank> &desc, void *&ioAddress,
                          nvinfer1::Dims &ioShape) {
  ioAddress = desc.bufferStart;
  // 0-d descriptor doesn't have shape, so guard this with constexpr.
  if constexpr (Rank > 0) {
    for (uint32_t i = 0; i < Rank; ++i) {
      ioShape.d[i] = desc.shape[i];
      ioShape.nbDims = Rank;
    }
  }
}

/// Dynamically dispatch over the parsing of the ranked descriptor:
/// desc = struct{ int64_t rank, void* rankedDescriptor }
static int dispatchPopulateArgumentPack(const UnrankedMemRef &desc,
                                        void *&ioAddress,
                                        nvinfer1::Dims &ioShape) {
  int64_t rank = desc.rank;
#define HANDLE_CASE(rank)                                                      \
  case rank: {                                                                 \
    const auto *ranked =                                                       \
        static_cast<const PtrAndShape<rank> *>(desc.rankedDescriptor);         \
    populateArgumentPack(*ranked, ioAddress, ioShape);                         \
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

/// Enqueue the execution of the TRT function with the given inputs/outputs onto
/// a stream.
void mtrt::tensorrt_enqueue(nvinfer1::IExecutionContext *context,
                            CUstream stream, int32_t numInputArgs,
                            UnrankedMemRef *inputs, int32_t numOutputArgs,
                            UnrankedMemRef *outputs) {
  const int32_t numTotalArgs = numInputArgs + numOutputArgs;
  for (int32_t i = 0; i < numTotalArgs; ++i) {
    const UnrankedMemRef &desc =
        i < numInputArgs ? inputs[i] : outputs[i - numInputArgs];

    // Extract shape and ptr from the descriptor.
    void *ioAddr{nullptr};
    nvinfer1::Dims ioShape{};
    if (dispatchPopulateArgumentPack(desc, ioAddr, ioShape)) {
      std::cerr << "failed to parse TensorRT enqueue parameter pack";
      return;
    }

    // Unfortunately TensorRT likes to address arguments by name. We use the
    // following convention in the compiler to generate names so that we don't
    // have to look them up. This is needlessly expensive, so we could also
    // pre-compute these names in static table if you know max number arguments.
    // Retaining names and orderings as metadata would require constructing an
    // alternative wrapper object around the execution context, which is another
    // option.
    std::string ioName = i < numInputArgs
                             ? "arg" + std::to_string(i)
                             : "result" + std::to_string(i - numInputArgs);
    bool result = context->setTensorAddress(ioName.c_str(), ioAddr);
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
      return;
    }

    if (i < numInputArgs) {
      bool result = context->setInputShape(ioName.c_str(), ioShape);
      if (!result) {
        std::cerr << "failed to set input tensor shape for input: " << ioName
                  << "\n";
        return;
      }
    }

    MTRT_DBGF("Set tensor address [%d] = %p", i, ioAddr);
  }
  context->enqueueV3(stream);
}

/// Load a TensorRT engine from a serialized plan file.
nvinfer1::ICudaEngine *
mtrt::tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                       const char *filename) {
  std::vector<char> data;
  if (readInputFile(filename, data))
    return nullptr;
  return runtime->deserializeCudaEngine(data.data(), data.size());
}

/// Destroy a TensorRT engine.
void mtrt::tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine) {
  ::delete engine;
}

/// Construct an execution context.
nvinfer1::IExecutionContext *
mtrt::tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine) {
  return engine->createExecutionContext();
}

/// Destroy an execution context.
void mtrt::tensorrt_execution_context_destroy(
    nvinfer1::IExecutionContext *context) {
  ::delete context;
}

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

void *mtrt::host_alloc(int64_t size, int32_t alignment) {
  if (size % alignment != 0)
    size = ((size + alignment - 1) / alignment) * alignment;
  return std::aligned_alloc(size, alignment);
}

void mtrt::host_free(void *ptr) { ::free(ptr); }

void *mtrt::constant_load_from_file(const char *filename, int32_t align,
                                    int32_t space) {

  size_t fileSize = getFileSize(filename);
  if (fileSize == 0)
    return nullptr;
  void *buffer;
  HANDLE_CUDART_ERROR(cudaMallocManaged(&buffer, fileSize), nullptr);
  if (!readInputFile(filename, reinterpret_cast<char *>(buffer)))
    return nullptr;
  return buffer;
}

void mtrt::constant_destroy(void *data, int32_t space) {
  HANDLE_CUDART_ERROR(cudaFree(data), );
}