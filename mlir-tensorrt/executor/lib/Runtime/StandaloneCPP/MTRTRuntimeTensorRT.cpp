//===- MTRTRuntimeTensorRT.cpp --------------------------------------------===//
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
/// \file
/// TensorRT wrapper implementations for generated EmitC C++ host code.
//===----------------------------------------------------------------------===//

#include "MTRTRuntimeTensorRT.h"
#include "MTRTRuntimeCuda.h"
#include "MTRTRuntimeStatus.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mtrt;

static const char *kDebugEnvironmentVariable = "MTRT_DEBUG";

/// Helper method that checks environment value for debugging.
static bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

#define MTRT_DBGF(fmt, ...)                                                    \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      std::fprintf(stderr, "%s:%d:%s(): " fmt "\n", "MTRTRuntimeTensorRT.cpp", \
                   __LINE__, __func__, __VA_ARGS__);                           \
  } while (0)

namespace mtrt::detail {
Status readInputFile(const std::string &filename, std::vector<char> &buffer);
} // namespace mtrt::detail

/// Given a ranked descriptor, populate the ioAddress and ioShape fields.
template <uint32_t Rank>
static void populateArgumentPack(const PtrAndShape<Rank> &desc,
                                 void *&ioAddress, nvinfer1::Dims &ioShape) {
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
static Status dispatchPopulateArgumentPack(const UnrankedMemRef &desc,
                                           void *&ioAddress,
                                           nvinfer1::Dims &ioShape) {
  int64_t rank = desc.rank;
#define HANDLE_CASE(R)                                                         \
  case R: {                                                                    \
    const auto *ranked =                                                       \
        static_cast<const PtrAndShape<R> *>(desc.rankedDescriptor);            \
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
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "unsupported TensorRT arg rank %lld",
                      static_cast<long long>(rank));
  }
#undef HANDLE_CASE
  return mtrt::ok();
}

Status mtrt::tensorrt_enqueue(nvinfer1::IExecutionContext *context,
                              CUstream stream, int32_t numInputArgs,
                              UnrankedMemRef *inputs, int32_t numOutputArgs,
                              UnrankedMemRef *outputs) {
  if (!context)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "context must not be null");
  if (numInputArgs < 0 || numOutputArgs < 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "numInputs/numOutputs must be >= 0");
  const int32_t numTotalArgs = numInputArgs + numOutputArgs;
  for (int32_t i = 0; i < numTotalArgs; ++i) {
    const UnrankedMemRef &desc =
        i < numInputArgs ? inputs[i] : outputs[i - numInputArgs];

    // Extract shape and ptr from the descriptor.
    void *ioAddr{nullptr};
    nvinfer1::Dims ioShape{};
    Status st = dispatchPopulateArgumentPack(desc, ioAddr, ioShape);
    if (st != mtrt::ok())
      return st;
    assert(ioAddr && "ioAddr must not be null");

    // Unfortunately TensorRT likes to address arguments by name. We use the
    // following convention in the compiler to generate names so that we don't
    // have to look them up.
    std::string ioName = i < numInputArgs
                             ? "arg" + std::to_string(i)
                             : "result" + std::to_string(i - numInputArgs);
    bool result = context->setTensorAddress(ioName.c_str(), ioAddr);
    if (!result) {
      const nvinfer1::ICudaEngine &engine = context->getEngine();
      // Keep the message short but actionable (full IO listing can be printed
      // by the caller if desired).
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT setTensorAddress failed for '%s' (index=%d, "
                        "nbIOTensors=%d)",
                        ioName.c_str(), i, engine.getNbIOTensors());
    }

    if (i < numInputArgs) {
      bool ok = context->setInputShape(ioName.c_str(), ioShape);
      if (!ok) {
        MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                          "TensorRT setInputShape failed for '%s'",
                          ioName.c_str());
      }
    }

    MTRT_DBGF("Set tensor address [%d] = %p", i, ioAddr);
  }
  if (!context->enqueueV3(stream))
    MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                      "TensorRT enqueueV3 failed");
  return mtrt::ok();
}

template <uint32_t Rank>
static Status fillRankedMemRefDescriptor(RankedMemRef<Rank> &out, void *ptr,
                                         const nvinfer1::Dims &shape,
                                         const nvinfer1::Dims &strides) {
  out.allocated = ptr;
  out.aligned = ptr;
  out.offset = 0;
  if constexpr (Rank > 0) {
    for (uint32_t i = 0; i < Rank; ++i) {
      out.shape[i] = static_cast<int64_t>(shape.d[i]);
      out.strides[i] = static_cast<int64_t>(strides.d[i]);
    }
  }
  return mtrt::ok();
}

static uint64_t roundUpTo(uint64_t m, uint64_t n) {
  if (n == 0)
    return m;
  uint64_t r = m % n;
  return r == 0 ? m : (m + (n - r));
}

namespace {
class NvInferResultAllocator : public nvinfer1::IOutputAllocator {
public:
  NvInferResultAllocator(nvinfer1::ICudaEngine const &engine, int32_t resultIdx,
                         CUstream enqueueStream)
      : enqueueStream(enqueueStream) {
    tensorName = "result" + std::to_string(resultIdx);
    // Default: allocate on device; use pinned host only for host IO tensors.
    isHostPinned = engine.getTensorLocation(tensorName.c_str()) ==
                   nvinfer1::TensorLocation::kHOST;
  }

  NvInferResultAllocator(const NvInferResultAllocator &) = delete;
  NvInferResultAllocator &operator=(const NvInferResultAllocator &) = delete;
  NvInferResultAllocator(NvInferResultAllocator &&) = delete;
  NvInferResultAllocator &operator=(NvInferResultAllocator &&) = delete;

  void *reallocateOutputAsync(const char *name, void *currentMemory,
                              uint64_t size, uint64_t alignment,
                              cudaStream_t stream) override {
    (void)currentMemory;
    if (!name || tensorName != name) {
      hadError = true;
      mtrt::detail::set_last_error_message(
          "%s:%d:%s(): TensorRT output allocator tensor name mismatch",
          __FILE__, __LINE__, __func__);
      return nullptr;
    }

    // TensorRT requires a non-null ptr even for empty tensors.
    size = std::max<uint64_t>(size, 1);
    size = roundUpTo(size, std::max<uint64_t>(alignment, 1));

    // If we already have a sufficiently large allocation, reuse it.
    if (allocatedPtr && size <= allocatedSizeBytes)
      return allocatedPtr;

    // Best-effort: cudaMalloc/cudaMallocAsync are typically >=256B aligned.
    // If TensorRT requests higher alignment, we still round the allocation size
    // up, but we do not over-allocate and manually align the returned pointer.
    if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      hadError = true;
      mtrt::detail::set_last_error_message(
          "%s:%d:%s(): TensorRT requested output allocation too large: %llu",
          __FILE__, __LINE__, __func__, static_cast<unsigned long long>(size));
      return nullptr;
    }

    // Free any previous allocation made by this allocator instance.
    if (allocatedPtr) {
      CUstream freeStream = enqueueStream;
      if (stream)
        freeStream = reinterpret_cast<CUstream>(stream);
      (void)mtrt::cuda_free(freeStream, allocatedPtr,
                            static_cast<int8_t>(isHostPinned),
                            /*isManaged=*/0);
      allocatedPtr = nullptr;
      allocatedSizeBytes = 0;
    }

    void *ptr = nullptr;
    CUstream allocStream = enqueueStream;
    if (stream)
      allocStream = reinterpret_cast<CUstream>(stream);
    Status st = mtrt::cuda_alloc(allocStream, static_cast<int64_t>(size),
                                 isHostPinned, /*isManaged=*/false, &ptr);
    if (st != mtrt::ok()) {
      hadError = true;
      return nullptr;
    }
    allocatedPtr = ptr;
    allocatedSizeBytes = size;
    return ptr;
  }

  void notifyShape(const char *name,
                   const nvinfer1::Dims &dims) noexcept override {
    if (!name || tensorName != name) {
      hadError = true;
      return;
    }
    outputDims = dims;
  }

  const std::string &getTensorName() const { return tensorName; }
  const nvinfer1::Dims &getOutputDims() const { return outputDims; }
  void *getAllocatedPtr() const { return allocatedPtr; }
  bool getHadError() const { return hadError; }

private:
  CUstream enqueueStream{nullptr};
  std::string tensorName;
  bool isHostPinned{false};
  void *allocatedPtr{nullptr};
  uint64_t allocatedSizeBytes{0};
  nvinfer1::Dims outputDims{};
  bool hadError{false};
};
} // namespace

/// Enqueue TensorRT execution with dynamic output allocation via
/// `nvinfer1::IOutputAllocator`, then populate the provided ranked memref
/// descriptors.
Status mtrt::tensorrt_enqueue_alloc(nvinfer1::IExecutionContext *context,
                                    CUstream stream, int32_t numInputArgs,
                                    UnrankedMemRef *inputs,
                                    int32_t numOutputArgs,
                                    UnrankedMemRefMut *outputs) {
  if (!context)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "context must not be null");
  if (numInputArgs < 0 || numOutputArgs < 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "numInputs/numOutputs must be >= 0");

  // Set input addresses and shapes.
  for (int32_t i = 0; i < numInputArgs; ++i) {
    const UnrankedMemRef &desc = inputs[i];
    void *ioAddr{nullptr};
    nvinfer1::Dims ioShape{};
    Status st = dispatchPopulateArgumentPack(desc, ioAddr, ioShape);
    if (st != mtrt::ok())
      return st;

    std::string ioName = "arg" + std::to_string(i);
    if (!context->setTensorAddress(ioName.c_str(), ioAddr))
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT setTensorAddress failed for '%s'",
                        ioName.c_str());
    if (!context->setInputShape(ioName.c_str(), ioShape))
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT setInputShape failed for '%s'",
                        ioName.c_str());
  }

  const nvinfer1::ICudaEngine &engine = context->getEngine();

  // Set output allocators for dynamic output allocation. We use one allocator
  // object per output tensor, matching TensorRT's name-based API.
  std::vector<std::unique_ptr<NvInferResultAllocator>> allocators;
  allocators.reserve(static_cast<size_t>(numOutputArgs));
  for (int32_t i = 0; i < numOutputArgs; ++i) {
    allocators.push_back(
        std::make_unique<NvInferResultAllocator>(engine, i, stream));
    std::string name = allocators.back()->getTensorName();
    // Clear any previous binding; TensorRT will call the allocator to provide
    // the buffer.
    (void)context->setTensorAddress(name.c_str(), nullptr);
    if (!context->setOutputAllocator(name.c_str(), allocators.back().get()))
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT setOutputAllocator failed for '%s'",
                        name.c_str());
  }

  bool enqOk = context->enqueueV3(stream);

  // Always clear allocators after enqueueV3 (success or failure) to avoid
  // leaving dangling pointers in the execution context.
  for (int32_t i = 0; i < numOutputArgs; ++i) {
    std::string name = "result" + std::to_string(i);
    (void)context->setOutputAllocator(name.c_str(), nullptr);
  }

  if (!enqOk)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                      "TensorRT enqueueV3 failed");

  // Populate ranked output memref descriptors using the allocator-provided ptr
  // and TensorRT-reported shapes/strides.
  for (int32_t i = 0; i < numOutputArgs; ++i) {
    NvInferResultAllocator *alloc = allocators[i].get();
    if (alloc->getHadError())
      return mtrt::make_status(mtrt::ErrorCode::TensorRTError);

    UnrankedMemRefMut &outDesc = outputs[i];
    const int64_t rank = outDesc.rank;
    if (rank < 0 || rank > 8)
      MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                        "unsupported TensorRT output rank %lld",
                        static_cast<long long>(rank));

    const nvinfer1::Dims &shape = alloc->getOutputDims();
    if (shape.nbDims != static_cast<int32_t>(rank))
      MTRT_RETURN_ERROR(
          mtrt::ErrorCode::TensorRTError,
          "TensorRT output '%s' rank mismatch: expected %lld, got "
          "%d",
          alloc->getTensorName().c_str(), static_cast<long long>(rank),
          shape.nbDims);

    for (int32_t d = 0; d < shape.nbDims; ++d)
      if (shape.d[d] < 0)
        MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                          "TensorRT output '%s' has dynamic dim at %d",
                          alloc->getTensorName().c_str(), d);

    nvinfer1::Dims strides =
        context->getTensorStrides(alloc->getTensorName().c_str());
    if (strides.nbDims != static_cast<int32_t>(rank))
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT output '%s' stride rank mismatch: expected "
                        "%lld, got %d",
                        alloc->getTensorName().c_str(),
                        static_cast<long long>(rank), strides.nbDims);

    void *ptr = alloc->getAllocatedPtr();
    if (!ptr)
      MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                        "TensorRT output allocator returned null ptr for '%s'",
                        alloc->getTensorName().c_str());

    switch (rank) {
#define HANDLE_CASE(R)                                                         \
  case R: {                                                                    \
    auto *ranked = static_cast<RankedMemRef<R> *>(outDesc.rankedDescriptor);   \
    if (!ranked)                                                               \
      MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,                      \
                        "null rankedDescriptor for output %d", i);             \
    Status st2 = fillRankedMemRefDescriptor<R>(*ranked, ptr, shape, strides);  \
    if (st2 != mtrt::ok())                                                     \
      return st2;                                                              \
    break;                                                                     \
  }
      HANDLE_CASE(0)
      HANDLE_CASE(1)
      HANDLE_CASE(2)
      HANDLE_CASE(3)
      HANDLE_CASE(4)
      HANDLE_CASE(5)
      HANDLE_CASE(6)
      HANDLE_CASE(7)
      HANDLE_CASE(8)
#undef HANDLE_CASE
    default:
      MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                        "unsupported TensorRT output rank %lld",
                        static_cast<long long>(rank));
    }
  }

  return mtrt::ok();
}

Status
mtrt::tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                       const char *filename,
                                       nvinfer1::ICudaEngine **outEngine) {
  if (!outEngine)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outEngine must not be null");
  *outEngine = nullptr;
  if (!runtime)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "runtime must not be null");
  if (!filename || filename[0] == '\0')
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "filename must not be empty");
  std::vector<char> data;
  Status st = detail::readInputFile(filename, data);
  if (st != mtrt::ok())
    return st;
  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(data.data(), data.size());
  if (!engine)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                      "deserializeCudaEngine failed for '%s'", filename);
  *outEngine = engine;
  return mtrt::ok();
}

void mtrt::tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine) {
  ::delete engine;
}

Status
mtrt::tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine,
                                        nvinfer1::IExecutionContext **outCtx) {
  if (!outCtx)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outCtx must not be null");
  *outCtx = nullptr;
  if (!engine)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "engine must not be null");
  nvinfer1::IExecutionContext *ctx = engine->createExecutionContext();
  if (!ctx)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::TensorRTError,
                      "createExecutionContext failed");
  *outCtx = ctx;
  return mtrt::ok();
}

void mtrt::tensorrt_execution_context_destroy(
    nvinfer1::IExecutionContext *context) {
  ::delete context;
}
