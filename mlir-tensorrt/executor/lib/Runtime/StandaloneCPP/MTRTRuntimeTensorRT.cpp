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
#include "MTRTRuntimeStatus.h"
#include <cstdio>
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
