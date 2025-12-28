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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
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

static int readInputFile(const std::string &filename,
                         std::vector<char> &buffer) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  buffer.resize(size);
  if (file.read(buffer.data(), size))
    return 0;
  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

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
static int dispatchPopulateArgumentPack(const UnrankedMemRef &desc,
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
    return 1;
  }
#undef HANDLE_CASE
  return 0;
}

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
    // have to look them up.
    std::string ioName = i < numInputArgs
                             ? "arg" + std::to_string(i)
                             : "result" + std::to_string(i - numInputArgs);
    bool result = context->setTensorAddress(ioName.c_str(), ioAddr);
    if (!result) {
      std::stringstream ss;
      const nvinfer1::ICudaEngine &engine = context->getEngine();
      ss << "Failed to set tensor address for IO tensor: " << ioName
         << " at position " << i << "; the IO tensors are:\n";
      for (int64_t j = 0; j < engine.getNbIOTensors(); j++) {
        const char *name = engine.getIOTensorName(j);
        ss << "[" << j << "] " << name << " : ";
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
      bool ok = context->setInputShape(ioName.c_str(), ioShape);
      if (!ok) {
        std::cerr << "failed to set input tensor shape for input: " << ioName
                  << "\n";
        return;
      }
    }

    MTRT_DBGF("Set tensor address [%d] = %p", i, ioAddr);
  }
  context->enqueueV3(stream);
}

nvinfer1::ICudaEngine *
mtrt::tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                       const char *filename) {
  std::vector<char> data;
  if (readInputFile(filename, data))
    return nullptr;
  return runtime->deserializeCudaEngine(data.data(), data.size());
}

void mtrt::tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine) {
  ::delete engine;
}

nvinfer1::IExecutionContext *
mtrt::tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine) {
  return engine->createExecutionContext();
}

void mtrt::tensorrt_execution_context_destroy(
    nvinfer1::IExecutionContext *context) {
  ::delete context;
}
