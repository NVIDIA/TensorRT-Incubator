//===- TensorRTModule.h -------------------------------------------*- C -*-===//
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
/// MTRT TensorRT C runtime module declarations.
///
//===----------------------------------------------------------------------===//
#ifndef RUNTIME_BACKEND_C_TENSORRTMODULE
#define RUNTIME_BACKEND_C_TENSORRTMODULE

#include "CUDAModule.h"

#ifdef __cplusplus
extern "C" {
#endif

MTRT_CAPI_EXPORTED void *mtrt_tensorrt_runtime_create();

MTRT_CAPI_EXPORTED void mtrt_tensorrt_runtime_destroy(void *runtime);

MTRT_CAPI_EXPORTED void
mtrt_tensorrt_enqueue(void *executionContext, CUstream stream,
                      int32_t numInputs, void *inputDescriptors,
                      int32_t numOutputs, void *outputDescriptors);

MTRT_CAPI_EXPORTED void
mtrt_tensorrt_enqueue_alloc(void *executionContext, CUstream stream,
                            int32_t numInputs, void *inputDescriptors,
                            int32_t numOutputs, void *outputDescriptors);

MTRT_CAPI_EXPORTED void *mtrt_load_tensorrt_engine(void *runtime, void *data,
                                                   size_t dataSize);
MTRT_CAPI_EXPORTED void *
mtrt_load_tensorrt_engine_from_file(void *runtime, const char *filename,
                                    size_t filenameSize);

MTRT_CAPI_EXPORTED void
mtrt_tensorrt_execution_context_destroy(void *executionContext);

//===----------------------------------------------------------------------===//
// EmitC-Compatible Wrappers
//
// These wrappers are used by emitted C/C++ code.
// The C/C++ generation currently uses opaque pointers like LLVM, so
// we must provide type-erased wrappers around each of the above functions.
//===----------------------------------------------------------------------===//

inline static void *mtrt_tensorrt_runtime_create_cwrapper() {
  return mtrt_tensorrt_runtime_create();
}

inline static void mtrt_tensorrt_runtime_destroy_cwrapper(void *runtime) {
  return mtrt_tensorrt_runtime_destroy(runtime);
}

inline static void
mtrt_tensorrt_enqueue_cwrapper(void *executionContext, void *stream,
                               int32_t numInputs, void *inputDescriptors,
                               int32_t numOutputs, void *outputDescriptors) {
  return mtrt_tensorrt_enqueue(executionContext, (CUstream)stream, numInputs,
                               inputDescriptors, numOutputs, outputDescriptors);
}

inline static void mtrt_tensorrt_enqueue_alloc_cwrapper(
    void *executionContext, CUstream stream, int32_t numInputs,
    void *inputDescriptors, int32_t numOutputs, void *outputDescriptors) {
  return mtrt_tensorrt_enqueue_alloc(executionContext, stream, numInputs,
                                     inputDescriptors, numOutputs,
                                     outputDescriptors);
}

inline static void *
mtrt_load_tensorrt_engine_cwrapper(void *runtime, void *data, size_t dataSize) {
  return mtrt_load_tensorrt_engine(runtime, data, dataSize);
}

inline static void *
mtrt_load_tensorrt_engine_from_file_cwrapper(void *runtime, void *filename,
                                             size_t filenameSize) {
  return mtrt_load_tensorrt_engine_from_file(runtime, (const char *)filename,
                                             filenameSize);
}

inline static void
mtrt_tensorrt_execution_context_destroy_cwrapper(void *executionContext) {
  return mtrt_tensorrt_execution_context_destroy(executionContext);
}

#ifdef __cplusplus
}
#endif

#endif // RUNTIME_BACKEND_C_TENSORRTMODULE
