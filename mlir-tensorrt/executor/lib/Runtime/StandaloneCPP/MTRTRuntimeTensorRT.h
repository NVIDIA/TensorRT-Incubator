//===- MTRTRuntimeTensorRT.h ----------------------------------------------===//
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
//===- MTRTRuntimeTensorRT.h -------------------------------------*- C++
//-*-===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
/// \file
/// TensorRT wrapper declarations used by generated EmitC C++ host code.
//===----------------------------------------------------------------------===//
#ifndef MTRT_RUNTIME_TENSORRT_H
#define MTRT_RUNTIME_TENSORRT_H

#include "MTRTRuntimeCore.h"

// TensorRT types.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInferRuntime.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// CUstream type.
#include "cuda.h"

namespace mtrt {

//===----------------------------------------------------------------------===//
// TensorRT Wrappers
//===----------------------------------------------------------------------===//

/// Enqueue the execution of the TRT function with the given inputs/outputs onto
/// a stream.
/// All UnrankedMemRefs here contain pointers to descriptors of 'PtrAndShape'
/// type.
Status tensorrt_enqueue(nvinfer1::IExecutionContext *context, CUstream stream,
                        int32_t numInputs, UnrankedMemRef *inputs,
                        int32_t numOutputs, UnrankedMemRef *outputs);

/// Load a TensorRT engine from a serialized plan file.
Status tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                        const char *filename,
                                        nvinfer1::ICudaEngine **outEngine);

/// Destroy a TensorRT engine.
void tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine);

/// Construct an execution context.
Status tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine,
                                         nvinfer1::IExecutionContext **outCtx);

/// Destroy an execution context.
void tensorrt_execution_context_destroy(nvinfer1::IExecutionContext *engine);

} // namespace mtrt

#endif // MTRT_RUNTIME_TENSORRT_H
