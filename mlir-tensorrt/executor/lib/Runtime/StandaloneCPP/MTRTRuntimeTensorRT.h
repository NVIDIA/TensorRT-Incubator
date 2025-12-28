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
void tensorrt_enqueue(nvinfer1::IExecutionContext *context, CUstream stream,
                      int32_t numInputs, UnrankedMemRef *inputs,
                      int32_t numOutputs, UnrankedMemRef *outputs);

/// Load a TensorRT engine from a serialized plan file.
nvinfer1::ICudaEngine *
tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                 const char *filename);

/// Destroy a TensorRT engine.
void tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine);

/// Construct an execution context.
nvinfer1::IExecutionContext *
tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine);

/// Destroy an execution context.
void tensorrt_execution_context_destroy(nvinfer1::IExecutionContext *engine);

} // namespace mtrt

#endif // MTRT_RUNTIME_TENSORRT_H
