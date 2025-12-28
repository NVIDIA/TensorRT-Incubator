//===- MTRTRuntime.h --------------------------------------------*- C++ -*-===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This file contains an example implementation of C++ functions required
/// to interact with generated C++ host code.
///
//===----------------------------------------------------------------------===//
#ifndef MTRTRUNTIME
#define MTRTRUNTIME

// Convenience umbrella header for generated EmitC C++.
//
// This intentionally does NOT include TensorRT headers. If your generated C++
// uses TensorRT wrappers (e.g. `mtrt::tensorrt_enqueue`), include
// `MTRTRuntimeTensorRT.h` in addition.

#include "MTRTRuntimeCore.h"
#include "MTRTRuntimeCuda.h"

#endif // MTRTRUNTIME
