//===- TensorRTVersionDetail.h ----------------------------------*- C++ -*-===//
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
/// Utilities for handling TensorRT versions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_TENSORRTVERSIONDETAIL_H
#define MLIR_TENSORRT_COMMON_UTILS_TENSORRTVERSIONDETAIL_H
#ifdef MLIR_TRT_TARGET_TENSORRT

#include "NvInferVersion.h"
#include <cstdint>

extern "C" int32_t getInferLibVersion() noexcept;
extern "C" int32_t getInferLibBuildVersion() noexcept;

#else // MLIR_TRT_TARGET_TENSORRT

// In cases where we are not compiling against TensorRT but still want to
// compile and test the TensorRT-related MLIR components of the compiler, we
// assume TensorRT is set to the latest GA release.
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 13
#define NV_TENSORRT_PATCH 2
#define NV_TENSORRT_BUILD 6

#endif // MLIR_TRT_TARGET_TENSORRT

/// Evaluates to true if the version of TensorRT that we are compiling against
/// is greater than or equal to the given version number.
#define MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(major, minor, patch)        \
  (NV_TENSORRT_MAJOR > major ||                                                \
   (NV_TENSORRT_MAJOR == major &&                                              \
    (NV_TENSORRT_MINOR > minor ||                                              \
     (NV_TENSORRT_MINOR == minor && NV_TENSORRT_PATCH >= patch))))

/// Evaluates to true if the version of TensorRT that we are compiling against
/// is less than the given version number.
#define MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(major, minor, patch)         \
  (NV_TENSORRT_MAJOR < major ||                                                \
   (NV_TENSORRT_MAJOR == major &&                                              \
    (NV_TENSORRT_MINOR < minor ||                                              \
     (NV_TENSORRT_MINOR == minor && NV_TENSORRT_PATCH < patch))))

#endif // MLIR_TENSORRT_COMMON_UTILS_TENSORRTVERSIONDETAIL_H
