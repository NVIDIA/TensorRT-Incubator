//===- CUDAToExecutor.h -----------------------------------------*- C++ -*-===//
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
/// Declarations for the `convert-cuda-to-executor` pass.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_CUDATOEXECUTOR_CUDATOEXECUTOR
#define MLIR_TENSORRT_CONVERSION_CUDATOEXECUTOR_CUDATOEXECUTOR

namespace mlir {
class TypeConverter;

/// Populate the type converter with conversions for CUDA types to Executor
/// types.
void populateCUDAToExecutorTypeConversions(TypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_CUDATOEXECUTOR_CUDATOEXECUTOR
