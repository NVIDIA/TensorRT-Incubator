//===- TensorRTRuntimeToExecutor.h ----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Declarations for the `convert-tensorrt-runtime-to-executor` pass.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_TENSORRTRUNTIMETOEXECUTOR_TENSORRTRUNTIMETOEXECUTOR
#define MLIR_TENSORRT_CONVERSION_TENSORRTRUNTIMETOEXECUTOR_TENSORRTRUNTIMETOEXECUTOR

namespace mlir {
class TypeConverter;

/// Populate the type converter with conversions for TensorRT Runtime types to
/// Executor types.
void populateTensorRTRuntimeToExecutorTypeConversions(
    TypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_TENSORRTRUNTIMETOEXECUTOR_TENSORRTRUNTIMETOEXECUTOR
