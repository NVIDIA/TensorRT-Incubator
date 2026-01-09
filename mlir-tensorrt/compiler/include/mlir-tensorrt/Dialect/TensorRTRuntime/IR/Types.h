//===- Types.h -----------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TYPES
#define MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TYPES

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsTypes.h.inc"

#endif // MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TYPES
