//===- StreamSchedulableOpInterface.h -------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Declarations for StreamSchedulableOp.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE

#include "mlir/IR/OpDefinition.h" // IWYU pragma: keep
#include "llvm/ADT/SmallVector.h" // IWYU pragma: keep

namespace mtrt::compiler {
class StreamSchedulableOp;
} // namespace mtrt::compiler

#include "mlir-tensorrt-common/Interfaces/StreamSchedulableOpInterface.h.inc"

#endif // MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE
