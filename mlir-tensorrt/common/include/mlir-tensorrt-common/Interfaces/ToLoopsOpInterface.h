//===- ToLoopsOpInterface.h -------------------------------*- C++ -*-===//
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
/// This file contains the declarations for the `ToLoopsOpInterface` interface.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class RewriterBase;
namespace scf {
class ForOp;
} // namespace scf

struct LowerToLoopsResult {
  /// Loops from outer to inner-most.
  SmallVector<Operation *> loops;
  /// Replacements for the original op.
  SmallVector<Value> replacements;
};
} // namespace mlir

#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h.inc"

#endif // MLIR_TENSORRT_COMMON_INTERFACES_TOLOOPSOPINTERFACE
