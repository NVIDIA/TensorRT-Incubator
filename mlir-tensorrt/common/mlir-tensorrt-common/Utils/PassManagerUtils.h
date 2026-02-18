//===- PassManagerUtils.h --------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
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
/// Utilities for mlir::OpPassManager.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_PASSMANAGERUTILS
#define MLIR_TENSORRT_COMMON_UTILS_PASSMANAGERUTILS

#include "mlir/Pass/PassManager.h"

namespace mlir {

/// Add nested passes to the given pass manager for the given operation type.
template <typename OpT>
static void
addNestedPasses(OpPassManager &pm,
                llvm::function_ref<void(OpPassManager &)> addPasses) {
  auto &nestedPM = pm.nest<OpT>();
  addPasses(nestedPM);
}

} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_UTILS_PASSMANAGERUTILS
