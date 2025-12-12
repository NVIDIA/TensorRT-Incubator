//===- Passes.h -----------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for Kernel-dialect related conversion passes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_CONVERSION_PASSES
#define MLIR_KERNEL_CONVERSION_PASSES

#include <memory>
#include <mlir/Pass/Pass.h>

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-kernel/Conversion/Passes.h.inc"
} // namespace mlir

#endif // MLIR_KERNEL_CONVERSION_PASSES
