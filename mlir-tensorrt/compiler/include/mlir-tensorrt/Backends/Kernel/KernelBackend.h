//===- KernelBackend.h ------------------------------------------*- C++ -*-===//
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
/// Declarations for the Kernel backend.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_BACKENDS_KERNEL_KERNELBACKEND
#define MLIR_TENSORRT_BACKENDS_KERNEL_KERNELBACKEND

#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Backends/Kernel/KernelBackendAttrs.h.inc"

namespace mtrt::compiler {

/// Register the Kernel backend extensions to the Plan dialect.
void registerKernelBackend(mlir::DialectRegistry &registry);

/// The attribute name used to mark a function as a kernel gen cluster.
llvm::StringRef getKernelGenClusterAttrName();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_BACKENDS_KERNEL_KERNELBACKEND
