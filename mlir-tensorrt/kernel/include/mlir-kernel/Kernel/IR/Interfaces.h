//===- Interfaces.h -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Kernel dialect interface declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_IR_INTERFACES
#define MLIR_KERNEL_KERNEL_IR_INTERFACES

#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/PassManager.h"

//===----------------------------------------------------------------------===//
// Kernel Attr Interface Declarations
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/AttrInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// Kernel Type Interface Declarations
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/TypeInterfaces.h.inc"

namespace mlir {
class DialectRegistry;
}

namespace mlir::kernel {

/// Register implementation of the PointerLikeTypeInterface for external
/// dialects.
void registerExternalDialectPtrTypeInterfaceImpls(DialectRegistry &registry);

} // namespace mlir::kernel

//===----------------------------------------------------------------------===//
// Kernel Op Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir-kernel/Kernel/IR/OpInterfaces.h.inc"

#endif // MLIR_KERNEL_KERNEL_IR_INTERFACES
