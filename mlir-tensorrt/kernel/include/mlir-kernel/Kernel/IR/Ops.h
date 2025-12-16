//===- Ops.h --------------------------------------------------------------===//
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
/// Declarations for Kernel dialect operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_IR_OPS
#define MLIR_KERNEL_KERNEL_IR_OPS

#include "mlir-kernel/Kernel/IR/Attributes.h"
#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

//===----------------------------------------------------------------------===//
// Kernel Dialect Op Declarations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-kernel/Kernel/IR/Ops.h.inc"

#endif // MLIR_KERNEL_KERNEL_IR_OPS
