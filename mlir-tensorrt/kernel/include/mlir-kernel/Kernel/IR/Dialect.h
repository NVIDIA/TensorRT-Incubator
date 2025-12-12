//===- Dialect.h ----------------------------------------------------------===//
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
/// Declarations for the Kernel dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_IR_DIALECT
#define MLIR_KERNEL_KERNEL_IR_DIALECT

#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/IR/TransformScheduleBase.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TensorEncoding.h"

//===----------------------------------------------------------------------===//
// Kernel Dialect Declaration
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Dialect.h.inc"

#endif // MLIR_KERNEL_KERNEL_IR_DIALECT
