//===- Plan.h ---------------------------------------------------*- C++ -*-===//
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
/// Plan dialect interface declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES

#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class DataFlowSolver;
}

namespace mlir::plan {
enum class InputKind : uint32_t;
} // namespace mlir::plan

#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.h.inc"

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
