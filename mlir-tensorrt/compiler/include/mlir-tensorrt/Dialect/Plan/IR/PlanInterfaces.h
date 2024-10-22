//===- Plan.h ---------------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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

/// Helper function used by cluster backends that returns 'true' if the given
/// clusterable operation should be executed on the host as determined by the
/// DataFlow analysis as well as the operation's types/results.

namespace mlir::plan::detail {
bool shouldRunOnHost(Operation *op, DataFlowSolver &solver);
}

#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.h.inc"

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLANINTERFACES
