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
/// Plan dialect declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN

#include "mlir-tensorrt-common/Interfaces/BoundsAttrInterface.h" // IWYU pragma: keep
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h" // IWYU pragma: keep
#include "mlir-tensorrt/Dialect/Plan/IR/Dialect.h"        // IWYU pragma: keep
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h" // IWYU pragma: keep
#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h" // IWYU pragma: keep
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"              // IWYU pragma: keep
#include "mlir/Interfaces/ControlFlowInterfaces.h"       // IWYU pragma: keep
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"        // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"        // IWYU pragma: keep
#include "mlir/Interfaces/ValueBoundsOpInterface.h"      // IWYU pragma: keep
#include "llvm/Support/ErrorHandling.h"

//===----------------------------------------------------------------------===//
// Plan Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.h.inc"

namespace mlir::plan::detail {
/// Verify a single bounds attribute.
LogicalResult
verifyBoundsAttr(StringRef argOrResult, unsigned idx, Type type,
                 BoundsAttr boundsAttr,
                 llvm::function_ref<InFlightDiagnostic()> emitOpError);
} // namespace mlir::plan::detail

//===----------------------------------------------------------------------===//
// Plan Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsTypes.h.inc"

namespace mlir::plan {
namespace PlanOpTrait {
template <typename ConcreteType>
class PlanDialectOp
    : public ::mlir::OpTrait::TraitBase<ConcreteType, PlanDialectOp> {};

} // namespace PlanOpTrait
} // namespace mlir::plan

//===----------------------------------------------------------------------===//
// Plan Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.h.inc"

//===----------------------------------------------------------------------===//
// Compiler-Runtime Interface Functions
//===----------------------------------------------------------------------===//

namespace mlir::plan {

/// Assign initial slot numbers to the results of the function.
///
/// The slot numbers are assigned to results as result attributes
/// `plan.result_slot` start from 0 to the number of results.
void assignInitialSlotNumbers(OpBuilder &builder, FunctionOpInterface func);

} // namespace mlir::plan

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN
