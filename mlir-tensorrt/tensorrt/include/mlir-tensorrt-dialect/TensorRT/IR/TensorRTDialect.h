//===- TensorRTDialect.h ----------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
/// TensorRT dialect declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTDIALECT
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTDIALECT

#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTBase.h" // IWYU pragma: keep
#include "mlir/Bytecode/BytecodeOpInterface.h"              // IWYU pragma: keep
#include "mlir/IR/Dialect.h"                                // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"                 // IWYU pragma: keep
#include "mlir/Interfaces/CallInterfaces.h"              // IWYU pragma: keep
#include "mlir/Interfaces/ControlFlowInterfaces.h"       // IWYU pragma: keep
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"    // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep

namespace mlir::func {
class FuncOp;
} // namespace mlir::func

namespace mlir {
using ReassociationIndices = SmallVector<int64_t, 2>;
}

//===----------------------------------------------------------------------===//
// TensorRT Dialect Interface Declaration
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttrInterfaces.h.inc"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// TensorRT Dialect Declaration
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// TensorRT Dialect Enum Definitions
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTEnums.h.inc"

//===----------------------------------------------------------------------===//
// TensorRT Dialect Attribute Declarations
//===----------------------------------------------------------------------===//
namespace mlir {
namespace tensorrt {
FailureOr<SmallVector<int64_t>> parseDimList(AsmParser &parser);
void printDimList(AsmPrinter &printer, ArrayRef<int64_t> ints);

/// Dynamic shape information for a single dimension.
struct DynamicDimensionBounds {
  int64_t min;
  int64_t opt;
  int64_t max;
};

using TensorValue = TypedValue<RankedTensorType>;

} // namespace tensorrt
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTAttributes.h.inc"

//===----------------------------------------------------------------------===//
// TensorRT Op Traits
//===----------------------------------------------------------------------===//

namespace mlir {

namespace tensorrt {
namespace detail {
/// Verify the op's result shapes and/or element types match the shapes and/or
/// types given by the result of `componentsTypeFn`. If the inferred components
/// do not include some information like shape or element type, then that
/// information is not verified.
LogicalResult verifyInferredTensorTypesWithPartialInfo(
    Operation *op,
    function_ref<LogicalResult(
        MLIRContext *, std::optional<Location>, ValueShapeRange, DictionaryAttr,
        OpaqueProperties, RegionRange, SmallVectorImpl<ShapedTypeComponents> &)>
        componentTypeFn,
    bool shapesEqualUpToDynamicAmbiguity = false);

bool isCompatibleReturnTypesShapes(TypeRange lhs, TypeRange rhs,
                                   bool shapesEqualUpToDynamicAmbiguity);
} // namespace detail

/// Implementation of trait that verifies result shapes using the op's
/// `inferReturnTypeComponents` method.
template <typename ConcreteType>
class TensorRTInferPartialTensorTypeTrait
    : public OpTrait::TraitBase<ConcreteType,
                                TensorRTInferPartialTensorTypeTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(
        ConcreteType::template hasTrait<InferShapedTypeOpInterface::Trait>(),
        "requires InferShapedTypeOpInterface to ensure successful "
        "invocation");

    return tensorrt::detail::verifyInferredTensorTypesWithPartialInfo(
        op, ConcreteType::inferReturnTypeComponents,
        /*shapesEqualUpToDynamicDim=*/true);
  }
};

/// Implementation of trait that verifies result shapes using the op's
/// `inferReturnTypeComponents` method.
template <typename ConcreteType>
class TensorRTInferCompleteTensorTypeTrait
    : public OpTrait::TraitBase<ConcreteType,
                                TensorRTInferCompleteTensorTypeTrait> {};

//===----------------------------------------------------------------------===//
// ExternalModels
//===----------------------------------------------------------------------===//
void registerTensorKindOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace tensorrt
} // namespace mlir

//===----------------------------------------------------------------------===//
// TensorRT Dialect Op Declarations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.h.inc"

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTDIALECT
