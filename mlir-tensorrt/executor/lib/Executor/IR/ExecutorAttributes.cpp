//===- ExecutorAttributes.cpp ---------------------------------------------===//
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
/// Implementation for Executor dialect attributes.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::executor;

//===----------------------------------------------------------------------===//
// DimensionBoundsAttr
//===----------------------------------------------------------------------===//

/// Check type of `min` matches type of `max` and that all elements of `min` are
/// <= elements of `max`
LogicalResult executor::DimensionBoundsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr min,
    DenseI64ArrayAttr max) {

  if (min.getElementType() != max.getElementType())
    return emitError() << "DimensionBoundsAttr 'min' and 'max' must have "
                          "matching element types; found min type: "
                       << min.getElementType()
                       << ", max type: " << max.getElementType();

  if (min.getSize() != max.getSize())
    return emitError() << "DimensionBoundsAttr 'min' and 'max' must have the "
                          "same size; found min size: "
                       << min.getSize() << ", max size: " << max.getSize();
  for (int i = 0; i < min.getSize(); ++i) {
    if (min[i] < 0)
      return emitError() << "DimensionBoundsAttr min[" << i << "] : " << min[i]
                         << " must be greater than or equal to 0";
    if (min[i] > max[i])
      return emitError() << "DimensionBoundsAttr min[" << i << "] : " << min[i]
                         << " must be less than equal to "
                         << "max[" << i << "] : " << max[i];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ValueBoundsAttr
//===----------------------------------------------------------------------===//

/// Check type of `min` matches type of `max` and that all elements of `min` are
/// <= elements of `max`.
LogicalResult executor::ValueBoundsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ElementsAttr min,
    ElementsAttr max) {
  if (min.getType() != max.getType())
    return emitError() << "ValueBoundsAttr 'min' and 'max' must have "
                          "matching types; found min type: "
                       << min.getType() << ", max type: " << max.getType();

  if (!min.getShapedType().getElementType().isSignlessIntOrIndex())
    return emitError()
           << "ValueBoundsAttr 'min' and 'max' value bounds element type must "
              "be a signless integer or "
              "an index type";

  // Compare underlying values.
  auto mins = min.tryGetValues<APInt>();
  auto maxs = max.tryGetValues<APInt>();
  if (!mins || !maxs)
    return emitError() << "failed to iterate the min/max values as integers";

  for (auto [i, minV, maxV] : llvm::enumerate(*mins, *maxs)) {
    if (minV.sgt(maxV))
      return emitError() << "ValueBoundsAttr min[" << i
                         << "] : " << minV.getSExtValue()
                         << " must be less than equal to "
                         << "max[" << i << "] : " << maxV.getSExtValue();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FunctionMetadataAttr
//===----------------------------------------------------------------------===//

static LogicalResult
checkDimensionBoundsAttr(Type type, DimensionBoundsAttr attr,
                         function_ref<InFlightDiagnostic()> emitError) {
  if (!isa<MemRefType>(type))
    return emitError() << "DimensionBoundsAttr is only for a memref type";
  return success();
}

static LogicalResult
checkValueBoundsAttr(Type type, ValueBoundsAttr attr,
                     function_ref<InFlightDiagnostic()> emitError) {

  if (!isa<MemRefType>(type) && !type.isIntOrIndexOrFloat())
    return emitError()
           << "ValueBoundsAttr is only for memref, index, int, or float type";

  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (!memref.hasStaticShape())
      return emitError()
             << "ValueBoundsAttr must not be present for dynamic memref";

    if (attr.getMin().getElementType() != memref.getElementType())
      return emitError() << "ValueBoundsAttr 'min/max' and corresponding "
                            "memref type must have matching element types; "
                            "found min/max type: "
                         << attr.getMin().getElementType()
                         << ", memref type: " << memref.getElementType();

    if (attr.getMin().getShapedType().getShape() != memref.getShape())
      return emitError()
             << "ValueBoundsAttr 'min/max' and corresponding "
                "memref type must have matching shapes; found min/max shape: "
             << attr.getMin().getShapedType().getShape()
             << ", memref shape: " << memref.getShape();
  }

  return success();
}

static LogicalResult
checkAttributeType(ArrayRef<Type> types, ArrayRef<Attribute> bounds,
                   function_ref<InFlightDiagnostic()> emitError) {
  for (unsigned i = 0; i < types.size(); ++i) {
    if (auto dimensionBounds = dyn_cast<DimensionBoundsAttr>(bounds[i])) {
      if (failed(
              checkDimensionBoundsAttr(types[i], dimensionBounds, emitError)))
        return emitError() << "Unsupported DimensionBoundsAttr";
    }
    if (auto valueBounds = dyn_cast<ValueBoundsAttr>(bounds[i])) {
      if (failed(checkValueBoundsAttr(types[i], valueBounds, emitError)))
        return emitError() << "Unsupported ValueBoundsAttr";
    }
    if (!isa<UnitAttr>(bounds[i]) && !isa<DimensionBoundsAttr>(bounds[i]) &&
        !isa<ValueBoundsAttr>(bounds[i]))
      return emitError() << "Unsupported attribute type";
  }
  return success();
}

LogicalResult executor::FunctionMetadataAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Type> args,
    ArrayRef<Type> results, int64_t nbOutArgs, ArrayRef<Attribute> argBounds,
    ArrayRef<Attribute> resBounds, FlatSymbolRefAttr shapeFuncName,
    CallingConvention callingConvention) {
  if (nbOutArgs < 0)
    return emitError() << "Number of output arguments must be non-negative";

  // Check sizes of arguments and bounds.
  if (args.size() != argBounds.size())
    return emitError() << "Size of args and arg bounds must be same";
  if (results.size() != resBounds.size())
    return emitError() << "Size of results and result bounds must be same";

  // Check for appropriate attribute types for arguments and results.
  if (failed(checkAttributeType(args, argBounds, emitError)))
    return failure();

  return checkAttributeType(results, resBounds, emitError);
}

//===----------------------------------------------------------------------===//
// Custom print/parse directives for attributes
//===----------------------------------------------------------------------===//

static void printTypesWithBoundsAttrs(AsmPrinter &printer, ArrayRef<Type> types,
                                      ArrayRef<Attribute> attrs) {
  assert(types.size() == attrs.size() &&
         "Mismatched sizes of types and bounds attributes");
  printer.getStream() << "[";
  for (int i = 0, e = types.size(); i != e; ++i) {
    if (i != 0)
      printer.getStream() << ", ";
    types[i].print(printer.getStream());
    // Skip UnitAttr which depict a NoneType.
    if (!isa<UnitAttr>(attrs[i])) {
      printer.getStream() << " {";
      attrs[i].print(printer.getStream());
      printer.getStream() << "}";
    }
  }
  printer.getStream() << "]";
}

static ParseResult
parseTypesWithBoundsAttrs(AsmParser &parser, SmallVectorImpl<Type> &types,
                          SmallVectorImpl<Attribute> &attrs) {
  // Parse the opening '['
  if (parser.parseLSquare())
    return failure();

  // Loop until ']' is encountered. `true` mean missing `]`.
  while (parser.parseOptionalRSquare()) {
    Type type;
    Attribute attr;

    auto res = parser.parseOptionalType(type);
    if (!res.has_value())
      continue; // No type is present.

    if (res.has_value() && res.value())
      return failure(); // Type is present, but failed to parse.

    // Parsed type successfully.
    types.push_back(type);

    // Check for optional attribute enclosed in '{' '}'
    if (parser.parseOptionalLBrace()) {
      // If no attribute is present, push a null attribute
      attrs.push_back(parser.getBuilder().getUnitAttr());
      // Optional comma between elements. If no comma is found, move on.
      (void)parser.parseOptionalComma();
      continue; // No attribute, move on.
    }

    if (parser.parseAttribute(attr))
      return failure(); // Attribute must exist following a `{`

    if (parser.parseRBrace())
      return failure(); // Attribute must be followed by a closing `}`

    attrs.push_back(attr);

    // Optional comma between elements. If no comma is found, move on.
    (void)parser.parseOptionalComma();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Function Arg Attributes Access and Verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyValueBoundsAttribute(Operation *op,
                                                unsigned argIndex,
                                                executor::ValueBoundsAttr attr,
                                                StringRef attrName) {
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return op->emitError()
           << attrName
           << " should only be used for FunctionOpInterface argument "
              "and result attributes";

  ShapedType valuesType = attr.getMin().getShapedType();

  Type argType = func.getArgument(argIndex).getType();
  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    if (valuesType.getShape() != shapedType.getShape() ||
        (valuesType.getElementType().isIndex() &&
         !shapedType.getElementType().isIntOrIndex()) ||
        (!valuesType.getElementType().isIndex() &&
         shapedType.getElementType() != shapedType.getElementType()))
      return op->emitError()
             << attrName << " value bounds type " << valuesType
             << " is not compatible with the argument type " << argType;

    return success();
  }

  if (argType.isIntOrIndexOrFloat()) {
    if (attr.getMin().getShapedType().getRank() != 0)
      return op->emitError()
             << attrName << " bounds of type " << valuesType
             << " must be a 0-rank shaped type for scalar argument type "
             << argType;
  }

  // If the type is not a shaped type or scalar, then we don't do any
  // validation. It may could correspond to whatever type that the memref was
  // lowered into (e.g. pointer or table), so there's not much validation that
  // is possible.
  return success();
}

static Attribute getFuncBounds(
    func::FuncOp func, int64_t idx,
    std::function<Attribute(func::FuncOp, unsigned, StringRef)> getAttrFunc) {
  if (Attribute shapeBounds =
          getAttrFunc(func, idx, ExecutorDialect::getShapeBoundsAttrName())) {
    assert(isa<executor::DimensionBoundsAttr>(shapeBounds) &&
           "expected ExecutorDimensionBoundsAttr");
    return shapeBounds;
  }
  if (Attribute valueBounds =
          getAttrFunc(func, idx, ExecutorDialect::getValueBoundsAttrName())) {
    assert(isa<executor::ValueBoundsAttr>(valueBounds) &&
           "expected executor::ValueBoundsAttr");
    return valueBounds;
  }

  return UnitAttr::get(func.getContext());
}

// Usage for argument attributes
Attribute executor::getFuncArgsBounds(func::FuncOp func, int64_t argIdx) {
  return getFuncBounds(func, argIdx,
                       [](func::FuncOp op, unsigned index, StringRef name) {
                         return op.getArgAttr(index, name);
                       });
}

// Usage for result attributes
Attribute executor::getFuncResultBounds(func::FuncOp func, int64_t argIdx) {
  return getFuncBounds(
      func, argIdx,
      [](func::FuncOp op, unsigned index, StringRef name) -> Attribute {
        return op.getResultAttr(index, name);
      });
}

LogicalResult
ExecutorDialect::verifyOperationAttribute(Operation *op,
                                          NamedAttribute attribute) {
  return success();
}

LogicalResult
ExecutorDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                          unsigned argIndex,
                                          NamedAttribute attribute) {
  if (attribute.getName() == getValueBoundsAttrName()) {
    auto boundsAttr = dyn_cast<ValueBoundsAttr>(attribute.getValue());
    if (!boundsAttr)
      return op->emitError()
             << "expected named attribute \"" << getValueBoundsAttrName()
             << "\" to be a \"#executor.value_bounds\" attribute containing "
                "value bounds";
    return verifyValueBoundsAttribute(op, argIndex, boundsAttr,
                                      attribute.getName());
  }

  if (attribute.getName() == ExecutorDialect::kResultArgAttrName) {
    auto resultIdx = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!resultIdx || !resultIdx.getType().isInteger(32))
      return op->emitError()
             << "expected " << ExecutorDialect::kResultArgAttrName
             << " attribute to have i32 value";
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Module Attributes Utilities
//===----------------------------------------------------------------------===//

StringRef executor::getExecutorGlobalInitializerFuncNameAttr() {
  return "executor.global_init_func";
}

FailureOr<ArrayRef<int64_t>>
executor::getModuleProcessGridShape(Operation *op) {
  DenseI64ArrayAttr attr = op->getAttrOfType<DenseI64ArrayAttr>(
      ExecutorDialect::kProcessGridShapeAttrName);
  if (!attr)
    return failure();
  return attr.asArrayRef();
}

LogicalResult executor::setModuleProcessGridShape(Operation *op,
                                                  ArrayRef<int64_t> shape) {
  FailureOr<ArrayRef<int64_t>> existingShape = getModuleProcessGridShape(op);
  if (failed(existingShape)) {
    op->setAttr(ExecutorDialect::kProcessGridShapeAttrName,
                DenseI64ArrayAttr::get(op->getContext(), shape));
    return success();
  }
  return success(shape == *existingShape);
}

//===----------------------------------------------------------------------===//
// TableGen'd enum definition.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd attributes definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute registration
//===----------------------------------------------------------------------===//

void ExecutorDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-executor/Executor/IR/ExecutorAttributes.cpp.inc"
      >();
}
