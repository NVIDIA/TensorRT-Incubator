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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
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

static Attribute
getFuncBounds(FunctionOpInterface func, int64_t idx,
              std::function<Attribute(FunctionOpInterface, unsigned, StringRef)>
                  getAttrFunc) {
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
Attribute executor::getFuncArgsBounds(FunctionOpInterface func,
                                      int64_t argIdx) {
  return getFuncBounds(
      func, argIdx, [](FunctionOpInterface op, unsigned index, StringRef name) {
        return op.getArgAttr(index, name);
      });
}

// Usage for result attributes
Attribute executor::getFuncResultBounds(FunctionOpInterface func,
                                        int64_t argIdx) {
  return getFuncBounds(
      func, argIdx,
      [](FunctionOpInterface op, unsigned index, StringRef name) -> Attribute {
        return op.getResultAttr(index, name);
      });
}

static bool isScalarType(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

static LogicalResult verifyABIFuncArgType(Location loc, Type type) {
  if (isScalarType(type) || isa<PointerType>(type))
    return success();
  return emitError(loc)
         << "arguments to ABI wrapper functions must be IntegerType, "
            "FloatType, IndexType, or host pointer type but got "
         << type;
}

LogicalResult
ExecutorDialect::verifyOperationAttribute(Operation *op,
                                          NamedAttribute attribute) {

  if (attribute.getName() == ExecutorDialect::kFuncABIAttrName) {
    auto func = dyn_cast<FunctionOpInterface>(op);
    if (!func)
      return op->emitError() << "expected " << ExecutorDialect::kFuncABIAttrName
                             << " attribute to be attached to a function";

    auto typeAttr = dyn_cast<TypeAttr>(attribute.getValue());
    if (!typeAttr)
      return op->emitError()
             << "expected " << ExecutorDialect::kFuncABIAttrName
             << " to be a TypeAttr but got " << attribute.getValue();
    auto abiFuncType = dyn_cast<FunctionType>(typeAttr.getValue());
    if (!abiFuncType)
      return op->emitError()
             << "expected " << ExecutorDialect::kFuncABIAttrName
             << " to be a FunctionType but got " << typeAttr.getValue();

    if (!func->getResultTypes().empty() &&
        func.getNumResults() != func.getResultTypes().size()) {
      return op->emitError()
             << "function " << func.getName() << " is decorated with a "
             << ExecutorDialect::kFuncABIAttrName
             << " attribute but it returns " << func.getNumResults()
             << " results while the ABI function type has "
             << abiFuncType.getNumResults() << " results";
    }

    const unsigned numInputs = abiFuncType.getNumInputs();
    const unsigned numOutputs = abiFuncType.getNumResults();
    if (numInputs + numOutputs != func.getNumArguments()) {
      return op->emitError()
             << "function " << func.getName() << " has "
             << func.getNumArguments()
             << " arguments, but the ABI function type has function type "
             << abiFuncType << ", which requires " << numInputs + numOutputs
             << " parameters";
    }

    for (unsigned i = 0; i < numInputs; ++i) {
      Type argType = func.getArgument(i).getType();
      if (failed(verifyABIFuncArgType(func.getArgument(i).getLoc(), argType)))
        return failure();

      if (isScalarType(argType))
        continue;

      Location loc = func.getArgument(i).getLoc();

      auto argABIAttr = func.getArgAttrOfType<executor::ArgumentABIAttr>(
          i, ExecutorDialect::kArgABIAttrName);
      if (!argABIAttr)
        return emitError(loc)
               << "expected " << ExecutorDialect::kArgABIAttrName
               << " argument ABI attribute for input argument " << i;
      if (argABIAttr.getAbi() != executor::ArgABIKind::byval)
        return emitError(loc)
               << "expected " << ExecutorDialect::kArgABIAttrName
               << " input argument " << i
               << " to have 'byval' ABI kind but got " << argABIAttr;
    }

    for (unsigned i = 0; i < numOutputs; ++i) {
      Location loc = func.getArgument(i + numInputs).getLoc();
      Type argType = func.getArgument(i + numInputs).getType();
      if (failed(verifyABIFuncArgType(loc, argType)))
        return failure();
      if (isScalarType(argType))
        continue;

      auto argABIAttr = func.getArgAttrOfType<executor::ArgumentABIAttr>(
          i + numInputs, ExecutorDialect::kArgABIAttrName);
      if (!argABIAttr)
        return emitError(loc)
               << "expected " << ExecutorDialect::kArgABIAttrName
               << " argument ABI attribute for output argument " << i;
      if (argABIAttr.getAbi() != executor::ArgABIKind::byref)
        return emitError(loc)
               << "expected " << ExecutorDialect::kArgABIAttrName
               << " output argument " << i
               << " to have 'byref' ABI kind but got " << argABIAttr;
    }

    return success();
  }

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

  if (attribute.getName() == ExecutorDialect::kArgABIAttrName) {
    auto argABIAttr = dyn_cast<ArgumentABIAttr>(attribute.getValue());
    if (!argABIAttr)
      return op->emitError()
             << "expected " << ExecutorDialect::kArgABIAttrName
             << " attribute to be a #executor.arg<...> attribute";

    auto func = dyn_cast<FunctionOpInterface>(op);
    if (!func)
      return op->emitError()
             << "expected " << ExecutorDialect::kArgABIAttrName
             << " attribute to be attached to a function argument";

    FailureOr<FunctionType> abiFuncType = abi::getABIFunctionType(func);
    if (failed(abiFuncType))
      return failure();

    auto ptrType = dyn_cast<PointerType>(func.getArgument(argIndex).getType());
    if (!ptrType || ptrType.getAddressSpace() != executor::MemoryType::host) {
      return op->emitError() << "expected " << ExecutorDialect::kArgABIAttrName
                             << " attribute to be attached to a host pointer "
                                "type argument but got "
                             << func.getArgument(argIndex).getType();
    }

    const unsigned numInputs = abiFuncType->getNumInputs();
    const bool isInput = argIndex < numInputs;
    Location loc = func.getArgument(argIndex).getLoc();
    if (isInput) {
      if (argABIAttr.getAbi() != executor::ArgABIKind::byval) {
        return emitError(loc) << "expected " << ExecutorDialect::kArgABIAttrName
                              << " attribute to be #executor.arg<byval, ...> "
                                 "for input arguments";
      }
      if (isa<IntegerType, IndexType, FloatType, Float4E2M1FNType,
              Float8E4M3FNType>(argABIAttr.getValueType())) {
        return emitError(loc)
               << "function " << func.getName() << " argument " << argIndex
               << " has ABI " << argABIAttr
               << " but input arguments passed by-val cannot have scalar value "
                  "types";
      }
    }
    if (!isInput && argABIAttr.getAbi() != executor::ArgABIKind::byref) {
      return emitError(loc) << "expected " << ExecutorDialect::kArgABIAttrName
                            << " attribute to be #executor.arg<byref, ...> "
                               "for output arguments but got "
                            << argABIAttr;
    }

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
// ArgumentABIAttr
//===----------------------------------------------------------------------===//

static void printArgumentABIAttrFlags(AsmPrinter &printer, bool undef) {
  if (undef)
    printer << ", undef";
}

static ParseResult parseArgumentABIAttrFlags(AsmParser &parser, bool &undef) {
  undef = false;
  if (parser.parseOptionalComma() || parser.parseOptionalKeyword("undef"))
    return success();
  undef = true;
  return success();
}

//===----------------------------------------------------------------------===//
// ABI Utilities
//===----------------------------------------------------------------------===//

FailureOr<FunctionType>
executor::abi::getABIFunctionType(FunctionOpInterface func) {
  auto abiFuncTypeAttr =
      func->getAttrOfType<TypeAttr>(ExecutorDialect::kFuncABIAttrName);
  if (!abiFuncTypeAttr || !isa<FunctionType>(abiFuncTypeAttr.getValue()))
    return func->emitError() << "expected " << ExecutorDialect::kFuncABIAttrName
                             << " attribute to be TypeAttr with a FunctionType "
                                "attached to the function "
                                "containing arguments "
                                "decorated with "
                             << ExecutorDialect::kArgABIAttrName;
  return cast<FunctionType>(abiFuncTypeAttr.getValue());
}

bool executor::abi::isABIWrapperFunction(FunctionOpInterface func) {
  auto abiFuncTypeAttr =
      func->getAttrOfType<TypeAttr>(ExecutorDialect::kFuncABIAttrName);
  return abiFuncTypeAttr != nullptr;
}

unsigned executor::abi::getNumInputArguments(FunctionOpInterface func) {
  FailureOr<FunctionType> abiFuncType = abi::getABIFunctionType(func);
  assert(succeeded(abiFuncType) && "expected ABI function type");
  return abiFuncType->getNumInputs();
}

unsigned executor::abi::getNumOutputArguments(FunctionOpInterface func) {
  FailureOr<FunctionType> abiFuncType = abi::getABIFunctionType(func);
  assert(succeeded(abiFuncType) && "expected ABI function type");
  return abiFuncType->getNumResults();
}

unsigned executor::abi::getOutputArgumentIndex(FunctionOpInterface func,
                                               BlockArgument arg) {
  assert(isOutputArgument(func, arg).has_value() && "expected output argument");
  auto argIt = llvm::find(func.getArguments(), arg);
  assert(argIt != func.getArguments().end() && "expected argument of func");
  unsigned argIndex = std::distance(func.getArguments().begin(), argIt);
  assert(argIndex >= getNumInputArguments(func) && "expected output argument");
  return argIndex - getNumInputArguments(func);
}

unsigned executor::abi::getInputArgumentIndex(FunctionOpInterface func,
                                              BlockArgument arg) {
  auto argIt = llvm::find(func.getArguments(), arg);
  assert(argIt != func.getArguments().end() && "expected argument of func");
  unsigned argIndex = std::distance(func.getArguments().begin(), argIt);
  assert(argIndex < getNumInputArguments(func) && "expected input argument");
  return argIndex;
}

BlockArgument executor::abi::getInputArgument(FunctionOpInterface func,
                                              unsigned index) {
  assert(index < getNumInputArguments(func) &&
         "expected input argument index to be within range");
  return func.getArgument(index);
}

BlockArgument executor::abi::getOutputArgument(FunctionOpInterface func,
                                               unsigned index) {
  assert(index < getNumOutputArguments(func) &&
         "expected output argument index to be within range");
  return func.getArgument(index + getNumInputArguments(func));
}

void executor::abi::setABIFunctionType(FunctionOpInterface func,
                                       TypeRange inputTypes,
                                       TypeRange resultTypes) {
  func->setAttr(ExecutorDialect::kFuncABIAttrName,
                TypeAttr::get(FunctionType::get(func.getContext(), inputTypes,
                                                resultTypes)));
}

void executor::abi::updateABIInputArgumentValueType(FunctionOpInterface func,
                                                    unsigned inputIdx,
                                                    Type valueType) {
  assert(inputIdx < getNumInputArguments(func) &&
         "expected input argument index to be within range");
  BlockArgument arg = getInputArgument(func, inputIdx);
  if (auto argABIAttr = getArgumentABIAttr(func, arg);
      argABIAttr && argABIAttr.getValueType() != valueType) {
    argABIAttr = argABIAttr.cloneWithValueType(valueType);
    abi::setArgumentABIAttr(func, arg, argABIAttr);
  }

  FailureOr<FunctionType> abiFuncType = getABIFunctionType(func);
  assert(succeeded(abiFuncType) && "expected ABI function type");
  if (abiFuncType->getInput(inputIdx) != valueType) {
    SmallVector<Type> newInputTypes(abiFuncType->getInputs());
    newInputTypes[inputIdx] = valueType;
    setABIFunctionType(func, newInputTypes, abiFuncType->getResults());
  }
}

void executor::abi::updateABIOutputArgumentValueType(FunctionOpInterface func,
                                                     unsigned outputIdx,
                                                     Type valueType) {
  assert(outputIdx < getNumOutputArguments(func) &&
         "expected output argument index to be within range");
  BlockArgument arg = getOutputArgument(func, outputIdx);
  if (auto argABIAttr = getArgumentABIAttr(func, arg);
      argABIAttr && argABIAttr.getValueType() != valueType) {
    argABIAttr = argABIAttr.cloneWithValueType(valueType);
    abi::setArgumentABIAttr(func, arg, argABIAttr);
  }
  FailureOr<FunctionType> abiFuncType = getABIFunctionType(func);
  assert(succeeded(abiFuncType) && "expected ABI function type");
  if (abiFuncType->getResult(outputIdx) != valueType) {
    SmallVector<Type> newResultTypes(abiFuncType->getResults());
    newResultTypes[outputIdx] = valueType;
    setABIFunctionType(func, abiFuncType->getInputs(), newResultTypes);
  }
}

std::optional<unsigned> executor::abi::isInputArgument(FunctionOpInterface func,
                                                       unsigned argIndex) {
  FailureOr<FunctionType> abiFuncType = getABIFunctionType(func);
  if (failed(abiFuncType))
    return std::nullopt;

  const unsigned numInputs = abiFuncType->getNumInputs();
  if (argIndex < numInputs)
    return argIndex;

  return std::nullopt;
}

bool executor::abi::isScalarArgumentType(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

ArgumentABIAttr executor::abi::getArgumentABIAttr(FunctionOpInterface func,
                                                  BlockArgument arg) {
  Block::BlockArgListType args = func.getArguments();
  auto it = llvm::find(args, arg);
  assert(it != args.end() && "expected argument of func");
  unsigned argIndex = std::distance(args.begin(), it);
  return getArgumentABIAttr(func, argIndex);
}

ArgumentABIAttr executor::abi::getArgumentABIAttr(FunctionOpInterface func,
                                                  unsigned argIndex) {
  return func.getArgAttrOfType<executor::ArgumentABIAttr>(
      argIndex, ExecutorDialect::kArgABIAttrName);
}

void executor::abi::setArgumentABIAttr(FunctionOpInterface func,
                                       BlockArgument arg,
                                       ArgumentABIAttr abiAttr) {
  Block::BlockArgListType args = func.getArguments();
  auto it = llvm::find(args, arg);
  assert(it != args.end() && "expected argument of func");
  unsigned argIndex = std::distance(args.begin(), it);
  func.setArgAttr(argIndex, ExecutorDialect::kArgABIAttrName, abiAttr);
}

std::optional<unsigned>
executor::abi::isOutputArgument(FunctionOpInterface func, unsigned argIndex) {
  FailureOr<FunctionType> abiFuncType = getABIFunctionType(func);
  if (failed(abiFuncType))
    return std::nullopt;

  const unsigned numInputs = abiFuncType->getNumInputs();
  if (argIndex >= numInputs)
    return argIndex - numInputs;

  return std::nullopt;
}

std::optional<unsigned>
executor::abi::isOutputArgument(FunctionOpInterface func, BlockArgument arg) {
  // Function arguments may not correspond directly to all block argument,
  // for example see `gpu.func`.
  Block::BlockArgListType args = func.getArguments();
  auto it = llvm::find(args, arg);
  assert(it != args.end() && "expected argument of func");
  unsigned argIndex = std::distance(args.begin(), it);
  return isOutputArgument(func, argIndex);
}

Value executor::abi::getOrCreateABIRecv(OpBuilder &b, FunctionOpInterface func,
                                        BlockArgument arg, Type expectedType) {
  auto it = llvm::find(func.getArguments(), arg);
  assert(it != func.getArguments().end() && "expected argument of func");
  unsigned argIndex = std::distance(func.getArguments().begin(), it);
  return getOrCreateABIRecv(b, func, argIndex, expectedType);
}

Value executor::abi::getOrCreateABIRecv(OpBuilder &b, FunctionOpInterface func,
                                        unsigned argIndex, Type expectedType) {
  assert(isABIWrapperFunction(func) && "func must be an ABI wrapper function");
  // Look for an existing ABIRecvOp that uses this argument
  for (Operation *user : func.getArgument(argIndex).getUsers()) {
    if (auto recvOp = dyn_cast<executor::ABIRecvOp>(user)) {
      if (recvOp.getPtr() == func.getArgument(argIndex)) {
        if (!expectedType || expectedType == recvOp.getResult().getType())
          return recvOp.getResult();
      }
    }
  }

  // No existing ABIRecvOp found, create a new one
  Type resultType = expectedType;
  if (!resultType) {
    // If the argument has an ArgumentABIAttr, use its value_type
    auto argABIAttr = func.getArgAttrOfType<executor::ArgumentABIAttr>(
        argIndex, ExecutorDialect::kArgABIAttrName);
    assert(argABIAttr && "expected ArgumentABIAttr");
    resultType = argABIAttr.getValueType();
  }

  // Create the ABIRecvOp at the beginning of the function body
  OpBuilder::InsertionGuard guard(b);
  Block *entryBlock = &func.getFunctionBody().front();
  b.setInsertionPointToStart(entryBlock);
  BlockArgument arg = func.getArgument(argIndex);
  return b.create<executor::ABIRecvOp>(arg.getLoc(), resultType, arg)
      .getResult();
}

FailureOr<SmallVector<FunctionOpInterface>>
executor::abi::collectAndValidateABIFuncs(Operation *module) {
  SymbolTableCollection symbolTables;
  SymbolUserMap symbolUserMap(symbolTables, module);
  SmallVector<FunctionOpInterface> abiFuncs;
  for (auto func : module->getRegion(0).getOps<FunctionOpInterface>()) {
    if (!func.isPublic() || func.isDeclaration() ||
        !executor::abi::isABIWrapperFunction(func))
      continue;
    abiFuncs.push_back(func);
    for (auto user : symbolUserMap.getUsers(func)) {
      if (isa<CallOpInterface>(user)) {
        return func->emitError(
            "ABI function " + func.getName().str() +
            " is called by a non-ABI operation, which is not allowed");
      }
    }
  }
  return abiFuncs;
}

//===----------------------------------------------------------------------===//
// Plugin ABI Decode Spec
//===----------------------------------------------------------------------===//

FailureOr<abi::plugin::DecodeSpec> executor::abi::plugin::ParseArgSpec(
    Operation *op, unsigned numInputArgs, unsigned numOutputArgs,
    llvm::ArrayRef<llvm::StringRef> argSpec, DictionaryAttr config,
    llvm::SmallVectorImpl<NamedAttribute> &immediateArgs) {

  std::vector<DecodeItem> items;
  items.reserve(argSpec.size());

  static constexpr llvm::StringRef kAttrsPrefix = "attrs.";
  static constexpr llvm::StringRef kArgs = "args.";
  static constexpr llvm::StringRef kRets = "rets.";
  static constexpr llvm::StringRef kNoneAttrName = "none";

  for (size_t i = 0; i < static_cast<size_t>(argSpec.size()); ++i) {
    llvm::StringRef specItem = argSpec[i];
    if (specItem.starts_with(kArgs)) {
      unsigned argIndex;
      if (specItem.drop_front(kArgs.size()).getAsInteger(10, argIndex))
        return op->emitError("expected \"attrs.[integer]\", but got \"")
               << specItem << "\"";
      if (argIndex >= numInputArgs)
        return op->emitError("argument index out of bounds: ") << argIndex;
      items.push_back(DecodeItem{DecodeArg{argIndex}, i, specItem});
    } else if (specItem.starts_with(kRets)) {
      unsigned retIndex;
      if (specItem.drop_front(kRets.size()).getAsInteger(10, retIndex))
        return op->emitError("expected \"rets.[integer]\", but got \"")
               << specItem << "\"";
      if (retIndex >= numOutputArgs)
        return op->emitError("result index out of bounds: ") << retIndex;
      items.push_back(DecodeItem{DecodeRet{retIndex}, i, specItem});

    } else if (specItem.starts_with(kAttrsPrefix)) {
      llvm::StringRef attrKey = specItem.drop_front(kAttrsPrefix.size());
      items.push_back(DecodeItem{DecodeAttr{attrKey}, i, specItem});
      if (!config || !config.contains(attrKey))
        return op->emitError("attribute key \"")
               << attrKey
               << "\" was specified in the argument specification but was not "
                  "found in the stablehlo.custom_call backend config";
      immediateArgs.push_back(NamedAttribute(attrKey, config.get(attrKey)));
    } else if (specItem == kNoneAttrName) {
      items.push_back(DecodeItem{OptionalNoneTag{}, i, specItem});
    } else {
      return op->emitError("unknown argument specification item: ") << specItem;
    }
  }

  return DecodeSpec{items};
}

FailureOr<abi::plugin::DecodeSpec> executor::abi::plugin::ParseArgSpec(
    Operation *op, unsigned numInputArgs, unsigned numOutputArgs,
    llvm::StringRef argSpecString, DictionaryAttr config,
    llvm::SmallVectorImpl<llvm::StringRef> &argSpecComponents,
    llvm::SmallVectorImpl<NamedAttribute> &immediateArg) {
  llvm::SplitString(argSpecString, argSpecComponents, ";");
  return executor::abi::plugin::ParseArgSpec(
      op, numInputArgs, numOutputArgs, argSpecComponents, config, immediateArg);
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
