//===- PlanAttributes.cpp -------------------------------------------------===//
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
/// Definitions of Plan dialect attributes.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

TensorKindInfo MemorySpaceAttr::getTensorKind() const {
  switch (getValue()) {
  case MemorySpace::device:
    return TensorKind::Device;
  case MemorySpace::host:
    return TensorKind::Host;
  case MemorySpace::host_pinned:
    return TensorKind::Host;
  case MemorySpace::unified:
    return TensorKind::Both;
  case MemorySpace::unknown:
    return TensorKindInfo();
  }
  llvm_unreachable("unknown plan MemorySpace kind");
}

bool MemorySpaceAttr::isHostVisible() const {
  return isVisible(plan::DeviceKind::CPU);
}

bool MemorySpaceAttr::isGpuVisible() const {
  return isVisible(plan::DeviceKind::GPU);
}

bool MemorySpaceAttr::isVisible(plan::DeviceKind deviceKind) const {
  bool isGPU = deviceKind == plan::DeviceKind::GPU;
  bool isHost = deviceKind == plan::DeviceKind::CPU;
  switch (getValue()) {
  case MemorySpace::device:
    return isGPU;
  case MemorySpace::host:
    return isHost;
  case MemorySpace::host_pinned:
    return isHost;
  case MemorySpace::unified:
    return true;
  case MemorySpace::unknown:
    return false;
  }
  llvm_unreachable("unknown plan MemorySpace kind");
}

//===----------------------------------------------------------------------===//
// BoundsAttr
//===----------------------------------------------------------------------===//

LogicalResult BoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 plan::BoundsKind kind,
                                 DenseI64ArrayAttr min_shape,
                                 DenseI64ArrayAttr max_shape,
                                 ElementsAttr min_values,
                                 ElementsAttr max_values) {
  if (kind == BoundsKind::None) {
    if (min_shape || max_shape || min_values || max_values)
      return emitError()
             << "expected no shape or value bounds for bounds of kind 'none'";
    return success();
  }
  if (kind == BoundsKind::Shape) {
    if (!max_shape || !min_shape)
      return emitError()
             << "when kind=shape max_shape and min_shape must both be provided";
    if (max_shape.size() != min_shape.size())
      return emitError() << "max_shape size (" << max_shape.size()
                         << ") must equal min_shape size (" << min_shape.size()
                         << ")";

    unsigned linearIndex = 0;
    for (auto [lhs, rhs] :
         llvm::zip_equal(min_shape.asArrayRef(), max_shape.asArrayRef())) {
      if (lhs > rhs)
        return emitError() << llvm::formatv(
                   "min_shape must be pointwise less-than-or-equal-to "
                   "max_shape, but"
                   " min_shape[{0}] = {1} > max_shape[{0}] = {2}",
                   linearIndex, lhs, rhs);
      linearIndex++;
    }

    return success();
  }

  if (!min_values || !max_values)
    return emitError()
           << "when kind=value max_values and min_values must both be provided";
  if (min_values.getType() != max_values.getType())
    return emitError() << "min_values type (" << min_values.getType()
                       << ") and max_values type (" << max_values.getType()
                       << ") must be the same";

  if (!min_values.getElementType().isSignlessIntOrIndexOrFloat())
    return emitError()
           << "min_values and max_values must have an element type "
              "of signless integer, index, or a floating point type";

  SmallVector<int64_t> basis =
      mlir::computeSuffixProduct(min_values.getShapedType().getShape());

  if (min_values.getElementType().isIntOrIndex()) {
    unsigned linearIndex = 0;
    for (auto [lhs, rhs] : llvm::zip_equal(min_values.getValues<APInt>(),
                                           max_values.getValues<APInt>())) {
      if (lhs.getSExtValue() > rhs.getSExtValue()) {
        SmallVector<int64_t> coord = mlir::delinearize(linearIndex, basis);
        return emitError() << llvm::formatv(
                   "min_values must be pointwise less-than-or-equal-to "
                   "max_values, but"
                   " min_values[{0:$[, ]}] = {1} > max_values[{0:$[, ]}] = {2}",
                   llvm::make_range(coord.begin(), coord.end()), lhs, rhs);
      }
      linearIndex++;
    }
  }

  if (llvm::isa<FloatType>(min_values.getElementType())) {
    unsigned linearIndex = 0;
    for (auto [lhs, rhs] : llvm::zip_equal(min_values.getValues<APFloat>(),
                                           max_values.getValues<APFloat>())) {
      if (lhs > rhs) {
        SmallVector<int64_t> coord = mlir::delinearize(linearIndex, basis);
        return emitError() << llvm::formatv(
                   "min_values must be pointwise less-than-or-equal-to "
                   "max_values, but"
                   " min_values[{0:$[, ]}] = {1} > max_values[{0:$[, ]}] = {2}",
                   llvm::make_range(coord.begin(), coord.end()),
                   lhs.convertToDouble(), rhs.convertToDouble());
      }
      linearIndex++;
    }
  }

  return success();
}

Attribute BoundsAttr::parse(::mlir::AsmParser &odsParser,
                            ::mlir::Type odsType) {
  if (odsParser.parseLess())
    return {};

  if (succeeded(odsParser.parseOptionalKeyword("none"))) {
    if (odsParser.parseGreater())
      return {};
    return odsParser.getChecked<BoundsAttr>(odsParser.getContext());
  }

  if (succeeded(odsParser.parseOptionalKeyword("shape"))) {
    if (odsParser.parseComma())
      return {};
    auto minShape = llvm::dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(odsParser, Type{}));
    if (!minShape)
      return {};
    if (odsParser.parseComma())
      return {};
    auto maxShape = llvm::dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(odsParser, Type{}));
    if (!maxShape)
      return {};
    if (odsParser.parseGreater())
      return {};
    return odsParser.getChecked<BoundsAttr>(
        odsParser.getContext(), BoundsKind::Shape, minShape, maxShape,
        DenseElementsAttr{}, DenseElementsAttr{});
  }

  DenseElementsAttr maxValues{}, minValues{};
  if (odsParser.parseKeyword("value") || odsParser.parseComma() ||
      odsParser.parseAttribute<DenseElementsAttr>(minValues) ||
      odsParser.parseComma() ||
      odsParser.parseAttribute<DenseElementsAttr>(maxValues) ||
      odsParser.parseGreater())
    return {};
  return odsParser.getChecked<BoundsAttr>(
      odsParser.getContext(), BoundsKind::Value, DenseI64ArrayAttr{},
      DenseI64ArrayAttr{}, minValues, maxValues);
}

void BoundsAttr::print(AsmPrinter &p) const {
  p << "<";
  if (getKind() == BoundsKind::None) {
    p << "none>";
    return;
  }

  p << (getKind() == BoundsKind::Shape ? "shape" : "value");
  p << ", ";
  if (getKind() == BoundsKind::Value) {
    p << getMinValues() << ", " << getMaxValues();
  } else {
    getMinShape().print(p);
    p << ", ";
    getMaxShape().print(p);
  }
  p << ">";
}

BoundsAttr BoundsAttr::get(MLIRContext *ctx) {
  return BoundsAttr::get(ctx, BoundsKind::None, {}, {}, {}, {});
}

BoundsAttr
BoundsAttr::getChecked(llvm::function_ref<InFlightDiagnostic()> emitError,
                       MLIRContext *ctx) {
  if (failed(verify(emitError, BoundsKind::None, {}, {}, {}, {})))
    return {};
  return BoundsAttr::get(ctx, BoundsKind::None, {}, {}, {}, {});
}

BoundsAttr BoundsAttr::get(MLIRContext *ctx, BoundsKind kind,
                           ArrayRef<int64_t> min, ArrayRef<int64_t> max) {
  assert(min.size() == max.size() && "expected equal-length arrays");
  assert((kind == BoundsKind::Shape || kind == BoundsKind::Value) &&
         "expected shape or value kind");
  if (kind == BoundsKind::Shape)
    return BoundsAttr::get(ctx, kind, DenseI64ArrayAttr::get(ctx, min),
                           DenseI64ArrayAttr::get(ctx, max), {}, {});
  RankedTensorType type = RankedTensorType::get(
      {static_cast<int64_t>(min.size())}, IndexType::get(ctx));
  return BoundsAttr::get(ctx, kind, {}, {},
                         DenseIntElementsAttr::get(type, min),
                         DenseIntElementsAttr::get(type, max));
}

BoundsAttr
BoundsAttr::getChecked(llvm::function_ref<InFlightDiagnostic()> emitError,
                       MLIRContext *ctx, BoundsKind kind, ArrayRef<int64_t> min,
                       ArrayRef<int64_t> max) {
  if (kind == BoundsKind::Shape) {
    auto minAttr = DenseI64ArrayAttr::get(ctx, min);
    auto maxAttr = DenseI64ArrayAttr::get(ctx, max);
    return getChecked(emitError, ctx, kind, minAttr, maxAttr,
                      DenseElementsAttr{}, DenseElementsAttr{});
  }
  if (kind == BoundsKind::Value) {
    auto i64Ty = IntegerType::get(ctx, 64);
    auto minAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(min.size())}, i64Ty), min);
    auto maxAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(max.size())}, i64Ty), max);
    return getChecked(emitError, ctx, kind, DenseI64ArrayAttr{},
                      DenseI64ArrayAttr{}, minAttr, maxAttr);
  }
  return BoundsAttr::get(ctx);
}

LogicalResult BoundsAttr::getShapeRange(SmallVectorImpl<int64_t> &min,
                                        SmallVectorImpl<int64_t> &max) const {
  if (!isShapeBound())
    return failure();
  ArrayRef<int64_t> minShape = getMinShape();
  min.assign(minShape.begin(), minShape.end());
  ArrayRef<int64_t> maxShape = getMaxShape();
  max.assign(maxShape.begin(), maxShape.end());
  return success();
}

std::optional<ConstantIntRanges>
BoundsAttr::getDimensionRange(int64_t dimension) const {
  if (!isShapeBound())
    return std::nullopt;
  return ConstantIntRanges::fromUnsigned(
      APInt(IndexType::kInternalStorageBitWidth, getMinShape()[dimension]),
      APInt(IndexType::kInternalStorageBitWidth, getMaxShape()[dimension]));
}

std::optional<ConstantIntRanges> BoundsAttr::getIntegerValueRange() const {
  if (!isValueBound() || !getMinValues().getElementType().isIntOrIndex() ||
      getValuesType().getNumElements() != 1)
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APInt>();
  auto maxs = getMaxValues().tryGetValues<APInt>();

  if (!mins || !maxs || mins->empty() || maxs->empty())
    return std::nullopt;

  return ConstantIntRanges::fromSigned(*mins->begin(), *maxs->begin());
}

std::optional<SmallVector<ConstantIntRanges>>
BoundsAttr::getIntegerValueRanges() const {
  if (!isValueBound() || !getMinValues().getElementType().isIntOrIndex())
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APInt>();
  auto maxs = getMaxValues().tryGetValues<APInt>();

  if (!mins || !maxs || mins->empty() || maxs->empty())
    return std::nullopt;

  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(mins->size());
  for (auto [min, max] : llvm::zip_equal(*mins, *maxs))
    ranges.push_back(ConstantIntRanges::fromSigned(min, max));
  return ranges;
}

std::optional<ConstantIntRanges> BoundsAttr::getIntegerValueRangeAtCoordinate(
    ArrayRef<uint64_t> coordinate) const {
  if (!isValueBound() || !getMinValues().getElementType().isIntOrIndex() ||
      !ElementsAttr::isValidIndex(getValuesType(), coordinate))
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APInt>();
  auto maxs = getMaxValues().tryGetValues<APInt>();

  if (!mins || !maxs)
    return std::nullopt;

  return ConstantIntRanges::fromSigned((*mins)[coordinate],
                                       (*maxs)[coordinate]);
}

std::optional<FloatValueRange> BoundsAttr::getFloatValueRange() const {
  if (!isValueBound() || !getValuesType().isFloat() ||
      getValuesType().getNumElements() != 1)
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APFloat>();
  auto maxs = getMaxValues().tryGetValues<APFloat>();

  if (!mins || !maxs || mins->empty() || maxs->empty())
    return std::nullopt;

  return FloatValueRange{*mins->begin(), *maxs->begin()};
}

std::optional<SmallVector<FloatValueRange>>
BoundsAttr::getFloatValueRanges() const {
  if (!isValueBound() || !getValuesType().getElementType().isFloat())
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APFloat>();
  auto maxs = getMaxValues().tryGetValues<APFloat>();

  if (!mins || !maxs || mins->empty() || maxs->empty())
    return std::nullopt;

  SmallVector<FloatValueRange> ranges;
  ranges.reserve(mins->size());
  for (auto [min, max] : llvm::zip_equal(*mins, *maxs))
    ranges.push_back(FloatValueRange{min, max});
  return ranges;
}

std::optional<FloatValueRange> BoundsAttr::getFloatValueRangeAtCoordinate(
    ArrayRef<uint64_t> coordinate) const {
  if (!isValueBound() || !getValuesType().isFloat() ||
      !ElementsAttr::isValidIndex(getValuesType(), coordinate))
    return std::nullopt;

  auto mins = getMinValues().tryGetValues<APFloat>();
  auto maxs = getMaxValues().tryGetValues<APFloat>();

  if (!mins || !maxs)
    return std::nullopt;

  return FloatValueRange{(*mins)[coordinate], (*maxs)[coordinate]};
}

LogicalResult plan::detail::verifyBoundsAttr(
    StringRef argOrResult, unsigned idx, Type type, BoundsAttr boundsAttr,
    llvm::function_ref<InFlightDiagnostic()> emitOpError) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    if (boundsAttr.isNone())
      return success();
    if (boundsAttr.isShapeBound()) {
      const int64_t boundsLength = boundsAttr.getMinShape().size();
      if (shapedType.getRank() != boundsLength)
        return emitOpError()
               << argOrResult << " #" << idx << " has type " << type
               << ", whose rank is not equal to the rank of the "
                  "corresponding shape "
                  "bounds "
               << boundsAttr;
    }
    if (boundsAttr.isValueBound() && !shapedType.hasStaticShape())
      return emitOpError() << argOrResult << " #" << idx << " has type "
                           << shapedType
                           << ", but has a corresponding bounds attribute of "
                              "'value' kind, which is "
                              "only allowed for staticly shaped operands";

    if (boundsAttr.isValueBound()) {
      Type elType = boundsAttr.getValuesType().getElementType();
      if (elType != shapedType.getElementType())
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected element type of value bounds elements (" << elType
               << ") to be compatible with the type (" << type << ")";
      if (boundsAttr.getValuesType().getShape() != shapedType.getShape())
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected type of values bounds elements ("
               << boundsAttr.getValuesType()
               << ") to be compatible with the type (" << type << ")";
    }

    return success();
  }
  if (type.isIntOrIndexOrFloat()) {
    if (boundsAttr.isNone())
      return success();

    if (boundsAttr.isShapeBound())
      return emitOpError() << "expected only value bounds or none bounds for "
                              "scalar "
                           << argOrResult << " #" << idx << " of type " << type
                           << ", but got " << boundsAttr;
    if (boundsAttr.isValueBound()) {

      if (boundsAttr.getValuesType().getRank() != 0)
        return emitOpError()
               << argOrResult << " #" << idx
               << " type expects rank-0 value bounds type, but got "
               << boundsAttr.getValuesType();

      Type elType = boundsAttr.getValuesType().getElementType();
      if (elType != type)
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected element type of value bounds elements (" << elType
               << ") to be compatible with the type (" << type << ")";
    }
    return success();
  }

  if (!boundsAttr.isNone())
    return emitOpError() << "expected only 'none' bounds for type " << type;
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd attributes definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// PlanDialect Hooks
//===----------------------------------------------------------------------===//

Attribute PlanDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&keyword)))
    return nullptr;

  auto it = attrParsingHooks.find(keyword);
  if (it == attrParsingHooks.end()) {
    parser.emitError(loc) << "unknown type mnemonic: " << keyword;
    return nullptr;
  }

  return it->getValue()(parser, type);
}

void PlanDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  auto it = attrPrintingHooks.find(attr.getTypeID());
  assert(it != attrPrintingHooks.end() && "printing unknown type");
  it->getSecond()(attr, printer);
}

/// Verify a bounds attribute on a function argument.
static LogicalResult verifyBoundsAttribute(Operation *op, unsigned argIndex,
                                           BoundsAttr attr,
                                           StringRef attrName) {
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return success();

  Type valueType = func.getArgument(argIndex).getType();

  // If the function is an ABI wrapper function, we need to get the value type
  // from the ABI function type since the type of the argument may be something
  // like `!executor.ptr<host>`.
  if (executor::abi::isABIWrapperFunction(func)) {
    FailureOr<FunctionType> abiFuncType =
        executor::abi::getABIFunctionType(func);
    if (failed(abiFuncType))
      return failure();
    if (std::optional<unsigned> inputIndex =
            executor::abi::isInputArgument(func, argIndex)) {
      valueType = abiFuncType->getInput(*inputIndex);
    } else {
      std::optional<unsigned> outputIndex =
          executor::abi::isOutputArgument(func, argIndex);
      assert(outputIndex.has_value() && "expected output index");
      valueType = abiFuncType->getResult(*outputIndex);
    }
  }

  return plan::detail::verifyBoundsAttr(
      "arg", argIndex, valueType, attr,
      [&]() -> InFlightDiagnostic { return op->emitOpError(); });
}

LogicalResult PlanDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attribute) {
  if (attribute.getName() == PlanDialect::kBackendsAttrName) {
    auto backendsAttr = dyn_cast<ArrayAttr>(attribute.getValue());
    if (!backendsAttr)
      return op->emitError()
             << PlanDialect::kBackendsAttrName << " must be an array attribute";
    if (!isa<ModuleOp>(op))
      return op->emitError() << PlanDialect::kBackendsAttrName
                             << " must be attached to a module";
    return success();
  }
  if (attribute.getName() == PlanDialect::kShapeFuncAttrName) {
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << PlanDialect::kShapeFuncAttrName
                             << " must be attached to a function";
    if (!isa<FlatSymbolRefAttr>(attribute.getValue()))
      return op->emitError() << PlanDialect::kShapeFuncAttrName
                             << " must be a FlatSymbolRefAttr";
    return success();
  }
  if (attribute.getName() == PlanDialect::kShapeFuncMarkerAttrName) {
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << PlanDialect::kShapeFuncMarkerAttrName
                             << " must be attached to a function";
    if (!isa<UnitAttr>(attribute.getValue()))
      return op->emitError()
             << PlanDialect::kShapeFuncMarkerAttrName << " must be a UnitAttr";
    return success();
  }
  return success();
}

LogicalResult PlanDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute attribute) {
  if (attribute.getName() == PlanDialect::kValueBoundsAttrName) {
    auto boundsAttr = dyn_cast<BoundsAttr>(attribute.getValue());
    if (!boundsAttr || !boundsAttr.isValueBound())
      return op->emitError()
             << "expected named attribute \""
             << PlanDialect::kValueBoundsAttrName
             << "\" to be a \"#plan.bounds\" attribute containing value bounds";

    return verifyBoundsAttribute(op, argIndex, boundsAttr, attribute.getName());
  }

  if (attribute.getName() == PlanDialect::kFuncTargetKind) {
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << PlanDialect::kFuncTargetKind
                             << " must decorate a function argument";
    return success();
  }

  if (attribute.getName() == PlanDialect::kShapeBoundsAttrName) {
    auto boundsAttr = dyn_cast<BoundsAttr>(attribute.getValue());
    if (!boundsAttr || !boundsAttr.isShapeBound())
      return op->emitError()
             << "expected named attribute \""
             << PlanDialect::kShapeBoundsAttrName
             << "\" to be a \"#plan.bounds\" attribute containing shape bounds";
    return verifyBoundsAttribute(op, argIndex, boundsAttr, attribute.getName());
  }

  if (attribute.getName() == PlanDialect::kResultArgAttrName) {
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << PlanDialect::kResultArgAttrName
                             << " must be attached to a function";
    // Check attribute has i32 value
    auto resultIdx = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!resultIdx || !resultIdx.getType().isInteger(32))
      return op->emitError() << "expected " << PlanDialect::kResultArgAttrName
                             << " attribute to have i32 value";
    return success();
  }

  if (attribute.getName() == PlanDialect::kDonationArgAttrName) {
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << PlanDialect::kDonationArgAttrName
                             << " must decorate a function argument";
    // Check attribute has i32 value
    auto resultIdx = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!resultIdx || !resultIdx.getType().isInteger(32))
      return op->emitError() << "expected " << PlanDialect::kDonationArgAttrName
                             << " attribute to have i32 value";
  }

  return success();
}

void PlanDialect::registerAttributes() {
  addAttributesExt<
#define GET_ATTRDEF_LIST
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.cpp.inc"
      >();

  // We don't use the generated attribute printer/parser.
  (void)&generatedAttributePrinter;
  (void)&generatedAttributeParser;
}

//===----------------------------------------------------------------------===//
// Compiler-Runtime Interface Functions
//===----------------------------------------------------------------------===//

void plan::assignInitialSlotNumbers(OpBuilder &builder,
                                    FunctionOpInterface func) {
  ArrayRef<Type> resultTypes = func.getResultTypes();
  unsigned slotIndex = 0;
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    func.setResultAttr(i, plan::PlanDialect::kResultArgAttrName,
                       builder.getI32IntegerAttr(slotIndex++));
  }
}
