//===- Plan.cpp  ----------------------------------------------------------===//
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
/// Definitions of Plan dialect operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
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

//===----------------------------------------------------------------------===//
// BoundsAttr
//===----------------------------------------------------------------------===//

LogicalResult BoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 plan::BoundsKind kind,
                                 DenseI64ArrayAttr min_shape,
                                 DenseI64ArrayAttr max_shape,
                                 DenseElementsAttr min_values,
                                 DenseElementsAttr max_values) {
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
                   " min_values[{0}] = {1} > max_values[{0}] = {2}",
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
      mlir::computeSuffixProduct(min_values.getType().getShape());

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
  assert(kind == BoundsKind::Shape ||
         kind == BoundsKind::Value && "expected shape or value kind");
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

//===----------------------------------------------------------------------===//
// InlineGroupOp
//===----------------------------------------------------------------------===//

LogicalResult InlineGroupOp::verify() {
  YieldOp yield = getYield();

  if (yield->getNumOperands() != getNumResults())
    return emitOpError() << "expected terminator to yield " << getNumResults()
                         << " values but got " << yield.getNumOperands();

  if (yield->getOperandTypes() != getResultTypes())
    return emitOpError() << "expected types of yielded operands ("
                         << yield.getOperandTypes()
                         << ") to equal types of results (" << getResultTypes()
                         << ")";

  return success();
}

void InlineGroupOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the InlineGroupOp, branch into the body.
  if (point.isParent()) {
    regions.assign({RegionSuccessor(&getRegion())});
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.assign({RegionSuccessor(getResults())});
}

//===----------------------------------------------------------------------===//
// InlineClosedGroupOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyBoundsAttr(StringRef argOrResult, unsigned idx, Type type,
                 BoundsAttr boundsAttr,
                 llvm::function_ref<InFlightDiagnostic()> emitOpError) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    if (boundsAttr.isNone())
      return success();
    if (boundsAttr.isShapeBound()) {
      int64_t boundsLength = boundsAttr.getMinShape().size();
      if (std::max<int64_t>(shapedType.getRank(), 1) !=
          std::max<int64_t>(boundsLength, 1))
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
      Type elType = boundsAttr.getMinValues().getElementType();
      if (elType != shapedType.getElementType())
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected element type of value bounds elements (" << elType
               << ") to be compatible with the type (" << type << ")";
      if (boundsAttr.getMinValues().getType().getShape() !=
          shapedType.getShape())
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected type of values bounds elements ("
               << boundsAttr.getMinValues().getType()
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
      int64_t numEls = boundsAttr.getMinValues().getNumElements();
      if (numEls != 1)
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected number of values bounds elements (" << numEls
               << ") to equal number of elements of the type (1)";
      Type elType = boundsAttr.getMinValues().getElementType();
      if (elType != type)
        return emitOpError()
               << argOrResult << " #" << idx
               << " expected element type of value bounds elements (" << elType
               << ") to be compatible with the type (" << type << ")";
    }
  }
  // For all other types, the bounds kind must be none.
  if (!boundsAttr.isNone())
    return emitOpError() << "expected only 'none' bounds for type " << type;
  return success();
}

LogicalResult InlineClosedGroupOp::verify() {
  SmallVector<BoundsAttr> inputAttrs =
      llvm::to_vector(getInputAttrs().getAsRange<BoundsAttr>());
  if (inputAttrs.size() != getInputs().size())
    return emitOpError("expected number of inputs (")
           << getInputs().size()
           << " to equal the number of input_attrs BoundsAttrs ("
           << inputAttrs.size() << ")";

  for (auto [idx, type] : llvm::enumerate(TypeRange(getInputs()))) {
    BoundsAttr boundsAttr = inputAttrs[idx];
    if (failed(verifyBoundsAttr("input argument", idx, type, boundsAttr,
                                [&]() { return emitOpError(); })))
      return failure();
  }

  return success();
}

void InlineClosedGroupOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the InlineClosedGroupOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

OperandRange
InlineClosedGroupOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperands();
}

void InlineClosedGroupOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  assert(region.front().getNumArguments() == getInputs().size() &&
         "expected one block arg for each input argument");
  for (BlockArgument arg : region.front().getArguments()) {
    setNameFn(arg, "in");
  }
}

void InlineClosedGroupOp::build(OpBuilder &b, OperationState &state,
                                TypeRange resultTypes, Attribute target,
                                ValueRange inputs,
                                ArrayRef<BoundsAttr> input_attrs) {
  state.addTypes(resultTypes);
  state.addOperands(inputs);
  state.getOrAddProperties<Properties>().target = target;
  state.getOrAddProperties<Properties>().setInputAttrs(b.getArrayAttr(
      SmallVector<Attribute>(input_attrs.begin(), input_attrs.end())));
  Region *body = state.addRegion();
  auto getLocs = [](ValueRange r) {
    SmallVector<Location> locs;
    locs.reserve(r.size());
    for (Value v : r)
      locs.push_back(v.getLoc());
    return locs;
  };
  (void)body->emplaceBlock();
  body->addArguments(TypeRange(inputs), getLocs(inputs));
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::build(OpBuilder &b, OperationState &result) {
  build(b, result, {});
}

//===----------------------------------------------------------------------===//
// WithShapeOp
//===----------------------------------------------------------------------===//

OpFoldResult WithShapeOp::fold(FoldAdaptor adaptor) {
  if (getOperand().getType().hasStaticShape())
    return getOperand();
  return {};
}

LogicalResult WithShapeOp::verify() {
  RankedTensorType operandType = getOperand().getType();
  if (static_cast<int64_t>(getShape().size()) != operandType.getRank())
    return emitOpError() << "expected number of shape dimension extent values ("
                         << getShape().size()
                         << ") to equal the operand type and result type rank ("
                         << operandType.getRank() << ")";

  // Detect any obvious errors which can be seen from static dims.
  for (auto [idx, dim] : llvm::enumerate(operandType.getShape())) {
    if (ShapedType::isDynamic(dim))
      continue;
    IntegerAttr attr{};
    if (matchPattern(getShape()[idx], m_Constant(&attr))) {
      if (attr.getInt() != dim)
        emitOpError()
            << "dimension #" << idx << " is equal to " << dim
            << ", but the corresponding index value can be constant-folded to "
            << attr;
    }
  }

  return success();
}

namespace {
/// If any of the dimension operands of the `plan.with_shape` operation are
/// IndexType and produced by `arith.index_cast`, then just replace the use with
/// the operand of the cast. Using the result of a cast to `IndexType` doesn't
/// give any useful information, since we will always lower `IndexType` to an
/// integer with bit-width at least as wide as the input IR's representation of
/// shape values. However, casting from IndexType to a more specific type could
/// technically have have a truncating semantic, so we don't absorb those casts.
struct WithShapeAbsorbIndexCastPattern : OpRewritePattern<WithShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WithShapeOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::pair<unsigned, Value>> sparseUpdates;
    for (auto [idx, value] : llvm::enumerate(op.getShape())) {
      auto indexCast = value.getDefiningOp<arith::IndexCastOp>();
      if (!indexCast || !indexCast.getType().isIndex())
        continue;
      sparseUpdates.push_back(std::make_pair(idx, indexCast.getOperand()));
    }

    if (sparseUpdates.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto [idx, replacement] : sparseUpdates)
        op.getShapeMutable()[idx].assign(replacement);
    });

    return success();
  }
};
} // namespace

void WithShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<WithShapeAbsorbIndexCastPattern>(context);
}

//===----------------------------------------------------------------------===//
// WithValuesOp
//===----------------------------------------------------------------------===//

static ParseResult
parseWithValuesTypes(OpAsmParser &parser,
                     ArrayRef<OpAsmParser::UnresolvedOperand> elements,
                     Type &operandType, SmallVectorImpl<Type> &elementsTypes) {
  if (parser.parseType(operandType))
    return ParseResult::failure();

  Type elType = getElementTypeOrSelf(operandType);
  elementsTypes.assign(elements.size(), elType);
  return ParseResult::success();
}

static void printWithValuesTypes(OpAsmPrinter &printer, Operation *, ValueRange,
                                 Type operandType, TypeRange elementsTypes) {
  printer << operandType;
}

LogicalResult WithValuesOp::verify() {
  if (static_cast<int64_t>(getElements().size()) != getType().getNumElements())
    return emitOpError("expected number of 'elements' (")
           << getElements().size()
           << ") to equal volume of the result's tensor type ("
           << getType().getNumElements() << ")";

  return success();
}

OpFoldResult WithValuesOp::fold(FoldAdaptor adaptor) {
  if (getType().getNumElements() == 0)
    return getOperand();
  return {};
}

TensorKind WithValuesOp::getStaticOperandTensorKind(OpOperand &operand) {
  return TensorKind::Unknown;
}

void WithValuesOp::inferOperandKind(
    ArrayRef<TensorKindLattice *> operands,
    ArrayRef<const TensorKindLattice *> results,
    llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) {
  assert(results.size() == 1 && "expected one result");
  if (results[0] && !results[0]->getValue().isUninitialized())
    setOperandKind(getOperandMutable(), results[0]->getValue().getKind());
}

//===----------------------------------------------------------------------===//
// PlanDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with func operations.
struct PlanInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// `tensorrt.enqueue` cannot be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return false;
  }

  /// All operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All functions can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd attributes definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.cpp.inc"

#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum definition.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void PlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsTypes.cpp.inc"
      >();

  addAttributesExt<
#define GET_ATTRDEF_LIST
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.cpp.inc"
      >();

  // We don't use the generated attribute printer/parser.
  (void)&generatedAttributePrinter;
  (void)&generatedAttributeParser;

  addInterfaces<PlanInlinerInterface>();
}

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
