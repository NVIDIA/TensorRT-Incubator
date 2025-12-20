//===- Plan.cpp  ----------------------------------------------------------===//
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
/// Definitions of Plan dialect operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::plan;
using namespace mtrt::compiler;

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

static LogicalResult verifyBoundsAttrs(Operation *op, ValueRange operands,
                                       ArrayAttr attrsArray, StringRef attrName,
                                       StringRef boundName) {
  SmallVector<BoundsAttr> attrs =
      llvm::to_vector(attrsArray.getAsRange<BoundsAttr>());
  if (attrs.size() != operands.size())
    return op->emitOpError("expected number of ")
           << attrName << " (" << operands.size() << ") to equal the number of "
           << boundName << " BoundsAttrs (" << attrs.size() << ")";

  for (auto [idx, type] : llvm::enumerate(TypeRange(operands))) {
    BoundsAttr boundsAttr = attrs[idx];
    if (failed(plan::detail::verifyBoundsAttr(
            attrName, idx, type, boundsAttr,
            [&]() { return op->emitOpError(); })))
      return failure();
  }

  return success();
}

LogicalResult InlineClosedGroupOp::verify() {
  if (failed(verifyBoundsAttrs(getOperation(), getInputs(), getInputAttrs(),
                               "inputs", "input_attrs")))
    return failure();

  if (failed(verifyBoundsAttrs(getOperation(), getResults(), getResAttrs(),
                               "results", "result_attrs")))
    return failure();

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
  assert(region.front().getNumArguments() ==
             getInputs().size() + getOuts().size() &&
         "expected one block arg for each input and destination argument");
  unsigned numInputs = getInputs().size();
  for (BlockArgument arg : region.front().getArguments()) {
    StringRef name = arg.getArgNumber() < numInputs ? "in" : "out";
    setNameFn(arg, name);
  }
}

void InlineClosedGroupOp::build(OpBuilder &b, OperationState &state,
                                Attribute target, ValueRange inputs,
                                ValueRange outs,
                                ArrayRef<BoundsAttr> input_attrs,
                                ArrayRef<BoundsAttr> result_attrs) {
  state.addOperands(inputs);
  state.addOperands(outs);
  state.getOrAddProperties<Properties>().target = target;
  state.getOrAddProperties<Properties>().setInputAttrs(b.getArrayAttr(
      SmallVector<Attribute>(input_attrs.begin(), input_attrs.end())));
  state.getOrAddProperties<Properties>().setResAttrs(b.getArrayAttr(
      SmallVector<Attribute>(result_attrs.begin(), result_attrs.end())));

  llvm::copy(
      ArrayRef<int32_t>{static_cast<int32_t>(inputs.size()),
                        static_cast<int32_t>(outs.size())},
      state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
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
  body->addArguments(TypeRange(outs), getLocs(outs));
  state.addTypes(TypeRange(outs));
}

//===----------------------------------------------------------------------===//
// InlineClosedAllocGroupOp
//===----------------------------------------------------------------------===//

LogicalResult InlineClosedAllocGroupOp::verify() {
  Operation *op = getOperation();
  return verifyBoundsAttrs(op, getInputs(), getInputAttrs(), "inputs",
                           this->getInputAttrsAttrName().strref());
}

void InlineClosedAllocGroupOp::getSuccessorRegions(
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
InlineClosedAllocGroupOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperands();
}

void InlineClosedAllocGroupOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  assert(region.getNumArguments() == getInputs().size() &&
         "expected one block arg for each input argument");
  for (BlockArgument arg : region.getArguments())
    setNameFn(arg, "in");
}

void InlineClosedAllocGroupOp::build(OpBuilder &b, OperationState &state,
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

void WithValuesOp::inferResultRangesFromOptional(
    ArrayRef<IntOrTensorValueRange> argBounds,
    SetTensorValueLatticeFn setResultRanges) {
  if (!BoundsArray::shouldAnalyzeValueBounds(getResult())) {
    setResultRanges(getResult(), BoundsArray());
    return;
  }

  const auto *tensorBounds = argBounds.front().dyn_cast<const BoundsArray *>();

  SmallVector<ConstantIntRanges> ranges;
  ArrayRef<IntOrTensorValueRange> scalarBounds = argBounds.drop_front();
  ranges.reserve(scalarBounds.size());
  for (unsigned i = 0, e = scalarBounds.size(); i < e; i++) {
    const auto *scalarBound =
        scalarBounds[i].dyn_cast<const IntegerValueRange *>();
    bool scalarIsInvalid = !scalarBound || scalarBound->isUninitialized();
    if (tensorBounds && !tensorBounds->isUninitialized()) {
      assert(
          tensorBounds->getValue().size() == scalarBounds.size() &&
          "expected number of tensor bounds to equal number of scalar bounds");
      if (!scalarIsInvalid) {
        ranges.push_back(
            scalarBound->getValue().rangeUnion(tensorBounds->getValue()[i]));
        continue;
      }
      ranges.push_back(tensorBounds->getValue()[i]);
      continue;
    }

    if (!scalarIsInvalid) {
      ranges.push_back(scalarBound->getValue());
      continue;
    }

    setResultRanges(getResult(), BoundsArray());
    return;
  }

  setResultRanges(getResult(), BoundsArray(ranges));
}

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
// OptimizationBarrierOp
//===----------------------------------------------------------------------===//

bool OptimizationBarrierOp::bufferizesToMemoryRead(
    OpOperand &operand, const bufferization::AnalysisState &state) {
  // no reads occur.
  return false;
}

bool OptimizationBarrierOp::bufferizesToMemoryWrite(
    OpOperand &operand, const bufferization::AnalysisState &state) {
  // no writes occur.
  return false;
}

bufferization::AliasingValueList OptimizationBarrierOp::getAliasingValues(
    OpOperand &operand, const bufferization::AnalysisState &state) {
  // operands are tied to results.
  return {{getOperation()->getResult(operand.getOperandNumber()),
           bufferization::BufferRelation::Equivalent, /*isDefinite=*/true}};
}

LogicalResult OptimizationBarrierOp::bufferize(
    RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) {
  // Just forward input buffers as the  result buffers.
  SmallVector<Value> buffers;
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    FailureOr<Value> buffer =
        bufferization::getBuffer(rewriter, operand.get(), options, state);
    if (failed(buffer))
      return failure();
    buffers.push_back(*buffer);
  }
  bufferization::replaceOpWithBufferizedValues(rewriter, *this, buffers);

  return success();
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

LogicalResult TransferOp::verify() { return success(); }

OpFoldResult TransferOp::fold(FoldAdaptor adaptor) {
  if (getOperand().getType() == getResult().getType())
    return getOperand();
  if (auto producerOp = getOperand().getDefiningOp<TransferOp>()) {
    if (producerOp.getOperand().getType() == getType())
      return producerOp.getOperand();
  }
  return {};
}

namespace {
struct TensorEmptyAbsorbTransferPattern : OpRewritePattern<TransferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getOperand();
    auto emptyOp = source.getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        op, emptyOp.getType().getShape(), emptyOp.getType().getElementType(),
        emptyOp.getDynamicSizes(), op.getType().getEncoding());
    return success();
  }
};
} // namespace

void TransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<TensorEmptyAbsorbTransferPattern>(context);
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PlanDialect Definitions
//===----------------------------------------------------------------------===//

namespace {
class PlanDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  /// Tells MLIR assembly printer/parser that the BoundsAttr can be
  /// aliased using #bounds[num]. This make the IR more readable.
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<plan::BoundsAttr>(attr)) {
      os << "bounds";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void PlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.cpp.inc"
      >();

  registerTypes();
  registerAttributes();

  addInterfaces<PlanInlinerInterface, PlanDialectOpAsmInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, PlanDialect>();
}
