//===- CreateClosedRegions.cpp --------------------------------------------===//
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
/// Implementation of the `plan-create-closed-regions` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-common/Utils/RegionUtils.h"
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "plan-create-closed-regions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

namespace mlir::plan {
#define GEN_PASS_DEF_CREATECLOSEDREGIONSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Shape/bounds analysis utilities
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<int64_t>>
getShapeBoundsForValue(Value v, presburger::BoundType boundType,
                       DataFlowSolver &solver) {
  assert((boundType == presburger::BoundType::UB ||
          boundType == presburger::BoundType::LB) &&
         "expected either UB or LB bound type");
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  if (rtt.hasStaticShape())
    return llvm::to_vector(rtt.getShape());

  LLVM_DEBUG({
    DBGS() << "trying to infer "
           << (boundType == presburger::BoundType::UB ? "upper" : "lower")
           << " bound for the shape of ";
    if (isa<OpResult>(v))
      llvm::dbgs() << v;
    else
      llvm::dbgs()
          << v << " of "
          << *llvm::cast<BlockArgument>(v).getParentBlock()->getParentOp();
    llvm::dbgs() << "\n";
  });

  const ShapeBoundsLattice *lattice = solver.lookupState<ShapeBoundsLattice>(v);
  if (!lattice || lattice->getValue().isUninitialized())
    return failure();

  LLVM_DEBUG(DBGS() << "found lattice value: " << lattice->getValue() << "\n");

  return llvm::map_to_vector(lattice->getValue().getValue(),
                             [&](const ConstantIntRanges &r) {
                               if (boundType == presburger::BoundType::UB)
                                 return r.smax().getSExtValue();
                               return r.smin().getSExtValue();
                             });
}

//===----------------------------------------------------------------------===//
// Main pass logic
//===----------------------------------------------------------------------===//

namespace {
enum class DestinationOperandMaterializationStrategy {
  /// The destination operand was materialized with a static or dynamic shape
  /// that matches the shape of the output,
  /// A `tensor.empty` is used to create the destination tensor above the
  /// `plan.inline_group` operation, and the result of the
  /// `plan.inline_closed_group` can be used without slicing it.
  ExactShape,
  /// We materialized a tensor based on the upper bound derived from one of our
  /// analysese. The tensor could be static or dynamic based on where we
  /// computed the upper bound. This indicates to the algorithm below that we
  /// need to slice the result of our `plan.inline_closed_group` operation to
  /// cut
  /// the tensor  to the exact size.
  UpperBound
};

struct DestinationOperandMaterializationResult {
  DestinationOperandMaterializationStrategy strategy;
  Value destinationOperand;
  /// The exact shape of the output.
  std::optional<SmallVector<OpFoldResult>> exactShape = {};
  SmallVector<int64_t> constantShapeUpperBound;
  SmallVector<int64_t> constantShapeLowerBound;
};
} // namespace

static FailureOr<SmallVector<OpFoldResult>>
getShapeAndVerifyDefinedAbove(RewriterBase &rewriter,
                              plan::InlineGroupOp regionOp,
                              plan::WithShapeOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<OpFoldResult> result;
  result.reserve(op.getShape().size());

  for (Value v : op.getShape()) {
    IntegerAttr constVal{};
    if (matchPattern(v, m_Constant(&constVal))) {
      result.push_back(rewriter.getIndexAttr(constVal.getInt()));
      continue;
    }

    // Check for DDS cases.
    if (OpResult res = dyn_cast<OpResult>(v)) {
      if (!res.getOwner()->getParentRegion()->isProperAncestor(
              &regionOp.getBody()))
        return failure();
      if (v.getType().isIndex()) {
        result.push_back(v);
        continue;
      }
      rewriter.setInsertionPointAfterValue(v);
      result.push_back(rewriter
                           .create<arith::IndexCastOp>(
                               v.getLoc(), rewriter.getIndexType(), v)
                           .getResult());
      continue;
    }

    auto blockArg = cast<BlockArgument>(v);
    if (!blockArg.getOwner()->getParent()->isProperAncestor(
            &regionOp.getBody()))
      return failure();

    if (v.getType().isIndex()) {
      result.push_back(v);
      continue;
    }

    rewriter.setInsertionPointAfterValue(v);
    result.push_back(
        rewriter
            .create<arith::IndexCastOp>(v.getLoc(), rewriter.getIndexType(), v)
            .getResult());
  }

  return result;
}

static FailureOr<DestinationOperandMaterializationResult>
materializeDestinationOperand(RewriterBase &rewriter, Location loc,
                              plan::InlineGroupOp op, unsigned resultIdx,
                              DataFlowSolver &solver) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  op->getParentOfType<ModuleOp>();
  auto rtt = dyn_cast<RankedTensorType>(op.getResultTypes()[resultIdx]);

  if (rtt.hasStaticShape()) {
    DestinationOperandMaterializationResult result;
    result.strategy = DestinationOperandMaterializationStrategy::ExactShape,
    result.destinationOperand =
        rewriter
            .create<tensor::EmptyOp>(loc, rtt.getShape(), rtt.getElementType())
            .getResult();
    result.constantShapeLowerBound = llvm::to_vector(rtt.getShape());
    result.constantShapeUpperBound = llvm::to_vector(rtt.getShape());
    return result;
  };

  // Any time the result has dynamic shape, we should be yielding a
  // `plan.with_shape` result.
  auto withShapeOp =
      op.getYieldedValueForResult(resultIdx).getDefiningOp<plan::WithShapeOp>();
  if (!withShapeOp)
    return emitError(op.getLoc()) << "expected cluster to yield the result of "
                                     "a 'plan.with_shape' operation but got "
                                  << op.getYieldedValueForResult(resultIdx);

  // Check that all dimensions are constants or defined above:
  FailureOr<SmallVector<OpFoldResult>> exactShape =
      getShapeAndVerifyDefinedAbove(rewriter, op, withShapeOp);
  if (failed(exactShape))
    return op->emitOpError()
           << "expected the shape calculation to be materialized above the "
              "cluster region but some dynamic dimension "
              " extent value is defined within the cluster region; this "
              "indicates that the result shape is in the so-called "
              "'data-dependent dynamic shapes' regime, "
              "which is currently not supported for TensorRT cluster regions";

  DestinationOperandMaterializationResult result;
  result.destinationOperand = Value{};
  result.exactShape = std::move(*exactShape);

  FailureOr<SmallVector<int64_t>> minShape =
      getShapeBoundsForValue(op.getYieldedValueForResult(resultIdx),
                             presburger::BoundType::LB, solver);
  if (succeeded(minShape)) {
    result.constantShapeLowerBound = *minShape;
  } else {
    LLVM_DEBUG(
        DBGS() << "failed to derive shape lower bound, filling with zeros\n");
    result.constantShapeLowerBound = SmallVector<int64_t>(rtt.getRank(), 0);
  }

  // If there are not static shapes, then first we can try to query the
  // value bounds analysis for the max shape.
  FailureOr<SmallVector<int64_t>> maxShape =
      getShapeBoundsForValue(op.getYieldedValueForResult(resultIdx),
                             presburger::BoundType::UB, solver);

  if (failed(maxShape))
    // TODO: try materializing non-static upper bound
    return failure();

  int64_t shapeProduct = mlir::computeProduct(*maxShape);

  LLVM_DEBUG(DBGS() << llvm::formatv(
                 "computed max shape = {0:$[, ]}; product={1}\n",
                 llvm::make_range(maxShape->begin(), maxShape->end()),
                 shapeProduct));

  // Any shaped type that is < 0 is invalid as it indicates an
  // overflow from signed integer arithmetic.
  if (llvm::any_of(*maxShape, [](int64_t dim) { return dim < 0; }) ||
      shapeProduct < 0)
    return op->emitOpError() << llvm::formatv(
               "failed to calculate upper bound for cluster result #{0}; got "
               "max shape {1:$[, ]} which has linear size {2}",
               resultIdx, llvm::make_range(maxShape->begin(), maxShape->end()),
               shapeProduct);

  // We allocate a linearized buffer since the output of dynamic regions
  // will create a dynamic tensor with a packed layout.
  {
    Value maxEmptyLinearValue = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{shapeProduct}, rtt.getElementType());

    // materialize the linearized size.
    SmallVector<AffineExpr> shape;
    shape.reserve(maxShape->size());
    for (unsigned symbolIdx = 0; symbolIdx < maxShape->size(); symbolIdx++)
      shape.push_back(rewriter.getAffineSymbolExpr(symbolIdx));

    OpFoldResult linearizedActualSize = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mlir::computeProduct(rewriter.getContext(), shape),
        *result.exactShape);

    auto sliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, maxEmptyLinearValue,
        /*offsets=*/
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
        /*sizes=*/ArrayRef<OpFoldResult>{linearizedActualSize},
        /*strides=*/
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
    if (sliceOp.getType() == op->getResultTypes()[resultIdx]) {
      result.destinationOperand = sliceOp.getResult();
    } else {
      SmallVector<Value> shapeElements =
          getValueOrCreateConstantIndexOp(rewriter, loc, *result.exactShape);
      Value shapeValue =
          rewriter.create<tensor::FromElementsOp>(loc, shapeElements);
      result.destinationOperand = rewriter.create<tensor::ReshapeOp>(
          loc, op.getResultTypes()[resultIdx], sliceOp.getResult(), shapeValue);
    }
  }

  result.constantShapeUpperBound = *maxShape;
  // Cast back to a dynamic shape so that it matches the originally required
  // type.
  result.strategy = DestinationOperandMaterializationStrategy::UpperBound;
  return result;
}

/// Remap relevant analysis state of type T from `original` to `replacement`.
template <typename T>
static void remapLatticeState(DataFlowSolver &solver, Value original,
                              Value replacement) {
  if (const T *lattice = solver.lookupState<T>(original)) {
    T *latticeReplacement = solver.getOrCreateState<T>(replacement);
    latticeReplacement->getValue() = lattice->getValue();
  }
}

/// Remap relevant analysis states from `originals` to `replacements`.
static void remapAnalysisState(DataFlowSolver &solver, ValueRange originals,
                               ValueRange replacements) {
  for (auto [original, replacement] :
       llvm::zip_equal(originals, replacements)) {
    remapLatticeState<TensorKindLattice>(solver, original, replacement);
    remapLatticeState<ShapeBoundsLattice>(solver, original, replacement);
    remapLatticeState<IntegerValueRangeLattice>(solver, original, replacement);
    remapLatticeState<Lattice<ConstantValue>>(solver, original, replacement);
    remapLatticeState<TensorValueBoundsLattice>(solver, original, replacement);
  }
}

static FailureOr<SmallVector<BoundsAttr>>
getInputAttributes(RewriterBase &rewriter, DataFlowSolver &solver, Location loc,
                   ValueRange inputs) {
  // Compute input tensor kinds.
  SmallVector<TensorKindInfo> inputTensorKinds;
  inputTensorKinds.reserve(inputs.size());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (!isa<RankedTensorType>(input.getType())) {
      inputTensorKinds.push_back(TensorKindInfo());
      continue;
    }
    const TensorKindLattice *tensorKindLattice =
        solver.lookupState<TensorKindLattice>(input);
    if (!tensorKindLattice || tensorKindLattice->getValue().isUninitialized()) {
      inputTensorKinds.push_back(TensorKindInfo());
      continue;
    }

    inputTensorKinds.push_back(tensorKindLattice->getValue());
  }

  // Create the shape profile attributes for the inputs.
  SmallVector<BoundsAttr> inputAttrs;
  inputAttrs.reserve(inputTensorKinds.size());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto tensorType = dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType || inputTensorKinds[idx].isUninitialized()) {
      inputAttrs.push_back(BoundsAttr::get(rewriter.getContext()));
      continue;
    }

    // Check if we need tensor value bounds (shape/host tensor bounds).
    if (inputTensorKinds[idx].isHostVisible()) {
      const auto *lattice = solver.lookupState<TensorValueBoundsLattice>(input);
      if (!lattice)
        return emitError(loc)
               << "host-visible input operand #" << idx << " of type "
               << input.getType()
               << " does not have value bounds information attached";

      plan::BoundsArray bounds = lattice->getValue();
      if (bounds.isUninitialized()) {
        // TensorKindAnalysis may classify a tensor as being host-visible even
        // if it does not have integer or index element type. This can occur if
        // e.g. there is a `tensor.extract` present, such that must be lowered
        // into a host access. However, our bounds analysis and attributes only
        // deal with integer tensors, so populate an empty bounds attribute if
        // this is not an integer tensor.
        if (!isa<IntegerType, IndexType>(tensorType.getElementType())) {
          inputAttrs.push_back(BoundsAttr::get(rewriter.getContext()));
          continue;
        }
        bounds = BoundsArray::getMaxRangeForValueBounds(input);
      }
      auto [lbAttr, ubAttr] = bounds.getAsElementsAttr(tensorType);

      BoundsAttr boundsAttr = BoundsAttr::getChecked(
          loc, rewriter.getContext(), BoundsKind::Value, DenseI64ArrayAttr{},
          DenseI64ArrayAttr{}, static_cast<ElementsAttr>(lbAttr),
          static_cast<ElementsAttr>(ubAttr));
      if (!boundsAttr)
        return failure();
      inputAttrs.push_back(boundsAttr);
      continue;
    }

    // We don't need shape bounds attributes for statically shaped tensors.
    if (tensorType.hasStaticShape()) {
      inputAttrs.push_back(BoundsAttr::get(rewriter.getContext()));
      continue;
    }

    // Get the upper bounds of the shape.
    FailureOr<SmallVector<int64_t>> ub =
        getShapeBoundsForValue(input, presburger::BoundType::UB, solver);
    if (failed(ub))
      return emitError(input.getLoc())
             << "failed to derive upper bound for " << input;

    // Get the lower bound of the shape.
    FailureOr<SmallVector<int64_t>> lb =
        getShapeBoundsForValue(input, presburger::BoundType::LB, solver);
    if (failed(lb))
      lb = SmallVector<int64_t>(tensorType.getRank(), 0);
    BoundsAttr boundsAttr = BoundsAttr::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(loc), rewriter.getContext(),
        BoundsKind::Shape, ArrayRef(*lb), ArrayRef(*ub));
    if (!boundsAttr)
      return failure();
    inputAttrs.push_back(boundsAttr);
  }

  return inputAttrs;
}

static LogicalResult materializeDestinationOperands(
    RewriterBase &rewriter, plan::InlineGroupOp op, DataFlowSolver &solver,
    SmallVectorImpl<DestinationOperandMaterializationResult>
        &destinationOperands) {
  destinationOperands.reserve(op.getNumResults());
  for (OpResult res : op->getOpResults()) {
    FailureOr<DestinationOperandMaterializationResult> destResult =
        materializeDestinationOperand(rewriter, res.getLoc(), op,
                                      res.getResultNumber(), solver);
    if (failed(destResult))
      return emitError(res.getLoc())
             << "failed to materialize destination operand of type "
             << res.getType();
    destinationOperands.push_back(*destResult);
  }
  return success();
}

// Helper function to create DPS closed group op.
static LogicalResult createInlineClosedGroupOp(
    RewriterBase &rewriter, plan::InlineGroupOp op, DataFlowSolver &solver,
    const ValueRange &inputs,
    ArrayRef<DestinationOperandMaterializationResult> destinationOperands) {
  InlineClosedGroupOp closedGroupOp = rewriter.create<InlineClosedGroupOp>(
      op.getLoc(), /*target=*/op.getTarget(),
      /*inputs=*/inputs,
      /*outs=*/
      llvm::map_to_vector(destinationOperands,
                          [](const auto &x) { return x.destinationOperand; }));

  rewriter.inlineBlockBefore(
      &op.getRegion().front(), &closedGroupOp.getRegion().front(),
      closedGroupOp.getRegion().front().end(),
      closedGroupOp.getRegion().getArguments().take_front(
          op.getRegion().getNumArguments()));

  SmallVector<Value> replacements;
  replacements.reserve(destinationOperands.size());
  for (auto [newResult, destOperand, originalType] :
       llvm::zip_equal(closedGroupOp->getResults(), destinationOperands,
                       op->getResultTypes())) {
    if (destOperand.strategy ==
        DestinationOperandMaterializationStrategy::ExactShape) {
      replacements.push_back(newResult);
      continue;
    }

    assert(destOperand.exactShape &&
           "expected materialized shape values to be provided for "
           "this materialization strategy");

    replacements.push_back(newResult);
  }

  // Since we are about to replace values that may be inputs to other regions
  // ops, we need to update the solver to populate the replacement TensorKind
  // information.
  remapAnalysisState(solver, op->getResults(), replacements);
  rewriter.replaceOp(op, replacements);

  // Create the closed region result profile attrs.
  SmallVector<BoundsAttr> resultAttrs;
  for (const DestinationOperandMaterializationResult &dest :
       destinationOperands) {
    auto boundsAttr = BoundsAttr::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(closedGroupOp.getLoc()),
        rewriter.getContext(), BoundsKind::Shape,
        ArrayRef(dest.constantShapeLowerBound),
        ArrayRef(dest.constantShapeUpperBound));
    if (!boundsAttr)
      return failure();
    resultAttrs.push_back(boundsAttr);
  }
  closedGroupOp.setResAttrsAttr(resultAttrs);

  // Create the closed region input profilw attrs.
  FailureOr<SmallVector<BoundsAttr>> inputAttr = getInputAttributes(
      rewriter, solver, closedGroupOp->getLoc(), closedGroupOp.getInputs());

  if (failed(inputAttr))
    return emitError(closedGroupOp.getLoc())
           << "failed to compute input attribute ";

  closedGroupOp.setInputAttrsAttr(*inputAttr);

  return success();
}

// Helper function to create non-DPS closed group op.
static LogicalResult
createInlineClosedAllocGroupOp(RewriterBase &rewriter, plan::InlineGroupOp op,
                               DataFlowSolver &solver,
                               const SmallVector<Value> &inputs) {
  // Create a new closed group op and move blocks into it.
  InlineClosedAllocGroupOp closedGroupOp =
      rewriter.create<InlineClosedAllocGroupOp>(
          op.getLoc(), /*result type*/ op->getResultTypes(),
          /*target=*/op.getTarget(),
          /*inputs=*/inputs);

  rewriter.inlineBlockBefore(
      &op.getRegion().front(), &closedGroupOp.getRegion().front(),
      closedGroupOp.getRegion().front().end(),
      closedGroupOp.getRegion().getArguments().take_front(
          op.getRegion().getNumArguments()));

  // Since we are about to replace values that may be inputs to other regions
  // ops, we need to update the solver to populate the replacement TensorKind
  // information.
  remapAnalysisState(solver, op->getResults(), closedGroupOp->getResults());
  rewriter.replaceOp(op, closedGroupOp->getResults());

  // Create the closed region input profilw attrs.
  FailureOr<SmallVector<BoundsAttr>> inputAttr = getInputAttributes(
      rewriter, solver, closedGroupOp->getLoc(), closedGroupOp.getInputs());

  if (failed(inputAttr))
    return emitError(closedGroupOp.getLoc())
           << "failed to compute input attribute ";

  closedGroupOp.setInputAttrsAttr(*inputAttr);

  return success();
}

static LogicalResult
createClosedGroupOp(RewriterBase &rewriter, plan::InlineGroupOp op,
                    DataFlowSolver &solver,
                    bool disableDestinationStyleCallingConvention) {
  OpBuilder::InsertionGuard g(rewriter);

  CompilerBackendAttrInterface backend = op.getTarget();
  bool useDestinationStyleCallingConvention =
      !disableDestinationStyleCallingConvention &&
      backend.supportsDestinationStyleCallingConvention(op) &&
      backend.preferDestinationStyleCallingConvention(op) &&
      llvm::all_of(op->getResultTypes(), llvm::IsaPred<RankedTensorType>);

  // Materialize destination operands if not using non-DPS call convention.
  SmallVector<DestinationOperandMaterializationResult> destinationOperands;
  if (useDestinationStyleCallingConvention) {
    useDestinationStyleCallingConvention &=
        succeeded(materializeDestinationOperands(rewriter, op, solver,
                                                 destinationOperands));
  }

  // Make the region isolated from above. This captures the input operands.
  SmallVector<Value> inputs = mlir::createClosedRegion(
      rewriter, op.getRegion(), [&](Operation *producer) -> bool {
        return backend.shouldCloneProducer(op, producer);
      });

  rewriter.setInsertionPoint(op);

  // Create and populate the appropriate closed group op based on call
  // convention.
  if (useDestinationStyleCallingConvention)
    return createInlineClosedGroupOp(rewriter, op, solver, inputs,
                                     destinationOperands);
  return createInlineClosedAllocGroupOp(rewriter, op, solver, inputs);
}

namespace {
class CreateClosedRegionsPass
    : public plan::impl::CreateClosedRegionsPassBase<CreateClosedRegionsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp op = getOperation();
    SmallVector<plan::InlineGroupOp> groupOps;
    MLIRContext *ctx = op->getContext();

    auto opFilterFn = [&](plan::InlineGroupOp groupOp) { return true; };

    // Filter by target for those that require DPS transform.
    if (testPreWalkOrder) {
      op->walk<WalkOrder::PreOrder>([&](plan::InlineGroupOp groupOp) {
        if (opFilterFn(groupOp))
          groupOps.push_back(groupOp);
      });
    } else {
      op->walk<WalkOrder::PostOrder>([&](plan::InlineGroupOp groupOp) {
        if (opFilterFn(groupOp))
          groupOps.push_back(groupOp);
      });
    }

    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<ShapeIntegerRangeAnalysis>();
    solver.load<ShapeBoundsForwardAnalysis>();
    solver.load<ShapeBoundsBackwardsAnalysis>(symbolTable);
    solver.load<TensorKindAnalysis>(symbolTable);
    solver.load<TensorValueBoundsAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return;

    IRRewriter rewriter(ctx);
    for (InlineGroupOp groupOp : llvm::make_early_inc_range(groupOps)) {
      if (failed(createClosedGroupOp(rewriter, groupOp, solver,
                                     disableDestinationStyleCallingConvention)))
        return signalPassFailure();
    }
  }
};
} // namespace
