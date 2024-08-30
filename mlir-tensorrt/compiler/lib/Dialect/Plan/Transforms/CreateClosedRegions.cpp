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
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  Value destinationOperand;
  /// The exact shape of the output.
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

/// Determines whether a cluster being outlined should clone a constant or
/// pass constant by value.
static bool shouldCloneProducer(Operation *producer, Region &cluster) {
  if (!producer->hasTrait<OpTrait::ConstantLike>() ||
      producer->getNumResults() != 1)
    return false;
  RankedTensorType type =
      dyn_cast<RankedTensorType>(producer->getResultTypes().front());
  if (!type || !type.hasStaticShape())
    return false;

  // A value should be cloned if all of its uses are in the cluster.
  if (llvm::all_of(producer->getUsers(), [&](Operation *user) {
        return cluster.isAncestor(user->getParentRegion());
      }))
    return true;
  return type.getNumElements() *
             llvm::divideCeil(type.getElementTypeBitWidth(), 8) <
         100 * 1024 * 1024;
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

static LogicalResult createClosedGroupOp(RewriterBase &rewriter,
                                         plan::InlineGroupOp op,
                                         DataFlowSolver &solver) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = op.getLoc();

  // Make the region isolated from above. This captures the input operands.
  SmallVector<Value> inputs = makeRegionIsolatedFromAbove(
      rewriter, op.getRegion(), [&](Operation *producer) {
        return shouldCloneProducer(producer, op.getRegion());
      });

  rewriter.setInsertionPoint(op);

  // Create a new closed group op and move blocks into it.
  auto closedGroupOp = rewriter.create<InlineClosedGroupOp>(
      op.getLoc(), /*result type*/ op->getResultTypes(), /*target=*/op.getTarget(),
      /*inputs=*/inputs);

  rewriter.inlineBlockBefore(
      &op.getRegion().front(), &closedGroupOp.getRegion().front(),
      closedGroupOp.getRegion().front().end(),
      closedGroupOp.getRegion().getArguments().take_front(
          op.getRegion().getNumArguments()));

  DBGS() << "Number of inline op results: " << op->getNumResults() << "\n"; 
  DBGS() << "Number of closed group op results: " << closedGroupOp->getNumResults() << "\n"; 

  rewriter.replaceOp(op, closedGroupOp->getResults());

  SmallVector<TensorKindInfo> inputTensorKinds;
  inputTensorKinds.reserve(inputs.size());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (!isa<RankedTensorType>(input.getType())) {
      inputTensorKinds.push_back(TensorKindInfo());
      continue;
    }
    const TensorKindLattice *tensorKindLattice =
        solver.lookupState<TensorKindLattice>(input);
    if (!tensorKindLattice || tensorKindLattice->getValue().isUninitialized())
      return closedGroupOp->emitOpError("input operand #")
             << idx << " of type " << input.getType()
             << " does not have a TensorKind associated with it";
    inputTensorKinds.push_back(tensorKindLattice->getValue());
  }

  // Create the shape profile attributes for the inputs.
  SmallVector<BoundsAttr> inputAttrs;
  inputAttrs.reserve(inputTensorKinds.size());
  for (auto [idx, input] : llvm::enumerate(closedGroupOp.getInputs())) {
    auto tensorType = dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      inputAttrs.push_back(BoundsAttr::get(rewriter.getContext()));
      continue;
    }

    // Check if we need tensor value bounds (shape/host tensor bounds).
    if (inputTensorKinds[idx].isHostVisible()) {
      const auto *lattice = solver.lookupState<TensorValueBoundsLattice>(input);
      if (!lattice)
        return emitError(closedGroupOp.getLoc())
               << "host-visible input operand #" << idx << " of type "
               << input.getType()
               << " does not have value bounds information attached";

      plan::BoundsArray bounds = lattice->getValue();
      if (bounds.isUninitialized())
        bounds = BoundsArray::getMaxRangeForValueBounds(input);
      auto [lbAttr, ubAttr] = bounds.getAsElementsAttr(tensorType);

      BoundsAttr boundsAttr = BoundsAttr::getChecked(
          closedGroupOp.getLoc(), rewriter.getContext(), BoundsKind::Value,
          DenseI64ArrayAttr{}, DenseI64ArrayAttr{}, lbAttr, ubAttr);
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
  closedGroupOp.setInputAttrsAttr(inputAttrs);

  return success();
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

    auto opFilterFn = [](plan::InlineGroupOp groupOp) {
      return llvm::isa<TensorRTClusterKindAttr>(groupOp.getTarget());
    };

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
    for (InlineGroupOp groupOp : groupOps) {
      if (failed(createClosedGroupOp(rewriter, groupOp, solver)))
        return signalPassFailure();
    }
  }
};
} // namespace
