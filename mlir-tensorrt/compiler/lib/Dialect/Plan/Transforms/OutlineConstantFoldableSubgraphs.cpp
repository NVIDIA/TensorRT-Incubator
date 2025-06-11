//===- OutlineConstantFoldableSubgraphs.cpp -------------------------------===//
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
/// Implementation of the `outline-constant-foldable-subgraphs` pass.
///
//===----------------------------------------------------------------------===//

#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/DataFlowUtils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANOUTLINECONSTANTFOLDABLESUBGRAPHSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;

/// State for constant foldability analysis.
///     true
///      |
/// uninitialized(⊥)
/// We don't have top (⊤) element (generally, unknown information)
/// because constant foldability analysis is definitive on pure ops.
/// Note that we don't have false state.
namespace {
class ConstantFoldabilityState {
public:
  ConstantFoldabilityState(std::optional<bool> value = std::nullopt)
      : value(std::move(value)) {}

  bool isInitialized() const { return value.has_value(); }
  bool isUninitialized() const { return !value.has_value(); }
  static ConstantFoldabilityState getUninitialized() {
    return ConstantFoldabilityState{};
  }
  bool getKnownState() const {
    assert(isInitialized());
    return *value;
  }

  bool operator==(const ConstantFoldabilityState &rhs) const {
    return value == rhs.value;
  }

  static ConstantFoldabilityState join(const ConstantFoldabilityState &lhs,
                                       const ConstantFoldabilityState &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return lhs;
  }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "uninitialized";
      return;
    }
    os << *value;
  }

private:
  std::optional<bool> value;
};

class ConstantFoldabilityLattice
    : public dataflow::Lattice<ConstantFoldabilityState> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldabilityLattice)
  using Lattice::Lattice;
};

/// Implements forward dataflow analysis that find constant foldable
/// values. This is simple analysis that works only for pure ops. Operation
/// results are considered constant foldable, if all of its operands are
/// constant foldable, and it has no memory effects.
class SparseConstantFoldabilityAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          ConstantFoldabilityLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const ConstantFoldabilityLattice *> operands,
                 ArrayRef<ConstantFoldabilityLattice *> results) override {

    if (!isPure(op)) {
      setAllToEntryStates(results);
      return success();
    }

    // If op is constant, it is constant foldable.
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      ConstantFoldabilityLattice *lattice = results.front();
      propagateIfChanged(lattice,
                         lattice->join(ConstantFoldabilityState(true)));
      return success();
    }

    // For other operations, check if all operands are constant.
    bool areAllOperandsConstantFoldable = true;
    for (auto *operandLattice : operands) {
      if (operandLattice->getValue().isUninitialized())
        return success();
      areAllOperandsConstantFoldable &=
          operandLattice->getValue().getKnownState();
    }

    // If all operands are constant foldable, results are constant foldable.
    for (auto *resultLattice : results)
      propagateIfChanged(resultLattice,
                         resultLattice->join(ConstantFoldabilityState(
                             areAllOperandsConstantFoldable)));

    return success();
  }

  // Set up entry state for lattices to be uninitialized.
  void setToEntryState(ConstantFoldabilityLattice *lattice) override {
    propagateIfChanged(
        lattice, lattice->join(ConstantFoldabilityState::getUninitialized()));
  }
};
} // namespace

/// Given `cluster`, this function creates a new private `FuncOp` containing all
/// ops from `cluster`and returns it after adding to the `moduleSymbolTable`.
/// Function op returned has a single block with no arguments and return types
/// same as types of values in `clusterValuesUsedOutsideCluster`. During
/// outlining, first, constant ops consumed by ops inside the cluster
/// (represented by `constantsUsedInsideCluster`) are cloned into the newly
/// created function body. Later, cluster ops are moved inside the function
/// body. Finally, uses of original constants (from outside) by operations
/// inside the cluster are replaced with newly cloned constants.
static func::FuncOp
outlineClusterToFunction(IRRewriter &rewriter, Location loc,
                         SymbolTable &moduleSymbolTable, const Cluster &cluster,
                         ArrayRef<Value> clusterValuesUsedOutsideCluster,
                         ArrayRef<Operation *> constantsUsedInsideCluster) {

  // Create `func::FuncOp` op.
  FunctionType funcType = FunctionType::get(
      rewriter.getContext(), {},
      llvm::to_vector(llvm::map_range(clusterValuesUsedOutsideCluster,
                                      [](Value v) { return v.getType(); })));
  func::FuncOp funcOp =
      rewriter.create<func::FuncOp>(loc, "constant_subgraph", funcType);
  funcOp->setAttr("plan.constant_foldable", rewriter.getUnitAttr());
  funcOp.setPrivate();

  // Create function body.
  Block *entryBlock = funcOp.addEntryBlock();

  {
    // Create `func::ReturnOp`.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<func::ReturnOp>(loc, clusterValuesUsedOutsideCluster);
  }

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    Operation *term = funcOp.getBody().getBlocks().front().getTerminator();

    // First clone the constant ops used by cluster.
    IRMapping constantMapping;
    for (Operation *op : constantsUsedInsideCluster)
      rewriter.clone(*op, constantMapping);

    // Move non-constant ops.
    for (Operation *op : cluster) {
      if (!op->hasTrait<OpTrait::ConstantLike>())
        rewriter.moveOpBefore(op, term);
    }

    // Update use of outside constants to cloned constants.
    for (Operation *op : constantsUsedInsideCluster)
      rewriter.replaceUsesWithIf(
          op->getResults().front(),
          constantMapping.lookup(op->getResults().front()),
          [&](OpOperand &user) {
            return funcOp->isProperAncestor(user.getOwner());
          });
  }

  if (funcOp->getParentOp())
    funcOp->remove();
  moduleSymbolTable.insert(funcOp);
  return funcOp;
}

/// Given `op`, for each of its operand, if producer has `ConstantLike` trait,
/// push producer to `constantsUsedByCluster`. Region/s of `op` are traveled
/// recursively, doing the same.
static void collectConstantParentsOfOperands(
    Operation *op, SmallVectorImpl<Operation *> &constantsUsedByCluster) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    if (defOp->hasTrait<OpTrait::ConstantLike>())
      constantsUsedByCluster.push_back(defOp);
    continue;
  }

  // Visit regions of op, if applicable.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block)
        collectConstantParentsOfOperands(&nestedOp, constantsUsedByCluster);
    }
  }
}

/// Visits every op in the cluster and if parent of its operand has
/// `ConstantLike` trait, adds it to `constantsUsedByCluster`. Cluster outlining
/// function uses `constantsUsedByCluster` to first copy constant ops into the
/// cluster before moving other ops.
static void collectConstantOpsUsedInsideCluster(
    const Cluster &cluster,
    SmallVectorImpl<Operation *> &constantsUsedByCluster) {
  for (Operation *op : cluster)
    collectConstantParentsOfOperands(op, constantsUsedByCluster);
}

/// Returns true if `op` is pure region op and doesn't have
/// `FunctionOpInterface`.
static bool isPureAndNonFuncRegionOp(Operation *op) {
  return (op->getNumRegions() > 0 &&
          !isa<RegionBranchOpInterface, FunctionOpInterface>(op));
}

/// Traverse `op` recursively and return `true` if op is standalone. Op is
/// standalone if every operand is either result of a constant op OR result
/// of another op which is inside the cluster.
/// There is a special case when op is inside the body of single region
/// carrying ops (for example, `stablehlo.reduce` and `linalg.generic`). In
/// this case, if parent region op is standalone, ops using entry block
/// arguments of this region are also standalone. However, one exception to
/// this is `func::Func` op.
static bool isOpStandalone(Operation *op, DenseSet<Operation *> &clusterOps) {

  auto isBlockArgOfPureAndNonFuncRegionOp = [&](Value v) {
    // First check if it's a block argument
    if (!isa<BlockArgument>(v))
      return false;
    // Get the parent operation of the block
    Operation *parentOp = cast<BlockArgument>(v).getOwner()->getParentOp();
    if (!parentOp)
      return false;
    return isPureAndNonFuncRegionOp(parentOp);
  };

  auto checkIfOpInCluster = [&](Operation *op) {
    if (isPureAndNonFuncRegionOp(op->getParentOp()))
      return clusterOps.contains(op->getParentOp());
    return clusterOps.contains(op);
  };

  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    // `operand` is block argument.
    if (!defOp) {
      if (isBlockArgOfPureAndNonFuncRegionOp(operand))
        continue;
      return false;
    }
    // `defOp` should be either constant OR member of this cluster.
    if (defOp->hasTrait<OpTrait::ConstantLike>() || checkIfOpInCluster(defOp))
      continue;
    return false;
  }

  // Visit regions of op, if applicable.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (!isOpStandalone(&nestedOp, clusterOps))
          return false;
      }
    }
  }
  return true;
}

/// Returns true if cluster is standalone, given ops in the cluster. Cluster
/// is standalone if, for each op, every operand is either result of
/// constant op OR result of another op which is inside cluster. To keep
/// things simple, we only outline standalone clusters. We ignore clusters
/// where one of the ops need result of other cluster as operand.
/// TODO: @Sagar Remove this limitation.
static bool isClusterStandalone(DenseSet<Operation *> &clusterOps) {
  for (Operation *op : clusterOps) {
    if (!isOpStandalone(op, clusterOps))
      return false;
  }
  return true;
}

/// Find constant foldable clusters by running clustering on `func` with
/// given clustering `opts` and outline each cluster to a function inside
/// module with `symbolTable`.
static LogicalResult findClustersAndOutlineToFuncs(func::FuncOp func,
                                                   ModuleOp moduleOp,
                                                   IRRewriter &rewriter,
                                                   SymbolTable &symbolTable,
                                                   const ClusteringOpts &opts,
                                                   DataFlowSolver &solver) {
  FailureOr<SmallVector<Cluster>> clusters =
      analyzeAndClusterOperations(func, opts);
  if (failed(clusters))
    return failure();

  for (const Cluster &cluster : *clusters) {
    DenseSet<Operation *> clusterOpSet;
    for (Operation *op : cluster)
      clusterOpSet.insert(op);

    // Check if cluster is standalone.
    if (!isClusterStandalone(clusterOpSet))
      continue;

    // It is still possible for clusters to have only non-compute ops.
    // For example, `tensor.empty()` followed by `linalg.generic` where
    // later one is not constant foldable. Outlining of such clusters is
    // skipped.
    if (llvm::all_of(clusterOpSet,
                     [](Operation *op) { return op->getNumOperands() == 0; }))
      continue;

    Block *clusterRootBlock = cluster.getRoot()->getBlock();

    // Find cluster values used outside the cluster. These values
    // should be returned from the outlined function.
    SetVector<Value> valuesUsedOutsideCluster;
    for (Operation *op : cluster) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (clusterOpSet.contains(user))
            continue;
          if (valuesUsedOutsideCluster.contains(result))
            continue;
          valuesUsedOutsideCluster.insert(result);
        }
      }
    }

    // Find constant ops used inside the cluster. Remember, constant ops
    // are not part of any cluster.
    SmallVector<Operation *> constantOpsUsedInsideCluster;
    collectConstantOpsUsedInsideCluster(cluster, constantOpsUsedInsideCluster);

    func::FuncOp outlinedFunc = outlineClusterToFunction(
        rewriter, moduleOp->getLoc(), symbolTable, cluster,
        valuesUsedOutsideCluster.getArrayRef(), constantOpsUsedInsideCluster);

    // Insert call to the outline function.
    rewriter.setInsertionPointToStart(clusterRootBlock);
    auto callOp =
        rewriter.create<func::CallOp>(moduleOp->getLoc(), outlinedFunc);
    // Set the call result values to 'uninitialized'.
    for (Value v : callOp->getResults())
      solver.getOrCreateState<ConstantFoldabilityLattice>(v);

    // Replace uses of cluster values used outside with the result of call op.
    for (auto [originalValue, callResult] :
         llvm::zip(valuesUsedOutsideCluster, callOp.getResults()))
      rewriter.replaceUsesWithIf(
          originalValue, callResult, [&](OpOperand &operand) {
            return !outlinedFunc->isProperAncestor(operand.getOwner());
          });
  }
  return success();
}

/// Returns true if `op` is should be clustered.
static bool shouldClusterOp(Operation *op, const DataFlowSolver &solver) {

  // Don't cluster terminator otherwise constant foldable terminator will be
  // outlined.
  // Don't cluster constants since they might be shared across clusters and
  // will be cloned later.
  // Don't cluster control-flow op itself. If control-flow op is clusterable
  // (i.e. added to `ClusteringState`), clustering algorithm doesn't visit ops
  // in its region/s. This causes issue when not all regions of control-flow
  // op are standalone to outline. Consider the example,
  //
  // %0 = scf.if %true->(tensor<4xf32>) {
  //   %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  //   %1 = stablehlo.add %cst, %cst_0 : tensor<4xf32>
  //   %2 = stablehlo.subtract %cst_1, %1 : tensor<4xf32>
  //   scf.yield %2 : tensor<4xf32>
  // } else {
  //   %1 = stablehlo.add %arg0, %cst : tensor<4xf32>
  //   %2 = stablehlo.subtract %1, %cst_0 : tensor<4xf32>
  //   scf.yield %2 : tensor<4xf32>
  // }
  // Here, DFA decides correctly that `%0` is constant foldable. However,
  // our logic to check whether an op is standalone (within cluster) doesn't
  // check which region might be taken for control flow ops. It says op is
  // standalone only if all regions (and thus ops inside those regions) have
  // dependencies inside the cluster.
  // In above example, this causes an issue because even though only `then`
  // region is going to be executed, since `else` region has dependency on
  // %arg0 (which is argument of top level `func.func`), we say this is `scf.if`
  // is not standalone.
  // Skipping control-flow ops all together solves this issue as follows,
  // 1. Clustering analyzes ops within all regions and outline constant foldable
  // clusters.
  // 2. If control-flow op is constant foldable, like above, its own
  // canonicalizer kicks in to keep clusters only in executable regions, as
  // shown below.
  // %0 = call @constant_subgraph() : () -> tensor<4xf32>
  // func.func private @constant_subgraph() -> tensor<4xf32> attributes
  // {plan.constant_foldable} {
  //   %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  //   %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  //   %0 = stablehlo.add %cst, %cst: tensor<4xf32>
  //   %1 = stablehlo.subtract %cst_0, %0 : tensor<4xf32>
  //   return %1 : tensor<4xf32>
  // }
  if (op->hasTrait<OpTrait::IsTerminator>() ||
      op->hasTrait<OpTrait::ConstantLike>() || isa<RegionBranchOpInterface>(op))
    return false;

  // Don't cluster ops inside pure region ops.
  if (isPureAndNonFuncRegionOp(op->getParentOp()))
    return false;

  bool areAllResultsConstantFoldable = true;
  for (Value result : op->getResults()) {
    const ConstantFoldabilityLattice *lattice =
        solver.lookupState<ConstantFoldabilityLattice>(result);
    if (!lattice || lattice->getValue().isUninitialized())
      return false;
    areAllResultsConstantFoldable &= lattice->getValue().getKnownState();
  }
  return areAllResultsConstantFoldable;
}

/// Returns clustering options for constant foldable clusters generation.
static ClusteringOpts
getClusteringOpts(const DataFlowSolver &solver,
                  const std::function<bool(Operation *)> &skipClustering) {
  ClusteringOpts opts;
  opts.clusterTarget = Attribute{};
  opts.isClusterableOp = [&solver, &skipClustering](Operation *op) {
    if (skipClustering && skipClustering(op))
      return false;
    return shouldClusterOp(op, solver);
  };
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  return opts;
}

namespace {
class PlanOutlineConstantFoldableSubgraphsPass
    : public mlir::plan::impl::PlanOutlineConstantFoldableSubgraphsPassBase<
          PlanOutlineConstantFoldableSubgraphsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    IRRewriter rewriter(&getContext());
    SymbolTable symbolTable(moduleOp);

    // Initialize and run data flow analysis to determine
    // constant foldable ops.
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SparseConstantFoldabilityAnalysis>();
    if (failed(solver.initializeAndRun(moduleOp)))
      return signalPassFailure();

    ClusteringOpts opts = getClusteringOpts(solver, skipClustering);

    SmallVector<func::FuncOp> originalFuncs;
    for (func::FuncOp func : moduleOp.getOps<func::FuncOp>())
      originalFuncs.push_back(func);

    for (func::FuncOp func : originalFuncs) {
      if (failed(findClustersAndOutlineToFuncs(func, moduleOp, rewriter,
                                               symbolTable, opts, solver))) {
        emitError(moduleOp->getLoc()) << " failed to process clusters\n";
        return signalPassFailure();
      }
    }
  }
};
} // namespace