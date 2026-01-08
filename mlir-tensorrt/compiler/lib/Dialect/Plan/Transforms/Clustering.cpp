//===- Clustering.cpp -----------------------------------------------------===//
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
/// Implementation of the `plan-clustering` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "mlir-tensorrt-common/Utils/DataFlowUtils.h"
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "plan-clustering"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir::plan {
#define GEN_PASS_DEF_CLUSTERINGPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

using SignatureConversion = TypeConverter::SignatureConversion;

using SolverAwareListener =
    SolverStateListener<TensorKindLattice,
                        dataflow::Lattice<dataflow::ConstantValue>,
                        dataflow::Executable>;

/// A listener that is DataFlowSolver-aware, but it also is aware of inserting
/// `plan::ClusterOp` operations. It sets the `plan::ClusterOp` lattice
/// values to be equivalent to the values yielded by the `plan::YieldOp`
/// operations in the body.
class ClusteringListener
    : public SolverStateListener<TensorKindLattice,
                                 dataflow::Lattice<dataflow::ConstantValue>,
                                 dataflow::Executable> {
public:
  using SolverStateListener::SolverStateListener;

  void updateClusterResultStates(plan::YieldOp yieldOp) {
    auto clusterOp = dyn_cast<plan::ClusterOp>(yieldOp->getParentOp());
    if (!clusterOp || clusterOp->getNumResults() != yieldOp->getNumOperands())
      return;
    ValueRange yieldedValues = yieldOp->getOperands();
    ValueRange clusterResults = clusterOp.getResults();
    for (auto [yielded, result] :
         llvm::zip_equal(yieldedValues, clusterResults))
      this->copyLatticeStates(yielded, result);
  }

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    if (previous.isSet())
      return;

    // Ensure that the `plan::ClusterOp` lattice values are updated to
    // match the values yielded by the `plan::YieldOp` operations in the body.
    if (auto clusterOp = dyn_cast<plan::ClusterOp>(op)) {
      if (auto yieldOp = clusterOp.getYield())
        updateClusterResultStates(yieldOp);
      return;
    }
    if (auto yieldOp = dyn_cast<plan::YieldOp>(op)) {
      updateClusterResultStates(yieldOp);
      return;
    }
  }

  void notifyOperationModified(Operation *op) override {
    if (auto yieldOp = dyn_cast<plan::YieldOp>(op))
      updateClusterResultStates(yieldOp);
  }
};

/// Creates an empty `plan.cluster` operation for a given type of cluster
/// target.
static Operation *createClusterOp(OpBuilder &b, Location loc, TypeRange types,
                                  Attribute target) {
  auto regionOp = b.create<plan::ClusterOp>(
      loc, types, cast<CompilerBackendAttrInterface>(target));
  b.setInsertionPointToStart(&regionOp.getRegion().emplaceBlock());
  b.create<plan::YieldOp>(loc);
  return regionOp;
}

/// Returns true if the op is already contained in a region op that is used to
/// encapsulate clusters.
static bool isOpInClusterRegion(Operation *op) {
  return op->getParentOfType<plan::ClusterOp>();
}

/// Apply cluster-and-outline using the given options to the `func`.
static LogicalResult
applyClusteringToFunc(RewriterBase &rewriter, FunctionOpInterface func,
                      DataFlowSolver &solver,
                      ArrayRef<CompilerBackendAttrInterface> clusters,
                      plan::InputKind inputKind) {
  ClusteringPatternSet<ClusteringRewriter> patterns;
  for (const auto &[idx, target] : llvm::enumerate(clusters)) {
    FailureOr<ClusteringOpts> clusteringOpts =
        target.getClusterKindOptions(inputKind, func, solver);
    if (failed(clusteringOpts))
      return failure();
    patterns.add(*clusteringOpts, createClusterOp, isOpInClusterRegion,
                 target.getClusterFilter(inputKind),
                 PatternBenefit(target.getClusterBenefit(inputKind)));
  }

  for (const std::unique_ptr<ClusteringRewriter> &rewrite : patterns) {
    FailureOr<SmallVector<Operation *>> regionOps =
        rewrite->findClusterAndCreateRegionOp(func, rewriter);
    if (failed(regionOps))
      return emitError(func.getLoc())
             << "clustering rewrite " << rewrite->getTarget() << " failed ";
  }

  return success();
}

namespace {
class ClusteringPass : public plan::impl::ClusteringPassBase<ClusteringPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Cluster all functions that:
    // - Are not declarations/external
    // - Don't already have a cluster_kind attribute (already processed)
    // - Are not private decomposition functions
    SmallVector<FunctionOpInterface> funcs;
    llvm::append_range(
        funcs,
        llvm::make_filter_range(
            module.getOps<FunctionOpInterface>(), [](FunctionOpInterface func) {
              return !func.isDeclaration() && !func.isExternal() &&
                     !func->hasAttr(PlanDialect::kFuncTargetKind) &&
                     !(func.isPrivate() && func->hasAttr("plan.decomposition"));
            }));

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    auto listener = std::make_unique<ClusteringListener>(solver);
    IRRewriter rewriter(module->getContext());
    rewriter.setListener(listener.get());

    // If the `plan.backends` already exists on the module, use that,
    // otherwise, populate defaults.
    SmallVector<CompilerBackendAttrInterface> schedule;
    if (ArrayAttr clusterKind = module->getAttrOfType<ArrayAttr>(
            plan::PlanDialect::kBackendsAttrName)) {
      for (Attribute kind : clusterKind) {
        auto kindAttr = llvm::dyn_cast<CompilerBackendAttrInterface>(kind);
        if (!kindAttr) {
          emitError(module.getLoc())
              << "in '" << plan::PlanDialect::kBackendsAttrName
              << "' found attribute " << kind
              << ", but it is not a CompilerBackendAttrInterface";
          return signalPassFailure();
        }
        schedule.push_back(kindAttr);
      }
    }
    llvm::sort(schedule, [&](CompilerBackendAttrInterface lhs,
                             CompilerBackendAttrInterface rhs) {
      return lhs.getClusterBenefit(inputKind) >
             rhs.getClusterBenefit(inputKind);
    });

    for (FunctionOpInterface func : funcs) {
      if (failed(applyClusteringToFunc(rewriter, func, solver, schedule,
                                       inputKind)))
        return signalPassFailure();
    }

    // Drop clustering attributes since they are no longer needed.
    module->removeAttr(plan::PlanDialect::kBackendsAttrName);
  }
};
} // namespace
