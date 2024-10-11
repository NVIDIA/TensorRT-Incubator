//===- StablehloClustering.cpp --------------------------------------------===//
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
/// Clustering pipeline that operates on Stable HLO IR. It separates the IR into
/// functions that go down different pipelines: stablehlo-to-arith (scalarizable
/// op clusters), stablehlo-to-tensorrt, and code generation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

#define DEBUG_TYPE "stablehlo-clustering"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir::plan {
#define GEN_PASS_DEF_STABLEHLOCLUSTERINGPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// StablehloClusteringPass
//===----------------------------------------------------------------------===//

using SignatureConversion = TypeConverter::SignatureConversion;

/// Creates an empty `plan.inline_group` operation for a given type of cluster
/// target.
static Operation *createInlineGroupOp(OpBuilder &b, Location loc,
                                      TypeRange types, Attribute target) {
  auto regionOp = b.create<plan::InlineGroupOp>(loc, types, target);
  b.setInsertionPointToStart(&regionOp.getRegion().emplaceBlock());
  b.create<plan::YieldOp>(loc);
  return regionOp;
}

/// Returns true if the op is already constained in a region op that is used to
/// encapsulate clusters.
static bool isOpInClusterRegion(Operation *op) {
  return op->getParentOfType<plan::InlineGroupOp>();
}

/// Apply cluster-and-outline using the given options to the `func`.
static LogicalResult
applyClusteringToFunc(RewriterBase &rewriter, func::FuncOp func,
                      DataFlowSolver &solver,
                      ArrayRef<ClusterKindAttrInterface> clusters,
                      const StablehloClusteringPassOptions &opts) {
  ClusteringPatternSet<ClusteringRewriter> patterns;
  for (const auto &[idx, target] : llvm::enumerate(clusters)) {
    if (target.getClusterKindName() == "tensorrt") {
      patterns.add(target.getClusterKindOptions(solver, opts.trtMajorVersion),
                   createInlineGroupOp, isOpInClusterRegion,
                   target.getClusterFilter(),
                   PatternBenefit(target.getClusterBenefit()));
    } else {
      patterns.add(target.getClusterKindOptions(solver, std::nullopt),
                   createInlineGroupOp, isOpInClusterRegion,
                   target.getClusterFilter(),
                   PatternBenefit(target.getClusterBenefit()));
    }
  }

  for (const std::unique_ptr<ClusteringRewriter> &rewrite : patterns) {
    FailureOr<SmallVector<Operation *>> regionOps =
        rewrite->findClusterAndCreateRegionOp(func, rewriter);
    if (failed(regionOps))
      return emitError(func.getLoc())
             << "clustering rewrite " << rewrite->getTarget() << " failed ";

    // The IR probably changed, so re-run the data flow solver. Any new values
    // yielded by the new region operations won't have lattice values associated
    // with them.
    if (failed(solver.initializeAndRun(func)))
      return func->emitError() << "failed to run dataflow solver";
  }

  return success();
}

/// For all single-result pure producers in `op` that return true from
/// `isProducerToClone`, clone them for each use. The use-cases for this method
/// are for improving fusion or clustering performance. See below uses for
/// examples.
static void cloneProducersWhereProfitable(
    RewriterBase &rewriter, Operation *op,
    llvm::function_ref<bool(Operation *op)> isProducerToClone) {
  SmallVector<Operation *> producersToClone;
  op->walk([&](Operation *op) {
    if (op->getNumResults() != 1 || !isMemoryEffectFree(op))
      return;
    if (isProducerToClone(op))
      producersToClone.push_back(op);
  });

  for (Operation *producerToClone : producersToClone) {
    rewriter.setInsertionPoint(producerToClone);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use :
         llvm::make_early_inc_range(producerToClone->getUses()))
      uses.push_back(&use);

    for (OpOperand *use : ArrayRef(uses).drop_front(1)) {
      rewriter.setInsertionPoint(use->getOwner());
      Operation *clone = rewriter.clone(*producerToClone);
      use->assign(clone->getResult(0));
    }
  }
}

/// Sometimes constant values (especially i32 scalars) have both 'host' and
/// 'device' uses and thus have TensorKind of 'both'. This function finds such
/// constants, duplicates them over their users, and then re-runs the
/// TensorKind analysis. This enables us to more reliably determine which
/// operations have all their operands/results located purely on the host.
static LogicalResult
deconflictConstantsOnHostAndDevice(RewriterBase &rewriter, Operation *op,
                                   DataFlowSolver &solver) {

  auto isProducerToClone = [&](Operation *op) -> bool {
    if (!op->hasTrait<OpTrait::ConstantLike>() || op->getNumResults() != 1)
      return false;
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op->getResult(0));
    if (!lattice || lattice->getValue().isUninitialized() ||
        !lattice->getValue().isHostVisible())
      return false;
    if (RankedTensorType rtt =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      if (rtt.getNumElements() > 16)
        return false;
    }
    return true;
  };

  cloneProducersWhereProfitable(rewriter, op, isProducerToClone);
  if (failed(solver.initializeAndRun(op)))
    return failure();

  return success();
}

/// Find `stablehlo.convert(constant)` ops and clone them for each use. This
/// helps to ensure that each clustered TensorRT engines segment has
/// self-contained weights and doesn't contain live-out conversions of weights,
/// which can cause significant perf issues (e.g. 2x latency on FP16 GPT
/// models).
static void deconflictStablehloConstConvertOps(RewriterBase &rewriter,
                                               Operation *op) {
  auto isProducerToClone = [](Operation *op) -> bool {
    auto convertOp = dyn_cast<stablehlo::ConvertOp>(op);
    return convertOp && !convertOp->hasOneUse() &&
           convertOp.getOperand().getDefiningOp<stablehlo::ConstantOp>();
  };
  cloneProducersWhereProfitable(rewriter, op, isProducerToClone);
}

static auto getIntegerAttrOrDefault(Operation *op, StringRef name,
                                    int64_t defaultValue) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return defaultValue;
}

namespace {
class StablehloClusteringPass
    : public plan::impl::StablehloClusteringPassBase<StablehloClusteringPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    /// If an entrypoint is specified, we only run clustering on the
    /// entrypoint. Otherwise, run on all functions.
    SmallVector<func::FuncOp> funcs;
    if (entrypoint.empty()) {
      llvm::append_range(
          funcs, llvm::make_filter_range(
                     module.getOps<func::FuncOp>(), [](func::FuncOp func) {
                       return !func.isDeclaration() && !func.isExternal() &&
                              !(func.isPrivate() &&
                                func->hasAttr("plan.decomposition"));
                     }));
    } else {
      auto mainFunc = dyn_cast_or_null<func::FuncOp>(
          SymbolTable(module).lookup(entrypoint));
      if (!mainFunc) {
        emitError(module.getLoc())
            << "module does not have a function with symbol name = "
            << entrypoint;
        return signalPassFailure();
      }
      funcs.push_back(mainFunc);
    }

    IRRewriter rewriter(module->getContext());

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    // Duplicate constants that have a placement of 'both' (host and device
    // access).
    if (failed(deconflictConstantsOnHostAndDevice(rewriter, module, solver)))
      return signalPassFailure();

    // Duplicate `stablehlo.convert(stablehlo.const)` chains per user to improve
    // segmentation performance.
    deconflictStablehloConstConvertOps(rewriter, module);

    // If the `plan.cluster_kinds` already exists on the module, use that,
    // otherwise, populate defaults.
    SmallVector<ClusterKindAttrInterface> schedule;
    if (ArrayAttr clusterKind =
            module->getAttrOfType<ArrayAttr>("plan.cluster_kinds")) {
      for (Attribute kind : clusterKind) {
        auto kindAttr = llvm::dyn_cast<ClusterKindAttrInterface>(kind);
        if (!kind) {
          emitError(module.getLoc())
              << "in 'plan.cluster_kinds' found attribute " << kind
              << ", but it is not a ClusterKindAttrInterface";
          return signalPassFailure();
        }
        schedule.push_back(kindAttr);
      }
    } else {
      // Use default cluster kind schedule.
      schedule.push_back(TensorRTClusterKindAttr::get(
          module->getContext(), this->disallowHostTensorsInTensorRTClusters,
          10));
      schedule.push_back(HostClusterKindAttr::get(module->getContext(), 9));
    }
    llvm::sort(schedule,
               [](ClusterKindAttrInterface lhs, ClusterKindAttrInterface rhs) {
                 return lhs.getClusterBenefit() > rhs.getClusterBenefit();
               });

    for (func::FuncOp func : funcs) {
      if (failed(applyClusteringToFunc(
              rewriter, func, solver, schedule,
              StablehloClusteringPassOptions{entrypoint, false, false,
                                             trtMajorVersion})))
        return signalPassFailure();
    }

    // Check for StableHLO partitioning attributes and attach the executor
    // grid shape attribute.
    /// TODO: move this logic into a standalone pass that handles partitioning.
    {
      auto numReplicas =
          getIntegerAttrOrDefault(module, "mhlo.num_replicas", 1);
      auto numPartitions =
          getIntegerAttrOrDefault(module, "mhlo.num_partitions", 1);
      if (failed(executor::setModuleProcessGridShape(
              module, {numReplicas, numPartitions}))) {
        emitError(module->getLoc())
            << "failed to set the Executor process grid shape attribute";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
