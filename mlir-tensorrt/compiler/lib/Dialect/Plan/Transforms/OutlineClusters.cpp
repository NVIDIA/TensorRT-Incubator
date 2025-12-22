//===- OutlineClusters.cpp  -----------------------------------------------===//
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
/// Implementation of the `plan-outline-clusters` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "plan-outline-clusters"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

namespace mlir::plan {
#define GEN_PASS_DEF_OUTLINECLUSTERSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

static CompilerBackendAttrInterface getClusterTargetForRegionOp(Operation *op) {
  if (auto regionOp = dyn_cast<plan::ClusterOp>(op))
    return regionOp.getTargetAttr();
  if (auto regionOp = dyn_cast<plan::DpsClusterOp>(op))
    return regionOp.getTargetAttr();
  if (auto regionOp = dyn_cast<plan::AllocClusterOp>(op))
    return regionOp.getTargetAttr();
  llvm_unreachable("unknown cluster region op kind");
}

/// Returns the parameters that should be used for region outlining.
static FailureOr<OutlineRegionOptions>
getOutliningParam(InputKind inputKind, Operation *op,
                  SymbolTable &moduleSymbolTable) {
  CompilerBackendAttrInterface target = getClusterTargetForRegionOp(op);
  if (!target)
    return op->emitError("missing target attribute");
  std::optional<OutlineRegionOptions> opts = target.getClusterOutliningOptions(
      inputKind, op->getContext(), moduleSymbolTable);
  if (!opts)
    return failure();
  return *opts;
}

/// Create outlined functions for each `scf.execute_region` operation within
/// `region`.
static FailureOr<SmallVector<FunctionOpInterface>>
createFunctionsFromRegions(InputKind inputKind, RewriterBase &rewriter,
                           Region &region, SymbolTable &moduleSymbolTable) {
  SmallVector<FunctionOpInterface> outlinedFuncs;

  WalkResult result = region.walk([&](Operation *op) {
    if (!isa<plan::ClusterOp, plan::DpsClusterOp, plan::AllocClusterOp>(op))
      return WalkResult::advance();

    CompilerBackendAttrInterface backend = getClusterTargetForRegionOp(op);
    rewriter.setInsertionPoint(op);

    /// TODO: currently the interface has two different ways to specify
    /// outlining. We should reduce this to a single interface method.
    FailureOr<OutlineRegionOptions> opts =
        getOutliningParam(inputKind, op, moduleSymbolTable);
    if (llvm::succeeded(opts)) {
      FailureOr<std::pair<FunctionOpInterface, SetVector<Value>>>
          outlineResult = outlineRegionOp(rewriter, op, *opts);
      if (failed(outlineResult)) {
        emitError(op->getLoc())
            << "failed to outline cluster region op to function";
        return WalkResult::interrupt();
      }
      auto [outlinedFunc, callOperands] = *outlineResult;
      outlinedFunc->setAttr(plan::PlanDialect::kFuncTargetKind, backend);
      outlinedFuncs.push_back(outlinedFunc);
      return WalkResult::advance();
    }

    if (failed(backend.outlineClosedCluster(inputKind, rewriter, op,
                                            moduleSymbolTable)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  return outlinedFuncs;
}

namespace {
class OutlineClustersPass
    : public plan::impl::OutlineClustersPassBase<OutlineClustersPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable moduleSymbolTable(module);

    SmallVector<FunctionOpInterface> funcs =
        llvm::to_vector(llvm::make_filter_range(
            module.getOps<FunctionOpInterface>(), [](FunctionOpInterface func) {
              return !func.isDeclaration() && !func.isExternal() &&
                     !(func.isPrivate() && func->hasAttr("plan.decomposition"));
            }));

    IRRewriter rewriter(module->getContext());
    for (FunctionOpInterface func : funcs) {
      SmallVector<plan::ClusterOp> clusters;
      func->walk([&](plan::ClusterOp clusterOp) {
        if (!clusterOp.getTargetAttr())
          return WalkResult::advance();
        clusters.push_back(clusterOp);
        return WalkResult::skip();
      });

      if (failed(createFunctionsFromRegions(
              inputKind, rewriter, func.getFunctionBody(), moduleSymbolTable)))
        return signalPassFailure();
    }
  }
};
} // namespace
