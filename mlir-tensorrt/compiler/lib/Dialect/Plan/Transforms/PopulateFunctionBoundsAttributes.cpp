//===- RegionArgBounds.cpp  -----------------------------------------------===//
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
/// Implementation of the `plan-create-result-arg-bounds` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANPOPULATEFUNCTIONBOUNDSATTRIBUTESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Return the upper and lower bounds
template <typename LatticeType>
static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getBounds(Value v, DataFlowSolver &solver) {
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  if constexpr (std::is_same_v<ShapeBoundsLattice, LatticeType>) {
    if (rtt.hasStaticShape())
      return std::make_pair(llvm::to_vector(rtt.getShape()),
                            llvm::to_vector(rtt.getShape()));
  }

  const LatticeType *lattice = solver.lookupState<LatticeType>(v);
  if (!lattice || lattice->getValue().isUninitialized())
    return failure();

  auto bound = [&](bool isUB) -> SmallVector<int64_t> {
    return llvm::map_to_vector(lattice->getValue().getValue(),
                               [&](const ConstantIntRanges &r) {
                                 if (isUB)
                                   return r.smax().getSExtValue();
                                 return r.smin().getSExtValue();
                               });
  };

  return std::make_pair(bound(/*isUB=*/false), bound(/*isUB=*/true));
}

namespace {
class PlanPopulateFunctionBoundsAttributesPass
    : public plan::impl::PlanPopulateFunctionBoundsAttributesPassBase<
          PlanPopulateFunctionBoundsAttributesPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || func.isDeclaration())
      return;

    if (func.getBlocks().size() != 1) {
      emitError(func.getLoc())
          << "the " << getArgument()
          << " pass requires functions to have a single-block region";
      return signalPassFailure();
    }

    // Skip all functions without shape profile information.
    if (!func.getArgAttrs() ||
        llvm::none_of(
            func.getArgAttrs()->getAsRange<DictionaryAttr>(),
            [&](DictionaryAttr dict) {
              return dict.getNamed(PlanDialect::getShapeBoundsAttrName()) ||
                     dict.getNamed(PlanDialect::getValueBoundsAttrName());
            }))
      return;

    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<ShapeIntegerRangeAnalysis>();
    solver.load<ShapeBoundsForwardAnalysis>();
    solver.load<ShapeBoundsBackwardsAnalysis>(symbolTable);
    solver.load<TensorValueBoundsAnalysis>();
    if (failed(solver.initializeAndRun(func))) {
      func.emitError() << "failed to run result arg bounds analyses.";
      return signalPassFailure();
    }

    func::ReturnOp returnOp =
        cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
    for (const auto [idx, result] :
         llvm::enumerate(returnOp->getOpOperands())) {
      auto rtt = dyn_cast<RankedTensorType>(result.get().getType());
      if (!rtt)
        continue; // No bound information for a scalar result.

      if (!rtt.hasStaticShape()) {
        FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
            bounds = getBounds<ShapeBoundsLattice>(result.get(), solver);
        if (failed(bounds)) {
          emitError(func.getLoc())
              << "failed to calculate shape bounds for return operand #" << idx;
          return signalPassFailure();
        }
        auto boundsAttr = BoundsAttr::getChecked(
            mlir::detail::getDefaultDiagnosticEmitFn(func->getLoc()),
            func.getContext(), BoundsKind::Shape, ArrayRef(bounds->first),
            ArrayRef(bounds->second));
        if (!boundsAttr || boundsAttr.isNone()) {
          emitError(func.getLoc())
              << "failed to compute lower/upper shape bounds attribute";
          return signalPassFailure();
        }
        func.setResultAttr(idx, plan::PlanDialect::getShapeBoundsAttrName(),
                           boundsAttr);
        continue;
      }

      // At this point, we have a statically shaped tensor. Check to see if we
      // should have value information attached.
      auto withValuesOp = result.get().getDefiningOp<plan::WithValuesOp>();
      if (!withValuesOp)
        continue;

      FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> bounds =
          getBounds<TensorValueBoundsLattice>(result.get(), solver);
      if (failed(bounds)) {
        emitError(func.getLoc())
            << "failed to calculate shape bounds for return operand #" << idx;
        return signalPassFailure();
      }

      auto boundsAttr = BoundsAttr::getChecked(
          mlir::detail::getDefaultDiagnosticEmitFn(func->getLoc()),
          func.getContext(), BoundsKind::Value, ArrayRef(bounds->first),
          ArrayRef(bounds->second));
      if (!boundsAttr || boundsAttr.isNone()) {
        emitError(func.getLoc())
            << "failed to compute lower/upper shape bounds attribute";
        return signalPassFailure();
      }

      func.setResultAttr(idx, plan::PlanDialect::getValueBoundsAttrName(),
                         boundsAttr);
    }
  }
};

} // namespace
