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
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-tensorrt-common/Support/Status.h"
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

static FailureOr<SmallVector<Value>> getReturnedValues(func::FuncOp func) {
  if (executor::abi::isABIWrapperFunction(func)) {
    FailureOr<FunctionType> abiFuncType =
        mlir::executor::abi::getABIFunctionType(func);
    assert(succeeded(abiFuncType) && "expected ABI function type");
    SmallVector<Value> returnedValues(abiFuncType->getNumResults(), nullptr);
    for (BlockArgument arg :
         func.getArguments().drop_front(abiFuncType->getNumInputs())) {
      std::optional<unsigned> outputIdx =
          mlir::executor::abi::isOutputArgument(func, arg);
      assert(outputIdx.has_value() && "expected output index");
      assert(*outputIdx < returnedValues.size() &&
             "expected valid output index");

      executor::ABISendOp sendOp{};
      for (Operation *user : arg.getUsers()) {
        if (auto abiOp = dyn_cast<executor::ABISendOp>(user)) {
          // Multiple send ops are not supported. This can only happen if the
          // entrypoint has multiple returns (e.g. it has unstructured control
          // flow from the start).
          if (sendOp)
            return failure();
          sendOp = abiOp;
        }
      }
      // An argument may not be used. This is not an error.
      if (!sendOp)
        continue;
      returnedValues[*outputIdx] = sendOp.getValue();
    }
    return returnedValues;
  }
  func::ReturnOp returnOp =
      cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
  return llvm::SmallVector<Value>(returnOp->getOperands());
}

static void updateFunctionBoundsAttribute(func::FuncOp func, StringRef attrName,
                                          plan::BoundsAttr boundsAttr,
                                          unsigned resultIndex) {
  if (executor::abi::isABIWrapperFunction(func)) {
    FailureOr<FunctionType> abiFuncType =
        mlir::executor::abi::getABIFunctionType(func);
    assert(succeeded(abiFuncType) && "expected ABI function type");
    func.setArgAttr(abiFuncType->getNumInputs() + resultIndex, attrName,
                    boundsAttr);
    return;
  }

  func.setResultAttr(resultIndex, attrName, boundsAttr);
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
        llvm::none_of(func.getArgAttrs()->getAsRange<DictionaryAttr>(),
                      [&](DictionaryAttr dict) {
                        return dict.getNamed(
                                   PlanDialect::kShapeBoundsAttrName) ||
                               dict.getNamed(PlanDialect::kValueBoundsAttrName);
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

    FailureOr<SmallVector<Value>> returnedValues = getReturnedValues(func);
    if (failed(returnedValues))
      return signalPassFailure();
    for (const auto [idx, result] : llvm::enumerate(*returnedValues)) {
      auto rtt = dyn_cast<RankedTensorType>(result.getType());
      if (!rtt)
        continue; // No bound information for a scalar result.

      if (!rtt.hasStaticShape()) {
        FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
            bounds = getBounds<ShapeBoundsLattice>(result, solver);
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
        updateFunctionBoundsAttribute(
            func, plan::PlanDialect::kShapeBoundsAttrName, boundsAttr, idx);
        continue;
      }

      // At this point, we have a statically shaped tensor. Check to see if we
      // should have value information attached.
      auto withValuesOp = result.getDefiningOp<plan::WithValuesOp>();
      if (!withValuesOp)
        continue;

      FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> bounds =
          getBounds<TensorValueBoundsLattice>(result, solver);
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

      updateFunctionBoundsAttribute(
          func, plan::PlanDialect::kValueBoundsAttrName, boundsAttr, idx);
    }
  }
};

} // namespace
