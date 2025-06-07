//===- AssignMemorySpaces.cpp ---------------------------------------------===//
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
///  Implementation of the `plan-assign-memory-spaces` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "plan-assign-memory-spaces"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANASSIGNMEMORYSPACESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {

// Generic pattern that rewrites any op by rewriting its operands and result
// types. Regions are also rewritten.
class GenericConvertSpace : public ConversionPattern {
public:
  GenericConvertSpace(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region &before = std::get<0>(regions);
      Region &parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// A pattern that converts the type of the attribute used as an operand for
// arith.constant
class ConvertConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  ConvertConstantPattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = dyn_cast_if_present<ShapedType>(
        typeConverter->convertType(op.getType()));
    if (!newType)
      return failure();

    ElementsAttr newAttr{};
    if (auto elementsAttr = dyn_cast<DenseElementsAttr>(op.getValue()))
      newAttr = elementsAttr.reshape(newType);
    if (auto resourceAttr =
            dyn_cast<DenseResourceElementsAttr>(op.getValue())) {
      DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
      newAttr = DenseResourceElementsAttr::get(newType, handle);
    }
    if (!newAttr)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
    return success();
  }
};
} // namespace

/// Return true if the op is likely in a compute region, like the region of
/// `stablehlo.reduce` or `linalg.generic`.
static bool inComputeRegion(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<func::FuncOp>(parent))
      return false;
    if (!isa<RegionBranchOpInterface>(parent))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

namespace {
/// Use an explicit 'host_pinned' staging tensor to materialie the
/// 'from_elements' before creating explicitly moving it to the 'device' space.
/// Other optimization patterns below help avoid the host-device transfer when
/// possible.
struct FixUpFromElements : public OpRewritePattern<tensor::FromElementsOp> {
  FixUpFromElements(MLIRContext *ctx, const DataFlowSolver &solver,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), solver(solver) {}

  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    auto space = dyn_cast_or_null<MemorySpaceAttr>(op.getType().getEncoding());
    if (!space)
      return failure();
    if (space.getValue() != plan::MemorySpace::device)
      return failure();

    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op.getResult());
    if (!lattice || lattice->getValue().isUninitialized() ||
        !lattice->getValue().isHostVisible())
      return failure();

    RankedTensorType originalType = op.getType();
    RankedTensorType newType = RankedTensorType::get(
        originalType.getShape(), originalType.getElementType(),
        MemorySpaceAttr::get(originalType.getContext(),
                             plan::MemorySpace::host_pinned));
    auto newOp = rewriter.create<tensor::FromElementsOp>(op.getLoc(), newType,
                                                         op.getElements());
    Value deviceTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), originalType.getShape(), originalType.getElementType(),
        originalType.getEncoding());
    Value rematDevReplacement =
        rewriter
            .create<bufferization::MaterializeInDestinationOp>(
                op.getLoc(), originalType, newOp.getResult(), deviceTensor)
            .getResult();
    rewriter.replaceOp(op, rematDevReplacement);
    return success();
  }

  const DataFlowSolver &solver;
};

static bool isHostVisible(TypedValue<RankedTensorType> v) {
  auto space = dyn_cast_or_null<MemorySpaceAttr>(v.getType().getEncoding());
  if (!space)
    return false;
  switch (space.getValue()) {
  case plan::MemorySpace::host:
  case plan::MemorySpace::host_pinned:
  case plan::MemorySpace::unified:
    return true;
  default:
    return false;
  }
}

/// For any 'shape' parameter of a 'tensor.reshape', get the shape by skipping
/// past any unnecessary explicit host-device transfers.
struct ReshapeAbsorbDeviceCast : public OpRewritePattern<tensor::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (isHostVisible(op.getShape()))
      return failure();
    auto matOp =
        op.getShape()
            .getDefiningOp<bufferization::MaterializeInDestinationOp>();
    if (!matOp)
      return failure();
    auto source = dyn_cast<TypedValue<RankedTensorType>>(matOp.getSource());
    if (!source || !isHostVisible(source))
      return failure();
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getShapeMutable().assign(source); });
    return success();
  }
};

/// Rewrite `memref.load` that acts on device memory to first copy the buffer to
/// the host and load from the host buffer.
struct TensorDeviceExtractRewriter
    : public OpRewritePattern<tensor::ExtractOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getTensor();
    if (isHostVisible(source))
      return failure();

    if (inComputeRegion(op))
      return failure();

    rewriter.setInsertionPointAfterValue(source);
    Value hostTensor = rewriter.create<tensor::CastOp>(
        op.getLoc(),
        RankedTensorType::get(
            source.getType().getShape(), source.getType().getElementType(),
            plan::MemorySpaceAttr::get(op->getContext(),
                                       plan::MemorySpace::host_pinned)),
        source);

    rewriter.replaceUsesWithIf(op.getTensor(), hostTensor, [&](OpOperand &use) {
      return isa<tensor::ExtractOp>(use.getOwner());
    });

    return success();
  }
};

/// Remap relevant analysis state of type T from `original` to `replacement`.
template <typename T>
static void remapLatticeState(DataFlowSolver &solver, Value original,
                              Value replacement) {
  if constexpr (!std::is_same_v<T, dataflow::Executable>) {
    if (const T *lattice = solver.lookupState<T>(original)) {
      T *latticeReplacement = solver.getOrCreateState<T>(replacement);
      latticeReplacement->getValue() = lattice->getValue();
    }
  } else {
    // do nothing for liveness analysis for the moment except create the state
    if (const auto *oldState =
            solver.lookupState<dataflow::Executable>(original)) {
      dataflow::Executable *newState = solver.getOrCreateState<T>(replacement);
      // Set to live if old state is live. We ignore change status.
      if (oldState->isLive())
        (void)newState->setToLive();
    }
  }
}

/// A rewrite listener that transfers replacements to updates to the solver
/// state.
class SolverStateListener : public RewriterBase::Listener {
public:
  SolverStateListener(DataFlowSolver &solver)
      : RewriterBase::Listener(), solver(solver) {}

private:
  void notifyOperationReplaced(Operation *op,
                               ValueRange replacements) override {
    for (auto [original, replacement] :
         llvm::zip_equal(op->getResults(), replacements)) {
      remapLatticeState<TensorKindLattice>(solver, original, replacement);
      remapLatticeState<dataflow::Lattice<dataflow::ConstantValue>>(
          solver, original, replacement);
      remapLatticeState<dataflow::Executable>(solver, original, replacement);
    }
    solver.eraseState(solver.getProgramPointAfter(op));
  }
  void notifyOperationReplaced(Operation *op, Operation *replacement) override {
    notifyOperationReplaced(op, replacement->getResults());
  }

  void notifyOperationErased(Operation *op) override {
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value res : op->getResults())
      solver.eraseState(res);
  }

  DataFlowSolver &solver;
};

} // namespace

namespace {
struct AssignMemorySpacesPass
    : public plan::impl::PlanAssignMemorySpacesPassBase<
          AssignMemorySpacesPass> {
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    TypeConverter converter;
    converter.addConversion(
        [&](Type type) -> std::optional<Type> { return type; });

    // The default tensor type converter just adds the 'device' memory type
    // info.
    auto deviceEncoding =
        plan::MemorySpaceAttr::get(context, plan::MemorySpace::device);
    converter.addConversion([&](RankedTensorType type) -> std::optional<Type> {
      if (type.getEncoding())
        return type;
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   deviceEncoding);
    });

    // Ops are legal if they are in a nested module or if their operand and
    // result types are legal.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (op->getParentWithTrait<OpTrait::SymbolTable>() != getOperation())
        return true;
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      if (op->getParentWithTrait<OpTrait::SymbolTable>() != getOperation())
        return true;
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.markOpRecursivelyLegal<func::FuncOp>(
        [&](func::FuncOp op) -> std::optional<bool> {
          if (op->getParentWithTrait<OpTrait::SymbolTable>() != getOperation())
            return true;
          return false;
        });
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      if (op->getParentWithTrait<OpTrait::SymbolTable>() != getOperation())
        return true;
      return converter.isLegal(op.getType()) &&
             converter.isLegal(op.getValue().getType());
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<GenericConvertSpace, ConvertConstantPattern>(converter,
                                                              context);

    // FuncOp is special as it has type encoding via attributes.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      emitError(module.getLoc(), "failed to assign memory spaces");
      return signalPassFailure();
    }

    // Perform some minor optimizations involving tensor.from_elements.
    {
      SymbolTableCollection symbolTables;
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      solver.load<TensorKindAnalysis>(symbolTables);

      if (failed(solver.initializeAndRun(getOperation()))) {
        emitError(getOperation().getLoc())
            << "failed to run TensorKindAnalysis";
        return signalPassFailure();
      }

      SolverStateListener solverAwareListener(solver);
      GreedyRewriteConfig config;
      config.listener = &solverAwareListener;
      FrozenRewritePatternSet patterns = [&]() {
        RewritePatternSet patterns_(&getContext());
        patterns_.insert<FixUpFromElements>(&getContext(), solver);
        patterns_.insert<ReshapeAbsorbDeviceCast>(&getContext());
        patterns_.insert<TensorDeviceExtractRewriter>(&getContext());
        return patterns_;
      }();
      for (FunctionOpInterface func :
           getOperation().getOps<FunctionOpInterface>()) {
        if (failed(applyPatternsGreedily(func, patterns))) {
          emitError(func.getLoc()) << "failed to run " << getArgument();
          return signalPassFailure();
        }
      }
    }
  }
};
} // namespace
