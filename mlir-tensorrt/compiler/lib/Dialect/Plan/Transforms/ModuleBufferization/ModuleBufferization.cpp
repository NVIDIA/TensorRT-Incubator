//===- ModuleBufferization.cpp -------------------------------------------===//
//
// Modified from upstream 'OneShotModuleBufferize.cpp', part of the LLVM
// Project, under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Changes: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Implementation of joint bufferization of host and device program.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Interfaces/BufferizationScopeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "plan-module-bufferize"
#define DBGS() (llvm::dbgs() << "[plan-module-bufferize] ")
#define DBGF(fmt, ...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv(                                    \
                 stderr, "{0}:{1}:{2}(): ", "ModuleBufferization.cpp",         \
                 __LINE__, __func__);                                          \
             llvm::dbgs() << llvm::formatv(fmt "\n", __VA_ARGS__));

namespace mlir::plan {
#define GEN_PASS_DEF_PLANMODULEBUFFERIZEPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

using OneShotBufferizationOptions = bufferization::OneShotBufferizationOptions;
using BufferizationStatistics = bufferization::BufferizationStatistics;

/// Return all func.return ops in the given function.
static FailureOr<SmallVector<Operation *>>
getReturnLikeOps(func::FuncOp funcOp) {
  SmallVector<Operation *> result;
  for (Block &b : funcOp.getFunctionBody()) {
    Operation *term = b.getTerminator();
    if (!term->hasTrait<OpTrait::ReturnLike>())
      return failure();
    result.push_back(term);
  }
  return result;
}

/// Remove bufferization attributes on FuncOp arguments.
static void removeBufferizationAttributes(BlockArgument bbArg) {
  auto funcOp = cast<func::FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(
      bbArg.getArgNumber(),
      bufferization::BufferizationDialect::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       bufferization::BufferizationDialect::kWritableAttrName);
}
static void removeBufferizationAttributesInModule(Operation *moduleOp) {
  moduleOp->walk([&](func::FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      ::removeBufferizationAttributes(bbArg);
  });
}

/// Helper function that extracts the source from a memref.cast. If the given
/// value is not a memref.cast result, simply returns the given value.
static Value unpackCast(Value v) {
  auto castOp = v.getDefiningOp<memref::CastOp>();
  if (!castOp)
    return v;
  return castOp.getSource();
}

/// Helper function that returns the return types (skipping casts) of the given
/// func.return ops. This function returns as many types as the return ops have
/// operands. If the i-th operand is not the same for all func.return ops, then
/// the i-th returned type is an "empty" type.
static SmallVector<Type> getReturnTypes(ArrayRef<Operation *> returnOps) {
  assert(!returnOps.empty() && "expected at least one ReturnOp");
  int numOperands = returnOps.front()->getNumOperands();

  // Helper function that unpacks memref.cast ops and returns the type.
  auto getSourceType = [&](Value v) { return unpackCast(v).getType(); };

  SmallVector<Type> result;
  for (int i = 0; i < numOperands; ++i) {
    // Get the type of the i-th operand of the first func.return ops.
    Type t = getSourceType(returnOps.front()->getOperand(i));

    // Check if all other func.return ops have a matching operand type.
    for (auto otherReturnOp : returnOps.drop_front())
      if (getSourceType(otherReturnOp->getOperand(i)) != t)
        t = Type();

    result.push_back(t);
  }

  return result;
}

/// Fold return values that are memref casts and update function return types.
///
/// During FuncOp bufferization, the exact type of the returned memrefs (if any)
/// is not known yet. Therefore, the bufferization uses memref types with the
/// most generic layout map as function return types. After bufferizing the
/// entire function body, a more concise memref type can potentially be used for
/// the return type of the function.
static void foldMemRefCasts(func::FuncOp funcOp) {
  // There is nothing to do for bodiless ops.
  if (funcOp.isDeclaration())
    return;

  FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!type)
    return;

  // Compute the common result types of all return ops.
  FailureOr<SmallVector<Operation *>> returnOps = getReturnLikeOps(funcOp);
  if (failed(returnOps))
    return;

  SmallVector<Type> resultTypes = getReturnTypes(*returnOps);

  // Remove direct casts.
  for (Operation *returnOp : *returnOps) {
    for (OpOperand &operand : returnOp->getOpOperands()) {
      // Bail if no common result type was found.
      if (resultTypes[operand.getOperandNumber()])
        operand.set(unpackCast(operand.get()));
    }
  }

  // Fill in the missing result types that were not the same among all
  // func.return ops.
  for (int i = 0; i < static_cast<int>(resultTypes.size()); ++i) {
    if (resultTypes[i])
      continue;
    resultTypes[i] = type.getResult(i);
  }

  // Update the function type.
  auto newFuncType =
      FunctionType::get(funcOp.getContext(), type.getInputs(), resultTypes);
  funcOp.setType(newFuncType);
  return;
}

static LogicalResult bufferizeOneModuleLikeOp(
    ModuleLikeOp moduleOp,
    const bufferization::OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics) {
  assert(options.bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  IRRewriter rewriter(moduleOp->getContext());

  // A list of non-circular functions in the order in which they are analyzed
  // and bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;
  // A list of all other functions. I.e., functions that call each other
  // recursively. For these, we analyze the function body but not the function
  // boundary.
  SmallVector<func::FuncOp> remainingFuncOps;

  // Try to bufferize functions in calling order. I.e., first bufferize
  // functions that do not call other functions. This allows us to infer
  // accurate buffer types for function return values. Functions that call
  // each other recursively are bufferized in an unspecified order at the end.
  // We may use unnecessarily "complex" (in terms of layout map) buffer types.
  if (failed(getFuncOpsOrderedByCalls(
          moduleOp, orderedFuncOps, remainingFuncOps, [&](func::FuncOp func) {
            return func->getParentWithTrait<OpTrait::SymbolTable>() == moduleOp;
          })))
    return failure();
  llvm::append_range(orderedFuncOps, remainingFuncOps);

  // Bufferize functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    DBGF("bufferizing func: {0}", funcOp.getName());

    if (llvm::is_contained(options.noAnalysisFuncFilter, funcOp.getName())) {
      // This function was not analyzed and RaW conflicts were not resolved.
      // Buffer copies must be inserted before every write.
      bufferization::OneShotBufferizationOptions updatedOptions = options;
      updatedOptions.copyBeforeWrite = true;
      if (failed(bufferization::bufferizeOp(funcOp.getOperation(),
                                            updatedOptions, statistics)))
        return failure();
    } else {
      if (failed(bufferization::bufferizeOp(funcOp.getOperation(), options,
                                            statistics)))
        return failure();
    }

    // Change buffer return types to more precise layout maps.
    if (options.inferFunctionResultLayout)
      foldMemRefCasts(funcOp);
  }

  // Bufferize all other ops.
  for (Operation &op : llvm::make_early_inc_range(moduleOp.getOps())) {

    // Most functions were already bufferized.
    if (isa<func::FuncOp>(op))
      continue;
    if (ModuleLikeOp(&op))
      continue;

    DBGF("bufferizing op: {0}", op);

    if (failed(bufferizeOp(&op, options, statistics)))
      return failure();
  }

  // Post-pass cleanup of function argument attributes.
  ::removeBufferizationAttributesInModule(moduleOp);

  return success();
}

static void checkConflicts(Operation *op,
                           const bufferization::AnalysisState &state) {
  SmallVector<OpOperand *> outOfPlaceOpOperands;
  DenseSet<OpOperand *> copiedOpOperands;
  SmallVector<Value> outOfPlaceValues;
  DenseSet<Value> copiedOpValues;

  // Find all out-of-place OpOperands.
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (!llvm::isa<TensorType>(operandType))
      continue;
    if (state.isInPlace(opOperand))
      continue;

    bufferization::AliasingValueList aliasingValues =
        state.getAliasingValues(opOperand);
    if (aliasingValues.getNumAliases() == 1 &&
        isa<OpResult>(aliasingValues.getAliases()[0].value) &&
        !state.bufferizesToMemoryWrite(opOperand) &&
        state.getAliasingOpOperands(aliasingValues.getAliases()[0].value)
                .getNumAliases() == 1 &&
        !isa<UnrankedTensorType>(
            aliasingValues.getAliases()[0].value.getType())) {
      // The op itself does not write but may create exactly one alias. Instead
      // of copying the OpOperand, copy the OpResult. The OpResult can sometimes
      // be smaller than the OpOperand (e.g., in the case of an extract_slice,
      // where the result is usually a smaller part of the source). Do not apply
      // this optimization if the OpResult is an unranked tensor (because those
      // cannot be copied at the moment).
      Value value = aliasingValues.getAliases()[0].value;
      outOfPlaceValues.push_back(value);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpValues.insert(value);
    } else {
      // In all other cases, make a copy of the OpOperand.
      outOfPlaceOpOperands.push_back(&opOperand);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpOperands.insert(&opOperand);
    }
  }
}

static LogicalResult
insertTensorCopiesWithinModuleScope(ModuleLikeOp op,
                                    const bufferization::AnalysisState &state) {
  IRRewriter rewriter(op->getContext());

  // We must walk the IR in pre-order because we don't want to walk ops in
  // nested symbol tables.
  WalkResult result =
      op->getRegion(0).walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        if (ModuleLikeOp(nestedOp))
          return WalkResult::skip();

        auto bufferizableOp =
            state.getOptions().dynCastBufferizableOp(nestedOp);
        if (!bufferizableOp)
          return WalkResult::advance();

        // Find inplacability conflicts and resolve them. (Typically with
        // explicit tensor copies in the form of AllocTensorOps.)
        rewriter.setInsertionPoint(nestedOp);
        if (failed(bufferizableOp.resolveConflicts(rewriter, state)))
          return WalkResult::interrupt();

        checkConflicts(nestedOp, state);

        return WalkResult::advance();
      });

  return failure(result.wasInterrupted());
}

static LogicalResult insertTensorCopiesInModule(
    ModuleLikeOp module,
    const bufferization::OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics) {
  bufferization::OneShotAnalysisState state(module, options);
  if (failed(analyzeOneModuleOp(module, state, statistics)))
    return failure();

  if (options.testAnalysisOnly)
    return success();

  return insertTensorCopiesWithinModuleScope(module, state);
}

static LogicalResult
bufferizeOneModule(ModuleLikeOp moduleOp,
                   const bufferization::OneShotBufferizationOptions &options,
                   BufferizationStatistics *statistics) {
  assert(options.bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  assert(!(options.copyBeforeWrite && options.testAnalysisOnly) &&
         "invalid combination of bufferization flags");
  if (!options.copyBeforeWrite) {
    if (options.noAnalysisFuncFilter.empty()) {
      if (failed(insertTensorCopiesInModule(moduleOp, options, statistics)))
        return failure();
    } else {
      // FuncOps whose names are specified in options.noAnalysisFuncFilter will
      // not be analyzed. Ops in these FuncOps will not be analyzed as well.
      bufferization::OpFilter::Entry::FilterFn analysisFilterFn =
          [=](Operation *op) {
            auto func = dyn_cast<func::FuncOp>(op);
            if (!func)
              func = op->getParentOfType<func::FuncOp>();

            if (func)
              return llvm::is_contained(options.noAnalysisFuncFilter,
                                        func.getName());
            return false;
          };
      bufferization::OneShotBufferizationOptions updatedOptions(options);
      updatedOptions.opFilter.denyOperation(analysisFilterFn);
      if (failed(
              insertTensorCopiesInModule(moduleOp, updatedOptions, statistics)))
        return failure();
    }
  }
  if (options.testAnalysisOnly)
    return success();
  if (failed(bufferizeOneModuleLikeOp(moduleOp, options, statistics)))
    return failure();
  return success();
}

static OneShotBufferizationOptions
getDefaultHostProgramBufferizationOptions(MLIRContext *ctx) {
  OneShotBufferizationOptions options;
  auto deviceSpace = plan::MemorySpaceAttr::get(ctx, plan::MemorySpace::device);
  options.bufferizeFunctionBoundaries = true;
  options.defaultMemorySpaceFn =
      [=](TensorType type) -> std::optional<Attribute> {
    if (auto rtt = dyn_cast<RankedTensorType>(type))
      if (auto planSpace =
              dyn_cast_if_present<plan::MemorySpaceAttr>(rtt.getEncoding()))
        return planSpace;
    return deviceSpace;
  };
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  options.bufferAlignment = 16;
  return options;
}

static std::optional<OneShotBufferizationOptions>
getBufferizationOptions(ModuleLikeOp op,
                        const OneShotBufferizationOptions &baseOptions) {
  if (auto scopeOp = dyn_cast<mtc::BufferizationScopeOpInterface>(*op)) {
    std::optional<OneShotBufferizationOptions> options =
        scopeOp.getBufferizationOptions();
    if (!options)
      return std::nullopt;
    options->testAnalysisOnly = baseOptions.testAnalysisOnly;
    options->printConflicts = baseOptions.printConflicts;
    options->analysisFuzzerSeed = baseOptions.analysisFuzzerSeed;
    options->dumpAliasSets = baseOptions.dumpAliasSets;
    return options;
  }
  return {};
}

static LogicalResult
runOneShotMultiModuleBufferize(ModuleLikeOp moduleOp,
                               BufferizationStatistics *statistics,
                               const OneShotBufferizationOptions &baseOptions) {
  SmallVector<ModuleLikeOp> modulesToBufferize;
  SymbolTable::walkSymbolTables(moduleOp, true,
                                [&](Operation *symbolTable, bool) {
                                  if (!ModuleLikeOp(symbolTable))
                                    return;
                                  modulesToBufferize.push_back(symbolTable);
                                });

  assert(modulesToBufferize.back() == moduleOp &&
         "expected moduleOp to be last in module bufferization queue");

  IRRewriter rewriter(moduleOp->getContext());
  for (ModuleLikeOp nestedModule : llvm::reverse(modulesToBufferize)) {
    std::optional<OneShotBufferizationOptions> options =
        getBufferizationOptions(nestedModule, baseOptions);
    if (!options) {
      if (moduleOp == nestedModule) {
        options = baseOptions;
      } else {
        DBGF("ignoring module: {0}", nestedModule.getSymbolName());
        continue;
      }
    }

    DBGF("bufferizing module: {0}", nestedModule.getSymbolName());

    if (failed(bufferizeOneModule(nestedModule, *options, statistics)))
      return nestedModule->emitError("failed to bufferize module");

    if (auto scopeOp =
            dyn_cast<mtc::BufferizationScopeOpInterface>(*nestedModule)) {
      if (failed(scopeOp.performPostBufferizationActions(rewriter)))
        return failure();
    }
  }

  return success();
}

namespace {
class ModuleBufferizationPass
    : public plan::impl::PlanModuleBufferizePassBase<ModuleBufferizationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleLikeOp module(getOperation());
    if (!module) {
      getOperation()->emitError()
          << getArgument()
          << " can only be scheduled on a module-like operation";
      return signalPassFailure();
    }

    // Perform bufferization (analysis, tensor copy insertion, bufferization
    // rewrites), post-bufferization fixup actions.
    OneShotBufferizationOptions options =
        getDefaultHostProgramBufferizationOptions(module->getContext());
    options.testAnalysisOnly = testAnalysisOnly;
    options.printConflicts = printConflicts;
    options.analysisFuzzerSeed = analysisFuzzerSeed;
    options.dumpAliasSets = dumpAliasSets;
    options.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
    options.copyBeforeWrite = copyBeforeWrite;
    options.checkParallelRegions = checkParallelRegions;

    if (failed(runOneShotMultiModuleBufferize(module, nullptr, options)))
      return signalPassFailure();

    // Fix up actions on the host module.
    if (failed(fixupHostModule(module, options))) {
      emitError(module->getLoc())
          << "failed to ensure correctness of read/writes to device "
             "memory "
             "from the host";
      return signalPassFailure();
    }
  }
};
} // namespace
