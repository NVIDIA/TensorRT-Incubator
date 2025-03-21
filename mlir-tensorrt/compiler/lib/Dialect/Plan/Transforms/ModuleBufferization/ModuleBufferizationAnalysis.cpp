//===- ModuleBufferizationAnalysis.cpp ------------------------------------===//
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
/// Implementation of the Plan bufferization analysis.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define DEBUG_TYPE "plan-module-bufferize"
#define DBGS() (llvm::dbgs() << "[plan-module-bufferize] ")
#define DBGF(fmt, ...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv(                                    \
                 stderr, "{0}:{1}:{2}(): ", "ModuleBufferization.cpp",         \
                 __LINE__, __func__);                                          \
             llvm::dbgs() << llvm::formatv(fmt "\n", __VA_ARGS__));

using namespace mlir;
using namespace mlir::bufferization;
using FuncAnalysisState = mlir::bufferization::func_ext::FuncAnalysisState;
using namespace mlir::plan;

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

/// Get or create FuncAnalysisState.
static FuncAnalysisState &
getOrCreateFuncAnalysisState(OneShotAnalysisState &state) {
  auto *result = state.getExtension<FuncAnalysisState>();
  if (result)
    return *result;
  return state.addExtension<FuncAnalysisState>();
}

static void annotateFuncArgAccess(func::FuncOp funcOp, int64_t idx, bool isRead,
                                  bool isWritten) {
  OpBuilder b(funcOp.getContext());
  Attribute accessType;
  if (isRead && isWritten) {
    accessType = b.getStringAttr("read-write");
  } else if (isRead) {
    accessType = b.getStringAttr("read");
  } else if (isWritten) {
    accessType = b.getStringAttr("write");
  } else {
    accessType = b.getStringAttr("none");
  }
  funcOp.setArgAttr(idx, BufferizationDialect::kBufferAccessAttrName,
                    accessType);
}

/// Annotate IR with the results of the analysis. For testing purposes only.
static void annotateEquivalentReturnBbArg(OpOperand &returnVal,
                                          BlockArgument bbArg) {
  const char *kEquivalentArgsAttr = "__equivalent_func_args__";
  Operation *op = returnVal.getOwner();

  SmallVector<int64_t> equivBbArgs;
  if (op->hasAttr(kEquivalentArgsAttr)) {
    auto attr = cast<ArrayAttr>(op->getAttr(kEquivalentArgsAttr));
    equivBbArgs = llvm::to_vector<4>(llvm::map_range(attr, [](Attribute a) {
      return cast<IntegerAttr>(a).getValue().getSExtValue();
    }));
  } else {
    equivBbArgs.append(op->getNumOperands(), -1);
  }
  equivBbArgs[returnVal.getOperandNumber()] = bbArg.getArgNumber();

  OpBuilder b(op->getContext());
  op->setAttr(kEquivalentArgsAttr, b.getI64ArrayAttr(equivBbArgs));
}

/// Store function BlockArguments that are equivalent to/aliasing a returned
/// value in FuncAnalysisState.
static LogicalResult
aliasingFuncOpBBArgsAnalysis(func::FuncOp funcOp, OneShotAnalysisState &state,
                             FuncAnalysisState &funcState) {
  if (funcOp.isDeclaration()) {
    // No function body available. Conservatively assume that every tensor
    // return value may alias with any tensor bbArg.
    FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
    if (!type)
      return failure();
    for (const auto &inputIt : llvm::enumerate(type.getInputs())) {
      if (!isa<TensorType>(inputIt.value()))
        continue;
      for (const auto &resultIt : llvm::enumerate(type.getResults())) {
        if (!isa<TensorType>(resultIt.value()))
          continue;
        int64_t returnIdx = resultIt.index();
        int64_t bbArgIdx = inputIt.index();
        funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(returnIdx);
      }
    }
    return success();
  }

  // Find all func.return ops.
  FailureOr<SmallVector<Operation *>> returnOps = getReturnLikeOps(funcOp);
  if (failed(returnOps))
    return failure();
  assert(!returnOps->empty() && "expected at least one ReturnOp");

  // Build alias sets. Merge all aliases from all func.return ops.
  for (BlockArgument bbArg : funcOp.getArguments()) {
    if (isa<RankedTensorType>(bbArg.getType())) {
      int64_t bbArgIdx = bbArg.getArgNumber();
      // Store aliases in a set, so that we don't add the same alias twice.
      SetVector<int64_t> aliases;
      for (Operation *returnOp : *returnOps) {
        for (OpOperand &returnVal : returnOp->getOpOperands()) {
          if (isa<RankedTensorType>(returnVal.get().getType())) {
            int64_t returnIdx = returnVal.getOperandNumber();
            if (state.areAliasingBufferizedValues(returnVal.get(), bbArg))
              aliases.insert(returnIdx);
          }
        }
      }
      for (int64_t alias : aliases)
        funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(alias);
    }
  }

  // Build equivalence sets.
  // Helper function that finds an equivalent block argument index for the
  // given OpOperand. Return std::nullopt if no equivalent block argument could
  // be found.
  auto findEquivalentBlockArgIdx =
      [&](OpOperand &opOperand) -> std::optional<int64_t> {
    Value v = opOperand.get();
    if (!isa<TensorType>(v.getType()))
      return std::nullopt;
    for (BlockArgument bbArg : funcOp.getArguments()) {
      if (isa<RankedTensorType>(bbArg.getType())) {
        if (state.areEquivalentBufferizedValues(v, bbArg)) {
          if (state.getOptions().testAnalysisOnly)
            annotateEquivalentReturnBbArg(opOperand, bbArg);
          return bbArg.getArgNumber();
        }
      }
    }
    return std::nullopt;
  };

  int64_t numResults = returnOps->front()->getNumOperands();
  for (int64_t i = 0; i < numResults; ++i) {
    // Find the equivalent block argument index for the i-th operand of the
    // first func.return op.
    std::optional<int64_t> maybeEquiv =
        findEquivalentBlockArgIdx(returnOps->front()->getOpOperand(i));
    if (!maybeEquiv.has_value())
      continue;
    int64_t bbArgIdx = *maybeEquiv;
    bool allEquiv = true;

    // Check if all other func.return ops have the same equivalent block
    // argument for the i-th operand. In contrast to aliasing information,
    // which is just "merged", equivalence information must match across all
    // func.return ops.
    for (Operation *returnOp : ArrayRef(*returnOps).drop_front()) {
      std::optional<int64_t> maybeEquiv =
          findEquivalentBlockArgIdx(returnOp->getOpOperand(i));
      if (maybeEquiv != bbArgIdx) {
        allEquiv = false;
        break;
      }
    }

    // All func.return ops have the same equivalent block argument for the i-th
    // operand.
    if (allEquiv)
      funcState.equivalentFuncArgs[funcOp][i] = bbArgIdx;
  }

  return success();
}

/// Determine which FuncOp bbArgs are read and which are written. When run on a
/// function with unknown ops, we conservatively assume that such ops bufferize
/// to a read + write.
static LogicalResult
funcOpBbArgReadWriteAnalysis(func::FuncOp funcOp, OneShotAnalysisState &state,
                             FuncAnalysisState &funcState) {
  for (int64_t idx = 0, e = funcOp.getNumArguments(); idx < e; ++idx) {
    // Skip non-tensor arguments.
    if (!isa<TensorType>(funcOp.getFunctionType().getInput(idx)))
      continue;
    bool isRead;
    bool isWritten;
    if (auto accessAttr = funcOp.getArgAttrOfType<StringAttr>(
            idx, BufferizationDialect::kBufferAccessAttrName)) {
      // Buffer access behavior is specified on the function. Skip the analysis.
      StringRef str = accessAttr.getValue();
      isRead = str == "read" || str == "read-write";
      isWritten = str == "write" || str == "read-write";
    } else if (funcOp.isDeclaration()) {
      // If the function has no body, conservatively assume that all args are
      // read + written.
      isRead = true;
      isWritten = true;
    } else {
      // Analyze the body of the function.
      BlockArgument bbArg = funcOp.getArgument(idx);
      isRead = state.isValueRead(bbArg);
      isWritten = state.isValueWritten(bbArg);
    }

    if (state.getOptions().testAnalysisOnly)
      annotateFuncArgAccess(funcOp, idx, isRead, isWritten);
    if (isRead)
      funcState.readBbArgs[funcOp].insert(idx);
    if (isWritten)
      funcState.writtenBbArgs[funcOp].insert(idx);
  }

  return success();
}

LogicalResult plan::analyzeOneModuleOp(ModuleLikeOp moduleOp,
                                       OneShotAnalysisState &state,
                                       BufferizationStatistics *statistics) {
  assert(state.getOptions().bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  FuncAnalysisState &funcState = getOrCreateFuncAnalysisState(state);

  // A list of non-circular functions in the order in which they are analyzed
  // and bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;
  // A list of all other functions. I.e., functions that call each other
  // recursively. For these, we analyze the function body but not the function
  // boundary.
  SmallVector<func::FuncOp> remainingFuncOps;

  if (failed(getFuncOpsOrderedByCalls(
          moduleOp, orderedFuncOps, remainingFuncOps, [&](func::FuncOp func) {
            return func->getParentWithTrait<OpTrait::SymbolTable>() == moduleOp;
          })))
    return failure();

  // Analyze functions in order. Starting with functions that are not calling
  // any other functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    if (!state.getOptions().isOpAllowed(funcOp))
      continue;

    // Now analyzing function.
    funcState.startFunctionAnalysis(funcOp);

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, state, statistics)))
      return failure();

    // Run some extra function analyses.
    if (failed(aliasingFuncOpBBArgsAnalysis(funcOp, state, funcState)) ||
        failed(funcOpBbArgReadWriteAnalysis(funcOp, state, funcState)))
      return failure();

    // Mark op as fully analyzed.
    funcState.analyzedFuncOps[funcOp] = func_ext::FuncOpAnalysisState::Analyzed;
  }

  // Analyze all other functions. All function boundary analyses are skipped.
  for (func::FuncOp funcOp : remainingFuncOps) {
    if (!state.getOptions().isOpAllowed(funcOp))
      continue;

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, state, statistics)))
      return failure();

    // TODO: We currently skip all function argument analyses for functions
    // that call each other circularly. These analyses do not support recursive
    // calls yet. The `BufferizableOpInterface` implementations of `func`
    // dialect ops return conservative results in the absence of analysis
    // information.
  }

  return success();
}

void FuncAnalysisStateInfo::appendToState(
    bufferization::OneShotAnalysisState &state) const {
  func_ext::FuncAnalysisState &other = getOrCreateFuncAnalysisState(state);
  for (const auto &[func, s] : analyzedFuncOps)
    other.analyzedFuncOps[func] = s;
  for (const auto &[func, s] : equivalentFuncArgs)
    other.equivalentFuncArgs[func] = s;
  for (const auto &[func, s] : writtenBbArgs)
    other.writtenBbArgs[func] = s;
  for (const auto &[func, s] : readBbArgs)
    other.readBbArgs[func] = s;
  for (const auto &[func, s] : aliasingReturnVals)
    other.aliasingReturnVals[func] = s;
}

void plan::setupAnalysisStateForModule(ModuleLikeOp moduleOp,
                                       const ModuleFuncAnalysisCache &funcInfo,
                                       OneShotAnalysisState &newState) {
  for (const auto &[otherModule, funcInfo] : funcInfo) {
    if (!moduleOp->isProperAncestor(otherModule))
      continue;
    funcInfo.appendToState(newState);
  }
}

void plan::appendAnalysisResultsToCache(
    ModuleLikeOp op, ModuleFuncAnalysisCache &cache,
    const bufferization::OneShotAnalysisState &state) {
  const auto *result = state.getExtension<FuncAnalysisState>();
  if (!result)
    return;
  cache.insert(std::make_pair(*op, FuncAnalysisStateInfo(*result)));
}
