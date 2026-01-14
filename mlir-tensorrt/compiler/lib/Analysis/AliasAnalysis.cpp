//===- AliasAnalysis.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Extended alias analysis with support for executor-specific constructs.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Analysis/AliasAnalysis.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;

/// Check if a value is a function argument.
static bool isFuncArg(Value val) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg)
    return false;
  return isa_and_nonnull<FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
}

/// Check if a function argument has the "executor.restrict" attribute.
static bool hasRestrictAttr(Value val) {
  auto blockArg = cast<BlockArgument>(val);
  auto func = cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  return func.getArgAttr(blockArg.getArgNumber(), "executor.restrict") !=
         nullptr;
}

/// Get the function argument that an executor.abi.recv reads from, if any.
static BlockArgument getABIRecvSourceArg(Value val) {
  if (auto *defOp = val.getDefiningOp()) {
    if (auto recvOp = dyn_cast<executor::ABIRecvOp>(defOp)) {
      if (auto blockArg = dyn_cast<BlockArgument>(recvOp.getPtr()))
        return blockArg;
    }
  }
  return nullptr;
}

AliasResult RestrictAwareAliasAnalysis::aliasImpl(Value lhs, Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;

  // Check if both values are function arguments with restrict attribute.
  if (isFuncArg(lhs) && isFuncArg(rhs)) {
    if (hasRestrictAttr(lhs) || hasRestrictAttr(rhs))
      return AliasResult::NoAlias;
  }

  // Check if values come from executor.abi.recv with different source args.
  BlockArgument lhsRecvArg = getABIRecvSourceArg(lhs);
  BlockArgument rhsRecvArg = getABIRecvSourceArg(rhs);
  if (lhsRecvArg && rhsRecvArg && lhsRecvArg != rhsRecvArg)
    return AliasResult::NoAlias;

  // Also handle the case where one is a func arg and the other is from
  // abi.recv with a different func arg source.
  if (isFuncArg(lhs) && rhsRecvArg) {
    auto lhsBlockArg = cast<BlockArgument>(lhs);
    if (lhsBlockArg != rhsRecvArg)
      return AliasResult::NoAlias;
  }
  if (isFuncArg(rhs) && lhsRecvArg) {
    auto rhsBlockArg = cast<BlockArgument>(rhs);
    if (rhsBlockArg != lhsRecvArg)
      return AliasResult::NoAlias;
  }

  return LocalAliasAnalysis::aliasImpl(lhs, rhs);
}

AliasAnalysis mlir::createRestrictAwareAliasAnalysis(Operation *op) {
  AliasAnalysis aa(op);
  aa.addAnalysisImplementation(RestrictAwareAliasAnalysis());
  return aa;
}
