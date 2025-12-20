//===- DimAnalysisConstraints.cpp - Constraint Solver Impl ------*- C++ -*-===//
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
/// Implementation of the Union-Find constraint solver using LLVM's
/// EquivalenceClasses utility.
///
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysisConstraints.h"

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// UnionFindConstraintSolver Implementation
//===----------------------------------------------------------------------===//

UnionFindConstraintSolver::UnionFindConstraintSolver(DimExprFactory &factory)
    : factory(factory) {}

DimExprRef UnionFindConstraintSolver::getOrCreateExpr(const DimVariable &var) {
  auto it = variableToExpr.find(var);
  if (it != variableToExpr.end())
    return it->second;

  DimExprRef expr = std::visit(
      [this](const auto &v) -> DimExprRef {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, IntegerSSAValue>) {
          return factory.getOrCreateValueExpr(v.value);
        } else if constexpr (std::is_same_v<T, TensorDim>) {
          return factory.getOrCreateTensorDimExpr(v.tensor, v.dimIndex);
        } else if constexpr (std::is_same_v<T, DimSymbol>) {
          return factory.getOrCreateSymbolExpr(v.symbolName);
        }
      },
      var);
  variableToExpr[var] = expr;
  return expr;
}

AddConstraintResult
UnionFindConstraintSolver::addConstraint(const Constraint &constraint) {
  assert(constraint.kind == ConstraintKind::Equal &&
         "Only equality constraints are supported");

  if (!constraint.lhs->isVariable()) {
    return AddConstraintResult::unsupported(
        "LHS of equality is not a simple variable: " +
        constraint.lhs->describe());
  }

  if (!constraint.rhs->isVariable()) {
    return AddConstraintResult::unsupported(
        "RHS of equality is not a simple variable: " +
        constraint.rhs->describe());
  }

  // Get variables and union the sets
  const auto &lhsVar = cast<VariableExpr>(constraint.lhs)->getVariable();
  const auto &rhsVar = cast<VariableExpr>(constraint.rhs)->getVariable();

  getOrCreateExpr(lhsVar);
  getOrCreateExpr(rhsVar);
  equivalences.unionSets(lhsVar, rhsVar);

  return AddConstraintResult::success();
}

bool UnionFindConstraintSolver::areEqual(DimExprRef a, DimExprRef b) {
  if (!a->isVariable() || !b->isVariable())
    return false;

  const auto &varA = cast<VariableExpr>(a)->getVariable();
  const auto &varB = cast<VariableExpr>(b)->getVariable();

  return equivalences.isEquivalent(varA, varB);
}

SmallVector<DimExprRef>
UnionFindConstraintSolver::getEquivalenceClass(DimExprRef expr) {
  SmallVector<DimExprRef> result;

  if (!expr->isVariable())
    return result;

  const auto &var = cast<VariableExpr>(expr)->getVariable();

  // Find the leader member_iterator for this variable
  auto leaderMI = equivalences.findLeader(var);
  if (leaderMI == equivalences.member_end())
    return result;

  // Iterate over all members starting from the leader
  for (auto mi = leaderMI; mi != equivalences.member_end(); ++mi) {
    result.push_back(getOrCreateExpr(*mi));
  }

  return result;
}

DimExprRef
UnionFindConstraintSolver::getCanonicalRepresentative(DimExprRef expr) {
  if (!expr->isVariable())
    return expr;

  const auto &var = cast<VariableExpr>(expr)->getVariable();

  // Get the leader value
  auto it = equivalences.findLeader(var);
  if (it == equivalences.member_end())
    return expr;

  const DimVariable &leader = equivalences.getLeaderValue(var);
  return getOrCreateExpr(leader);
}

std::optional<StringRef>
UnionFindConstraintSolver::getEquivalentSymbol(DimExprRef expr) {
  auto equivClass = getEquivalenceClass(expr);

  for (DimExprRef e : equivClass) {
    if (const auto *varExpr = dyn_cast<VariableExpr>(e)) {
      if (const auto *symbol =
              std::get_if<DimSymbol>(&varExpr->getVariable())) {
        return StringRef(symbol->symbolName);
      }
    }
  }

  return std::nullopt;
}
