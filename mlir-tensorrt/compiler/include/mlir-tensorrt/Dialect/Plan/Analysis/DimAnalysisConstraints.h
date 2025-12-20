//===- DimAnalysisConstraints.h - Constraint System Interface ---*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Constraint solver interface and Union-Find implementation for dimension
/// relationship analysis. This provides an extensible interface that can
/// support more complex constraint solving in future phases.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISCONSTRAINTS_H
#define MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISCONSTRAINTS_H

#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysisTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <optional>
#include <string>

namespace mlir {
namespace plan {

//===----------------------------------------------------------------------===//
// Constraint Types
//===----------------------------------------------------------------------===//

/// Kind of constraint relationship.
enum class ConstraintKind {
  Equal, // expr1 == expr2
};

/// A constraint between two expressions.
struct Constraint {
  ConstraintKind kind;
  DimExprRef lhs;
  DimExprRef rhs;

  /// Source location for diagnostics.
  Location loc;
};

//===----------------------------------------------------------------------===//
// Constraint Solver Interface
//===----------------------------------------------------------------------===//

/// Result of attempting to add a constraint.
struct AddConstraintResult {
  enum class Status {
    Success,      // Constraint added successfully
    Unsupported,  // Constraint type not yet supported
    Contradiction // Constraint contradicts existing constraints
  };

  Status status;
  std::string message; // Diagnostic message if not Success

  static AddConstraintResult success() { return {Status::Success, ""}; }

  static AddConstraintResult unsupported(StringRef reason) {
    return {Status::Unsupported, reason.str()};
  }

  static AddConstraintResult contradiction(StringRef reason) {
    return {Status::Contradiction, reason.str()};
  }

  bool succeeded() const { return status == Status::Success; }
  bool isUnsupported() const { return status == Status::Unsupported; }
};

/// Abstract interface for constraint solving.
class ConstraintSolver {
public:
  virtual ~ConstraintSolver() = default;

  /// Add a constraint to the system.
  virtual AddConstraintResult addConstraint(const Constraint &constraint) = 0;

  /// Check if two expressions are known to be equal.
  virtual bool areEqual(DimExprRef a, DimExprRef b) = 0;

  /// Get all expressions equivalent to the given expression.
  virtual SmallVector<DimExprRef> getEquivalenceClass(DimExprRef expr) = 0;

  /// Get a canonical representative for an expression's equivalence class.
  /// If the expression is equivalent to a DimSymbol, prefer returning that.
  virtual DimExprRef getCanonicalRepresentative(DimExprRef expr) = 0;

  /// Try to find a DimSymbol equivalent to the given expression.
  virtual std::optional<StringRef> getEquivalentSymbol(DimExprRef expr) = 0;
};

//===----------------------------------------------------------------------===//
// Phase 1: Union-Find Solver using llvm::EquivalenceClasses
//===----------------------------------------------------------------------===//

/// Union-Find based constraint solver for equality constraints.
/// Uses LLVM's EquivalenceClasses which implements Tarjan's union-find
/// with path compression.
class UnionFindConstraintSolver : public ConstraintSolver {
public:
  explicit UnionFindConstraintSolver(DimExprFactory &factory);

  AddConstraintResult addConstraint(const Constraint &constraint) override;
  bool areEqual(DimExprRef a, DimExprRef b) override;
  SmallVector<DimExprRef> getEquivalenceClass(DimExprRef expr) override;
  DimExprRef getCanonicalRepresentative(DimExprRef expr) override;
  std::optional<StringRef> getEquivalentSymbol(DimExprRef expr) override;

private:
  DimExprFactory &factory;

  /// LLVM's EquivalenceClasses for DimVariable with custom comparator.
  llvm::EquivalenceClasses<DimVariable> equivalences;

  /// Map from variable to its expression (for returning DimExprRef).
  llvm::DenseMap<DimVariable, DimExprRef> variableToExpr;

  /// Get or create the expression for a variable.
  DimExprRef getOrCreateExpr(const DimVariable &var);
};

} // namespace plan
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISCONSTRAINTS_H
