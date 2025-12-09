//===- DimAnalysis.h - Dimension Relationship Analysis ----------*- C++ -*-===//
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
/// Main dimension relationship analysis class. This analysis determines
/// equality relationships among dynamic dimension extent values for different
/// function arguments.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSIS_H
#define MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSIS_H

#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysisConstraints.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysisTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <optional>

namespace mlir {
namespace plan {

//===----------------------------------------------------------------------===//
// Symbol Maps
//===----------------------------------------------------------------------===//

/// Information about JAX dimension symbols in functions.
struct FunctionSymbolInfo {
  /// Map from symbol name to the function argument.
  llvm::StringMap<BlockArgument> symbolToArg;

  /// Map from argument index to optional symbol name.
  SmallVector<std::optional<StringRef>> argToSymbol;
};

//===----------------------------------------------------------------------===//
// Analysis Options
//===----------------------------------------------------------------------===//

/// Options controlling the dimension analysis behavior.
struct DimAnalysisOptions {
  /// Attribute name for JAX global constant annotations.
  StringRef globalConstantAttrName = "jax.global_constant";
};

//===----------------------------------------------------------------------===//
// Main Analysis Class
//===----------------------------------------------------------------------===//

/// Analysis that determines relationships between dynamic dimension values.
/// The analysis is scoped to a specific entrypoint function and only considers
/// operations in the top-level blocks of functions reachable from the
/// entrypoint. This avoids issues with control-flow operations like scf.if
/// where assertions inside branches may not always execute.
class DimensionRelationshipAnalysis {
public:
  /// Create an analysis for the given entrypoint function.
  /// The entrypoint should be a public function in the module.
  explicit DimensionRelationshipAnalysis(func::FuncOp entrypoint,
                                         DimAnalysisOptions options = {});

  /// Run the analysis.
  LogicalResult run();

  /// Get the entrypoint function.
  func::FuncOp getEntrypoint() const { return entrypoint; }

  /// Get the module.
  ModuleOp getModule() const { return module; }

  //===--------------------------------------------------------------------===//
  // Query API
  //===--------------------------------------------------------------------===//

  DimExprRef getExprFromValue(Value v);
  DimExprRef getExprFromTensorDim(Value tensor, int32_t dim);
  DimExprRef getExprFromSymbol(StringRef symbol);

  /// Check if two valus are provable equal.
  bool areEqual(DimExprRef v1, DimExprRef v2);

  /// Check if two tensor dimensions are known to be equal.
  bool areDimensionsEqual(Value tensor1, int32_t dim1, Value tensor2,
                          int32_t dim2);

  /// Check if a tensor dimension equals a named symbol.
  bool dimensionEqualsSymbol(Value tensor, int32_t dim, StringRef symbol);

  /// Get the symbol name for a tensor dimension, if known.
  std::optional<StringRef> getSymbolForDimension(Value tensor, int32_t dim);

  /// Get all tensor dimensions known to be equal to a given dimension.
  SmallVector<TensorDim> getEqualDimensions(Value tensor, int32_t dim);

  /// Check if two SSA values represent the same dimension value.
  bool areValuesEqual(Value v1, Value v2);

  /// Get the symbol name for an SSA value, if known.
  std::optional<StringRef> getSymbolForValue(Value v);

  /// Return the function symbol info if it exists for the given function.
  const FunctionSymbolInfo *getFunctionSymbolInfo(func::FuncOp func) const;

  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//

  /// Get the expression factory for creating new expressions.
  DimExprFactory &getExprFactory() { return exprFactory; }

  /// Get the constraint solver.
  ConstraintSolver &getSolver() { return *solver; }

  /// Print all equivalence classes to the given output stream.
  void printEquivalenceClasses(raw_ostream &os) const;

private:
  ModuleOp module;
  func::FuncOp entrypoint;
  DimAnalysisOptions options;

  /// Expression factory (manages expression allocation).
  DimExprFactory exprFactory;

  /// Constraint solver.
  std::unique_ptr<ConstraintSolver> solver;

  /// Functions reachable from the entrypoint.
  DenseSet<func::FuncOp> reachableFunctions;

  /// Symbol information per function.
  DenseMap<func::FuncOp, FunctionSymbolInfo> functionSymbols;

  /// Collect functions reachable from the entrypoint via the call graph.
  void collectReachableFunctions();

  /// Build symbol maps from function arguments.
  void buildSymbolMaps();

  /// Walk IR and collect constraints.
  LogicalResult collectConstraints();
};

} // namespace plan
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSIS_H
