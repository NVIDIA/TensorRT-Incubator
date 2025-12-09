//===- DimAnalysisTypes.h - Dimension Analysis Types ------------*- C++ -*-===//
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
/// Types and expressions for dimension relationship analysis. This file defines
/// the variable types (IntegerSSAValue, TensorDim, DimSymbol) and expression
/// hierarchy used to represent dimension constraints.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISTYPES_H
#define MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISTYPES_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace mlir {
namespace plan {

//===----------------------------------------------------------------------===//
// Variable Types (Leaf nodes in expressions)
//===----------------------------------------------------------------------===//

/// An integer SSA value (scalar i32/i64 or 0-rank tensor<iN>).
struct IntegerSSAValue {
  Value value;

  bool operator==(const IntegerSSAValue &other) const {
    return value == other.value;
  }
  bool operator!=(const IntegerSSAValue &other) const {
    return !(*this == other);
  }
  bool operator<(const IntegerSSAValue &other) const {
    return value.getAsOpaquePointer() < other.value.getAsOpaquePointer();
  }
};

/// A dimension of a tensor-typed SSA value.
struct TensorDim {
  Value tensor;
  int32_t dimIndex;

  bool operator==(const TensorDim &other) const {
    return tensor == other.tensor && dimIndex == other.dimIndex;
  }
  bool operator!=(const TensorDim &other) const { return !(*this == other); }
  bool operator<(const TensorDim &other) const {
    if (tensor.getAsOpaquePointer() != other.tensor.getAsOpaquePointer())
      return tensor.getAsOpaquePointer() < other.tensor.getAsOpaquePointer();
    return dimIndex < other.dimIndex;
  }
};

/// A named symbolic dimension (e.g., from jax.export.symbolic_shape).
struct DimSymbol {
  std::string symbolName;

  bool operator==(const DimSymbol &other) const {
    return symbolName == other.symbolName;
  }
  bool operator!=(const DimSymbol &other) const { return !(*this == other); }
  bool operator<(const DimSymbol &other) const {
    return symbolName < other.symbolName;
  }
};

/// A variable that can appear in dimension expressions.
/// This is extensible - we can add more variants as needed.
using DimVariable = std::variant<IntegerSSAValue, TensorDim, DimSymbol>;

/// Comparator for DimVariable to enable use with ordered containers.
struct DimVariableComparator {
  bool operator()(const DimVariable &lhs, const DimVariable &rhs) const {
    // First compare by variant index (type)
    if (lhs.index() != rhs.index())
      return lhs.index() < rhs.index();
    // Then compare values within the same type
    return std::visit(
        [](const auto &l, const auto &r) -> bool {
          using T = std::decay_t<decltype(l)>;
          using U = std::decay_t<decltype(r)>;
          if constexpr (std::is_same_v<T, U>) {
            return l < r;
          }
          return false; // Should not happen since indices are equal
        },
        lhs, rhs);
  }
};

/// Get a human-readable description of a DimVariable.
std::string describeDimVariable(const DimVariable &var);

//===----------------------------------------------------------------------===//
// Expression Types
//===----------------------------------------------------------------------===//

/// Forward declaration for expression pointer.
class DimExpr;
using DimExprRef = const DimExpr *;

/// Base class for dimension expressions.
class DimExpr {
public:
  /// Check if this is a variable expression.
  bool isVariable() const { return isVariableExpr; }

  /// Get a human-readable description for diagnostics.
  virtual std::string describe() const = 0;

  virtual ~DimExpr() = default;

protected:
  explicit DimExpr(bool isVariable) : isVariableExpr(isVariable) {}

private:
  bool isVariableExpr;
};

/// A variable expression (leaf node).
class VariableExpr : public DimExpr {
public:
  explicit VariableExpr(DimVariable var)
      : DimExpr(/*isVariable=*/true), variable(std::move(var)) {}

  const DimVariable &getVariable() const { return variable; }

  std::string describe() const override {
    return describeDimVariable(variable);
  }

  static bool classof(const DimExpr *e) { return e->isVariable(); }

private:
  DimVariable variable;
};

//===----------------------------------------------------------------------===//
// Expression Factory
//===----------------------------------------------------------------------===//

/// Factory for creating and managing dimension expressions.
/// Provides caching to ensure the same variable always maps to the same
/// expression.
class DimExprFactory {
public:
  /// Get or create an expression for an integer SSA value.
  DimExprRef getOrCreateValueExpr(Value value);

  /// Get or create an expression for a tensor dimension.
  DimExprRef getOrCreateTensorDimExpr(Value tensor, int32_t dim);

  /// Get or create an expression for a named symbol.
  DimExprRef getOrCreateSymbolExpr(StringRef symbol);

  /// Get all cached value expressions (for iteration).
  const llvm::DenseMap<Value, DimExprRef> &getValueExprs() const {
    return valueToExpr;
  }

  /// Get all cached tensor dimension expressions (for iteration).
  const llvm::DenseMap<std::pair<Value, int32_t>, DimExprRef> &
  getTensorDimExprs() const {
    return tensorDimToExpr;
  }

private:
  /// Create a variable expression (internal, uncached).
  DimExprRef createVariable(DimVariable var);

  /// Storage for variable expressions.
  std::vector<std::unique_ptr<VariableExpr>> variableExprs;

  /// Cache from SSA values to their expressions.
  llvm::DenseMap<Value, DimExprRef> valueToExpr;

  /// Cache from (tensor, dim) to expressions.
  llvm::DenseMap<std::pair<Value, int32_t>, DimExprRef> tensorDimToExpr;

  /// Cache from symbol name to expressions.
  llvm::StringMap<DimExprRef> symbolToExpr;
};

} // namespace plan
} // namespace mlir

//===----------------------------------------------------------------------===//
// DenseMapInfo specializations
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct DenseMapInfo<mlir::plan::IntegerSSAValue> {
  static mlir::plan::IntegerSSAValue getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey()};
  }
  static mlir::plan::IntegerSSAValue getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey()};
  }
  static unsigned getHashValue(const mlir::plan::IntegerSSAValue &val) {
    return DenseMapInfo<mlir::Value>::getHashValue(val.value);
  }
  static bool isEqual(const mlir::plan::IntegerSSAValue &lhs,
                      const mlir::plan::IntegerSSAValue &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::plan::TensorDim> {
  static mlir::plan::TensorDim getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey(), -1};
  }
  static mlir::plan::TensorDim getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey(), -2};
  }
  static unsigned getHashValue(const mlir::plan::TensorDim &val) {
    return llvm::hash_combine(
        DenseMapInfo<mlir::Value>::getHashValue(val.tensor), val.dimIndex);
  }
  static bool isEqual(const mlir::plan::TensorDim &lhs,
                      const mlir::plan::TensorDim &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::plan::DimSymbol> {
  static mlir::plan::DimSymbol getEmptyKey() {
    return {DenseMapInfo<StringRef>::getEmptyKey().str()};
  }
  static mlir::plan::DimSymbol getTombstoneKey() {
    return {DenseMapInfo<StringRef>::getTombstoneKey().str()};
  }
  static unsigned getHashValue(const mlir::plan::DimSymbol &val) {
    return llvm::hash_value(StringRef(val.symbolName));
  }
  static bool isEqual(const mlir::plan::DimSymbol &lhs,
                      const mlir::plan::DimSymbol &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<mlir::plan::DimVariable> {
  static mlir::plan::DimVariable getEmptyKey() {
    return mlir::plan::IntegerSSAValue{
        DenseMapInfo<mlir::Value>::getEmptyKey()};
  }
  static mlir::plan::DimVariable getTombstoneKey() {
    return mlir::plan::IntegerSSAValue{
        DenseMapInfo<mlir::Value>::getTombstoneKey()};
  }
  static unsigned getHashValue(const mlir::plan::DimVariable &val) {
    return std::visit(
        [](const auto &v) {
          using T = std::decay_t<decltype(v)>;
          return DenseMapInfo<T>::getHashValue(v);
        },
        val);
  }
  static bool isEqual(const mlir::plan::DimVariable &lhs,
                      const mlir::plan::DimVariable &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_DIMANALYSISTYPES_H
