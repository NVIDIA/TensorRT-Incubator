//===- DimAnalysis.cpp - Dimension Analysis Implementation ------*- C++ -*-===//
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
/// Implementation of the dimension relationship analysis.
///
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// ConstraintCollector - Helper for collecting constraints from operations
//===----------------------------------------------------------------------===//

namespace {

/// Helper struct that collects constraints from operations.
/// This is a friend of DimensionRelationshipAnalysis to access internals.
struct ConstraintCollector {
  DimensionRelationshipAnalysis &analysis;

  explicit ConstraintCollector(DimensionRelationshipAnalysis &analysis)
      : analysis(analysis) {}

  //===--------------------------------------------------------------------===//
  // Expression Building Helpers
  //===--------------------------------------------------------------------===//

  DimExprRef exprFromValue(Value value) {
    return analysis.getExprFactory().getOrCreateValueExpr(value);
  }

  DimExprRef exprFromTensorDim(Value tensor, int32_t dim) {
    return analysis.getExprFactory().getOrCreateTensorDimExpr(tensor, dim);
  }

  DimExprRef exprFromSymbol(StringRef symbol) {
    return analysis.getExprFactory().getOrCreateSymbolExpr(symbol);
  }

  void addConstraint(const Constraint &constraint) {
    (void)analysis.getSolver().addConstraint(constraint);
  }

  //===--------------------------------------------------------------------===//
  // Operation Visitors - One per operation type
  //===--------------------------------------------------------------------===//

  void visit(func::CallOp callOp) {
    auto callee =
        analysis.getModule().lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee)
      return;

    const FunctionSymbolInfo *symbolInfo =
        analysis.getFunctionSymbolInfo(callee);
    if (!symbolInfo)
      return;

    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      assert(i < symbolInfo->argToSymbol.size() &&
             "Argument index out of bounds");
      if (std::optional<StringRef> symbolName = symbolInfo->argToSymbol[i]) {
        Value operand = callOp.getOperand(i);
        DimExprRef lhs = exprFromValue(operand);
        DimExprRef rhs = exprFromSymbol(*symbolName);
        addConstraint({ConstraintKind::Equal, lhs, rhs, callOp.getLoc()});
      }
    }
  }

  void visit(stablehlo::GetDimensionSizeOp op) {
    Value tensor = op.getOperand();
    Value result = op.getResult();
    int32_t dim = static_cast<int32_t>(op.getDimension());

    DimExprRef lhs = exprFromValue(result);
    DimExprRef rhs = exprFromTensorDim(tensor, dim);
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(tensor::DimOp op) {
    auto constIndex = op.getConstantIndex();
    if (!constIndex)
      return;

    Value tensor = op.getSource();
    Value result = op.getResult();
    int32_t dim = static_cast<int32_t>(*constIndex);

    DimExprRef lhs = exprFromValue(result);
    DimExprRef rhs = exprFromTensorDim(tensor, dim);
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() != "shape_assertion")
      return;
    if (op.getInputs().empty())
      return;

    Value cmpResult = op.getInputs().front();
    auto equality = getEqualityFromCmp(cmpResult);
    if (!equality)
      return;

    auto [lhsValue, rhsValue] = *equality;
    DimExprRef lhs = exprFromValue(lhsValue);
    DimExprRef rhs = exprFromValue(rhsValue);
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(cf::AssertOp op) {
    Value cmpResult = op.getArg();
    auto equality = getEqualityFromCmp(cmpResult);
    if (!equality)
      return;

    auto [lhsValue, rhsValue] = *equality;
    DimExprRef lhs = exprFromValue(lhsValue);
    DimExprRef rhs = exprFromValue(rhsValue);
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(tensor::ExtractOp op) {
    auto tensorType = dyn_cast<RankedTensorType>(op.getTensor().getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return;
    if (tensorType.getNumElements() != 1)
      return;
    if (!tensorType.getElementType().isIntOrIndex())
      return;

    DimExprRef lhs = exprFromValue(op.getResult());
    DimExprRef rhs = exprFromValue(op.getTensor());
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(tensor::FromElementsOp op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType || !resultType.hasStaticShape())
      return;
    if (resultType.getNumElements() != 1)
      return;
    if (!resultType.getElementType().isIntOrIndex())
      return;

    assert(op.getElements().size() == 1);
    Value scalarElement = op.getElements().front();

    DimExprRef lhs = exprFromValue(op.getResult());
    DimExprRef rhs = exprFromValue(scalarElement);
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(plan::WithShapeOp op) {
    // plan.with_shape links a tensor with SSA values representing its shape.
    // For each dimension i: shape[i] == dim(operand, i)
    Value tensor = op.getOperand();
    Value tensorResult = op.getResult();
    for (auto [dimIdx, shapeValue] : llvm::enumerate(op.getShape())) {
      DimExprRef lhs = exprFromValue(shapeValue);
      DimExprRef rhs = exprFromTensorDim(tensor, static_cast<int32_t>(dimIdx));
      DimExprRef res =
          exprFromTensorDim(tensorResult, static_cast<int32_t>(dimIdx));
      addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
      addConstraint({ConstraintKind::Equal, res, lhs, op.getLoc()});
    }
  }

  void visit(stablehlo::ReshapeOp op) {
    // For single-element tensors, reshape preserves the value.
    auto inputType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType)
      return;
    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return;
    if (inputType.getNumElements() != 1 || resultType.getNumElements() != 1)
      return;
    if (!inputType.getElementType().isIntOrIndex())
      return;

    DimExprRef lhs = exprFromValue(op.getResult());
    DimExprRef rhs = exprFromValue(op.getOperand());
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(arith::IndexCastOp op) {
    // index_cast is treated as identity for dimension analysis.
    DimExprRef lhs = exprFromValue(op.getResult());
    DimExprRef rhs = exprFromValue(op.getIn());
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  void visit(arith::IndexCastUIOp op) {
    // index_castui is treated as identity for dimension analysis.
    DimExprRef lhs = exprFromValue(op.getResult());
    DimExprRef rhs = exprFromValue(op.getIn());
    addConstraint({ConstraintKind::Equal, lhs, rhs, op.getLoc()});
  }

  //===--------------------------------------------------------------------===//
  // Helper to extract equality from comparison operations
  //===--------------------------------------------------------------------===//

  std::optional<std::pair<Value, Value>> getEqualityFromCmp(Value cmpResult) {
    if (auto cmpOp = cmpResult.getDefiningOp<stablehlo::CompareOp>()) {
      if (cmpOp.getComparisonDirection() == stablehlo::ComparisonDirection::EQ)
        return std::make_pair(cmpOp.getLhs(), cmpOp.getRhs());
    }

    if (auto cmpOp = cmpResult.getDefiningOp<arith::CmpIOp>()) {
      if (cmpOp.getPredicate() == arith::CmpIPredicate::eq)
        return std::make_pair(cmpOp.getLhs(), cmpOp.getRhs());
    }

    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // Main dispatch
  //===--------------------------------------------------------------------===//

  void processOperation(Operation &op) {
    llvm::TypeSwitch<Operation *>(&op)
        .Case<func::CallOp, stablehlo::GetDimensionSizeOp, tensor::DimOp,
              stablehlo::CustomCallOp, cf::AssertOp, tensor::ExtractOp,
              tensor::FromElementsOp, plan::WithShapeOp, stablehlo::ReshapeOp,
              arith::IndexCastOp, arith::IndexCastUIOp>(
            [&](auto typedOp) { visit(typedOp); });
  }

  void collectFromFunction(func::FuncOp func) {
    if (func.isDeclaration())
      return;

    // Only process operations in the entry block (top-level) to avoid
    // operations nested in control-flow regions that may not always execute.
    for (Operation &op : func.getBody().front())
      processOperation(op);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// DimensionRelationshipAnalysis Implementation
//===----------------------------------------------------------------------===//

DimensionRelationshipAnalysis::DimensionRelationshipAnalysis(
    func::FuncOp entrypoint, DimAnalysisOptions options)
    : module(entrypoint->getParentOfType<ModuleOp>()), entrypoint(entrypoint),
      options(std::move(options)),
      solver(std::make_unique<UnionFindConstraintSolver>(exprFactory)) {}

LogicalResult DimensionRelationshipAnalysis::run() {
  collectReachableFunctions();
  buildSymbolMaps();
  return collectConstraints();
}

DimExprRef DimensionRelationshipAnalysis::getExprFromValue(Value v) {
  return exprFactory.getOrCreateValueExpr(v);
}

DimExprRef DimensionRelationshipAnalysis::getExprFromTensorDim(Value tensor,
                                                               int32_t dim) {
  return exprFactory.getOrCreateTensorDimExpr(tensor, dim);
}

DimExprRef DimensionRelationshipAnalysis::getExprFromSymbol(StringRef symbol) {
  return exprFactory.getOrCreateSymbolExpr(symbol);
}

const FunctionSymbolInfo *
DimensionRelationshipAnalysis::getFunctionSymbolInfo(func::FuncOp func) const {
  auto it = functionSymbols.find(func);
  if (it == functionSymbols.end())
    return nullptr;
  return &it->second;
}

void DimensionRelationshipAnalysis::collectReachableFunctions() {
  // BFS from the entrypoint to find all reachable functions
  SmallVector<func::FuncOp> worklist;
  worklist.push_back(entrypoint);
  reachableFunctions.insert(entrypoint);

  while (!worklist.empty()) {
    func::FuncOp func = worklist.pop_back_val();
    if (func.isDeclaration())
      continue;

    // Only look at operations in the entry block (top-level)
    for (Operation &op : func.getBody().front()) {
      if (auto callOp = dyn_cast<func::CallOp>(&op)) {
        if (auto callee =
                module.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
          if (reachableFunctions.insert(callee).second)
            worklist.push_back(callee);
        }
      }
    }
  }
}

void DimensionRelationshipAnalysis::buildSymbolMaps() {
  for (func::FuncOp func : reachableFunctions) {
    FunctionSymbolInfo info;
    info.argToSymbol.resize(func.getNumArguments());

    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      // Check for jax.global_constant attribute
      if (auto symbolAttr = func.getArgAttrOfType<StringAttr>(
              i, options.globalConstantAttrName)) {
        StringRef symbolName = symbolAttr.getValue();
        BlockArgument arg = func.getArgument(i);

        info.symbolToArg[symbolName] = arg;
        info.argToSymbol[i] = symbolName;
      }
    }

    if (!info.symbolToArg.empty())
      functionSymbols[func] = std::move(info);
  }
}

LogicalResult DimensionRelationshipAnalysis::collectConstraints() {
  ConstraintCollector collector(*this);
  for (func::FuncOp func : reachableFunctions)
    collector.collectFromFunction(func);
  return success();
}

//===----------------------------------------------------------------------===//
// Query API
//===----------------------------------------------------------------------===//

bool DimensionRelationshipAnalysis::areEqual(DimExprRef v1, DimExprRef v2) {
  return solver->areEqual(v1, v2);
}

bool DimensionRelationshipAnalysis::areDimensionsEqual(Value tensor1,
                                                       int32_t dim1,
                                                       Value tensor2,
                                                       int32_t dim2) {
  ConstraintCollector collector(*this);
  DimExprRef expr1 = collector.exprFromTensorDim(tensor1, dim1);
  DimExprRef expr2 = collector.exprFromTensorDim(tensor2, dim2);
  return solver->areEqual(expr1, expr2);
}

bool DimensionRelationshipAnalysis::dimensionEqualsSymbol(Value tensor,
                                                          int32_t dim,
                                                          StringRef symbol) {
  ConstraintCollector collector(*this);
  DimExprRef dimExpr = collector.exprFromTensorDim(tensor, dim);
  DimExprRef symExpr = collector.exprFromSymbol(symbol);
  return solver->areEqual(dimExpr, symExpr);
}

std::optional<StringRef>
DimensionRelationshipAnalysis::getSymbolForDimension(Value tensor,
                                                     int32_t dim) {
  ConstraintCollector collector(*this);
  DimExprRef expr = collector.exprFromTensorDim(tensor, dim);
  return solver->getEquivalentSymbol(expr);
}

SmallVector<TensorDim>
DimensionRelationshipAnalysis::getEqualDimensions(Value tensor, int32_t dim) {
  ConstraintCollector collector(*this);
  DimExprRef expr = collector.exprFromTensorDim(tensor, dim);
  auto equivClass = solver->getEquivalenceClass(expr);

  SmallVector<TensorDim> result;
  for (DimExprRef e : equivClass) {
    if (auto *varExpr = dyn_cast<VariableExpr>(e)) {
      if (auto *td = std::get_if<TensorDim>(&varExpr->getVariable())) {
        result.push_back(*td);
      }
    }
  }
  return result;
}

bool DimensionRelationshipAnalysis::areValuesEqual(Value v1, Value v2) {
  ConstraintCollector collector(*this);
  DimExprRef expr1 = collector.exprFromValue(v1);
  DimExprRef expr2 = collector.exprFromValue(v2);
  return solver->areEqual(expr1, expr2);
}

std::optional<StringRef>
DimensionRelationshipAnalysis::getSymbolForValue(Value v) {
  ConstraintCollector collector(*this);
  DimExprRef expr = collector.exprFromValue(v);
  return solver->getEquivalentSymbol(expr);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

void DimensionRelationshipAnalysis::printEquivalenceClasses(
    raw_ostream &os) const {
  // Collect all unique equivalence class representatives
  DenseSet<DimExprRef> visited;
  SmallVector<std::pair<DimExprRef, SmallVector<DimExprRef>>> classes;

  // We need const_cast because getEquivalenceClass modifies internal state
  // (path compression in Union-Find). This is safe because it doesn't change
  // the logical state of the solver.
  auto &mutableSolver = const_cast<ConstraintSolver &>(*solver);

  for (const auto &[value, expr] : exprFactory.getValueExprs()) {
    DimExprRef rep = mutableSolver.getCanonicalRepresentative(expr);
    if (!visited.contains(rep)) {
      visited.insert(rep);
      classes.push_back({rep, mutableSolver.getEquivalenceClass(rep)});
    }
  }

  for (const auto &[key, expr] : exprFactory.getTensorDimExprs()) {
    DimExprRef rep = mutableSolver.getCanonicalRepresentative(expr);
    if (!visited.contains(rep)) {
      visited.insert(rep);
      classes.push_back({rep, mutableSolver.getEquivalenceClass(rep)});
    }
  }

  // Print the equivalence classes
  os << "Dimension Equivalence Classes:\n";
  for (const auto &[rep, members] : classes) {
    if (members.size() <= 1)
      continue;

    os << "  Class [";
    bool first = true;
    for (DimExprRef member : members) {
      if (!first)
        os << " == ";
      first = false;
      os << member->describe();
    }
    os << "]\n";
  }
}
