//===- TestBoundsAnalysis.cpp ---------------------------------------------===//
//
// Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Test pass for the PlanDialect shape/value bounds analysis tools.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
void registerTestBoundsAnalysisPass();
}

/// Print out the lattice information for the given value `v`.
template <typename T>
static void printLatticeInfo(llvm::raw_ostream &os, Value v,
                             DataFlowSolver &solver) {
  const T *typeInfo = solver.lookupState<T>(v);
  if (!typeInfo) {
    os << "<<nullptr>>";
    return;
  }
  typeInfo->print(os);
}

/// Print out the lattice information for each operand and result of `op`.
template <typename T>
static void printLatticeInfo(llvm::raw_ostream &os, Operation *op,
                             DataFlowSolver &solver) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    os << " operand #" << index << ": ";
    printLatticeInfo<T>(os, operand, solver);
    os << "\n";
  }
  for (auto [index, value] : llvm::enumerate(op->getResults())) {
    os << " result #" << index << ": ";
    printLatticeInfo<T>(os, value, solver);
    os << "\n";
  }
  for (auto [index, region] : llvm::enumerate(op->getRegions())) {
    os << "  Region #" << index << ":\n";
    for (auto [argIdx, arg] : llvm::enumerate(region.getArguments())) {
      os << "    arg #" << argIdx << ": ";
      printLatticeInfo<T>(os, arg, solver);
      os << "\n";
    }
  }
}

namespace {

struct TestBoundsAnalysisPass
    : public PassWrapper<TestBoundsAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBoundsAnalysisPass)

  StringRef getArgument() const override { return "test-bounds-analysis"; }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<plan::ShapeIntegerRangeAnalysis>();
    solver.load<plan::ShapeBoundsForwardAnalysis>();
    solver.load<plan::ShapeBoundsBackwardsAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::errs();

    op->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto tag = op->getAttrOfType<StringAttr>("tag")) {
        os << "test_tag: " << tag.getValue() << ":\n";
        printLatticeInfo<plan::ShapeBoundsLattice>(os, op, solver);
      }

      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        os << "func " << funcOp.getName() << ":\n";
        for (auto [index, operand] : llvm::enumerate(funcOp.getArguments())) {
          os << " arg #" << index << ": ";
          printLatticeInfo<plan::ShapeBoundsLattice>(os, operand, solver);
          os << "\n";
        }
      }
    });
  }
};

struct TestTensorValueBoundsAnalysisPass
    : public PassWrapper<TestBoundsAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBoundsAnalysisPass)

  StringRef getArgument() const override {
    return "test-tensor-value-bounds-analysis";
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<plan::ShapeIntegerRangeAnalysis>();
    solver.load<plan::TensorValueBoundsAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::errs();
    op->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto tag = op->getAttrOfType<StringAttr>("tag")) {
        os << "test_tag: " << tag.getValue() << ":\n";
        printLatticeInfo<plan::TensorValueBoundsLattice>(os, op, solver);
      }

      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        os << "func " << funcOp.getName() << ":\n";
        for (auto [index, operand] : llvm::enumerate(funcOp.getArguments())) {
          os << " arg #" << index << ": ";
          printLatticeInfo<plan::TensorValueBoundsLattice>(os, operand, solver);
          os << "\n";
        }
      }
    });
  }
};
} // namespace

void mlir::registerTestBoundsAnalysisPass() {
  PassRegistration<TestBoundsAnalysisPass>();
  PassRegistration<TestTensorValueBoundsAnalysisPass>();
}
