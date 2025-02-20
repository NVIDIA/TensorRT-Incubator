//===- TestTensorKindAnalysis.cpp -----------------------------------------===//
//
// Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Add tensor kind analysis test pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

/// Print out the lattice information for the given value `v`.
static void printLatticeInfo(llvm::raw_ostream &os, Value v,
                             DataFlowSolver &solver) {
  const TensorKindLattice *typeInfo = solver.lookupState<TensorKindLattice>(v);
  if (!typeInfo) {
    os << "<<nullptr>>";
    return;
  }
  typeInfo->print(os);
}

/// Print out the lattice information for each operand and result of `op`.
static void printLatticeInfo(llvm::raw_ostream &os, Operation *op,
                             DataFlowSolver &solver) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    os << " operand #" << index << ": ";
    printLatticeInfo(os, operand, solver);
    os << "\n";
  }
  for (auto [index, value] : llvm::enumerate(op->getResults())) {
    os << " result #" << index << ": ";
    printLatticeInfo(os, value, solver);
    os << "\n";
  }
  for (auto [index, region] : llvm::enumerate(op->getRegions())) {
    os << "  Region #" << index << ":\n";
    for (auto [argIdx, arg] : llvm::enumerate(region.getArguments())) {
      os << "    arg #" << argIdx << ": ";
      printLatticeInfo(os, arg, solver);
      os << "\n";
    }
  }
}

namespace {
/// A test pass for the TensorKindAnalysis. We declare it directly here
/// (rather than going through a Passes.td) so that we can exclude it from the
/// build in the future if required.
struct TestTensorKindPass
    : public PassWrapper<TestTensorKindPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTensorKindPass)

  StringRef getArgument() const override { return "test-tensor-kind-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::errs();

    // Walk in pre-order for readability. For each operation with the "tag"
    // attribute, print out the tensor kinds of its operands and results. This
    // is the same kind of testing mechanism used by the tests in upstream
    // MLIR analysis test passes.
    op->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto tag = op->getAttrOfType<StringAttr>("tag")) {
        os << "test_tag: " << tag.getValue() << ":\n";
        printLatticeInfo(os, op, solver);
      }

      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        os << "func " << funcOp.getName() << ":\n";
        for (auto [index, operand] : llvm::enumerate(funcOp.getArguments())) {
          os << " arg #" << index << ": ";
          printLatticeInfo(os, operand, solver);
          os << "\n";
        }
      }
    });
  }
};
} // namespace

namespace mlir {
void registerTestTensorKindAnalysisPass() {
  PassRegistration<TestTensorKindPass>();
}
} // namespace mlir
