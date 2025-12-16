//===- TestDimAnalysis.cpp ------------------------------------------------===//
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
/// Test pass for the PlanDialect dimension relationship analysis.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::plan;

namespace mlir {
namespace plan {
void registerTestDimAnalysisPass();
} // namespace plan
} // namespace mlir

namespace {

struct TestDimAnalysisPass
    : public PassWrapper<TestDimAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDimAnalysisPass)

  StringRef getArgument() const override { return "test-dim-analysis"; }
  StringRef getDescription() const override {
    return "Test pass for dimension relationship analysis";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    raw_ostream &os = llvm::errs();

    // Find public functions to use as entrypoints
    SmallVector<func::FuncOp> entrypoints;
    module.walk([&](func::FuncOp func) {
      if (func.isPublic() && !func.isDeclaration())
        entrypoints.push_back(func);
    });

    for (func::FuncOp entrypoint : entrypoints) {
      os << "=== Analysis for entrypoint: " << entrypoint.getName() << " ===\n";

      DimAnalysisOptions options{};
      DimensionRelationshipAnalysis analysis(entrypoint, options);

      if (failed(analysis.run())) {
        emitError(entrypoint.getLoc())
            << "Failed to run dimension relationship analysis";
        return signalPassFailure();
      }

      // Print the equivalence classes
      analysis.printEquivalenceClasses(os);

      // Print information about operations tagged with "test_tag"
      entrypoint.walk([&](Operation *op) {
        auto tag = op->getAttrOfType<StringAttr>("test_tag");
        if (!tag)
          return;

        os << "Operation: " << tag.getValue() << "\n";

        // For get_dimension_size ops, print the symbol if known
        if (auto dimSizeOp = dyn_cast<stablehlo::GetDimensionSizeOp>(op)) {
          Value result = dimSizeOp.getResult();
          if (auto symbol = analysis.getSymbolForValue(result)) {
            os << "  Result symbol: " << *symbol << "\n";
          } else {
            os << "  Result symbol: <unknown>\n";
          }
        }

        // For func.call ops, print mappings
        if (auto callOp = dyn_cast<func::CallOp>(op)) {
          for (auto [idx, operand] : llvm::enumerate(callOp.getOperands())) {
            if (auto symbol = analysis.getSymbolForValue(operand)) {
              os << "  Operand " << idx << " -> symbol: " << *symbol << "\n";
            }
          }
        }
      });

      // Print function argument symbol mappings for reachable functions
      os << "Symbol mappings:\n";
      module.walk([&](func::FuncOp func) {
        bool hasSymbols = false;
        for (unsigned i = 0; i < func.getNumArguments(); ++i) {
          if (auto symbolAttr =
                  func.getArgAttrOfType<StringAttr>(i, "jax.global_constant")) {
            if (!hasSymbols) {
              os << "  Function: " << func.getName() << "\n";
              hasSymbols = true;
            }
            os << "    Arg " << i << " is symbol: " << symbolAttr.getValue()
               << "\n";
          }
        }
      });
    }
  }
};

} // namespace

void mlir::plan::registerTestDimAnalysisPass() {
  PassRegistration<TestDimAnalysisPass>();
}
