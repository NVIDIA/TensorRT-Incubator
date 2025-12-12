//===- TestOutlining.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Clustering transforms test pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Utils/OutliningUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::kernel {
void registerTestOutliningPass();
}

namespace {
class TestOutliningPass
    : public PassWrapper<TestOutliningPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOutliningPass)
  TestOutliningPass() {}
  TestOutliningPass(const TestOutliningPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-outlining"; }
  StringRef getDescription() const final {
    return "Test cluster-and-outline transformation";
  }

  void runOnOperation() override {
    // This pass must run on a module because it creates new functions. We will
    // run the cluster-and-outline on the first function in the module, which is
    // how are tests are setup.
    ModuleOp op = getOperation();

    IRRewriter rewriter(op->getContext());

    rewriter.setInsertionPointToEnd(op.getBody());
    gpu::GPUModuleOp kernelModule =
        rewriter.create<gpu::GPUModuleOp>(op.getLoc(), "kernels",
                                          /*targets=*/ArrayAttr{});
    SymbolTable kernelModuleSymbolTable(kernelModule);

    auto walkResult = op.walk([&](scf::ForallOp op) {
      FailureOr<ForallOutliningResult> forallOutlineResult =
          mlir::outlineForall(
              rewriter, op, "forall", kernelModuleSymbolTable,
              mlir::getInductionVarReplacementsUsingGpuBlockId,
              [&](RewriterBase &rewriter, scf::ForallOp op, ValueRange inputs,
                  func::FuncOp callee) -> Operation * {
                Location loc = op.getLoc();
                SmallVector<OpFoldResult> ub = op.getMixedUpperBound();
                SmallVector<Value> gridShape =
                    mlir::getValueOrCreateConstantIndexOp(rewriter, loc, ub);
                SmallVector<Value> blockShape = {
                    rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getIndexAttr(1))};
                unsigned numOutputs = op.getOutputs().size();
                return rewriter.create<kernel::CallOp>(
                    loc, op.getResultTypes(), gridShape, blockShape,
                    inputs.drop_back(numOutputs), inputs.take_back(numOutputs),
                    SymbolRefAttr::get(kernelModule.getSymNameAttr(),
                                       {SymbolRefAttr::get(callee)}));
              },
              [](Operation *op) -> bool {
                return op->hasTrait<OpTrait::ConstantLike>();
              });
      if (failed(forallOutlineResult)) {
        op.emitOpError() << "failed to outline forall op";
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, tensor::TensorDialect, func::FuncDialect,
                    kernel::KernelDialect, gpu::GPUDialect,
                    mlir::bufferization::BufferizationDialect>();
  }
};
} // namespace

void kernel::registerTestOutliningPass() {
  PassRegistration<TestOutliningPass>();
}
