//===- TestCUDASideEffects.cpp --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// A test pass for verifying memory side effects on CUDA operations.
///
//===----------------------------------------------------------------------===//
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::cuda {
void registerTestCUDASideEffectsPass();
}

namespace {
class TestCUDASideEffectsPass
    : public PassWrapper<TestCUDASideEffectsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCUDASideEffectsPass)

  StringRef getArgument() const override { return "cuda-test-side-effects"; }
  StringRef getDescription() const override {
    return "Test CUDA side effects interfaces";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk operations detecting side effects.
    SmallVector<MemoryEffects::EffectInstance, 8> effects;
    module.walk([&](MemoryEffectOpInterface op) {
      effects.clear();
      op.getEffects(effects);

      if (op->hasTrait<OpTrait::IsTerminator>())
        return;

      // Check to see if this operation has any memory effects.
      if (effects.empty()) {
        op.emitRemark() << "operation has no memory effects";
        return;
      }

      for (const MemoryEffects::EffectInstance &instance : effects) {
        auto diag = op.emitRemark() << "found an instance of ";

        if (isa<MemoryEffects::Allocate>(instance.getEffect()))
          diag << "'allocate'";
        else if (isa<MemoryEffects::Free>(instance.getEffect()))
          diag << "'free'";
        else if (isa<MemoryEffects::Read>(instance.getEffect()))
          diag << "'read'";
        else if (isa<MemoryEffects::Write>(instance.getEffect()))
          diag << "'write'";

        if (instance.getValue()) {
          if (OpOperand *operand = instance.getEffectValue<OpOperand *>())
            diag << " on operand #" << operand->getOperandNumber() << ",";
          else if (OpResult result = instance.getEffectValue<OpResult>())
            diag << " on result #" << result.getResultNumber() << ",";
          else if (BlockArgument blockArgument =
                       instance.getEffectValue<BlockArgument>())
            diag << " on block argument #" << blockArgument.getArgNumber()
                 << ",";
        } else if (SymbolRefAttr symbolRef = instance.getSymbolRef()) {
          diag << " on a symbol '" << symbolRef << "',";
        }

        diag << " on resource '" << instance.getResource()->getName() << "'";
      }
    });
  }
};
} // namespace

void cuda::registerTestCUDASideEffectsPass() {
  PassRegistration<TestCUDASideEffectsPass>();
}
