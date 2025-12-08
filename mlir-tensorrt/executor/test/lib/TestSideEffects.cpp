//===- TestSideEffects.cpp ------------------------------------------------===//
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
/// A test pass for verifying memory side effects on operations.
/// This differs from the upstream MLIR pass in that it actually prints which
/// operand/result/block argument the effect is on.
///
//===----------------------------------------------------------------------===//
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::executor {
void registerTestSideEffectsPass();
}

namespace {
class TestSideEffectsPass
    : public PassWrapper<TestSideEffectsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSideEffectsPass)

  StringRef getArgument() const override {
    return "executor-test-side-effects";
  }
  StringRef getDescription() const override {
    return "Test executor side effects interfaces";
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

void executor::registerTestSideEffectsPass() {
  PassRegistration<TestSideEffectsPass>();
}
