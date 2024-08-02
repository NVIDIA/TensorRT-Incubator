//===- DropNestedModules.cpp ----------------------------------------------===//
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
/// Implementation of `executor-drop-nested-modules` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Transforms.h"

#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
#define GEN_PASS_DEF_DROPNESTEDMODULESPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

void mlir::dropNestedModules(RewriterBase &rewriter, ModuleOp op) {
  for (Operation &op :
       llvm::make_early_inc_range(op.getBody()->getOperations())) {
    if (op.hasTrait<OpTrait::SymbolTable>() &&
        op.hasTrait<OpTrait::IsIsolatedFromAbove>())
      rewriter.eraseOp(&op);
  }
}

namespace {

class DropNestedModulesPass
    : public mlir::impl::DropNestedModulesPassBase<DropNestedModulesPass> {
  void runOnOperation() override {
    ModuleOp op = getOperation();
    IRRewriter rewriter(&getContext());
    dropNestedModules(rewriter, op);
  }
};
} // namespace
