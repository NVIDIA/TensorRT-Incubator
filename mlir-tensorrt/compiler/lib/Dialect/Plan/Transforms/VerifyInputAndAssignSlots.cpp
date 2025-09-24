//===- VerifyInputAndAssignSlots.cpp --------------------------------------===//
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
///  Implementation of the `plan-verify-input-and-assign-slots` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::plan {
#define GEN_PASS_DEF_VERIFYINPUTANDASSIGNSLOTSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {

struct VerifyInputAndAssignSlotsPass
    : public plan::impl::VerifyInputAndAssignSlotsPassBase<
          VerifyInputAndAssignSlotsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    ModuleOp module = getOperation();
    for (auto func : module.getOps<FunctionOpInterface>())
      assignInitialSlotNumbers(rewriter, func);
  }
};

} // namespace
