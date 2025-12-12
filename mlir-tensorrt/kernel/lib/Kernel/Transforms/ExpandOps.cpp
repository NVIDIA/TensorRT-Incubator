//===- ExpandOps.cpp ------------------------------------------------------===//
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
/// Definition of the `kernel-expand-ops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::kernel {
#define GEN_PASS_DEF_KERNELEXPANDOPSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

namespace {

class KernelExpandOpsPass
    : public kernel::impl::KernelExpandOpsPassBase<KernelExpandOpsPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();
    IRRewriter rewriter(ctx);

    SmallVector<CombinerOp> combinerOps;
    op->walk([&](CombinerOp combinerOp) { combinerOps.push_back(combinerOp); });

    for (CombinerOp combinerOp : combinerOps) {
      replaceOpWithRegion(rewriter, combinerOp, combinerOp.getBody(),
                          combinerOp.getInputs());
    }
  }
};
} // namespace
