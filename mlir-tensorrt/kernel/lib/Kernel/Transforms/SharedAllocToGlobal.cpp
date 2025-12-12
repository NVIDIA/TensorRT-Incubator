//===- SharedAllocToGlobal.cpp --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of `kernel-shared-alloc-to-global` pass that lowers
/// `memref.alloc` representing shared memory to `memref.global`.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_SHAREDALLOCTOGLOBALPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

/// Get the `op`'s alignemnt value or return the default.
static uint64_t getAlignmentOr(memref::AllocOp op, uint64_t defaultValue) {
  if (std::optional<uint64_t> allocAlign = op.getAlignment())
    return *allocAlign;
  return defaultValue;
}

/// Rewrite `memref.alloc`
struct SharedMemAllocToGlobalRewriter
    : public OpRewritePattern<memref::AllocOp> {
  SharedMemAllocToGlobalRewriter(MLIRContext *ctx, uint64_t defaultAlignment,
                                 PatternBenefit benefit = PatternBenefit(1))
      : OpRewritePattern(ctx, benefit), defaultAlignment(defaultAlignment) {}

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    // We recognize shared memory based on the address space identifier. We use
    // GPU dialect's attribute for this. Shared memory allocations should always
    // have static shape at this point.
    MemRefType allocType = op.getType();
    if (allocType.getMemorySpace() !=
            gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                       gpu::AddressSpace::Workgroup) ||
        !op.getType().hasStaticShape())
      return failure();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto moduleOp = funcOp->getParentOfType<gpu::GPUModuleOp>();
    if (!moduleOp)
      return failure();

    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp.getLoc(), "__shared_memory__", rewriter.getStringAttr("private"),
        allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/
        rewriter.getI64IntegerAttr(
            static_cast<int64_t>(getAlignmentOr(op, defaultAlignment))));
    symbolTable.insert(global);

    // Place the memref.get_global at the start of the function.
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, global.getType(),
                                                     global.getName());
    return success();
  }

  uint64_t defaultAlignment;
};

namespace {
class SharedAllocToGlobalPass
    : public kernel::impl::SharedAllocToGlobalPassBase<
          SharedAllocToGlobalPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    /// TODO: this pass could also consolidate the shared memory in order to
    /// convert staticly declared shared memory to dynamic shared memory.
    /// TODO: this pass could also do liveness analysis to allow SMEM reuse.
    patterns.add<SharedMemAllocToGlobalRewriter>(ctx, defaultAlignment);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
