//===- CUDASimplifyStreamWait.cpp -----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// Implementation of `cuda-simplify-stream-wait`.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::cuda {

#define GEN_PASS_DEF_CUDASIMPLIFYSTREAMWAITPASS
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"

} // namespace mlir::cuda

using namespace mlir;
using namespace mlir::cuda;

namespace {

static void simplifyRedundantStreamWaitEvents(IRRewriter &rewriter,
                                              Operation *op) {
  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "expected isolated operation");

  DominanceInfo domInfo(op);

  llvm::DenseMap<std::pair<Value, Value>,
                 llvm::DenseSet<cuda::StreamWaitEventOp>>
      streamWaitEvents;

  auto getEquivalenceSet = [&](cuda::StreamWaitEventOp waitOp)
      -> llvm::DenseSet<cuda::StreamWaitEventOp> & {
    auto [it, _] =
        streamWaitEvents.try_emplace({waitOp.getStream(), waitOp.getEvent()},
                                     llvm::DenseSet<cuda::StreamWaitEventOp>());
    return it->second;
  };

  op->walk<WalkOrder::PreOrder, ForwardIterator>(
      [&](cuda::StreamWaitEventOp waitOp) {
        auto &equivalentSet = getEquivalenceSet(waitOp);
        for (cuda::StreamWaitEventOp other : equivalentSet) {
          // We already waited on this event, so we can remove the redundant
          // wait.
          if (domInfo.dominates(other, waitOp)) {
            rewriter.eraseOp(waitOp);
            return WalkResult::skip();
          }
        }
        equivalentSet.insert(waitOp);
        return WalkResult::advance();
      });
}

class CUDASimplifyStreamWaitPass
    : public cuda::impl::CUDASimplifyStreamWaitPassBase<
          CUDASimplifyStreamWaitPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    IRRewriter rewriter(func.getContext());
    simplifyRedundantStreamWaitEvents(rewriter, func);
  }
};

} // namespace
