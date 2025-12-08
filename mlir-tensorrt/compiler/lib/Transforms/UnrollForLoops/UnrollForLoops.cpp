//===- SCFDetensorizeLoops.cpp --------------------------------------------===//
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
/// Implementation of `scf-ext-unroll-for-loops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "scf-ext-unroll-for-loops"
#define DBGV(fmt, ...)                                                         \
  llvm::dbgs() << llvm::formatv("[" DEBUG_TYPE "] " fmt "\n", __VA_ARGS__)

namespace mtrt {
#define GEN_PASS_DEF_UNROLLFORLOOPSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;
using namespace mlir::scf;

/// Returns the trip count of `forOp` if its' low bound, high bound and step are
/// constants, or optional otherwise. Trip count is computed as
/// ceilDiv(highBound - lowBound, step).
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
  if (!lbCstOp.has_value() || !ubCstOp.has_value() || !stepCstOp.has_value())
    return {};

  // Constant loop bounds computation.
  int64_t lbCst = lbCstOp.value();
  int64_t ubCst = ubCstOp.value();
  int64_t stepCst = stepCstOp.value();
  assert(lbCst >= 0 && ubCst >= 0 && stepCst > 0 &&
         "expected positive loop bounds and step");
  return llvm::divideCeilSigned(ubCst - lbCst, stepCst);
}

/// Estimate the unrolling "cost" in terms of program size.
static uint64_t countOperationsInBody(scf::ForOp op) {
  uint64_t numOps = 0;
  op.getBody()->walk([&](Operation *op) {
    // We don't count terminators or control flow operations (only operations in
    // the bodies).
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<RegionBranchOpInterface>(op))
      return;
    numOps++;
  });
  return numOps;
}

#ifndef NDEBUG
//// For debug logging -- print the operation without regions.
static std::string printWithoutRegions(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);
  op->print(os, OpPrintingFlags().skipRegions());
  return result;
}
#endif

/// Unrolls `op` if its trip count is static and less than `unrollThreshold`.
/// Returns `success()` if the loop is unrolled or ignored, `failure()` if the
/// transformation fails.
static LogicalResult
unrollForLoopWithStaticTripCount(IRRewriter &rewriter, scf::ForOp op,
                                 uint64_t unrollThreshold) {
  std::optional<int64_t> tripCount = getConstantTripCount(op);
  if (!tripCount)
    return success();

  if (*tripCount == 0)
    return success();

  if (*tripCount == 1)
    return op.promoteIfSingleIteration(rewriter);

  uint64_t numOps = countOperationsInBody(op) * *tripCount;
  bool shouldUnroll = numOps <= unrollThreshold;
  LLVM_DEBUG(
      DBGV("{0} unroll loop {1} because it will result in {2} operations "
           "vs. threshold {3}",
           shouldUnroll ? "will" : "will not", printWithoutRegions(op), numOps,
           unrollThreshold));
  if (!shouldUnroll)
    return success();

  if (failed(mlir::loopUnrollByFactor(op, *tripCount)))
    return emitError(op->getLoc()) << "failed to unroll for op";

  return success();
}

namespace {

class UnrollForLoopsPass
    : public mtrt::impl::UnrollForLoopsPassBase<UnrollForLoopsPass> {
public:
  using Base::Base;

  void runOnOperation() override {

    SmallVector<scf::ForOp> forOps;
    // Make sure we walk inner-to-outer in case of nested loops.
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](scf::ForOp forOp) { forOps.push_back(forOp); });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp forOp : forOps) {
      rewriter.setInsertionPoint(forOp);
      // An error is emitted in all failure cases, so we don't need to emit one
      // here.
      if (failed(unrollForLoopWithStaticTripCount(rewriter, forOp,
                                                  unrollThreshold)))
        return signalPassFailure();
    }
  }
};
} // namespace
