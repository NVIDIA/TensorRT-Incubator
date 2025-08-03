//===- ConfirmArgumentDonation.cpp ----------------------------------------===//
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
///  Implementation of the `plan-confirm-argument-donation` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "plan-confirm-argument-donation"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "
#define DBGF(fmt, ...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv(                                    \
                 stderr, "{0}:{1}:{2}(): ", "ConfirmArgumentDonation.cpp",     \
                 __LINE__, __func__);                                          \
             llvm::dbgs() << llvm::formatv(fmt, __VA_ARGS__));

namespace mlir::plan {
#define GEN_PASS_DEF_PLANCONFIRMARGUMENTDONATIONPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Argument donation is denoted by an argument attribute `plan.aliasing_output
/// = N` where N is index of result for which this argument is donated. This
/// function uses `BufferOriginAnalysis` to determine if argument and result it
/// is donated for truly uses same buffer. If same buffer is used, it doesn't do
/// anything. If same buffer is not used and `failOnDonationArgumentRejection`
/// flag is set, it returns failure, resulting in pass failure.
static LogicalResult
confirmArgumentDonation(func::FuncOp func,
                        bool failOnDonationArgumentRejection) {
  if (func.isDeclaration())
    return success();
  Block *entryBlock = &func.getBody().front();
  auto termOp = dyn_cast<func::ReturnOp>(entryBlock->getTerminator());
  if (!termOp)
    return success();
  BufferOriginAnalysis analysis(func);

  // Argument is donated if it has `plan.aliasing_output` argument.
  auto isArgumentDonated = [&](BlockArgument blockArgument) {
    return func.getArgAttrOfType<IntegerAttr>(
        blockArgument.getArgNumber(), plan::PlanDialect::kDonationArgAttrName);
  };

  // Donation is said to be accepted, if for an argument with attribute
  // `plan.aliasing_output = N`, if Nth result bufferizes to the same buffer as
  // that of argument.
  auto isDonationAccepted = [&](BlockArgument blockArgument) -> bool {
    auto donatedResultIdx = func.getArgAttrOfType<IntegerAttr>(
        blockArgument.getArgNumber(), plan::PlanDialect::kDonationArgAttrName);
    Value returnedVal = termOp->getOperand(donatedResultIdx.getInt());

    // Use BufferOriginAnalysis to determine if the returned value originates
    // from the same buffer as the block argument.
    std::optional<bool> result =
        analysis.isSameAllocation(blockArgument, returnedVal);

    // If we can't decide whether buffers are same, return failure
    // conservatively which says donation is not accepted.
    return result.has_value() && *result;
  };

  // Remove `plan.aliasing_output` argument attribute.
  auto markDonationRejected = [&](int64_t argIdx) {
    func.removeArgAttr(argIdx, plan::PlanDialect::kDonationArgAttrName);
  };

  for (BlockArgument arg : entryBlock->getArguments()) {
    if (!isArgumentDonated(arg))
      continue;
    if (isDonationAccepted(arg)) {
      continue;
    } else {
      if (failOnDonationArgumentRejection)
        return failure();
      markDonationRejected(arg.getArgNumber());
    }
  }
  return success();
}

namespace {
class ConfirmArgumentDonationPass
    : public plan::impl::PlanConfirmArgumentDonationPassBase<
          ConfirmArgumentDonationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    if (failed(confirmArgumentDonation(op, failOnDonationArgumentRejection))) {
      emitError(op.getLoc())
          << "failed to confirm argument donation for " << op.getName()
          << " with function type " << op.getFunctionType() << " .";
      return signalPassFailure();
    }
  }
};
} // namespace
