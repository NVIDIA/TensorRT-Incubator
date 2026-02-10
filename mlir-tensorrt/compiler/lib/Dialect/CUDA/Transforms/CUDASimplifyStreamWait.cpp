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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cuda-simplify-stream-wait"

namespace mlir::cuda {

#define GEN_PASS_DEF_CUDASIMPLIFYSTREAMWAITPASS
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"

} // namespace mlir::cuda

using namespace mlir;
using namespace mlir::cuda;

namespace {

static void simplifyDuplicateStreamWaitEvents(IRRewriter &rewriter,
                                              Operation *op) {
  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "expected isolated operation");

  LLVM_DEBUG(
      llvm::dbgs() << "=== Simplifying duplicate stream wait events ===\n");

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
        LLVM_DEBUG(llvm::dbgs() << "  Checking wait op: " << waitOp << "\n");
        LLVM_DEBUG(llvm::dbgs() << "    Stream: " << waitOp.getStream()
                                << ", Event: " << waitOp.getEvent() << "\n");
        auto &equivalentSet = getEquivalenceSet(waitOp);
        for (cuda::StreamWaitEventOp other : equivalentSet) {
          // We already waited on this event, so we can remove the redundant
          // wait.
          if (domInfo.dominates(other, waitOp)) {
            LLVM_DEBUG(llvm::dbgs() << "    Found dominating wait, erasing: "
                                    << waitOp << "\n");
            rewriter.eraseOp(waitOp);
            return WalkResult::skip();
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "    Adding to equivalence set\n");
        equivalentSet.insert(waitOp);
        return WalkResult::advance();
      });
}

static Operation *getUniqueRecordOp(Value event, Block &block) {
  Operation *recordOp = event.getDefiningOp<cuda::EventCreateOnStreamOp>();
  for (OpOperand &use : event.getUses()) {
    auto record = dyn_cast<cuda::StreamRecordEventOp>(use.getOwner());
    if (!record)
      continue;
    if (recordOp && recordOp != record)
      return nullptr;
    recordOp = record;
  }

  if (!recordOp || recordOp->getBlock() != &block)
    return nullptr;

  return recordOp;
}

static Value getStreamFromRecordOp(Operation *recordOp) {
  if (auto record = dyn_cast<cuda::StreamRecordEventOp>(recordOp))
    return record.getStream();
  if (auto create = dyn_cast<cuda::EventCreateOnStreamOp>(recordOp))
    return create.getStream();
  llvm_unreachable("expected stream record or create on stream operation");
}

static bool collectEventUsesForElimination(
    cuda::StreamWaitEventOp waitOp, Block &block, Operation *&recordOp,
    llvm::SmallVectorImpl<cuda::EventReleaseOp> &releaseOps) {
  Value event = waitOp.getEvent();
  recordOp = event.getDefiningOp<cuda::EventCreateOnStreamOp>();
  releaseOps.clear();

  LLVM_DEBUG(llvm::dbgs() << "      Collecting event uses for elimination\n");

  for (OpOperand &use : event.getUses()) {
    Operation *user = use.getOwner();
    if (auto record = dyn_cast<cuda::StreamRecordEventOp>(user)) {
      if (recordOp && recordOp != record) {
        LLVM_DEBUG(llvm::dbgs()
                   << "        Multiple different record ops found\n");
        return false;
      }
      recordOp = record;
      continue;
    }
    if (auto wait = dyn_cast<cuda::StreamWaitEventOp>(user)) {
      if (wait != waitOp) {
        LLVM_DEBUG(llvm::dbgs() << "        Event used by other wait op\n");
        return false;
      }
      continue;
    }
    if (auto release = dyn_cast<cuda::EventReleaseOp>(user)) {
      releaseOps.push_back(release);
      continue;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "        Event has unexpected use: " << *user << "\n");
    return false;
  }

  if (!recordOp || recordOp->getBlock() != &block) {
    LLVM_DEBUG(llvm::dbgs() << "        No record op or wrong block\n");
    return false;
  }

  if (releaseOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "        Multiple release ops found\n");
    return false;
  }

  for (cuda::EventReleaseOp release : releaseOps) {
    if (release->getBlock() != &block) {
      LLVM_DEBUG(llvm::dbgs() << "        Release op in wrong block\n");
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "        Event uses collected successfully\n");
  return true;
}

static Operation *findNextUseInBlock(Operation *start, Value value) {
  for (Operation *it = start->getNextNode(); it; it = it->getNextNode()) {
    for (Value operand : it->getOperands()) {
      if (operand == value)
        return it;
    }
  }
  return nullptr;
}

static void simplifyOrderedEventWaits(IRRewriter &rewriter, Operation *op) {
  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "expected isolated operation");

  LLVM_DEBUG(llvm::dbgs() << "=== Simplifying ordered event waits ===\n");

  llvm::SmallDenseSet<Operation *, 8> eraseSet;
  llvm::SmallVector<Operation *, 8> eraseList;
  llvm::SmallVector<Value, 8> eventsToCleanup;

  auto markForErase = [&](Operation *op) {
    if (eraseSet.insert(op).second)
      eraseList.push_back(op);
  };

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation *cur = block.empty() ? nullptr : &block.front(); cur;) {
        auto wait1 = dyn_cast<cuda::StreamWaitEventOp>(cur);
        if (!wait1) {
          cur = cur->getNextNode();
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "  Found wait1: " << *wait1 << "\n");

        Value waitStream = wait1.getStream();
        Operation *nextUse = findNextUseInBlock(wait1, waitStream);
        if (!nextUse) {
          LLVM_DEBUG(llvm::dbgs() << "    No next use of stream found\n");
          cur = cur->getNextNode();
          continue;
        }

        auto wait2 = dyn_cast<cuda::StreamWaitEventOp>(nextUse);
        if (!wait2 || wait2.getStream() != waitStream ||
            wait1.getEvent() == wait2.getEvent()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    Next use is not a compatible wait2\n");
          cur = cur->getNextNode();
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "  Found wait2: " << *wait2 << "\n");

        Operation *record1;
        llvm::SmallVector<cuda::EventReleaseOp, 1> releaseOps;
        if (!collectEventUsesForElimination(wait1, block, record1,
                                            releaseOps)) {
          LLVM_DEBUG(llvm::dbgs() << "    Cannot eliminate wait1 event\n");
          cur = cur->getNextNode();
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "    Event1 can be eliminated, record1: "
                                << *record1 << "\n");

        auto record2 = getUniqueRecordOp(wait2.getEvent(), block);
        if (!record2 ||
            getStreamFromRecordOp(record1) != getStreamFromRecordOp(record2) ||
            !record1->isBeforeInBlock(record2)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    record2 invalid or streams mismatch or "
                        "incorrect order\n");
          cur = cur->getNextNode();
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "    record2: " << *record2 << "\n");
        LLVM_DEBUG(llvm::dbgs()
                   << "    Conditions met! Marking for elimination\n");

        LLVM_DEBUG(llvm::dbgs()
                   << "    Marking wait1 for erase: " << *wait1 << "\n");
        markForErase(wait1.getOperation());
        for (cuda::EventReleaseOp release : releaseOps) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    Marking release for erase: " << release << "\n");
          markForErase(release.getOperation());
        }
        // Only enqueue events whose defining op is not being erased.
        // EventCreateOnStreamOp defines the event result; erasing it would
        // invalidate the Value and make later cleanup a use-after-free.
        if (!isa<cuda::EventCreateOnStreamOp>(record1))
          eventsToCleanup.push_back(wait1.getEvent());
        LLVM_DEBUG(llvm::dbgs()
                   << "    Marking record1 for erase: " << *record1 << "\n");
        markForErase(record1);
        cur = wait2;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Erasing " << eraseList.size()
                          << " operations\n");
  for (Operation *opToErase : eraseList) {
    LLVM_DEBUG(llvm::dbgs() << "    Erasing: " << *opToErase << "\n");
    rewriter.eraseOp(opToErase);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Cleaning up " << eventsToCleanup.size()
                          << " events\n");
  for (Value event : eventsToCleanup) {
    if (!event.use_empty()) {
      LLVM_DEBUG(llvm::dbgs() << "    Event still has uses, skipping\n");
      continue;
    }
    if (auto createOp = event.getDefiningOp<cuda::EventCreateOp>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Erasing unused EventCreateOp: " << *createOp << "\n");
      rewriter.eraseOp(createOp);
    } else if (auto recordOp =
                   event.getDefiningOp<cuda::EventCreateOnStreamOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "    Erasing unused EventCreateOnStreamOp: "
                              << *recordOp << "\n");
      rewriter.eraseOp(recordOp);
    }
  }
}

static void simplifyRedundantStreamWaitEvents(IRRewriter &rewriter,
                                              Operation *op) {
  simplifyDuplicateStreamWaitEvents(rewriter, op);
  simplifyOrderedEventWaits(rewriter, op);
}

class CUDASimplifyStreamWaitPass
    : public cuda::impl::CUDASimplifyStreamWaitPassBase<
          CUDASimplifyStreamWaitPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    LLVM_DEBUG(llvm::dbgs()
               << "\n*** Running CUDASimplifyStreamWait on function: "
               << func.getName() << " ***\n");

    IRRewriter rewriter(func.getContext());
    simplifyRedundantStreamWaitEvents(rewriter, func);

    LLVM_DEBUG(llvm::dbgs() << "*** Finished CUDASimplifyStreamWait ***\n\n");
  }
};

} // namespace
