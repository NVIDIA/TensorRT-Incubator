//===- CUDAInsertHostSync.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// Implementation of `cuda-insert-host-sync` using MLIR's Data Flow Analysis
/// framework for principled handling of control flow.
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt-common/Interfaces/StreamSchedulableOpInterface.h"
#include "mlir-tensorrt/Analysis/AliasAnalysis.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"    // IWYU pragma: keep
#include "llvm/ADT/SetVector.h"   // IWYU pragma: keep
#include "llvm/ADT/SmallPtrSet.h" // IWYU pragma: keep
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cuda-insert-host-sync"

namespace mlir::cuda {
#define GEN_PASS_DEF_CUDAINSERTHOSTSYNCPASS
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"
} // namespace mlir::cuda

using namespace mlir;
using namespace mlir::cuda;

//===----------------------------------------------------------------------===//
// PendingD2HAccess - Tracks which D2H ops may have written to a memref
//===----------------------------------------------------------------------===//

namespace {

/// Represents the set of D2H copy operations that may have last written to
/// a memory location. We track both the D2H operation and its stream so we
/// can insert appropriate synchronization.
class PendingD2HAccess {
public:
  using D2HOpSet = llvm::SmallSetVector<CopyD2HOp, 4>;

  ArrayRef<CopyD2HOp> get() const { return ops.getArrayRef(); }
  bool empty() const { return ops.empty(); }
  bool isKnown() const { return !unknown; }

  /// Merge another PendingD2HAccess into this one.
  ChangeResult merge(const PendingD2HAccess &other) {
    if (unknown)
      return ChangeResult::NoChange;
    if (other.unknown) {
      unknown = true;
      ops.clear();
      return ChangeResult::Change;
    }
    bool changed = ops.set_union(other.ops);
    return !changed ? ChangeResult::NoChange : ChangeResult::Change;
  }

  /// Set the pending D2H ops to a single operation.
  ChangeResult set(CopyD2HOp op) {
    if (!unknown && ops.size() == 1 && *ops.begin() == op)
      return ChangeResult::NoChange;
    unknown = false;
    ops.clear();
    ops.insert(op);
    return ChangeResult::Change;
  }

  /// Clear all pending D2H ops (e.g., after a host-side write).
  ChangeResult clear() {
    if (!unknown && ops.empty())
      return ChangeResult::NoChange;
    unknown = false;
    ops.clear();
    return ChangeResult::Change;
  }

  /// Mark as unknown (pessimistic fixpoint).
  ChangeResult setUnknown() {
    if (unknown)
      return ChangeResult::NoChange;
    ops.clear();
    unknown = true;
    return ChangeResult::Change;
  }

  bool operator==(const PendingD2HAccess &other) const {
    return unknown == other.unknown && ops == other.ops;
  }

private:
  bool unknown = false;
  D2HOpSet ops;
};

//===----------------------------------------------------------------------===//
// PendingD2HLattice - Dense lattice tracking D2H writes per memref
//===----------------------------------------------------------------------===//

/// Dense lattice that maps each memref value to the set of D2H operations
/// that may have last written to it.
class PendingD2HLattice : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PendingD2HLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  /// Join with another lattice (union of pending D2H ops per memref).
  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const PendingD2HLattice &>(rhs);
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &[memref, access] : other.pendingAccesses) {
      result |= pendingAccesses[memref].merge(access);
    }
    return result;
  }

  /// Reset to entry state (no pending D2H ops).
  ChangeResult reset() {
    if (pendingAccesses.empty())
      return ChangeResult::NoChange;
    pendingAccesses.clear();
    return ChangeResult::Change;
  }

  /// Record that a D2H operation wrote to a memref.
  ChangeResult setD2HWrite(Value memref, CopyD2HOp d2hOp) {
    return pendingAccesses[memref].set(d2hOp);
  }

  /// Clear pending D2H info for a memref (e.g., after host-side sync or write).
  ChangeResult clearMemref(Value memref) {
    auto it = pendingAccesses.find(memref);
    if (it == pendingAccesses.end())
      return ChangeResult::NoChange;
    ChangeResult result = it->second.clear();
    pendingAccesses.erase(it);
    return result;
  }

  /// Get pending D2H accesses for a memref. Returns nullptr if none.
  const PendingD2HAccess *getPendingAccess(Value memref) const {
    auto it = pendingAccesses.find(memref);
    if (it == pendingAccesses.end())
      return nullptr;
    return &it->second;
  }

  /// Get all memrefs with pending D2H accesses.
  const DenseMap<Value, PendingD2HAccess> &getAllPendingAccesses() const {
    return pendingAccesses;
  }

  void print(raw_ostream &os) const override {
    os << "PendingD2HLattice:\n";
    for (const auto &[memref, access] : pendingAccesses) {
      os << "  " << memref << ": ";
      if (!access.isKnown()) {
        os << "<unknown>\n";
        continue;
      }
      os << "[";
      llvm::interleaveComma(access.get(), os,
                            [&](CopyD2HOp op) { os << op->getLoc(); });
      os << "]\n";
    }
  }

private:
  DenseMap<Value, PendingD2HAccess> pendingAccesses;
};

//===----------------------------------------------------------------------===//
// PendingD2HAnalysis - Forward DFA tracking D2H writes
//===----------------------------------------------------------------------===//

/// Forward dense data-flow analysis that tracks which D2H copy operations
/// may have last written to each memref at each program point.
class PendingD2HAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<PendingD2HLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, const PendingD2HLattice &before,
                               PendingD2HLattice *after) override {
    LLVM_DEBUG({
      llvm::dbgs() << "[PendingD2HAnalysis] visitOperation: " << op->getName()
                   << " at " << op->getLoc() << "\n";
      llvm::dbgs() << "[PendingD2HAnalysis]   before: ";
      before.print(llvm::dbgs());
    });

    // Start by propagating the incoming state.
    ChangeResult result = after->join(before);

    // Handle D2H copy operations: record them as pending writes.
    if (auto d2hOp = dyn_cast<CopyD2HOp>(op)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[PendingD2HAnalysis] D2H write to "
                     << d2hOp.getTarget() << " at " << op->getLoc() << "\n";
      });
      result |= after->setD2HWrite(d2hOp.getTarget(), d2hOp);
      propagateIfChanged(after, result);
      LLVM_DEBUG({
        llvm::dbgs() << "[PendingD2HAnalysis]   after: ";
        after->print(llvm::dbgs());
      });
      return success();
    }

    // Skip other CUDA dialect ops (they don't affect host-visible state).
    if (isa<CUDADialect>(op->getDialect())) {
      propagateIfChanged(after, result);
      return success();
    }

    // For non-CUDA ops with memory write effects on memrefs, clear the
    // pending D2H state for those memrefs (the host has now written to them).
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    if (memEffects) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memEffects.getEffects(effects);
      for (const auto &effect : effects) {
        Value affectedValue = effect.getValue();
        if (!affectedValue || !isa<MemRefType>(affectedValue.getType()))
          continue;

        // If it's a write effect, clear pending D2H for that memref.
        if (isa<MemoryEffects::Write>(effect.getEffect())) {
          LLVM_DEBUG({
            llvm::dbgs() << "[PendingD2HAnalysis] Host write to "
                         << affectedValue << " clears pending D2H\n";
          });
          result |= after->clearMemref(affectedValue);
        }
      }
    }

    propagateIfChanged(after, result);
    return success();
  }

  /// At entry points, there are no pending D2H operations.
  void setToEntryState(PendingD2HLattice *lattice) override {
    LLVM_DEBUG(
        { llvm::dbgs() << "[PendingD2HAnalysis] setToEntryState called\n"; });
    propagateIfChanged(lattice, lattice->reset());
  }

  /// Override to trace region branch control flow.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const PendingD2HLattice &before,
                                            PendingD2HLattice *after) override {
    LLVM_DEBUG({
      llvm::dbgs()
          << "[PendingD2HAnalysis] visitRegionBranchControlFlowTransfer: "
          << branch->getName() << " at " << branch->getLoc() << " from region "
          << (regionFrom ? std::to_string(*regionFrom) : "parent")
          << " to region " << (regionTo ? std::to_string(*regionTo) : "parent")
          << "\n";
      llvm::dbgs() << "[PendingD2HAnalysis]   before: ";
      before.print(llvm::dbgs());
    });

    // Use default behavior: join the states.
    DenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after);

    LLVM_DEBUG({
      llvm::dbgs() << "[PendingD2HAnalysis]   after: ";
      after->print(llvm::dbgs());
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Helper functions for synchronization insertion
//===----------------------------------------------------------------------===//

static bool isStreamSchedulableOp(Operation *op) {
  if (auto iface = dyn_cast<mtrt::compiler::StreamSchedulableOp>(op))
    return iface.getStreamOperand() != nullptr;
  return false;
}

/// Check if an operation performs a host-side access (read or write) of a
/// memref. Both reads and writes require synchronization with pending D2H
/// because the async copy must complete before the host can safely access
/// the memory.
static bool hasHostMemRefAccess(Operation *op,
                                SmallVectorImpl<Value> &accessedMemrefs) {
  // Check for pure ops first.
  if (isPure(op))
    return false;

  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects)
    return false;

  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffects.getEffects(effects);
  for (const auto &effect : effects) {
    Value affectedValue = effect.getValue();
    if (!affectedValue || !isa<MemRefType>(affectedValue.getType()))
      continue;

    // Both host reads and writes require synchronization with pending D2H.
    if (isa<MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Free>(
            effect.getEffect()))
      accessedMemrefs.push_back(affectedValue);
  }

  return !accessedMemrefs.empty();
}

/// Collect all D2H operations that need synchronization before accessing
/// the given memrefs, using alias analysis to handle subviews/casts.
static void
collectD2HOpsNeedingSync(const PendingD2HLattice *lattice,
                         ArrayRef<Value> readMemrefs, AliasAnalysis &aa,
                         llvm::SmallSetVector<CopyD2HOp, 4> &d2hOpsToSync) {
  if (!lattice)
    return;

  for (Value readMemref : readMemrefs) {
    for (const auto &[pendingMemref, pendingAccess] :
         lattice->getAllPendingAccesses()) {
      if (!pendingAccess.isKnown())
        continue;

      // Check if the read memref may alias the pending memref.
      if (aa.alias(readMemref, pendingMemref) != AliasResult::NoAlias) {
        for (CopyD2HOp d2hOp : pendingAccess.get())
          d2hOpsToSync.insert(d2hOp);
      }
    }
  }
}

/// Check if a Value is in scope (dominates) at a given operation.
/// For block arguments, checks if the argument's block dominates the op's
/// block. For op results, checks if the defining op properly dominates the
/// target op.
static bool valueInScopeAt(Value value, Operation *op, DominanceInfo &domInfo) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    // Block arguments dominate all ops in their block and nested regions
    Block *argBlock = blockArg.getOwner();
    Block *opBlock = op->getBlock();
    return domInfo.dominates(argBlock, opBlock);
  }
  // For op results, the defining op must properly dominate
  Operation *defOp = value.getDefiningOp();
  return defOp && domInfo.properlyDominates(defOp, op);
}

//===----------------------------------------------------------------------===//
// CUDAInsertHostSyncPass
//===----------------------------------------------------------------------===//
namespace {

class CUDAInsertHostSyncPass
    : public cuda::impl::CUDAInsertHostSyncPassBase<CUDAInsertHostSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "[cuda-insert-host-sync] running on module\n");

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;

      if (failed(processFunction(func)))
        return signalPassFailure();
    }
  }

  LogicalResult processFunction(func::FuncOp func) {
    LLVM_DEBUG({
      llvm::dbgs() << "[cuda-insert-host-sync] processing func @"
                   << func.getName() << "\n";
    });

    // Phase 1: Run the data-flow analysis to compute pending D2H state at
    // each program point.
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    solver.load<PendingD2HAnalysis>();

    if (failed(solver.initializeAndRun(func))) {
      return func.emitError() << "data-flow analysis failed";
    }

    // Phase 2: Use the analysis results to insert synchronization.
    AliasAnalysis aa = createRestrictAwareAliasAnalysis(func);
    IRRewriter rewriter(func.getContext());

    // Track which D2H ops have had events created for them and their events.
    DenseMap<Operation *, Value> d2hToEvent;
    Value device{};

    auto lazilyCreateDevice = [&]() -> Value {
      if (device)
        return device;
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&func.getBody().front());
      Value zeroI32 = rewriter.create<arith::ConstantOp>(
          func.getLoc(), rewriter.getI32IntegerAttr(0));
      device = rewriter.create<GetProgramDeviceOp>(func.getLoc(), zeroI32);
      return device;
    };

    // Track created events and their defining blocks for later release.
    // Events are released at the end of their defining block (before
    // terminator).
    SmallVector<std::pair<Value, Block *>> createdEvents;

    // Create events for D2H ops in the same scope as the sync point.
    auto createEventForD2H = [&](CopyD2HOp d2hOp) -> Value {
      auto it = d2hToEvent.find(d2hOp.getOperation());
      if (it != d2hToEvent.end())
        return it->second;

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(d2hOp);
      Location loc = d2hOp.getLoc();
      Value event = rewriter.create<EventCreateOp>(loc, lazilyCreateDevice());
      rewriter.create<StreamRecordEventOp>(loc, d2hOp.getStream(), event);
      d2hToEvent[d2hOp.getOperation()] = event;

      // Track for release at end of defining block
      createdEvents.push_back({event, d2hOp->getBlock()});

      LLVM_DEBUG({
        llvm::dbgs() << "[cuda-insert-host-sync] created event for D2H at "
                     << loc << "\n";
      });
      return event;
    };

    // Function-scope tracking: maps D2H op -> block where it was synced.
    // Used to skip redundant syncs when the previous sync dominates current.
    DenseMap<Operation *, Block *> d2hSyncedInBlock;
    // Track D2H ops synced immediately after (tier 3).
    DenseSet<Operation *> immediatelySyncedD2HOps;
    // Per-block stream sync tracking (streams can be safely synced multiple
    // times, but we avoid redundant syncs within a block).
    DenseSet<Value> syncedStreamsInBlock;
    // Track the current block to detect block transitions for clearing
    // per-block caches.
    Block *currentBlock = nullptr;

    DominanceInfo domInfo(func);

    // Helper to insert synchronization for D2H ops using a 3-tier approach:
    // 1. Event sync: If D2H properly dominates sync point
    // 2. Stream sync at use: If D2H doesn't dominate but stream is in scope
    // 3. Immediate sync: If neither D2H nor stream dominates, sync after D2H
    auto insertSyncs =
        [&](Operation *beforeOp, ArrayRef<CopyD2HOp> d2hOpsToSync) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPoint(beforeOp);

          for (CopyD2HOp d2hOp : d2hOpsToSync) {
            // Skip if this D2H was already synced immediately after it
            if (immediatelySyncedD2HOps.contains(d2hOp.getOperation()))
              continue;

            // Tier 1: If D2H properly dominates, use event-based sync (optimal)
            if (domInfo.properlyDominates(d2hOp.getOperation(), beforeOp)) {
              // Check if already synced in a dominating block - can skip
              // entirely
              auto it = d2hSyncedInBlock.find(d2hOp.getOperation());
              if (it != d2hSyncedInBlock.end() &&
                  domInfo.dominates(it->second, beforeOp->getBlock())) {
                continue; // Already synced in a dominating block, skip
              }

              Value event = createEventForD2H(d2hOp);

              LLVM_DEBUG({
                llvm::dbgs()
                    << "[cuda-insert-host-sync] inserting event.sync for "
                    << event << "\n";
              });

              // We want to insert the event.sync immediately prior to
              // the first use. This delays sync until as long as possible.
              // This may place event synchronization inside of loops, but
              // that's OK because we only ever write (record) an event once.
              // After the first sync, subsequent syncs should have minimal
              // cost. If it turns out to have high overhead, we can adjust this
              // insertion point to hoist synchronizations out of repetitive
              // regions (e.g. loops) if the point before the loop is dominated
              // by the d2h.
              rewriter.create<EventSyncOp>(beforeOp->getLoc(), event);

              // Note: event release is handled after the walk, at end of
              // defining block
              d2hSyncedInBlock[d2hOp.getOperation()] = beforeOp->getBlock();
              continue;
            }

            // D2H doesn't dominate - need fallback to stream sync
            Value stream = d2hOp.getStream();

            // Tier 2: If stream is in scope at sync point, use stream sync
            // there
            if (valueInScopeAt(stream, beforeOp, domInfo)) {
              if (syncedStreamsInBlock.contains(stream))
                continue;

              LLVM_DEBUG({
                llvm::dbgs()
                    << "[cuda-insert-host-sync] D2H doesn't dominate, using "
                       "stream.sync for "
                    << stream << "\n";
              });
              rewriter.create<StreamSyncOp>(beforeOp->getLoc(), stream);
              syncedStreamsInBlock.insert(stream);
              continue;
            }

            // Tier 3: Stream not in scope - insert sync immediately after D2H
            LLVM_DEBUG({
              llvm::dbgs()
                  << "[cuda-insert-host-sync] stream not in scope, inserting "
                     "immediate sync after D2H at "
                  << d2hOp.getLoc() << "\n";
            });
            OpBuilder::InsertionGuard g2(rewriter);
            rewriter.setInsertionPointAfter(d2hOp);
            rewriter.create<StreamSyncOp>(d2hOp.getLoc(), stream);
            immediatelySyncedD2HOps.insert(d2hOp.getOperation());
          }
        };

    // Walk all operations in PreOrder. For region-bearing ops, check if
    // nested ops need sync and insert before the parent op.
    auto result =
        func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          // Skip CUDA dialect ops and the function itself.
          if (isStreamSchedulableOp(op) || isa<func::FuncOp>(op))
            return WalkResult::advance();

          // Clear per-block stream cache when transitioning to a new block.
          // Note: D2H sync tracking (d2hSyncedInBlock) and event release
          // tracking (releasedEvents) are function-scope and NOT cleared here.
          if (op->getBlock() != currentBlock) {
            syncedStreamsInBlock.clear();
            currentBlock = op->getBlock();
          }

          // Get the lattice state before this operation.
          const PendingD2HLattice *lattice =
              solver.lookupState<PendingD2HLattice>(
                  solver.getProgramPointBefore(op));

          // For regular ops, check if this operation accesses any memref.
          SmallVector<Value> accessedMemrefs;
          if (!hasHostMemRefAccess(op, accessedMemrefs))
            return WalkResult::advance();

          LLVM_DEBUG({
            llvm::dbgs() << "[cuda-insert-host-sync] checking op at "
                         << op->getLoc() << " (" << op->getName() << ") with "
                         << accessedMemrefs.size() << " accessed memrefs\n";
            if (lattice) {
              llvm::dbgs() << "[cuda-insert-host-sync]   lattice: ";
              lattice->print(llvm::dbgs());
            } else {
              llvm::dbgs() << "[cuda-insert-host-sync]   lattice: nullptr\n";
            }
          });

          // Collect D2H ops that need synchronization.
          llvm::SmallSetVector<CopyD2HOp, 4> d2hOpsToSync;
          collectD2HOpsNeedingSync(lattice, accessedMemrefs, aa, d2hOpsToSync);

          LLVM_DEBUG({
            llvm::dbgs() << "[cuda-insert-host-sync]   found "
                         << d2hOpsToSync.size() << " D2H ops needing sync\n";
          });

          if (d2hOpsToSync.empty())
            return WalkResult::advance();

          LLVM_DEBUG({
            llvm::dbgs() << "[cuda-insert-host-sync] op at " << op->getLoc()
                         << " needs sync for " << d2hOpsToSync.size()
                         << " D2H ops\n";
          });

          insertSyncs(op, d2hOpsToSync.getArrayRef());
          return WalkResult::advance();
        });

    // Release all created events at the end of their defining blocks.
    // Events cannot escape their defining block (we don't yield/return them),
    // so releasing before the terminator is always safe.
    for (auto [event, block] : createdEvents) {
      Operation *terminator = block->getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.create<EventReleaseOp>(terminator->getLoc(), event);
      LLVM_DEBUG({
        llvm::dbgs() << "[cuda-insert-host-sync] releasing event " << event
                     << " at end of block\n";
      });
    }

    return success(!result.wasInterrupted());
  }
};
} // namespace

#undef DEBUG_TYPE
