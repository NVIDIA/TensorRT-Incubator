//===- SlotAssignment.cpp -------------------------------------------------===//
//
// Based on code from the "ArmSME tile allocation" transformation:
// https://github.com/llvm/llvm-project/blob/dc8e89b2b3787defa9ef1d72014c8a68c1b09a5f/mlir/lib/Dialect/ArmSME/Transforms/TileAllocation.cpp
// The original code has the following license:
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
/// Implementation of slot assignment.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Target/SlotAssignment/SlotAssignment.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <stack>

#define DEBUG_TYPE "slot-assignment"

using namespace mlir;

namespace {

/// Utility class for slot allocation.
class SlotAllocator {
public:
  SlotAllocator(unsigned maxLocalSlots)
      : maxLocalSlots(maxLocalSlots), localsInUse(maxLocalSlots),
        spillId(maxLocalSlots + 1) {}

  /// Mark local ID as unused.
  void releaseLocalId(unsigned n) {
    assert(n < maxLocalSlots && "Invalid local ID");
    localsInUse.reset(n);
  }

  /// Find the first available local ID.
  FailureOr<unsigned> allocateLocalId() {
    // We reserve the last local ID for spilling to a local table.
    auto it = localsInUse.find_first_unset_in(0, maxLocalSlots);
    if (it != -1) {
      localsInUse.set(it);
      return it;
    }
    return failure(); // All used
  }

  /// Find the first available local ID.
  void acquireLocalId(unsigned localId) {
    assert(localId < maxLocalSlots && "Invalid local ID");
    assert(!localsInUse.test(localId) && "Local ID already in use");
    localsInUse.set(localId);
  }

  /// Returns true if the local ID is marked active (in use).
  bool isLocalIdInUse(unsigned localId) const {
    assert(localId < maxLocalSlots && "Invalid local ID");
    return localsInUse.test(localId);
  }

  unsigned acquireSpillId() { return spillId++; }

  unsigned getMaxLocalSlots() const { return maxLocalSlots; }

private:
  const unsigned maxLocalSlots;
  llvm::BitVector localsInUse{maxLocalSlots};
  unsigned spillId{maxLocalSlots + 1};
};

/// A live range for one or more values. A live range is built up
/// of non-overlapping intervals [start, end) which represent parts of the
/// program where a value in the range needs to be live. Note that as the
/// intervals are non-overlapping all values within a live range can be
/// allocated to the same slot.
struct LiveRange {
  using RangeSet = llvm::IntervalMap<uint64_t, uint8_t, 16,
                                     llvm::IntervalMapHalfOpenInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;
  // Dummy value for the IntervalMap. Only the keys matter (the intervals).
  static constexpr uint8_t kValidLiveRange = 0xff;

  LiveRange(Allocator &allocator)
      : ranges(std::make_unique<RangeSet>(allocator)) {}

  /// Returns true if this range overlaps with `otherRange`.
  bool overlaps(LiveRange const &otherRange) const {
    return llvm::IntervalMapOverlaps<RangeSet, RangeSet>(*ranges,
                                                         *otherRange.ranges)
        .valid();
  }

  /// Returns true if this range is active at `point` in the program.
  bool overlaps(uint64_t point) const {
    return ranges->lookup(point) == kValidLiveRange;
  }

  /// Unions this live range with `otherRange`, aborts if the ranges
  /// overlap.
  void unionWith(LiveRange const &otherRange) {
    for (auto it = otherRange.ranges->begin(); it != otherRange.ranges->end();
         ++it)
      ranges->insert(it.start(), it.stop(), kValidLiveRange);
    values.set_union(otherRange.values);
  }

  /// Inserts an interval [start, end) for `value` into this range.
  void insert(Value value, unsigned start, unsigned end) {
    values.insert(value);
    if (start != end)
      ranges->insert(start, end, kValidLiveRange);
  }

  bool empty() const { return ranges->empty(); }
  unsigned start() const { return ranges->start(); }
  unsigned end() const { return ranges->stop(); }
  bool operator<(LiveRange const &other) const {
    return start() < other.start();
  }

  std::optional<unsigned> getLocalId() const { return localId; }

  void setLocalId(unsigned localId) { this->localId = localId; }

  const RangeSet &getRanges() const { return *ranges; }

  const SetVector<Value> &getValues() const { return values; }

private:
  /// The values contained in this live range.
  SetVector<Value> values;

  /// A set of (non-overlapping) intervals that mark where any value in
  /// `values` is live.
  std::unique_ptr<RangeSet> ranges;

  std::optional<unsigned> localId{};
};

} // namespace

void SlotAssignmentManager::print(llvm::raw_ostream &os) const {
  os << "\n\n==========SlotAssignment Results: @"
     << const_cast<FunctionOpInterface &>(funcOp).getName() << "\n";
  for (auto &[value, allocationResult] : assignment) {
    os << "  ";
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      os << "BlockArgument #" << blockArg.getArgNumber() << " of ";
      blockArg.getOwner()->printAsOperand(os);
    } else {
      os << value;
    }
    os << " -> ";
    switch (allocationResult.type) {
    case SlotType::Local:
      os << "local " << allocationResult.id;
      break;
    case SlotType::Spill:
      os << "spill " << allocationResult.id;
      break;
    }
    os << "\n";
  }
  os << "==========\n\n";
}

/// Number operations within a function to allow computing live ranges.
/// Operations are numbered consecutively within blocks, and the blocks are
/// topologically sorted (using forward edges).
static DenseMap<Operation *, unsigned>
generateOperationNumbering(FunctionOpInterface function) {
  unsigned index = 0;
  SetVector<Block *> blocks =
      getBlocksSortedByDominance(function.getFunctionBody());
  DenseMap<Operation *, unsigned> operationToIndexMap;
  for (Block *block : blocks) {
    index++; // We want block args to have their own number.
    for (Operation &op : block->getOperations())
      operationToIndexMap.try_emplace(&op, index++);
  }
  return operationToIndexMap;
}

/// Prints live ranges alongside operation names for debugging.
void dumpLiveRanges(DenseMap<Operation *, unsigned> const &operationToIndexMap,
                    ArrayRef<LiveRange const *> liveRanges,
                    FunctionOpInterface function) {
  llvm::errs() << "Liveness: @" << function.getName() << "\n";
  for (auto [blockIdx, block] : llvm::enumerate(function.getBlocks())) {
    llvm::errs() << "^bb" << blockIdx << ":\n";
    for (Operation &op : block.getOperations()) {
      unsigned operationIndex = operationToIndexMap.at(&op);
      for (LiveRange const *range : liveRanges) {
        char liveness = ' ';
        for (auto it = range->getRanges().begin();
             it != range->getRanges().end(); ++it) {
          if (it.start() == operationIndex)
            liveness = (liveness == 'E' ? '|' : 'S');
          else if (it.stop() == operationIndex)
            liveness = (liveness == 'S' ? '|' : 'E');
          else if (operationIndex >= it.start() && operationIndex < it.stop())
            liveness = '|';
        }
        llvm::errs() << liveness;
      }
      llvm::errs() << ' ' << op.getName() << '\n';
    }
  }
  llvm::errs() << "==========\n";
}

/// Gather live ranges for values from the MLIR liveness analysis.
static llvm::MapVector<Value, LiveRange> gatherValueLiveRanges(
    const DenseMap<Operation *, unsigned> &operationToIndexMap,
    LiveRange::Allocator &liveRangeAllocator, Liveness &liveness,
    FunctionOpInterface function) {
  assert(!operationToIndexMap.empty() && "expected operation numbering");
  llvm::MapVector<Value, LiveRange> liveRanges;
  /// Defines or updates a live range for a value. Live-ins may update
  /// an existing live range (rather than define a new one). Note: If
  /// `liveAtBlockEntry` is true then `firstUseOrDef` is the first operation in
  /// the block.
  auto defineOrUpdateValueLiveRange = [&](Value value, Operation *firstUseOrDef,
                                          LivenessBlockInfo const &livenessInfo,
                                          bool liveAtBlockEntry = false) {
    // Find or create a live range for `value`.
    auto [it, _] = liveRanges.try_emplace(value, liveRangeAllocator);
    LiveRange &valueLiveRange = it->second;
    auto lastUseInBlock = livenessInfo.getEndOperation(value, firstUseOrDef);
    // Add the interval [firstUseOrDef, lastUseInBlock) to the live range.
    unsigned startOpIdx =
        operationToIndexMap.at(firstUseOrDef) + (liveAtBlockEntry ? -1 : 0);
    unsigned endOpIdx = operationToIndexMap.at(lastUseInBlock);
    if (endOpIdx == startOpIdx)
      endOpIdx++;
    valueLiveRange.insert(value, startOpIdx, endOpIdx);
  };

  for (Block &block : function.getBlocks()) {
    LivenessBlockInfo const *livenessInfo = liveness.getLiveness(&block);
    // Handle block arguments:
    for (Value argument : block.getArguments())
      defineOrUpdateValueLiveRange(argument, &block.front(), *livenessInfo,
                                   /*liveAtBlockEntry=*/true);
    // Handle live-ins:
    for (Value liveIn : livenessInfo->in())
      defineOrUpdateValueLiveRange(liveIn, &block.front(), *livenessInfo,
                                   /*liveAtBlockEntry=*/true);
    // Handle new definitions:
    for (Operation &op : block) {
      for (Value result : op.getResults())
        defineOrUpdateValueLiveRange(result, &op, *livenessInfo);
    }
  }

  return liveRanges;
}

/// Choose a live range to spill (via some heuristics). This picks either a live
/// range from `overlappingRanges`, or the new live range `newRange`.
template <typename OverlappingRangesIterator>
LiveRange *
chooseSpillUsingHeuristics(OverlappingRangesIterator overlappingRanges,
                           LiveRange *newRange) {
  return newRange;
}

/// Iterate over all predecessor values to a block argument.
static void forEachPredecessorValue(BlockArgument blockArg,
                                    function_ref<void(Value)> callback) {
  Block *block = blockArg.getOwner();
  unsigned argNumber = blockArg.getArgNumber();
  for (Block *pred : block->getPredecessors()) {
    llvm::TypeSwitch<Operation *>(pred->getTerminator())
        .Case<cf::BranchOp>([&](auto branch) {
          Value predecessorOperand = branch.getDestOperands()[argNumber];
          callback(predecessorOperand);
        })
        .Case<cf::CondBranchOp>([&](auto condBranch) {
          if (condBranch.getFalseDest() == block) {
            Value predecessorOperand =
                condBranch.getFalseDestOperands()[argNumber];
            callback(predecessorOperand);
          }
          if (condBranch.getTrueDest() == block) {
            Value predecessorOperand =
                condBranch.getTrueDestOperands()[argNumber];
            callback(predecessorOperand);
          }
        });
  }
}

/// Coalesce live ranges where it would prevent unnecessary moves.
static SmallVector<LiveRange *>
coalesceLiveRanges(llvm::MapVector<Value, LiveRange> &initialLiveRanges,
                   const SlotAssignmentManager::Options &options) {
  llvm::MapVector<Value, LiveRange *> liveRanges;
  for (auto &[value, liveRange] : initialLiveRanges) {
    liveRanges.insert({value, &liveRange});
  }

  // Merge the live ranges of values `a` and `b` into one (if they do not
  // overlap). After this, the values `a` and `b` will both point to the same
  // live range (which will contain multiple values).
  auto mergeValuesIfNonOverlapping = [&](Value a, Value b) {
    LiveRange *aLiveRange = liveRanges.find(a)->second;
    LiveRange *bLiveRange = liveRanges.find(b)->second;
    if (aLiveRange != bLiveRange && !aLiveRange->overlaps(*bLiveRange)) {
      aLiveRange->unionWith(*bLiveRange);
      for (Value value : bLiveRange->getValues())
        liveRanges[value] = aLiveRange;
    }
  };

  // Merge the live ranges of new definitions with their operands.
  auto unifyDefinitionsWithOperands = [&](Value value) {
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1)
      return;
    mergeValuesIfNonOverlapping(value, op->getOperand(0));
  };

  // Merge the live ranges of block arguments with their predecessors.
  auto unifyBlockArgumentsWithPredecessors = [&](Value value) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return;
    forEachPredecessorValue(blockArg, [&](Value predecessorValue) {
      mergeValuesIfNonOverlapping(blockArg, predecessorValue);
    });
  };

  auto applyRule = [&](auto rule) {
    llvm::for_each(llvm::make_first_range(initialLiveRanges), rule);
  };

  // Unify as many live ranges as we can. This prevents unnecessary moves.
  if (options.enableBlockArgumentCoalescing)
    applyRule(unifyBlockArgumentsWithPredecessors);

  if (options.enablePureOpCoalescing)
    applyRule(unifyDefinitionsWithOperands);

  // Remove duplicate live range entries.
  SetVector<LiveRange *> uniqueLiveRanges;
  for (auto [_, liveRange] : liveRanges) {
    if (!liveRange->empty())
      uniqueLiveRanges.insert(liveRange);
  }

  // Sort the new live ranges by starting point.
  auto coalescedLiveRanges = uniqueLiveRanges.takeVector();
  llvm::stable_sort(coalescedLiveRanges,
                    [](LiveRange *a, LiveRange *b) { return *a < *b; });
  return std::move(coalescedLiveRanges);
}

/// Greedily allocate slot IDs to live ranges. Spill using simple heuristics.
static void
allocateSlotsToLiveRanges(ArrayRef<LiveRange *> liveRangesSortedByStartPoint,
                          unsigned maxLocalSlots) {
  SlotAllocator slotAllocator(maxLocalSlots);
  // `activeRanges` = Live ranges that need to be in a slot at the
  // `currentPoint` in the program.
  SetVector<LiveRange *> activeRanges;
  // `inactiveRanges` = Live ranges that _do not_ need to be in a slot
  // at the `currentPoint` in the program but could become active again later.
  // An inactive section of a live range can be seen as a 'hole' in the live
  // range, where it is possible to reuse the live range's slot ID _before_ it
  // has ended. By identifying 'holes', the allocator can reuse slots more
  // often, which helps avoid costly slot spills.
  SetVector<LiveRange *> inactiveRanges;
  for (LiveRange *nextRange : liveRangesSortedByStartPoint) {
    auto currentPoint = nextRange->start();
    // 1. Update the `activeRanges` at `currentPoint`.
    activeRanges.remove_if([&](LiveRange *activeRange) {
      // Check for live ranges that have expired.
      if (activeRange->end() <= currentPoint) {
        slotAllocator.releaseLocalId(*activeRange->getLocalId());
        return true;
      }
      // Check for live ranges that have become inactive.
      if (!activeRange->overlaps(currentPoint)) {
        slotAllocator.releaseLocalId(*activeRange->getLocalId());
        inactiveRanges.insert(activeRange);
        return true;
      }
      return false;
    });
    // 2. Update the `inactiveRanges` at `currentPoint`.
    inactiveRanges.remove_if([&](LiveRange *inactiveRange) {
      // Check for live ranges that have expired.
      if (inactiveRange->end() <= currentPoint) {
        return true;
      }
      // Check for live ranges that have become active.
      if (inactiveRange->overlaps(currentPoint)) {
        slotAllocator.acquireLocalId(*inactiveRange->getLocalId());
        activeRanges.insert(inactiveRange);
        return true;
      }
      return false;
    });

    // 3. Collect inactive live ranges that overlap with the new live range.
    // Note: The overlap checks in steps 1 and 2 only look at the `currentPoint`
    // whereas this checks if there is an overlap at any future point too.
    SmallVector<LiveRange *> overlappingInactiveRanges;
    for (LiveRange *inactiveRange : inactiveRanges) {
      if (inactiveRange->overlaps(*nextRange)) {
        if (*inactiveRange->getLocalId() < slotAllocator.getMaxLocalSlots() &&
            !slotAllocator.isLocalIdInUse(*inactiveRange->getLocalId())) {
          // We need to reserve the slot IDs of overlapping inactive ranges to
          // prevent two (overlapping) live ranges from getting the same slot
          // ID.
          slotAllocator.acquireLocalId(*inactiveRange->getLocalId());
          overlappingInactiveRanges.push_back(inactiveRange);
        }
      }
    }

    // 4. Allocate a slot ID to `nextRange`.
    FailureOr<unsigned> localId = slotAllocator.allocateLocalId();
    if (succeeded(localId)) {
      nextRange->setLocalId(*localId);
    } else {
      // Create an iterator over all overlapping live ranges.
      auto allOverlappingRanges = llvm::concat<LiveRange>(
          llvm::make_pointee_range(activeRanges.getArrayRef()),
          llvm::make_pointee_range(overlappingInactiveRanges));
      // Choose an overlapping live range to spill.
      LiveRange *rangeToSpill =
          chooseSpillUsingHeuristics(allOverlappingRanges, nextRange);
      if (rangeToSpill != nextRange) {
        // Spill an (in)active live range (so release its slot ID first).
        slotAllocator.releaseLocalId(*rangeToSpill->getLocalId());
        // This will always succeed after a spill (of an active live range).
        nextRange->setLocalId(*slotAllocator.allocateLocalId());
        // Remove the live range from the active/inactive sets.
        if (!activeRanges.remove(rangeToSpill)) {
          bool removed = inactiveRanges.remove(rangeToSpill);
          assert(removed && "expected a range to be removed!");
          (void)removed;
        }
      }
      rangeToSpill->setLocalId(slotAllocator.acquireSpillId());
    }

    // 5. Insert the live range into the active ranges.
    if (nextRange->getLocalId() < slotAllocator.getMaxLocalSlots())
      activeRanges.insert(nextRange);

    // 6. Release slots reserved for inactive live ranges (in step 3).
    for (LiveRange *range : overlappingInactiveRanges) {
      if (*range->getLocalId() < slotAllocator.getMaxLocalSlots())
        slotAllocator.releaseLocalId(*range->getLocalId());
    }
  }
}

SlotAssignmentManager::SlotAssignmentManager(FunctionOpInterface func_,
                                             Options options_)
    : options(std::move(options_)), funcOp(func_) {
  // Number operations within the function.
  DenseMap<Operation *, unsigned> operationToIndexMap =
      generateOperationNumbering(funcOp);

  LiveRange::Allocator liveRangeAllocator;
  Liveness liveness(funcOp);
  llvm::MapVector<Value, LiveRange> initialLiveRanges = gatherValueLiveRanges(
      operationToIndexMap, liveRangeAllocator, liveness, funcOp);

  LLVM_DEBUG({
    // Wrangle initial live ranges into a form suitable for printing.
    auto nonEmpty = llvm::make_filter_range(
        llvm::make_second_range(initialLiveRanges),
        [&](LiveRange const &liveRange) { return !liveRange.empty(); });
    auto initialRanges = llvm::to_vector(llvm::map_range(
        nonEmpty, [](LiveRange const &liveRange) { return &liveRange; }));
    llvm::stable_sort(
        initialRanges,
        [](LiveRange const *a, LiveRange const *b) { return *a < *b; });
    llvm::errs() << "\n========== Initial Live Ranges:\n";
    dumpLiveRanges(operationToIndexMap, initialRanges, funcOp);
  });

  SmallVector<LiveRange *> coalescedLiveRanges =
      coalesceLiveRanges(initialLiveRanges, options);

  allocateSlotsToLiveRanges(coalescedLiveRanges, options.maxLocalSlots);

  LLVM_DEBUG({
    llvm::errs() << "\n========== Coalesced Live Ranges:\n";
    dumpLiveRanges(operationToIndexMap, coalescedLiveRanges, funcOp);
  });

  for (const LiveRange *liveRange : coalescedLiveRanges) {
    for (Value value : liveRange->getValues()) {
      if (liveRange->getLocalId() <= options.maxLocalSlots) {
        assignment.try_emplace(value, SlotType::Local,
                               *liveRange->getLocalId());
      } else {
        assignment.try_emplace(value, SlotType::Spill,
                               *liveRange->getLocalId());
        this->spill = true;
      }
    }
  }
}

namespace {
enum class SlotSwapStatus : int8_t { NotStarted, Started, Done };
enum class SwapWorkItem : int8_t { Start, Emit };

struct SlotSwapWorkItem {
  unsigned idx;
  SwapWorkItem kind;
};

template <typename T>
struct SlotSwapState {
  std::stack<SlotSwapWorkItem> workItems;
  SmallVector<SlotSwapStatus> statuses;
  MutableArrayRef<T> sourceSlots;
  ArrayRef<T> targetSlots;
  T tempSlotId;
};
} // namespace

/// Given a particular index into the swap arrays, enqueue the
/// command to emit the swap and then recursively enqueue all work
/// that this swap depends on.
template <typename T>
void handleSlotSwapStart(
    unsigned idx, SlotSwapState<T> &state,
    llvm::function_ref<void(T sourceSlot, T targetSlot)> emitMove) {
  switch (state.statuses[idx]) {
  case SlotSwapStatus::NotStarted:
    state.statuses[idx] = SlotSwapStatus::Started;
    state.workItems.push({idx, SwapWorkItem::Emit});
    // Enqueue all dependencies on top of the "emit" work item.
    for (unsigned i = 0, e = state.sourceSlots.size(); i < e; ++i) {
      if (i != idx && state.sourceSlots[i] == state.targetSlots[idx])
        state.workItems.push({i, SwapWorkItem::Start});
    }
    break;
  case SlotSwapStatus::Started:
    // We visited this index before, so emit the move to a temporary slot and
    // update the source to be the temporary slot.
    emitMove(state.sourceSlots[idx], state.tempSlotId);
    state.sourceSlots[idx] = state.tempSlotId;
    break;
  case SlotSwapStatus::Done:
    break;
  }
}

/// Main loop for the slot swap algorithm. This is responsible for
/// enqueuing all swaps required for index `idx`. At completion, swap
/// for `idx` and possible other swaps that it depends on will be complete.
template <typename T>
void emitSlotSwapsHelper(
    unsigned idx, SlotSwapState<T> &state,
    llvm::function_ref<void(T sourceSlot, T targetSlot)> emitMove) {
  auto &workItems = state.workItems;
  workItems.push({idx, SwapWorkItem::Start});
  while (!workItems.empty()) {
    SlotSwapWorkItem workItem = workItems.top();
    workItems.pop();
    switch (workItem.kind) {
    case SwapWorkItem::Start: {
      handleSlotSwapStart(workItem.idx, state, emitMove);
      break;
    }
    case SwapWorkItem::Emit:
      state.statuses[workItem.idx] = SlotSwapStatus::Done;
      emitMove(state.sourceSlots[workItem.idx],
               state.targetSlots[workItem.idx]);

      break;
    }
  }
}

/// Emits a sequence of sequential swaps, using a temporary slot as
/// necessary. The ID for the temporary slot is passed in as `tempSlotId`.
template <typename T>
void emitSlotSwapImpl(
    llvm::MutableArrayRef<T> sourceSlots, llvm::ArrayRef<T> targetSlots,
    T tempSlotId,
    llvm::function_ref<void(T sourceSlot, T targetSlot)> emitMove) {
  assert(sourceSlots.size() == targetSlots.size() &&
         "expected same number of source and target slots");

  SlotSwapState<T> state;
  state.statuses.assign(sourceSlots.size(), SlotSwapStatus::NotStarted);
  state.sourceSlots = sourceSlots;
  state.targetSlots = targetSlots;
  state.tempSlotId = tempSlotId;
  for (unsigned idx = 0, e = sourceSlots.size(); idx < e; ++idx) {
    if (state.statuses[idx] == SlotSwapStatus::NotStarted)
      emitSlotSwapsHelper(idx, state, emitMove);
  }
}

void mlir::emitSlotSwap(
    llvm::MutableArrayRef<int32_t> sourceSlots,
    llvm::ArrayRef<int32_t> targetSlots, int32_t tempSlotId,
    llvm::function_ref<void(int32_t sourceSlot, int32_t targetSlot)> emitMove) {
  emitSlotSwapImpl<int32_t>(sourceSlots, targetSlots, tempSlotId, emitMove);
}

void mlir::emitSlotSwap(
    llvm::MutableArrayRef<StringRef> sourceSlots,
    llvm::ArrayRef<StringRef> targetSlots, StringRef tempSlotId,
    llvm::function_ref<void(StringRef sourceSlot, StringRef targetSlot)>
        emitMove) {
  emitSlotSwapImpl<StringRef>(sourceSlots, targetSlots, tempSlotId, emitMove);
}
