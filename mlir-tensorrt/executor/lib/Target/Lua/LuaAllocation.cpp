//===- LuaAllocation.cpp  -------------------------------------------------===//
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
/// Implementation of Lua variable allocation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Target/Lua/LuaAllocation.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lua-allocation"

using namespace mlir;

namespace {

/// Utility class for Lua local variable allocation.
class LuaLocalAllocator {
public:
  static constexpr unsigned kMaxLocalNum = 200;

  /// Mark local ID as unused.
  void releaseLocalId(unsigned n) {
    assert(n < kMaxLocalNum - 1 && "Invalid local ID");
    localsInUse.reset(n);
  }

  /// Find the first available local ID.
  FailureOr<unsigned> acquireLocalId() {
    // We reserve the last local ID for spilling to a local table.
    auto it = localsInUse.find_first_unset_in(0, kMaxLocalNum - 1);
    if (it != -1) {
      localsInUse.set(it);
      return it;
    }
    return failure(); // All used
  }

  unsigned acquireSpillId() {
    if (spillId == 0) {
      assert(!localsInUse.test(kMaxLocalNum - 1) &&
             "Failed to allocate local ID for a spill table");
      localsInUse.set(kMaxLocalNum - 1);
    }
    return spillId++;
  }

private:
  llvm::BitVector localsInUse{kMaxLocalNum};
  unsigned spillId{0};
};

/// A live range for a (collection of) values. A live range is built up
/// of non-overlapping intervals [start, end) which represent parts of the
/// program where a value in the range needs to be live. Note that as the
/// intervals are non-overlapping all values within a live range can be
/// allocated to the same Lua local variable.
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

  /// The values contained in this live range.
  SetVector<Value> values;

  /// A set of (non-overlapping) intervals that mark where any value in
  /// `values` is live.
  std::unique_ptr<RangeSet> ranges;
};

} // namespace

void LuaAllocation::print(llvm::raw_ostream &os) const {
  os << "\n\n==========LuaAllocation Results: @"
     << const_cast<func::FuncOp &>(funcOp).getName() << "\n";
  for (auto &[value, allocationResult] : allocationResults) {
    os << "  ";
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      os << "BlockArgument #" << blockArg.getArgNumber() << " of ";
      blockArg.getOwner()->printAsOperand(os);
    } else {
      os << value;
    }
    os << " -> ";
    switch (allocationResult.type) {
    case LuaAllocationType::Local:
      os << "local " << allocationResult.id;
      break;
    case LuaAllocationType::Spill:
      os << "spill " << allocationResult.id;
      break;
    case LuaAllocationType::Global:
      os << "global " << allocationResult.id;
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

static void
dumpLiveRanges(func::FuncOp funcOp,
               const DenseMap<Operation *, unsigned> &operationToIndexMap,
               const llvm::MapVector<Value, LiveRange> &liveRanges) {
  auto function = const_cast<func::FuncOp &>(funcOp);
  LLVM_DEBUG(llvm::dbgs() << "\n\n==========LuaAllocation Live Ranges: @"
                          << const_cast<func::FuncOp &>(funcOp).getName()
                          << "\n");
  for (auto [blockIdx, block] : llvm::enumerate(function.getBlocks())) {
    LLVM_DEBUG(llvm::dbgs() << "^bb" << blockIdx << ":\n");
    for (Operation &op : block.getOperations()) {
      unsigned operationIndex = operationToIndexMap.at(&op);
      for (auto &[value, liveRange] : liveRanges) {
        char liveness = ' ';
        for (auto it = liveRange.ranges->begin(); it != liveRange.ranges->end();
             ++it) {
          if (it.start() + 1 == it.stop())
            continue;
          if (it.start() == operationIndex)
            liveness = (liveness == 'E' ? '|' : 'S');
          else if (it.stop() == operationIndex + 1)
            liveness = (liveness == 'S' ? '|' : 'E');
          else if (operationIndex >= it.start() &&
                   operationIndex + 1 < it.stop())
            liveness = '|';
        }
        LLVM_DEBUG(llvm::dbgs() << liveness);
      }
      LLVM_DEBUG(llvm::dbgs() << ' ' << op.getName() << '\n');
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "==========\n\n");
}

void LuaAllocation::allocate(func::FuncOp funcOp) {
  // Number operations within the function.
  DenseMap<Operation *, unsigned> operationToIndexMap =
      generateOperationNumbering(funcOp);

  LiveRange::Allocator liveRangeAllocator;
  Liveness liveness(funcOp);
  llvm::MapVector<Value, LiveRange> liveRanges;
  // Define a live range for a value.
  auto getValueLivenessRange = [&](Value value, bool isBlockArg = false) {
    auto [it, _] = liveRanges.try_emplace(value, liveRangeAllocator);
    auto liveOperations = liveness.resolveLiveness(value);
    auto minId = std::numeric_limits<unsigned>::max();
    auto maxId = std::numeric_limits<unsigned>::min();
    // Since Lua lacks the concept of block arguments, we simulate them by
    // assigning values just before the `goto` that branches to the target
    // block. As a result, the live ranges for block arguments must begin at the
    // branch site, not at the destination block’s entry. The MLIR builtin
    // liveness analysis does not capture this behavior, as it treats the block
    // entry as the start of the live range.
    if (isBlockArg) {
      BlockArgument blockArg = cast<BlockArgument>(value);
      Block *destBlock = blockArg.getOwner();
      if (blockArg.getOwner()->hasNoPredecessors()) {
        minId = 0;
      } else {
        for (Block *pred : destBlock->getPredecessors()) {
          Operation *terminator = pred->getTerminator();
          if (isa<BranchOpInterface>(terminator) &&
              llvm::is_contained(terminator->getSuccessors(), destBlock)) {
            minId = std::min(minId, operationToIndexMap.at(terminator));
            maxId = std::max(maxId, operationToIndexMap.at(terminator) + 1);
          }
        }
      }
    }
    llvm::for_each(liveOperations, [&](Operation *liveOp) {
      minId = std::min(minId, operationToIndexMap.at(liveOp));
      maxId = std::max(maxId, operationToIndexMap.at(liveOp) + 1);
    });
    it->second.insert(value, minId, maxId);
  };

  // Compute live ranges for all block arguments and results.
  for (Block &block : funcOp.getFunctionBody()) {
    // Handle block arguments.
    for (Value arg : block.getArguments())
      getValueLivenessRange(arg, true);

    // Handle operation results.
    for (Operation &op : block) {
      for (Value result : op.getResults())
        getValueLivenessRange(result);
    }
  }
  // Sort the live ranges by starting point (ready for variable allocation).
  llvm::stable_sort(liveRanges, [](const auto &a, const auto &b) {
    return a.second < b.second;
  });

  LuaLocalAllocator localAllocator;
  SetVector<LiveRange *> activeLiveRanges;
  for (auto &[_, liveRange] : liveRanges) {
    auto currentPoint = liveRange.start();

    // Remove any live ranges that are no longer active.
    activeLiveRanges.remove_if([&](LiveRange *activeLiveRange) {
      if (!activeLiveRange->overlaps(currentPoint) &&
          allocationResults.at(activeLiveRange->values[0]).type ==
              LuaAllocationType::Local) {
        auto localId = allocationResults.at(activeLiveRange->values[0]).id;
        localAllocator.releaseLocalId(localId);
        return true;
      }
      return false;
    });

    // Add any new live ranges that are active.
    activeLiveRanges.insert(&liveRange);

    // Allocate variables for the values in the live range.
    for (Value value : liveRange.values) {
      auto localId = localAllocator.acquireLocalId();
      if (succeeded(localId)) {
        allocationResults.try_emplace(value, LuaAllocationType::Local,
                                      *localId);
      } else {
        allocationResults.try_emplace(value, LuaAllocationType::Spill,
                                      localAllocator.acquireSpillId());
        spill = true;
      }
    }
  }
  dumpLiveRanges(funcOp, operationToIndexMap, liveRanges);
}
