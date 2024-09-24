//===- AllocsToGlobals.cpp ------------------------------------------------===//
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
/// Implementation of the `executor-allocs-to-globals` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "executor-allocs-to-globals"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTORALLOCSTOGLOBALSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

struct BlockAllocLiveIntervals {
  DenseMap<Operation *, unsigned> opToIndex;
  DenseMap<memref::AllocOp, llvm::BitVector> allocRanges;
  DenseMap<memref::AllocOp, unsigned> liveRangeSize;
};

/// Return the dealloc associated with `op` or nullptr if none is found.
static memref::DeallocOp getDealloc(memref::AllocOp op) {
  for (Operation *user : op->getUsers())
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
      return dealloc;
  return nullptr;
}

/// Compute start/end intervals for the relevant allocations in each block.
static BlockAllocLiveIntervals getAllocIntervals(Block *block) {
  SmallVector<std::pair<memref::AllocOp, Operation *>> allocOps;
  mlir::BufferViewFlowAnalysis analysis(block->getParentOp());
  using ValueSet = mlir::BufferViewFlowAnalysis::ValueSetT;

  for (auto allocOp : block->getOps<memref::AllocOp>()) {
    // Get the dealloc.
    memref::DeallocOp deallocOp = getDealloc(allocOp);
    if (!deallocOp || allocOp->getBlock() != deallocOp->getBlock() ||
        !allocOp.getDynamicSizes().empty())
      continue;

    // Find all potentially aliasing values of this allocation.
    ValueSet aliasValues = analysis.resolve(allocOp);
    Operation *lastUser = allocOp;
    for (Value v : aliasValues) {
      for (Operation *user : v.getUsers()) {
        while (user->getBlock() != block)
          user = user->getParentOp();
        if (deallocOp != user && lastUser->isBeforeInBlock(user))
          lastUser = user;
      }
    }
    allocOps.emplace_back(allocOp, lastUser);
  }

  BlockAllocLiveIntervals intervals;
  for (auto [idx, op] : llvm::enumerate(block->getOperations()))
    intervals.opToIndex[&op] = idx;
  intervals.allocRanges.reserve(intervals.opToIndex.size());

  for (auto [allocOp, deallocOp] : allocOps) {
    llvm::BitVector vec(intervals.opToIndex.size());
    unsigned start = intervals.opToIndex[allocOp];
    unsigned end = intervals.opToIndex[deallocOp];
    vec.set(start, end);
    intervals.allocRanges[allocOp] = std::move(vec);
    intervals.liveRangeSize[allocOp] = end - start;
  }
  return intervals;
}

/// Returns the number of bytes required for a buffer assigned to hold `t`.
/// TODO: we should be using the data layout here.
static int64_t getRequiredBufferSize(MemRefType t) {
  // This is an upper bound since index type could also be 32 bits.
  if (t.getElementType().isIndex())
    return t.getNumElements() *
           llvm::divideCeil(IndexType::kInternalStorageBitWidth, 8);
  return t.getNumElements() * llvm::divideCeil(t.getElementTypeBitWidth(), 8);
}

/// Determine if two memref types are "compatible" in the sense that the `to`
/// memref type can re-use a block of memory that was used to store `from`. We
/// use the most trivial rule, which is to just check that buffer size of "from"
/// is >= buffer size of 'to'.
/// TODO: take alignment into account
static unsigned isCompatible(MemRefType from, MemRefType to) {
  return from == to;
  // return getRequiredBufferSize(to) <= getRequiredBufferSize(from);
}

/// Print the live ranges to `os` as a textual diagram. Useful for debugging and
/// visualization. For each allocation, print out a line '[type] ____xxxx___' to
/// represent the live range where the 'x's are where the allocation is live. We
/// will repeat the range part of the string at a height roughly correlated with
/// the byte size.
static void printLiveRanges(llvm::raw_ostream &os,
                            const BlockAllocLiveIntervals &intervals) {
  // Bin the allocation sizes to find how we should map byte size to block
  // heights.
  int64_t minAllocationSize = std::numeric_limits<int64_t>::max(),
          maxAllocationSize = 0;
  for (auto [op, liveRange] : intervals.allocRanges) {
    int64_t bytes = getRequiredBufferSize(op.getType());
    minAllocationSize = std::min(minAllocationSize, bytes);
    maxAllocationSize = std::max(maxAllocationSize, bytes);
  }

  // Make 10 different bins.
  int64_t spread = maxAllocationSize - minAllocationSize;
  int64_t numBins = 10;
  int64_t binSize = std::max<int64_t>(static_cast<float>(spread) / numBins, 1);

  auto getHeight = [&](int64_t numBytes) {
    return std::max<int64_t>((numBytes - minAllocationSize) / binSize, 1);
  };

  for (auto [op, liveRange] : intervals.allocRanges) {
    std::string typeStr;
    {
      // Formatv doesn't seem to support fixed width with range
      // object, so we using streamer.
      llvm::raw_string_ostream ss(typeStr);
      llvm::interleave(op.getType().getShape(), ss, "x");
      ss << "x" << op.getType().getElementType();
    }

    std::string liveRangeStr;
    {
      llvm::raw_string_ostream ss(liveRangeStr);
      int64_t pre = liveRange.find_first();
      int64_t last = liveRange.find_last();
      if (pre > 0)
        ss << std::string(pre, '_');
      if (last - pre + 1 > 0)
        ss << std::string(last - pre + 1, 'x');
      if (last < liveRange.size() - 1)
        ss << std::string(liveRange.size() - 1 - last, '_');
    }
    std::string paddingStr = llvm::formatv("{0,-20}", "").str();
    for (int64_t row = getHeight(getRequiredBufferSize(op.getType())); row > 1;
         row--)
      os << paddingStr << liveRangeStr << "\n";
    os << llvm::formatv("{0,-20}", StringRef(typeStr).take_front(20))
       << liveRangeStr << "\n";
  }
}

static GlobalOp findAvailableGlobal(
    memref::AllocOp alloc, const llvm::BitVector &range,
    llvm::DenseMap<executor::GlobalOp, SmallVector<memref::AllocOp>>
        &assignedTo,
    BlockAllocLiveIntervals &intervals) {
  for (auto [globalOp, userSet] : assignedTo) {
    if (!isCompatible(llvm::cast<MemRefType>(globalOp.getType()),
                      llvm::cast<MemRefType>(alloc.getType())))
      continue;
    if (llvm::any_of(userSet, [&](memref::AllocOp &user) {
          BitVector tmp = intervals.allocRanges[user];
          tmp &= range;
          return tmp.any();
        }))
      continue;
    return globalOp;
  }
  return nullptr;
}

static FailureOr<std::pair<GlobalOp, GetGlobalOp>>
outlineAllocToGlobal(RewriterBase &rewriter, memref::AllocOp op, unsigned id) {
  rewriter.setInsertionPoint(op);
  // Find deallocation
  memref::DeallocOp dealloc = getDealloc(op);
  if (!dealloc)
    return std::make_pair(GlobalOp(nullptr), GetGlobalOp(nullptr));

  // Drop the deallocation since the runtime will handle the cleanup.
  rewriter.eraseOp(dealloc);

  std::string name = llvm::formatv("workspace_{0}", id);

  executor::GlobalOp global = createUniqueGlobalOp(
      op.getLoc(), op->getParentOfType<ModuleOp>(), name, op.getType(), false,
      [&op](OpBuilder &b, Location loc) {
        auto newAllocOp = cast<memref::AllocOp>(b.clone(*op));
        b.create<executor::ReturnOp>(loc, newAllocOp.getResult());
      });

  return std::make_pair(
      global, rewriter.replaceOpWithNewOp<executor::GetGlobalOp>(op, global));
}

LogicalResult
assignBlocks(RewriterBase &rewriter, BlockAllocLiveIntervals &intervals,
             llvm::DenseMap<memref::AllocOp, executor::GlobalOp> &assignment) {
  llvm::DenseMap<executor::GlobalOp, SmallVector<memref::AllocOp>> assignedTo;
  unsigned globalCount = 0;
  unsigned memoryUsedBytes = 0;
  unsigned memorySavedBytes = 0;
  for (auto [alloc, range] : intervals.allocRanges) {

    int64_t allocBytes = getRequiredBufferSize(alloc.getType());
    if (GlobalOp avail =
            findAvailableGlobal(alloc, range, assignedTo, intervals)) {
      assignment[alloc] = avail;
      assignedTo[avail].push_back(alloc);

      memref::DeallocOp deallocOp = getDealloc(alloc);
      assert(deallocOp && "expected matching dealloc");
      rewriter.eraseOp(deallocOp);
      rewriter.setInsertionPoint(alloc);
      rewriter.replaceOpWithNewOp<GetGlobalOp>(alloc, avail);
      memorySavedBytes += allocBytes;
      continue;
    }

    FailureOr<std::pair<GlobalOp, GetGlobalOp>> newGlobal =
        outlineAllocToGlobal(rewriter, alloc, globalCount++);
    if (failed(newGlobal))
      return failure();

    assignment[alloc] = std::get<0>(*newGlobal);
    assignedTo[std::get<0>(*newGlobal)] = SmallVector<memref::AllocOp>{alloc};
    memoryUsedBytes += allocBytes;
  }

  LLVM_DEBUG(DBGS() << " memory used: " << memoryUsedBytes << "\n";
             DBGS() << " memory saved: " << memorySavedBytes << "\n");

  return success();
}

namespace {
class ExecutorAllocsToGlobalsPass
    : public ::executor::impl::ExecutorAllocsToGlobalsPassBase<
          ExecutorAllocsToGlobalsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(checkIsModuleLike(op)))
      return signalPassFailure();

    StringAttr moduleName = op->hasAttr(SymbolTable::getSymbolAttrName())
                                ? SymbolTable::getSymbolName(op)
                                : nullptr;

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    SmallVector<Block *> blocks;
    for (func::FuncOp func : op->getRegion(0).getOps<func::FuncOp>()) {
      if (func.isPrivate() || func.isExternal())
        continue;
      func.walk([&](Block *block) { blocks.push_back(block); });
    }

    for (Block *block : blocks) {
      BlockAllocLiveIntervals intervals = getAllocIntervals(block);
      LLVM_DEBUG({
        if (!intervals.allocRanges.empty()) {
          Operation *parent = block->getParentOp();
          if (auto funcOp = dyn_cast<func::FuncOp>(parent)) {
            DBGS() << "Diagram for func "
                   << (moduleName ? moduleName.strref() : StringRef(""))
                   << "::" << funcOp.getName() << ":\n";
            printLiveRanges(llvm::dbgs(), intervals);
          }
        }
      });

      llvm::DenseMap<memref::AllocOp, executor::GlobalOp> assignment;
      if (failed(assignBlocks(rewriter, intervals, assignment))) {
        emitError(op->getLoc())
            << "failed to analyze block in " << getArgument();
        return signalPassFailure();
      }
    }
  }
};
} // namespace
