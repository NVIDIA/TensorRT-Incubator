//===- CUDAScheduleAsync.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// Implementation of `cuda-schedule-async`.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/StreamSchedulableOpInterface.h"
#include "mlir-tensorrt/Analysis/AliasAnalysis.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::cuda {

#define GEN_PASS_DEF_CUDASCHEDULEASYNCPASS
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"
} // namespace mlir::cuda

using namespace mlir;
using namespace mlir::cuda;

/// The program stream is the 'main stream' for the program. The user gives this
/// in the runtime API. They expect the program will use this stream to
/// determine when inputs are ready and add events to it that indicate when the
/// outputs are ready.
static constexpr int64_t kProgramStreamIndex = 0;
/// The copy stream is used for all copy operations internal to the program.
static constexpr int64_t kCopyStreamIndex = 1;
/// The compute stream is used for all compute operations internal to the
/// program.
static constexpr int64_t kComputeStreamIndex = 2;

[[maybe_unused]] static constexpr int64_t kNumStreams = 3;

namespace {

struct MemEffects {
  SmallVector<Value> reads;
  SmallVector<Value> writes;
  SmallVector<Value> frees;
  bool hasUnknownEffects{false};
};
} // namespace

static MemEffects getMemEffects(Operation *op) {
  MemEffects result;

  // Use standard utility to check for unknown effects.
  if (hasUnknownEffects(op)) {
    result.hasUnknownEffects = true;
    return result;
  }

  auto iface = cast<MemoryEffectOpInterface>(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);

  for (const auto &eff : effects) {
    Value v = eff.getValue();
    if (!v) {
      result.hasUnknownEffects = true;
      continue;
    }
    if (!isa<MemRefType>(v.getType()))
      continue;

    if (isa<MemoryEffects::Read>(eff.getEffect())) {
      result.reads.push_back(v);
    } else if (isa<MemoryEffects::Write>(eff.getEffect())) {
      result.writes.push_back(v);
    } else if (isa<MemoryEffects::Free>(eff.getEffect())) {
      result.frees.push_back(v);
    } else if (isa<MemoryEffects::Allocate>(eff.getEffect())) {
      // Allocate effects don't create dependencies - they just create new
      // memory that doesn't alias with anything yet. Safe to ignore.
      continue;
    } else {
      // Conservatively treat truly unknown effects as needing synchronization.
      result.hasUnknownEffects = true;
    }
  }

  return result;
}

static bool mayAliasAny(AliasAnalysis &aa, Value a, ArrayRef<Value> bs) {
  for (Value b : bs) {
    if (aa.alias(a, b) != AliasResult::NoAlias)
      return true;
  }
  return false;
}

static bool hasMemoryConflict(AliasAnalysis &aa, const MemEffects &a,
                              const MemEffects &b) {
  if (a.hasUnknownEffects || b.hasUnknownEffects)
    return true;

  // RAW / WAW / WAF / F*
  SmallVector<Value> bReadWriteFree;
  bReadWriteFree.append(b.reads.begin(), b.reads.end());
  bReadWriteFree.append(b.writes.begin(), b.writes.end());
  bReadWriteFree.append(b.frees.begin(), b.frees.end());

  for (Value w : a.writes)
    if (mayAliasAny(aa, w, bReadWriteFree))
      return true;
  for (Value f : a.frees)
    if (mayAliasAny(aa, f, bReadWriteFree))
      return true;

  // WAR / FAR
  SmallVector<Value> bWriteFree;
  bWriteFree.append(b.writes.begin(), b.writes.end());
  bWriteFree.append(b.frees.begin(), b.frees.end());
  for (Value r : a.reads)
    if (mayAliasAny(aa, r, bWriteFree))
      return true;

  return false;
}

static bool isCopyLike(Operation *op) {
  return isa<cuda::CopyD2DOp, cuda::CopyD2HOp, cuda::CopyH2DOp>(op);
}

static bool hasStreamOperands(Operation *op) {
  if (auto iface = dyn_cast<mtrt::compiler::StreamSchedulableOp>(op))
    return iface.getStreamOperand() != nullptr;
  return false;
}

static bool isSchedulableCudaCommand(Operation *op) {
  if (!op)
    return false;

  // Exclude stream/event management ops and other non-command ops.
  if (isa<cuda::GetActiveDeviceOp, cuda::GetGlobalStreamOp,
          cuda::StreamCreateOp, cuda::StreamDestroyOp, cuda::StreamSyncOp,
          cuda::StreamWaitEventOp, cuda::StreamRecordEventOp,
          cuda::EventCreateOp, cuda::EventCreateOnStreamOp, cuda::EventSyncOp,
          cuda::EventElapsedTimeOp, cuda::EventReleaseOp, cuda::DeviceCountOp,
          cuda::GetDeviceOp, cuda::SetActiveDeviceOp, cuda::GetFunctionOp,
          cuda::CompiledModuleOp>(op))
    return false;

  return hasStreamOperands(op);
}

static void setAllStreamOperandsTo(Operation *op, Value stream) {
  auto iface = dyn_cast<mtrt::compiler::StreamSchedulableOp>(op);
  if (!iface)
    return;
  OpOperand *streamOperand = iface.getStreamOperand();
  if (streamOperand)
    streamOperand->set(stream);
}

static Value getOrCreateProducerEvent(RewriterBase &rewriter,
                                      DenseMap<Operation *, Value> &eventMap,
                                      Operation *producer, Value producerStream,
                                      llvm::function_ref<Value()> getDevice) {
  if (auto it = eventMap.find(producer); it != eventMap.end())
    return it->second;

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(producer);
  Location loc = producer->getLoc();
  Value e = rewriter.create<cuda::EventCreateOnStreamOp>(loc, producerStream);
  eventMap[producer] = e;
  return e;
}

static void scheduleBlock(Block &block,
                          llvm::function_ref<Value(int64_t)> getStream,
                          AliasAnalysis &aa, IRRewriter &rewriter,
                          llvm::function_ref<Value()> getDevice) {
  SmallVector<Operation *> ops;
  ops.reserve(block.getOperations().size());

  for (Operation &op : block)
    if (isSchedulableCudaCommand(&op))
      ops.push_back(&op);

  if (ops.empty())
    return;

  // Assign streams.
  DenseMap<Operation *, int64_t> streamAssignment;
  for (Operation *op : ops) {
    int64_t idx = 1;
    if (isCopyLike(op)) {
      idx = kCopyStreamIndex;
    } else {
      idx = kComputeStreamIndex;
    }
    streamAssignment[op] = idx;
    setAllStreamOperandsTo(op, getStream(idx));
  }

  // Compute memory effects once.
  DenseMap<Operation *, MemEffects> effectsMap;
  effectsMap.reserve(ops.size());
  for (Operation *op : ops)
    effectsMap[op] = getMemEffects(op);

  // Insert cross-stream synchronization.
  DenseMap<Operation *, Value> producerToEvent;
  for (unsigned i = 0; i < ops.size(); ++i) {
    Operation *consumer = ops[i];
    int64_t consumerStreamIdx = streamAssignment.lookup(consumer);

    llvm::SmallPtrSet<Value, 8> waitedEvents;
    for (unsigned j = 0; j < i; ++j) {
      Operation *producer = ops[j];
      int64_t producerStreamIdx = streamAssignment.lookup(producer);
      if (producerStreamIdx == consumerStreamIdx)
        continue;
      if (!hasMemoryConflict(aa, effectsMap.lookup(producer),
                             effectsMap.lookup(consumer)))
        continue;

      Value e =
          getOrCreateProducerEvent(rewriter, producerToEvent, producer,
                                   getStream(producerStreamIdx), getDevice);
      if (!waitedEvents.insert(e).second)
        continue;

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(consumer);
      rewriter.create<cuda::StreamWaitEventOp>(consumer->getLoc(),
                                               getStream(consumerStreamIdx), e);
    }
  }

  // Release all events created in this block before the terminator.
  // This guarantees events are created and released in the same block,
  // avoiding scope issues with nested regions.
  if (!producerToEvent.empty()) {
    Operation *terminator = block.getTerminator();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(terminator);
    Location loc = terminator->getLoc();
    for (auto &kv : producerToEvent)
      rewriter.create<cuda::EventReleaseOp>(loc, kv.second);
  }
}

static void scheduleRegion(Region &region,
                           llvm::function_ref<Value(int64_t)> getStream,
                           AliasAnalysis &aa, IRRewriter &rewriter,
                           llvm::function_ref<Value()> getDevice) {
  for (Block &block : region) {
    // Collect nested regions first to avoid iterator invalidation.
    SmallVector<Region *> nested;
    for (Operation &op : block)
      for (Region &r : op.getRegions())
        nested.push_back(&r);

    scheduleBlock(block, getStream, aa, rewriter, getDevice);
    for (Region *r : nested)
      scheduleRegion(*r, getStream, aa, rewriter, getDevice);
  }
}

namespace {

class CUDAScheduleAsyncPass
    : public cuda::impl::CUDAScheduleAsyncPassBase<CUDAScheduleAsyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    if (!func.getBody().hasOneBlock()) {
      func.emitError("function has multiple blocks");
      return signalPassFailure();
    }

    // Materialize a stream pool at the start of the entry block.
    Block &entry = func.front();
    OpBuilder b(&entry, entry.begin());
    Location loc = func.getLoc();
    AliasAnalysis aa = createRestrictAwareAliasAnalysis(func);
    IRRewriter rewriter(func.getContext());

    Value device{};
    auto lazilyCreateDevice = [&]() -> Value {
      if (device)
        return device;
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&entry);
      Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(0));
      device = b.create<cuda::GetProgramDeviceOp>(loc, zeroI32);
      return device;
    };

    SmallVector<Value> streams;
    auto lazilyGetStream = [&](int64_t i) -> Value {
      OpBuilder::InsertionGuard g(rewriter);
      Value device = lazilyCreateDevice();
      rewriter.setInsertionPointAfterValue(device);
      if (streams.empty()) {
        Value programStream =
            b.create<cuda::GetGlobalStreamOp>(loc, device, kProgramStreamIndex);
        Value copyStream =
            b.create<cuda::GetGlobalStreamOp>(loc, device, kCopyStreamIndex);
        Value computeStream =
            b.create<cuda::GetGlobalStreamOp>(loc, device, kComputeStreamIndex);
        Value inputsReadyEvent =
            rewriter.create<cuda::EventCreateOnStreamOp>(loc, programStream);
        rewriter.create<cuda::StreamWaitEventOp>(loc, copyStream,
                                                 inputsReadyEvent);
        rewriter.create<cuda::StreamWaitEventOp>(loc, computeStream,
                                                 inputsReadyEvent);
        rewriter.create<cuda::EventReleaseOp>(loc, inputsReadyEvent);
        streams.resize(3, {});
        streams[kProgramStreamIndex] = programStream;
        streams[kCopyStreamIndex] = copyStream;
        streams[kComputeStreamIndex] = computeStream;
      }
      return streams[i];
    };

    // Schedule CUDA commands across streams. Events are created and released
    // within the same block to guarantee proper scoping.
    scheduleRegion(func.getBody(), lazilyGetStream, aa, rewriter,
                   lazilyCreateDevice);

    if (!streams.empty()) {
      rewriter.setInsertionPoint(entry.getTerminator());
      SmallVector<Value> events;
      for (Value st : ArrayRef(streams).drop_front()) {
        assert(device && streams[kProgramStreamIndex] &&
               "main program stream must be created");
        Value event = rewriter.create<cuda::EventCreateOnStreamOp>(loc, st);
        events.push_back(event);
      }
      for (Value e : events) {
        rewriter.create<cuda::StreamWaitEventOp>(
            loc, streams[kProgramStreamIndex], e);
        rewriter.create<cuda::EventReleaseOp>(loc, e);
      }
    }
  }
};

} // namespace
