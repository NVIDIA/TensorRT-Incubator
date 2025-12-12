//===- GenerateSort.cpp ---------------------------------------------------===//
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
/// Implementation of GPU merge sort kernel generation.
///
//===----------------------------------------------------------------------===//

#include "mlir-kernel/Kernel/Transforms/GenerateSort.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/GenerateSortValueWrapper.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::kernel;

/// Helper function to get the size of a type in bytes.
/// Used for type-based tuning of items per thread.
static unsigned getTypeSizeInBytes(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8; // Round up to bytes
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  return 4; // Default to 4 bytes for unknown types
}

/// Copy elements from `source[i] -> dest[storeIndexBase + i * storeStride]` for
/// `i` in [0, sourceSize). A conditional is inserted so that each element is
/// only copied if `storeIndexBase + i < destSize`. It is assumed that
/// `storeIndexBase` is always in-bounds.
static void threadCopy(Context &ctx, int64_t sourceSize,
                       ValueWrapper storeIndexBase, ValueWrapper storeStride,
                       ValueWrapper destSize, MemRefWrapper source,
                       MemRefWrapper dest) {
  // HotPath: If `storeIndexBase+sourceSize < destSize`, we
  // don't need any predication. This is the vast majority of cases across
  // threads in the program, so we emit separate code for this case.
  for (int64_t item = 0; item < sourceSize; ++item) {
    auto itemIdx = ctx.constantI32(item);
    auto globalIdx = storeIndexBase + itemIdx * storeStride;
    auto inBounds = globalIdx < destSize;
    ctx.buildIf(inBounds, [&]() {
      auto sourceValue = source.load(itemIdx);
      dest.store(sourceValue, globalIdx);
    });
  }
}

int64_t MergeSortConfig::getActualItemsPerThread(Type type) const {
  unsigned typeSize = getTypeSizeInBytes(type);
  // Scale items per thread to maintain constant register pressure.
  // Formula: actual_items = min(nominal, max(1, nominal * 4 / sizeof(T)))
  // This follows CUB's Nominal4BItemsToItems strategy.
  int64_t scaled = (itemsPerThread * 4) / static_cast<int64_t>(typeSize);
  return std::max(static_cast<int64_t>(1), std::min(itemsPerThread, scaled));
}

/// Generate merge path computation using ValueWrapper for cleaner syntax
/// This implementation mirrors the CUB algorithm directly
static ValueWrapper generateMergePath(Context &ctx, MemRefWrapper keys1,
                                      MemRefWrapper keys2,
                                      ValueWrapper keys1Count,
                                      ValueWrapper keys2Count,
                                      ValueWrapper diag, Type elementType) {
  // MergePath binary search implementation (exact CUB algorithm)
  OpBuilder &builder = keys1Count.getBuilder();
  Location loc = keys1Count.getLoc();

  auto zero = ctx.constantI32(0);
  auto one = ctx.constantI32(1);
  auto two = ctx.constantI32(2);

  // CUB line 64: keys1_begin = diag < keys2_count ? 0 : diag - keys2_count
  auto keys1_begin = (diag < keys2Count).select(zero, diag - keys2Count);

  // CUB line 65: keys1_end = min(diag, keys1_count)
  auto keys1_end = ctx.min(diag, keys1Count);

  // CUB line 67: while (keys1_begin < keys1_end)
  // We need to use scf::WhileOp directly to track both bounds

  auto whileOp = builder.create<scf::WhileOp>(
      loc,
      TypeRange{keys1_begin.getValue().getType(),
                keys1_end.getValue().getType()},
      ValueRange{keys1_begin.getValue(), keys1_end.getValue()});

  // Before region
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *before = &whileOp.getBefore().emplaceBlock();
    before->addArgument(keys1_begin.getValue().getType(), loc);
    before->addArgument(keys1_end.getValue().getType(), loc);
    builder.setInsertionPointToStart(before);

    ValueWrapper current_begin(builder, loc, before->getArgument(0));
    ValueWrapper current_end(builder, loc, before->getArgument(1));

    // Condition: keys1_begin < keys1_end
    auto condition = current_begin < current_end;
    builder.create<scf::ConditionOp>(
        loc, condition.getValue(),
        ValueRange{current_begin.getValue(), current_end.getValue()});
  }

  // After region
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *after = &whileOp.getAfter().emplaceBlock();
    after->addArgument(keys1_begin.getValue().getType(), loc);
    after->addArgument(keys1_end.getValue().getType(), loc);
    builder.setInsertionPointToStart(after);

    ValueWrapper current_begin(builder, loc, after->getArgument(0));
    ValueWrapper current_end(builder, loc, after->getArgument(1));

    // CUB line 69: mid = MidPoint(keys1_begin, keys1_end)
    auto mid = (current_begin + current_end) / two;

    // CUB line 71: key1 = keys1[mid]
    auto key1 = keys1.load(mid);

    // CUB line 72: key2 = keys2[diag - 1 - mid]
    auto key2 = keys2.load(diag - one - mid);

    // CUB line 73: if (binary_pred(key2, key1))
    auto key2_less_key1 = ctx.compareKeys(key2, key1, elementType);

    // Lines 74-79: Update bounds based on comparison (EXACT CUB algorithm)
    // if (key2 < key1) then keys1_end = mid
    // else keys1_begin = mid + 1
    auto new_begin = key2_less_key1.select(current_begin, mid + one);
    auto new_end = key2_less_key1.select(mid, current_end);

    builder.create<scf::YieldOp>(
        loc, ValueRange{new_begin.getValue(), new_end.getValue()});
  }

  // Return the final keys1_begin value (first result)
  return ValueWrapper(builder, loc, whileOp.getResult(0));
}

/// Generate serial merge operation using ValueWrapper
/// Returns indices array if needed for value gathering
static void generateSerialMerge(
    Context &ctx, MemRefWrapper sharedKeys, ValueWrapper keys1Beg,
    ValueWrapper keys2Beg, ValueWrapper keys1Count, ValueWrapper keys2Count,
    MemRefWrapper localKeys,
    MemRefWrapper localIndices, // NEW: indices for value gathering
    Type keyType, int itemsPerThread) {
  auto one = ctx.constantI32(1);

  auto keys1End = keys1Beg + keys1Count;
  auto keys2End = keys2Beg + keys2Count;

  // Serial merge loop - merge itemsPerThread items
  SmallVector<ValueWrapper> mergeState = {keys1Beg, keys2Beg};
  for (int64_t i = 0; i < itemsPerThread; i++) {
    auto item = ctx.constant(i);
    auto currentKeys1Pos = mergeState[0];
    auto currentKeys2Pos = mergeState[1];

    // Check if sequences are exhausted
    auto keys1Valid = currentKeys1Pos < keys1End;
    auto keys2Valid = currentKeys2Pos < keys2End;

    // Load keys conditionally to avoid out-of-bounds access
    // Use sentinel value for exhausted sequences (INT_MAX for integers,
    // +inf for floats)
    auto sentinelKey = ctx.sentinelValue(keyType);
    auto key1 =
        keys1Valid.select(sharedKeys.load(currentKeys1Pos), sentinelKey);
    auto key2 =
        keys2Valid.select(sharedKeys.load(currentKeys2Pos), sentinelKey);

    // CUB logic: take from keys2 if (keys2 valid) AND (keys1 exhausted OR
    // key2 < key1) With INT_MAX sentinel: exhausted sequence will never be
    // selected over valid keys
    auto key2Less = ctx.compareKeys(key2, key1, keyType);
    auto takeFromKeys2 = keys2Valid & ((!keys1Valid) | key2Less);

    // Select key and update positions
    auto selectedKey = takeFromKeys2.select(key2, key1);
    auto nextKeys1Pos =
        takeFromKeys2.select(currentKeys1Pos, currentKeys1Pos + one);
    auto nextKeys2Pos =
        takeFromKeys2.select(currentKeys2Pos + one, currentKeys2Pos);

    // Store the selected key
    localKeys.store(selectedKey, item);

    // Store the index where this key came from (for value gathering later)
    auto selectedIndex = takeFromKeys2.select(currentKeys2Pos, currentKeys1Pos);
    localIndices.store(selectedIndex, item);

    mergeState[0] = nextKeys1Pos;
    mergeState[1] = nextKeys2Pos;
  }
}

FailureOr<MergeSortKernelResult>
MergeSortKernelGenerator::createMergeSortKernels(
    OpBuilder &builder, Location loc, Type keyType, Type valueType,
    ModuleOp module, gpu::GPUModuleOp gpuModule,
    SymbolTableCollection &symbolTables, const MergeSortConfig &config) {
  OpBuilder::InsertionGuard g(builder);

  builder.setInsertionPointToStart(gpuModule.getBody());

  // Create individual kernels
  FailureOr<func::FuncOp> blockSortResult = createBlockSortKernel(
      builder, loc, gpuModule, keyType, valueType, config);
  if (failed(blockSortResult))
    return failure();

  FailureOr<func::FuncOp> partitionResult = createPartitionKernel(
      builder, loc, gpuModule, keyType, valueType, config);
  if (failed(partitionResult))
    return failure();

  FailureOr<func::FuncOp> mergeResult =
      createMergeKernel(builder, loc, gpuModule, keyType, valueType, config);
  if (failed(mergeResult))
    return failure();

  // Create dispatch function
  builder.setInsertionPointAfter(gpuModule);
  FailureOr<func::FuncOp> dispatchResult = createDispatchFunction(
      builder, loc, module, gpuModule, *blockSortResult, *partitionResult,
      *mergeResult, keyType, valueType, symbolTables, config);
  if (failed(dispatchResult))
    return failure();

  return MergeSortKernelResult{*dispatchResult, *blockSortResult,
                               *partitionResult, *mergeResult};
}

/// Helper to create the GPU module
FailureOr<gpu::GPUModuleOp>
MergeSortKernelGenerator::createGPUModule(ModuleOp parentModule,
                                          StringRef moduleName) {
  OpBuilder builder(parentModule.getContext());
  builder.setInsertionPointToStart(parentModule.getBody());

  Location loc = parentModule.getLoc();
  auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, moduleName);

  // Add the required gpu_module_kind attribute
  auto moduleKindAttr =
      kernel::DefaultGPUModuleKindAttr::get(builder.getContext());
  gpuModule->setAttr(kernel::KernelDialect::getGpuModuleKindAttrName(),
                     moduleKindAttr);

  return gpuModule;
}

/// Creates the block sort kernel using ValueWrapper
FailureOr<func::FuncOp> MergeSortKernelGenerator::createBlockSortKernel(
    OpBuilder &builder, Location loc, gpu::GPUModuleOp gpuModule, Type keyType,
    Type valueType, const MergeSortConfig &config) {

  OpBuilder::InsertionGuard g(builder);
  Context ctx(builder, loc);

  auto i1Type = builder.getI1Type();

  // Create dynamic memref types with strided layout (no memory space)
  auto stridedLayout = StridedLayoutAttr::get(
      builder.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
  auto keyMemRefType =
      MemRefType::get({ShapedType::kDynamic}, keyType, stridedLayout);
  auto valueMemRefType =
      config.keysOnly
          ? Type{}
          : MemRefType::get({ShapedType::kDynamic}, valueType, stridedLayout);

  // Function signature: (keys, values?, count, temp_keys, temp_values?, ping)
  SmallVector<Type> argTypes = {keyMemRefType,
                                keyMemRefType}; // input keys, temp keys
  if (!config.keysOnly)
    argTypes.insert(argTypes.end(), {valueMemRefType,
                                     valueMemRefType}); // input and temp values
  argTypes.push_back(i1Type);                           // ping

  auto funcType = builder.getFunctionType(argTypes, {});
  auto func = builder.create<func::FuncOp>(loc, "block_sort_kernel", funcType);
  func->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                builder.getUnitAttr());

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get arguments using ValueWrapper
  unsigned argIdx = 0;
  MemRefWrapper keysIn(builder, loc, entryBlock->getArgument(argIdx++));
  MemRefWrapper tempKeys(builder, loc, entryBlock->getArgument(argIdx++));

  MemRefWrapper valuesIn(builder, loc,
                         config.keysOnly ? Value{}
                                         : entryBlock->getArgument(argIdx++));
  MemRefWrapper tempValues(builder, loc,
                           config.keysOnly ? Value{}
                                           : entryBlock->getArgument(argIdx++));

  ValueWrapper ping(builder, loc, entryBlock->getArgument(argIdx));

  // Get count from memref dimension
  Type i32Type = builder.getI32Type();
  auto c0 = ctx.constantI32(0);
  ValueWrapper count(
      builder, loc,
      builder.create<memref::DimOp>(loc, keysIn.getValue(), c0.toIndex()));
  count = count.toType(i32Type);

  // Get thread and block dimensions
  auto blockId = ctx.blockId();
  auto threadId = ctx.threadId();
  auto blockDim = ctx.blockDim();

  // Calculate tile parameters
  auto itemsPerThread = ctx.constantI32(config.itemsPerThread);
  auto itemsPerTile = blockDim * itemsPerThread;
  auto tileBase = blockId * itemsPerTile;

  // Create local arrays for thread's items
  auto localKeys = ctx.allocaStatic(keyType, config.itemsPerThread);
  auto localIndices =
      ctx.allocaStatic(builder.getI32Type(), config.itemsPerThread);
  auto localValues = config.keysOnly
                         ? MemRefWrapper(builder, loc, Value{})
                         : ctx.allocaStatic(valueType, config.itemsPerThread);

  // Allocate shared memory for the key/value tiles.
  auto sharedKeys =
      ctx.allocShared(keyType, config.blockThreads * config.itemsPerThread + 1);
  auto sharedValues =
      config.keysOnly
          ? MemRefWrapper(builder, loc, Value{})
          : ctx.allocShared(valueType,
                            config.blockThreads * config.itemsPerThread + 1);

  // Calculate valid items in this tile (bound by count)
  auto tileSize =
      ctx.constantInt(config.blockThreads * config.itemsPerThread, i32Type);
  auto validItems = ctx.min(count - tileBase, tileSize);

  // Load the entire tile from global to shared.
  auto threadBase = tileBase + threadId;

  // Initialize OOB elements to sentinel value (like CUB does)
  // CUB uses BlockLoad with an OOB default parameter: Load(..., num_remaining,
  // *(keys_in + tile_base)) This ensures out-of-bounds elements are initialized
  // rather than garbage We use INT_MAX/+inf as sentinel so they sort to the end
  auto sentinelKey = ctx.sentinelValue(keyType);

  for (int64_t item = 0; item < config.itemsPerThread; ++item) {
    auto itemIdx = ctx.constantI32(item);
    auto globalIdx = threadBase + itemIdx * blockDim;
    auto sharedIdx = threadId + itemIdx * blockDim;

    // Bounds check and load, or initialize with sentinel
    auto inBounds = globalIdx < count;
    ctx.buildIfElse(
        inBounds,
        [&]() {
          auto keyValue = keysIn.load(globalIdx);
          sharedKeys.store(keyValue, sharedIdx);

          if (!config.keysOnly) {
            auto valueValue = valuesIn.load(globalIdx);
            sharedValues.store(valueValue, sharedIdx);
          }
        },
        [&]() {
          // Initialize OOB slots with sentinel so they sort to the end
          sharedKeys.store(sentinelKey, sharedIdx);
        });
  }

  // Block synchronization
  ctx.syncThreads();

  // Load the thread tile into local memory.
  for (int64_t item = 0; item < config.itemsPerThread; ++item) {
    auto itemIdx = ctx.constantInt(item, i32Type);
    auto sharedIdx = (threadId * itemsPerThread + itemIdx).toIndex();
    localKeys.store(sharedKeys.load(sharedIdx), itemIdx);
    if (!config.keysOnly) {
      localValues.store(sharedValues.load(sharedIdx), itemIdx);
    }
  }

  // Implement stable odd-even sort (matches CUB's StableOddEvenSort)
  for (int64_t i = 0; i < config.itemsPerThread; i++) {
    for (int64_t j = i & 1; j < config.itemsPerThread - 1; j += 2) {

      // Calculate starting point: j = 1 & i (odd-even alternation)
      auto jIdx = ctx.constant(j);
      auto jPlus1 = ctx.constant(j + 1);

      // Load and compare adjacent elements
      auto key1 = localKeys.load(jIdx);
      auto key2 = localKeys.load(jPlus1);
      auto shouldSwap = ctx.compareKeys(key2, key1, keyType);
      auto newKey1 = shouldSwap.select(key2, key1);
      auto newKey2 = shouldSwap.select(key1, key2);

      // Swap keys
      localKeys.store(newKey1, jIdx);
      localKeys.store(newKey2, jPlus1);

      // Swap values if present
      if (!config.keysOnly) {
        auto val1 = localValues.load(jIdx);
        auto val2 = localValues.load(jPlus1);
        auto newVal1 = shouldSwap.select(val2, val1);
        auto newVal2 = shouldSwap.select(val1, val2);
        localValues.store(newVal1, jIdx);
        localValues.store(newVal2, jPlus1);
      }
    }
  }

  // Manually generate merge iterations to avoid loop variable issues
  // CUB: for (int target = 2; target <= NUM_THREADS; target *= 2)
  auto generateMergeIteration = [&](int64_t targetMergeSizeVal,
                                    int64_t mergedThreadsVal) {
    // Create constants directly as index type
    auto targetMergedThreadsIdx = ctx.constantInt(targetMergeSizeVal, i32Type);
    auto mergedThreadsIdx = ctx.constantInt(mergedThreadsVal, i32Type);

    auto mask = targetMergedThreadsIdx - ctx.constantInt(1, i32Type);

    // Synchronize before storing to shared memory
    ctx.syncThreads();

    // Store local keys to shared memory (unrolled for coalescing)
    for (int64_t item = 0; item < config.itemsPerThread; ++item) {
      auto itemIdx = ctx.constantInt(item, i32Type);
      auto sharedIdx = (threadId * itemsPerThread) + itemIdx;
      auto localKey = localKeys.load(itemIdx);
      sharedKeys.store(localKey, sharedIdx.toIndex());

      if (!config.keysOnly) {
        auto localValue = localValues.load(itemIdx);
        sharedValues.store(localValue, sharedIdx.toIndex());
      }
    }

    // Synchronize after storing to shared memory
    ctx.syncThreads();

    // Compute merge parameters (analogous to CUB's MergePath calculation)
    auto maskInverted = ~mask;
    auto firstThreadInGroup = maskInverted & threadId;
    // Compute size using the index type mergedThreads
    auto start = ctx.constantI32(config.itemsPerThread) * firstThreadInGroup;
    auto size = ctx.constantI32(config.itemsPerThread) * mergedThreadsIdx;

    auto threadInGroup = mask & threadId;
    auto diagUnbounded = ctx.constantI32(config.itemsPerThread) * threadInGroup;
    auto diag = ctx.min(validItems, diagUnbounded);

    // Calculate keys ranges for merge path (bounded by validItems like CUB)
    auto startBounded = ctx.min(validItems, start);
    auto keys1Beg = startBounded;
    auto keys1End = ctx.min(validItems, keys1Beg + size);
    auto keys2Beg = keys1End;
    auto keys2End = ctx.min(validItems, keys2Beg + size);

    auto keys1Count = keys1End - keys1Beg;
    auto keys2Count = keys2End - keys2Beg;

    // Compute merge path partition for this thread
    auto keys1SharedPtr =
        sharedKeys.subview(keys1Beg, keys1Count, ctx.constantI32(1));
    auto keys2SharedPtr =
        sharedKeys.subview(keys2Beg, keys2Count, ctx.constantI32(1));

    auto partitionDiag =
        generateMergePath(ctx, keys1SharedPtr, keys2SharedPtr, keys1Count,
                          keys2Count, diag, keyType);

    // Calculate local merge ranges
    auto keys1BegLocal = keys1Beg + partitionDiag;
    auto keys2BegLocal = keys2Beg + (diag - partitionDiag);

    auto keys1CountLocal = keys1End - keys1BegLocal;
    auto keys2CountLocal = keys2End - keys2BegLocal;

    // Serial merge within thread
    generateSerialMerge(ctx, sharedKeys, keys1BegLocal, keys2BegLocal,
                        keys1CountLocal, keys2CountLocal, localKeys,
                        localIndices, keyType, config.itemsPerThread);

    // CUB: Gather values INSIDE each merge iteration using the indices from
    // THIS merge
    if (!config.keysOnly) {
      ctx.syncThreads();

      // Store values to shared memory (unrolled for coalescing)
      for (int64_t item = 0; item < config.itemsPerThread; ++item) {
        auto itemIdx = ctx.constantI32(item);
        auto sharedIdx = (threadId * itemsPerThread) + itemIdx;
        sharedValues.store(localValues.load(itemIdx), sharedIdx);
      }

      ctx.syncThreads();

      // Gather values from shared memory using indices from THIS merge
      // iteration (unrolled to expose all loads to optimizer)
      for (int64_t item = 0; item < config.itemsPerThread; ++item) {
        auto itemIdx = ctx.constantI32(item);
        auto index = localIndices.load(itemIdx);
        auto gatheredValue = sharedValues.load(index);
        localValues.store(gatheredValue, itemIdx);
      }
    }
  }; // End of generateMergeIteration lambda

  // Generate merge iterations dynamically for all powers of 2 up to
  // blockThreads We need iterations for 2, 4, 8, 16, ... up to blockThreads
  // Each iteration merges sorted sequences of size (target/2) into size target
  for (int64_t targetSize = 2; targetSize <= config.blockThreads;
       targetSize *= 2) {
    int64_t mergedSize = targetSize / 2;
    generateMergeIteration(targetSize, mergedSize);
  }

  // Final synchronization before storing to global memory
  ctx.syncThreads();

  // Store results to shared so that we can re-load for coalesced global stores.
  auto one = ctx.constantI32(1);
  threadCopy(ctx, config.itemsPerThread, threadId * itemsPerThread, one, count,
             localKeys, sharedKeys);
  if (!config.keysOnly) {
    threadCopy(ctx, config.itemsPerThread, threadId * itemsPerThread, one,
               count, localValues, sharedValues);
  }
  ctx.syncThreads();

  // Re-load results from shared with a coalesced layout and store to global.
  // Choose output buffer based on ping.
  // ping=true writes to original buffer, ping=false writes to temp
  // To ensure coallesced loads, we load and store in a block-cyclic (e.g.
  // striped) manner across threads:
  // global[tileId + threadId + ThreadsPerBlock * idx] <= shared[threadId +
  // ThreadsPerBlock * idx] for all idx in [0, itemsPerThread).
  auto pingCond = ConditionWrapper(builder, loc, ping.getValue());
  auto numThreads = ctx.constantI32(config.blockThreads);
  MemRefWrapper keysSharedSource =
      sharedKeys.subview(threadId, itemsPerThread, numThreads);
  MemRefWrapper valuesSharedSource =
      config.keysOnly
          ? MemRefWrapper(builder, loc, Value{})
          : sharedValues.subview(threadId, itemsPerThread, numThreads);
  auto storeIndexBase = tileBase + threadId;
  ctx.buildIfElse(
      pingCond,
      [&]() { // ping = true: write to original buffer
        threadCopy(ctx, config.itemsPerThread, storeIndexBase, numThreads,
                   count, keysSharedSource, keysIn);
        if (!config.keysOnly) {
          threadCopy(ctx, config.itemsPerThread, storeIndexBase, numThreads,
                     count, valuesSharedSource, valuesIn);
        }
      },
      [&]() { // ping = false: write to temp buffer
        threadCopy(ctx, config.itemsPerThread, storeIndexBase, numThreads,
                   count, keysSharedSource, tempKeys);
        if (!config.keysOnly) {
          threadCopy(ctx, config.itemsPerThread, storeIndexBase, numThreads,
                     count, valuesSharedSource, tempValues);
        }
      });

  builder.create<func::ReturnOp>(loc);
  return func;
}

/// Creates the partition kernel using ValueWrapper
FailureOr<func::FuncOp> MergeSortKernelGenerator::createPartitionKernel(
    OpBuilder &builder, Location loc, gpu::GPUModuleOp gpuModule, Type keyType,
    Type valueType, const MergeSortConfig &config) {

  OpBuilder::InsertionGuard g(builder);
  Context ctx(builder, loc);

  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  // Create dynamic memref types with strided layout (no memory space)
  auto stridedLayout = StridedLayoutAttr::get(
      builder.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
  auto keyMemRefType =
      MemRefType::get({ShapedType::kDynamic}, keyType, stridedLayout);
  auto partitionMemRefType =
      MemRefType::get({ShapedType::kDynamic}, i32Type, stridedLayout);

  // Function signature
  SmallVector<Type> argTypes = {
      keyMemRefType, // keys_ping
      keyMemRefType, // keys_pong
  };

  // Note: partition kernel doesn't need values/tempValues for computation
  // but they are passed through for consistency with the call site
  if (!config.keysOnly) {
    auto valueMemRefType =
        MemRefType::get({ShapedType::kDynamic}, valueType, stridedLayout);
    argTypes.push_back(valueMemRefType); // values_ping
    argTypes.push_back(valueMemRefType); // values_pong
  }

  argTypes.insert(argTypes.end(), {
                                      i32Type,             // num_partitions
                                      partitionMemRefType, // merge_partitions
                                      i1Type,              // ping
                                      i32Type, // target_merged_tiles_number
                                      i32Type  // items_per_tile
                                  });

  auto funcType = builder.getFunctionType(argTypes, {});
  auto func = builder.create<func::FuncOp>(loc, "partition_kernel", funcType);
  func->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                builder.getUnitAttr());

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments wrapped
  unsigned argIdx = 0;
  MemRefWrapper keysPing(builder, loc, entryBlock->getArgument(argIdx++));
  MemRefWrapper keysPong(builder, loc, entryBlock->getArgument(argIdx++));

  // Skip values/tempValues if present (not used in partition kernel)
  if (!config.keysOnly)
    argIdx += 2;

  ValueWrapper numPartitions(builder, loc, entryBlock->getArgument(argIdx++));
  MemRefWrapper mergePartitions(builder, loc,
                                entryBlock->getArgument(argIdx++));
  ValueWrapper ping(builder, loc, entryBlock->getArgument(argIdx++));
  ValueWrapper targetMergedTiles(builder, loc,
                                 entryBlock->getArgument(argIdx++));
  ValueWrapper itemsPerTile(builder, loc, entryBlock->getArgument(argIdx++));

  // Get count from memref dimension
  auto c0 = ctx.constantI32(0);
  auto count = ValueWrapper(
      builder, loc,
      builder.create<memref::DimOp>(loc, keysPing.getValue(), c0.toIndex()));
  count = count.toType(i32Type);

  // Get thread ID - this is the partition index
  auto partitionIdx = ctx.blockId() * ctx.blockDim() + ctx.threadId();

  // Check bounds - early exit if out of range
  auto inBounds = partitionIdx < numPartitions;
  ctx.buildIf(inBounds, [&]() {
    // CUB's exact algorithm implementation
    auto mergedTiles = targetMergedTiles / ctx.constantI32(2);
    auto mask = targetMergedTiles - ctx.constantI32(1);

    // Calculate segment boundaries (exactly like CUB AgentPartition::Process())
    auto maskInverted = ~mask;
    auto list = maskInverted & partitionIdx;
    auto start = itemsPerTile * list;
    auto size = itemsPerTile * mergedTiles;

    auto localTileIdx = mask & partitionIdx;

    // Calculate keys ranges with proper bounds checking
    auto keys1Beg = ctx.min(count, start);
    auto keys1End = ctx.min(count, start + size);
    auto keys2Beg = keys1End;
    auto keys2End = ctx.min(count, keys2Beg + size);

    // Handle the special case of the last partition (one-past-the-end marker)
    auto numPartitionsMinus1 = numPartitions - ctx.constantI32(1);
    auto isLastPartition = partitionIdx == numPartitionsMinus1;

    ctx.buildIfElse(
        isLastPartition,
        [&]() { mergePartitions.store(keys1End, partitionIdx); },
        [&]() {
          // Calculate partition point
          auto keys1Count = keys1End - keys1Beg;
          auto keys2Count = keys2End - keys2Beg;
          auto totalCount = keys2End - keys1Beg;
          auto partitionAt = ctx.min(totalCount, itemsPerTile * localTileIdx);

          // Select keys buffer based on ping and create offset pointers
          auto pingCond = ConditionWrapper(builder, loc, ping.getValue());
          Value selectedKeysBuffer =
              pingCond
                  .select(ValueWrapper(builder, loc, keysPing.getValue()),
                          ValueWrapper(builder, loc, keysPong.getValue()))
                  .getValue();
          MemRefWrapper keysPtr(builder, loc, selectedKeysBuffer);

          auto one = ctx.constantI32(1);
          auto keys1SubPtr = keysPtr.subview(keys1Beg, keys1Count, one);
          auto keys2SubPtr = keysPtr.subview(keys2Beg, keys2Count, one);

          auto partitionDiag =
              generateMergePath(ctx, keys1SubPtr, keys2SubPtr, keys1Count,
                                keys2Count, partitionAt, keyType);
          auto result = keys1Beg + partitionDiag;

          // Store partition result
          mergePartitions.store(result, partitionIdx);
        });
  });

  builder.create<func::ReturnOp>(loc);
  return func;
}

/// Creates the merge kernel using ValueWrapper
FailureOr<func::FuncOp> MergeSortKernelGenerator::createMergeKernel(
    OpBuilder &builder, Location loc, gpu::GPUModuleOp gpuModule, Type keyType,
    Type valueType, const MergeSortConfig &config) {

  OpBuilder::InsertionGuard g(builder);
  Context ctx(builder, loc);

  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  // Create dynamic memref types with strided layout (no memory space)
  auto stridedLayout = StridedLayoutAttr::get(
      builder.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
  auto keyMemRefType =
      MemRefType::get({ShapedType::kDynamic}, keyType, stridedLayout);
  auto valueMemRefType =
      config.keysOnly
          ? Type{}
          : MemRefType::get({ShapedType::kDynamic}, valueType, stridedLayout);
  auto partitionMemRefType =
      MemRefType::get({ShapedType::kDynamic}, i32Type, stridedLayout);

  // Function signature
  SmallVector<Type> argTypes = {
      keyMemRefType, // keys_ping
      keyMemRefType, // keys_pong
  };
  if (!config.keysOnly) {
    argTypes.push_back(valueMemRefType); // values_ping
    argTypes.push_back(valueMemRefType); // values_pong
  }
  argTypes.insert(argTypes.end(), {
                                      partitionMemRefType, // merge_partitions
                                      i1Type,              // ping
                                      i32Type // target_merged_tiles_number
                                  });

  auto funcType = builder.getFunctionType(argTypes, {});
  auto func = builder.create<func::FuncOp>(loc, "merge_kernel", funcType);
  func->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                builder.getUnitAttr());

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments wrapped
  unsigned argIdx = 0;
  MemRefWrapper keysPing(builder, loc, entryBlock->getArgument(argIdx++));
  MemRefWrapper keysPong(builder, loc, entryBlock->getArgument(argIdx++));
  MemRefWrapper valuesPing(builder, loc,
                           config.keysOnly ? Value{}
                                           : entryBlock->getArgument(argIdx++));
  MemRefWrapper valuesPong(builder, loc,
                           config.keysOnly ? Value{}
                                           : entryBlock->getArgument(argIdx++));
  MemRefWrapper mergePartitions(builder, loc,
                                entryBlock->getArgument(argIdx++));
  ValueWrapper ping(builder, loc, entryBlock->getArgument(argIdx++));
  ValueWrapper targetMergedTiles(builder, loc, entryBlock->getArgument(argIdx));

  // Get count from memref dimension
  auto c0 = ctx.constantI32(0);
  auto count = ValueWrapper(
      builder, loc,
      builder.create<memref::DimOp>(loc, keysPing.getValue(), c0.toIndex()));
  count = count.toType(i32Type);

  // Get thread and block information
  auto blockId = ctx.blockId();
  auto threadId = ctx.threadId();
  auto blockDim = ctx.blockDim();

  auto itemsPerThread = ctx.constantI32(config.itemsPerThread);
  auto itemsPerTile = blockDim * itemsPerThread;
  auto tileBase = blockId * itemsPerTile;

  // Debug prints removed - vector.print not supported by mlir-tensorrt-compiler

  // Get partition boundaries for this tile
  auto partitionBeg = mergePartitions.load(blockId);
  auto partitionEnd = mergePartitions.load(blockId + ctx.constantI32(1));

  // Calculate merge parameters (exactly like CUB AgentMerge)
  auto mergedTiles = targetMergedTiles / ctx.constantI32(2);
  auto mask = targetMergedTiles - ctx.constantI32(1);

  auto maskInverted = ~mask;
  auto list = maskInverted & blockId;
  auto start = itemsPerTile * list;
  auto size = itemsPerTile * mergedTiles;

  // Calculate diagonal position (CUB line 523)
  auto diag = itemsPerTile * blockId - start;

  // Calculate keys1 ranges (CUB lines 525-526)
  auto keys1Beg = partitionBeg - start;
  auto keys1End = partitionEnd - start;

  // Calculate max_keys2 (CUB lines 528-529)
  auto keysEndDistFromStart = count - start;
  auto sizeCompare = keysEndDistFromStart > size;
  auto max_keys2 =
      sizeCompare.select(keysEndDistFromStart - size, ctx.constantI32(0));

  // Calculate keys2 ranges (CUB lines 535-538)
  auto keys2BegUnbounded = diag - keys1Beg;
  auto keys2Beg = ctx.min(max_keys2, keys2BegUnbounded);

  auto diagPlusTile = diag + itemsPerTile;
  auto keys2EndUnbounded = diagPlusTile - keys1End;
  auto keys2End = ctx.min(max_keys2, keys2EndUnbounded);

  // Handle last tile in group (CUB lines 541-545)
  auto maskedBlockId = mask & blockId;
  auto isLastTileInGroup = mask == maskedBlockId;
  auto keys1EndLastTile = ctx.min(count - start, size);
  auto keys2EndLastTile = ctx.min(max_keys2, size);
  keys1End = isLastTileInGroup.select(keys1EndLastTile, keys1End);
  keys2End = isLastTileInGroup.select(keys2EndLastTile, keys2End);

  // Create shared memory for the merge operation
  auto sharedKeys =
      ctx.allocShared(keyType, config.blockThreads * config.itemsPerThread + 1);
  auto sharedValues =
      config.keysOnly
          ? MemRefWrapper(builder, loc, Value{})
          : ctx.allocShared(valueType,
                            config.blockThreads * config.itemsPerThread + 1);

  // Number of keys per type - ensure they don't exceed tile size
  auto numKeys1 = keys1End - keys1Beg;
  auto numKeys2 = keys2End - keys2Beg;

  // Cap to tile size to ensure we don't exceed shared memory bounds
  auto tileSize = ctx.constantI32(config.blockThreads * config.itemsPerThread);
  numKeys1 = ctx.min(numKeys1, tileSize);
  numKeys2 = ctx.min(numKeys2, tileSize);

  // Also ensure the total doesn't exceed tile size
  auto totalKeys = numKeys1 + numKeys2;
  auto cappedTotal = ctx.min(totalKeys, tileSize);

  // If total was capped, adjust numKeys2
  auto cappedKeys2 = cappedTotal - numKeys1;
  numKeys2 = ctx.min(numKeys2, cappedKeys2);

  // Load keys and values into shared memory with COALESCED global loads
  // Strategy: Load in striped pattern from global, store in blocked pattern to
  // shared This ensures coalesced global memory access regardless of partition
  // boundaries
  auto zero = ctx.constantI32(0);
  auto one = ctx.constantI32(1);

  // totalKeys already calculated above

  // Load keys1 range with coalesced access
  auto keys1Start = start + keys1Beg;
  for (int64_t item = 0; item < config.itemsPerThread; ++item) {
    auto itemIdx = ctx.constantI32(item);
    // Striped global load: base + threadId + blockDim * item
    auto globalIdx = keys1Start + threadId + blockDim * itemIdx;
    auto sharedIdx = threadId + blockDim * itemIdx;

    auto inBounds = (sharedIdx < numKeys1) & (globalIdx < count);
    ctx.buildIf(inBounds, [&]() {
      auto pingCond = ConditionWrapper(builder, loc, ping.getValue());
      ctx.buildIfElse(
          pingCond,
          [&]() {
            auto keyVal = keysPing.load(globalIdx);
            sharedKeys.store(keyVal, sharedIdx);
            if (!config.keysOnly) {
              auto valVal = valuesPing.load(globalIdx);
              sharedValues.store(valVal, sharedIdx);
            }
          },
          [&]() {
            auto keyVal = keysPong.load(globalIdx);
            sharedKeys.store(keyVal, sharedIdx);
            if (!config.keysOnly) {
              auto valVal = valuesPong.load(globalIdx);
              sharedValues.store(valVal, sharedIdx);
            }
          });
    });
  }

  // Load keys2 range with coalesced access
  auto keys2Start = start + size + keys2Beg;
  for (int64_t item = 0; item < config.itemsPerThread; ++item) {
    auto itemIdx = ctx.constantI32(item);
    // Striped global load: base + threadId + blockDim * item
    auto globalIdx = keys2Start + threadId + blockDim * itemIdx;
    auto sharedIdx = numKeys1 + threadId + blockDim * itemIdx;

    auto inBounds = (sharedIdx < totalKeys) & (globalIdx < count);
    ctx.buildIf(inBounds, [&]() {
      auto pingCond = ConditionWrapper(builder, loc, ping.getValue());
      ctx.buildIfElse(
          pingCond,
          [&]() {
            auto keyVal = keysPing.load(globalIdx);
            sharedKeys.store(keyVal, sharedIdx);
            if (!config.keysOnly) {
              auto valVal = valuesPing.load(globalIdx);
              sharedValues.store(valVal, sharedIdx);
            }
          },
          [&]() {
            auto keyVal = keysPong.load(globalIdx);
            sharedKeys.store(keyVal, sharedIdx);
            if (!config.keysOnly) {
              auto valVal = valuesPong.load(globalIdx);
              sharedValues.store(valVal, sharedIdx);
            }
          });
    });
  }

  // Synchronize after loading
  ctx.syncThreads();

  // Compute merge path for this thread
  auto diag0Local = ctx.min(numKeys1 + numKeys2, itemsPerThread * threadId);

  // Find merge path partition for this thread
  auto keys1SharedPtr = sharedKeys.subview(zero, numKeys1, one);
  auto keys2SharedPtr = sharedKeys.subview(numKeys1, numKeys2, one);

  auto keys1BegLocal =
      generateMergePath(ctx, keys1SharedPtr, keys2SharedPtr, numKeys1, numKeys2,
                        diag0Local, keyType);
  auto keys1EndLocal = numKeys1;
  auto keys2BegLocal = diag0Local - keys1BegLocal;
  auto keys2EndLocal = numKeys2;

  auto numKeys1Local = keys1EndLocal - keys1BegLocal;
  auto numKeys2Local = keys2EndLocal - keys2BegLocal;

  // Create local arrays for thread's output
  auto localKeys = ctx.allocaStatic(keyType, config.itemsPerThread);
  auto localIndices =
      ctx.allocaStatic(builder.getI32Type(), config.itemsPerThread);
  auto localValues = config.keysOnly
                         ? MemRefWrapper(builder, loc, Value{})
                         : ctx.allocaStatic(valueType, config.itemsPerThread);

  // Debug prints removed - vector.print not supported by mlir-tensorrt-compiler

  // Perform serial merge
  generateSerialMerge(ctx, sharedKeys, keys1BegLocal, keys2BegLocal + numKeys1,
                      numKeys1Local, numKeys2Local, localKeys, localIndices,
                      keyType, config.itemsPerThread);

  // Gather values using indices if not keys-only
  if (!config.keysOnly) {
    ctx.syncThreads();

    // Gather values from shared memory using the indices from merge
    // UNROLLED: Generate explicit loop iterations so indices are constant
    for (int64_t item = 0; item < config.itemsPerThread; ++item) {
      auto itemIdx = ctx.constantI32(item);
      auto index = localIndices.load(itemIdx);
      auto gatheredValue = sharedValues.load(index);
      localValues.store(gatheredValue, itemIdx);
    }
  }

  // Synchronize before writing
  ctx.syncThreads();

  // Store results to shared so that we can re-load for coalesced global stores.
  // First, store local results to shared memory in BLOCKED layout
  threadCopy(ctx, config.itemsPerThread, threadId * itemsPerThread, one, count,
             localKeys, sharedKeys);
  if (!config.keysOnly) {
    threadCopy(ctx, config.itemsPerThread, threadId * itemsPerThread, one,
               count, localValues, sharedValues);
  }
  ctx.syncThreads();

  // Re-load results from shared with a coalesced (STRIPED) layout and store to
  // global. Choose output buffer based on ping. ping=true writes to pong
  // (temp), ping=false writes to ping (original) To ensure coalesced stores, we
  // load and store in a striped manner: global[tileBase + threadId +
  // blockDim * idx] <= shared[threadId + blockDim * idx]
  auto pingCond = ConditionWrapper(builder, loc, ping.getValue());
  MemRefWrapper keysSharedSource =
      sharedKeys.subview(threadId, itemsPerThread, blockDim);
  MemRefWrapper valuesSharedSource =
      config.keysOnly
          ? MemRefWrapper(builder, loc, Value{})
          : sharedValues.subview(threadId, itemsPerThread, blockDim);
  auto storeIndexBase = tileBase + threadId;

  ctx.buildIfElse(
      pingCond,
      [&]() { // ping = true: write to pong (temp) buffer
        threadCopy(ctx, config.itemsPerThread, storeIndexBase, blockDim, count,
                   keysSharedSource, keysPong);
        if (!config.keysOnly) {
          threadCopy(ctx, config.itemsPerThread, storeIndexBase, blockDim,
                     count, valuesSharedSource, valuesPong);
        }
      },
      [&]() { // ping = false: write to ping (original) buffer
        threadCopy(ctx, config.itemsPerThread, storeIndexBase, blockDim, count,
                   keysSharedSource, keysPing);
        if (!config.keysOnly) {
          threadCopy(ctx, config.itemsPerThread, storeIndexBase, blockDim,
                     count, valuesSharedSource, valuesPing);
        }
      });

  builder.create<func::ReturnOp>(loc);
  return func;
}

/// Creates the main dispatch function using ValueWrapper
FailureOr<func::FuncOp> MergeSortKernelGenerator::createDispatchFunction(
    OpBuilder &builder, Location loc, ModuleOp module,
    gpu::GPUModuleOp gpuModule, func::FuncOp blockSortFunc,
    func::FuncOp partitionFunc, func::FuncOp mergeFunc, Type keyType,
    Type valueType, SymbolTableCollection &symbolTables,
    const MergeSortConfig &config) {

  OpBuilder::InsertionGuard g(builder);
  Context ctx(builder, loc);

  auto i32Type = builder.getI32Type();

  // Create tensor types for dispatch function
  auto keyTensorType = RankedTensorType::get({ShapedType::kDynamic}, keyType);
  auto valueTensorType =
      config.keysOnly
          ? Type{}
          : RankedTensorType::get({ShapedType::kDynamic}, valueType);

  // Function signature: (keys, count, values?) -> (keys, values?)
  SmallVector<Type> argTypes = {keyTensorType, i32Type};
  SmallVector<Type> resultTypes = {keyTensorType};

  if (!config.keysOnly) {
    argTypes.push_back(valueTensorType);    // values
    resultTypes.push_back(valueTensorType); // values result
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto func =
      builder.create<func::FuncOp>(loc, "merge_sort_dispatch", funcType);
  func.setPrivate();

  symbolTables.getSymbolTable(module).insert(func);

  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments
  int argIdx = 0;
  Value keysTensor = entryBlock->getArgument(argIdx++);
  ValueWrapper count(builder, loc, entryBlock->getArgument(argIdx++));

  Value valuesTensor;
  if (!config.keysOnly)
    valuesTensor = entryBlock->getArgument(argIdx++);

  // Create temporary buffers as tensors
  Value tempKeysTensor = builder.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{ShapedType::kDynamic}, keyType,
      ValueRange{count.toIndex()});

  Value tempValuesTensor;
  if (!config.keysOnly) {
    tempValuesTensor = builder.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{ShapedType::kDynamic}, valueType,
        ValueRange{count.toIndex()});
  }

  // Calculate grid dimensions
  auto blockThreads = ctx.constantI32(config.blockThreads);
  auto itemsPerThread = ctx.constantI32(config.itemsPerThread);
  auto itemsPerTile = blockThreads * itemsPerThread;

  // numTiles = (count + itemsPerTile - 1) / itemsPerTile
  auto numTiles = (count + itemsPerTile - ctx.constantI32(1)) / itemsPerTile;

  // Calculate number of merge passes using ceil(log2(numTiles))
  // This matches CUB's ceil_ilog2: 64 - clz(numTiles - 1)
  // For numTiles = 17: ceil(log2(17)) = 5 (not floor(log2(17)) = 4)
  auto numTilesMinus1 = numTiles - ctx.constantI32(1);
  auto clz = ValueWrapper(
      builder, loc,
      builder.create<math::CountLeadingZerosOp>(loc, numTilesMinus1));
  auto bitwidth = ctx.constantI32(32);
  auto numPasses = bitwidth - clz;

  // Calculate numPartitions for partition kernel
  auto numPartitions = numTiles + ctx.constantI32(1);

  // Create partition offsets tensor
  Value partitionOffsets = builder.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{ShapedType::kDynamic}, builder.getI32Type(),
      ValueRange{numPartitions.toIndex()});

  // Initial ping state - result should end up in original buffers
  auto initialPing = numPasses;
  Value two = ctx.constantI32(2).getValue();
  Value rem = builder.create<arith::RemSIOp>(loc, initialPing, two);
  Value pingValue = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, rem, ctx.constantI32(0).getValue());
  ValueWrapper ping(builder, loc, pingValue);

  // Launch block sort kernel
  auto blockSortGrid = numTiles;
  auto blockSortBlock = blockThreads;

  // Use ext_call for block sort - pass count and ping as separate scalar args
  SmallVector<Value> blockSortInputs = {keysTensor, tempKeysTensor};
  SmallVector<Type> blockSortInputTypes = {keyTensorType, keyTensorType};
  // Removed duplicate declaration - these are defined later

  if (!config.keysOnly) {
    blockSortInputs.insert(blockSortInputs.end(),
                           {valuesTensor, tempValuesTensor});
    blockSortInputTypes.insert(blockSortInputTypes.end(),
                               {valueTensorType, valueTensorType});
  }
  // Add ping as additional scalar argument
  blockSortInputs.push_back(ping.getValue());
  blockSortInputTypes.push_back(builder.getI1Type());

  // Create symbol reference for block sort kernel
  auto blockSortSymbol = SymbolRefAttr::get(
      builder.getContext(), gpuModule.getName(),
      {SymbolRefAttr::get(builder.getContext(), blockSortFunc.getName())});

  // Block sort needs to return updated tensors in tensor semantics
  SmallVector<Type> blockSortResultTypes = {keysTensor.getType(),
                                            tempKeysTensor.getType()};

  if (!config.keysOnly) {
    blockSortResultTypes.push_back(valuesTensor.getType());
    blockSortResultTypes.push_back(tempValuesTensor.getType());
  }

  // Combine inputs and outs into single args list
  SmallVector<Value> blockSortArgs = blockSortInputs;

  // Build aliasing array: result[i] aliases arg[aliasing[i]]
  SmallVector<int32_t> blockSortAliasing;
  // result 0 aliases arg 0 (keys)
  blockSortAliasing.push_back(0);
  // result 1 aliases arg 1 (tempKeys)
  blockSortAliasing.push_back(1);

  // Build effects array
  SmallVector<StringRef> blockSortEffects = {"rw", "rw"}; // keys, tempKeys

  if (!config.keysOnly) {
    // result 2 aliases arg 2 (values)
    blockSortAliasing.push_back(2);
    // result 3 aliases arg 3 (tempValues)
    blockSortAliasing.push_back(3);
    blockSortEffects.push_back("rw"); // values
    blockSortEffects.push_back("rw"); // tempValues
  }
  blockSortEffects.push_back("-"); // ping (scalar)

  auto blockSortResults = builder.create<kernel::ExtCallOp>(
      loc, blockSortResultTypes, ValueRange{blockSortGrid.toIndex()},
      ValueRange{blockSortBlock.toIndex()}, blockSortArgs, blockSortSymbol,
      /*aliasingArgs=*/blockSortAliasing,
      /*effects=*/blockSortEffects);

  // Update tensor values with results
  keysTensor = blockSortResults->getResult(0);
  tempKeysTensor = blockSortResults->getResult(1);
  size_t resultIdx = 2;
  if (!config.keysOnly) {
    valuesTensor = blockSortResults->getResult(resultIdx++);
    tempValuesTensor = blockSortResults->getResult(resultIdx++);
  }

  // Merge passes loop - thread tensor values through as iter_args
  SmallVector<Value> loopIterArgs = {ping.getValue(), keysTensor,
                                     tempKeysTensor};
  if (!config.keysOnly) {
    loopIterArgs.push_back(valuesTensor);
    loopIterArgs.push_back(tempValuesTensor);
  }

  auto passLoop = ctx.buildForWithState(
      ctx.constantI32(0), numPasses, ctx.constantI32(1), loopIterArgs,
      [&](ValueWrapper pass, SmallVector<ValueWrapper> &state) {
        ValueWrapper currentPing(builder, loc, state[0].getValue());
        Value currentKeysTensor = state[1].getValue();
        Value currentTempKeysTensor = state[2].getValue();
        Value currentValuesTensor =
            !config.keysOnly ? state[3].getValue() : Value{};
        Value currentTempValuesTensor =
            !config.keysOnly ? state[4].getValue() : Value{};

        // Calculate target merged tiles for this pass (CUB line 360: 2 << pass)
        Value shiftAmount = pass.getValue();
        Value targetMergedTilesVal = builder.create<arith::ShLIOp>(
            loc, ctx.constantI32(2).getValue(), shiftAmount);
        ValueWrapper targetMergedTiles(builder, loc, targetMergedTilesVal);

        // Launch partition kernel using ext_call
        auto partitionGrid =
            (numPartitions + ctx.constantI32(255)) / ctx.constantI32(256);

        SmallVector<Value> partitionInputs;
        // Always pass buffers in consistent order - ping value tells kernel
        // which to use
        partitionInputs = {currentKeysTensor, currentTempKeysTensor};

        if (!config.keysOnly) {
          partitionInputs.insert(
              partitionInputs.end(),
              {currentValuesTensor, currentTempValuesTensor});
        }

        // Add scalar arguments
        partitionInputs.insert(
            partitionInputs.end(),
            {numPartitions.getValue(), partitionOffsets, currentPing.getValue(),
             targetMergedTiles.getValue(), itemsPerTile.getValue()});

        // Create symbol reference for partition kernel
        auto partitionSymbol =
            SymbolRefAttr::get(builder.getContext(), gpuModule.getName(),
                               {SymbolRefAttr::get(builder.getContext(),
                                                   partitionFunc.getName())});

        // Combine partition inputs and outs into args
        SmallVector<Value> partitionArgs = partitionInputs;

        // Build aliasing array: result[i] aliases arg[aliasing[i]]
        SmallVector<int32_t> partitionAliasing;
        if (!config.keysOnly) {
          // result 0 aliases arg 5 (partitionOffsets)
          partitionAliasing.push_back(5);
        } else {
          // result 0 aliases arg 3 (partitionOffsets)
          partitionAliasing.push_back(3);
        }

        // Build effects array
        SmallVector<StringRef> partitionEffects;
        if (!config.keysOnly) {
          partitionEffects = {
              "r",  "r", "r", "r", "-",
              "rw", "-", "-", "-"}; // keys, tempKeys, values, tempValues,
                                    // numPartitions, partitionOffsets, ping,
                                    // targetMergedTiles, itemsPerTile
        } else {
          partitionEffects = {"r", "r", "-", "rw",
                              "-", "-", "-"}; // keys, tempKeys, numPartitions,
                                              // partitionOffsets, ping,
                                              // targetMergedTiles, itemsPerTile
        }

        // Partition kernel modifies partitionOffsets
        auto partitionOp = builder.create<kernel::ExtCallOp>(
            loc, TypeRange{partitionOffsets.getType()},
            ValueRange{partitionGrid.toIndex()}, ValueRange{ctx.constant(256)},
            partitionArgs, partitionSymbol,
            /*aliasingArgs=*/partitionAliasing,
            /*effects=*/partitionEffects);

        // Update partitionOffsets with result
        partitionOffsets = partitionOp->getResult(0);

        // Launch merge kernel using ext_call
        SmallVector<Value> mergeInputs;
        // Always pass buffers in the same order - ping value controls which to
        // use inside kernel
        mergeInputs = {currentKeysTensor, currentTempKeysTensor};

        if (!config.keysOnly) {
          mergeInputs.insert(mergeInputs.end(),
                             {currentValuesTensor, currentTempValuesTensor});
        }

        // Add scalar arguments
        mergeInputs.insert(mergeInputs.end(),
                           {partitionOffsets, currentPing.getValue(),
                            targetMergedTiles.getValue()});

        // Create symbol reference for merge kernel
        auto mergeSymbol = SymbolRefAttr::get(
            builder.getContext(), gpuModule.getName(),
            {SymbolRefAttr::get(builder.getContext(), mergeFunc.getName())});

        // Merge kernel outputs and aliasing
        SmallVector<Type> mergeResultTypes = {currentKeysTensor.getType(),
                                              currentTempKeysTensor.getType()};
        SmallVector<int32_t> mergeAliasing = {
            0, 1}; // outputs alias first two inputs

        if (!config.keysOnly) {
          mergeResultTypes.push_back(currentValuesTensor.getType());
          mergeResultTypes.push_back(currentTempValuesTensor.getType());
          mergeAliasing.push_back(2); // out[2] aliases in[2]
          mergeAliasing.push_back(3); // out[3] aliases in[3]
        }

        // Combine merge inputs and outs into args
        SmallVector<Value> mergeArgs = mergeInputs;

        // Build effects array
        SmallVector<StringRef> mergeEffects = {"rw", "rw"}; // keys, tempKeys

        if (!config.keysOnly) {
          mergeEffects.push_back("rw"); // values
          mergeEffects.push_back("rw"); // tempValues
        }
        mergeEffects.push_back("r"); // partitionOffsets
        mergeEffects.push_back("-"); // ping (scalar)
        mergeEffects.push_back("-"); // targetMergedTiles (scalar)

        auto mergeOp = builder.create<kernel::ExtCallOp>(
            loc, mergeResultTypes, ValueRange{numTiles.toIndex()},
            ValueRange{blockThreads.toIndex()}, mergeArgs, mergeSymbol,
            /*aliasingArgs=*/mergeAliasing,
            /*effects=*/mergeEffects);

        // Get updated tensor values
        Value updatedKeysTensor = mergeOp->getResult(0);
        Value updatedTempKeysTensor = mergeOp->getResult(1);
        Value updatedValuesTensor =
            !config.keysOnly ? mergeOp->getResult(2) : Value{};
        Value updatedTempValuesTensor =
            !config.keysOnly ? mergeOp->getResult(3) : Value{};

        // Flip ping for next iteration
        Value trueVal = ctx.constantBool(true).getValue();
        Value nextPingVal =
            builder.create<arith::XOrIOp>(loc, currentPing.getValue(), trueVal);
        ValueWrapper nextPing(builder, loc, nextPingVal);

        // Return all updated values for next iteration
        SmallVector<ValueWrapper> yieldValues = {
            nextPing, ValueWrapper(builder, loc, updatedKeysTensor),
            ValueWrapper(builder, loc, updatedTempKeysTensor)};

        if (!config.keysOnly) {
          yieldValues.push_back(
              ValueWrapper(builder, loc, updatedValuesTensor));
          yieldValues.push_back(
              ValueWrapper(builder, loc, updatedTempValuesTensor));
        }

        return yieldValues;
      });

  // Extract the final tensor values from the loop
  Value finalKeysTensor = passLoop[1].getValue();
  Value finalValuesTensor = !config.keysOnly ? passLoop[3].getValue() : Value{};

  // CUB ensures the final result is always in the original buffer
  // by choosing the initial ping value as: ping = (numPasses % 2 == 0)
  // This guarantees that after all merge passes, data ends up in the original
  // buffer
  //
  // With the fix to make merge read/write opposite buffers:
  // - Block sort writes based on initialPing
  // - Each merge flips ping and moves data between buffers
  // - After numPasses flips, data is back in original buffer
  //
  // With correct ping initialization and merge read/write logic,
  // data always ends up in finalKeysTensor (original buffer)
  SmallVector<Value> results = {finalKeysTensor};

  if (!config.keysOnly)
    results.push_back(finalValuesTensor);

  builder.create<func::ReturnOp>(loc, results);
  return func;
}
