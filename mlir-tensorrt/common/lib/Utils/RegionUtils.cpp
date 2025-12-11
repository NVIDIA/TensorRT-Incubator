//===- RegionUtils.cpp ----------------------------------------------------===//
//
// This file contains code modified from upstream MLIR project.
// The original code is licensed under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Utils/RegionUtils.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include <deque>

using namespace mlir;

SmallVector<Value> mlir::createClosedRegion(
    RewriterBase &rewriter, Region &region,
    llvm::function_ref<bool(Operation *)> cloneOperationIntoRegion,
    llvm::function_ref<void(SmallVectorImpl<Value> &)> reorderCapturedValues) {
  // Get initial list of values used within region but defined above.
  llvm::SetVector<Value> initialCapturedValues;
  mlir::getUsedValuesDefinedAbove(region, initialCapturedValues);

  std::deque<Value> worklist(initialCapturedValues.begin(),
                             initialCapturedValues.end());
  llvm::DenseSet<Value> visited;
  llvm::DenseSet<Operation *> visitedOps;

  llvm::SetVector<Value> finalCapturedValuesSet;
  llvm::SetVector<Operation *> finalProducerOps;

  SmallVector<Operation *> clonedOperations;
  while (!worklist.empty()) {
    Value currValue = worklist.front();
    worklist.pop_front();
    if (visited.count(currValue))
      continue;
    visited.insert(currValue);

    Operation *definingOp = currValue.getDefiningOp();
    if (!definingOp || visitedOps.count(definingOp)) {
      finalCapturedValuesSet.insert(currValue);
      finalProducerOps.insert(
          definingOp
              ? definingOp
              : cast<BlockArgument>(currValue).getOwner()->getParentOp());
      continue;
    }
    visitedOps.insert(definingOp);

    if (!cloneOperationIntoRegion(definingOp)) {
      // Defining operation isnt cloned, so add the current value to final
      // captured values list.
      finalCapturedValuesSet.insert(currValue);
      finalProducerOps.insert(definingOp);
      continue;
    }

    // Add all operands of the operation to the worklist and mark the op as to
    // be cloned.
    for (Value operand : definingOp->getOperands()) {
      if (visited.count(operand))
        continue;
      worklist.push_back(operand);
    }
    clonedOperations.push_back(definingOp);
  }

  finalProducerOps = mlir::topologicalSort(finalProducerOps);
  SetVector<Value> finalCapturedValues = [&finalProducerOps,
                                          &finalCapturedValuesSet]() {
    llvm::SetVector<Value> finalCapturedValues;
    for (Operation *op : finalProducerOps) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (BlockArgument arg : block.getArguments()) {
            if (finalCapturedValuesSet.contains(arg))
              finalCapturedValues.insert(arg);
          }
        }
      }
      for (Value v : op->getResults()) {
        if (finalCapturedValuesSet.contains(v))
          finalCapturedValues.insert(v);
      }
    }
    return finalCapturedValues;
  }();
  assert(finalCapturedValues.size() == finalCapturedValuesSet.size() &&
         "final captured values size mismatch");

  // The operations to be cloned need to be ordered in topological order
  // so that they can be cloned into the region without violating use-def
  // chains.
  mlir::computeTopologicalSorting(clonedOperations);

  OpBuilder::InsertionGuard g(rewriter);
  // Collect types of existing block
  Block *entryBlock = &region.front();
  SmallVector<Type> newArgTypes =
      llvm::to_vector(entryBlock->getArgumentTypes());
  SmallVector<Location> newArgLocs = llvm::to_vector(llvm::map_range(
      entryBlock->getArguments(), [](BlockArgument b) { return b.getLoc(); }));

  // Append the types of the captured values.
  for (auto value : finalCapturedValues) {
    newArgTypes.push_back(value.getType());
    newArgLocs.push_back(value.getLoc());
  }

  // Create a new entry block.
  Block *newEntryBlock =
      rewriter.createBlock(&region, region.begin(), newArgTypes, newArgLocs);
  auto newEntryBlockArgs = newEntryBlock->getArguments();

  // Create a mapping between the captured values and the new arguments added.
  IRMapping map;
  auto replaceIfFn = [&](OpOperand &use) {
    return region.isAncestor(use.getOwner()->getParentRegion());
  };
  for (auto [arg, capturedVal] :
       llvm::zip(newEntryBlockArgs.take_back(finalCapturedValues.size()),
                 finalCapturedValues)) {
    map.map(capturedVal, arg);
    rewriter.replaceUsesWithIf(capturedVal, arg, replaceIfFn);
  }
  rewriter.setInsertionPointToStart(newEntryBlock);
  for (auto *clonedOp : clonedOperations) {
    Operation *newOp = rewriter.clone(*clonedOp, map);
    rewriter.replaceOpUsesWithIf(clonedOp, newOp->getResults(), replaceIfFn);
  }
  rewriter.mergeBlocks(
      entryBlock, newEntryBlock,
      newEntryBlock->getArguments().take_front(entryBlock->getNumArguments()));
  return llvm::to_vector(finalCapturedValues);
}
