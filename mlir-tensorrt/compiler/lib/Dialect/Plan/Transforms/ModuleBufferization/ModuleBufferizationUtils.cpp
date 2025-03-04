//===- ModuleBufferizationUtils.cpp ---------------------------------------===//
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
/// This file contains definitions for some simple utilities needed
/// for bufferization. We use the CallGraph to find functions which
/// cannot call each other circularly and enumerate them in the order
/// of 'fewer incoming call edges' to 'more incoming call edges'.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using OneShotBufferizationOptions = bufferization::OneShotBufferizationOptions;

/// Create a subview containing a single element.
static Value createUnitSubView(RewriterBase &rewriter, Location loc, Value base,
                               ValueRange offsets) {
  auto type = cast<MemRefType>(base.getType());
  if (type.hasStaticShape() && type.getNumElements() == 1)
    return base;
  SmallVector<OpFoldResult> coords =
      llvm::map_to_vector(offsets, [](Value v) { return OpFoldResult(v); });
  SmallVector<OpFoldResult> ones(coords.size(), rewriter.getIndexAttr(1));
  return rewriter.create<memref::SubViewOp>(loc, base, coords, ones, ones);
}

static FailureOr<Value>
copyElementToHost(RewriterBase &rewriter, Location loc, Value source,
                  ValueRange coord,
                  const bufferization::OneShotBufferizationOptions &options) {
  MemRefType sourceType = cast<MemRefType>(source.getType());

  // Create the type for the new allocation -- just enough to hold a single
  // element.
  auto allocType = MemRefType::get(
      SmallVector<int64_t>(coord.size(), 1), sourceType.getElementType(),
      MemRefLayoutAttrInterface{},
      plan::MemorySpaceAttr::get(rewriter.getContext(),
                                 plan::MemorySpace::host_pinned));

  // Create the subview of the original source.
  Value sourceView = createUnitSubView(rewriter, loc, source, coord);

  // Allocate the staging buffer, copy the element to it, and synchronize.
  FailureOr<Value> allocated =
      options.createAlloc(rewriter, loc, allocType, ValueRange{});
  if (failed(allocated))
    return failure();
  if (failed(options.createMemCpy(rewriter, loc, sourceView, *allocated)))
    return failure();
  return *allocated;
}

static LogicalResult
copyElementToDevice(RewriterBase &rewriter, Location loc, Value scalarToStore,
                    Value destMemRef, ValueRange coord,
                    const bufferization::OneShotBufferizationOptions &options) {
  MemRefType destType = cast<MemRefType>(destMemRef.getType());

  // Create the type for the new allocation -- just enough to hold a single
  // element.
  auto allocType =
      MemRefType::get(SmallVector<int64_t>(coord.size(), 1),
                      destType.getElementType(), MemRefLayoutAttrInterface{},
                      plan::MemorySpaceAttr::get(rewriter.getContext(),
                                                 plan::MemorySpace::host));

  // Create the subview of the original source.
  Value destView = createUnitSubView(rewriter, loc, destMemRef, coord);

  // Allocate the staging buffer, copy the element to it, and synchronize.
  FailureOr<Value> alloc = options.createAlloc(rewriter, loc, allocType, {});
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<memref::StoreOp>(
      loc, scalarToStore, *alloc, SmallVector<Value>(destType.getRank(), zero));
  return options.createMemCpy(rewriter, loc, *alloc, destView);
}

LogicalResult plan::fixupHostModule(
    ModuleLikeOp module,
    const bufferization::OneShotBufferizationOptions &options) {
  IRRewriter rewriter(module);

  auto walkResult = module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (ModuleLikeOp(op))
      return op == module ? WalkResult::advance() : WalkResult::skip();

    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      MemRefType fromType = loadOp.getMemRefType();
      auto space = dyn_cast<plan::MemorySpaceAttr>(fromType.getMemorySpace());
      if (!space || space.isHostVisible())
        return WalkResult::skip();
      rewriter.setInsertionPoint(loadOp);
      FailureOr<Value> copiedToHost =
          copyElementToHost(rewriter, loadOp.getLoc(), loadOp.getMemRef(),
                            loadOp.getIndices(), options);
      if (failed(copiedToHost))
        return WalkResult::interrupt();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loadOp.getLoc(), 0);
      rewriter.replaceOpWithNewOp<memref::LoadOp>(
          loadOp, *copiedToHost, SmallVector<Value>(fromType.getRank(), zero));
      return WalkResult::skip();
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      MemRefType fromType = storeOp.getMemRefType();
      auto space = dyn_cast<plan::MemorySpaceAttr>(fromType.getMemorySpace());
      if (!space || space.isHostVisible())
        return WalkResult::skip();
      rewriter.setInsertionPoint(storeOp);
      if (failed(copyElementToDevice(rewriter, storeOp.getLoc(),
                                     storeOp.getValue(), storeOp.getMemRef(),
                                     storeOp.getIndices(), options)))
        return WalkResult::interrupt();
      rewriter.eraseOp(storeOp);
      return WalkResult::skip();
    }

    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}