//===- OutliningUtils.cpp -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Definition of various outlining transform utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Utils/OutliningUtils.h"
#include "mlir-tensorrt-common/Utils/RegionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

/// Create a memref type with a fully dynamic strided layout attribute.
static MemRefType getDynamicMemRefType(OpBuilder &b, RankedTensorType type) {
  return MemRefType::get(
      type.getShape(), type.getElementType(),
      StridedLayoutAttr::get(
          b.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

/// Replace a `scf.in_parallel` terminator containing a number of
/// `tensor.parallel_insert_slice` operations with a corresponding number of
/// copy operations using this pattern:
///
/// ```mlir
/// parallel_insert_slice %src into %argN[offsets, sizes, strides]
/// ```
///
/// becomes
///
/// ```mlir
/// %source = to_memref %src
/// %dest = to_memref %argN
/// %dest_subview = subview %dest[offsets, sizes, strides]
/// copy %source, %dest_subview
/// ```
///
static void inParallelToMemRefCopy(RewriterBase &rewriter,
                                   scf::InParallelOp terminator) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(terminator);
  for (Operation &yieldingOp : terminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(yieldingOp);

    MemRefType destMemRefType =
        getDynamicMemRefType(rewriter, insertOp.getDest().getType());
    auto destMemRef = rewriter.create<bufferization::ToBufferOp>(
        insertOp.getLoc(), destMemRefType, insertOp.getDest(),
        /*read_only=*/false);
    auto srcMemRef = rewriter.create<bufferization::ToBufferOp>(
        insertOp.getLoc(),
        getDynamicMemRefType(rewriter, insertOp.getSource().getType()),
        insertOp.getSource(), /*read_only=*/false);

    SmallVector<OpFoldResult> offsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = insertOp.getMixedStrides();
    // Create a subview of the destination to match the hyper-rectangular area
    // of the `tensor.parallel_insert_slice` operation. Rank-reduce the subview
    // if the `tensor.parallel_insert_slice` operation is rank-expanding the
    // source.
    MemRefType destSubViewType =
        destMemRefType.getRank() !=
                cast<MemRefType>(srcMemRef.getType()).getRank()
            ? memref::SubViewOp::inferRankReducedResultType(
                  cast<MemRefType>(srcMemRef.getType()).getShape(),
                  destMemRefType, offsets, sizes, strides)
            : memref::SubViewOp::inferResultType(destMemRefType, offsets, sizes,
                                                 strides);
    auto destSubView =
        rewriter.create<memref::SubViewOp>(insertOp.getLoc(), destSubViewType,
                                           destMemRef, offsets, sizes, strides);
    rewriter.create<memref::CopyOp>(insertOp.getLoc(), srcMemRef, destSubView);
  }
}

static SmallVector<Value> makeForallBodyIsolatedFromAbove(
    RewriterBase &rewriter, scf::ForallOp op,
    std::function<SmallVector<Value>(RewriterBase &, Location loc,
                                     ValueRange ivs, ArrayRef<OpFoldResult> ubs,
                                     std::optional<ArrayAttr>)>
        ivReplacementBuilder,
    std::function<bool(Operation *)> cloneOperationIntoRegion) {

  // Update uses of the induction variable.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(op.getBody());
  SmallVector<Value> ivReplacements =
      ivReplacementBuilder(rewriter, op.getLoc(), op.getInductionVars(),
                           op.getMixedUpperBound(), op.getMapping());
  for (auto [iv, replacement] :
       llvm::zip_equal(op.getInductionVars(), ivReplacements))
    rewriter.replaceAllUsesWith(iv, replacement);

  // Erase the original induction variables.
  for (unsigned i = 0, e = op.getInductionVars().size(); i < e; i++)
    op.getBodyRegion().eraseArgument(0);

  // Make the region isolated from above.
  SmallVector<Value> arguments = mlir::createClosedRegion(
      rewriter, op.getBodyRegion(), cloneOperationIntoRegion);

  // Rewrite the terminator to use `memref.copy` operations.
  inParallelToMemRefCopy(rewriter, op.getTerminator());
  rewriter.setInsertionPoint(op.getTerminator());
  rewriter.create<func::ReturnOp>(op.getLoc(), ValueRange{});
  rewriter.eraseOp(op.getTerminator());

  // Move the outs arguments to the end.
  Block *body = op.getBody();

  for (unsigned i = 0, e = op->getNumResults(); i < e; i++) {
    // Unfortunately there isn't an easy way to rearange the arguments besides
    // manually creating a new one and doing a RAUW.
    Type tmpType = body->getArgument(0).getType();
    Location tmpLoc = body->getArgument(0).getLoc();
    BlockArgument newArg = body->addArgument(tmpType, tmpLoc);
    rewriter.replaceAllUsesWith(body->getArgument(0), newArg);
    body->eraseArgument(0);
  }
  const unsigned numInputArgs = arguments.size();
  llvm::append_range(arguments, op.getInits());
  rewriter.setInsertionPointToStart(op.getBody());

  for (BlockArgument arg : body->getArguments()) {
    if (auto rtt = dyn_cast<RankedTensorType>(arg.getType())) {
      arg.setType(getDynamicMemRefType(rewriter, rtt));
      auto ToBufferOp = rewriter.create<bufferization::ToTensorOp>(
          arg.getLoc(), rtt, arg, /*restrict=*/true,
          /*writable=*/arg.getArgNumber() >= numInputArgs);
      rewriter.replaceAllUsesExcept(arg, ToBufferOp, ToBufferOp);
    }
  }

  return arguments;
}

static FunctionType makeForallOutlinedBodyType(RewriterBase &rewriter,
                                               TypeRange callArgs) {
  SmallVector<Type> newTypes;
  newTypes.reserve(callArgs.size());
  for (Type t : callArgs) {
    if (auto rtt = dyn_cast<RankedTensorType>(t)) {
      newTypes.push_back(getDynamicMemRefType(rewriter, rtt));
      continue;
    }
    newTypes.push_back(t);
  }
  return FunctionType::get(rewriter.getContext(), TypeRange(newTypes),
                           TypeRange{});
}

FailureOr<ForallOutliningResult> mlir::outlineForall(
    RewriterBase &rewriter, scf::ForallOp op, StringRef name,
    SymbolTable &moduleForOutlinedBody,
    std::function<SmallVector<Value>(RewriterBase &, Location loc,
                                     ValueRange ivs, ArrayRef<OpFoldResult> ubs,
                                     std::optional<ArrayAttr>)>
        ivReplacementBuilder,
    std::function<Operation *(RewriterBase &, scf::ForallOp forallOp,
                              ValueRange args, func::FuncOp callee)>
        callBuilder,
    std::function<bool(Operation *)> cloneOperationIntoRegion) {

  SmallVector<Value> callArgs = makeForallBodyIsolatedFromAbove(
      rewriter, op, ivReplacementBuilder, std::move(cloneOperationIntoRegion));

  func::FuncOp funcOp = func::FuncOp::create(
      op.getLoc(), name,
      makeForallOutlinedBodyType(rewriter, TypeRange(callArgs)));
  Block *outlineBlock = funcOp.addEntryBlock();
  rewriter.inlineBlockBefore(op.getBody(), outlineBlock, outlineBlock->end(),
                             outlineBlock->getArguments());

  moduleForOutlinedBody.insert(funcOp);
  rewriter.setInsertionPoint(op);
  Operation *callOp = callBuilder(rewriter, op, callArgs, funcOp);

  rewriter.replaceOp(op, callOp);

  ForallOutliningResult result;
  result.outlinedBody = funcOp;
  result.forallReplacement = callOp;
  return result;
}

SmallVector<Value> mlir::getInductionVarReplacementsUsingGpuBlockId(
    RewriterBase &rewriter, Location loc, ValueRange ivs,
    ArrayRef<OpFoldResult> ubs, std::optional<ArrayAttr> attr) {
  SmallVector<Value> result;

  ValueRange delinearizedCoord{};

  if (attr && llvm::any_of(attr->getValue(), [](Attribute a) {
        auto mappingAttr = llvm::dyn_cast<gpu::GPUBlockMappingAttr>(a);
        return mappingAttr && mappingAttr.isLinearMapping();
      })) {
    SmallVector<OpFoldResult> linearizationOperands;
    OpFoldResult linearIndex{};
    AffineExpr s0, s1, s2, d0, d1, d2;
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    bindDims(rewriter.getContext(), d0, d1, d2);

    // Assemble the affine map operands (dimension vars first
    // followed by symbol vars).
    linearizationOperands = SmallVector<OpFoldResult>{
        OpFoldResult(rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x)),
        OpFoldResult(rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::y)),
        OpFoldResult(rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::z))};
    linearizationOperands.append({
        OpFoldResult(rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x)),
        OpFoldResult(rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::y)),
        OpFoldResult(rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::z)),
    });

    // Calculate the linear processor index.
    SmallVector<AffineExpr> basis = computeSuffixProduct({s0, s1, s2});
    linearIndex = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0 * basis[0] + d1 * basis[1] + d2 * basis[2],
        linearizationOperands);
    delinearizedCoord =
        rewriter
            .create<affine::AffineDelinearizeIndexOp>(
                loc,
                getValueOrCreateConstantIndexOp(rewriter, loc, linearIndex),
                ubs)
            .getResults();
  }

  for (unsigned index = 0, e = ivs.size(); index < e; index++) {
    if (attr && index < attr->size()) {
      if (auto mappingAttr =
              dyn_cast<gpu::GPUBlockMappingAttr>(attr->getValue()[index])) {
        if (!mappingAttr.isLinearMapping()) {
          int64_t id = mappingAttr.getMappingId();
          assert(id < 3 && "expected GPU mapping ID to be < 2");
          result.push_back(rewriter.create<gpu::BlockIdOp>(
              loc, static_cast<gpu::Dimension>(id)));
          continue;
        }

        // Delinearize into the Forall basis.
        result.push_back(delinearizedCoord[mappingAttr.getMappingId() -
                                           static_cast<int64_t>(
                                               gpu::MappingId::LinearDim0)]);
        continue;
      }
    }
    assert(index < 3 && "index is out-of-bounds");
    result.push_back(rewriter.create<gpu::BlockIdOp>(
        loc, static_cast<gpu::Dimension>(index)));
  }
  return result;
}
