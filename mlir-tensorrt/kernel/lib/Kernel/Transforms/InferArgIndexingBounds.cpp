//===- InferArgIndexingBounds.cpp -----------------------------------------===//
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
/// Implementation of a pass that finds the bounds of memory access indices
/// for a given input argument of a function.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arg-indexing-bounds-inference"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::kernel {
#define GEN_PASS_DEF_INFERARGINDEXINGBOUNDSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

static bool isFuncArg(Value value, func::FuncOp funcOp) {
  if (auto bbArg = dyn_cast<BlockArgument>(value))
    return bbArg.getOwner()->getParentOp() == funcOp;
  return false;
}

static bool isConstant(Value value) {
  APInt constVal{};
  return matchPattern(value, m_ConstantInt(&constVal));
}

static FailureOr<SmallVector<BoundAttr>>
computeBoundFromIndexValues(RewriterBase &rewriter, func::FuncOp funcOp,
                            BlockArgument bbArg, ArrayRef<Value> indexValues,
                            presburger::BoundType boundType) {
  assert(boundType == presburger::BoundType::UB ||
         boundType == presburger::BoundType::LB);

  AffineMap boundMap;
  ValueDimList mapOperands;
  std::function<bool(Value, std::optional<int64_t>,
                     ValueBoundsConstraintSet & cstr)>
      stopCondition = [&](Value v, std::optional<int64_t> d,
                          ValueBoundsConstraintSet &cstr) {
        // Stop when reaching a block argument of the function body.
        if (isFuncArg(v, funcOp))
          return true;
        // Continue if it's the starting point
        Operation *op = v.getDefiningOp();
        if (!op)
          return false;
        // Stop when reaching a memref.load where both the memref is block arg
        // and the indices are constants
        auto loadOp = dyn_cast<memref::LoadOp>(op);
        if (loadOp && isFuncArg(loadOp.getMemRef(), funcOp) &&
            llvm::all_of(loadOp.getIndices(), isConstant))
          return true;
        // Stop when reaching a value that is defined outside of the function
        // body
        return funcOp.getRegion().findAncestorOpInRegion(*op) == nullptr;
      };

  // Set lower/upper bound string for detailed error msg
  std::string boundStr;
  if (boundType == presburger::BoundType::UB)
    boundStr = "upper";
  else
    boundStr = "lower";

  // Solve bound for each index value
  SmallVector<BoundAttr> indexBoundMaps;
  MLIRContext *ctx = funcOp->getContext();
  for (Value indexValue : indexValues) {
    if (failed(ValueBoundsConstraintSet::computeBound(boundMap, mapOperands,
                                                      boundType, indexValue,
                                                      stopCondition, true))) {
      funcOp.emitError() << "failed to compute indexing " << boundStr
                         << " bound for function argument " << bbArg;
      return failure();
    }
    // Get info for map operands
    SmallVector<IndexArgAttr> argNumbers;
    for (auto [v, d] : mapOperands) {
      Operation *operation = v.getDefiningOp();
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        argNumbers.push_back(IndexArgAttr::get(ctx, arg.getArgNumber(), {}));
      } else if (operation && isa<memref::LoadOp>(operation)) {
        // if the map operand is an load op, its memref must be a BlockArgument
        // and indices must be constants
        auto loadOp = dyn_cast_or_null<memref::LoadOp>(operation);
        Value memrefOperand = loadOp->getOperand(0);
        ValueRange loadIndices = loadOp.getIndices();
        assert(isFuncArg(memrefOperand, funcOp) &&
               llvm::all_of(loadIndices, isConstant));
        SmallVector<int64_t> coord =
            llvm::map_to_vector(loadIndices, [](Value v) {
              APInt constVal{};
              matchPattern(v, m_ConstantInt(&constVal));
              return constVal.getSExtValue();
            });
        argNumbers.push_back(IndexArgAttr::get(
            ctx, dyn_cast<BlockArgument>(memrefOperand).getArgNumber(), coord));
      } else {
        funcOp->emitError()
            << boundStr
            << " bound map depends on operands which are not function argument:"
            << v;
        return failure();
      }
    }
    // Form the maps and their operands as attributes
    BoundAttr indexBoundMap = BoundAttr::get(ctx, argNumbers, boundMap);
    indexBoundMaps.push_back(indexBoundMap);
  }
  return indexBoundMaps;
}

namespace {

struct InferArgIndexingBoundsPass
    : public kernel::impl::InferArgIndexingBoundsPassBase<
          InferArgIndexingBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    // Get the map from an arg to its page dim
    llvm::SmallDenseMap<BlockArgument, int64_t> argToPageDim;
    for (BlockArgument bbArg : funcOp.getArguments()) {
      std::optional<NamedAttribute> pageDimAttr =
          funcOp.getArgAttrDict(bbArg.getArgNumber())
              .getNamed("kernel.page_dim");
      if (!pageDimAttr)
        continue;
      if (auto pageDimIntAttr = dyn_cast<IntegerAttr>(pageDimAttr->getValue()))
        argToPageDim[bbArg] = pageDimIntAttr.getInt();
      else
        funcOp->emitError()
            << "expect an IntegerAttr for kernel.page_dim, but got "
            << pageDimAttr->getValue();
    }

    // Get the map from an arg to a list of index values which are
    // accessing the dims to be paged
    llvm::SmallDenseMap<BlockArgument, SmallVector<Value>> argToIndexValues;
    WalkResult walkResult = funcOp->walk([&](Operation *op) {
      // If op is not load or store, if it is other op that has read/write
      // effect on any bbArg in argToPageDim, we should abort
      if (!isa<memref::LoadOp, memref::StoreOp>(op)) {
        auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
        auto viewLike = dyn_cast<ViewLikeOpInterface>(op);
        if (!memEffect && !viewLike)
          return WalkResult::advance();
        for (auto [bbArg, pageDim] : argToPageDim) {
          if (memEffect &&
              (memEffect.getEffectOnValue<MemoryEffects::Read>(bbArg) ||
               memEffect.getEffectOnValue<MemoryEffects::Write>(bbArg))) {
            op->emitError() << "Cannot find bounds if the function argument is "
                               "consumed by ops with read/write memory effect";
            return WalkResult::interrupt();
          } else if (viewLike && viewLike.getViewSource() == bbArg) {
            op->emitError() << "Cannot find bounds if the function argument "
                               "has aliases (created by "
                            << viewLike->getName() << ")";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }

      // The memref operand is the second last operand in both Store and Load
      Value operand = op->getOperand(op->getNumOperands() - 2);
      auto bbArg = llvm::dyn_cast<BlockArgument>(operand);
      // If the arg does not contain page_dim, it will not be paged, so skip
      // it
      if (!bbArg || !argToPageDim.contains(bbArg))
        return WalkResult::advance();
      int64_t pageDim = argToPageDim[bbArg];
      SmallVector<Value> indices;
      if (auto loadOp = dyn_cast<memref::LoadOp>(op))
        indices = loadOp.getIndices();
      else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
        indices = storeOp.getIndices();
      for (auto [idx, indexValue] : llvm::enumerate(indices)) {
        if (static_cast<int64_t>(idx) == pageDim)
          argToIndexValues[bbArg].push_back(indexValue);
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    // For each arg to be paged, get the LB and UB for the index values
    for (auto [bbArg, indexValues] : argToIndexValues) {
      if (!isa<MemRefType>(bbArg.getType())) {
        funcOp->emitError()
            << "Function argument at position " << bbArg.getArgNumber()
            << " is not memref type, but " << bbArg.getType();
        return signalPassFailure();
      }
      FailureOr<SmallVector<BoundAttr>> upperBounds =
          computeBoundFromIndexValues(rewriter, funcOp, bbArg, indexValues,
                                      presburger::BoundType::UB);
      if (failed(upperBounds))
        return signalPassFailure();
      FailureOr<SmallVector<BoundAttr>> lowerBounds =
          computeBoundFromIndexValues(rewriter, funcOp, bbArg, indexValues,
                                      presburger::BoundType::LB);
      if (failed(lowerBounds))
        return signalPassFailure();

      // Set lower and upper bounds as argument attributes
      int64_t argNumber = bbArg.getArgNumber();
      funcOp.setArgAttr(argNumber, PageBoundsAttr::name,
                        PageBoundsAttr::get(funcOp->getContext(), *lowerBounds,
                                            *upperBounds));
    }
  }
};
} // namespace
