//===- ToLoopsOpInterfaceImpl.cpp ----------------------------------------===//
//
// Modified from original LLVM/MLIR code under Linalg dialect transforms.
// Original license:
// "Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"
//
// Modifiecations:
// - Changed code so that it expects linalg operations to operate on tensor
//   instead of buffer types.
//
// Modifications Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// This file contains the implementation of the ToLoopsOpInterface extensions
/// to the Linalg dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Dialect/LinalgExt/Transforms/ToLoopsOpInterfaceImpl.h"
#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;

/// Make canonical affine applies for the given map and values.
static SmallVector<Value>
makeCanonicalAffineApplies(OpBuilder &b, Location loc, AffineMap map,
                           ArrayRef<OpFoldResult> vals) {
  if (map.isEmpty())
    return {};
  assert(map.getNumInputs() == vals.size());
  SmallVector<OpFoldResult> res;
  res.reserve(map.getNumResults());
  unsigned numDoms = map.getNumDims();
  for (AffineExpr e : map.getResults()) {
    auto exprMap = AffineMap::get(numDoms, map.getNumSymbols(), e);
    res.push_back(affine::makeComposedFoldedAffineApply(b, loc, exprMap, vals));
  }
  return getValueOrCreateConstantIndexOp(b, loc, res);
}

/// Inline the region of the LinalgOp and emit the "store" (tensor.insert)
/// operations.
static SmallVector<Value>
inlineRegionAndEmitStore(OpBuilder &b, Location loc, linalg::LinalgOp op,
                         ValueRange allIvs, ArrayRef<Value> indexedValues,
                         ArrayRef<SmallVector<Value>> indexing,
                         ValueRange outputBuffers) {
  Block &block = *op.getBlock();
  IRMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(&op)) {
      map.map(op.getResult(0), allIvs[indexOp.getDim()]);
      continue;
    }
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  linalg::YieldOp terminator = cast<linalg::YieldOp>(block.getTerminator());
  SmallVector<Value> storeValues;
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    storeValues.push_back(b.create<tensor::InsertOp>(
        loc, toStore, outputBuffers[operand.getOperandNumber()],
        indexing[operand.getOperandNumber()]));
  }
  return storeValues;
}

/// Emit the scalar implementation for the LinalgOp operation.
static scf::ValueVector
emitScalarImplementation(OpBuilder &b, Location loc, ValueRange allIvs,
                         linalg::LinalgOp linalgOp,
                         ValueRange operandValuesToUse) {
  assert(linalgOp.hasPureTensorSemantics() &&
         "expected linalg op with tensor semantics");
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());

  SmallVector<OpFoldResult> allIvsPlusDims(allIvs);

  for (auto [inputOperand, operandValue] :
       llvm::zip(linalgOp.getDpsInputOperands(),
                 operandValuesToUse.take_front(linalgOp.getNumDpsInputs()))) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(operandValue);
      continue;
    }
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, operandValue, indexing));
  }

  SmallVector<SmallVector<Value>, 2> outputIndexing;
  for (auto [outputOperand, outputValue] :
       llvm::zip(linalgOp.getDpsInitsMutable(),
                 operandValuesToUse.take_back(linalgOp.getNumDpsInits()))) {
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
        allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, outputValue, indexing));
    outputIndexing.push_back(indexing);
  }

  return inlineRegionAndEmitStore(
      b, loc, linalgOp, allIvs, indexedValues,
      ArrayRef(outputIndexing).take_back(linalgOp.getNumDpsInits()),
      operandValuesToUse.take_back(linalgOp.getNumDpsInits()));
}

/// Lower the LinalgOp to a 'scf.for' loop nest.
FailureOr<SmallVector<Operation *>>
mlir::linalg_ext::convertLinalgOpToLoops(RewriterBase &rewriter,
                                         linalg::LinalgOp linalgOp) {
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  if (!linalgOp.hasPureTensorSemantics())
    return emitError(linalgOp.getLoc())
           << "expected linalg op with tensor semantics";

  SmallVector<Range, 4> loopRanges =
      linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  // Generate the loop nest using the 'mlir::linalg::GenerateLoopNest' utility.
  SmallVector<Operation *> loops;
  mlir::linalg::GenerateLoopNest<scf::ForOp>::doit(
      rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
        for (Value v : ivs) {
          BlockArgument ivVal = cast<BlockArgument>(v);
          loops.push_back(ivVal.getOwner()->getParentOp());
        }
        return emitScalarImplementation(b, loc, ivs, linalgOp,
                                        operandValuesToUse);
      });

  return loops;
}

namespace {
template <typename OpTy>
struct ToLoopsOpInterfaceImpl
    : public ToLoopsOpInterface::ExternalModel<ToLoopsOpInterfaceImpl<OpTy>,
                                               OpTy> {
  FailureOr<Operation *> lowerToLoops(Operation *op,
                                      RewriterBase &rewriter) const {
    FailureOr<SmallVector<Operation *>> loops =
        convertLinalgOpToLoops(rewriter, cast<linalg::LinalgOp>(op));
    if (failed(loops))
      return failure();
    rewriter.replaceOp(op, loops->front());
    return loops->front();
  }
};
} // namespace

void linalg_ext::registerToLoopsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::GenericOp::attachInterface<ToLoopsOpInterfaceImpl<GenericOp>>(*ctx);
    // linalg::MapOp::attachInterface<ToLoopsOpInterfaceImpl<MapOp>>(*ctx);
  });
}
