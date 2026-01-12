//===- StablehloBuilder.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/StablehloExt/Utils/StablehloBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace stablehlo;

stablehlo::BroadcastInDimOp
StablehloBuilder::broadcastInDim(Value v, RankedTensorType newType,
                                 ArrayRef<int64_t> broadcastDims) {
  return rewriter.create<stablehlo::BroadcastInDimOp>(v.getLoc(), newType, v,
                                                      broadcastDims);
}

stablehlo::ReshapeOp StablehloBuilder::expandDims(Value v, int64_t idx) {
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  assert(idx < rtt.getRank());
  RankedTensorType newType = RankedTensorType::Builder(rtt).insertDim(1, idx);
  return rewriter.create<stablehlo::ReshapeOp>(v.getLoc(), newType, v);
}

stablehlo::ReshapeOp StablehloBuilder::reshape(Value v,
                                               ArrayRef<int64_t> shape) {
  return rewriter.create<stablehlo::ReshapeOp>(
      v.getLoc(), cast<RankedTensorType>(v.getType()).clone(shape), v);
}

stablehlo::ReshapeOp StablehloBuilder::squeezeDims(Value v, int64_t idx) {
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  assert(idx < rtt.getRank() && rtt.getDimSize(idx) == 1);
  RankedTensorType newType = RankedTensorType::Builder(rtt).dropDim(idx);
  return rewriter.create<stablehlo::ReshapeOp>(v.getLoc(), newType, v);
}

stablehlo::TransposeOp StablehloBuilder::transpose(Value v,
                                                   ArrayRef<int64_t> perm) {
  assert(static_cast<int64_t>(perm.size()) ==
             cast<RankedTensorType>(v.getType()).getRank() &&
         "expected size of 'permutation' list to match rank of tensor");
  return rewriter.create<stablehlo::TransposeOp>(v.getLoc(), v, perm);
}

stablehlo::TransposeOp StablehloBuilder::transpose(Value v, AffineMap perm) {
  assert(perm.isPermutation() && "expected permutation");
  assert(perm.getNumResults() == cast<RankedTensorType>(v.getType()).getRank());
  SmallVector<int64_t, 4> elements;
  for (unsigned i = 0; i < perm.getNumResults(); i++)
    elements.push_back(
        llvm::cast<AffineDimExpr>(perm.getResult(i)).getPosition());
  return rewriter.create<stablehlo::TransposeOp>(v.getLoc(), v, elements);
}

stablehlo::SliceOp StablehloBuilder::indexSelect(Value v, int64_t dim,
                                                 int64_t idx) {
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  assert(dim < rtt.getRank() && idx < rtt.getDimSize(dim) &&
         "expected dim < rank and idx < shape[dim]");
  SmallVector<int64_t> limits(rtt.getShape());
  SmallVector<int64_t> offsets(rtt.getRank(), 0);
  limits[dim] = 1 + idx;
  offsets[dim] = idx;
  return rewriter.create<stablehlo::SliceOp>(
      v.getLoc(), v, offsets, limits, SmallVector<int64_t>(offsets.size(), 1));
}

stablehlo::ConcatenateOp StablehloBuilder::concat(ValueRange v, int64_t dim) {
  return rewriter.create<stablehlo::ConcatenateOp>(v.front().getLoc(), v, dim);
}
