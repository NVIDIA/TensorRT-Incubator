//===- Utils.cpp ------------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
/// TensorRT dialect utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::tensorrt;

/// Check that the batch dim and only the batch dim is dynamic.
static bool onlyBatchDimStatic(TensorType argType) {
  if (!argType.isDynamicDim(0))
    return false;
  for (unsigned idx = 1; idx < argType.getRank(); idx++) {
    if (argType.isDynamicDim(idx))
      return false;
  }
  return true;
}

bool tensorrt::hasArgumentShapeProfile(FunctionOpInterface op,
                                       unsigned argIndex) {
  return succeeded(getArgumentShapeProfile(op, argIndex));
}

bool tensorrt::hasHostTensorValueBounds(FunctionOpInterface op,
                                        unsigned argIndex) {
  return succeeded(getArgumentValueBounds(op, argIndex));
}

/// Return a ShapeProfileAttr for the given specified arg assuming only the
/// batch size is dynamic.
static FailureOr<ShapeProfileAttr> getArgumentShapeInfoDynamicBatchOnly(
    FunctionOpInterface op, const DynamicDimensionBounds &batchSizeRange,
    unsigned argIndex) {
  auto argType = cast<TensorType>(op.getArgument(argIndex).getType());
  if (argType.hasStaticShape() || !argType.hasRank())
    return failure();
  if (!onlyBatchDimStatic(argType))
    return failure();
  ArrayRef<int64_t> nonBatchDims = argType.getShape().drop_front(1);
  SmallVector<int64_t> shapeMin, shapeOpt, shapeMax;
  shapeMin = {batchSizeRange.min};
  llvm::append_range(shapeMin, nonBatchDims);
  shapeOpt = {batchSizeRange.opt};
  llvm::append_range(shapeOpt, nonBatchDims);
  shapeMax = {batchSizeRange.max};
  llvm::append_range(shapeMax, nonBatchDims);
  return ShapeProfileAttr::get(op->getContext(), shapeMin, shapeOpt, shapeMax);
}

FailureOr<ShapeProfileAttr>
tensorrt::getArgumentShapeProfile(FunctionOpInterface op, unsigned argIndex) {
  auto tensorType = cast<RankedTensorType>(op.getArgument(argIndex).getType());
  if (tensorType.hasStaticShape())
    return ShapeProfileAttr::get(tensorType);
  auto shapeProfile = op.getArgAttrOfType<ShapeProfileAttr>(
      argIndex, TensorRTDialect::getShapeProfileArgAttrName());
  if (!shapeProfile)
    return failure();
  return shapeProfile;
}

FailureOr<ShapeProfileAttr>
tensorrt::getArgumentValueBounds(FunctionOpInterface op, unsigned argIndex) {
  auto shapeProfile = op.getArgAttrOfType<ShapeProfileAttr>(
      argIndex, TensorRTDialect::getShapeTensorValueBoundsArgAttrName());
  if (!shapeProfile)
    return failure();
  return shapeProfile;
}

FailureOr<ShapeProfileAttr> tensorrt::inferArgShapeProfile(
    FunctionOpInterface op, unsigned argIndex,
    std::optional<DynamicDimensionBounds> batchSizeRange) {
  BlockArgument arg = op.getArgument(argIndex);
  RankedTensorType t = dyn_cast<RankedTensorType>(arg.getType());
  if (!t)
    return failure();

  if (t.hasStaticShape())
    return ShapeProfileAttr::get(t);

  // If the argument has a dynamic shape, try to resolve the the min/max/opt
  // information from the batch information (if only batch dim is static) or
  // the argument attributes if batch dim range is not available.
  if (onlyBatchDimStatic(t) && batchSizeRange.has_value())
    return getArgumentShapeInfoDynamicBatchOnly(op, *batchSizeRange,
                                                arg.getArgNumber());

  // Otherwise, return the shape profile if present.
  return getArgumentShapeProfile(op, arg.getArgNumber());
}

TypedValue<RankedTensorType>
tensorrt::createConstShapeTensor(RewriterBase &b, Location loc,
                                 ArrayRef<int32_t> values) {
  return b.create<tensorrt::ConstantOp>(loc, b.getI32TensorAttr(values));
}

TypedValue<RankedTensorType>
tensorrt::scatterShapeTensor(RewriterBase &b, Location loc,
                             ArrayRef<int64_t> baseShape, int32_t scatterDim,
                             TypedValue<RankedTensorType> update) {
  assert(!ShapedType::isDynamicShape(baseShape) &&
         "baseShape must be a static shape");
  assert(scatterDim < static_cast<int32_t>(baseShape.size()) &&
         "scatterDim must be in range [0, rank)");
  assert(update.getType().getNumElements() == 1 &&
         update.getType().getElementType().isInteger(32) &&
         "expected scalar update tensor with element type i32");

  auto getI32TensorAttr = [&](auto x) {
    return b.getI32TensorAttr(SmallVector<int32_t>(x.begin(), x.end()));
  };
  int64_t rank = static_cast<int64_t>(baseShape.size());

  // Reshape scalar to 1xi32 if required.
  if (update.getType().getRank() == 0)
    update = b.create<tensorrt::ExpandRankOp>(loc, update.getType().clone({1}),
                                              update);

  // Handle the trivial case.
  if (rank == 1)
    return update;

  SmallVector<Value> parts;
  ArrayRef<int64_t> partShape = baseShape.slice(0, scatterDim);
  if (!partShape.empty())
    parts.push_back(
        b.create<tensorrt::ConstantOp>(loc, getI32TensorAttr(partShape)));
  parts.push_back(update);
  if (scatterDim < static_cast<int32_t>(baseShape.size()) - 1) {
    partShape = baseShape.drop_front(scatterDim + 1);
    parts.push_back(
        b.create<tensorrt::ConstantOp>(loc, getI32TensorAttr(partShape)));
  }

  return b.create<tensorrt::ConcatenationOp>(loc, parts, 0);
}

FailureOr<Attribute> tensorrt::getSplatConstantElementAttribute(Value x) {
  while (true) {
    if (auto expandRank = x.getDefiningOp<tensorrt::ExpandRankOp>())
      x = expandRank.getInput();
    else if (auto collapseRank = x.getDefiningOp<tensorrt::CollapseRankOp>())
      x = collapseRank.getInput();
    else if (auto reshape = x.getDefiningOp<tensorrt::ReshapeOp>())
      x = reshape.getInput();
    else if (auto broadcast = x.getDefiningOp<tensorrt::BroadcastOp>())
      x = broadcast.getInput();
    else if (auto cast = x.getDefiningOp<tensorrt::CastOp>())
      x = cast.getInput();
    else if (auto identity = x.getDefiningOp<tensorrt::IdentityOp>())
      x = identity.getInput();
    else if (auto slice = x.getDefiningOp<tensorrt::SliceOp>())
      x = slice.getInput();
    else if (auto constant = x.getDefiningOp<tensorrt::ConstantOp>()) {
      SplatElementsAttr els{};
      if (!matchPattern(x, m_Constant(&els)))
        return failure();
      Attribute value = els.getSplatValue<Attribute>();
      if (!isa<FloatAttr, IntegerAttr>(value))
        return failure();
      return value;
    } else {
      return failure();
    }
  }
  return failure();
}
