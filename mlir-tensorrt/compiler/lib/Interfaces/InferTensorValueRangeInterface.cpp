//===- InferTensorValueRangeInterface.cpp -------------------------------===//
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
/// InferTensorValueRangeInterface definitions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"

using namespace mlirtrt::compiler;
using namespace mlir;

//===----------------------------------------------------------------------===//
// BoundsValue
//===----------------------------------------------------------------------===//

bool BoundsArray::shouldAnalyzeValueBounds(Type type) {
  if (auto rtt = dyn_cast<RankedTensorType>(type))
    return rtt.getElementType().isSignlessIntOrIndex() &&
           rtt.hasStaticShape() && rtt.getNumElements() <= kMaxVolumeThreshold;
  return false;
}
bool BoundsArray::shouldAnalyzeValueBounds(Value value) {
  return shouldAnalyzeValueBounds(value.getType());
}

ConstantIntRanges BoundsArray::getMaxDimRange() {
  APInt smin = APInt(IndexType::kInternalStorageBitWidth, 0);
  APInt smax = APInt(IndexType::kInternalStorageBitWidth,
                     std::numeric_limits<int32_t>::max());
  return ConstantIntRanges::fromSigned(smin, smax);
}

BoundsArray BoundsArray::getMaxRangeForShapeBounds(Value v) {
  auto type = cast<ShapedType>(v.getType());
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(type.getRank());
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      ranges.push_back(getMaxDimRange());
      continue;
    }
    ranges.push_back(ConstantIntRanges::constant(APInt(64, dim)));
  }
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::getMaxRangeForValueBounds(Value v) {
  assert(shouldAnalyzeValueBounds(v) && "value is unsuitable for analysis");
  Type elementType = mlir::getElementTypeOrSelf(v);
  unsigned numBits = ConstantIntRanges::getStorageBitwidth(elementType);
  APInt smin = APInt::getSignedMinValue(numBits);
  APInt smax = APInt::getSignedMaxValue(numBits);
  SmallVector<ConstantIntRanges> ranges(
      cast<ShapedType>(v.getType()).getNumElements(),
      ConstantIntRanges::fromSigned(smin, smax));
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::getFromConstantValue(DenseIntElementsAttr v) {
  assert(shouldAnalyzeValueBounds(v.getType()) &&
         "attribute type is unsuitable for creating value bound state");
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(cast<ShapedType>(v.getType()).getNumElements());
  for (const APInt &element : v.getValues<APInt>())
    ranges.push_back(ConstantIntRanges::constant(element));
  return BoundsArray(std::move(ranges));
}

BoundsArray BoundsArray::fromShapeBounds(ArrayRef<int64_t> min,
                                         ArrayRef<int64_t> max) {
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(min, max))
    res.push_back(ConstantIntRanges::fromSigned(APInt(64, l), APInt(64, r)));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::fromIntegerValueBounds(unsigned bitWidth,
                                                ArrayRef<int64_t> min,
                                                ArrayRef<int64_t> max) {
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(min, max))
    res.push_back(
        ConstantIntRanges::fromSigned(APInt(64, l).sextOrTrunc(bitWidth),
                                      APInt(64, r).sextOrTrunc(bitWidth)));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::fromIntegerValueBounds(ArrayRef<llvm::APInt> min,
                                                ArrayRef<llvm::APInt> max) {
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(min, max))
    res.push_back(ConstantIntRanges::fromSigned(l, r));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::join(const BoundsArray &lhs, const BoundsArray &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(lhs.getValue(), rhs.getValue()))
    res.push_back(l.rangeUnion(r));
  return BoundsArray(std::move(res));
}

BoundsArray BoundsArray::meet(const BoundsArray &lhs, const BoundsArray &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  SmallVector<ConstantIntRanges> res;
  for (auto [l, r] : llvm::zip_equal(lhs.getValue(), rhs.getValue()))
    res.push_back(l.intersection(r));
  return BoundsArray(std::move(res));
}

void BoundsArray::print(raw_ostream &os) const {
  if (!value) {
    os << "<<uninitialized>>";
    return;
  }
  os << "<";
  llvm::interleaveComma(*value, os, [&](const ConstantIntRanges &r) {
    os << "[" << r.smin() << ", " << r.smax() << "]";
  });
  os << ">";
}

llvm::raw_ostream &mlirtrt::compiler::operator<<(llvm::raw_ostream &os,
                                                 const BoundsArray &v) {
  v.print(os);
  return os;
}

std::pair<DenseElementsAttr, DenseElementsAttr>
BoundsArray::getAsElementsAttr(RankedTensorType type) const {
  assert(!isUninitialized() && "expected initialized value");
  assert(type.getNumElements() == static_cast<int64_t>(value->size()) &&
         "specified tensor type's volume does not match lattice value volume");
  SmallVector<APInt> lbs;
  lbs.reserve(type.getNumElements());
  SmallVector<APInt> ubs;
  ubs.reserve(type.getNumElements());
  for (const ConstantIntRanges &r : *value) {
    lbs.push_back(r.smin());
    ubs.push_back(r.smax());
  }
  return std::make_pair(DenseElementsAttr::get(type, lbs),
                        DenseElementsAttr::get(type, ubs));
}

/// Returns true if the element ranges are constant (single-value) ranges.
std::optional<DenseElementsAttr>
BoundsArray::getConstantValues(RankedTensorType type) const {
  assert(!isUninitialized() && "expected initialized value");
  assert(type.getNumElements() == static_cast<int64_t>(value->size()) &&
         "specified tensor type's volume does not match lattice value volume");
  SmallVector<APInt> lbs;
  lbs.reserve(type.getNumElements());
  for (const ConstantIntRanges &r : *value) {
    if (r.smin() != r.smax())
      return {};
    lbs.push_back(r.smin());
  }
  return DenseElementsAttr::get(type, lbs);
}

//===----------------------------------------------------------------------===//
// Generated interface class implmenetations.
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.cpp.inc"
