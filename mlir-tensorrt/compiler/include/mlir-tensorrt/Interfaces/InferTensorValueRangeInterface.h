//===- InferTensorValueRangeInterface.h --------------------------*- C++
//-*-===//
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
/// Declarations for InferTensorValueRangeInterface.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE
#define MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include <optional>

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// BoundsArray
//===----------------------------------------------------------------------===//

/// A BoundsArray is simply an array of mlir::ConstantIntRanges used to
/// represent either the bounds on a shape of a tensor-typed SSA value or the
/// bounds of the element values of a statically shaped integer tensor-typed SSA
/// value. When it is used to represent the bounds for the value of a tensor, we
/// use a canonical packed generalized row-major layout mapping from tensor
/// coordinates to storage index.
class BoundsArray {
public:
  BoundsArray() : value(std::nullopt) {}

  BoundsArray(llvm::ArrayRef<mlir::ConstantIntRanges> value)
      : value(std::make_optional(llvm::to_vector(value))) {}

  bool isUninitialized() const { return !value.has_value(); }

  bool operator==(const BoundsArray &rhs) const { return value == rhs.value; }

  llvm::ArrayRef<mlir::ConstantIntRanges> getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  /// Return the most conservative integer scalar bounds for an dynamic/unknown
  /// dimension extent.
  static mlir::ConstantIntRanges getMaxDimRange();

  /// Create a BoundsValue from the min/max bounds of shape. Using this method
  /// ensures that the `value` are created with the correct storage bitwidth
  /// (an implementation detail of the analysis).
  static BoundsArray fromShapeBounds(llvm::ArrayRef<int64_t> min,
                                     llvm::ArrayRef<int64_t> max);

  /// Create a `BoundsValue` using the given scalar values encoded as int64_t
  /// values. However, when storing the bounds, use the given bitwidth.
  /// TODO: remove this when we migrate away from using
  /// `#tensorrt.shape_profile` for value bounds.
  static BoundsArray fromIntegerValueBounds(unsigned bitwidth,
                                            llvm::ArrayRef<int64_t> min,
                                            llvm::ArrayRef<int64_t> max);
  static BoundsArray fromIntegerValueBounds(llvm::ArrayRef<llvm::APInt> min,
                                            llvm::ArrayRef<llvm::APInt> max);

  /// For the given tensor-typed value, return the most conservative bounds for
  /// the shape of `v`. For each unknown dimension of the shape of `v` the
  /// `getMaxDimRange()` bound is used.
  static BoundsArray getMaxRangeForShapeBounds(mlir::Value v);

  /// For the given statically shaped integer tensor-typed value, return the
  /// most conservative bounds for the value of `v`.
  static BoundsArray getMaxRangeForValueBounds(mlir::Value v);

  /// For the given DenseIntElementsAttr, return a corresponding BoudnsValue
  /// representing constant bounds as indicated by the attribute.
  static BoundsArray getFromConstantValue(mlir::DenseIntElementsAttr attr);

  /// Join two BoundsValues by performing a pointwise union of the integer
  /// scalar a ranges.
  static BoundsArray join(const BoundsArray &lhs, const BoundsArray &rhs);

  /// Meet two BoundsValues by performing a pointwise intersection of the
  /// integer scalar a ranges.
  static BoundsArray meet(const BoundsArray &lhs, const BoundsArray &rhs);

  /// Print a human-readable representation of the bounds.
  void print(llvm::raw_ostream &os) const;

  /// Return the min/max bounds representation as two DenseElementsAttrs.
  std::pair<mlir::DenseElementsAttr, mlir::DenseElementsAttr>
  getAsElementsAttr(mlir::RankedTensorType type) const;

  /// Returns DenseElementsAttr representation if the element ranges are all
  /// constant (single-value) ranges, otherwise nullopt.
  std::optional<mlir::DenseElementsAttr>
  getConstantValues(mlir::RankedTensorType type) const;

  /// The maximum allowed volume of a tensor that we allow tracking the value
  /// of. This is used to avoid edge cases where tracking the bounds would
  /// require an excess amount of memory.
  static constexpr int64_t kMaxVolumeThreshold = 32;

  /// Whether the analysis should consider a value. To consider
  /// a value, it must be a ranked tensor of static shape and signless-or-index
  /// integer element type and have a total volume <= kMaxVolumeThreshold.
  static bool shouldAnalyzeValueBounds(mlir::Type type);

  /// Whether the analysis should consider a value. To consider
  /// a value, it must be a ranked tensor of static shape and signless-or-index
  /// integer element type and have a total volume <= kMaxVolumeThreshold.
  static bool shouldAnalyzeValueBounds(mlir::Value value);

private:
  std::optional<llvm::SmallVector<mlir::ConstantIntRanges>> value;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BoundsArray &v);

/// Represents either a BoundsArray lattice or a InterValueRange lattice.
struct IntOrTensorValueRange
    : public llvm::PointerUnion<const BoundsArray *,
                                const mlir::IntegerValueRange *> {
  using PointerUnion::PointerUnion;
};

/// Similar to SetIntRangeFn, but operating on IntegerValueRange lattice values.
/// This is the `setResultRanges` callback for the BoundsArray based
/// interface method.
using SetTensorValueLatticeFn =
    llvm::function_ref<void(mlir::Value, BoundsArray)>;

class InferTensorValueRangeInterface;

namespace detail {} // namespace detail

} // namespace mlirtrt::compiler

#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h.inc"

#endif // MLIR_TENSORRT_INTERFACES_INFERTENSORVALUERANGEINTERFACE
