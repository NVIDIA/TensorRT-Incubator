//===- TensorKindOpInterface.h ----------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for the TensorKindOpInterface. See the
/// [`TensorKindAnalysis` documentation](docs/Analysis/TensorKindAnalysis.md)
/// for more information.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H
#define MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

enum class TensorKind { Unknown = 0, Device, Host, Both };

llvm::StringRef stringifyTensorKind(TensorKind kind);

/// Return the name of the function arg attr UnitAttr
/// that should be used to mark an argument as a shape tensor.
StringRef getHostTensorArgAttrName();

/// Represents a lattice point. The lattice picture looks like this:
///
/// ```
///     "both" (top)
///     /    \               (host and device states are at same level)
/// "host"  "device"
///    \     /
///     unknown
///       |
///      uninitialized (bottom)
///
/// ```
struct TensorKindInfo {
  /// Create the type info in an uninitialized state.
  TensorKindInfo() : kind(std::nullopt) {}
  TensorKindInfo(TensorKind initialKind) : kind(initialKind) {}

  /// Print a description of this lattice point to the stream.
  void print(raw_ostream &os) const;

  /// Set the value and return whether there was a change.
  ChangeResult setKind(TensorKind kind) {
    auto newKind = kind;
    ChangeResult result = isUninitialized() || newKind != *this->kind
                              ? ChangeResult::Change
                              : ChangeResult::NoChange;
    this->kind = newKind;
    return result;
  }

  /// Return whether this lattice point has an uninitialied value.
  bool isUninitialized() const { return !kind.has_value(); }

  /// Return whether this lattice point is unknown.
  bool isUnknown() const { return *kind == TensorKind::Unknown; }

  static TensorKindInfo join(const TensorKindInfo &lhs,
                             const TensorKindInfo &rhs);
  static TensorKindInfo meet(const TensorKindInfo &lhs,
                             const TensorKindInfo &rhs);

  TensorKind getKind() const {
    assert(!isUninitialized());
    return *kind;
  }

  /// Return true if the kind has `Host` value.
  bool isHostOnly() const {
    assert(!isUninitialized() && "expected initialized value");
    return getKind() == TensorKind::Host;
  }

  /// Returns true if the kind is `host` or `both`.
  bool isHostVisible() const {
    assert(!isUninitialized() && "expected initialized value");
    return getKind() == TensorKind::Host || getKind() == TensorKind::Both;
  }

  /// Return true if this is an device tensor and not 'both'.
  bool isDeviceOnly() const {
    assert(!isUninitialized() && "expected initialized value");
    return getKind() == TensorKind::Device;
  }

  /// Returns true if  this is a tensor visible on the device (device or both).
  bool isDeviceVisible() const {
    assert(!isUninitialized() && "expected initialized value");
    return getKind() == TensorKind::Device || getKind() == TensorKind::Both;
  }

  /// Return true if the kind is both a host tensor and an device tensor.
  bool isBothHostAndDevice() const {
    assert(!isUninitialized() && "expected initialized value");
    return getKind() == TensorKind::Both;
  }

  bool operator<(const TensorKindInfo &other) const {
    // Uninitialized value is always less than other.
    if (isUninitialized())
      return true;
    // Uninitialized other can't be greater than anything.
    if (other.isUninitialized())
      return false;
    // If we are unknown, we are less than any host/device/both.
    if (isUnknown())
      return static_cast<int32_t>(getKind()) <
             static_cast<int32_t>(other.getKind());
    // If we are known, we are less than 'both'.
    if ((isHostOnly() || isDeviceOnly()) && other.isBothHostAndDevice())
      return true;
    return false;
  }

  bool operator==(const TensorKindInfo &other) const {
    return other.kind == this->kind;
  }
  bool operator!=(const TensorKindInfo &other) const {
    return !(*this == other);
  }

  std::optional<TensorKind> kind;
};

/// Wraps the `TensorKindInfo` into a lattice class. This is required because
/// upstream `dataflow::Lattice` has some issues that make it incompatible with
/// backward analysis. TODO: fix the static switch for the `meet` operation
/// upstream so that we can use `dataflow::Lattice<>`.
class TensorKindLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;
  Value getPoint() const { return point.get<Value>(); }
  TensorKindInfo &getValue() { return value; }
  const TensorKindInfo &getValue() const {
    return const_cast<TensorKindLattice *>(this)->getValue();
  }
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    return join(static_cast<const TensorKindLattice &>(rhs).getValue());
  }
  ChangeResult meet(const AbstractSparseLattice &rhs) override {
    return meet(static_cast<const TensorKindLattice &>(rhs).getValue());
  }
  ChangeResult join(const TensorKindInfo &rhs) {
    TensorKindInfo newValue = TensorKindInfo::join(value, rhs);
    assert(TensorKindInfo::join(newValue, value) == newValue &&
           "expected `join` to be monotonic");
    assert(TensorKindInfo::join(newValue, rhs) == newValue &&
           "expected `join` to be monotonic");
    if (newValue == value)
      return ChangeResult::NoChange;
    value = newValue;
    return ChangeResult::Change;
  }
  ChangeResult meet(const TensorKindInfo &rhs) {
    TensorKindInfo newValue = TensorKindInfo::meet(value, rhs);
    if (newValue == value)
      return ChangeResult::NoChange;
    value = newValue;
    return ChangeResult::Change;
  }
  void print(raw_ostream &os) const override { value.print(os); }

private:
  TensorKindInfo value;
};

} // namespace mlir

// Include the generated interface declarations.
#include "mlir-tensorrt-dialect/Interface/TensorKindAttrInterface.h.inc"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h.inc"

#endif // MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H
