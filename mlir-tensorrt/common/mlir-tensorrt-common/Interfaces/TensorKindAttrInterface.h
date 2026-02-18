//===- TensorKindAttrInterface.h ------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_INTERFACES_TENSORKINDATTRINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_TENSORKINDATTRINTERFACE

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
namespace mlir {

enum class TensorKind { Unknown = 0, Device, Host, Both };

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
  TensorKindInfo();
  TensorKindInfo(TensorKind initialKind);

  /// Print a description of this lattice point to the stream.
  void print(raw_ostream &os) const;

  /// Set the value and return whether there was a change.
  ChangeResult setKind(TensorKind kind);

  /// Return whether this lattice point has an uninitialied value.
  bool isUninitialized() const;

  /// Return whether this lattice point is unknown.
  bool isUnknown() const;

  static TensorKindInfo join(const TensorKindInfo &lhs,
                             const TensorKindInfo &rhs);
  static TensorKindInfo meet(const TensorKindInfo &lhs,
                             const TensorKindInfo &rhs);

  TensorKind getKind() const;

  /// Return true if the kind has `Host` value.
  bool isHostOnly() const;

  /// Returns true if the kind is `host` or `both`.
  bool isHostVisible() const;

  /// Return true if this is an device tensor and not 'both'.
  bool isDeviceOnly() const;

  /// Returns true if  this is a tensor visible on the device (device or both).
  bool isDeviceVisible() const;

  /// Return true if the kind is both a host tensor and an device tensor.
  bool isBothHostAndDevice() const;

  bool operator<(const TensorKindInfo &other) const;

  bool operator==(const TensorKindInfo &other) const;
  bool operator!=(const TensorKindInfo &other) const;

  std::optional<TensorKind> kind;
};

llvm::StringRef stringifyTensorKind(TensorKind kind);

/// Return the name of the function arg attr UnitAttr
/// that should be used to mark an argument as a shape tensor.
StringRef getHostTensorArgAttrName();

} // namespace mlir

#include "mlir-tensorrt-common/Interfaces/TensorKindAttrInterface.h.inc"

#endif // MLIR_TENSORRT_COMMON_INTERFACES_TENSORKINDATTRINTERFACE
