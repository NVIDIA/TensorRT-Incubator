//===- TensorKindOpInterface.h ----------------------------------*- C++ -*-===//
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
/// Declarations for the TensorKindOpInterface. See the
/// [`TensorKindAnalysis` documentation](docs/Analysis/TensorKindAnalysis.md)
/// for more information.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H
#define MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H

#include "mlir-tensorrt-common/Interfaces/TensorKindAttrInterface.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {

llvm::StringRef stringifyTensorKind(TensorKind kind);

/// Return the name of the function arg attr UnitAttr
/// that should be used to mark an argument as a shape tensor.
StringRef getHostTensorArgAttrName();

/// Wraps the `TensorKindInfo` into a lattice class. This is required because
/// upstream `dataflow::Lattice` has some issues that make it incompatible with
/// backward analysis. TODO: fix the static switch for the `meet` operation
/// upstream so that we can use `dataflow::Lattice<>`.
class TensorKindLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;
  Value getPoint() const { return this->getAnchor(); }
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

namespace detail {
/// Returns true if the given value is a candidate for a host tensor based on
/// its type information. It must be a statically-shaped integer tensor with
/// fewer than 8 elements.
bool isHostTensorCandidate(Type type);

} // namespace detail

} // namespace mlir

// Include the generated interface declarations.
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h.inc"

#endif // MLIR_TENSORRT_INTERFACE_TENSORKINDOPINTERFACE_H
