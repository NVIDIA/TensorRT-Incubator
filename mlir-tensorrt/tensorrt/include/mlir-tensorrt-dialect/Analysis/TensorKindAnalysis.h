//===- TensorKindAnalysis.h -------------------------------------*- C++ -*-===//
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
/// This file defines an analysis that tries to determine whether each tensor
/// SSA value that is part of a program is a "host" tensor or a "device" tensor
/// (or both). See the
/// [`TensorKindAnalysis` documentation](docs/Analysis/TensorKindAnalysis.md)
/// for more information.
///
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_TENSORRT_DIALECT_ANALYSIS_TENSORKINDANALYSIS
#define INCLUDE_MLIR_TENSORRT_DIALECT_ANALYSIS_TENSORKINDANALYSIS

#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {

namespace func {
class FuncOp;
}

/// A data flow analysis that tries to determines whether a tensor is a `host`
/// or `device` tensor (or both).
class TensorKindAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<TensorKindLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  /// Return whether the `operand` is known to require a certain placement (e.g.
  /// host or device) or whether the placement is unknown. This doesn't actually
  /// use TensorKindAnalysis (since it is a static property of operations) but
  /// is a place where we either 1) query TensorKindOpInterface if the operand's
  /// owner uses that interface or 2) lookup the info for certain upstream
  /// operations which don't implement this interface (as opposed to providing
  /// external interface definition).
  /// See https://github.com/llvm/llvm-project/issues/64212 for a discussion of
  /// why we should avoid using external models for dialects that we don't own.
  static TensorKind getStaticOperandTensorKind(OpOperand &operand);

  /// Visit the given operation and transfer info from results to operands.
  LogicalResult
  visitOperation(Operation *op, ArrayRef<TensorKindLattice *> operands,
                 ArrayRef<const TensorKindLattice *> results) override;

  /// Set the state of the given lattice point at region exit state. This
  /// function is just called on terminators of regions that are not owned by a
  /// CallInterface or RegionBranchOpInterface operation. For the most part (at
  /// least for our use cases) this means that this function is called  to
  /// initialize the `func.return` operands of a function. The current logic is
  /// that it will set each operand (corresponding to a func result) to an
  /// device tensor __unless__ the owning `func.func` has a corresponding
  /// result attribute named `tensorrt.host_tensor` (UnitAttr) set, in which
  /// case it is initialized as a host tensor.
  void setToExitState(TensorKindLattice *lattice) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;
};

} // namespace mlir

#endif // INCLUDE_MLIR_TENSORRT_DIALECT_ANALYSIS_TENSORKINDANALYSIS
