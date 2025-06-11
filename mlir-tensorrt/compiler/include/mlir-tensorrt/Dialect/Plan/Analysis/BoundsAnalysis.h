//===- BoundsAnalysis.h -----------------------------------------*- C++ -*-===//
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
/// Definition of dataflow analyses that calculate bounds on shape dimensions
/// (for dynamic shapes) and tensor values (e.g. shape tensors).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_BOUNDSANALYSIS
#define MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_BOUNDSANALYSIS

#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir::plan {

using BoundsArray = mlirtrt::compiler::BoundsArray;

//===----------------------------------------------------------------------===//
// Shape Bounds Analyses
//===----------------------------------------------------------------------===//

class ShapeBoundsLattice : public dataflow::Lattice<BoundsArray> {
public:
  using Lattice::Lattice;
};

class ShapeBoundsForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ShapeBoundsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// Populate bounds for entry values (e.g. function block arguments).
  void setToEntryState(ShapeBoundsLattice *lattice) override;

  /// Visit `op` and propagate bounds from `operands` to `results`.
  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const ShapeBoundsLattice *> operands,
                               ArrayRef<ShapeBoundsLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow.
  /// This function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<ShapeBoundsLattice *> argLattices,
                                    unsigned firstIndex) override;
};

class ShapeBoundsBackwardsAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<ShapeBoundsLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(ShapeBoundsLattice *lattice) override;

  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<ShapeBoundsLattice *> operands,
                 ArrayRef<const ShapeBoundsLattice *> results) override;
};

//===----------------------------------------------------------------------===//
// ShapeIntegerRangeAnalysis
//===----------------------------------------------------------------------===//

/// ShapeIntegerRangeAnalysis is a sparse forward dataflow analysis that
/// analyzes the value bounds of integer scalar-typed SSA values. It is exactly
/// equivalent to upstream IntegerRangeAnalysis with the addition of more
/// support for our special bounds attributes on function arguments as well
/// as `tensor.extract` and `tensor.dim`.
///
/// As in the other MLIR dataflow analyses in this file, "sparse" means that a
/// lattice state is associated with each SSA value. The entry states of
/// function BlockArguments are populated from the `tensorrt.value_bounds`
/// function argument attributes.
///
/// It is a "forward" dataflow analysis because bounds are propagated along the
/// flow of control of the program from operands to results, and the analysis
/// machinery iteratively updates bounds of an SSA value. Every SSA value
/// maintains a state, and the state is only updated by "joining" it with
/// an updated bound (in this case "join" = union).
///
/// The dataflow framework handles joining bounds of SSA values (op results
/// and block arguments) that are connected due to the flow of control as long
/// as the ops with control flow behavior (e.g. Func, SCF, CF ops) implement
/// the standard MLIR control flow interfaces.
///
/// While operations that implement control flow are handled automatically by
/// the dataflow framework, a transfer function must be defined for all other
/// operations that describes how the bounds of the operands propagate to the
/// bounds of the results. In the case of this analysis, the transfer function
/// is implemented by any operation that implements the upstream
/// InferIntRangeInterface.
class ShapeIntegerRangeAnalysis : public dataflow::IntegerRangeAnalysis {
public:
  using IntegerRangeAnalysis::IntegerRangeAnalysis;

  /// At an entry point, we cannot reason about integer value ranges except
  /// if we are dealing with  a function where the bounds are encoded into the
  /// function arg attributes.
  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) override;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override;
};

//===----------------------------------------------------------------------===//
// TensorValueBoundsAnalysis
//===----------------------------------------------------------------------===//

class TensorValueBoundsLattice : public dataflow::Lattice<BoundsArray> {
public:
  using Lattice::Lattice;
};

/// TensorBoundsAnalysis is a sparse forward dataflow analysis that analyzes
/// value bounds of integer tensor-typed SSA values. As in the other MLIR
/// dataflow analyses in this file, "sparse" means that a lattice state is
/// associated with each SSA value. The entry states of functions BlockArguments
/// are populated from the `tensorrt.value_bounds` function argument attributes.
///
/// The state associated with an SSA value is similar to that of the
/// IntegerRangeAnalysis. The state is considered uninitialized unless the Value
/// has an integer tensor type. The state simply consists of an array of integer
/// bounds (an array of `ConstantIntRanges`). Each element in the tensor is
/// mapped to a `ConstantIntRanges` by storing the `ConstantIntRanges` in a flat
/// array using a packed row-major layout.
///
/// It is a "forward" dataflow analysis because bounds are propagated along the
/// flow of control of the program from operands to results. The analysis
/// machinery iteratively updates bounds of an SSA value. Every SSA value
/// maintains a state, and the state is only updated by "joining" it with
/// an updated bound (in this case "join" = union).
///
/// The dataflow framework handles joining bounds of SSA values (op results
/// and block arguments) that are connected due to the flow of control as long
/// as the ops with control flow behavior (e.g. Func, SCF, CF ops) implement
/// the standard MLIR control flow interfaces.
///
/// While operations that implement control flow are handled automatically by
/// the dataflow framework, a transfer function must be defined for all other
/// operations that describes how the bounds of the operands propagate to the
/// bounds of the results. In the case of this analysis, the transfer function
/// is only implemented for a handful of functions that query the result of
/// `ShapeIntegerRangeAnalysis` and SCCP to update their state. This generally
/// includes operations that create or update tensors using scalars or
/// have simple clone-like semantics:
///   - `tensor.from_elements|splat`
///   - `tensor.insert`
///   - `tensor.reshape|expand_shape|collapse_shape`
///   - `bufferization.alloc_tensor`
///   - `bufferization.materialize_in_destination`
///   - `plan.with_values`
class TensorValueBoundsAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TensorValueBoundsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// Populate bounds for entry values (e.g. function block arguments).
  void setToEntryState(TensorValueBoundsLattice *lattice) override;

  /// Visit `op` and propagate bounds from `operands` to `results`.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const TensorValueBoundsLattice *> operands,
                 ArrayRef<TensorValueBoundsLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow.
  /// This function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<TensorValueBoundsLattice *> argLattices,
                               unsigned firstIndex) override;

  /// The maximum allowed volume of a tensor before we stop tracking its
  /// analyzing its value bounds.
  static constexpr int64_t kMaxVolumeThreshold = 32;

  /// Whether the analysis should consider a value. To consider
  /// a value, it must be a ranked tensor of static shape and signless-or-index
  /// integer element type and have a total volume <= kMaxVolumeThreshold.
  static bool shouldAnalyzeValueBounds(Type type);

  /// Whether the analysis should consider a value. To consider
  /// a value, it must be a ranked tensor of static shape and signless-or-index
  /// integer element type and have a total volume <= kMaxVolumeThreshold.
  static bool shouldAnalyzeValueBounds(Value value);
};

} // namespace mlir::plan

#endif // MLIR_TENSORRT_DIALECT_PLAN_ANALYSIS_BOUNDSANALYSIS
