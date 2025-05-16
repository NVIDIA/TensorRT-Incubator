//===- GatherScatterUtils.h -------------------------------------*- C++ -*-===//
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
/// Utilities for dealing with `stablehlo.scatter` and `stablehlo.gather`
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_GATHERSCATTERUTILS_H
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_GATHERSCATTERUTILS_H

#include "mlir-tensorrt/Utils/ShapeInfo.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {

class OpBuilder;

namespace stablehlo {
class DynamicGatherOp;
class GatherOp;
class ScatterOp;

} // namespace stablehlo
namespace stablehlo_ext {

//===----------------------------------------------------------------------===//
// GatherOp Categorization
//===----------------------------------------------------------------------===//

/// # Simple, Single Dimension Gather
///
/// We say that a `stablehlo.gather` operation is a "simple, single dimension
/// gather" if it has an index vector of size 1 and certain requirements are
/// met. Since the index vector is of size 1, then the `start_index_map`
/// contains a single operand dimension index which we call the "gather
/// dimension". The op must meet the following requirements:
///
/// - (C0) The `collapsed_slice_dims` contains a single dim index equal to the
///   "gather dimension".
/// - (C1) The `slice_sizes` match the shape of the input operand, except that
///   `slice_sizes[gather_dim] == 1`.
///
/// These requirements (plus the additional two below, which depend on
/// `index_vector_dim`) correspond to the semantic of a simple ONNX-like
/// 'gather' operation, since you are taking full slices of the operand in all
/// dimensions except for the "gather dimension", which is sliced with size 1
/// and indexed according to the indices from `start_indices`.
///
/// Note that a "simple, single dimension" gather can be in either "implicit
/// index dimension" or "explicit index dimension" form, and the following
/// additional requirements depend on which form it is in:
///
/// "Implicit index dimension form" as the following extra requirement:
///
/// - (C2) The `offset_dims` are a contiguous sequence ranging from
///   `rank(start_indices)` to `rank(result)`.
/// - (C3) The `index_vector_dim` is equal to the rank of the `start_indices`.
///
/// "Explicit index dimension form" as the following extra requirement:
///
/// - (C2) The `offset_dims` are a contiguous sequence ranging from
///   `rank(start_indices)-1` to `rank(result)`.
/// - (C3) The `index_vector_dim` is equal to `rank(start_indices) - 1` and
///   `shape(start_indices)[index_vector_dim] == 1`.
///
/// # Simple, Leading Multi-Dimension Gather
///
/// We say that a `stablehlo.gather` operation is a "simple, leading
/// multi-dimension gather" if it has an explicit index vector of size N and
/// certain requirements are met.
///
/// - (C0) The `collapsed_slice_dims` matches the contiguous sequence from 0
///   to `rank(start_indices) - 1`.
/// - (C1) The `start_index_map` matches `collapsed_slice_dims`.
/// - (C2) The `slice_sizes` match the shape of the input operand, except that
///   all leading dimensions corresponding to the `collapsed_slice_dims` are
///   set to `1`.
/// - (C3) The `offset_dims` is equal to the contiguous sequence from
///   `rank(start_indices)-1` to `rank(result)`.
/// - (C4) The size of the index vector dim is > 1.
///
/// The requirements say that the index vector indexes into the operand at the
/// leading dimensions and that the slices taken are size-1 in those dimensions
/// but full size everywhere else. In these cases the op corresponds closely to
/// the semantic of ONNX 'gather_nd'.
/// Note that requirement C4 makes it so that ops which are "simple, leading
/// multi-dimensional gather" do not intersect with explicit index form of
/// "simple, single-dimensional gather".
///
/// # Simple, Leading Multi-Dimension Gather with Degenerate Dims
///
/// This is the same as "Simple, Leading Multi-Dimensional Gather" except that
/// the size-1 slices are not collapsed. The requirements are:
///
/// - (C0) The `start_index_map` matches the contiguous sequence from 0
///   to `rank(start_indices) - 1`.
/// - (C1) The `collapsed_slice_dims` are empty.
/// - (C2) The `slice_sizes` match the shape of the input operand, except that
///   all leading dimensions corresponding to the `collapsed_slice_dims` are
///   set to `1`.
/// - (C3) The `offset_dims` is equal to the contiguous sequence from
///   `rank(start_indices)-1` to `rank(result)`.
/// - (C4) The size of the index vector dim is > 1.

/// Returns the "gather dimension" if `op` is a 'simple, single dimension'
/// gather op with implicit index vector dimension (see above for definition).
std::optional<int64_t>
isSingleDimSimpleGatherWithImplicitIndexDim(stablehlo::GatherOp op);

/// Returns the "gather dimension" if `op` is a 'simple, single dimension'
/// gather op with implicit index vector dimension (see above for definition).
/// This version works for `stablehlo.dynamic_gather` using pattern matching
/// against the expected canonical form when the operand shape along some
/// "offset dimensions" is dynamic.
std::optional<int64_t> isSingleDimSimpleGatherWithImplicitIndexDim(
    stablehlo::DynamicGatherOp op,
    const ShapeInfoCallbacks &shapeInfoCallbacks);

/// Returns the "gather dimension" if `op` is a 'simple, single dimension'
/// gather op with explicit size-1 index vector dimension (see above for
/// definition).
std::optional<int64_t>
isSingleDimSimpleGatherWithExplicitIndexDim(stablehlo::GatherOp op);

/// Returns true if the `op` corresponds to a 'simple, leading multi-dimensional
/// gather' (see definition above).
bool isSimpleLeadingMultiDimGather(stablehlo::GatherOp op);

/// Returns true if the `op` corresponds to a 'simple, leading multi-dimensional
/// gather' (see definition above).
bool isSimpleLeadingMultiDimGatherWithDegenerateDims(stablehlo::GatherOp op);

/// Attempts to construct a `stablehlo.reshape` if result type is statically
/// shaped, otherwise creates `stablehlo.dynamic_reshape`.
Value createCollapsingReshape(OpBuilder &b, Location loc, Value input,
                              ArrayRef<ReassociationIndices> reassociation);

/// Attempts to construct a `stablehlo.reshape` if `resultType` is statically
/// shaped, otherwise creates a `stablehlo.dynamic_reshape`.
Value createExpandingReshape(OpBuilder &b, Location loc,
                             RankedTensorType resultType, Value input,
                             ArrayRef<ReassociationIndices> reassociation);

/// Check that the "update_computation" region of a 'stablehlo.scatter' op
/// yields the "update" scalars directly.
bool checkUpdateComputationReturnsUpdateValues(stablehlo::ScatterOp op);

/// Returns true if the `scatterOp` has a configuration that corresponds to the
/// ONNX ScatterNd operation semantic.
///
/// The ONNX scatter ND semantic can be found here:
/// https://onnx.ai/onnx/operators/onnx__ScatterND.html
///
/// For `stablehlo.scatter` to represent `onnx.scatter_nd`, we require:
/// - (C0) the `index_vector_dim` must be the last dimension of the
///   `scatter_indices`.
/// - (C1) the `update_window_dims` must correspond to the a tail sequence
///   of dimensions of the `updates`, which should be
///   '[scatter_indices.rank - 1, ..., rank(updates)-1]`.
/// - (C2) `input_batching_dims|scatter_indices_batching_dims` are empty.
/// - (C3) `scatter_dims_to_operand_dims` must be identity permutation
///   which maps to start of dims(result). This means it must be
///   "[0, 1, ..., scatter_indices.shape[-1]-1]".
/// - (C4) there can't be overlap between non-zero dims in
///   "full start index" and non-zero dims in "full window index" (see
///   diagram at
///   https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter
///   for an example where overlap does occur in result dim 2).
///   The potential for overlap is a unique feature of
///   `stablehlo.scatter` that gives it more expressive power.
///   Requiring no overlap is checked by verifying that `inserted_window_dims`
///   is the sequence
///   "[0, 1, ..., scatter_indices.shape[-1]-1]" and C3.
/// - (C5) The rank of `scatter_updates` must be equal to
///   `rank(scatter_indices) - 1 + rank(input) - scatter_indices.shape[-1]`.
///   This final check may have some overlap with the previous 5 but is a good
///   initial sanity check.
bool isCanonicalScatterNd(stablehlo::ScatterOp scatterOp);

//===----------------------------------------------------------------------===//
// Code below this point was adapted from the MLIR-HLO project (part of OpenXLA
// project) `xla/mlir_hlo/mhlo/utils/mhlo_scatter_gather_utils.h` and has the
// original license: Apache License v2.0. See
// https://github.com/openxla/xla/blob/main/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// Checks if the scatter has the following characteristics:
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [0, 1, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
bool isCanonicalScatter(stablehlo::ScatterOp scatterOp);

// Checks if the gather has the following characteristics:
// - start_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - collapsed_slice_dims is []
// - offset_dims is [1, 2, ...]
// - start_index_map is [0, 1, ...]
bool isCanonicalGather(stablehlo::GatherOp gatherOp);

/// Expands the shape of `tensor`, inserting degenerate dimensions.
///
/// For example, tensor<10x4xf32> and dimsToInsert = {0, 2}
/// will result in tensor<1x10x1x4xf32>.
Value insertDegenerateDimensions(OpBuilder &b, Location loc, Value tensor,
                                 ArrayRef<int64_t> dimsToInsert);

// Given a map from index vector positions to dimension numbers, creates a
// permutation that when applied to the operand, let you replace the map with
// the identity permutation. Also returns its inverse. In gather, the map is
// called `start_index_map`. In scatter, it's `scatter_dims_to_operand_dims`.
std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
makeOperandStartIndexPermutations(ArrayRef<int64_t> dimMap, int operandRank);

// Insert transposes and reshapes to bring `indices` to the 2D shape, where
// the dim0 is the product of all dimensions that are not equal to
// `indexVectorDim` and dim1 is the index vector dim.
//
// Examples.
//
// [a, I, b] will be transposed to [a, b, I], then reshaped into [ab, I].
// [a, b] will be reshaped to [a, b, I(1)] and then reshaped into [ab, I(1)].
Value canonicalizeStartIndices(OpBuilder &b, Location loc, Value indices,
                               int64_t indexVectorDim);

} // namespace stablehlo_ext
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_GATHERSCATTERUTILS_H
