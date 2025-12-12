//===- TensorUtils.h ------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Extra utilities for the Tensor dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_TENSORUTILS_H
#define MLIR_TENSORRT_UTILS_TENSORUTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;
class OpFoldResult;
namespace scf {
class ForallOp;
}
namespace tensor {
class ExpandShapeOp;
class ExtractSliceOp;
} // namespace tensor

namespace tensor_ext {

/// Try to materialize a slice of a `tensor.expand_shape` op by slicing the
/// source of the reshape instead of the result. It will return the new slice
/// in canonical rank-reduced form.
FailureOr<tensor::ExtractSliceOp>
materializeSliceOfExpandShape(RewriterBase &rewriter, tensor::ExpandShapeOp op,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides);

/// This pattern is the same as `tensor:replaceExtractSliceWithTiledProducer`
/// except that it actually performs a replacement. In addition, it handles
/// cases where the producer does not implement `TilingInterface` such as
/// special reshape producer cases.
FailureOr<TilingResult> replaceExtractSliceWithTiledProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp, OpResult producer);

/// Attempt to fuse the producer of a slice of a `scf.forall` `shared_outs`
/// argument (`blockArg`). On success, a pair (original `shared_outs` tied
/// operand, the replacement for `sliceOp`) are returned. This function ensures
/// that the fused producer's dest operands are correctly updated.
FailureOr<std::pair<OpResult, OpResult>> tryToFuseThroughSharedOutsBlockArg(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp,
    scf::ForallOp forallOp, BlockArgument blockArg);

/// Fuse all fusable producers greedily into the given `target`
/// op. It runs in an iterative manner until nothing can be further fused.
LogicalResult fuseGreedily(Operation *target, RewriterBase &rewriter,
                           bool removeProducer = false);

/// Includes patterns to interchange `scf.for` and `scf.forall` so that
/// the `scf.forall` is outer-most. This is possible if the `scf.forall`
/// effectively implements a subset insertion/extraction on the `scf.for`
/// iteration arguments. In that case, we can hoist the `scf.forall` up
/// without changing the semantics of the program.
///
/// Finds instances of the following pattern:
///
/// ```
/// %result = for ... iter_args(%iter = $init) {
///   %update = forall ... outs(%out = %iter) {
///     %0 = extract_slice %out[%o][%s][1]
///     ...
///     %tile = ...
///     parallel_insert_slice %tile into %out[%o][%s][1]
///   }
///   yield %update
/// }
/// ```
///
/// And produces:
///
/// ```
/// %result = forall ... outs(%out = %init) {
///   %0 = extract_slice %out[%o][%s][1]
///   %1 = for ... iter_args(%iter = $0) {
///     ...
///     %tile = ...
///     yield %tile
///   }
///   parallel_insert_slice %1 into %out[%o][%s][1]
/// }
/// ```
///
/// Note that `forOp` does not have to be provided. The `forOp` to interchange
/// is discovered by inspecting `forallOp`. If `forOp` is provided, then the
/// function returns failure if the discovered op does not match `forOp`.
LogicalResult interchangeForallAndFor(RewriterBase &rewriter,
                                      scf::ForallOp forallOp,
                                      scf::ForOp forOp = {});

} // namespace tensor_ext
} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_TENSORUTILS_H
