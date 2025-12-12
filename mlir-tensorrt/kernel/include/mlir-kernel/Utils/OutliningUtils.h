//===- OutliningUtils.h ---------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Outlining transformation utilities.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_TRANSFORMS_OUTLINING_OUTLINING_H
#define MLIR_EXECUTOR_TRANSFORMS_OUTLINING_OUTLINING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

namespace func {
class FuncOp;
}

/// Encapsulates the result of outlining a `scf.forall` operation into a
/// function + a call-like operation.
struct ForallOutliningResult {
  Operation *forallReplacement;
  Operation *outlinedBody;
};

/// Outline a `scf.forall` operation's body into a function which is inserted
/// into `symbolTable`. The `scf.forall` operation itself is replaced by a
/// call-like operation which is constructed using `callBuilder`. The induction
/// variables of the `scf.forall` operation are replaced in the body prior to
/// outlining using `ivReplacementBuilder`.
FailureOr<ForallOutliningResult> outlineForall(
    RewriterBase &rewriter, scf::ForallOp op, StringRef name,
    SymbolTable &moduleSymbolTable,
    std::function<SmallVector<Value>(RewriterBase &, Location loc,
                                     ValueRange ivs, ArrayRef<OpFoldResult> ubs,
                                     std::optional<ArrayAttr>)>
        ivReplacementBuilder,
    std::function<Operation *(RewriterBase &, scf::ForallOp forallOp,
                              ValueRange args, func::FuncOp)>
        callBuilder,
    std::function<bool(Operation *)> cloneOperationIntoRegion =
        [](Operation *op) { return false; });

/// Construct a value corresponding to the `index-th` block coordinate of a GPU
/// thread block within a grid. The grid has a shape given by `ubs`. The mapping
/// consists of an optional set of `gpu::MappingIdAttr` attributes specifying
/// how dimensions should be calculated.
SmallVector<Value> getInductionVarReplacementsUsingGpuBlockId(
    RewriterBase &rewriter, Location loc, ValueRange ivs,
    ArrayRef<OpFoldResult> ubs, std::optional<ArrayAttr> mapping);

} // namespace mlir

#endif // MLIR_EXECUTOR_TRANSFORMS_OUTLINING_OUTLINING_H
