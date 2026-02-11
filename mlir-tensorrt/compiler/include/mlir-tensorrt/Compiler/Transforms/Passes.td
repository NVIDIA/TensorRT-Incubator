//===- Passes.td -------------------------------------------*- Tablegen -*-===//
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
#ifndef MLIR_TENSORRT_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// OutlineTensorRTOpsPass
//===----------------------------------------------------------------------===//

def OutlineTensorRTOpsPass : Pass<"outline-tensorrt-ops",
      "::mlir::ModuleOp"> {
  let summary = "Segment and outline all TensorRT Dialect ops into a 'tensorrt.module'";

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// DropNestedModulesPass
//===----------------------------------------------------------------------===//
def DropNestedModulesPass : Pass<"drop-nested-modules", "::mlir::ModuleOp"> {
  let summary = "Drops any operation with the SymbolTable attribute directly"
    " nested within the top-level Module";
}

//===----------------------------------------------------------------------===//
// LinalgElementwiseFusionPass
//===----------------------------------------------------------------------===//

def LinalgElementwiseFusionPass : Pass<"mtrt-linalg-elementwise-fusion"> {
  let summary =
    "Performs elementwise fusion on linalg operations specific to MLIR-TensorRT";

  let description = [{
    Runs linalg elementwise fusion with a control function that is specific
    to MLIR-TensorRT.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MemRefCastEliminationPass
//===----------------------------------------------------------------------===//
def MemRefCastEliminationPass : Pass<"memref-cast-elimination"> {
  let summary = "Runs auxillary patterns to eliminate `memref.cast` operations";

  let description = [{
    Attempts to eliminate `memref.cast` operations where they can be statically
    determined to be superfluous but for whatever reason are not being
    removed by canonicalization.
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFDetensorizePass
//===----------------------------------------------------------------------===//

def SCFDetensorizePass : Pass<"mtrt-scf-detensorize"> {
  let summary = "try to replace loop-carried scalar tensors with scalar-typed values";

  let description = [{

    Currently this pass only supports `scf.while`. It rewrites loop-carried SSA
    tensor values to scalar values if the below conditions are true:

    - The tensor type has one element.

    - In the 'before' region, the relevant block arguments are either forwarded
      to the successor region) or are only accessed via `tensor.extract`.

    - In the `after` region, the relevant block arguments are used by
      `tensor.extract` and the input to the same position in the yield
      terminator is defined by a `tensor.from_elements` op.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFUnrollPass
//===----------------------------------------------------------------------===//

def SCFUnrollPass : Pass<"mtrt-scf-unroll"> {
   let summary = "Unroll for loops with static trip count";

   let description = [{
    Unroll `scf.for` loops with static trip count. Loops are fully unrolled
    if they have a cost below the `unroll-threshold`.

    Currently the cost is estimated by counting the number of operations in the
    loop body and multiplying it by the trip count.
   }];

   let dependentDialects = ["::mlir::scf::SCFDialect"];

   let options = [
    Option<"unrollThreshold", "unroll-threshold", "uint64_t",
    /*default=*/"100",
    "Unroll for loops if the cost is below this limit">
   ];
}

//===----------------------------------------------------------------------===//
// SCFForCyclicIterArgsPass
//===----------------------------------------------------------------------===//

def SCFForCyclicIterArgsPass : Pass<"mtrt-scf-for-cyclic-iter-args"> {
  let summary =
    "Eliminate scf.for iter args that only participate in cyclic shifts";

  let description = [{
    Identifies groups of `scf.for` iter args that are yielded as a cyclic
    permutation of the same block arguments. The loop-carried args are removed,
    and uses inside the loop are replaced with values selected from the init
    args based on the iteration index. The loop results are rewritten similarly
    based on the trip count.
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFFloatStrengthReducePass
//===----------------------------------------------------------------------===//

def SCFFloatStrengthReducePass : Pass<"mtrt-scf-float-strength-reduce"> {
  let summary = "Convert float loop-carried variables to integer counters";

  let description = [{
    Identifies `scf.while` loops with floating-point loop-carried variables that
    are updated with constant additive steps (e.g., `arith.addf %iv, %const`).
    Transforms these loops to use integer counters with appropriate scale
    factors, and attempts to uplift to `scf.for` when possible.

    For example, a loop counting down from 1.0 to 0.05 with step -0.1 can be
    transformed to an integer loop counting from 10 to 1 with step -1, with
    a scale factor of 0.1 applied when the float value is needed.

    This transformation improves performance by:
    1. Using faster integer arithmetic for loop control
    2. Enabling further optimizations like loop unrolling
    3. Making loop bounds statically analyzable
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuncExtDuplicateFunctionEliminationPass
//===----------------------------------------------------------------------===//

def FuncExtDuplicateFunctionEliminationPass : Pass<"func-ext-duplicate-function-elimination",
  "::mlir::ModuleOp"> {
  let summary = "Deduplicates func.func operations";

  let description = [{
    This pass is based on the upstream `duplicate-function-elimination` pass
    and modified to handle more general cases and fix a couple bugs.

    An open TODO is to upstream these changes to the `duplicate-function-elimination` pass.

    This pass attempts to identify `func.func` that are equivalent representations
    of deduplicates them. It does this by first scanning all `func::FuncOp` operations
    (even those in nested symbol tables) and attempts to identify unique instances
    within each SymbolTable scope. Functions are only deduplicated within the scope
    of a single SymbolTable, but the pass will look at all scopes from inner to outer.

    It creates equivalence classes and rewrites the call operations to use only one
    func per class (which is selected to be the first one from each equivalence class).
    The other functions are discarded. A function is only selected for removal if
    we can be sure that all references to the symbol can be safely updated.

    Note that this pass does not try to reconcile argument or result attributes if
    two func.func are identical except for argument attributes, then they will be
    considered separate functions.
  }];
}

//===----------------------------------------------------------------------===//
// TensorExtPadToInsertSlicePass
//===----------------------------------------------------------------------===//

def TensorExtPadToInsertSlicePass : Pass<"tensor-ext-pad-to-insert-slice"> {
  let summary =
    "Convert tensor.pad to linalg.fill + tensor.insert_slice";

  let description = [{
    This pass converts `tensor.pad` operations to `linalg.fill` and
    `tensor.insert_slice` operations.
  }];

  let dependentDialects = [
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LinalgSimplifyExtractSlicePass
//===----------------------------------------------------------------------===//

def LinalgSimplifyExtractSlicePass : Pass<"linalg-simplify-extract-slice"> {
  let summary =
    "Runs auxillary patterns to eliminate `tensor.extract_slice` operations";

  let description = [{
    This pass tries to eliminate `tensor.extract_slice` operations in special
    cases not covered by the existing `tensor.extract_slice` transforms upstream
    (e.g. fusion and sequential tensor.extract_slice elimination). In
    particular, it removes `tensor.extract_slice` by fusing with a
    `TilingInterface` operation when both the slice op and the producer are in
    the same block and the producer has one use.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanExecuteConstantFoldableSubgraphs
//===----------------------------------------------------------------------===//

def PlanExecuteConstantFoldableSubgraphsPass : Pass<
      "plan-execute-constant-foldable-subgraphs",
      "::mlir::ModuleOp"> {
  let summary = "Compile and execute constant foldable subgraphs";

  let description = [{
    This pass compiles and executes e2e constant foldable subgraphs
    outlined by `PlanOutlineConstantFoldableSubgraphsPass`. In first step, outlined
    `func.func` is cloned into a separate `ModuleOp` and e2e execution happens.
    In second step, operands of `func.return` in the original `func.func`
    are replaced with the constants created from results of e2e execution.
    Inlining pass running after this pass will eventually replace
    `func.call` ops created in `PlanOutlineConstantFoldableSubgraphsPass`
    with result of constant folding entire subgraph.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect",
    "::mlir::cuda::CUDADialect",
    "::mlir::executor::ExecutorDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
  ];
}

#endif // MLIR_TENSORRT_TRANSFORMS_PASSES_TD
