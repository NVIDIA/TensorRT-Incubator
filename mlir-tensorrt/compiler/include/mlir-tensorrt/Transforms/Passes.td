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

#ifdef MLIR_TENSORRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// StablehloInputPreprocessingPass
//===----------------------------------------------------------------------===//
def StablehloInputPreprocessingPass : Pass<"tensorrt-stablehlo-input-preprocessing"> {
  let summary = "Prepares Stable HLO dialect operations for conversion to TensorRT";

  let description = [{
   This pass contains a set of patterns for simplifying or transforming Stable HLO
   input IR so that conversion to the TensorRT dialect is more straightforward.

   In particular:

    - Simplify certain patterns commonly found in IR emitted for JAX programs
      but not covered by existing Stable HLO canonicalizations/transforms.

    - Prepare convolutions to be NCHW/FCRS format and have at least two
      "spatial" dimensions.

    - Canonicalize `stablehlo.scatter` operations so that they can be converted to
      `tensorrt.scatter` in a straightforward manner.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::stablehlo::StablehloDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StablehloRaiseQDQPass
//===----------------------------------------------------------------------===//

def StablehloRaiseQDQPass : Pass<"stablehlo-raise-qdq"> {
  let summary = "Recognizes Q(quantize) and DQ(dequantize) patterns in the StableHLO IR";
  let description = [{
    This pass matches Q(quantize) and DQ(dequantize) patterns in the StableHLO
    IR and raises these patterns to `stablehlo.composite` op where
    decomposition is a private function implementing Q or DQ.
  }];

}

//===----------------------------------------------------------------------===//
// TestStablehloMatchersPass to test different matchers for stablehlo
//===----------------------------------------------------------------------===//
def TestStablehloMatchersPass : Pass<"test-stablehlo-matchers"> {
  let summary = "tests pattern matching utilities for StableHLO.";
  let description =[{
   StableHLOMatchers.h defines matchers to raise to different patterns for
   dot-product attention, softmax, etc. This pass tests a matcher by looking
   for the `__matched__` attribute added by a matcher.
   }];
}
#endif // MLIR_TENSORRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// DropNestedModulesPass
//===----------------------------------------------------------------------===//
def DropNestedModulesPass : Pass<"drop-nested-modules", "::mlir::ModuleOp"> {
  let summary = "Drops any operation with the SymbolTable attribute directly"
    " nested within the top-level Module";
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
// SCFDetensorizeLoopsPass
//===----------------------------------------------------------------------===//
def SCFDetensorizeLoopsPass : Pass<"scf-detensorize-loops"> {
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

#endif // MLIR_TENSORRT_TRANSFORMS_PASSES_TD
