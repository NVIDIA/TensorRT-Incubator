//===- Passes.td  ---------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StablehloExtSimplificationsPass
//===----------------------------------------------------------------------===//

def StablehloExtSimplificationsPass : Pass<"stablehlo-ext-simplifications"> {
  let summary = "Applies Stablehlo device-agnostic simplifications along with additional patterns";
  let description = [{
    This pass runs the patterns defined by
    `stablehlo-aggressive-simplification` along with additional custom
    patterns not yet upstreamed.
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect"
  ];

  let options = [
    Option<"foldOpElementLimit", "fold-op-element-limit", "int64_t", "1024",
      "The computation size limit for constant folding">
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
// ConstantFoldingPass
//===----------------------------------------------------------------------===//

def ConstantFoldingPass : Pass<"stablehlo-ext-constant-folding"> {
  let summary = "runs extra constant-folding patterns on StableHLO IR";

  let description = [{

    Applies constant-folding rewrites to Stable HLO operations in addition to
    the folders provided by the Stable HLO dialect. In particular, this pass
    is more aggressive (and expensive) than the Stable HLO folders. It folds
    transpose operations aggressively, even if cloning is required, since that
    often benefits performance when offloading to opaque downstream compilers
    like TensorRT when the constant is used as an operand to a compuatation such
    as matrix multiplication.

    In addition to the more aggressive folding, it the provided folding routines
    can handle simulation of folding of `dense_resource<__elided__>` attributes.
    This makes it easier to run tests and debug when running pipelines on
    input IR that has all its large constants elided. Otherwise,the results of
    a pipeline could look very different when running on IR that has elided vs
    non-elided attributes.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];

  let options = [
    Option<"constantFoldSizeLimit", "constant-fold-size-limit", "int64_t",
     "65536", "The computation size limit for constant folding.">
  ];
}

//===----------------------------------------------------------------------===//
// RefineShapesPass
//===----------------------------------------------------------------------===//

def RefineShapesPass : Pass<"stablehlo-ext-refine-shapes", "ModuleOp"> {
  let summary = "refines the shapes of stablehlo operations";

  let description = [{
    Performs shape refinement. This pass includes the upstream
    `stablehlo-refine-shapes` patterns as well as some additional patterns
    for handling `tensor.cast` operations.
  }];

  let options = [
    Option<"interprocedural", "interprocedural", "bool", "true",
      "whether to try to simplify function types">
  ];
}

//===----------------------------------------------------------------------===//
// CanonicalizeShapesPass
//===----------------------------------------------------------------------===//

def CanonicalizeShapesPass : Pass<"stablehlo-ext-canonicalize-shapes", "ModuleOp"> {
  let summary = "iteratively canonicalizes dynamic shape op variants and refines shapes";

  let description = [{
    This pass uses runs a dynamic pipeline to perform dynamic op canonicalization
    (`stablehlo-canonicalize-dynamism`) along with shape refinement
    (`stablehlo-ext-refine-shapes`) iteratively until a fixed point is reached or
    until the `maxIterations` is exceeded. Failure to converge within `maxIterations`
    is currently considered an error.
  }];

  let options = [
    Option<"maxIterations", "max-iterations", "int64_t", "8",
      "the maximum number of iterations to run the dynamism simplification and "
      "shape refinement if a fixed-point is not reached">,
    Option<"interprocedural", "interprocedural", "bool", "true",
      "whether to try to simplify function types">
  ];
}

//===----------------------------------------------------------------------===//
// GatherToSlicePass
//===----------------------------------------------------------------------===//

def GatherToSlicePass : Pass<"stablehlo-gather-to-slice"> {
  let summary = "converts slice-like stablehlo.gather to stablehlo.slice";
  let description = [{
    The `stablehlo.gather` operation is capable of representing static
    strided slice operations. Certain frontends (e.g. JAX) may use fact
    to represent slice operations using `stablehlo.gather`. This pass
    matches such patterns and rewrites them to `stablehlo.slice`.
  }];
}

//===----------------------------------------------------------------------===//
// CanonicalizeDotGeneralPass
//===----------------------------------------------------------------------===//

def CanonicalizeDotGeneralPass : Pass<"stablehlo-canonicalize-dot-general"> {
  let summary = "performs canonicalizations of stablehlo.dot_general";

  let description = [{

    Performs canonicalizations of stablehlo.dot_general such that:

    1. If there are an unbalanced number of M/N dimensions (e.g.
      'outer-product dimensions') then collapsing reshapes are inserted
      in order to create an equal number of M/N dimensions.

    2. Inserts transpose operations to ensure tha the batch dims
       are the leading dims of each operand.

    3. Some `stablehlo.dot_general` operations are actually a pure
       element-wise multiplication. Recognize such ops and lower them
       to `stablehlo.mul`.

  }];
}

//===----------------------------------------------------------------------===//
// CanonicalizeGatherPass
//===----------------------------------------------------------------------===//

def CanonicalizeGatherPass : Pass<"stablehlo-ext-canonicalize-gather"> {
  let summary = "Rewrites gather into transposes, reshapes and a simple gather.";
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

//===----------------------------------------------------------------------===//
// ExpandTuplesPass
//===----------------------------------------------------------------------===//

def ExpandTuplesPass : Pass<"stablehlo-ext-expand-tuples", "ModuleOp"> {
  let summary = "Expand Stable HLO tuple for the entry function of the module.";
  let options = [
    Option<"entryFunctionName", "entry-function-name", "std::string",
           /*default=*/"\"main\"", "the name of entry function of the module">,
  ];

  let dependentDialects = ["::mlir::stablehlo::StablehloDialect"];
}

//===----------------------------------------------------------------------===//
// CanonicalizeScatterPass
//===----------------------------------------------------------------------===//

def CanonicalizeScatterPass : Pass<"stablehlo-ext-canonicalize-scatter"> {
  let summary = "Rewrites scatter into transposes, reshapes and a simple scatter.";
  let dependentDialects = ["stablehlo::StablehloDialect", "tensor::TensorDialect"];
}

//===----------------------------------------------------------------------===//
// LowerSpecialCustomCalls
//===----------------------------------------------------------------------===//

def LowerSpecialCustomCalls : Pass<"stablehlo-ext-lower-special-custom-calls"> {
  let summary = "Rewrites special `stablehlo.custom_call` operations that are emitted "
    "by frontends like JAX.";
  let description = [{

    Certain frontends like JAX emit special custom calls in order to:

    1. To represent CHLO operations. The StableHLO/VHLO dialects don't encompass CHLO.
       Therefore, an ad-hoc method is used by JAX where a CHLO opeation is serialized
       as a `stablehlo.custom_call`. These can get lowered directly to corresponding
       CHLO operations. Note that this breaks portability.

    2. For inserting sharding annotations, JAX emits `stablehlo.custom_call @Sharding`
       and attaches an attribute containing the string representation of the XLA
       sharding attribute. At the point in the pipeline where this pass is being
       invoked, we can't handle such annotations unless they are no-ops/identity shardings,
       which are sometimes inserted by JAX even for single-gpu programs.
       This pass looks for these and operations and eliminates them if they represent
       no-ops, otherwise it will fail. A different pass upstream in the
       pipeline would have to perform partitioning of the program to handle other
       cases.
  }];
  let dependentDialects = ["::mlir::chlo::ChloDialect"];
}

//===----------------------------------------------------------------------===//
// CanonicalizeConvolutionPass
//===----------------------------------------------------------------------===//

def CanonicalizeConvolutionPass : Pass<"stablehlo-ext-canonicalize-convolution"> {
  let summary = "Canonicalizes stablehlo convolution operations";
}

//===----------------------------------------------------------------------===//
// TargetSpecificOptimizationsPass
//===----------------------------------------------------------------------===//

def TargetSpecificOptimizationsPass : Pass<"stablehlo-ext-target-specific-optimizations"> {
  let summary = "Canonicalizes stablehlo operations";
  let options = [
    ListOption<"disablePatterns", "disable-patterns", "std::string",
      "Comma-separated list of target-specific optimization patterns to disable. "
      "Available patterns: "
      "dot-general, gather, scatter, convolution, gather-to-slice">,
    Option<"constantFoldSizeLimit", "constant-fold-size-limit", "int64_t",
     "65536", "The computation size limit for constant folding.">
  ];
}

//===----------------------------------------------------------------------===//
// MaterializeDenseResourcePass
//===----------------------------------------------------------------------===//

def MaterializeDenseResourceElementsAttrPass : Pass<"stablehlo-ext-materialize-dense-resource-elements-attr"> {
  let summary = "Materializes `DenseElementsAttr` into `DenseResourceElementsAttr` in"
    "`stablehlo.constant` operations.";
  let dependentDialects = ["stablehlo::StablehloDialect"];

  let options = [
    Option<"elementCountThreshold", "element-count-threshold", "int64_t", "8",
      "Optional threshold for converting `DenseElementsAttr` to `DenseResourceElementsAttr`. "
      "Conversion only occurs if number of elements is greater than or equal to this value. "
      "Defaults to 8 if not specified.">
  ];
}

//===----------------------------------------------------------------------===//
// StablehloLowerCheckCustomCallsPass
//===----------------------------------------------------------------------===//

def StablehloLowerCheckCustomCallsPass : Pass<"stablehlo-ext-lower-check-custom-calls",
      "::mlir::ModuleOp"> {
  let summary = "Lowers `stablehlo.custom_call` operations like 'check.eq' used in tests";

  let description = [{
    The Stablehlo repo has a large set of MLIR files which were
    generated from JAX test programs. These files contain Stablehlo
    programs in a particular format. For comparing a tensor SSA value
    against an expected value, they use special
    `stablehlo.custom_call` operations with call target names like
    'check.eq'. Their semantics correspond to the 'Check' MLIR dialect
    operations defined in the Stablehlo repo (e.g.
    `third_party/stablehlo/stablehlo/tests/CheckOps.td`).

    Since MLIR-TensorRT doesn't have a built-in set of operations
    corresponding to the 'Check' dialect operations, this pass lowers
    the `stablehlo.custom_call` operations targeting `check.eq`
    operations by generating and inserting MLIR functions which
    implement the semantics of the 'Check' op and replacing the
    `stablehlo.custom_call` with a `func.call` operation.

    ## Example Input

    ```mlir
    func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @inputs() : () -> tensor<20x20xf32>
      %1 = call @expected() : () -> tensor<20x20xf32>
      %2 = stablehlo.abs %0 : tensor<20x20xf32>
      stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
      return %2 : tensor<20x20xf32>
    }
    func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
      %cst = stablehlo.constant dense<__elided__> : tensor<20x20xf32>
      return %cst : tensor<20x20xf32>
    }
    func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
      %cst = stablehlo.constant dense<__elided__> : tensor<20x20xf32>
      return %cst : tensor<20x20xf32>
    }
    ```

    ## Example Output

    ```mlir
    func.func @check_expect_close_0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                                   %min_ulp: tensor<i64>, %max_ulp: tensor<i64>) attributes {no_inline} {
      // For floating-point comparison based on ULP (Units in Last Place) distance.
      // The implementation handles special cases: bitwise equal values, NaNs, and finite values.
      // Reference: third_party/stablehlo/stablehlo/tests/CheckOps.cpp::ULPDifference

      // 1. Check if values are bitwise equal
      %actual_bits = stablehlo.bitcast_convert %arg0 : (tensor<?x?xf32>) -> tensor<?x?xi32>
      %expected_bits = stablehlo.bitcast_convert %arg1 : (tensor<?x?xf32>) -> tensor<?x?xi32>
      %bitwise_equal = stablehlo.compare EQ, %actual_bits, %expected_bits : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>

      // 2. Check if both values are NaN (isNaN = !isFinite)
      %actual_is_finite = stablehlo.is_finite %arg0 : (tensor<?x?xf32>) -> tensor<?x?xi1>
      %actual_is_nan = stablehlo.not %actual_is_finite : tensor<?x?xi1>
      %expected_is_finite = stablehlo.is_finite %arg1 : (tensor<?x?xf32>) -> tensor<?x?xi1>
      %expected_is_nan = stablehlo.not %expected_is_finite : tensor<?x?xi1>
      %both_nan = stablehlo.and %actual_is_nan, %expected_is_nan : tensor<?x?xi1>

      // 3. For finite values, compute ULP distance
      // Convert to absolute value in float domain, then bitcast (matching reference impl)
      %abs_actual = stablehlo.abs %arg0 : tensor<?x?xf32>
      %abs_expected = stablehlo.abs %arg1 : tensor<?x?xf32>
      %abs_actual_bits = stablehlo.bitcast_convert %abs_actual : (tensor<?x?xf32>) -> tensor<?x?xi32>
      %abs_expected_bits = stablehlo.bitcast_convert %abs_expected : (tensor<?x?xf32>) -> tensor<?x?xi32>

      // Check if original values have different signs
      %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
      %actual_negative = stablehlo.compare LT, %arg0, %zero_f32 : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
      %expected_negative = stablehlo.compare LT, %arg1, %zero_f32 : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
      %different_sign = stablehlo.compare NE, %actual_negative, %expected_negative : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>

      // ULP distance when same sign: |abs_actual_bits - abs_expected_bits|
      %ulp_diff_same_sign = stablehlo.subtract %abs_actual_bits, %abs_expected_bits : tensor<?x?xi32>
      %ulp_diff_same_sign_abs = stablehlo.abs %ulp_diff_same_sign : tensor<?x?xi32>

      // ULP distance when different sign: abs_actual_bits + abs_expected_bits
      %ulp_diff_diff_sign = stablehlo.add %abs_actual_bits, %abs_expected_bits : tensor<?x?xi32>

      // Select based on sign difference
      %ulp_diff_i32 = stablehlo.select %different_sign, %ulp_diff_diff_sign, %ulp_diff_same_sign_abs : tensor<?x?xi1>, tensor<?x?xi32>
      %ulp_diff = stablehlo.convert %ulp_diff_i32 : (tensor<?x?xi32>) -> tensor<?x?xi64>

      // 4. Check if ULP difference is within bounds [min_ulp, max_ulp]
      %min_ulp_bcast = stablehlo.broadcast_in_dim %min_ulp, dims = [] : (tensor<i64>) -> tensor<?x?xi64>
      %max_ulp_bcast = stablehlo.broadcast_in_dim %max_ulp, dims = [] : (tensor<i64>) -> tensor<?x?xi64>
      %ulp_ge_min = stablehlo.compare GE, %ulp_diff, %min_ulp_bcast : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
      %ulp_le_max = stablehlo.compare LE, %ulp_diff, %max_ulp_bcast : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
      %ulp_in_range = stablehlo.and %ulp_ge_min, %ulp_le_max : tensor<?x?xi1>

      // 5. Combine all conditions: bitwise_equal OR both_nan OR (both_finite AND ulp_in_range)
      %both_finite = stablehlo.and %actual_is_finite, %expected_is_finite : tensor<?x?xi1>
      %finite_and_close = stablehlo.and %both_finite, %ulp_in_range : tensor<?x?xi1>
      %is_close_temp = stablehlo.or %bitwise_equal, %both_nan : tensor<?x?xi1>
      %is_close = stablehlo.or %is_close_temp, %finite_and_close : tensor<?x?xi1>

      // 6. Reduce to check all elements
      %c1 = stablehlo.constant dense<true> : tensor<i1>
      %res = stablehlo.reduce (%is_close init: %c1)
        applies stablehlo.and across dimensions = [0, 1]
        : (tensor<?x?xi1>, tensor<i1>) -> tensor<i1>
      %0 = tensor.extract %res[] : tensor<i1>
      cf.assert %0, "check_expect_close failed"
      return
    }

    func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @inputs() : () -> tensor<20x20xf32>
      %1 = call @expected() : () -> tensor<20x20xf32>
      %2 = stablehlo.abs %0 : tensor<20x20xf32>
      %2_dynamic = tensor.cast %2 : tensor<20x20xf32> -> tensor<?x?xf32>
      %1_dynamic = tensor.cast %1 : tensor<20x20xf32> -> tensor<?x?xf32>
      func.call @check_expect_close_0(%2_dynamic, %1_dynamic) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
      return %2 : tensor<20x20xf32>
    }
    func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
      %cst = stablehlo.constant dense<__elided__> : tensor<20x20xf32>
      return %cst : tensor<20x20xf32>
    }
    func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
      %cst = stablehlo.constant dense<__elided__> : tensor<20x20xf32>
      return %cst : tensor<20x20xf32>
    }
    ```

    ## Additional Notes

    - Generated functions should be marked as 'no_inline' since inlining
      check operations could allow computation to be folded away, defeating
      the purpose of the test.

    - Checks which are "almost equal" either have a 'tolerance' attribute or
      use a default value of 1e-4.

    - Check dialect operations may not have an exact semantic specified
      in the Tablegen definition (e.g. what counts as "almost equal" or "close");
      to see exact semantics refer to the Stablehlo interpreter:
      - third_party/stablehlo/stablehlo/reference/Element.cpp

  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::chlo::ChloDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect"
  ];
}

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
