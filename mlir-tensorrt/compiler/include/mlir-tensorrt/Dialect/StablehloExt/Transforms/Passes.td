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
      "shape refinement if a fixed-point is not reached">
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
    ListOption<"patternSetNames", "patterns", "std::string",
      "Comma-separated list of target-specific optimization patterns to enable. "
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
#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
