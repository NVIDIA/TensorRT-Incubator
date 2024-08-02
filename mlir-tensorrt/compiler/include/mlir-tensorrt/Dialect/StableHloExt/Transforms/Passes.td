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
    Option<"maxIterations", "max-iterations", "int64_t", "4",
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

def CanonicalizeGatherPass : Pass<"stablehlo-ext-canonicalize-gather", "func::FuncOp"> {
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

def CanonicalizeScatterPass : Pass<"stablehlo-ext-canonicalize-scatter", "func::FuncOp"> {
  let summary = "Rewrites scatter into transposes, reshapes and a simple scatter.";
  let dependentDialects = ["stablehlo::StablehloDialect", "tensor::TensorDialect"];
}

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
