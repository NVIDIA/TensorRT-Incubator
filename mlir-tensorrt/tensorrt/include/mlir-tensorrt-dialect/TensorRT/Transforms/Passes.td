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
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES
#define MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RaiseActivationsPass
//===----------------------------------------------------------------------===//

def RaiseActivationsPass : Pass<"tensorrt-raise-activations"> {
  let summary = "raise sequence of operations to higher-level `tensorrt.activation`";

  let description = [{
    This pass matches and raises sequences of more primitive operations to
    single higher-level `tensorrt.activation` ops (e.g. GELU).
  }];

  let dependentDialects = [
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];

  let options = [
    Option<"targetTensorRTVersion", "target-tensorrt-version", "TensorRTVersion",
      "TensorRTVersion()",
      "the TensorRT version that is being targeted">
  ];
}

//===----------------------------------------------------------------------===//
// PopulatePluginShapeRegions
//===----------------------------------------------------------------------===//

def InferPluginShapesPass : Pass<"tensorrt-infer-plugin-shapes"> {

  let summary = "populates tensorrt.opaque_plugin shape regions by loading and "
                "querying the TensorRT plugin";

  let description = [{
    The purpose of this pass is to populate the scalar shape calculation region
    of `tensorrt.opaque_plugin` operation that result shapes with at least
    one unknown extent.

    While this region can be populated by hand, this pass will attempt to load
    and instantiate the plugin and query the plugin's methods to retrieve
    the expressions for the result shapes. We provide the plugin with a
    special implementation of the IDimExprBuilder interface which constructs
    the scalar `arith` dialect IR in the plugin region.

    Failure to load the plugin or to construct the scalar operations in
    the shape region is considered a hard failure since later passes may
    need the shape calculations for e.g. bounds analysis.
  }];
}

//===----------------------------------------------------------------------===//
// RaiseNormalizations
//===----------------------------------------------------------------------===//
def RaiseNormalizationsPass : Pass<"tensorrt-raise-normalizations">{
  let summary = "raise sequence of operations to higher-level `tensorrt.normalization`";

  let description = [{
    This pass matches and raises the sequence of more primitive operations that
    perform instance, batch or layer normalizations to single higher-level
    `tensorrt.normalization` ops.
  }];

  let dependentDialects =  [
      "::mlir::pdl::PDLDialect",
      "::mlir::pdl_interp::PDLInterpDialect",
      "::mlir::tensorrt::TensorRTDialect"];
}


//===----------------------------------------------------------------------===//
// ExpandOpsPass
//===----------------------------------------------------------------------===//
def ExpandOpsPass : Pass<"tensorrt-expand-ops"> {
  let summary = "Expand tensorrt extension ops into one or more lower-level operations";
  let dependentDialects = [];
  let options = [];
}

//===----------------------------------------------------------------------===//
// BroadcastElimination
//===----------------------------------------------------------------------===//
def BroadcastEliminationPass : Pass<"tensorrt-broadcast-elimination"> {
  let summary = "Try to eliminate `tensorrt.broadcast` operations";
  let description = [{
    The `tensorrt-broadcast-elimination` pass tries to eliminate
    `tensorrt.broadcast` operations by absorbing them into compatible
    tensorrt operations that support implicit broadcasting. Note that
    this pass will also try to "push down" the broadcasts below reshapes
    in order to move broadcasts closer to the consuming computational
    operation, making it more likely that the broadcast can be eliminated.
  }];
  let dependentDialects = [];
  let options = [];

  let statistics = [
    Statistic<"numBroadcastEliminated", "num-broadcast-eliminated",
              "The number of eliminated broadcast operations">
  ];
}

//===----------------------------------------------------------------------===//
// ApplyBugWorkarounds
//===----------------------------------------------------------------------===//
def ApplyWorkaroundsPass : Pass<"tensorrt-apply-wars"> {
  let summary = "Apply workarounds for known bugs and other issues for a specific TRT version";
  let description = [{
    Applies a set of patterns that rewrite certain IR patterns so as to
    1. Avoid known TensorRT bugs.
    2. Overcome TensorRT layer limitations.
    The patterns are specific to the given TensorRT version.
  }];

  let dependentDialects = [];
  let options = [
    Option<"tensorrtVersion", "tensorrt-version", "std::string", "\"8.5\"",
      "TensorRT version in the form MAJOR.MINOR">,
    Option<"tensorrtStronglyTyped", "tensorrt-strongly-typed", "bool", "false",
      "Whether TensorRT strongly typed mode is enabled">,
    Option<"forceDefaultSliceInBounds", "force-default-slice-in-bounds", "bool",
      "false",
      "Specifies that we should insert min/max operations to force the "
      "dynamic offset and size tensors of all 'default' slice operations "
      "to be in bounds. This should be used when shape tensor input bounds are "
      "not known in order to workaround TRT limitations.">
  ];
}

//===----------------------------------------------------------------------===//
// LegalizeInt8Pass
//===----------------------------------------------------------------------===//
def LegalizeInt8Pass : Pass<"tensorrt-legalize-int8", "func::FuncOp"> {
  let summary = "performs required transformations where int8 tensors are used";
  let description = [{
    This pass first determines the int8 "mode" that will be used during TensorRT
    translation:
      - "QDQ mode" is used if any Quantize or Dequantize operations
        are in the network.
      - "dynamic range mode" is used otherwise.

    Each mode has its own quirks and it's not really worth spending effort to
    understand what the differences are because neither one will really allow
    the user to create int8 programs that represent an 'explicit' management
    of precision (i.e. just like using int8 in numpy or JAX).

    This pass will attempt to determine which "mode" will be used during
    translation and insert some rewrite workarounds so that the translated tensorrt
    engine behaves *as close as possible* to the semantics of the IR, but
    since we don't control TensorRT, and sometimes TensorRT may behave eratically,
    so results may vary.
  }];
  let dependentDialects = ["::mlir::tensorrt::TensorRTDialect"];
}

//===----------------------------------------------------------------------===//
// TransposeEliminationPass
//===----------------------------------------------------------------------===//
def TransposeEliminationPass : Pass<"tensorrt-transpose-elimination"> {
  let summary = "try to eliminate tensorrt.transpose operations";

  let description = [{

    It is well-known that excessive number of transpose ops (either
    "tensorrt.transpose" or "tensorrt.shuffle" operations with identity reshape)
    can cause performance issues with TensorRT. This commonly occurs when the
    input source being converted represents convolutions in "NHWC" format vs.
    TensorRT's preferred "NCHW" format. In the conversion of these types of
    convolutions, a number of transpose operations must be inserted. These
    transpose operations can prevent fusions. For example, a transpose operation
    between a convolution and a pointwise addition can prevent convolution-bias
    fusion.

    This pass tries to eliminate transpose operations by applying the following
    patterns in a greedy manner:

    1) rotating `tensorrt.transpose` "forward" certain computational operations,
    especially `tensorrt.element_wise` ops. This means that the transpose will
    be applied to the result of the elementwise operation as well as the other
    branch of the operation. To avoid an infinite ping-pong application of this
    pattern certain heuristics are applied to determine whether or not this is
    beneficial. For example:

    ```
    func.func @transpose_pushdown_switch(%arg0: tensor<2x2xf32>, %arg1: tensor<1x2xf32>)
                  -> tensor<2x2xf32> {
      %1 = tensorrt.transpose {
        permutation = affine_map<(d0, d1)->(d1, d0)>
      } %arg0 : tensor<2x2xf32> to tensor<2x2xf32>
      %2 = tensorrt.element_wise <kSUM> (
        %1, %arg1: tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
      return %2 : tensor<2x2xf32>
    }
    ```

    becomes

    ```
    func.func @transpose_pushdown_switch(%arg0: tensor<2x2xf32>,
        %arg1: tensor<1x2xf32>) -> tensor<2x2xf32> {
      %0 = tensorrt.transpose {permutation = #map}
        %arg1 : tensor<1x2xf32> to tensor<2x1xf32>
      %1 = tensorrt.element_wise <kSUM>
        (%arg0, %0 : tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
      %2 = tensorrt.transpose {permutation = #map} %1 : tensor<2x2xf32> to tensor<2x2xf32>
      return %2 : tensor<2x2xf32>
    }
    ```

    In this case, moving the transpose to the other branch results in lower
    memory cost on the inputs, but higher total memory cost (because a transpose
    on the result is also added). However, we always prefer to push transpose
    operations as far forward as possible in this transformation.

    2) Const-folding transpose operations. Often, it is undesirable to let
    weights be transposed at runtime. Instead, weights should be pre-transformed
    to put them into a form that is suitable for TensorRT convolutions.
    Therefore, we apply global transpose-const folding. This can be quite
    expensive for large weights but is important to reduce runtime transpose
    costs.

  }];
}

//===----------------------------------------------------------------------===//
// ReshapeEliminationPass
//===----------------------------------------------------------------------===//
def ReshapeEliminationPass : Pass<"tensorrt-reshape-elimination"> {
  let summary = "try to eliminate tensorrt.reshape operations";

  let description = [{
    Reshape elimination pass captures pattern with un-necessary reshape and
    simplifies it by eliminating reshape operations.
  }];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES
