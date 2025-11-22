//===- Passes.td -------------------------------------------*- Tablegen -*-===//
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
// TransposeReshapeEliminationPass
//===----------------------------------------------------------------------===//
def TransposeReshapeEliminationPass : Pass<"tensorrt-transpose-reshape-elimination"> {
  let summary = "try to eliminate tensorrt.transpose, tensorrt.reshape, and tensorrt.shuffle operations";

  let description = [{
    It is well-known that excessive number of transpose or reshapes ops (either
    "tensorrt.transpose", "tensorrt.reshape" or "tensorrt.shuffle")
    can cause performance issues with TensorRT. For example, this commonly occurs
    when the input source being converted represents convolutions in "NHWC" format
    vs. TensorRT's preferred "NCHW" format. In the conversion of these types of
    convolutions, a number of transpose operations must be inserted. These
    transpose operations can prevent fusions. For example, a transpose operation
    between a convolution and a pointwise addition can prevent convolution-bias
    fusion.  Fusions can also be blocked if a reshape is placed between a
    matrix multiplication and an activation.

    The pass tries to eliminate transposes and reshapes by pushing the transposes
    and reshapes around to combine them into identity transposes and reshapes which
    can be eliminated from the program.  To accomplish this, this pass uses several
    rewrite rules that push transposes and reshapes from the output of an Operation
    to the operation's input, and vice-versa.  The rewrites currently included in this
    pass handle common cases, though currently does not handle every possible scenario---one
    may wish to extend this pass in the future as needed.

    The process is as follows:
    1) Normalize transpose, reshape, shuffle and matrix multiply into a common set of ops.
       - shuffle(x) -> reshape(transpose(reshape(x)))
       - matrix_multiply(x, y) -> einsum("ij,jk->ik", x, y)
       - expand_rank(x) -> reshape(x)
       - collapse_rank(x) -> reshape(x)
    2) Push down reshape and transpose, eliminating when possible. E.g. op(transpose(x)) -> transpose(op(x))
       - einsum(transpose(x), ...) -> einsum(x, ...)
       - einsum(...) -> transpose(einsum(...))      Pull transposes out of einsum (to try to match matrix multiply pattern)
       - einsum(reshape(x), y, ...) -> transpose(reshape(einsum(x, reshape(transpose(y)), ...))
       - unary(transpose(x)) -> transpose(unary(x))
       - activation(transpose(x)) -> transpose(activation(x))
       - identity_op(transpose(x)) -> transpose(identity_op(x))
       - activation(reshape(x)) -> reshape(activation(x))
       - unary(reshape(x)) -> reshape(unary(x))
       - identity_op(reshape(x)) -> reshape(identity_op(x))
       - reshape(transpose(x)) -> transpose(reshape(x)) if possible put reshape before transpose
       - dequantize(quantize(transpose(x))) -> transpose(dequantize(quantize((x))) if the scale is 0-dim
       - dequantize(quantize(reshape(x))) -> reshape(dequantize(quantize(x))) if the scale is 0-dim
       - reshape(reshape(x)) -> reshape(x)
       - transpose(transpose(x)) -> transpose(x)
       - reshape(x) -> x if reshape is identity
       - transpose(x) -> x if transpose is identity
       - elementwise(reshape(a), b) -> reshape(elementwise(a, reshape(b))) conditioned on heuristic
       - elementwise(transpose(a), b) -> transpose(elementwise(a, transpose(b)))
    3) Push up reshape and transpose, eliminating when possible.  E.g. transpose(op(x)) -> op(transpose(x))
       - transpose(einsum(...)) -> einsum(...)
       - einsum(...) -> einsum(transpose(x), ...)   Pull transposes out of einsum (to try to match matrix multiply pattern)
       - reshape(einsum(...)) -> einsum(reshape(transpose(x)), ...)
       - transpose(activation(x)) -> activation(transpose(x))
       - transpose(unary(x)) -> unary(transpose(x))
       - transpose(identity_op(x)) -> identity_op(transpose(x))
       - reshape(activation(x)) -> activation(reshape(x))
       - reshape(unary(x)) -> unary(reshape(x))
       - reshape(identity_op(x)) -> identity_op(reshape(x))
       - reshape(reshape(x)) -> reshape(x)
       - transpose(transpose(x)) -> transpose(x)
       - reshape(x) -> x if reshape is identity
       - transpose(x) -> x if transpose is identity
       - transpose(reshape(x)) -> reshape(transpose(x)) if possible put transpose before reshape
       - transpose(dequantize(quantize(x))) -> dequantize(quantize(transpose(x))) if the scale is 0-dim
       - reshape(dequantize(quantize(x))) -> dequantize(quantize(reshape(x))) if the scale is 0-dim
       - reshape(elementwise(a, b)) -> elementwise(reshape(a), reshape(b))
       - transpose(elementwise(a, b)) -> elementwise(transpose(a), transpose(b))
    4) Final clean up.  Fuse leftover transposes and reshapes with other ops.
       - einsum("ij,jk->ik", x, y) -> matrix_multiply(x, y) if einsum matches a matrix multiply pattern
       - matrix_multiply(transpose(x), y) -> matrix_multiply(x, y) merge transpose if possible
       - transpose(einsum(...)) -> einsum(...)
       - einsum(transpose(x), ...) -> einsum(...)
       - einsum(collapse_rank(x), ...) -> einsum(...)
       - expand_rank(einsum(...)) -> einsum(...)

    To avoid an infinite ping-pong application of these patterns, heuristics are
    applied to determine when a pattern is beneficial.
  }];
}


#endif // MLIR_TENSORRT_DIALECT_TENSORRT_TRANSFORMS_PASSES
