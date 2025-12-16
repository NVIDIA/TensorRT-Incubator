//===- KernelTransformOps.td ----------------------------------------------===//
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
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMOPS_KERNELTRANSFORMOPS_TD
#define MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMOPS_KERNELTRANSFORMOPS_TD

include "mlir-kernel/Kernel/IR/Attributes.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// KernelVectorizeChildrenAndApplyPatternsOp
//===----------------------------------------------------------------------===//

def KernelVectorizeChildrenAndApplyPatternsOp :
  Op<Transform_Dialect, "kernel.vectorize_children_and_apply_patterns",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This is a clone of the upstream linalg transform op
    "transform.structured.vectorize_children_and_apply_patterns", but it
    removes the "insert slice vectorization" which was added in upstream
    PR https://github.com/llvm/llvm-project/pull/111349.

    It is not allowed for us to "vectorize" the `tensor.insert_slice` operation
    in the manner introduced in that PR because we treat `tensor.insert_slice`
    as a concurrent operation in our compiler (e.g. more like
    `tensor.parallel_insert_slice`, which is currently only allowed within
    a "ParallelCombiningOp" region).

    To resolve these discrepancies we need to fix our usage of
    the `tensor.insert_slice` operation, which we create during outlining
    of `scf.forall`. Instead, we should introduce our own parallel combining
    operation in the Kernel dialect and use `tensor.parallel_insert_slice`.

  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   UnitAttr:$vectorize_padding,
                   UnitAttr:$vectorize_nd_extract,
                   UnitAttr:$flatten_1d_depthwise_conv,
                   UnitAttr:$disable_multi_reduction_to_contract_patterns,
                   UnitAttr:$disable_transfer_permutation_map_lowering_patterns);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
      "$target attr-dict `:`"
      "functional-type(operands, results)";

  let builders = [
    OpBuilder<(ins "Value":$target,
               CArg<"bool", "false">:$vectorizePadding,
               CArg<"bool", "false">:$vectorizeNDExtract,
               CArg<"bool", "false">:$flatten1DDepthwise)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// FuseGreedily
//===----------------------------------------------------------------------===//

def FuseGreedilyOp : Op<Transform_Dialect, "kernel.fuse_greedily_op",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    The `kernel.fuse_greedily_op` operation loops over all operands in the given
    `target` op and tries to replace any operands that are the result of a
    `tensor.extract_slice` with an equivalent inlined computation based on the
    slices' input operand. An example would be to replace a slice of a
    `linalg.matmul` with an instance of `linalg.matmul` that computes the slice.

    This procedure is called "fusion" because the target op is typically
    located within the body of a loop nest.

    Example:

    ```
    %0 = linalg.matmul ins(%arg0, %arg1 : ... ) ... -> tensor<?x?xf32>
    %1 = scf.forall (...) -> tensor<?x?xf32> {
      %2 = tensor.extract_slice %0[...][...][...] : tensor<?x?xf32> to tensor<128x64xf32>
      %3 = linalg.matmul ins(%2, %arg1 : tensor<128x64xf32>, tensor<64x64xf32f32>)
              ... -> tensor <128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into ...
      }
    }
    ```

    can be transformed to

    ```
    %1 = scf.forall (...) -> tensor<?x?xf32> {
      %arg0_slice = tensor.extract_slice %arg0 ....
      %arg1_slice = tensor.extract_slice %arg1 ....
      %2 = linalg.matmul ins(%arg0_slice, %arg1_slice : ... ) ... -> tensor<128x64xf32>
      %3 = linalg.matmul ins(%2, %arg1 : tensor<128x64xf32>, tensor<64x64xf32f32>)
              ... -> tensor <128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into ...
      }
    }
    ```

    The transformation is made generic by leveraging the `TilingInterface`. Any
    operation that implements the `TilingInterface` specifies how to replace a
    slice of the operation's result(s) with a new inlined computation.
    Therefore, any operation that implements `TilingInterface` can potentially
    be fused.

    This inlining/fusion process is applied greedily by repeating the procedure
    on all fused operations until no fusions can be made. In the above example,
    the procedure is applied at first to `%3`, which is the initial `target`.
    Then, it would be applied to `%2`. Since `%2`'s operands are slices of block
    arguments, no further fusion is possible and the procedure terminates.

    The initial `target` op is usually a `linalg.generic`, but it can be any
    operation that accepts tensor parameters.

    More specifically, an operand of the `target` is a fusion candidate if:

    1. It is produced by a `tensor.extract_slice` operation
    2. The input to the `tensor.extract_slice` is produced by a `TilingInterface` operation.

    #### Example usage:

    ```
    %1 = transform.kernel.fuse_producers_greedily %0
    ```

    #### Return Modes

    This operation consumes the `target` handle and returns a new handle
    representing the same operation as the `target`.

  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        Operation *target,
        transform::ApplyToEachResultList &results,
        transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// ForallToKernelOp
//===----------------------------------------------------------------------===//

def ForallToKernelOp : Op<Transform_Dialect,
      "kernel.forall_to_kernel",
      [TransformEachOpTrait,
       FunctionalStyleTransformOpTrait,
       MemoryEffectsOpInterface,
       TransformOpInterface]> {

  let description = [{
    Outlines the body of an `scf.forall` operation to a new `func.func` nested
    under a `gpu.module` operation. The `scf.forall` operation is replaced by
    a `kernel.call` operation that represents "launching" the kernel.

    The block arguments for the `scf.forall` body representing the indices of
    the processing elements are replaced by `gpu.block_id` operations inserted
    at the start of the body block. Up to three indices are replaced by
    `gpu.block_id` with the position arguments `x`, `y`, and `z` corresponding
    to the first, second, and third block arguments representing procesing
    element indices. Having more than three distribution indices is considered a
    definite error.

    All `shared_outs` parameters and well as used-values-defined-above become
    arguments to the created `func.func`. The only exception is scalar and
    vector constants, which are inlined into the body.

    The result is a `func.func` that represents a thead-block level SPMD
    program.

    If `reuse_existing_gpu_module` is true, then the kernel function will
    be inserted into the first existing `gpu.module` that is found. If
    no existing `gpu.module` is found or if `reuse_existing_gpu_module` is
    false, then a new `gpu.module` will be created.

    If `extra_module_attrs` is provided, then the attributes will be added
    to the created `gpu.module` only if `reuse_existing_gpu_module` is true.

    #### Example

    ```mlir
    func.func @main(%arg0: tensor<1024x1024xf32>,
                %arg1: tensor<1024x1024xf32>,
                %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %0 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg2) -> tensor<1024x1024xf32> {
        ... body ...
      }
      return %0 : tensor<1024x1024xf32>
    }
    ```

    is transformed to

    ```mlir
    func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>,
                    %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
      %c8 = arith.constant 8 : index
      %0 = kernel.call @kernels::@main_kernel[%c8, %c8] (%arg0, %arg1) outs(%arg2)
        : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
      return %0 : tensor<1024x1024xf32>
    }
    gpu.module @kernels {
      func.func @main_kernel(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>,
                             %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
        %0 = gpu.block_id  x
        %1 = gpu.block_id  y
        ... body ...
      }
    }
    ```

    #### Return modes

    If `target` is not a top-level `scf.forall` operation, then it is silently
    ignored.

    The operation consumes the `target` (`scf.forall`) handle and produces
    handles for the created `kernel.call`, `gpu.module`, and `func.func`
    operations.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$num_threads,
                       DefaultValuedAttr<BoolAttr, "true">:$reuse_existing_gpu_module,
                       OptionalAttr<DictionaryAttr>:$extra_module_attrs);

  let results = (outs TransformHandleTypeInterface:$kernel_call,
                      TransformHandleTypeInterface:$kernel_module,
                      TransformHandleTypeInterface:$kernel_func);

  let assemblyFormat = [{
    $target `threads` `(` $num_threads `)` attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$num_threads,
                   CArg<"bool", "true">:$reuse_existing_gpu_module,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extra_module_attrs), [{
        auto anyOpType = transform::AnyOpType::get($_builder.getContext());
        build($_builder, $_state,
          {anyOpType, anyOpType, anyOpType},
          target, num_threads, reuse_existing_gpu_module,
          !extra_module_attrs.empty() ? $_builder.getDictionaryAttr(extra_module_attrs)
            : DictionaryAttr{});
      }]>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// NestScalarLinalgInForallOp
//===----------------------------------------------------------------------===//

def NestScalarLinalgInForallOp : Op<Transform_Dialect,
      "kernel.nest_scalar_linalg_in_forall",
      [TransformEachOpTrait,
       FunctionalStyleTransformOpTrait,
       MemoryEffectsOpInterface,
       TransformOpInterface]> {

  let description = [{
    The `transform.kernel.nest_scalar_linalg_in_forall` operation nests the given
    `linalg.generic` target under a single-iteration `scf.forall` operation.
    The target is replaced by the results of the `scf.forall` operation.

    The purpose is to enable generating `scf.forall` that represent single-CTA,
    single-warp kernels for a given `linalg.generic` operation that may have no
    loops (and thus cannot use the `transform.structured.tile_to_forall` op
    to accomplish the same thing).


    #### Return Modes

    The operation returns new handles to the `scf.forall` operation and the
    nested clone of the Linalg target op. The target handle is consumed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$forall_op, TransformHandleTypeInterface:$linalg_op);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        ::mlir::DestinationStyleOpInterface target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

  let builders = [
    OpBuilder<(ins "Value":$target), [{
      auto anyOpType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state,
        {anyOpType, anyOpType},
        target);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ForallToSubgroupsOp
//===----------------------------------------------------------------------===//

def ForallToSubgroupsOp : Op<Transform_Dialect,
      "kernel.forall_to_subgroups",
      [TransformEachOpTrait,
       FunctionalStyleTransformOpTrait,
       MemoryEffectsOpInterface,
       TransformOpInterface]> {

  let description = [{

    Inlines the body of an `scf.forall` operation immediately above the op
    itself and then deletes the `scf.forall` operation. The block arguments
    representing distribution indices are replaced by a calculation on
    `gpu.thread_idx x` representing the calculation of the subgroup id and then
    the subgroup id is delinearized to the shape of the total indices in each
    index dimension of the `scf.forall` operation.

    Let x = gpu.thread_id x, then the subgroup id will be:
    (x mod (prod(loop_upper_bound) * subgroup_size)) floor_div subgroup_size.
    This subgroup id will be delinearized to the dimension of the
    loop and to replace the loop iterator variable.

    Usage:
      `transform.kernel.forall_to_subgroups %forall subgroup_size(s)`

    Thus, this transformation assumes that the threadblock is launched as a 1D
    linear array of threads. For example, if the number of threads is `128`,
    then the threadblock shape at launch time should be given as `(NumThreads,
    1, 1)`, otherwise the result of this transformation is undefined.

    #### Example

    Given `%1 = transform.kernel.forall_to_subgroups %arg0 subgroup_size(32)`
    ```mlir
    func.func @kernel(%arg0: tensor<128x32xf32>,
                  %arg1: tensor<32x128xf32>,
                  %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
      %c2 = arith.constant 2: index
      %0 = scf.forall (%i, %j) in (%c2, %c2) shared_outs(%out = %arg2) -> tensor<128x128xf32> {
        ... body ...
      }
      return %0 : tensor<128x128xf32>
    }
    ```

    is transformed to

    ```
    func.func @kernel(%arg0: tensor<128x32xf32>,
                      %arg1: tensor<32x128xf32>,
                      %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
      %c2 = arith.constant 2 : index
      %0 = gpu.thread_id  x
      %1 = affine.apply affine_map<()[s0] -> ((s0 mod 128) floordiv 32)>()[%0]
      %2:2 = affine.delinearize_index %1 into (%c2, %c2) : index, index
      ... body ...
    }
    ```
    where the constant 128 in the affine_map is by subgroup_size (32)*%c2*%c2.

    #### Return modes

    The operatin consumes the `target` (`scf.forall`) handle and produces
    handles for the created operation representing the `group_id` (before
    delinearization to the different dimensions).
  }];

  let arguments = (ins TransformHandleTypeInterface:$target, I64Attr:$subgroup_size);
  let results = (outs TransformHandleTypeInterface:$group_id);
  let assemblyFormat = "$target `subgroup_size` `(` $subgroup_size `)` attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// VerifyPostTilingOp
//===----------------------------------------------------------------------===//

def VerifyPostTilingOp : Op<Transform_Dialect,
      "kernel.verify_post_tiling",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Performs checks that should occur after tiling and fusion have occurred
    but prior to distributing/outlining the `scf.forall` operations.

    The `forall` parameter should be the outer-most `scf.forall` (which
    is mapped to blocks in the grid).

    #### Return modes

    Reads the target handle and produces no results. If errors are discovered,
    an error is emitted and a definite failure is returned.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs );
  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// LowerToLoopsOp
//===----------------------------------------------------------------------===//

def LowerToLoopsOp : Op<Transform_Dialect,
        "kernel.lower_to_loops",
        [TransformEachOpTrait,
         FunctionalStyleTransformOpTrait,
         MemoryEffectsOpInterface,
         TransformOpInterface]> {
  let description = [{
    Lower the given operation to SCF loops.

    #### Return modes

    This operation consumes the target handle and produces a new handle
    representing the outer-most loop of the `scf.for` loop nest used
    to replace the target.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    DiagnosedSilenceableFailure applyToOne(
        transform::TransformRewriter &rewriter,
        ToLoopsOpInterface target,
        transform::ApplyToEachResultList &results,
        transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOps
//===----------------------------------------------------------------------===//

def ApplyRewriteVectorTransferReadToConstantPatternOp : Op<Transform_Dialect,
    "apply_patterns.vector.rewrite_vector_transfer_read_to_constant",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Includes a vector simplification pattern in the set. The pattern
    detects when a `vector.transfer_read` is reading from the result of
    a `vector.transfer_write` which has an overwrite/fill-like effect.

    Such `vector.transfer_reads` are then replaced with `arith.constant`
    with a splat constant value.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyInterchangeForAndForallPatternsOp : Op<Transform_Dialect,
    "apply_patterns.kernel.interchange_for_and_forall",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Includes patterns to interchange `scf.for` and `scf.forall` so that
    the `scf.forall` is outer-most. This is possible if the `scf.forall`
    effectively implements a subset insertion/extraction on the `scf.for`
    iteration arguments. In that case, we can hoist the `scf.forall` up
    without changing the semantics of the program.

    Finds instances of the following pattern:

    ```
    %result = for ... iter_args(%iter = $init) {
      %update = forall ... outs(%out = %iter) {
        %0 = extract_slice %out[%o][%s][1]
        ...
        %tile = ...
        parallel_insert_slice %tile into %out[%o][%s][1]
      }
      yield %update
    }
    ```

    And produces:

    ```
    %result = forall ... outs(%out = %init) {
      %0 = extract_slice %out[%o][%s][1]
      %1 = for ... iter_args(%iter = $0) {
        ...
        %tile = ...
        yield %tile
      }
      parallel_insert_slice %1 into %out[%o][%s][1]
    }
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyAffineBoundsOptimizationPatternsOp : Op<Transform_Dialect,
    "apply_patterns.kernel.affine_bounds_optimization",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply simplification patterns for `arith` and `affine.apply|min|max`
    based on constraint information derived from
    `ValueBoundsConstraintSet`/`ValueBoundsOpInterface`.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMOPS_KERNELTRANSFORMOPS_TD
