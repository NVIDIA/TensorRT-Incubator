//===- Attributes.td ------------------------------------------------------===//
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
#ifndef MLIR_KERNEL_KERNEL_IR_KERNELATTRIBUTES
#define MLIR_KERNEL_KERNEL_IR_KERNELATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir-kernel/Kernel/IR/Dialect.td"
include "mlir-kernel/Kernel/IR/Enums.td"
include "mlir-kernel/Kernel/IR/Interfaces.td"

class Kernel_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Kernel_Dialect, name, traits> {
        let mnemonic = attrMnemonic;
}

def Kernel_DimsParam : ArrayRefParameter<"int64_t", "list of i64 dimension magnitudes"> {
  let cppStorageType = "llvm::SmallVector<int64_t>";
  // The printer and parse will parse this as `[` dims... `]`.
  let parser = "::parseIntArray($_parser)";
  let printer = "::printIntArray($_printer, $_self)";
}

def Kernel_TensorCoreInfoAttr : Kernel_Attr<"TensorCoreInfo", "tensorcore"> {
  let description = [{
    The `#kernel.tensorcore` carries information about the type of NVIDIA
    TensorCore (Warp-level Matrix Multiply-Accumulate) PTX instruction.

    For complete documentation, see this section in the NVIDIA PTX docs:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
  }];
  let parameters = (ins
    "Type":$a_type,
    "Type":$b_type,
    "Type":$c_type,
    "Type":$d_type,
    "int64_t":$m,
    "int64_t":$n,
    "int64_t":$k,
    "bool":$sparse
  );
  let genVerifyDecl = 1;
  let genAccessors = 1;
  let assemblyFormat = [{
    `<`
    custom<TensorCoreInfo>($a_type, $b_type,
                           $c_type, $d_type,
                           $m, $n, $k, $sparse)
    `>`
  }];
  let extraClassDeclaration = [{
    /// Return M, N, K parameters as an array.
    std::array<int64_t, 3> getMNK() {
      return {getM(), getN(), getK()};
    }
  }];
}

def Kernel_IndexArgAttr : Kernel_Attr<"IndexArg", "index_arg"> {
  let summary = "describes an index value defined by a function argument";

  let description = [{
     The `kernel.index_arg` encodes a value of index type, defined by a
     function argument. Specifically, the value can be (1) from a
     function argument of type `index`, or (2) loaded from a function
     argument of type `memref<...xindex>` with constant coordinates.
     We store the argument number in the `arg` parameter. For case
     (1), there will be no `coordinate` parameter; for case (2), the
     `coordinate` parameter stores the loaded coordinates.

     E.g. case (1):
     ```
     func.func (%arg0: memref<128xf16>, %idx: index, ...) {
        memref.store %arg0[%idx]
     }
     ```
     The kernel.index_arg for the stored index value will be <arg = 1>.

     E.g. case (2):
     ```
     func.func (%arg0: memref<128xf16>, %indices: memref<2xindex>, ...) {
        %c1 = arith.constant 1
        %idx = memref.load %indices[%c1]
        memref.store %arg0[%idx]
     }
     ```
     The kernel.index_arg for the stored index value will be
     <arg = 1, coordinate = [1]>.
  }];

  let parameters = (ins
    "int64_t":$arg,
    OptionalArrayRefParameter<"int64_t">:$coordinate
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` `arg`  `=` $arg
        ( `,` `coordinate`  `=` `[` $coordinate^ `]` )?
    `>`
  }];
}

def Kernel_BoundAttr : Kernel_Attr<"Bound", "bound"> {
  let summary = "describes a lower or upper bound as an affine map";

  let description = [{
     The `kernel.bound` attribute describes a lower or upper bound
     using an affine map. If the bound depends on other parameters,
     those parameters are encoded using `IndexArgAttrs`.
  }];

  let parameters = (ins
    ArrayRefParameter<"IndexArgAttr">:$args,
    "AffineMap":$map
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` `args` `=` `[` $args `]` `,`
        `map`  `=` $map
    `>`
  }];
}

def Kernel_PageBoundsAttr : Kernel_Attr<"PageBounds", "page_bounds"> {
  let summary = "describes the bounds for the function arg to be paged";

  let description = [{
    The `kernel.page_bounds` attribute stores the lower and upper
    bounds of the indexing of the function arguments that are consumed
    by `memref.load` or `memref.store`. The lower and upper bounds are
    described using affine maps wrapped as `BoundAttr`s. Because there
    can be multiple `memref.load` and `memref.store`s on one argument,
    and thus multiple lower and upper bounds, we store all these bounds
    as lists with the parameters `lower` and `upper`.
  }];

  let parameters = (ins
    ArrayRefParameter<"BoundAttr">:$lower,
    ArrayRefParameter<"BoundAttr">:$upper
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` `lower`  `=` `[` $lower `]` `,`
        `upper`  `=` `[` $upper `]`
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// DefaultGPUModuleKindAttr
//===----------------------------------------------------------------------===//

def Kernel_DefaultGPUModuleKindAttr : Kernel_Attr<
      "DefaultGPUModuleKind", "gpu_module_kind.default"> {
  let summary = "default lowering for a GPU module";

  let description = [{
    The `#kernel.gpu_module_kind.default` attribute describes how to lower
    a `gpu.module` its interface methods describe separate pipelines
    for pre/post bufferization optimizations and final lowering to
    NVVM IR.
  }];
}

//===----------------------------------------------------------------------===//
// Attributes representing Transform IR generators (schedule generators)
//===----------------------------------------------------------------------===//

def Kernel_MatMulScheduleParametersAttr: Kernel_Attr<"MatMulScheduleParameters",
    "matmul_parameters",
    [DeclareAttrInterfaceMethods<CodegenScheduleAttrInterface>]> {
  let summary = "An attribute representing the MatMul parameters: CTA tile "
    "sizes, the number of warps, and the number of stages";
  let description = [{
    The `#kernel.matmul_parameters` attribute describes
    the CTA tile sizes and the number of warps, which are used to
    generate tile sizes in the transform schedule. The CTA tile sizes
    should be given as a triple of the form [m_size, n_size, k_size]; the
    warp size should be one of 1, 2, 4, or 8. }];

  let parameters = (ins
    "::mlir::gpu::TargetAttrInterface":$gpu_target,
    ArrayRefParameter<"int64_t">:$cta_tile_sizes,
    "int64_t":$num_of_warps,
    OptionalArrayRefParameter<"int64_t">:$warp_tile_sizes,
    OptionalParameter<"std::optional<int64_t>">:$num_of_stages,
    OptionalParameter<"TensorCoreInfoAttr">:$tensor_core
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` `gpu_target` `=` $gpu_target `,`
        `cta_tile_sizes`  `=` `[` $cta_tile_sizes `]` `,`
        `num_of_warps` `=` $num_of_warps
        (`,` `warp_tile_sizes` `=` `[` $warp_tile_sizes^ `]` )?
        (`,` `num_of_stages` `=` $num_of_stages^ )?
        (`,` `tensor_core` `=` $tensor_core^ )?
      `>`
  }];

  let genVerifyDecl = 1;
}

def Kernel_ElementwiseScheduleParametersAttr: Kernel_Attr<"ElementwiseScheduleParameters",
    "elementwise_parameters",
    [DeclareAttrInterfaceMethods<CodegenScheduleAttrInterface>]> {
  let summary = "An attribute representing the Elementwise Ops parameters: "
    "the number of CTAs, the number of warps (subgroups) per CTA, and warp "
    "(subgroup) size.";
  let description = [{
    The `#kernel.matmul_parameters` attribute describes
    the CTA tile sizes, the number of warps (subgroups), and warp (subgroup)
    size, which are used to generate tile sizes in the transform schedule.
    Each parameters should be a vector representing the numbers along the
    corresponding input dimensions. The number of warps should be power of 2
    and <=4; and the warp size should be power of 2 and <= 32.
     }];

  let parameters = (ins
    "::mlir::gpu::TargetAttrInterface":$gpu_target,
    ArrayRefParameter<"int64_t">:$grid_shape,
    ArrayRefParameter<"int64_t">:$cta_shape,
    ArrayRefParameter<"int64_t">:$vector_shape
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` `gpu_target` `=` $gpu_target `,`
        `grid_shape`  `=` `[` $grid_shape `]` `,`
        `cta_shape`  `=` `[` $cta_shape `]` `,`
        `vector_shape`  `=` `[` $vector_shape `]`
    `>`
  }];
  let genVerifyDecl = 1;
}

def Kernel_FallbackScheduleParametersAttr: Kernel_Attr<"FallbackScheduleParameters",
    "fallback_parameters",
    [DeclareAttrInterfaceMethods<CodegenScheduleAttrInterface>]> {
  let summary = "An attribute representing the transform schedule parameters "
    "for all the other ops, however it should not be used for codegen "
    "workflow.";

  let description = [{
    The `#kernel.fallback_parameters` attribute contains parameters: the
    target device (target_device), the number of CTAs (grid_shape), the
    number of threads per CTA (cta_shape).

    Note: it is not designed to be used in the end-to-end codegen workflow. It
    is only meant to be used for tiling to distribute work to CTAs, currently
    for simulation and cost analysis purposes.
  }];

  let parameters = (ins
    "::mlir::gpu::TargetAttrInterface":$gpu_target,
    OptionalArrayRefParameter<"int64_t">:$cta_workload_shape,
    OptionalArrayRefParameter<"int64_t">:$cta_blocking_shape,
    OptionalArrayRefParameter<"int64_t">:$thread_tile_shape,
    OptionalArrayRefParameter<"int64_t">:$grid_shape,
    OptionalArrayRefParameter<"int64_t">:$cta_shape,
    DefaultValuedParameter<"bool", "true">:$disable_vectorization
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<`
      $gpu_target `,`
      `[` (`]`) : ( $cta_workload_shape^ `]`)? `,`
      `[` (`]`) : ( $cta_blocking_shape^ `]`)? `,`
      `[` (`]`) : ( $thread_tile_shape^ `]`)? `,`
      `[` (`]`) : ( $grid_shape^ `]`)? `,`
      `[` (`]`) : ( $cta_shape^ `]`)?
       (`,` `disable_vectorization` `=` $disable_vectorization^ )?
    `>`
  }];
}

def Kernel_ScatterScheduleParametersAttr: Kernel_Attr<
        "ScatterScheduleParameters",
        "scatter_parameters",
        [DeclareAttrInterfaceMethods<CodegenScheduleAttrInterface>]> {

  let summary = "An attribute representing the transform schedule parameters "
                "for scatter operations.";

  let description = [{
    The `#kernel.scatter_parameters` attribute contains
    parameters for the scatter transform IR generator.
  }];

  let parameters = (ins
    "::mlir::gpu::TargetAttrInterface":$gpu_target,
    Kernel_DimsParam:$cta_workload_shape,
    Kernel_DimsParam:$cta_blocking_shape,
    Kernel_DimsParam:$thread_tile_shape,
    Kernel_DimsParam:$grid_shape,
    Kernel_DimsParam:$cta_shape
  );

  let genAccessors = 1;

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

#endif // MLIR_KERNEL_KERNEL_IR_KERNELATTRIBUTES
