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
#ifndef MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// MaterializeShapeCalculationsPass
//===----------------------------------------------------------------------===//

def MaterializeShapeCalculationsPass : Pass<"plan-materialize-shape-calculations"> {

  let description = [{

    This pass materializes scalar arithmetic for calculation of the shapes
    of every dynamic tensor in the program.

    The materialized arithmetic IR consists of `arith` operations and
    `tensor.extract|dim` operations (e.g. when a shape is dependent on a
    calculation) and the method we use for materialization, described below,
    makes it possible to segment and outline the scalar arithmetic calculations
    into separate functions that can be used to e.g. expose shape calculations
    in the final executable.

    The materialization of the shape calculations occurs in two phases:


    1. Insertion of `plan.with_shape` operations. For every TensorType SSA
       value that has unknown dimension(s), we insert a `plan.with_shape`
       and `tensor.dim` operations at its definition. This is an identity
       transformation except that it materializes the scalar-result `tensor.dim`
       ops and associates them with the corresponding tensor SSA value.

    2. We run patterns that propagate the `tensor.dim` values upwards and
       canonicalize the resulting scalar arithmetic IR for the calculation of
       shapes.


    As an example, consider the following program:

    ```mlir

    func.func @test_dynamic_broadcast_max(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %0 = stablehlo.constant dense<0.0> : tensor<1x1xf32>

      %4 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
      %6 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
      %8 = "stablehlo.concatenate"(%5, %7) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

      %9 = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
      %11 = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
      %13 = "stablehlo.concatenate"(%10, %12) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

      %shape = stablehlo.maximum %8, %13 : tensor<2xi32>

      %result = "stablehlo.dynamic_broadcast_in_dim"(%0, %shape)
        {broadcast_dimensions = array<i64: 0, 1> : tensor<2xi64>}
          : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>

      return %result : tensor<?x?xf32>
    }

    ```

    After insertion of `plan.with_shape` operations (Step 1) we have:


    ```mlir

    func.func @test_dynamic_broadcast_max(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
      %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
      %3 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
      %5 = "stablehlo.concatenate"(%2, %4) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %6 = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
      %8 = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
      %10 = "stablehlo.concatenate"(%7, %9) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %11 = stablehlo.maximum %5, %10 : tensor<2xi32>
      %12 = "stablehlo.dynamic_broadcast_in_dim"(%0, %11) {broadcast_dimensions = array<i64: 0, 1> : tensor<2xi64>} : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>

      // --- begin scalar shape calculation ---
      %dim = tensor.dim %12, %c0 : tensor<?x?xf32>
      %dim_0 = tensor.dim %12, %c1 : tensor<?x?xf32>
      // --- end scalar shape calculation ---

      %15 = plan.with_shape %12(%dim, %dim_0) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
      return %15 : tensor<?x?xf32>
    }

    ```

    And finally after running rewrites we have (Step 2):

    ```mlir

    func.func @test_dynamic_broadcast_max(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
      %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
      %3 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
      %5 = "stablehlo.concatenate"(%2, %4) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %6 = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
      %8 = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
      %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
      %10 = "stablehlo.concatenate"(%7, %9) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %11 = stablehlo.maximum %5, %10 : tensor<2xi32>
      %12 = "stablehlo.dynamic_broadcast_in_dim"(%0, %11) {broadcast_dimensions = array<i64: 0, 1> : tensor<2xi64>} : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>

      // --- begin scalar shape calculation ---
      %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
      %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
      %13 = arith.maxsi %dim, %dim_0 : index
      %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
      %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
      %14 = arith.maxsi %dim_1, %dim_2 : index
      // --- end scalar shape calculation ---

      %15 = plan.with_shape %12(%13, %14) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
      return %15 : tensor<?x?xf32>
    }

    ```

    For illustration purposes, the IR of interest is highlighted between the
    `begin/end scalar shape calculation` comments. We can see that this segment
    of the IR is completely independent of the other StableHlo operations except for
    being tied to the function arguments and to the rest of the IR using the
    `plan.with_shape` operation. This helps facilitate outlining or use of
    additional analyses such as `IntegerRangeAnalysis`.

  }];


  let dependentDialects = [
    // The StableHLO shape reification functions may produce Shape dialect ops.
    "::mlir::shape::ShapeDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanRefineTypesPass
//===----------------------------------------------------------------------===//

def PlanRefineTypesPass : Pass<"plan-refine-types", "::mlir::ModuleOp"> {
  let description = [{
    This pass should run after the `plan-materialize-shape-computations` pass
    creates the `plan.with_values` and `plan.with_shapes` operations and before
    clustering of the IR into different segments. It attempts to refine the
    types of certain operations (e.g. StableHLO dynamic op variants) by using
    the side information that can be gleaned from the
    `plan.(with_values|with_shape)` operations. Often times this can result in
    more refined tensor types than could be achieved in the input IR
    preprocessing stage.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanCreateShapeFuncsPass
//===----------------------------------------------------------------------===//

def PlanCreateShapeFuncsPass : Pass<"plan-create-shape-funcs",
      "::mlir::ModuleOp"> {

  let description = [{
    This pass clusters groups of scalar operations that compute shapes. It
    then generates new public functions that compute the shapes and inserts
    metadata that link the original function to the shape calculation functions.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanPopulateFunctionBoundsAttributesPass
//===----------------------------------------------------------------------===//

def PlanPopulateFunctionBoundsAttributesPass
      : Pass<"plan-populate-func-bounds-attrs", "func::FuncOp"> {

  let description = [{
    This pass populates function result attributes containing shape/value bounds
    information for function results.
  }];

  let options = [];

  let dependentDialects = [
    "::mlir::tensorrt::TensorRTDialect",
  ];
}

//===----------------------------------------------------------------------===//
// StablehloClusteringPass
//===----------------------------------------------------------------------===//

def StablehloClusteringPass : Pass<"stablehlo-clustering", "::mlir::ModuleOp"> {
  let summary = "clusters and outlines `stablehlo` operations into a `func.func`";

  let description = [{
    This pass clusters groups of StableHLO operations into encapsulating region
    operations that are not isolated-from-above. The purpose of this pass
    is to achieve a course segmentation that specifies how clusters of
    operations will be compiled.
  }];

  let options = [
    Option<"entrypoint", "entrypoint", "std::string", "\"\"",
      "the name of the entrypoint function; if empty then the clustering runs"
      " on all functions">,
    Option<"disallowHostTensorsInTensorRTClusters",
      "disallow-host-tensors-in-tensorrt-clusters", "bool", "false",
      "don't cluster host tensors in TensorRT clusters">,
    Option<"disableCreateShapeFuncPass", "disable-create-shape-func-pass", "bool", "false",
      "don't apply create shape to func pass in TensorRT clusters">,
    Option<"trtMajorVersion", "trt-major-version", "int64_t", "NV_TENSORRT_MAJOR",
    "terget TensorRT version for clustering">
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tensorrt::TensorRTDialect",
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// CreateClosedRegionsPass
//===----------------------------------------------------------------------===//

def CreateClosedRegionsPass : Pass<"plan-create-closed-regions", "::mlir::ModuleOp"> {

  let description = [{
    This pass performs an intermediate lowering step `dps.inline_group`
    operations prior to the `plan-outline-clusters` pass.

    Certain cluster regions such as those targeting TensorRT are required to
    be replaced by a destination-passing-style (DPS) call-like operation
    in the outlining step. However, the clustered IR inside the region
    is StableHlo IR, whose operations do not follow
    destination-passing-style. When the results of the region op are
    dynamically shaped, the logic for materializing the output buffers
    above the region can become complicated.

    This pass replaces such cluster regions with `plan.inline_closed_group`
    region operations and materializes the destination output tensors above
    the region. This step is isolated as a dedicated pass in order to
    encapsulate the strategies required for materializing destination operands
    in the presence of dynamic shapes.

    There are three primary strategies required in order to handle all dynamic
    shape situations. The first two are applicable when it is possible to
    materialize the calculation of the output shapes as a set of scalar
    operations above the region (in other words, the shape of the outputs are
    not "data dependent dynamic shapes" / dependent on the result of an SSA
    value that is produced within the group region).

    Strategies:

    1. If an upper bound for the extents of all dynamic dimensions can be
       calculated for a result, then we create a empty tensor for the largest
       possible linear footprint required for that result. The linear block of
       memory is then sliced and reshaped to the exact desired output size
       (before the closed region) and is passed to the region's DPS argument.
       The results of the region are then held in the lower portion of the
       linear memory block corresponding to a packed tensor of the correct type.

    2. If an upper bound for the extents of all dynamic dimensions of a result
       can not be calculated, but the output shape calculation can still be
       materialized above, then we create a destination tensor that is
       dynamically sized to the exact required output shape. This is what occurs
       if our analysis machinery fails or shape bounds are not provided
       (assuming the backends do not require such bounds).

       Note: currently for backends such as TensorRT, we return an error if this
       occurs.

    3. [Unimplemented] In the case where the backend supports outputs whose
       shape is dependent on SSA values internal to the cluster/region, we say
       that the region op outputs have "data dependent dynamic shapes". In this
       case, we cannot materialize a `dps.inline_closed_group` operation and
       instead the region must be lowered to a call-like operation that for the
       relevant backend which takes on an allocation-like semantic when
       bufferized. This is currently not supported for any MLIR-TRT backend,
       although we plan to support this route with TensorRT in the future.

  }];

  let options = [
    Option<"testPreWalkOrder", "test-pre-walk-order", "bool", "false",
      "(used only in testing) specifies to outline regions by walking in "
      " pre-order; used for verifying results are not sensitive "
      "to traversal order">
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tensorrt::TensorRTDialect",
  ];
}

//===----------------------------------------------------------------------===//
// OutlineClustersPass
//===----------------------------------------------------------------------===//

def OutlineClustersPass : Pass<"plan-outline-clusters", "::mlir::ModuleOp"> {

  let description = [{
    This pass takes the clusters formed from the stablehlo-clustering pass and
    outlines them to function-like operation while replacing the cluster region
    results with the results of call-like operations.

    The specific types of "function-like" and "call-like" operations that
    are created depend on the specific cluster kind associated with each
    cluster region operation.

    Certain clusters such as those targeting TensorRT have additional special
    logic to capture the required metadata for each operand (e.g. shape
    profile).
  }];

  let options = [];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tensorrt::TensorRTDialect",
  ];
}

//===----------------------------------------------------------------------===//
// EliminateShapeOpsPass
//===----------------------------------------------------------------------===/

def EliminateShapeOpsPass : Pass<"plan-eliminate-shape-ops", "::mlir::ModuleOp"> {
  let description = [{

    The `plan-eliminate-shape-ops` pass replaces all `plan.with_shape`
    operations with their tensor operand. This pass is purely for cleanup after
    cluster outlining has occurred. In addition, the pass will cleanup unused
    arguments in functions after the elimination of `plan.with_shape`.

  }];
}

//===----------------------------------------------------------------------===//
// PostClusteringValidationPass
//===----------------------------------------------------------------------===//

def PostClusteringValidationPass : Pass<"post-clustering-validation", "func::FuncOp"> {
  let summary = "validates public `func.func` ops after cluster outlining";

  let description = [{
    This pass validates public `func.func` ops after cluster outlining.

    Validation logic checks the following:
    - Each public `func.func` has valid ops which can be bufferized
    - Each op in a public `func.func` has a valid datatype
  }];
}

//===----------------------------------------------------------------------===//
// PlanAllocTensorsPass
//===----------------------------------------------------------------------===//

def PlanAllocTensorsPass : Pass<"plan-alloc-tensors",
                                    "::mlir::ModuleOp"> {
  let summary = "creates `bufferization.alloc_tensor` operations";

  let description = [{
    This pass prepares the IR for bufferization using the `plan-bufferize`
    pass. Specifically, it eliminates `tensor.empty`, `tensor.from_elements`
    and other ops that represent the creation of values with tensor types.

    It does this by materializing `bufferization.alloc_tensor` operations.
    Certain decisions are also made regarding the placement of these
    allocations in terms of host vs. device. For example, scalar values in
    Executor IR functions are on the host, so `tensor.from_elements` must
    bufferize to a host allocation that is copied to a device tensor.

    Other `tensor.empty` operations by default bufferize to device allocations.
  }];

  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanBufferize
//===----------------------------------------------------------------------===//

def PlanBufferizePass : Pass<"plan-bufferize", "::mlir::ModuleOp"> {
  let summary = "Perform bufferization (one-shot-bufferize)";

  let description = [{
    This pass invokes one-shot-bufferization on the given module.
  }];

  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::plan::PlanDialect"
  ];
}


#endif // MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD
