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
#ifndef MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

defvar InputKindOption = Option<"inputKind", "input-kind", "::mlir::plan::InputKind",
      "::mlir::plan::InputKind::Stablehlo",
      "specifies the kind of input IR dialect",
      "::mlir::plan::detail::createInputKindClOptions()">;

//===----------------------------------------------------------------------===//
// PopulateDefaultBackendMetadataPass
//===----------------------------------------------------------------------===//

def PopulateDefaultBackendMetadataPass : Pass<"plan-populate-default-backend-metadata",
      "::mlir::ModuleOp"> {
  let summary = "Populate Plan dialect backend metadata information if not "
                "already present";

  let description = [{
    The compiler uses an attribute 'plan.backends' located on the module
    in order to understand the relative priority of each backend. If the
    attribute is not present, then this pass will populate it with a default
    value.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];

  let options = [
    ListOption<"backends", "backends", "std::string",
    "A list of default backend attributes to use for compilation if no "
    "'plan.backends' attribute is provided on the top-level module.",
    "llvm::cl::list_init<std::string>({\"#plan.host_backend<benefit=1>\"})">
  ];
}

//===----------------------------------------------------------------------===//
// VerifyInputAndAssignSlotsPass
//===----------------------------------------------------------------------===//

def VerifyInputAndAssignSlotsPass : Pass<"plan-verify-input-and-assign-slots",
                                         "::mlir::ModuleOp"> {
  let description = [{
    This pass verifies input IR pre-conditions and assigns slot numbers to
    the arguments and results.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LegalizeIOBoundsAttributesPass
//===----------------------------------------------------------------------===//

def LegalizeIOBoundsAttributesPass : Pass<"plan-legalize-io-bounds-attributes",
                                         "::mlir::ModuleOp"> {
  let description = [{
    This pass legalizes argument and result attributes that denote
    shape and value bounds.

    Some frontends use `tensorrt` dialect attributes for specifying the
    shape/value bounds of arguments. For compatibility, this pass supports
    converting them to `plan` dialect attributes.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

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

  let options = [
    InputKindOption
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

  let options = [
    InputKindOption
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
    "::mlir::plan::PlanDialect",
    "::mlir::executor::ExecutorDialect"
  ];

  let options = [
    Option<"forceUndefOutputArgs", "force-undef-output-args", "bool", "false",
      "force undef ABI attribute to all output arguments">,
    Option<"abiVersion", "abi-version", "int", "1",
      "the version of the ABI to use">,
  ];
}

//===----------------------------------------------------------------------===//
// PlanPopulateFunctionBoundsAttributesPass
//===----------------------------------------------------------------------===//

def PlanPopulateFunctionBoundsAttributesPass
      : InterfacePass<"plan-populate-func-bounds-attrs", "FunctionOpInterface"> {

  let description = [{
    This pass populates function result attributes containing shape/value bounds
    information for function results.
  }];
}

//===----------------------------------------------------------------------===//
// ClusteringPass
//===----------------------------------------------------------------------===//

def ClusteringPass : Pass<"plan-clustering", "::mlir::ModuleOp"> {
  let summary = "clusters operations into different regions based"
                " on provided backend configurations";

  let description = [{
    This pass clusters groups of operations into encapsulating region
    operations that are not isolated-from-above. The purpose of this pass
    is to achieve a course segmentation that specifies how clusters of
    operations will be compiled.

    The kinds of clusters that can be formed and the specific rules for
    clustering are defined by the clustering configuration specified
    by the module's `plan.backends` attribute. This is an array of
    attributes which all implement the
    [CompilerBackendAttrInterface](../IR/PlanInterfaces.td).
  }];

  let options =
      [Option<"entrypoint", "entrypoint", "std::string", "\"\"",
              "the name of the entrypoint function; if empty then the "
              "clustering runs"
              " on all functions">,
       Option<"forceEntrypointsReturnAllocs", "force-entrypoints-return-allocs",
              "bool", "false",
              "allow backend clusters to directly allocate outputs">,
       Option<"disableCreateShapeFuncPass", "disable-create-shape-func-pass",
              "bool", "false",
              "don't apply create shape to func pass in TensorRT clusters">,
       InputKindOption];
}

//===----------------------------------------------------------------------===//
// CreateClosedRegionsPass
//===----------------------------------------------------------------------===//

def CreateClosedRegionsPass : Pass<"plan-create-closed-regions", "::mlir::ModuleOp"> {

  let description = [{
    This pass performs an intermediate lowering step of `plan.cluster`
    operations prior to the `plan-outline-clusters` pass.

    Certain cluster regions such as those targeting TensorRT are required to
    be replaced by a destination-passing-style (DPS) call-like operation
    in the outlining step. However, the clustered IR inside the region
    is StableHlo IR, whose operations do not follow
    destination-passing-style. When the results of the region op are
    dynamically shaped, the logic for materializing the output buffers
    above the region can become complicated.

    This pass replaces such cluster regions with `plan.dps_cluster`
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
       case, we cannot materialize a `plan.dps_cluster` operation and
       instead the region must be lowered to a call-like operation that for the
       relevant backend which takes on an allocation-like semantic when
       bufferized. This is currently not supported for any MLIR-TRT backend,
       although we plan to support this route with TensorRT in the future.

  }];

  let options = [
    Option<"testPreWalkOrder", "test-pre-walk-order", "bool", "false",
      "(used only in testing) specifies to outline regions by walking in "
      " pre-order; used for verifying results are not sensitive "
      "to traversal order">,
    Option<"disableDestinationStyleCallingConvention", "disable-destination-style-calling-convention", "bool",
           /*default=*/"false",
           "Disable destination-passing-style (DPS) calling convention lowering">,
    InputKindOption
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect"
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

  let options = [
    InputKindOption
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
// PlanAssignMemorySpacesPass
//===----------------------------------------------------------------------===//

def PlanAssignMemorySpacesPass : Pass<"plan-assign-memory-spaces",
                                    "::mlir::ModuleOp"> {
  let summary = "assigns memory spaces encodings to tensor types";

  let description = [{
    This pass applies a type conversion that adds a '#plan.memory_space'
    attribute to all tensor types in the top-level module that do not already
    have an encoding.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanOptimizeMemorySpacesPass
//===----------------------------------------------------------------------===//

def PlanOptimizeMemorySpacesPass : InterfacePass<"plan-optimize-memory-spaces",
                                    "::mlir::FunctionOpInterface"> {
  let summary = "optimizes memory spaces encodings to tensor types";

  let description = [{
    This pass applies a set of transformations that attempt to optimize the
    memory space encodings of tensor types in terms of host vs. device
    placement. This includes changes such as (but not limited to):

    - Removing redundant memory space changes.
    - Hoisting memory space changes out of loops.
    - Ensuring operations that require certain operands to live in specific
      memory spaces (host vs. device) have such constraints met.

    Note that this pass only deals with 'host' and 'device' memory spaces. The
    current contract is that use of other specialized memory spaces (e.g.
    `host_pinned`) is done via follow-on specialized optimization passes.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanPromoteHostTensorsToHostPinnedPass
//===----------------------------------------------------------------------===//

def PlanPromoteHostTensorsToHostPinnedPass
        : InterfacePass<"plan-promote-host-tensors-to-host-pinned",
                        "::mlir::FunctionOpInterface"> {
  let summary = "promotes host tensors to host pinned tensors";

  let description = [{
    This pass finds host tensors which are ideal candidates for promotion to the
    'host-pinned' memory space. This pass must be run after the
    `plan-optimize-memory-spaces` pass.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanMaterializeExplicitTransfersPass
//===----------------------------------------------------------------------===//

def PlanMaterializeExplicitTransfersPass
                                : Pass<"plan-materialize-explicit-transfers"> {
  let summary = "Turn `tensor.cast` that cast between memory spaces into "
                "explicit transfers using bufferization ops.";

  let description = [{
    This pass materializes explicit transfers between memory spaces by
    lowering `tensor.cast` operations that change the memory space specified
    by the tensor encoding attributes of the operand/result types.

    The transfers are materialized as explicit `bufferization.alloc_tensor`
    and `bufferization.materialize_in_destination` operations to perform the
    copy (the more concise `bufferization.alloc_tensor` with `copy` operand
    currently cannot change between memory spaces).
  }];

  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::plan::PlanDialect"
  ];
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

    For public functions in the top-level module, this pass will
    also perform a transformation into destination-passing-style unless the
    'force-entrypoints-return-allocs' flag is set.
  }];

  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::plan::PlanDialect"
  ];

  let options = [
    Option<"forceEntrypointsReturnAllocs", "force-entrypoints-return-allocs", "bool",
           /*default=*/"false",
          "Require entrypoint functions to return allocations corresponding to"
          " the original tensor results, otherwise they are transformed"
          " into destination arguments whenever possible.">
  ];
}

//===----------------------------------------------------------------------===//
// PlanModuleBufferizePass
//===----------------------------------------------------------------------===//

def PlanModuleBufferizePass : Pass<"plan-module-bufferize",
                                   "::mlir::ModuleOp"> {
  let summary = "Run a specialized one-shot-module-bufferization";

  let description = [{
    The 'plan-module-bufferization' pass is equivalent to the upstream
    `one-shot-(module)-bufferization`, but it bufferizes a potentially nested
    set of modules at once. Nested modules are only bufferized if they implement
    the `BufferizationScopeOpInterface`.

    It is assumed that the outer-most module is the 'host' module.

    At the end of the pass, we fixup issues related to address spaces:

    - Any store/load from a non-host-visible address space in the host program
      are rewritten to insert the appropriate copies.
  }];

  let options = [
    Option<"allowReturnAllocsFromLoops", "allow-return-allocs-from-loops",
           "bool", /*default=*/"true",
           "Allows returning/yielding new allocations from a loop.">,
    Option<"analysisFuzzerSeed", "analysis-fuzzer-seed", "unsigned",
           /*default=*/"0",
           "Test only: Analyze ops in random order with a given seed (fuzzer)">,
    Option<"checkParallelRegions", "check-parallel-regions", "bool",
           /*default=*/"true", "Account for parallel regions in RaW analysis.">,
    Option<"copyBeforeWrite", "copy-before-write", "bool", /*default=*/"false",
           "Skip the analysis. Make a buffer copy on every write.">,
    Option<"dumpAliasSets", "dump-alias-sets", "bool", /*default=*/"false",
           "Test only: Annotate tensor IR with alias sets">,
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Test only: Only run inplaceability analysis and annotate IR">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Test only: Annotate IR with RaW conflicts. Requires "
           "test-analysis-only.">
  ];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanConfirmArgumentDonation
//===----------------------------------------------------------------------===//
def PlanConfirmArgumentDonationPass : Pass<"plan-confirm-argument-donation",
                                            "func::FuncOp"> {
  let summary = "Confirms if donated arguments are used after bufferization.";

  let description = [{
    For each argument marked with `plan.aliasing_output = N`, check whether the
    N-th returned value from the function is equivalent to the argument, after
    bufferization. If not, either emit an error (if `fail-on-donation-argument-
    rejection=true`) or remove the `plan.aliasing_output` attribute.
  }];

  let options = [
    Option<"failOnDonationArgumentRejection", "fail-on-donation-argument-rejection",
           "bool", /*default=*/"false",
           "Allows controlling whether pass should fail if aliasing fails after "
           "bufferization.">,
  ];
}

//===----------------------------------------------------------------------===//
// PlanRemoveEquivalentBufferResultsPass
//===----------------------------------------------------------------------===//

def PlanRemoveEquivalentBufferResultsPass : Pass<"plan-remove-equivalent-buffer-results"> {
  let summary = "Remove equivalent buffer results of functions";

  let description = [{
    This pass removes function memref results that are equivalent to block
    arguments. The logic is similar to the corresponding upstream pass, but
    we handle potentially nested modules.
  }];
}

//===----------------------------------------------------------------------===//
// PlanBufferResultsToOutParamsPass
//===----------------------------------------------------------------------===//

def PlanBufferResultsToOutParamsPass : Pass<"plan-buffer-results-to-out-params",
                                           "::mlir::ModuleOp"> {
  let summary = "Convert buffer results to out params";

  let description = [{
    This pass converts function memref results to out params. There is a similar
    upstream pass, but our version is more advanced and can handle promoting
    a set of memref results.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect"
  ];

  let options = [
    Option<"ignorePublicFunctions", "ignore-public-functions", "bool",
      "false", "do not apply the transformation on public functions">
  ];
}

//===----------------------------------------------------------------------===//
// PlanOwnershipBasedBufferDeallocationPass
//===----------------------------------------------------------------------===//

def PlanOwnershipBasedBufferDeallocationPass : Pass<
      "plan-ownership-based-buffer-deallocation",
      "::mlir::ModuleOp"> {
  let summary = "Perform ownership-based buffer deallocation";

  let description = [{
    This pass runs the ownership-based buffer deallocation transformation. We
    duplicate the pass stub locally from upstream since the upstream
    transformation also runs on nested modules, but we limit deallocation to
    just the host module.
  }];

  let options = [
    Option<"privateFuncDynamicOwnership", "private-function-dynamic-ownership",
           "bool", /*default=*/"false",
           "Allows to add additional arguments to private functions to "
           "dynamically pass ownership of memrefs to callees. This can enable "
           "earlier deallocations.">,
  ];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanOutlineConstantFoldableSubgraphs
//===----------------------------------------------------------------------===//

def PlanOutlineConstantFoldableSubgraphsPass : Pass<
      "plan-outline-constant-foldable-subgraphs",
      "::mlir::ModuleOp"> {
  let summary = "Analyze and outline constant foldable subgraphs";

  let description = [{
    This pass implements forward dataflow analysis (named `SparseConstantFoldabilityAnalysis`)
    to find out constant foldable ops. This analysis, unlike upstream
    `ConstantPropagationAnalysis` is very simple and works only for pure ops.
    If all operands of an operation are constant foldable, all results are marked
    as constant foldable.
    Constant foldability analysis is then used along with clustering to
    find constant foldable subgraphs. These constant foldable subgraphs are
    finally outlined to a private function with `plan.constant_foldable` attribute.
  }];
}

#endif // MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_TD
