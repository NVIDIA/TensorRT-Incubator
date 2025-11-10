// RUN: mlir-tensorrt-opt %s -split-input-file -plan-remove-equivalent-buffer-results | FileCheck %s

func.func @return_same_arg(%arg0: memref<10xf32, #plan.memory_space<device>>)
    -> memref<10xf32, #plan.memory_space<device>> {
  return %arg0 : memref<10xf32, #plan.memory_space<device>>
}
// CHECK-LABEL: @return_same_arg
// CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32, #plan.memory_space<device>>)
// CHECK-NOT: -> memref
// CHECK: return
// CHECK-NOT: %[[ARG0]]

// -----

func.func @return_after_cast(%arg0: memref<10xf32, #plan.memory_space<device>>)
    -> memref<?xf32, #plan.memory_space<device>> {
  %cast = memref.cast %arg0 : memref<10xf32, #plan.memory_space<device>> to memref<?xf32, #plan.memory_space<device>>
  return %cast : memref<?xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @return_after_cast
// CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32, #plan.memory_space<device>>)
// CHECK-NOT: -> memref
// CHECK: %[[CAST:.*]] = memref.cast
// CHECK: return
// CHECK-NOT: %[[CAST]]

// -----

func.func @return_after_reshape(%arg0: memref<12xf32, #plan.memory_space<device>>,
                                %shape: memref<2xindex>)
    -> memref<3x4xf32, #plan.memory_space<device>> {
  %reshape = memref.reshape %arg0(%shape) : (memref<12xf32, #plan.memory_space<device>>, memref<2xindex>)
      -> memref<3x4xf32, #plan.memory_space<device>>
  return %reshape : memref<3x4xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @return_after_reshape
// CHECK-SAME: (%[[ARG0:.*]]: memref<12xf32, #plan.memory_space<device>>, %[[SHAPE:.*]]: memref<2xindex>)
// CHECK-SAME: -> memref<3x4xf32, #plan.memory_space<device>>
// CHECK: %[[RESHAPE:.*]] = memref.reshape
// CHECK: return %[[RESHAPE]]

// -----

func.func @return_after_expand_shape(%arg0: memref<12xf32, #plan.memory_space<device>>)
    -> memref<3x4xf32, #plan.memory_space<device>> {
  %expand = memref.expand_shape %arg0 [[0, 1]] output_shape [3, 4] :
      memref<12xf32, #plan.memory_space<device>> into memref<3x4xf32, #plan.memory_space<device>>
  return %expand : memref<3x4xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @return_after_expand_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<12xf32, #plan.memory_space<device>>)
// CHECK-SAME: -> memref<3x4xf32, #plan.memory_space<device>>
// CHECK: %[[EXPAND:.*]] = memref.expand_shape
// CHECK: return %[[EXPAND]]

// -----

func.func @return_after_collapse_shape(%arg0: memref<3x4xf32, #plan.memory_space<device>>)
    -> memref<12xf32, #plan.memory_space<device>> {
  %collapse = memref.collapse_shape %arg0 [[0, 1]] :
      memref<3x4xf32, #plan.memory_space<device>> into memref<12xf32, #plan.memory_space<device>>
  return %collapse : memref<12xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @return_after_collapse_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x4xf32, #plan.memory_space<device>>)
// CHECK-SAME: -> memref<12xf32, #plan.memory_space<device>>
// CHECK: %[[COLLAPSE:.*]] = memref.collapse_shape
// CHECK: return %[[COLLAPSE]]

// -----

func.func @return_after_reshape_collapse_expand(
    %arg0: memref<3x64xf4E2M1FN, #plan.memory_space<device>>,
    %shape: memref<2xindex>)
    -> memref<3x64xf4E2M1FN, #plan.memory_space<device>> {
  %reshape = memref.reshape %arg0(%shape) :
      (memref<3x64xf4E2M1FN, #plan.memory_space<device>>, memref<2xindex>)
      -> memref<12x16xf4E2M1FN, #plan.memory_space<device>>
  %collapse = memref.collapse_shape %reshape [[0, 1]] :
      memref<12x16xf4E2M1FN, #plan.memory_space<device>> into memref<192xf4E2M1FN, #plan.memory_space<device>>
  %expand = memref.expand_shape %collapse [[0, 1]] output_shape [3, 64] :
      memref<192xf4E2M1FN, #plan.memory_space<device>> into memref<3x64xf4E2M1FN, #plan.memory_space<device>>
  return %expand : memref<3x64xf4E2M1FN, #plan.memory_space<device>>
}

// CHECK-LABEL: @return_after_reshape_collapse_expand
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x64xf4E2M1FN, #plan.memory_space<device>>, %[[SHAPE:.*]]: memref<2xindex>)
// CHECK-NOT: -> memref<3x64xf4E2M1FN
// CHECK: return

// -----

func.func @return_allocated_buffer(%arg0: memref<10xf32>) -> memref<10xf32> {
  %alloc = memref.alloc() : memref<10xf32>
  memref.copy %arg0, %alloc : memref<10xf32> to memref<10xf32>
  return %alloc : memref<10xf32>
}

// CHECK-LABEL: @return_allocated_buffer
// CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32>)
// CHECK-SAME: -> memref<10xf32>
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: return %[[ALLOC]]

// -----

func.func @multiple_args_results(
    %arg0: memref<10xf32>,
    %arg1: memref<20xf32>,
    %arg2: memref<30xf32>)
    -> (memref<10xf32>, memref<20xf32>, memref<30xf32>) {
  %alloc = memref.alloc() : memref<30xf32>
  memref.copy %arg2, %alloc : memref<30xf32> to memref<30xf32>
  return %arg0, %arg1, %alloc : memref<10xf32>, memref<20xf32>, memref<30xf32>
}

// CHECK-LABEL: @multiple_args_results
// CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: memref<20xf32>, %[[ARG2:.*]]: memref<30xf32>)
// CHECK-SAME: -> memref<30xf32>
// CHECK-NOT: -> (memref<10xf32>, memref<20xf32>, memref<30xf32>)
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: return %[[ALLOC]]

// -----

func.func @return_dynamic_to_static(%arg0: memref<?x?xf32>)
    -> memref<10x20xf32> {
  %cast = memref.cast %arg0 : memref<?x?xf32> to memref<10x20xf32>
  return %cast : memref<10x20xf32>
}

// CHECK-LABEL: @return_dynamic_to_static
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>)
// CHECK-NOT: -> memref
// CHECK: %[[CAST:.*]] = memref.cast
// CHECK: return
// CHECK-NOT: %[[CAST]]

// -----

func.func @return_partial_dynamic(%arg0: memref<10x?xf32>)
    -> memref<?x20xf32> {
  %cast = memref.cast %arg0 : memref<10x?xf32> to memref<?x20xf32>
  return %cast : memref<?x20xf32>
}

// CHECK-LABEL: @return_partial_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x?xf32>)
// CHECK-NOT: -> memref
// CHECK: %[[CAST:.*]] = memref.cast
// CHECK: return
// CHECK-NOT: %[[CAST]]

// -----

func.func public @complex_returns(
  %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, complex<f32>>},
  %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, complex<f32>>},
  %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>>})
   -> complex<f32> attributes {
    executor.func_abi = (complex<f32>, complex<f32>) -> complex<f32>
} {
  %0 = executor.abi.recv %arg0 : complex<f32>
  %1 = executor.abi.recv %arg1 : complex<f32>
  %2 = complex.add %0, %1 : complex<f32>
  %3 = executor.abi.send %2 to %arg2 : complex<f32>
  return %3 : complex<f32>
}
