// RUN: mlir-tensorrt-opt %s -split-input-file -plan-bufferize-pipeline | FileCheck %s

executor.plugin @test_plugin {config = {param = 3 : i64}, ffi_backend = #executor.ffi_backend<tvm_ffi>,
  function_name = "some_func", plugin_name = "some_lib.so"}
   : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
func.func @test_io_aliasing(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  // This config says that each input is written in-place.
  // Note that the relative order of the results is reversed.
  // Result 0 will be copied to output 1, and result 1 will be copied to output 0.
  %0:2 = executor.call_plugin @test_plugin
  ins(%arg0, %arg1 : tensor<5xf32>, tensor<5xf32>) outs(%arg1, %arg0 : tensor<5xf32>, tensor<5xf32>)
      {arg_spec = ["attrs.param;args;rets"],
      immediate_args = {param = 3 : i64},
      io_aliasing = array<i32: 1, 0>} : tensor<5xf32>, tensor<5xf32>
  return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
}

// CHECK-LABEL: func.func @test_io_aliasing
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}, %[[arg3:.+]]: {{.*}})
//       CHECK:     executor.call_plugin @test_plugin ins(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[arg1]], %[[arg0]] :
//       CHECK:     memref.copy %[[arg1]], %[[arg2]]
//       CHECK:     memref.copy %[[arg0]], %[[arg3]]
//       CHECK:     return

// -----

executor.plugin @test_plugin {config = {param = 3 : i64}, ffi_backend = #executor.ffi_backend<tvm_ffi>,
  function_name = "some_func", plugin_name = "some_lib.so"}
   : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
func.func @test_memory_space_requirements(
           %arg0: tensor<5xf32> {plan.memory_space = #plan.memory_space<device>},
           %arg1: tensor<5xf32> {plan.memory_space = #plan.memory_space<host>})
    -> (tensor<5xf32>, tensor<5xf32>) {
  %0:2 = executor.call_plugin @test_plugin
  ins(%arg0, %arg1 : tensor<5xf32>, tensor<5xf32>) outs(%arg0, %arg1 : tensor<5xf32>, tensor<5xf32>)
      {arg_spec = ["args;rets"],
      io_aliasing = array<i32: 0, 1>} : tensor<5xf32>, tensor<5xf32>
  return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
}

// CHECK-LABEL: func.func @test_memory_space_requirements
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}, %[[arg3:.+]]: {{.*}})
//       CHECK:     memref.copy %[[arg1]], %[[arg3]]
//       CHECK:     executor.call_plugin @test_plugin ins(%[[arg0]], %[[arg3]] : {{.*}}) outs(%[[arg0]], %[[arg3]] :
//       CHECK:     memref.copy %[[arg0]], %[[arg2]]
//       CHECK:     return

// -----

executor.plugin @test_plugin {config = {param = 3 : i64}, ffi_backend = #executor.ffi_backend<tvm_ffi>,
  function_name = "some_func", plugin_name = "some_lib.so"} : (tensor<?xf32>) -> (tensor<5xf32>)
func.func @test_input_layout(
           %arg0: tensor<?xf32> {plan.memory_space = #plan.memory_space<device>})
    -> (tensor<5xf32>) {
  %0 = tensor.empty() : tensor<5xf32>
  %1 = tensor.extract_slice %arg0[0] [5] [2] : tensor<?xf32> to tensor<5xf32>
  %2 = executor.call_plugin @test_plugin
    ins(%1 : tensor<5xf32>) outs(%0 : tensor<5xf32>) {arg_spec = ["args;rets"]} : tensor<5xf32>
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @test_input_layout
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
//       CHECK:     %[[subview:.+]] = memref.subview %[[arg0]][0] [5] [2] :
//       CHECK:     %[[alloc:.+]] = memref.alloc()
//       CHECK:     memref.copy %[[subview]], %[[alloc]] :
//       CHECK:     executor.call_plugin @test_plugin ins(%[[alloc]] : {{.*}}) outs(%[[arg1]] : {{.*}})
//       CHECK:     memref.dealloc %[[alloc]] : memref<5xf32, #plan.memory_space<device>>
//       CHECK:     return

// -----

executor.plugin @test_plugin {config = {param = 3 : i64}, ffi_backend = #executor.ffi_backend<tvm_ffi>,
  function_name = "some_func", plugin_name = "some_lib.so"} : (tensor<5xf32>) -> (tensor<5xf32>)
func.func @test_output_layout(%arg0: tensor<5xf32>, %arg1: tensor<?xf32>) -> (tensor<5xf32>) {
  %1 = tensor.extract_slice %arg1[0] [5] [2] : tensor<?xf32> to tensor<5xf32>
  %2 = executor.call_plugin @test_plugin
    ins(%arg0 : tensor<5xf32>) outs(%1 : tensor<5xf32>) {arg_spec = ["args;rets"]} : tensor<5xf32>
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @test_output_layout
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}, %[[arg2:.+]]: {{.*}}) {
//       CHECK:     %[[subview:.+]] = memref.subview %[[arg1]][0] [5] [2] : {{.*}} to {{.*}}
//       CHECK:     %[[alloc:.+]] = memref.alloc()
//       CHECK:     memref.copy %[[subview]], %[[alloc]] : {{.*}} to {{.*}}
//       CHECK:     executor.call_plugin @test_plugin ins(%[[arg0]] : {{.*}}) outs(%[[alloc]] 
//       CHECK:     memref.copy %[[alloc]], %[[arg2]] : {{.*}} to {{.*}}
//       CHECK:     memref.dealloc %[[alloc]] : {{.*}}
//       CHECK:     return
