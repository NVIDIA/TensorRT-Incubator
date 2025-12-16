// RUN: kernel-opt %s --kernel-lower-sort | FileCheck %s

// CHECK: gpu.module @[[gpu_module_name_kv_sort:[a-zA-Z0-9_]+]]
// CHECK:    func.func @[[block_sort_kernel_kv:.+]](
// CHECK:    func.func @[[partition_kernel_kv:.+]](
// CHECK:    func.func @[[merge_kernel_kv:.+]](

// CHECK: func.func private @[[merge_sort_dispatch_kv:[a-zA-Z0-9_]+]]
// CHECK:         kernel.ext_call @[[gpu_module_name_kv_sort]]::@[[block_sort_kernel_kv]]
// CHECK:         kernel.ext_call @[[gpu_module_name_kv_sort]]::@[[partition_kernel_kv]]
// CHECK:         kernel.ext_call @[[gpu_module_name_kv_sort]]::@[[merge_kernel_kv]]

// CHECK: gpu.module @[[gpu_module_name_keys_only:[a-zA-Z0-9_]+]]
// CHECK:    func.func @[[block_sort_kernel_keys_only:.+]](
// CHECK:    func.func @[[partition_kernel_keys_only:.+]](
// CHECK:    func.func @[[merge_kernel_keys_only:.+]](

// CHECK: func.func private @[[merge_sort_dispatch_keys_only:[a-zA-Z0-9_]+]](
// CHECK:         kernel.ext_call @[[gpu_module_name_keys_only]]::@[[block_sort_kernel_keys_only]]
// CHECK:         kernel.ext_call @[[gpu_module_name_keys_only]]::@[[partition_kernel_keys_only]]
// CHECK:         kernel.ext_call @[[gpu_module_name_keys_only]]::@[[merge_kernel_keys_only]]

// CHECK-LABEL: func.func @sort_keys_only
// CHECK-NOT:     kernel.sort
// CHECK:         %[[RESULT:.+]] = call @[[merge_sort_dispatch_keys_only]]
// CHECK:         return %[[RESULT]]
func.func @sort_keys_only(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = kernel.sort (%arg0) <block_threads = 128, items_per_thread = 4> : tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @sort_static
// CHECK-NOT:     kernel.sort
// CHECK:         %[[values_cast:.+]] = tensor.cast
// CHECK:         %[[keys_cast:.+]] = tensor.cast
// CHECK-DAG:     %[[RESULT:.+]]:2 = call @[[merge_sort_dispatch_kv]](%[[values_cast]], %{{.*}}, %[[keys_cast]])
// CHECK-DAG:     %[[keys_cast_2:.+]] = tensor.cast %[[RESULT]]#0 : tensor<?xi32> to tensor<17xi32>
// CHECK-DAG:     %[[values_cast_2:.+]] = tensor.cast %[[RESULT]]#1 : tensor<?xf32> to tensor<17xf32>
// CHECK:         return %[[keys_cast_2]], %[[values_cast_2]]
func.func @sort_static(%arg0: tensor<17xi32>, %arg1: tensor<17xf32>) -> (tensor<17xi32>, tensor<17xf32>) {
  %0, %1 = kernel.sort (%arg0, %arg1) <block_threads = 128, items_per_thread = 4> : tensor<17xi32>, tensor<17xf32>
  return %0, %1 : tensor<17xi32>, tensor<17xf32>
}
