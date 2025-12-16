// RUN: kernel-opt -split-input-file %s | kernel-opt | FileCheck %s

gpu.module @kernels {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xf32>
    memref.store %0, %arg1[%c0] : memref<1xf32>
    return
  }
  func.func @kernel_with_results(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<1xf32>
    %1 = tensor.insert %0 into %arg1[%c0] : tensor<1xf32>
    return %1 : tensor<1xf32>
  }
}

func.func @caller(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index, %arg3: index) {
  // These conventions are technically both fine, although we may consider requiring
  // calls to have non-empty outs.
  kernel.call @kernels::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

func.func @caller_memspace(%arg0: memref<1xf32>,
  %arg1: memref<1xf32>,
  %arg2: index, %arg3: index) {
  kernel.call @kernels::@kernel grid[%arg2] block[%arg3] (%arg0) outs(%arg1) :
    (memref<1xf32>, memref<1xf32>) -> ()
  return
}

func.func @caller_with_results(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: index, %arg3: index) -> tensor<1xf32> {
  %0 = kernel.call @kernels::@kernel_with_results grid[%arg2] block[%arg3] (%arg0) outs(%arg1) : (tensor<1xf32>, tensor<1xf32>)
     -> (tensor<1xf32>)
  return %0 : tensor<1xf32>
}

//       CHECK: gpu.module @kernels {
// CHECK-LABEL: func.func @kernel
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32>, %[[arg1:.+]]: memref<1xf32>)
//       CHECK:       %[[c0:.+]] = arith.constant 0 : index
//       CHECK:       %[[v0:.+]] = memref.load %[[arg0]][%[[c0]]] : memref<1xf32>
//       CHECK:       memref.store %[[v0]], %[[arg1]][%[[c0]]] : memref<1xf32>
// CHECK-LABEL: func.func @kernel_with_results
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32>, %[[arg1:.+]]: tensor<1xf32>) -> tensor<1xf32> {
//       CHECK:       %[[c0:.+]] = arith.constant 0 : index
//       CHECK:       %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xf32>
//       CHECK:       %[[inserted:.+]] = tensor.insert %[[extracted]] into %[[arg1]][%[[c0]]] : tensor<1xf32>
//       CHECK:       return %[[inserted]] : tensor<1xf32>

// CHECK-LABEL: @caller
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32>, %[[arg1:.+]]: memref<1xf32>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
//       CHECK:     kernel.call @kernels::@kernel grid[%[[arg2]]] block[%[[arg3]]] (%[[arg0]]) outs(%[[arg1]])

// CHECK-LABEL: @caller_memspace
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32>, %[[arg1:.+]]: memref<1xf32>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
//       CHECK:     kernel.call @kernels::@kernel grid[%[[arg2]]] block[%[[arg3]]] (%[[arg0]]) outs(%[[arg1]])

// CHECK-LABEL: func.func @caller_with_results
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32>, %[[arg1:.+]]: tensor<1xf32>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) -> tensor<1xf32>
//       CHECK:     %[[v0:.+]] = kernel.call @kernels::@kernel_with_results grid[%[[arg2]]] block[%[[arg3]]] (%[[arg0]]) outs(%[[arg1]])
//       CHECK:     return %[[v0]] : tensor<1xf32>

// -----

func.func @test_matmul_shape_attr() attributes {
  shape1 = #kernel.tensorcore<f32, f32, f32, f32, 16x8x8>
} {
  return
}

// CHECK-LABEL: func.func @test_matmul_shape_attr
//  CHECK-SAME:  shape1 = #kernel.tensorcore<f32, f32, f32, f32, 16x8x8>


// -----

func.func @test_associative(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
  %0, %1 = kernel.combiner (%arg0, %arg1, %arg2, %arg3) : f32, f32, f32, f32 {
  ^bb0(%a: f32, %c: f32, %b: f32, %d: f32):
    %max = arith.maximumf %a, %b : f32
    %0 = arith.subf %a, %max : f32
    %1 = arith.subf %b, %max : f32
    %2 = math.exp %0 : f32
    %3 = math.exp %1 : f32
    %4 = arith.mulf %2, %c : f32
    %5 = arith.addf %4, %3 : f32
    kernel.yield %max, %5 : f32, f32
  }
  return %0, %1 : f32, f32
}

// CHECK-LABEL: func.func @test_associative
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: f32, %[[arg3:.+]]: f32)
//       CHECK:     kernel.combiner(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]]) : f32, f32, f32, f32 {
//       CHECK:     ^bb0(%[[arg4:.+]]: f32, %[[arg5:.+]]: f32, %[[arg6:.+]]: f32, %[[arg7:.+]]: f32):
//       CHECK:       kernel.yield %{{.+}}, %{{.+}} : f32, f32

// -----

// CHECK-LABEL: func.func @test_sort(
func.func @test_sort(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: kernel.sort(%{{.+}}) <block_threads = 128, items_per_thread = 4> : tensor<?xf32>
  %0 = kernel.sort (%arg0) <block_threads = 128, items_per_thread = 4> : tensor<?xf32>
  return %0 : tensor<?xf32>
}
