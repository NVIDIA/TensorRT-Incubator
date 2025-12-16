// RUN: kernel-opt %s -split-input-file -kernel-refine-argument-layouts -verify-diagnostics | FileCheck %s

#dev = 1 : i64

gpu.module @test_full_specialization {
  func.func @test_kernel(%arg0: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<?xi32, strided<[?], offset: ?>>) {
    return
  }
}

// CHECK-LABEL: gpu.module @test_full_specialization
// CHECK-LABEL: func.func @test_kernel
//  CHECK-SAME: (%{{.+}}: memref<1x32x4xi32>, %{{.+}}: memref<?xi32>)

// CHECK-LABEL: func.func @main
func.func @main(%arg0: memref<1x32x4xi32, #dev>, %arg1: memref<?xi32, #dev>) -> memref<?xi32, #dev> {
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[cast0:.+]] = memref.cast %{{.*}} : memref<1x32x4xi32, 1> to memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, 1>
  // CHECK: %[[cast1:.+]] = memref.cast %{{.*}} : memref<?xi32, 1> to memref<?xi32, strided<[?], offset: ?>, 1>
  %arg0_casted = memref.cast %arg0 : memref<1x32x4xi32, #dev> to memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>
  %arg1_casted = memref.cast %arg1 : memref<?xi32, #dev> to memref<?xi32, strided<[?], offset: ?>, #dev>
  // CHECK: %[[in:.+]] = memref.cast %[[cast0]]
  // CHECK: %[[out:.+]] = memref.cast %[[cast1]]
  // CHECK: kernel.call @test_full_specialization::@test_kernel {{.*}} (%[[in]]) outs(%[[out]])
  kernel.call @test_full_specialization::@test_kernel grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0_casted) outs(%arg1_casted)
     : (memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>, memref<?xi32, strided<[?], offset: ?>, #dev>) -> ()

  // CHECK: %[[in:.+]] = memref.cast %[[cast0]]
  // CHECK: %[[out:.+]] = memref.cast %[[cast1]]
  // CHECK: kernel.ext_call @test_full_specialization::@test_kernel
  // CHECK-SAME: args(%[[in]], %[[out]] : memref<1x32x4xi32, 1>, memref<?xi32, 1>)
  kernel.ext_call @test_full_specialization::@test_kernel grid[%c1, %c1, %c1] block[%c32, %c1, %c1]
     args(%arg0_casted, %arg1_casted : memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>, memref<?xi32, strided<[?], offset: ?>, #dev>)
     result_aliases = [1], effects = ["r", "w"]


  return %arg1 : memref<?xi32, #dev>
}



// -----


#dev = 1 : i64

// Two callers, only arg1 can be refined.

gpu.module @test_partial_specialization {
  func.func @test_kernel(%arg0: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>) {
    return
  }
}

func.func @main(%arg0: memref<1x32x4xi32, #dev>, %arg1: memref<1x32x4xi32, #dev>, %arg2: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>) -> memref<1x32x4xi32, #dev> {
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  %arg0_casted = memref.cast %arg0 : memref<1x32x4xi32, #dev> to memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>
  %arg1_casted = memref.cast %arg1 : memref<1x32x4xi32, #dev> to memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>
  kernel.call @test_partial_specialization::@test_kernel grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0_casted) outs(%arg1_casted)
     : (memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>, memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>) -> ()
  kernel.call @test_partial_specialization::@test_kernel grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg2) outs(%arg1_casted)
     : (memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>, memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>, #dev>) -> ()
  return %arg1 : memref<1x32x4xi32, #dev>
}

// CHECK-LABEL: gpu.module @test_partial_specialization {
// CHECK-LABEL: func.func @test_kernel
//  CHECK-SAME: (%{{.+}}: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %{{.+}}: memref<1x32x4xi32>)



// -----

// No callers, assume it can be refined.

gpu.module @test_no_callers {
  func.func @test_kernel(%arg0: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %arg1: index, %arg2: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>) {
    return
  }
}

// CHECK-LABEL: gpu.module @test_no_callers
// CHECK-LABEL: func.func @test_kernel
//  CHECK-SAME: (%{{.+}}: memref<1x32x4xi32>, %{{.+}}: index, %{{.+}}: memref<1x32x4xi32>)


// -----

// Unknown symbol use, assume cannot refine.

gpu.module @test_unknown_symbol_use {
  func.func @test_kernel(%arg0: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %arg1: index, %arg2: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>) {
    return
  }

  // expected-warning @below {{cannot refine the argument types of kernel function test_kernel due to unknown symbol user}}
  func.func @other_user() attributes {
    other.symbol = @test_kernel
  } {
    return
  }
}

// CHECK-LABEL: gpu.module @test_unknown_symbol_use
//      CHECK: func.func @test_kernel
// CHECK-SAME: (%{{.+}}: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>, %{{.+}}: index, %{{.+}}: memref<1x32x4xi32, strided<[?, ?, ?], offset: ?>>)
