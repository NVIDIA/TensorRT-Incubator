// RUN: executor-opt %s -test-executor-bufferization-pipeline -inline -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe | FileCheck %s

!memref_type = memref<4xcomplex<f32>, strided<[?], offset: ?>, #executor.memory_type<host>>

func.func private @print_tensor(%arg0: !memref_type) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %el = memref.load %arg0 [%i] : !memref_type
    %re = complex.re %el : complex<f32>
    %im = complex.im %el : complex<f32>
    executor.print "[%d] = (%.2f, %.2f)"(%i, %re, %im : index, f32, f32)
  }
  return
}

func.func @main() -> i32 {

  %0 = arith.constant dense<[(1.0, 2.0), (2., 3.), (4., 5.), (5., 6.)]> : tensor<4xcomplex<f32>>
  %1 = arith.constant dense<[(1.0, -2.0), (-2., 3.), (-4., -5.), (5.9, 6.9)]> : tensor<4xcomplex<f32>>
  %2 = bufferization.alloc_tensor() : tensor<4xcomplex<f32>>
  %alloc1 = bufferization.alloc_tensor() : tensor<4xcomplex<f32>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %result:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %2, %acc1 = %alloc1) -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) {
    %lhs = tensor.extract %0[%i] : tensor<4xcomplex<f32>>
    %rhs = tensor.extract %1[%i] : tensor<4xcomplex<f32>>
    %add = complex.add %lhs, %rhs :complex<f32>
    %sub = complex.sub %lhs, %rhs :complex<f32>

    %updated0 = tensor.insert %add into %acc[%i] : tensor<4xcomplex<f32>>
    %updated1 = tensor.insert %sub into %acc1[%i] : tensor<4xcomplex<f32>>

    scf.yield %updated0, %updated1 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>
  }

  %memref = bufferization.to_memref %result#0 read_only : tensor<4xcomplex<f32>> to !memref_type
  func.call @print_tensor(%memref) : (!memref_type) -> ()
  %memref1 = bufferization.to_memref %result#1 read_only : tensor<4xcomplex<f32>> to !memref_type
  func.call @print_tensor(%memref1) : (!memref_type) -> ()

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}

//      CHECK: [0] = (2.00, 0.00)
// CHECK-NEXT: [1] = (0.00, 6.00)
// CHECK-NEXT: [2] = (0.00, 0.00)
// CHECK-NEXT: [3] = (10.90, 12.90)

// CHECK-NEXT: [0] = (0.00, 4.00)
// CHECK-NEXT: [1] = (4.00, 0.00)
// CHECK-NEXT: [2] = (8.00, 10.00)
// CHECK-NEXT: [3] = (-0.90, -0.90)
