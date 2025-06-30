// RUN: executor-opt %s -test-executor-bufferization-pipeline -inline -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

!scalar_type = f32

func.func private @print_tensor(
    %arg0: memref<4x!scalar_type, strided<[?], offset: ?>, #executor.memory_type<host>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %el = memref.load %arg0 [%i] :
       memref<4x!scalar_type, strided<[?], offset: ?>, #executor.memory_type<host>>
    executor.print "[%d] = %.2f"(%i, %el : index, !scalar_type)
  }
  return
}

func.func private @fill_tensor(%arg1: !scalar_type, %arg0: tensor<4x!scalar_type>) -> tensor<4x!scalar_type> {
  %1 = linalg.fill ins(%arg1: !scalar_type) outs(%arg0: tensor<4x!scalar_type>) -> tensor<4x!scalar_type>
  return %1 : tensor<4x!scalar_type>
}

func.func @main() -> i32 {

  %0 = arith.constant dense<0.0> : tensor<4x!scalar_type>
  %fill_value = arith.constant 1.1 : !scalar_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %1 = func.call @fill_tensor(%fill_value, %0) : (!scalar_type, tensor<4x!scalar_type>) -> tensor<4x!scalar_type>

  %memref = bufferization.to_memref %1 read_only :
     tensor<4x!scalar_type> to memref<4x!scalar_type, strided<[?], offset: ?>, #executor.memory_type<host>>
  func.call @print_tensor(%memref) :
     (memref<4x!scalar_type, strided<[?], offset: ?>, #executor.memory_type<host>>) -> ()

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}

//      CHECK: [0] = 1.1
// CHECK-NEXT: [1] = 1.1
// CHECK-NEXT: [2] = 1.1
// CHECK-NEXT: [3] = 1.1
