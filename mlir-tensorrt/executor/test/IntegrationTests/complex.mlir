 // RUN: executor-opt %s \
 // RUN:  -executor-lowering-pipeline \
 // RUN:  | executor-translate -mlir-to-runtime-executable \
 // RUN:  | executor-runner -input-type=rtexe -features=core \
 // RUN:  | FileCheck %s

func.func @print_complex(%arg0: complex<f32>) {
  %r = complex.re %arg0 : complex<f32>
  %i = complex.im %arg0 : complex<f32>
  executor.print "%f + %fj"(%r, %i : f32, f32)
  return
}

func.func @print_complex64(%arg0: complex<f64>) {
  %r = complex.re %arg0 : complex<f64>
  %i = complex.im %arg0 : complex<f64>
  executor.print "%f + %fj"(%r, %i : f64, f64)
  return
}

func.func @test_complex_add(%arg0: complex<f32>, %arg1: complex<f32>) {
  %0 = complex.add %arg0, %arg1 : complex<f32>
  func.call @print_complex(%0) : (complex<f32>) -> ()
  return
}

func.func @test_complex_sub(%arg0: complex<f64>, %arg1: complex<f64>) {
  %c = complex.sub %arg0, %arg1 : complex<f64>
  func.call @print_complex64(%c) : (complex<f64>) -> ()
  return
}

func.func @test_complex_mul(%arg0: complex<f64>, %arg1: complex<f64>) {
  %c = complex.mul %arg0, %arg1 : complex<f64>
  func.call @print_complex64(%c) : (complex<f64>) -> ()
  return
}

func.func @test_load(%arg0: memref<4xcomplex<f32>>, %arg1: index) {
  %0 = memref.load %arg0[%arg1] : memref<4xcomplex<f32>>
  executor.print "test_load[%d] = "(%arg1 : index)
  func.call @print_complex(%0) : (complex<f32>) -> ()
  return
}

memref.global @global_memref : memref<4xcomplex<f32>> = dense<
  [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]>

func.func @main() -> i32 {

  %cst = complex.constant [0.1 : f32, -1.0 : f32] : complex<f32>
  %cst1 = complex.constant [0.2 : f32, 1.0 : f32] : complex<f32>

  func.call @test_complex_add(%cst, %cst1) : (complex<f32>, complex<f32>) -> ()

  %a_re = arith.constant 1.2 : f64
  %a_im = arith.constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = arith.constant 5.6 : f64
  %b_im = arith.constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>

  func.call @test_complex_sub(%a, %b) : (complex<f64>, complex<f64>) -> ()
  func.call @test_complex_mul(%a, %b) : (complex<f64>, complex<f64>) -> ()

  %glob = memref.get_global @global_memref : memref<4xcomplex<f32>>
  %c2_index = arith.constant 2 : index
  func.call @test_load(%glob, %c2_index) : (memref<4xcomplex<f32>>, index) -> ()

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}

// CHECK: 0.3{{0+}} + 0.{{0+}}j
// CHECK: -4.4{{0+}} + -4.4{{0+}}j
// CHECK: -19.8{{0+}} + 28.4{{0+}}j
