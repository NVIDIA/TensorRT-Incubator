// RUN: rm %t.mlir || true
// RUN: executor-opt %s -executor-lowering-pipeline -o %t.mlir

// RUN: executor-translate -mlir-to-lua  %t.mlir \
// RUN:   | executor-runner -input-type=lua -features=core | FileCheck %s

// RUN: executor-opt %s -executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

func.func @test_ctpop_i64(%arg0: i64) {
  %0 = executor.ctpop %arg0 : i64
  executor.print "ctpop_i64(%d) = %d"(%arg0, %0 : i64, i64)
  return
}

func.func @test_ctpop_i32(%arg0: i32) {
  %0 = executor.ctpop %arg0 : i32
  executor.print "ctpop_i32(%d) = %d"(%arg0, %0 : i32, i32)
  return
}

func.func @main() -> i64 {
  // Test i64
  // 0 has 0 bits set
  %i64_0 = executor.constant 0 : i64
  // 1 (0b1) has 1 bit set
  %i64_1 = executor.constant 1 : i64
  // 7 (0b111) has 3 bits set
  %i64_7 = executor.constant 7 : i64
  // 255 (0b11111111) has 8 bits set
  %i64_255 = executor.constant 255 : i64
  // -1 has all 64 bits set
  %i64_m1 = executor.constant -1 : i64

  func.call @test_ctpop_i64(%i64_0) : (i64) -> ()
  func.call @test_ctpop_i64(%i64_1) : (i64) -> ()
  func.call @test_ctpop_i64(%i64_7) : (i64) -> ()
  func.call @test_ctpop_i64(%i64_255) : (i64) -> ()
  func.call @test_ctpop_i64(%i64_m1) : (i64) -> ()

  // Test i32
  %i32_0 = executor.constant 0 : i32
  // 15 (0b1111) has 4 bits set
  %i32_15 = executor.constant 15 : i32
  // -1 has all 32 bits set
  %i32_m1 = executor.constant -1 : i32

  func.call @test_ctpop_i32(%i32_0) : (i32) -> ()
  func.call @test_ctpop_i32(%i32_15) : (i32) -> ()
  func.call @test_ctpop_i32(%i32_m1) : (i32) -> ()

  %c0 = executor.constant 0 : i64
  return %c0 : i64
}

// CHECK: ctpop_i64(0) = 0
// CHECK: ctpop_i64(1) = 1
// CHECK: ctpop_i64(7) = 3
// CHECK: ctpop_i64(255) = 8
// CHECK: ctpop_i64(-1) = 64
// CHECK: ctpop_i32(0) = 0
// CHECK: ctpop_i32(15) = 4
// CHECK: ctpop_i32(-1) = 32
