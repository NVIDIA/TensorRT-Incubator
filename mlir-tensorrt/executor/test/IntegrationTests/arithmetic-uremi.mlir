// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline | \
// RUN: executor-translate -mlir-to-runtime-executable | \
// RUN: executor-runner -input-type=rtexe --features=core

//===----------------------------------------------------------------------===//
// Integration tests for executor.uremi
//===----------------------------------------------------------------------===//

module {
  func.func private @uremi_i64(%lhs: i64, %rhs: i64, %expected: i64) -> ()
      attributes {no_inline} {
    %r = executor.uremi %lhs, %rhs : i64
    %ok = executor.icmp <eq> %r, %expected : i64
    executor.assert %ok, "executor.uremi i64 failed"
    return
  }

  func.func private @uremi_i32(%lhs: i32, %rhs: i32, %expected: i32) -> ()
      attributes {no_inline} {
    %r = executor.uremi %lhs, %rhs : i32
    %ok = executor.icmp <eq> %r, %expected : i32
    executor.assert %ok, "executor.uremi i32 failed"
    return
  }

  func.func @main() -> i32 {
    // i64 cases.
    %c0_i64 = executor.constant 0 : i64
    %c1_i64 = executor.constant 1 : i64
    %c2_i64 = executor.constant 2 : i64
    %c3_i64 = executor.constant 3 : i64
    %c5_i64 = executor.constant 5 : i64
    %c7_i64 = executor.constant 7 : i64
    %c8_i64 = executor.constant 8 : i64
    %c15_i64 = executor.constant 15 : i64
    %c16_i64 = executor.constant 16 : i64
    %cn1_i64 = executor.constant -1 : i64
    %cn4_i64 = executor.constant -4 : i64
    %c123456789_i64 = executor.constant 123456789 : i64

    // Basic sanity.
    call @uremi_i64(%c5_i64, %c3_i64, %c2_i64) : (i64, i64, i64) -> ()
    call @uremi_i64(%c5_i64, %c7_i64, %c5_i64) : (i64, i64, i64) -> ()
    call @uremi_i64(%c0_i64, %c7_i64, %c0_i64) : (i64, i64, i64) -> ()

    // Power-of-two modulus.
    call @uremi_i64(%c123456789_i64, %c8_i64, %c5_i64) : (i64, i64, i64) -> ()

    // Unsigned semantics for negative bit-patterns.
    // -1 (all-ones) % 2 == 1.
    call @uremi_i64(%cn1_i64, %c2_i64, %c1_i64) : (i64, i64, i64) -> ()
    // -1 (all-ones) % 16 == 15.
    call @uremi_i64(%cn1_i64, %c16_i64, %c15_i64) : (i64, i64, i64) -> ()
    // (2^64 - 4) % 3 == 0.
    call @uremi_i64(%cn4_i64, %c3_i64, %c0_i64) : (i64, i64, i64) -> ()

    // i32 cases.
    %c0_i32 = executor.constant 0 : i32
    %c1_i32 = executor.constant 1 : i32
    %c2_i32 = executor.constant 2 : i32
    %c3_i32 = executor.constant 3 : i32
    %c5_i32 = executor.constant 5 : i32
    %c7_i32 = executor.constant 7 : i32
    %c8_i32 = executor.constant 8 : i32
    %c15_i32 = executor.constant 15 : i32
    %c16_i32 = executor.constant 16 : i32
    %cn1_i32 = executor.constant -1 : i32
    %cn4_i32 = executor.constant -4 : i32
    %c123456789_i32 = executor.constant 123456789 : i32

    call @uremi_i32(%c5_i32, %c3_i32, %c2_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%c5_i32, %c7_i32, %c5_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%c0_i32, %c7_i32, %c0_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%c123456789_i32, %c8_i32, %c5_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%cn1_i32, %c2_i32, %c1_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%cn1_i32, %c16_i32, %c15_i32) : (i32, i32, i32) -> ()
    call @uremi_i32(%cn4_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()

    return %c0_i32 : i32
  }
}
