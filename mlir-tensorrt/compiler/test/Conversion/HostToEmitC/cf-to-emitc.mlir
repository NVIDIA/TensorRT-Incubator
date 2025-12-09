// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc="artifacts-dir=%t" %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

// Test ControlFlow Assert conversion to EmitC

func.func @test_assert_i1_with_message(%arg0: i1) {
  cf.assert %arg0, "assertion failed"
  return
}

// CPP-LABEL: void test_assert_i1_with_message(bool v1) {
// CPP:   assert(v1 && "assertion failed");

// -----

func.func @test_assert_i32(%arg0: i32) {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi ne, %arg0, %c0 : i32
  cf.assert %cmp, "value must be non-zero"
  return
}

// CPP-LABEL: void test_assert_i32(int32_t v1) {
// CPP: int32_t v2 = 0;
// CPP: bool v3 = v1 != v2;
// CPP: assert(v3 && "value must be non-zero");

// -----

func.func @test_assert_string_escaping(%arg0: i1) {
  cf.assert %arg0, "message with \"quotes\" and \\backslashes"
  return
}

// CPP-LABEL: void test_assert_string_escaping(bool v1) {
// CPP: assert(v1 && "message with \"quotes\" and \\backslashes");

// -----

func.func @test_assert_curly_brace_escaping(%arg0: i1) {
  cf.assert %arg0, "message with {curly} braces"
  return
}

// CPP-LABEL: void test_assert_curly_brace_escaping(bool v1) {
// CPP: assert(v1 && "message with {curly} braces");
