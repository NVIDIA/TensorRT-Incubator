// Integration test for check.expect_close custom calls (ULP-based comparison)
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   mlir-tensorrt-runner -input-type=rtexe -features=core

// Test check.expect_close with ULP (Units in Last Place) based comparison
module @test_check_expect_close attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Test 1: Exact equality - 0 ULP difference - should pass
  %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  stablehlo.custom_call @check.expect_close(%a, %b) {has_side_effect = true}
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  // Test 2: With default ULP bounds (min=0, max=3) - exact match
  %c = stablehlo.constant dense<[1.5, 2.5, 3.5]> : tensor<3xf32>
  %d = stablehlo.constant dense<[1.5, 2.5, 3.5]> : tensor<3xf32>
  stablehlo.custom_call @check.expect_close(%c, %d) {has_side_effect = true}
    : (tensor<3xf32>, tensor<3xf32>) -> ()

  // Test 3: With explicit ULP bounds - should pass
  %e = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %f = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  stablehlo.custom_call @check.expect_close(%e, %f) {
    has_side_effect = true,
    min_ulp_difference = 0 : i64,
    max_ulp_difference = 10 : i64
  } : (tensor<2xf32>, tensor<2xf32>) -> ()

  // Test 4: f64 values with ULP check - exact match
  %g = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
  %h = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
  stablehlo.custom_call @check.expect_close(%g, %h) {has_side_effect = true}
    : (tensor<3xf64>, tensor<3xf64>) -> ()

  // Test 5: f16 values - exact match
  %i = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf16>
  %j = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf16>
  stablehlo.custom_call @check.expect_close(%i, %j) {has_side_effect = true}
    : (tensor<2xf16>, tensor<2xf16>) -> ()

  // Test 6: NaN values - both NaN should be considered equal (0 ULP)
  %nan1 = stablehlo.constant dense<0x7FC00000> : tensor<1xf32>  // NaN
  %nan2 = stablehlo.constant dense<0x7FC00000> : tensor<1xf32>  // NaN
  stablehlo.custom_call @check.expect_close(%nan1, %nan2) {has_side_effect = true}
    : (tensor<1xf32>, tensor<1xf32>) -> ()

  // Test 7: Verify computation result
  %x = stablehlo.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %y = stablehlo.constant dense<[3.0, 5.0]> : tensor<2xf32>
  %sum = stablehlo.add %x, %y : tensor<2xf32>
  %expected = stablehlo.constant dense<[5.0, 9.0]> : tensor<2xf32>
  stablehlo.custom_call @check.expect_close(%sum, %expected) {has_side_effect = true}
    : (tensor<2xf32>, tensor<2xf32>) -> ()

  // Test 8: 1D tensor with single element
  %s1 = stablehlo.constant dense<[42.0]> : tensor<1xf32>
  %s2 = stablehlo.constant dense<[42.0]> : tensor<1xf32>
  stablehlo.custom_call @check.expect_close(%s1, %s2) {has_side_effect = true}
    : (tensor<1xf32>, tensor<1xf32>) -> ()

  return
}

}
