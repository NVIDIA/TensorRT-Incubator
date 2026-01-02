// Integration test for check.expect_almost_eq and check.expect_almost_eq_const custom calls
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   mlir-tensorrt-runner -input-type=rtexe -features=core

// Test check.expect_almost_eq with float tensors
module @test_check_expect_almost_eq attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Test 1: Exact equality for floats - should pass
  %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  stablehlo.custom_call @check.expect_almost_eq(%a, %b) {has_side_effect = true}
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  // Test 2: Values within default tolerance (1e-3) - should pass
  // The default tolerance for check.expect_almost_eq is 1e-3
  %c = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %d = stablehlo.constant dense<[1.0005, 2.0005, 3.0005]> : tensor<3xf32>
  stablehlo.custom_call @check.expect_almost_eq(%c, %d) {has_side_effect = true}
    : (tensor<3xf32>, tensor<3xf32>) -> ()

  // Test 3: check.expect_almost_eq_const with default tolerance (1e-4) - should pass
  %e = stablehlo.constant dense<[10.0, 20.0]> : tensor<2xf32>
  %f = stablehlo.constant dense<[10.00005, 20.00005]> : tensor<2xf32>
  stablehlo.custom_call @check.expect_almost_eq_const(%e, %f) {has_side_effect = true}
    : (tensor<2xf32>, tensor<2xf32>) -> ()

  // Test 4: With explicit tolerance attribute - should pass
  %g = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %h = stablehlo.constant dense<[1.05, 2.05, 3.05]> : tensor<3xf32>
  stablehlo.custom_call @check.expect_almost_eq_const(%g, %h) {
    has_side_effect = true,
    tolerance = 0.1 : f64
  } : (tensor<3xf32>, tensor<3xf32>) -> ()

  // Test 5: f64 values - should pass
  %i = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %j = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  stablehlo.custom_call @check.expect_almost_eq(%i, %j) {has_side_effect = true}
    : (tensor<2xf64>, tensor<2xf64>) -> ()

  // Test 6: Verify computation result with tolerance
  %x = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %y = stablehlo.constant dense<[0.1, 0.2, 0.3]> : tensor<3xf32>
  %sum = stablehlo.add %x, %y : tensor<3xf32>
  %expected = stablehlo.constant dense<[1.1, 2.2, 3.3]> : tensor<3xf32>
  stablehlo.custom_call @check.expect_almost_eq(%sum, %expected) {has_side_effect = true}
    : (tensor<3xf32>, tensor<3xf32>) -> ()

  // Test 7: NaN equality (both NaN should be considered equal)
  %nan1 = stablehlo.constant dense<0x7FC00000> : tensor<1xf32>  // NaN
  %nan2 = stablehlo.constant dense<0x7FC00000> : tensor<1xf32>  // NaN
  stablehlo.custom_call @check.expect_almost_eq(%nan1, %nan2) {has_side_effect = true}
    : (tensor<1xf32>, tensor<1xf32>) -> ()

  return
}

}
