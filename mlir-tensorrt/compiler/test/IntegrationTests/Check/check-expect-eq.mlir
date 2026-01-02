// Integration test for check.expect_eq and check.expect_eq_const custom calls
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   mlir-tensorrt-runner -input-type=rtexe -features=core

// Test check.expect_eq with integer tensors (should pass - values are equal)
module @test_check_expect_eq_int attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Test 1: Integer equality check - should pass
  %a = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %b = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  stablehlo.custom_call @check.expect_eq(%a, %b) {has_side_effect = true}
    : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()

  // Test 2: Integer equality with 1D tensor - should pass
  %c = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi64>
  %d = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi64>
  stablehlo.custom_call @check.expect_eq(%c, %d) {has_side_effect = true}
    : (tensor<4xi64>, tensor<4xi64>) -> ()

  // Test 3: Boolean equality check - should pass
  %e = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %f = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  stablehlo.custom_call @check.expect_eq(%e, %f) {has_side_effect = true}
    : (tensor<3xi1>, tensor<3xi1>) -> ()

  // Test 4: check.expect_eq_const variant - should pass
  %g = stablehlo.constant dense<[100, 200, 300]> : tensor<3xi32>
  %h = stablehlo.constant dense<[100, 200, 300]> : tensor<3xi32>
  stablehlo.custom_call @check.expect_eq_const(%g, %h) {has_side_effect = true}
    : (tensor<3xi32>, tensor<3xi32>) -> ()

  // Test 5: Scalar tensor equality - should pass
  %i = stablehlo.constant dense<42> : tensor<i32>
  %j = stablehlo.constant dense<42> : tensor<i32>
  stablehlo.custom_call @check.expect_eq(%i, %j) {has_side_effect = true}
    : (tensor<i32>, tensor<i32>) -> ()

  // Test 6: Computation result equality - verify add operation
  %x = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %y = stablehlo.constant dense<[4, 5, 6]> : tensor<3xi32>
  %sum = stablehlo.add %x, %y : tensor<3xi32>
  %expected_sum = stablehlo.constant dense<[5, 7, 9]> : tensor<3xi32>
  stablehlo.custom_call @check.expect_eq(%sum, %expected_sum) {has_side_effect = true}
    : (tensor<3xi32>, tensor<3xi32>) -> ()

  return
}

}
