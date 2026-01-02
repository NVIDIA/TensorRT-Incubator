// Negative integration test for check.expect_eq - verifies assertion fails for unequal values
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   not mlir-tensorrt-runner -input-type=rtexe -features=core

// This test verifies that check.expect_eq correctly fails when values are NOT equal.
// The 'not' command inverts the exit code, so this test passes if the runner fails.
module @test_check_expect_eq_negative attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // These values are NOT equal - the assertion should fail
  %a = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %b = stablehlo.constant dense<[[1, 2, 999], [4, 5, 6]]> : tensor<2x3xi32>  // Different value!
  stablehlo.custom_call @check.expect_eq(%a, %b) {has_side_effect = true}
    : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()

  return
}

}
