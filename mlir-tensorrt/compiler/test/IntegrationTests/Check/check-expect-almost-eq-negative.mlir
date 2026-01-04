// Negative integration test for check.expect_almost_eq - verifies assertion fails for values outside tolerance
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   not mlir-tensorrt-runner -input-type=rtexe -features=core

// This test verifies that check.expect_almost_eq correctly fails when values
// differ by more than the tolerance. The 'not' command inverts the exit code.
module @test_check_expect_almost_eq_negative attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // These values differ by 1.0, which is way more than the default tolerance (1e-3)
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %b = stablehlo.constant dense<[1.0, 2.0, 4.0]> : tensor<3xf32>  // 3.0 vs 4.0 - too different!
  stablehlo.custom_call @check.expect_almost_eq(%a, %b) {has_side_effect = true}
    : (tensor<3xf32>, tensor<3xf32>) -> ()

  return
}

}
