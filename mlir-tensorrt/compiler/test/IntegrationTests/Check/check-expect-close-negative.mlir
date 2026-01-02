// Negative integration test for check.expect_close - verifies assertion fails for values outside ULP bounds
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   not mlir-tensorrt-runner -input-type=rtexe -features=core

// This test verifies that check.expect_close correctly fails when values
// differ by more than the allowed ULP range. The 'not' command inverts the exit code.
module @test_check_expect_close_negative attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // These values differ significantly - way more than 3 ULPs (default max)
  %a = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[1.0, 100.0]> : tensor<2xf32>  // 2.0 vs 100.0 - huge ULP difference!
  stablehlo.custom_call @check.expect_close(%a, %b) {has_side_effect = true}
    : (tensor<2xf32>, tensor<2xf32>) -> ()

  return
}

}
