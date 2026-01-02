// Negative integration test for check.expect_eq with complex types
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   not mlir-tensorrt-runner -input-type=rtexe -features=core

// This test verifies that check.expect_eq correctly fails for unequal complex values.
// The 'not' command inverts the exit code.
module @test_check_complex_negative attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Complex values with different imaginary parts - should fail
  %a = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>
  %b = stablehlo.constant dense<[(1.0, 999.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>  // Different imaginary!
  stablehlo.custom_call @check.expect_eq(%a, %b) {has_side_effect = true}
    : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()

  return
}

}
