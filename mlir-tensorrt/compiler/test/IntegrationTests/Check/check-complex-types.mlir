// Integration test for check.* operations with complex types
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   mlir-tensorrt-runner -input-type=rtexe -features=core

// Test check operations with complex<f32> and complex<f64> tensors
module @test_check_complex_types attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Test 1: check.expect_eq with complex<f32>
  %a = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>
  %b = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>
  stablehlo.custom_call @check.expect_eq(%a, %b) {has_side_effect = true}
    : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()

  // Test 2: check.expect_almost_eq with complex<f32>
  %c = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>
  %d = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f32>>
  stablehlo.custom_call @check.expect_almost_eq(%c, %d) {has_side_effect = true}
    : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()

  // Test 3: check.expect_close with complex<f32>
  %e = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %f = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  stablehlo.custom_call @check.expect_close(%e, %f) {has_side_effect = true}
    : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()

  // Test 4: check.expect_eq with complex<f64>
  %g = stablehlo.constant dense<[(1.0, 2.0)]> : tensor<1xcomplex<f64>>
  %h = stablehlo.constant dense<[(1.0, 2.0)]> : tensor<1xcomplex<f64>>
  stablehlo.custom_call @check.expect_eq(%g, %h) {has_side_effect = true}
    : (tensor<1xcomplex<f64>>, tensor<1xcomplex<f64>>) -> ()

  // Test 5: check.expect_close with complex<f64>
  %i = stablehlo.constant dense<[(5.0, 6.0), (7.0, 8.0)]> : tensor<2xcomplex<f64>>
  %j = stablehlo.constant dense<[(5.0, 6.0), (7.0, 8.0)]> : tensor<2xcomplex<f64>>
  stablehlo.custom_call @check.expect_close(%i, %j) {has_side_effect = true}
    : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()

  // Test 6: Complex addition verification
  %x = stablehlo.constant dense<[(1.0, 1.0), (2.0, 2.0)]> : tensor<2xcomplex<f32>>
  %y = stablehlo.constant dense<[(1.0, 1.0), (2.0, 2.0)]> : tensor<2xcomplex<f32>>
  %sum = stablehlo.add %x, %y : tensor<2xcomplex<f32>>
  %expected = stablehlo.constant dense<[(2.0, 2.0), (4.0, 4.0)]> : tensor<2xcomplex<f32>>
  stablehlo.custom_call @check.expect_eq(%sum, %expected) {has_side_effect = true}
    : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()

  // Test 7: 2D complex tensor
  %m = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 1.0)], [(0.0, -1.0), (1.0, 0.0)]]>
    : tensor<2x2xcomplex<f32>>
  %n = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 1.0)], [(0.0, -1.0), (1.0, 0.0)]]>
    : tensor<2x2xcomplex<f32>>
  stablehlo.custom_call @check.expect_eq(%m, %n) {has_side_effect = true}
    : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>) -> ()

  return
}

}
