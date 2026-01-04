// Integration test for check.eq custom call (predicate operation)
// RUN:   mlir-tensorrt-compiler %s --disable-all-extensions --entrypoint= -o - | \
// RUN:   mlir-tensorrt-runner -input-type=rtexe -features=core

// Test check.eq which is a predicate custom call (returns tensor<i1> result)
module @test_check_eq attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func @main() {
  // Test 1: Integer equality - should return true
  %a = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %b = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %eq1 = stablehlo.custom_call @check.eq(%a, %b) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
  %res1 = tensor.extract %eq1[] : tensor<i1>
  cf.assert %res1, "check.eq test 1 failed: integers should be equal"

  // Test 2: Float equality with tolerance (uses almost-equal semantics)
  %c = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %d = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %eq2 = stablehlo.custom_call @check.eq(%c, %d) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
  %res2 = tensor.extract %eq2[] : tensor<i1>
  cf.assert %res2, "check.eq test 2 failed: floats should be equal"

  // Test 3: Scalar tensors
  %e = stablehlo.constant dense<42> : tensor<i32>
  %f = stablehlo.constant dense<42> : tensor<i32>
  %eq3 = stablehlo.custom_call @check.eq(%e, %f) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %res3 = tensor.extract %eq3[] : tensor<i1>
  cf.assert %res3, "check.eq test 3 failed: scalar integers should be equal"

  // Test 4: Boolean values
  %g = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %h = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %eq4 = stablehlo.custom_call @check.eq(%g, %h) : (tensor<3xi1>, tensor<3xi1>) -> tensor<i1>
  %res4 = tensor.extract %eq4[] : tensor<i1>
  cf.assert %res4, "check.eq test 4 failed: booleans should be equal"

  // Test 5: Verify computation result
  %x = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %y = stablehlo.constant dense<[10, 20, 30]> : tensor<3xi32>
  %sum = stablehlo.add %x, %y : tensor<3xi32>
  %expected = stablehlo.constant dense<[11, 22, 33]> : tensor<3xi32>
  %eq5 = stablehlo.custom_call @check.eq(%sum, %expected) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
  %res5 = tensor.extract %eq5[] : tensor<i1>
  cf.assert %res5, "check.eq test 5 failed: sum should equal expected"

  // Test 6: 2D tensor equality
  %m1 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %m2 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %eq6 = stablehlo.custom_call @check.eq(%m1, %m2) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<i1>
  %res6 = tensor.extract %eq6[] : tensor<i1>
  cf.assert %res6, "check.eq test 6 failed: 2D tensors should be equal"

  // Test 7: Float with small tolerance (within almost-equal threshold)
  %f1 = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %f2 = stablehlo.constant dense<[1.00005, 2.00005, 3.00005]> : tensor<3xf32>
  %eq7 = stablehlo.custom_call @check.eq(%f1, %f2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
  %res7 = tensor.extract %eq7[] : tensor<i1>
  cf.assert %res7, "check.eq test 7 failed: floats within tolerance should be equal"

  // ========== NEGATIVE TESTS ==========
  // Test that check.eq returns false for unequal values
  %true_const = arith.constant true

  // Negative Test 1: Integer inequality - should return false
  %neg_a = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %neg_b = stablehlo.constant dense<[1, 2, 999]> : tensor<3xi32>  // Different!
  %neg_eq1 = stablehlo.custom_call @check.eq(%neg_a, %neg_b) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
  %neg_res1 = tensor.extract %neg_eq1[] : tensor<i1>
  %neg_res1_inverted = arith.xori %neg_res1, %true_const : i1
  cf.assert %neg_res1_inverted, "negative test 1 failed: unequal integers should return false"

  // Negative Test 2: Completely different values - should return false
  %neg_c = stablehlo.constant dense<[100, 200, 300]> : tensor<3xi32>
  %neg_d = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %neg_eq2 = stablehlo.custom_call @check.eq(%neg_c, %neg_d) : (tensor<3xi32>, tensor<3xi32>) -> tensor<i1>
  %neg_res2 = tensor.extract %neg_eq2[] : tensor<i1>
  %neg_res2_inverted = arith.xori %neg_res2, %true_const : i1
  cf.assert %neg_res2_inverted, "negative test 2 failed: different values should return false"

  // Negative Test 3: Boolean inequality - should return false
  %neg_e = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %neg_f = stablehlo.constant dense<[true, true, true]> : tensor<3xi1>  // Different!
  %neg_eq3 = stablehlo.custom_call @check.eq(%neg_e, %neg_f) : (tensor<3xi1>, tensor<3xi1>) -> tensor<i1>
  %neg_res3 = tensor.extract %neg_eq3[] : tensor<i1>
  %neg_res3_inverted = arith.xori %neg_res3, %true_const : i1
  cf.assert %neg_res3_inverted, "negative test 3 failed: unequal booleans should return false"

  // Negative Test 4: Float inequality - should return false (large difference)
  %neg_g = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %neg_h = stablehlo.constant dense<[1.0, 999.0]> : tensor<2xf32>  // Large difference!
  %neg_eq4 = stablehlo.custom_call @check.eq(%neg_g, %neg_h) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
  %neg_res4 = tensor.extract %neg_eq4[] : tensor<i1>
  %neg_res4_inverted = arith.xori %neg_res4, %true_const : i1
  cf.assert %neg_res4_inverted, "negative test 4 failed: unequal floats should return false"

  // Negative Test 5: 2D tensor inequality - should return false
  %neg_m1 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %neg_m2 = stablehlo.constant dense<[[1, 2], [3, 999]]> : tensor<2x2xi32>  // Different!
  %neg_eq5 = stablehlo.custom_call @check.eq(%neg_m1, %neg_m2) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<i1>
  %neg_res5 = tensor.extract %neg_eq5[] : tensor<i1>
  %neg_res5_inverted = arith.xori %neg_res5, %true_const : i1
  cf.assert %neg_res5_inverted, "negative test 5 failed: unequal 2D tensors should return false"

  // Negative Test 6: Float outside tolerance - should return false
  %neg_f1 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %neg_f2 = stablehlo.constant dense<[1.001, 2.0]> : tensor<2xf32>  // Just outside 1e-4 tolerance
  %neg_eq6 = stablehlo.custom_call @check.eq(%neg_f1, %neg_f2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
  %neg_res6 = tensor.extract %neg_eq6[] : tensor<i1>
  %neg_res6_inverted = arith.xori %neg_res6, %true_const : i1
  cf.assert %neg_res6_inverted, "negative test 6 failed: floats outside tolerance should return false"

  return
}

}
