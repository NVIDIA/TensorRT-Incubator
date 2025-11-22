// RUN: mlir-tensorrt-compiler %s -opts="disable-all-extensions entrypoint=" -o - | \
// RUN: mlir-tensorrt-runner -input-type=rtexe -features=core -split-input-file

!tensor_type = tensor<2xcomplex<f64>>
!compare_type = tensor<2xi1>

module @add_op_test_c128 attributes {
  plan.backends = [#plan.host_backend<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func private @check_almost_eq(%arg0: !tensor_type, %arg1: !tensor_type) attributes {no_inline} {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (!tensor_type, !tensor_type) -> !compare_type
  %c0 = stablehlo.constant dense<1> : tensor<i1>
  %2 = stablehlo.reduce (%0 init: %c0)
    applies stablehlo.and across dimensions = [0]
     : (!compare_type, tensor<i1>) -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  cf.assert %3, "check_almost_eq failed"
  return
}

func.func private @compute(%arg0: !tensor_type, %arg1: !tensor_type) -> !tensor_type
   attributes {no_inline} {
  %0 = stablehlo.add %arg0, %arg1 : !tensor_type
  return %0 : !tensor_type
}

func.func @main() -> i32 {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %2 = call @compute(%0, %1) : (!tensor_type, !tensor_type) -> !tensor_type
  %expected = stablehlo.constant dense<[(3.000000e+00, 5.000000e+00), (1.500000e+01, 1.100000e+01)]> : tensor<2xcomplex<f64>>
  call @check_almost_eq(%2, %expected) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()

  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

}
