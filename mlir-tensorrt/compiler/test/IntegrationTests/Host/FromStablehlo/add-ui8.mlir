// RUN: mlir-tensorrt-opt %s -stablehlo-to-executable-pipeline="disable-tensorrt-extension entrypoint=" \
// RUN: | mlir-tensorrt-translate -mlir-to-runtime-executable -allow-unregistered-dialect -split-input-file \
// RUN: | mlir-tensorrt-runner -input-type=rtexe -features=core -split-input-file

!tensor_type = tensor<2xui8>
!compare_type = tensor<2xi1>

module @add_op_test_ui8 attributes {
  plan.cluster_kinds = [#plan.host_cluster<benefit = 1>],
  plan.memory_space = #plan.memory_space<host>
} {

func.func private @check_eq(%arg0: !tensor_type, %arg1: !tensor_type) attributes {no_inline} {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (!tensor_type, !tensor_type) -> !compare_type
  %c0 = stablehlo.constant dense<1> : tensor<i1>
  %2 = stablehlo.reduce (%0 init: %c0)
    applies stablehlo.and across dimensions = [0]
     : (!compare_type, tensor<i1>) -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  executor.print "check: %d"(%3 : i1)
  cf.assert %3, "check_eq failed"
  return
}

func.func private @compute(%arg0: !tensor_type, %arg1: !tensor_type) -> !tensor_type
   attributes {no_inline} {
  %0 = stablehlo.add %arg0, %arg1 : !tensor_type
  return %0 : !tensor_type
}

func.func @main() -> i32 {
  %0 = stablehlo.constant dense<[0, 16]> : tensor<2xui8>
  %1 = stablehlo.constant dense<[255, 16]> : tensor<2xui8>
  %2 = call @compute(%0, %1) : (!tensor_type, !tensor_type) -> !tensor_type
  %expected = stablehlo.constant dense<[255, 32]> : tensor<2xui8>
  call @check_eq(%2, %expected) : (!tensor_type, !tensor_type) -> ()

  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

}
