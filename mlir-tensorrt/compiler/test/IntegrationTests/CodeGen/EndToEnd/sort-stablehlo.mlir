// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-compiler %s -input=stablehlo -opts="disable-tensorrt-extension backends=#plan.kernel_backend<benefit=2>,#plan.host_backend<benefit=1>" -o - \
// RUN: | mlir-tensorrt-runner -input-type=rtexe -features=core,cuda

func.func private @sort_i32(%arg0: tensor<128xi32>) -> tensor<128xi32> attributes {no_inline} {
  %sorted = "stablehlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %predicate = stablehlo.compare LT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64} : (tensor<128xi32>) -> tensor<128xi32>
  return %sorted : tensor<128xi32>
}

func.func private @sort_kv_i32_f32(%arg0: tensor<128xi32>, %arg1: tensor<128xf32>) -> (tensor<128xi32>, tensor<128xf32>) attributes {no_inline} {
  %sorted0, %sorted1 = "stablehlo.sort"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
    %predicate = stablehlo.compare LT, %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %predicate : tensor<i1>
  }) {dimension = 0 : i64} : (tensor<128xi32>, tensor<128xf32>) -> (tensor<128xi32>, tensor<128xf32>)
  return %sorted0, %sorted1 : tensor<128xi32>, tensor<128xf32>
}

func.func private @check_eq_i32(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i1 attributes {no_inline} {
  %equal = stablehlo.compare EQ, %arg0, %arg1 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
  %c1 = stablehlo.constant dense<1> : tensor<i1>
  %reduce_and = stablehlo.reduce (%equal init: %c1) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
  %val = tensor.extract %reduce_and[] : tensor<i1>
  return %val : i1
}

func.func private @check_eq_f32(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> i1 attributes {no_inline} {
  %equal = stablehlo.compare EQ, %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
  %c1 = stablehlo.constant dense<1> : tensor<i1>
  %reduce_and = stablehlo.reduce (%equal init: %c1) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
  %val = tensor.extract %reduce_and[] : tensor<i1>
  return %val : i1
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c10 = arith.constant 10 : index

  %0 = stablehlo.iota dim = 0 : tensor<128xi32>
  %1 = stablehlo.reverse %0, dims = [0] : tensor<128xi32>
  %2 = call @sort_i32(%1) : (tensor<128xi32>) -> tensor<128xi32>
  %3 = call @check_eq_i32(%0, %2) : (tensor<128xi32>, tensor<128xi32>) -> i1
  executor.assert %3, "sort failed"


  %vals = stablehlo.iota dim = 0 : tensor<128xf32>
  %vals_rev = stablehlo.reverse %vals, dims = [0] : tensor<128xf32>
  %sorted0, %sorted1 = call @sort_kv_i32_f32(%1, %vals_rev) : (tensor<128xi32>, tensor<128xf32>) -> (tensor<128xi32>, tensor<128xf32>)
  %4 = call @check_eq_i32(%0, %sorted0) : (tensor<128xi32>, tensor<128xi32>) -> i1
  executor.assert %4, "sort kv failed, keys not correct"
  %5 = call @check_eq_f32(%vals, %sorted1) : (tensor<128xf32>, tensor<128xf32>) -> i1
  executor.assert %5, "sort kv failed, values not correct"

  executor.print "all tests passed"()

  return %c0_i32 : i32
}
