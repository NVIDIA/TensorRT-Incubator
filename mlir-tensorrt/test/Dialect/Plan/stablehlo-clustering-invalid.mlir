// RUN: mlir-tensorrt-opt -split-input-file \
// RUN:  -plan-segmentation-pipeline="disallow-host-tensors-in-tensorrt-clusters" -verify-diagnostics %s

func.func @reduce_i64_not_divisible_by_32(%arg0: tensor<116xi64>) -> tensor<i64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  // expected-error @below {{is invalid, post clustering}}
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [0] : (tensor<116xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// -----

func.func @reduce_i64_not_power_of_2(%arg0: tensor<24xi64>) -> tensor<i64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  // expected-error @below {{is invalid, post clustering}}
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [0] : (tensor<24xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// -----

func.func @reduce_i64_not_tail(%arg0: tensor<128x128xi64>) -> tensor<128xi64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  // expected-error @below {{is invalid, post clustering}}
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [0] : (tensor<128x128xi64>, tensor<i64>) -> tensor<128xi64>
  return %0 : tensor<128xi64>
}

// -----

func.func @reduce_i64_multiple_dims(%arg0: tensor<128x128x128xi64>) -> tensor<128xi64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  // expected-error @below {{is invalid, post clustering}}
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<128x128x128xi64>, tensor<i64>) -> tensor<128xi64>
  return %0 : tensor<128xi64>
}

// -----

func.func @reduce_i64_not_all_static(%arg0: tensor<?x128xi64>) -> tensor<128xi64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  // expected-error @below {{is invalid, post clustering}}
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [1] : (tensor<?x128xi64>, tensor<i64>) -> tensor<128xi64>
  return %0 : tensor<128xi64>
}
