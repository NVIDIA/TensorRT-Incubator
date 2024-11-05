// RUN: mlir-tensorrt-opt %s -tensorrt-stablehlo-input-preprocessing -stablehlo-aggressive-simplification -split-input-file | FileCheck %s

func.func @trivial_right_shift(%arg0: tensor<i32>) -> tensor<i32> {
  %c32 = stablehlo.constant dense<32> : tensor<i32>
  %0 = stablehlo.shift_right_logical %arg0, %c32 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @trivial_right_shift
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i32>
//       CHECK:     return %[[v0]] : tensor<i32>

// -----

func.func @nontrivial_right_shift(%arg0: tensor<i32>) -> tensor<i32> {
  %c16 = stablehlo.constant dense<16> : tensor<i32>
  %0 = stablehlo.shift_right_logical %arg0, %c16 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @nontrivial_right_shift
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<16> : tensor<i32>
//       CHECK:     %[[v1:.+]] = stablehlo.shift_right_logical %[[arg0]], %[[v0]] : tensor<i32>
//       CHECK:     return %[[v1]] : tensor<i32>

// -----

func.func @jax_random_seed(%arg0: tensor<i32>) -> (tensor<2xi32>) {
  %0 = stablehlo.constant dense<32> : tensor<i32>
  %1 = stablehlo.shift_right_logical %arg0, %0 : tensor<i32>
  %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.constant dense<4294967295> : tensor<i32>
  %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i32>
  %6 = stablehlo.and %arg0, %5 : tensor<i32>
  %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i32>
  %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
  %9 = "stablehlo.concatenate"(%3, %8) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  return %9 : tensor<2xi32>
}

// CHECK-LABEL: @jax_random_seed
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<2xi32> {
//       CHECK: %[[v0:.+]] = stablehlo.constant dense<-1> : tensor<i32>
//       CHECK: %[[v1:.+]] = stablehlo.constant dense<0> : tensor<1xi32>
//       CHECK: %[[v2:.+]] = stablehlo.and %[[arg0]], %[[v0]] : tensor<i32>
//       CHECK: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<i32>) -> tensor<1xi32>
//       CHECK: %[[v4:.+]] = stablehlo.concatenate %[[v1]], %[[v3]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
//       CHECK: return %[[v4]] : tensor<2xi32>

// -----

func.func @erf_inv(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = chlo.erf_inv %arg0 : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @erf_inv
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>) -> tensor<4xf32> {
//   CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<0x7F800000> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_1:.+]] = stablehlo.constant dense<2.83297682> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_2:.+]] = stablehlo.constant dense<1.50140941> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_3:.+]] = stablehlo.constant dense<1.00167406> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_4:.+]] = stablehlo.constant dense<0.246640727> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_5:.+]] = stablehlo.constant dense<0.00943887047> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_6:.+]] = stablehlo.constant dense<-0.00417768164> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_7:.+]] = stablehlo.constant dense<-0.0076224613> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_8:.+]] = stablehlo.constant dense<-0.00125372503> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_9:.+]] = stablehlo.constant dense<0.00573950773> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_10:.+]] = stablehlo.constant dense<2.1858087E-4> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_11:.+]] = stablehlo.constant dense<-0.00367342844> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_12:.+]] = stablehlo.constant dense<-4.39150654E-6> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_13:.+]] = stablehlo.constant dense<0.00134934322> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_14:.+]] = stablehlo.constant dense<-3.5233877E-6> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_15:.+]] = stablehlo.constant dense<1.00950558E-4> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_16:.+]] = stablehlo.constant dense<3.43273939E-7> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_17:.+]] = stablehlo.constant dense<-2.00214257E-4> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_18:.+]] = stablehlo.constant dense<2.81022636E-8> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_19:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_20:.+]] = stablehlo.constant dense<2.500000e+00> : tensor<4xf32>
//   CHECK-DAG:     %[[cst_21:.+]] = stablehlo.constant dense<5.000000e+00> : tensor<4xf32>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.negate %[[arg0]] : tensor<4xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.multiply %[[arg0]], %[[v0]] : tensor<4xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.log_plus_one %[[v1]] : tensor<4xf32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.negate %[[v2]] : tensor<4xf32>
//   CHECK-DAG:     %[[v4:.+]] = stablehlo.compare  LT, %[[v3]], %[[cst_21]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
//   CHECK-DAG:     %[[v5:.+]] = stablehlo.subtract %[[v3]], %[[cst_20]] : tensor<4xf32>
//   CHECK-DAG:     %[[v6:.+]] = stablehlo.sqrt %[[v3]] : tensor<4xf32>
//   CHECK-DAG:     %[[v7:.+]] = stablehlo.subtract %[[v6]], %[[cst_19]] : tensor<4xf32>
//   CHECK-DAG:     %[[v8:.+]] = stablehlo.select %[[v4]], %[[v5]], %[[v7]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v9:.+]] = stablehlo.select %[[v4]], %[[cst_18]], %[[cst_17]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v10:.+]] = stablehlo.select %[[v4]], %[[cst_16]], %[[cst_15]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v11:.+]] = stablehlo.multiply %[[v9]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v12:.+]] = stablehlo.add %[[v10]], %[[v11]] : tensor<4xf32>
//   CHECK-DAG:     %[[v13:.+]] = stablehlo.select %[[v4]], %[[cst_14]], %[[cst_13]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v14:.+]] = stablehlo.multiply %[[v12]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v15:.+]] = stablehlo.add %[[v13]], %[[v14]] : tensor<4xf32>
//   CHECK-DAG:     %[[v16:.+]] = stablehlo.select %[[v4]], %[[cst_12]], %[[cst_11]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v17:.+]] = stablehlo.multiply %[[v15]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v18:.+]] = stablehlo.add %[[v16]], %[[v17]] : tensor<4xf32>
//   CHECK-DAG:     %[[v19:.+]] = stablehlo.select %[[v4]], %[[cst_10]], %[[cst_9]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v20:.+]] = stablehlo.multiply %[[v18]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v21:.+]] = stablehlo.add %[[v19]], %[[v20]] : tensor<4xf32>
//   CHECK-DAG:     %[[v22:.+]] = stablehlo.select %[[v4]], %[[cst_8]], %[[cst_7]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v23:.+]] = stablehlo.multiply %[[v21]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v24:.+]] = stablehlo.add %[[v22]], %[[v23]] : tensor<4xf32>
//   CHECK-DAG:     %[[v25:.+]] = stablehlo.select %[[v4]], %[[cst_6]], %[[cst_5]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v26:.+]] = stablehlo.multiply %[[v24]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v27:.+]] = stablehlo.add %[[v25]], %[[v26]] : tensor<4xf32>
//   CHECK-DAG:     %[[v28:.+]] = stablehlo.select %[[v4]], %[[cst_4]], %[[cst_3]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v29:.+]] = stablehlo.multiply %[[v27]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v30:.+]] = stablehlo.add %[[v28]], %[[v29]] : tensor<4xf32>
//   CHECK-DAG:     %[[v31:.+]] = stablehlo.select %[[v4]], %[[cst_2]], %[[cst_1]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     %[[v32:.+]] = stablehlo.multiply %[[v30]], %[[v8]] : tensor<4xf32>
//   CHECK-DAG:     %[[v33:.+]] = stablehlo.add %[[v31]], %[[v32]] : tensor<4xf32>
//   CHECK-DAG:     %[[v34:.+]] = stablehlo.multiply %[[v33]], %[[arg0]] : tensor<4xf32>
//   CHECK-DAG:     %[[v35:.+]] = stablehlo.abs %[[arg0]] : tensor<4xf32>
//   CHECK-DAG:     %[[v36:.+]] = stablehlo.compare  EQ, %[[v35]], %[[cst_0]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
//   CHECK-DAG:     %[[v37:.+]] = stablehlo.multiply %[[arg0]], %[[cst]] : tensor<4xf32>
//   CHECK-DAG:     %[[v38:.+]] = stablehlo.select %[[v36]], %[[v37]], %[[v34]] : tensor<4xi1>, tensor<4xf32>
//   CHECK-DAG:     return %[[v38]] : tensor<4xf32>