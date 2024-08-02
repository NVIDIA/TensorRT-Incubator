// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-scalar-to-arith | FileCheck %s

func.func @test_abs(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.abs"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_abs
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK: %[[v_0:.+]] = arith.cmpi sge, %[[extracted]], %[[c0_i32]] : i32
//       CHECK: %[[v_1:.+]] = arith.subi %[[c0_i32]], %[[extracted]] : i32
//       CHECK: %[[v_2:.+]] = arith.select %[[v_0]], %[[extracted]], %[[v_1]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_2]] : tensor<i32>

// -----

func.func @test_add(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_add
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.addi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_and(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.and"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_and
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.andi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_atan2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_atan2
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.atan2 %[[extracted]], %[[extracted_0]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_bitcast_f32_i32(%arg0: tensor<f32>) -> tensor<i32> {
  %1 = "stablehlo.bitcast_convert"(%arg0) : (tensor<f32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_bitcast_f32_i32
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = arith.bitcast %[[extracted]] : f32 to i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_cubic_root(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.cbrt"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_cubic_root
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.cbrt %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_ceil(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.ceil"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_ceil
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.ceil %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_clamp(%min: tensor<f32>, %arg0: tensor<f32>, %max: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.clamp"(%min, %arg0, %max) : (tensor<f32>, tensor<f32>, tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_clamp
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<f32>
//       CHECK: %[[extracted_1:.+]] = tensor.extract %[[arg2:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = arith.maximumf %[[extracted]], %[[extracted_0]] : f32
//       CHECK: %[[v_1:.+]] = arith.minimumf %[[v_0]], %[[extracted_1]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_1]] : tensor<f32>

// -----

func.func @test_clz(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_clz
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = math.ctlz %[[extracted]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_compare(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i1> {
  %1 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1: tensor<i1>
}
// CHECK-LABEL: test_compare
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.cmpi eq, %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i1>

// -----

func.func @test_cosine(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.cosine"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_cosine
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.cos %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_div(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_div
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = arith.divf %[[extracted]], %[[extracted_0]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_exp(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.exponential"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_exp
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.exp %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_exp_m1(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_exp_m1
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.expm1 %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_floor(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.floor"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_floor
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.floor %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_is_finite(%arg0: tensor<f32>) -> tensor<i1> {
  %1 = "stablehlo.is_finite"(%arg0) : (tensor<f32>) ->  tensor<i1>
  return %1: tensor<i1>
}
// CHECK-LABEL: test_is_finite
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//   CHECK-DAG: %[[cst:.+]] = arith.constant 0x7F800000 : f32
//       CHECK: %[[v_0:.+]] = math.absf %[[extracted]] : f32
//       CHECK: %[[v_1:.+]] = arith.cmpf one, %[[v_0]], %[[cst]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_1]] : tensor<i1>

// -----

func.func @test_log(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.log"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_log
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.log %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_log_1p(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.log_plus_one"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_log_1p
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.log1p %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_logistic(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.logistic"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_logistic
//       CHECK: %[[cst:.+]] = arith.constant 1.000000e+00 : f32
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = arith.negf %[[extracted]] : f32
//       CHECK: %[[v_1:.+]] = math.exp %[[v_0]] : f32
//       CHECK: %[[v_2:.+]] = arith.addf %[[v_1]], %[[cst]] : f32
//       CHECK: %[[v_3:.+]] = arith.divf %[[cst]], %[[v_2]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_3]] : tensor<f32>

// -----

func.func @test_max(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_max
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.maxsi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_min(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_min
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.minsi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_mul(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_mul
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.muli %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_negate(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.negate"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_negate
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK: %[[v_0:.+]] = arith.subi %[[c0_i32]], %[[extracted]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_not(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.not"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_not
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c_1_i32:.+]] = arith.constant -1 : i32
//       CHECK: %[[v_0:.+]] = arith.xori %[[extracted]], %[[c_1_i32]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_or(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.or"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_or
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.ori %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_pop_count(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.popcnt"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_pop_count
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = math.ctpop %[[extracted]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_pow(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.power"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_pow
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c_1_i32:.+]] = arith.constant -1 : i32
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG: %[[c2_i32:.+]] = arith.constant 2 : i32
//   CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[c6:.+]] = arith.constant 6 : index
//       CHECK: %[[v_0_3:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c6]] step %[[c1]] iter_args(%arg3 = %[[c1_i32]], %[[arg4:.+]] = %[[extracted]], %[[arg5:.+]] = %[[extracted_0]]) -> (i32, i32, i32) {
//       CHECK: %[[v_10:.+]] = arith.andi %[[arg5]], %[[c1_i32]] : i32
//       CHECK: %[[v_11:.+]] = arith.cmpi eq, %[[v_10]], %[[c1_i32]] : i32
//       CHECK: %[[v_12:.+]] = arith.muli %[[arg3:.+]], %[[arg4]] : i32
//       CHECK: %[[v_13:.+]] = arith.select %[[v_11]], %[[v_12]], %[[arg3]] : i32
//       CHECK: %[[v_14:.+]] = arith.muli %[[arg4]], %[[arg4]] : i32
//       CHECK: %[[v_15:.+]] = arith.shrui %[[arg5]], %[[c1_i32]] : i32
//       CHECK: scf.yield %[[v_13]], %[[v_14]], %[[v_15]] : i32, i32, i32
//       CHECK: }
//       CHECK: %[[v_1:.+]] = arith.remsi %[[extracted_0]], %[[c2_i32]] : i32
//       CHECK: %[[v_2:.+]] = arith.cmpi eq, %[[v_1]], %[[c0_i32]] : i32
//       CHECK: %[[v_3:.+]] = arith.cmpi slt, %[[extracted_0]], %[[c0_i32]] : i32
//       CHECK: %[[v_4:.+]] = arith.cmpi eq, %[[extracted]], %[[c1_i32]] : i32
//       CHECK: %[[v_5:.+]] = arith.cmpi eq, %[[extracted]], %[[c_1_i32]] : i32
//       CHECK: %[[v_6:.+]] = arith.select %[[v_4]], %[[c1_i32]], %[[c0_i32]] : i32
//       CHECK: %[[v_7:.+]] = arith.select %[[v_2]], %[[c1_i32]], %[[c_1_i32]] : i32
//       CHECK: %[[v_8:.+]] = arith.select %[[v_5]], %[[v_7]], %[[v_6]] : i32
//       CHECK: %[[v_9:.+]] = arith.select %[[v_3]], %[[v_8]], %[[v_0_0:.+]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_9]] : tensor<i32>

// -----

func.func @test_rem(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.remainder"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_rem
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG: %[[c_1_i32:.+]] = arith.constant -1 : i32
//   CHECK-DAG: %[[c_2147483648_i32:.+]] = arith.constant -2147483648 : i32
//       CHECK: %[[v_0:.+]] = arith.cmpi eq, %[[extracted_0]], %[[c0_i32]] : i32
//       CHECK: %[[v_1:.+]] = arith.cmpi eq, %[[extracted]], %[[c_2147483648_i32]] : i32
//       CHECK: %[[v_2:.+]] = arith.cmpi eq, %[[extracted_0]], %[[c_1_i32]] : i32
//       CHECK: %[[v_3:.+]] = arith.andi %[[v_1]], %[[v_2]] : i1
//       CHECK: %[[v_4:.+]] = arith.ori %[[v_0]], %[[v_3]] : i1
//       CHECK: %[[v_5:.+]] = arith.select %[[v_4]], %[[c1_i32]], %[[extracted_0]] : i32
//       CHECK: %[[v_6:.+]] = arith.remsi %[[extracted]], %[[v_5]] : i32
//       CHECK: %[[v_7:.+]] = arith.select %[[v_3]], %[[c0_i32]], %[[v_6]] : i32
//       CHECK: %[[v_8:.+]] = arith.select %[[v_0]], %[[extracted]], %[[v_7]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_8]] : tensor<i32>

// -----

func.func @test_round_nearest_afz(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.round_nearest_afz"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.round %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>


// -----

func.func @test_round_nearest_even(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.round_nearest_even"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_round_nearest_even
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.roundeven %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_rsqrt(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.rsqrt"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_rsqrt
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.rsqrt %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_select(%cond: tensor<i1>, %lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.select" (%cond, %lhs, %rhs) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i1>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[extracted_1:.+]] = tensor.extract %[[arg2:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.select %[[extracted]], %[[extracted_0]], %[[extracted_1]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_tensor4xi32_select(%cond: tensor<4xi1>, %lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
  %1 = "stablehlo.select" (%cond, %lhs, %rhs) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: @test_tensor4xi32_select
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi1>, %[[arg1:.+]]: tensor<4xi32>, %[[arg2:.+]]: tensor<4xi32>) -> tensor<4xi32> {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xi1>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xi1>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xi1>
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xi1>
//       CHECK:     %[[c0_3:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_4:.+]] = tensor.extract %[[arg1]][%[[c0_3]]] : tensor<4xi32>
//       CHECK:     %[[c1_5:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_6:.+]] = tensor.extract %[[arg1]][%[[c1_5]]] : tensor<4xi32>
//       CHECK:     %[[c2_7:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_8:.+]] = tensor.extract %[[arg1]][%[[c2_7]]] : tensor<4xi32>
//       CHECK:     %[[c3_9:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_10:.+]] = tensor.extract %[[arg1]][%[[c3_9]]] : tensor<4xi32>
//       CHECK:     %[[c0_11:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_12:.+]] = tensor.extract %[[arg2]][%[[c0_11]]] : tensor<4xi32>
//       CHECK:     %[[c1_13:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_14:.+]] = tensor.extract %[[arg2]][%[[c1_13]]] : tensor<4xi32>
//       CHECK:     %[[c2_15:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_16:.+]] = tensor.extract %[[arg2]][%[[c2_15]]] : tensor<4xi32>
//       CHECK:     %[[c3_17:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_18:.+]] = tensor.extract %[[arg2]][%[[c3_17]]] : tensor<4xi32>
//       CHECK:     %[[v0:.+]] = arith.select %[[extracted]], %[[extracted_4]], %[[extracted_12]] : i32
//       CHECK:     %[[v1:.+]] = arith.select %[[extracted_0]], %[[extracted_6]], %[[extracted_14]] : i32
//       CHECK:     %[[v2:.+]] = arith.select %[[extracted_1]], %[[extracted_8]], %[[extracted_16]] : i32
//       CHECK:     %[[v3:.+]] = arith.select %[[extracted_2]], %[[extracted_10]], %[[extracted_18]] : i32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]], %[[v1]], %[[v2]], %[[v3]] : tensor<4xi32>
//       CHECK:     return %[[from_elements]] : tensor<4xi32>

// -----

func.func @test_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_shift_left
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[c32_i32:.+]] = arith.constant 32 : i32
//       CHECK: %[[v_0:.+]] = arith.shli %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[v_1:.+]] = arith.cmpi ult, %[[extracted_0]], %[[c32_i32]] : i32
//       CHECK: %[[v_2:.+]] = arith.select %[[v_1]], %[[v_0]], %[[c0_i32]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_2]] : tensor<i32>

// -----

func.func @test_shift_right_arith(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_shift_right_arith
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c31_i32:.+]] = arith.constant 31 : i32
//   CHECK-DAG: %[[c32_i32:.+]] = arith.constant 32 : i32
//       CHECK: %[[v_0:.+]] = arith.shrsi %[[extracted]], %[[c31_i32]] : i32
//       CHECK: %[[v_1:.+]] = arith.shrsi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[v_2:.+]] = arith.cmpi ult, %[[extracted_0]], %[[c32_i32]] : i32
//       CHECK: %[[v_3:.+]] = arith.select %[[v_2]], %[[v_1]], %[[v_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_3]] : tensor<i32>

// -----

func.func @test_shift_right_logical(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_shift_right_logical
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[c32_i32:.+]] = arith.constant 32 : i32
//       CHECK: %[[v_0:.+]] = arith.shrui %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[v_1:.+]] = arith.cmpi ult, %[[extracted_0]], %[[c32_i32]] : i32
//       CHECK: %[[v_2:.+]] = arith.select %[[v_1]], %[[v_0]], %[[c0_i32]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_2]] : tensor<i32>

// -----

func.func @test_sign(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.sign"(%arg0) : (tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_sign
//   CHECK-DAG: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//   CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[c31_i32:.+]] = arith.constant 31 : i32
//   CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
//       CHECK: %[[v_0:.+]] = arith.cmpi eq, %[[extracted]], %[[c0_i32]] : i32
//       CHECK: %[[v_1:.+]] = arith.shrsi %[[extracted]], %[[c31_i32]] : i32
//       CHECK: %[[v_2:.+]] = arith.ori %[[v_1]], %[[c1_i32]] : i32
//       CHECK: %[[v_3:.+]] = arith.select %[[v_0]], %[[c0_i32]], %[[v_2]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_3]] : tensor<i32>

// -----

func.func @test_sine(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.sine"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_sine
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.sin %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_sqrt(%arg0: tensor<f32>) -> tensor<f32> {
  %1 = "stablehlo.sqrt"(%arg0) : (tensor<f32>) ->  tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: test_sqrt
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<f32>
//       CHECK: %[[v_0:.+]] = math.sqrt %[[extracted]] : f32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<f32>

// -----

func.func @test_sub(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_sub
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.subi %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_xor(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %1 = "stablehlo.xor"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) ->  tensor<i32>
  return %1: tensor<i32>
}
// CHECK-LABEL: test_xor
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0:.+]][] : tensor<i32>
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1:.+]][] : tensor<i32>
//       CHECK: %[[v_0:.+]] = arith.xori %[[extracted]], %[[extracted_0]] : i32
//       CHECK: %[[from_elements:.+]] = tensor.from_elements %[[v_0]] : tensor<i32>

// -----

func.func @test_tensor4xi32_xor(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %1 = "stablehlo.xor"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) ->  tensor<4xi32>
  return %1: tensor<4xi32>
}

// CHECK-LABEL: @test_tensor4xi32_xor
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<4xi32>) -> tensor<4xi32> {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xi32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xi32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xi32>
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xi32>
//       CHECK:     %[[c0_3:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_4:.+]] = tensor.extract %[[arg1]][%[[c0_3]]] : tensor<4xi32>
//       CHECK:     %[[c1_5:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_6:.+]] = tensor.extract %[[arg1]][%[[c1_5]]] : tensor<4xi32>
//       CHECK:     %[[c2_7:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_8:.+]] = tensor.extract %[[arg1]][%[[c2_7]]] : tensor<4xi32>
//       CHECK:     %[[c3_9:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_10:.+]] = tensor.extract %[[arg1]][%[[c3_9]]] : tensor<4xi32>
//       CHECK:     %[[v0:.+]] = arith.xori %[[extracted]], %[[extracted_4]] : i32
//       CHECK:     %[[v1:.+]] = arith.xori %[[extracted_0]], %[[extracted_6]] : i32
//       CHECK:     %[[v2:.+]] = arith.xori %[[extracted_1]], %[[extracted_8]] : i32
//       CHECK:     %[[v3:.+]] = arith.xori %[[extracted_2]], %[[extracted_10]] : i32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]], %[[v1]], %[[v2]], %[[v3]] : tensor<4xi32>
//       CHECK:     return %[[from_elements]] : tensor<4xi32>

// -----

func.func @test_iota() -> tensor<4xi32> {
  %1 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<4xi32>)
  return %1: tensor<4xi32>
}

// CHECK-LABEL: @test_iota
//       CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK:     %[[c1_i32:.+]] = arith.constant 1 : i32
//       CHECK:     %[[c2_i32:.+]] = arith.constant 2 : i32
//       CHECK:     %[[c3_i32:.+]] = arith.constant 3 : i32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[c0_i32]], %[[c1_i32]], %[[c2_i32]], %[[c3_i32]]
//       CHECK:     return %[[from_elements]] : tensor<4xi32>

// -----

func.func @test_iota_f32() -> tensor<4xf32> {
  %1 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<4xf32>)
  return %1: tensor<4xf32>
}

// CHECK-LABEL: @test_iota_f32
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[cst_0:.+]] = arith.constant 1.000000e+00 : f32
//       CHECK:     %[[cst_1:.+]] = arith.constant 2.000000e+00 : f32
//       CHECK:     %[[cst_2:.+]] = arith.constant 3.000000e+00 : f32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[cst]], %[[cst_0]], %[[cst_1]], %[[cst_2]]
//       CHECK:     return %[[from_elements]] : tensor<4xf32>

// -----

func.func @test_reshape_scalar(%arg0: tensor<i32>) -> tensor<1xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<i32>)->tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @test_reshape_scalar
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<1xi32> {
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[extracted]] : tensor<1xi32>
//       CHECK:     return %[[from_elements]] : tensor<1xi32

// -----

func.func @test_concat(%arg0: tensor<1xi32>, %arg1: tensor<3xi32>) -> tensor<4xi32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @test_concat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<3xi32>) -> tensor<4xi32> {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xi32>
//       CHECK:     %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c0_0]]] : tensor<3xi32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<3xi32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_3:.+]] = tensor.extract %[[arg1]][%[[c2]]] : tensor<3xi32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[extracted]], %[[extracted_1]], %[[extracted_2]], %[[extracted_3]] : tensor<4xi32>
//       CHECK:     return %[[from_elements]] : tensor<4xi32>

// -----

func.func @test_broadcast_in_dim(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0>} : (tensor<1xi32>) -> tensor<3xi32>
  %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64>} : (tensor<i32>) -> tensor<3xi32>
  return %0, %1 : tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: @test_broadcast_in_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<3xi32>, tensor<3xi32>) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xi32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[extracted]], %[[extracted]], %[[extracted]] : tensor<3xi32>
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][] : tensor<i32>
//       CHECK:     %[[from_elements_1:.+]] = tensor.from_elements %[[extracted_0]], %[[extracted_0]], %[[extracted_0]] : tensor<3xi32>
//       CHECK:     return %[[from_elements]], %[[from_elements_1]] : tensor<3xi32>, tensor<3xi32

// -----

func.func @test_reduce_1(%arg0: tensor<1xi8>, %arg1: tensor<i8>) -> tensor<i8> {
  %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.and across dimensions = [0] : (tensor<1xi8>, tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}

// CHECK-LABEL: @test_reduce_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi8>, %[[arg1:.+]]: tensor<i8>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xi8>
//       CHECK:     %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][] : tensor<i8>
//       CHECK:     %[[v0:.+]] = arith.andi %[[extracted]], %[[extracted_1]] : i8
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]] : tensor<i8>
//       CHECK:     return %[[from_elements]] : tensor<i8>

// -----

func.func @test_reduce_4(%arg0: tensor<4xi8>, %arg1: tensor<i8>) -> tensor<i8> {
  %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.and across dimensions = [0] : (tensor<4xi8>, tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}

// CHECK-LABEL: @test_reduce_4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi8>, %[[arg1:.+]]: tensor<i8>)
//       CHECK:     %[[c0:.+]] = arith.constant 0
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] :
//       CHECK:     %[[c1:.+]] = arith.constant 1
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xi8>
//       CHECK:     %[[c2:.+]] = arith.constant 2
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xi8>
//       CHECK:     %[[c3:.+]] = arith.constant 3
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xi8>
//       CHECK:     %[[c0_3:.+]] = arith.constant 0
//       CHECK:     %[[extracted_4:.+]] = tensor.extract %[[arg1]][]
//       CHECK:     %[[v0:.+]] = arith.andi %[[extracted]], %[[extracted_4]] : i8
//       CHECK:     %[[v1:.+]] = arith.andi %[[extracted_0]], %[[v0]] : i8
//       CHECK:     %[[v2:.+]] = arith.andi %[[extracted_1]], %[[v1]] : i8
//       CHECK:     %[[v3:.+]] = arith.andi %[[extracted_2]], %[[v2]] : i8
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v3]]
//       CHECK:     return %[[from_elements]] : tensor<i8>

// -----

func.func @test_add_2d(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @test_add_2d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2xf32>, %[[arg1:.+]]: tensor<2x2xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]], %[[c0_0]]]
//       CHECK:     %[[c0_1:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c0_1]], %[[c1]]]
//       CHECK:     %[[c1_3:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0_4:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_5:.+]] = tensor.extract %[[arg0]][%[[c1_3]], %[[c0_4]]] : tensor<2x2xf32>
//       CHECK:     %[[c1_6:.+]] = arith.constant 1 : index
//       CHECK:     %[[c1_7:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_8:.+]] = tensor.extract %[[arg0]][%[[c1_6]], %[[c1_7]]] : tensor<2x2xf32>
//       CHECK:     %[[c0_9:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_10:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_11:.+]] = tensor.extract %[[arg1]][%[[c0_9]], %[[c0_10]]] : tensor<2x2xf32>
//       CHECK:     %[[c0_12:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1_13:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_14:.+]] = tensor.extract %[[arg1]][%[[c0_12]], %[[c1_13]]] : tensor<2x2xf32>
//       CHECK:     %[[c1_15:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0_16:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_17:.+]] = tensor.extract %[[arg1]][%[[c1_15]], %[[c0_16]]] : tensor<2x2xf32>
//       CHECK:     %[[c1_18:.+]] = arith.constant 1 : index
//       CHECK:     %[[c1_19:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_20:.+]] = tensor.extract %[[arg1]][%[[c1_18]], %[[c1_19]]] : tensor<2x2xf32>
//       CHECK:     %[[v0:.+]] = arith.addf %[[extracted]], %[[extracted_11]] : f32
//       CHECK:     %[[v1:.+]] = arith.addf %[[extracted_2]], %[[extracted_14]] : f32
//       CHECK:     %[[v2:.+]] = arith.addf %[[extracted_5]], %[[extracted_17]] : f32
//       CHECK:     %[[v3:.+]] = arith.addf %[[extracted_8]], %[[extracted_20]] : f32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]], %[[v1]], %[[v2]], %[[v3]]
//       CHECK:     return %[[from_elements]]

// -----

func.func @test_reshape_2d(%arg0: tensor<1x4xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x4xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @test_reshape_2d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x4xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]], %[[c0_0]]] : tensor<1x4xf32>
//       CHECK:     %[[c0_1:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c0_1]], %[[c1]]] : tensor<1x4xf32>
//       CHECK:     %[[c0_3:.+]] = arith.constant 0 : index
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_4:.+]] = tensor.extract %[[arg0]][%[[c0_3]], %[[c2]]] : tensor<1x4xf32>
//       CHECK:     %[[c0_5:.+]] = arith.constant 0 : index
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_6:.+]] = tensor.extract %[[arg0]][%[[c0_5]], %[[c3]]] : tensor<1x4xf32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[extracted]], %[[extracted_2]], %[[extracted_4]], %[[extracted_6]] : tensor<2x2xf32>
//       CHECK:     return %[[from_elements]]

// -----

func.func @test_slice(%arg0: tensor<4xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1>,
    limit_indices = array<i64: 3>,
    strides = array<i64: 1>
  } : (tensor<4xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @test_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>
//       CHECK:      %[[c0:.+]] = arith.constant 0 : index
//       CHECK:      %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xf32>
//       CHECK:      %[[c1:.+]] = arith.constant 1 : index
//       CHECK:      %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xf32>
//       CHECK:      %[[c2:.+]] = arith.constant 2 : index
//       CHECK:      %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xf32>
//       CHECK:      %[[c3:.+]] = arith.constant 3 : index
//       CHECK:      %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xf32>
//       CHECK:      %[[from_elements:.+]] = tensor.from_elements %[[extracted_0]], %[[extracted_1]] : tensor<2xf32>
//       CHECK:      return %[[from_elements]] : tensor<2xf32>

