// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-refine-shapes | FileCheck %s

func.func @check_type_refinement() -> tensor<?xf32> {
  %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c_0 = stablehlo.constant dense<3> : tensor<i32>
  %c_1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c_2 = stablehlo.constant dense<3> : tensor<1xi32>
  %c_3 = stablehlo.constant dense<1> : tensor<i32>
  %c_4 = stablehlo.constant dense<1> : tensor<1xi32>
  %c_5 = stablehlo.constant dense<0> : tensor<i32>
  %c_6 = stablehlo.constant dense<1> : tensor<i32>
  %c_7 = stablehlo.constant dense<0> : tensor<1xi32>
  %c_8 = stablehlo.constant dense<1> : tensor<1xi32>
  %0 = stablehlo.compare  LE, %c_7, %c_8 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %1 = stablehlo.select %0, %c_7, %c_8 : tensor<1xi1>, tensor<1xi32>
  %c_9 = stablehlo.constant dense<1> : tensor<1xi32>
  %2 = stablehlo.real_dynamic_slice %c_4, %1, %c_8, %c_9 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %c_10 = stablehlo.constant dense<> : tensor<0xi32>
  %3 = stablehlo.dynamic_reshape %2, %c_10 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %c_11 = stablehlo.constant dense<-1> : tensor<i32>
  %c_12 = stablehlo.constant dense<> : tensor<0xi32>
  %4 = stablehlo.compare  EQ, %c_12, %c_10 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi1>
  %5 = stablehlo.select %4, %c_12, %c_12 : tensor<0xi1>, tensor<0xi32>
  %6 = stablehlo.dynamic_broadcast_in_dim %3, %5, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %7 = stablehlo.dynamic_broadcast_in_dim %c_11, %5, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %8 = stablehlo.add %6, %7 : tensor<i32>
  %c_13 = stablehlo.constant dense<0> : tensor<1xi32>
  %c_14 = stablehlo.constant dense<1> : tensor<1xi32>
  %9 = stablehlo.compare  LE, %c_13, %c_14 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %10 = stablehlo.select %9, %c_13, %c_14 : tensor<1xi1>, tensor<1xi32>
  %c_15 = stablehlo.constant dense<1> : tensor<1xi32>
  %11 = stablehlo.real_dynamic_slice %c_4, %10, %c_14, %c_15 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %12 = stablehlo.dynamic_reshape %11, %c_10 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %13 = stablehlo.compare  EQ, %c_12, %c_10 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi1>
  %14 = stablehlo.select %13, %c_12, %c_12 : tensor<0xi1>, tensor<0xi32>
  %15 = stablehlo.dynamic_broadcast_in_dim %12, %14, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %16 = stablehlo.dynamic_broadcast_in_dim %c_11, %14, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %17 = stablehlo.add %15, %16 : tensor<i32>
  %18 = stablehlo.compare  EQ, %c_12, %c_10 : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi1>
  %19 = stablehlo.select %18, %c_12, %c_12 : tensor<0xi1>, tensor<0xi32>
  %20 = stablehlo.dynamic_broadcast_in_dim %17, %19, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %21 = stablehlo.dynamic_broadcast_in_dim %c_6, %19, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %22 = stablehlo.add %20, %21 : tensor<i32>
  %23 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
  %24 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
  %25 = stablehlo.compare  LE, %23, %24 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %26 = stablehlo.select %25, %23, %24 : tensor<1xi1>, tensor<1xi32>
  %c_16 = stablehlo.constant dense<1> : tensor<1xi32>
  %27 = stablehlo.real_dynamic_slice %c_2, %26, %24, %c_16 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %28 = stablehlo.dynamic_reshape %27, %c_10 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %29 = stablehlo.dynamic_broadcast_in_dim %28, %c_1, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %30 = stablehlo.dynamic_broadcast_in_dim %cst, %29, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  return %30 : tensor<?xf32>
}

// CHECK-LABEL: func.func @check_type_refinement
//       CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
//       CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<-1> : tensor<i32>
//       CHECK-DAG:     %[[c_0:.+]] = stablehlo.constant dense<> : tensor<0xi32>
//       CHECK-DAG:     %[[c_1:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//       CHECK-DAG:     %[[c_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
//       CHECK-DAG:     %[[c_3:.+]] = stablehlo.constant dense<1> : tensor<i32>
//       CHECK-DAG:     %[[c_4:.+]] = stablehlo.constant dense<0> : tensor<1xi32>
//       CHECK-DAG:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[c_1]], %[[c_4]], %[[c_1]], %[[c_1]] : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
//       CHECK-DAG:     %[[v1:.+]] = stablehlo.dynamic_reshape %[[v0]], %[[c_0]] : (tensor<1xi32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v2:.+]] = stablehlo.dynamic_broadcast_in_dim %[[v1]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v3:.+]] = stablehlo.dynamic_broadcast_in_dim %[[c]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v4:.+]] = stablehlo.add %[[v2]], %[[v3]] : tensor<i32>
//       CHECK-DAG:     %[[v5:.+]] = stablehlo.real_dynamic_slice %[[c_1]], %[[c_4]], %[[c_1]], %[[c_1]] : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
//       CHECK-DAG:     %[[v6:.+]] = stablehlo.dynamic_reshape %[[v5]], %[[c_0]] : (tensor<1xi32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v7:.+]] = stablehlo.dynamic_broadcast_in_dim %[[v6]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v8:.+]] = stablehlo.dynamic_broadcast_in_dim %[[c]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v9:.+]] = stablehlo.add %[[v7]], %[[v8]] : tensor<i32>
//       CHECK-DAG:     %[[v10:.+]] = stablehlo.dynamic_broadcast_in_dim %[[v9]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v11:.+]] = stablehlo.dynamic_broadcast_in_dim %[[c_3]], %[[c_0]], dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v12:.+]] = stablehlo.add %[[v10]], %[[v11]] : tensor<i32>
//       CHECK-DAG:     %[[v13:.+]] = stablehlo.reshape %[[v4]] : (tensor<i32>) -> tensor<1xi32>
//       CHECK-DAG:     %[[v14:.+]] = stablehlo.reshape %[[v12]] : (tensor<i32>) -> tensor<1xi32>
//       CHECK-DAG:     %[[v15:.+]] = stablehlo.compare  LE, %[[v13]], %[[v14]] : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
//       CHECK-DAG:     %[[v16:.+]] = stablehlo.select %[[v15]], %[[v13]], %[[v14]] : tensor<1xi1>, tensor<1xi32>
//       CHECK-DAG:     %[[v17:.+]] = stablehlo.real_dynamic_slice %[[c_2]], %[[v16]], %[[v14]], %[[c_1]] : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//       CHECK-DAG:     %[[v18:.+]] = stablehlo.dynamic_reshape %[[v17]], %[[c_0]] : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
//       CHECK-DAG:     %[[v19:.+]] = stablehlo.dynamic_broadcast_in_dim %[[v18]], %[[c_1]], dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
//       CHECK-DAG:     %[[v20:.+]] = stablehlo.dynamic_broadcast_in_dim %[[cst]], %[[v19]], dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
//       CHECK-DAG:     return %[[v20]] : tensor<?xf32>