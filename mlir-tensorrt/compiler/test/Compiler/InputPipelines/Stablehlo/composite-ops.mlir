// RUN: mlir-tensorrt-opt %s -stablehlo-input-pipeline

func.func public @composite_call(%arg0: tensor<4xf32>) -> (tensor<4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.composite "my.tangent" %arg0 {composite_attributes = {dtype = f32, int = 1 : i64, str = "bar", tensor = dense<0.000000e+00> : tensor<1x2xf32>, tensor_r1 = dense<0.000000e+00> : tensor<2xf32>}, decomposition = @my.tangent} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
func.func private @my.tangent(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.sine %arg0 : tensor<4xf32>
    %1 = stablehlo.cosine %arg0 : tensor<4xf32>
    %2 = stablehlo.divide %0, %1 : tensor<4xf32>
    return %2 : tensor<4xf32>
}

// CHECK-LABEL: @composite_call
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.sine %[[arg0]] : tensor<4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.cosine %[[arg0]] : tensor<4xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.divide %[[v0]], %[[v1]] : tensor<4xf32>
//  CHECK-NEXT: return %[[v2]] : tensor<4xf32>
