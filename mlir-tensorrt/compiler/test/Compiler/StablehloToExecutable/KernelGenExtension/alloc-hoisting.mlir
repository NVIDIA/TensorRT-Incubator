// REQUIRES: cuda
// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-compiler -opts="hoist-allocs-to-globals=true disable-tensorrt-extension" %s -mlir -o - | \
// RUN: FileCheck %s --check-prefix=HOIST

// RUN: mlir-tensorrt-compiler -opts="hoist-allocs-to-globals=false disable-tensorrt-extension" %s -mlir -o - | \
// RUN: FileCheck %s --check-prefix=NOHOIST

builtin.module @test_hoist_allocs attributes {
  plan.backends = [
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

//       HOIST: executor.global @workspace{{.*}}
// NOHOIST-NOT: executor.global @workspace{{.*}}

func.func @main(%arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %1 = stablehlo.dot %arg1, %arg2 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = stablehlo.dot %1, %arg2 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1, %2 : tensor<128x128xf32>, tensor<128x128xf32>
}

}
