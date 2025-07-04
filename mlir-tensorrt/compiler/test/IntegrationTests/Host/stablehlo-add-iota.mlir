// RUN: mlir-tensorrt-opt %s -stablehlo-to-executable-pipeline="disable-tensorrt-extension entrypoint=" \
// RUN: | mlir-tensorrt-translate -mlir-to-runtime-executable -allow-unregistered-dialect \
// RUN: | mlir-tensorrt-runner -input-type=rtexe -features=core | \
// RUN: FileCheck %s

module @simple_add_iota attributes {plan.cluster_kinds = [#plan.host_cluster<benefit = 1>]} {

  func.func private @add_iota(%arg0: tensor<128xf32>) -> (tensor<128xf32>)
      attributes {no_inline, plan.memory_space = #plan.memory_space<host>} {
    %0 = stablehlo.iota dim = 0 : tensor<128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<128xf32>
    return %1 : tensor<128xf32>
  }

  func.func private @num_differences(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<i32>
      attributes {no_inline, plan.memory_space = #plan.memory_space<host>} {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<128xf32>
    %1 = stablehlo.abs %0 : tensor<128xf32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %cst0 = stablehlo.constant dense<0.000000> : tensor<128xf32>
    %diff = stablehlo.compare GT, %1, %cst0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %diff_i32 = stablehlo.convert %diff : (tensor<128xi1>) -> tensor<128xi32>
    %diff_f32 = stablehlo.convert %diff_i32 : (tensor<128xi32>) -> tensor<128xf32>
    %2 = stablehlo.reduce (%diff_i32 init: %c0) across dimensions = [0]
      : (tensor<128xi32>, tensor<i32>) -> tensor<i32>
      reducer(%arg6: tensor<i32>, %arg7: tensor<i32>) {
        %r = stablehlo.add %arg7, %arg6 : tensor<i32>
        stablehlo.return %r : tensor<i32>
      }
    return %2 : tensor<i32>
  }

  func.func private @print_tensor(%arg0: tensor<128xf32>) attributes {no_inline, plan.memory_space = #plan.memory_space<host>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %0 = tensor.extract %arg0[%i] : tensor<128xf32>
      executor.print "tensor[%d] = %f"(%i, %0 : index, f32)
    }
    return
  }

  func.func @main() attributes {plan.memory_space = #plan.memory_space<host>} {
    %0 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %1 = call @add_iota(%0) : (tensor<128xf32>) -> tensor<128xf32>
    call @print_tensor(%1) : (tensor<128xf32>) -> ()

    %expected = stablehlo.constant dense<
        [1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,  10.0,  11.0,
        12.0,  13.0,  14.0,  15.0,  16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,
        23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,  32.0,  33.0,
        34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,
        45.0,  46.0,  47.0,  48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,
        56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,  64.0,  65.0,  66.0,
        67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,
        78.0,  79.0,  80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,
        89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,
       100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
       111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
       122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0]> : tensor<128xf32>
    %2 = call @num_differences(%1, %expected) : (tensor<128xf32>, tensor<128xf32>) -> tensor<i32>
    %num = tensor.extract %2[] : tensor<i32>
    executor.print "num errors = %d"(%num : i32)
    return
  }
}

// CHECK: num errors = 0
