// RUN: mlir-tensorrt-compiler %s -opts="disable-tensorrt-extension entrypoint=" -o - | \
// RUN: mlir-tensorrt-runner -input-type=rtexe -features=core -split-input-file

// This test contains a nested loop structure. It is derived from an upstream JAX
// unit test in 'lax_control_flow_test.py' that exposed a bug in the executor
// Lua translation.

// The computation just computes

#cluster_kinds = [
  #plan.host_backend<benefit = 1>
]

module @nested_while attributes {
    plan.backends = #cluster_kinds,
    plan.memory_space = #plan.memory_space<host>
} {

  // Zero elements above the diagonal of `%arg2` (e.g. np.tril(arg2, k=0)).
  // %arg3 is output tensor to be updated using dynamic_update_slice.
  // %arg0 and %arg1 are 0 and size of matrix (5) respectively.
  // They are passed as arguments to match the original JAX test.
  // Important: computation must be marked no_inline to prevent folding.
  func.func private @compute(%arg0: tensor<i32>, %arg1: tensor<i32>,
                             %arg2: tensor<5x5xf32>, %arg3: tensor<5x5xf32>)
                              -> (tensor<i32> {jax.result_info = "[0]"}, tensor<i32> {jax.result_info = "[1]"}, tensor<5x5xf32> {jax.result_info = "[2]"}, tensor<5x5xf32> {jax.result_info = "[3]"})
      attributes {no_inline} {
    %c = stablehlo.constant dense<5> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %0:4 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %arg2, %iterArg_4 = %arg3) : tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>
     cond {
      %e = tensor.extract %iterArg[] : tensor<i32>
      %1 = stablehlo.compare  LT, %iterArg, %iterArg_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i32>
      %2:4 = stablehlo.while(%iterArg_5 = %iterArg, %iterArg_6 = %c_0, %iterArg_7 = %iterArg_3, %iterArg_8 = %iterArg_4) : tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>
       cond {
        %e = tensor.extract %iterArg_6[] : tensor<i32>
        %3 = stablehlo.compare  LE, %iterArg_6, %iterArg_5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %3 : tensor<i1>
      } do {
        %3 = stablehlo.compare  LT, %iterArg_5, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %4 = stablehlo.convert %iterArg_5 : tensor<i32>
        %5 = stablehlo.add %4, %c : tensor<i32>
        %6 = stablehlo.select %3, %5, %iterArg_5 : tensor<i1>, tensor<i32>
        %7 = stablehlo.dynamic_slice %iterArg_7, %6, %c_0, sizes = [1, 5] : (tensor<5x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
        %8 = stablehlo.reshape %7 : (tensor<1x5xf32>) -> tensor<5xf32>
        %9 = stablehlo.compare  LT, %iterArg_6, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %10 = stablehlo.convert %iterArg_6 : tensor<i32>
        %11 = stablehlo.add %10, %c : tensor<i32>
        %12 = stablehlo.select %9, %11, %iterArg_6 : tensor<i1>, tensor<i32>
        %13 = stablehlo.dynamic_slice %8, %12, sizes = [1] : (tensor<5xf32>, tensor<i32>) -> tensor<1xf32>
        %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
        %15 = stablehlo.reshape %14 : (tensor<f32>) -> tensor<1x1xf32>
        %16 = stablehlo.compare  LT, %iterArg_5, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %17 = stablehlo.convert %iterArg_5 : tensor<i32>
        %18 = stablehlo.add %17, %c : tensor<i32>
        %19 = stablehlo.select %16, %18, %iterArg_5 : tensor<i1>, tensor<i32>
        %20 = stablehlo.compare  LT, %iterArg_6, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %21 = stablehlo.convert %iterArg_6 : tensor<i32>
        %22 = stablehlo.add %21, %c : tensor<i32>
        %23 = stablehlo.select %20, %22, %iterArg_6 : tensor<i1>, tensor<i32>
        %24 = stablehlo.dynamic_update_slice %iterArg_8, %15, %19, %23 : (tensor<5x5xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<5x5xf32>
        %25 = stablehlo.add %iterArg_6, %c_1 : tensor<i32>
        stablehlo.return %iterArg_5, %25, %iterArg_7, %24 : tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>
      }
      stablehlo.return %1, %iterArg_2, %iterArg_3, %2#3 : tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>
    }
    return %0#0, %0#1, %0#2, %0#3 : tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>
  }

  func.func private @print_tensor(%arg0: tensor<5x5xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.for %i = %c0 to %c5 step %c1 {
      scf.for %j = %c0 to %c5 step %c1 {
        %0 = tensor.extract %arg0[%i, %j] : tensor<5x5xf32>
        executor.print "tensor[%d, %d] = %f"(%i, %j, %0 : index, index, f32)
      }
    }
    return
  }

  func.func @main() -> i32 {
    %input = stablehlo.constant dense<
      [[-2.14022618,  1.04267863, -1.60212933, -1.64424777,  1.40534482],
       [-0.00507625,  0.29647936, -0.67771467,  0.38206631,  0.81398157],
       [-0.11529496,  0.21862135, -0.21087151, -0.03706357, -2.0103956 ],
       [ 1.1299681,   0.14887056, -1.38651613, -0.0346357,  -0.67762548],
       [-2.33308187,  0.99721413, -1.4355879,  -0.23594572, -0.09954703]]> : tensor<5x5xf32>
    %expected = stablehlo.constant dense<
      [[-2.14022618,  0.,          0.,          0.,          0.        ],
       [-0.00507625,  0.29647936,  0.,          0.,          0.        ],
       [-0.11529496,  0.21862135, -0.21087151,  0.,          0.        ],
       [ 1.1299681,   0.14887056, -1.38651613, -0.0346357,   0.        ],
       [-2.33308187,  0.99721413, -1.4355879,  -0.23594572, -0.09954703]]> : tensor<5x5xf32>

    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<5> : tensor<i32>

    %zeros = stablehlo.constant dense<0.0> : tensor<5x5xf32>

    %0:4 = call @compute(%c0, %c1, %input, %zeros) : (tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>)
      -> (tensor<i32>, tensor<i32>, tensor<5x5xf32>, tensor<5x5xf32>)

    call @print_tensor(%0#3) : (tensor<5x5xf32>) -> ()

    // Check equality with expected.
    %comp = stablehlo.compare EQ, %0#3, %expected : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xi1>
    %true = stablehlo.constant dense<1> : tensor<i1>
    %3 = stablehlo.reduce (%comp init: %true) applies stablehlo.and across dimensions = [0, 1]
     : (tensor<5x5xi1>, tensor<i1>) -> tensor<i1>
    %e = tensor.extract %3[] : tensor<i1>
    executor.print "success = %d"(%e : i1)

    cf.assert %e, "result does not match expected"

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
