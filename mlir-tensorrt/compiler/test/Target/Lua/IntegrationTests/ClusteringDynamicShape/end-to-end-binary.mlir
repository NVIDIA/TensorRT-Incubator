// RUN: %pick-one-gpu mlir-tensorrt-opt %s  \
// RUN: -pass-pipeline="builtin.module(stablehlo-preprocessing-pipeline{disable-inliner},\
// RUN: stablehlo-clustering-pipeline{entrypoint=}, \
// RUN: post-clustering-pipeline, \
// RUN: executor-lowering-pipeline)" \
// RUN: | mlir-tensorrt-translate -mlir-to-runtime-executable -allow-unregistered-dialect |  \
// RUN: %pick-one-gpu mlir-tensorrt-runner -input-type=rtexe -features=core,cuda,tensorrt | FileCheck %s

#profile = #tensorrt.shape_profile<min = [2], opt = [4], max = [6]>
#profile1 = #tensorrt.shape_profile<min = [1], opt = [3], max = [5]>

builtin.module @end_to_end_binary attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=false, tensorrt_major_version = 10>,
    #plan.host_cluster<benefit = 0>
  ]
} {

  func.func private @add(%arg0: tensor<?xf32> {tensorrt.shape_profile = #profile},
                %arg1: tensor<?xf32> {tensorrt.shape_profile = #profile}) -> tensor<?xf32> {
      %2 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
      return %2 : tensor<?xf32>
  }

  func.func @print_tensor(%data: tensor<?xf32>) -> () {
    executor.print "\\n"()
    %c0 = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = tensor.dim %data, %c0 : tensor<?xf32>

    scf.for %i = %c0 to %stop step %step {
      %r = tensor.extract %data[%i] : tensor<?xf32>
      executor.print "result[%d] = %.3f"(%i, %r: index, f32)
    }

    return
  }

  func.func private @test_add() -> () {
    %opt = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %c4 = arith.constant 4 : index
    %opt_d = tensor.cast %opt : tensor<4xf32> to tensor<?xf32>
    %result_opt = func.call @add(%opt_d, %opt_d) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    func.call @print_tensor(%result_opt) : (tensor<?xf32>) -> ()

    %opt2 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>
    %c2 = arith.constant 2 : index
    %opt2_d = tensor.cast %opt2 : tensor<6xf32> to tensor<?xf32>
    %result_opt2 = func.call @add(%opt2_d, %opt2_d) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

    func.call @print_tensor(%result_opt2) : (tensor<?xf32>) -> ()

    return
  }

  func.func public @main() -> index {

    func.call @test_add() : () -> ()

    %c0 = arith.constant 0 : index
    return %c0 : index
  }

}

//      CHECK: result[0] = 2.000
// CHECK-NEXT: result[1] = 2.000
// CHECK-NEXT: result[2] = 2.000
// CHECK-NEXT: result[3] = 2.000

//      CHECK: result[0] = 2.000
// CHECK-NEXT: result[1] = 4.000
// CHECK-NEXT: result[2] = 6.000
// CHECK-NEXT: result[3] = 8.000
// CHECK-NEXT: result[4] = 10.000
// CHECK-NEXT: result[5] = 12.000

// CHECK-NOT: result
