// RUN: mlir-tensorrt-opt %s  \
// RUN: -pass-pipeline="builtin.module(stablehlo-preprocessing-pipeline{disable-inliner},\
// RUN: stablehlo-clustering-pipeline, \
// RUN: post-clustering-pipeline, \
// RUN: executor-lowering-pipeline)" \
// RUN: | mlir-tensorrt-translate -mlir-to-runtime-executable -allow-unregistered-dialect |  mlir-tensorrt-runner -input-type=rtexe

#profile0 = #tensorrt.shape_profile<min = [1], opt = [5], max = [10]>
#profile1 = #tensorrt.shape_profile<min = [2], opt = [5], max = [6]>

// This function computes `out[i] = e^(e^(arg0[i]))` for `0 <= i < arg1`. It returns a tensor
// contining the first `%arg1` number of elements.
func.func @test_separated_data_dependent(%arg0: tensor<?xf32> {tensorrt.shape_profile = #profile0},
                                                 %arg1: index {tensorrt.value_bounds = #profile1}) -> tensor<?xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<?xf32>
  %2 = tensor.extract_slice %0[0][%arg1][1] : tensor<?xf32> to tensor<?xf32>
  %3 = stablehlo.exponential %2 : tensor<?xf32>
  return %3 : tensor<?xf32>
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

func.func @main() -> index {
  %0 = arith.constant dense<[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]> : tensor<6xf32>
  %1 = tensor.cast %0 : tensor<6xf32> to tensor<?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %2 = func.call @test_separated_data_dependent(%1, %c3) : (tensor<?xf32>, index) -> tensor<?xf32>
  func.call @print_tensor(%2) : (tensor<?xf32>)->()

  // Call again with a different slice index.
  %3 = func.call @test_separated_data_dependent(%1, %c6) : (tensor<?xf32>, index) -> tensor<?xf32>
  func.call @print_tensor(%3) : (tensor<?xf32>)->()

  // Call again with a different input shape and different slice index.
  %4 = arith.constant dense<[1.0, 0.0, 1.0]> : tensor<3xf32>
  %5 = tensor.cast %4 : tensor<3xf32> to tensor<?xf32>
  %c2 = arith.constant 2 : index
  %6 = func.call @test_separated_data_dependent(%5, %c2) : (tensor<?xf32>, index) -> tensor<?xf32>
  func.call @print_tensor(%6) : (tensor<?xf32>)->()

  return %c0 : index
}


// CHECK-LABEL: result[0] = 2.718
//  CHECK-NEXT: result[1] = 2.718
//  CHECK-NEXT: result[2] = 2.718

//  CHECK-NEXT: result[0] = 2.718
//  CHECK-NEXT: result[1] = 2.718
//  CHECK-NEXT: result[2] = 2.718
//  CHECK-NEXT: result[3] = 15.154
//  CHECK-NEXT: result[4] = 15.154
//  CHECK-NEXT: result[5] = 15.154

//  CHECK-NEXT: result[0] = 15.154
//  CHECK-NEXT: result[1] = 2.718
//   CHECK-NOT: result
