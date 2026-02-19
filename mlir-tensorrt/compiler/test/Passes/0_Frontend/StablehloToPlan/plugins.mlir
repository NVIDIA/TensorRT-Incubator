// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-plan | FileCheck %s

module @jit_add_one_cuda {
  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>)
     -> (tensor<5xf32>, tensor<5xf32>) {
    %0:2 = stablehlo.custom_call @add_one_cuda(%arg0, %arg1) {
      backend_config = "",
      mhlo.backend_config = {
        arg_spec = "attrs.param;args.0;args.1",
        func = "add_one_cuda",
        mtrt_ffi_backend = "tvm_ffi",
        param = 3 : i64,
        plugin = "build/lib/TVMFFICUDATestPlugin.so"
      },
      operand_layouts = [dense<0> : tensor<1xindex>,
                         dense<0> : tensor<1xindex>],
      output_operand_aliases = [
        #stablehlo.output_operand_alias<
          output_tuple_indices = [1],
          operand_index = 0,
          operand_tuple_indices = []>,
        #stablehlo.output_operand_alias<
          output_tuple_indices = [0],
          operand_index = 1,
          operand_tuple_indices = []>
      ],
      result_layouts = [dense<0> : tensor<1xindex>,
                        dense<0> : tensor<1xindex>]
    } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
    return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
  }
}

//       CHECK: module @jit_add_one_cuda
//       CHECK:   executor.plugin @plugin_add_one_cuda0 {
//  CHECK-SAME: config = {
//  CHECK-SAME:  param = 3 : i64
//  CHECK-SAME:  ffi_backend = #executor.ffi_backend<tvm_ffi>,
//  CHECK-SAME:  function_name = "add_one_cuda",
//  CHECK-SAME:  plugin_name = "build/lib/TVMFFICUDATestPlugin.so"
// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5xf32>,
//  CHECK-SAME: %[[arg1:.+]]: tensor<5xf32>) ->
//       CHECK:     %[[v0:.+]]:2 = executor.call_plugin
//  CHECK-SAME: @plugin_add_one_cuda0 stream({{.*}}) ins(%[[arg0]], %[[arg1]] :
//  CHECK-SAME: tensor<5xf32>, tensor<5xf32>) outs(%[[arg1]],
//  CHECK-SAME: %[[arg0]] : tensor<5xf32>, tensor<5xf32>)
//  CHECK-SAME:  immediate_args = {param = 3 : i64}
//  CHECK-SAME:  io_aliasing = array<i32: 1, 0>
//       CHECK:     return %[[v0]]#0, %[[v0]]#1
