// RUN: executor-opt %s -split-input-file -canonicalize | FileCheck %s

module @plugin_side_effects {
  executor.plugin @plugin_add_one_cuda0 {
    config = {},
    ffi_backend = #executor.ffi_backend<tvm_ffi>,
    function_name = "add_one_cuda",
    plugin_name = "TVMFFICUDATestPlugin.so"
  } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  func.func public @main(
      %arg0: memref<5xf32, #executor.memory_type<device>>,
      %arg1: memref<5xf32, #executor.memory_type<device>>) -> () {
    executor.call_plugin @plugin_add_one_cuda0
        ins(%arg0, %arg1 :
            memref<5xf32, #executor.memory_type<device>>,
            memref<5xf32, #executor.memory_type<device>>)
        outs(%arg1, %arg0 :
             memref<5xf32, #executor.memory_type<device>>,
             memref<5xf32, #executor.memory_type<device>>)
        {arg_spec = ["args.0", "args.1"]}
    return
  }
}

// CHECK-LABEL: @plugin_side_effects
//       CHECK: func.func public @main
//       CHECK: executor.call_plugin @plugin_add_one_cuda0
