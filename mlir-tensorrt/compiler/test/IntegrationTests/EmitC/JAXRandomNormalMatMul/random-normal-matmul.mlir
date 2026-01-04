// REQUIRES: cuda
// REQUIRES: system-linux
// REQUIRES: tensorrt

// RUN: rm -rf %t/random_normal_matmul.cpp %t/random_normal_matmul_test || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-compiler %s --host-target=emitc --artifacts-dir=%t --entrypoint=random_normal_matmul_main --abi-version=0 \
// RUN:   -o %t/random_normal_matmul.cpp
// RUN: %host_cxx -c %t/random_normal_matmul.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeStatus.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCore.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCuda.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeTensorRT.cpp \
// RUN:  -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:   %cuda_toolkit_linux_cxx_flags \
// RUN:  -I%nvinfer_include_dir \
// RUN:  -L%nvinfer_lib_dir \
// RUN:  -lnvinfer

module @jit_random_funcs attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, plan.backends = [#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 3, tensorrt_major_version = 10>, #plan.kernel_backend<benefit = 2>, #plan.host_backend<benefit = 1>]} {
  func.func public @random_normal_matmul_main(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32> {jax.result_info = ""}) {
    %c = stablehlo.constant dense<4294967295> : tensor<ui32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<32> : tensor<i32>
    %0 = stablehlo.shift_right_logical %c_0, %c_1 : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %3 = stablehlo.convert %c : (tensor<ui32>) -> tensor<i32>
    %4 = stablehlo.and %c_0, %3 : tensor<i32>
    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<ui32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %7 = stablehlo.concatenate %2, %6, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %8 = call @_normal(%7) : (tensor<2xui32>) -> tensor<128x128xf32>
    %9 = stablehlo.dot_general %arg0, %8, contracting_dims = [1] x [0] : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %9 : tensor<128x128xf32>
  }
  func.func private @_normal(%arg0: tensor<2xui32>) -> tensor<128x128xf32> {
    %0 = call @_normal_real(%arg0) : (tensor<2xui32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func.func private @_normal_real(%arg0: tensor<2xui32>) -> tensor<128x128xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<128x128xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<128x128xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<128x128xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<128x128xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<128x128xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<128x128xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<128x128xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<128x128xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<128x128xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<128x128xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<128x128xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<128x128xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<128x128xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<128x128xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<128x128xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<128x128xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<128x128xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<128x128xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<128x128xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<128x128xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<128x128xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<128x128xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<128x128xf32>
    %1 = stablehlo.negate %0 : tensor<128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x128xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<128x128xf32>
    %4 = stablehlo.negate %3 : tensor<128x128xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<128x128xf32>
    %7 = stablehlo.sqrt %4 : tensor<128x128xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<128x128xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<128x128xi1>, tensor<128x128xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<128x128xi1>, tensor<128x128xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<128x128xi1>, tensor<128x128xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<128x128xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x128xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<128x128xi1>, tensor<128x128xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<128x128xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x128xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<128x128xi1>, tensor<128x128xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<128x128xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x128xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<128x128xi1>, tensor<128x128xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<128x128xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x128xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<128x128xi1>, tensor<128x128xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<128x128xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x128xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<128x128xi1>, tensor<128x128xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<128x128xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x128xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<128x128xi1>, tensor<128x128xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<128x128xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x128xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<128x128xi1>, tensor<128x128xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<128x128xf32>
    %34 = stablehlo.add %32, %33 : tensor<128x128xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<128x128xf32>
    %36 = stablehlo.abs %0 : tensor<128x128xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<128x128xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<128x128xi1>, tensor<128x128xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<128x128xf32>
    return %41 : tensor<128x128xf32>
  }
  func.func private @_uniform(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<128x128xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %c_1 = stablehlo.constant dense<5> : tensor<ui32>
    %c_2 = stablehlo.constant dense<4> : tensor<ui32>
    %c_3 = stablehlo.constant dense<2> : tensor<ui32>
    %c_4 = stablehlo.constant dense<8> : tensor<ui32>
    %c_5 = stablehlo.constant dense<24> : tensor<ui32>
    %c_6 = stablehlo.constant dense<16> : tensor<ui32>
    %c_7 = stablehlo.constant dense<3> : tensor<ui32>
    %c_8 = stablehlo.constant dense<29> : tensor<ui32>
    %c_9 = stablehlo.constant dense<1> : tensor<ui32>
    %c_10 = stablehlo.constant dense<6> : tensor<ui32>
    %c_11 = stablehlo.constant dense<26> : tensor<ui32>
    %c_12 = stablehlo.constant dense<17> : tensor<ui32>
    %c_13 = stablehlo.constant dense<15> : tensor<ui32>
    %c_14 = stablehlo.constant dense<19> : tensor<ui32>
    %c_15 = stablehlo.constant dense<13> : tensor<ui32>
    %c_16 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<16384xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:8192] : (tensor<16384xui32>) -> tensor<8192xui32>
    %8 = stablehlo.slice %2 [8192:16384] : (tensor<16384xui32>) -> tensor<8192xui32>
    %9 = stablehlo.xor %4, %6 : tensor<ui32>
    %10 = stablehlo.xor %9, %c_16 : tensor<ui32>
    %11 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %12 = stablehlo.add %7, %11 : tensor<8192xui32>
    %13 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %14 = stablehlo.add %8, %13 : tensor<8192xui32>
    %15 = stablehlo.add %12, %14 : tensor<8192xui32>
    %16 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %17 = stablehlo.shift_left %14, %16 : tensor<8192xui32>
    %18 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %19 = stablehlo.shift_right_logical %14, %18 : tensor<8192xui32>
    %20 = stablehlo.or %17, %19 : tensor<8192xui32>
    %21 = stablehlo.xor %15, %20 : tensor<8192xui32>
    %22 = stablehlo.add %15, %21 : tensor<8192xui32>
    %23 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %24 = stablehlo.shift_left %21, %23 : tensor<8192xui32>
    %25 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %26 = stablehlo.shift_right_logical %21, %25 : tensor<8192xui32>
    %27 = stablehlo.or %24, %26 : tensor<8192xui32>
    %28 = stablehlo.xor %22, %27 : tensor<8192xui32>
    %29 = stablehlo.add %22, %28 : tensor<8192xui32>
    %30 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %31 = stablehlo.shift_left %28, %30 : tensor<8192xui32>
    %32 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %33 = stablehlo.shift_right_logical %28, %32 : tensor<8192xui32>
    %34 = stablehlo.or %31, %33 : tensor<8192xui32>
    %35 = stablehlo.xor %29, %34 : tensor<8192xui32>
    %36 = stablehlo.add %29, %35 : tensor<8192xui32>
    %37 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %38 = stablehlo.shift_left %35, %37 : tensor<8192xui32>
    %39 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %40 = stablehlo.shift_right_logical %35, %39 : tensor<8192xui32>
    %41 = stablehlo.or %38, %40 : tensor<8192xui32>
    %42 = stablehlo.xor %36, %41 : tensor<8192xui32>
    %43 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %44 = stablehlo.add %36, %43 : tensor<8192xui32>
    %45 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %46 = stablehlo.add %42, %45 : tensor<8192xui32>
    %47 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %48 = stablehlo.add %46, %47 : tensor<8192xui32>
    %49 = stablehlo.add %44, %48 : tensor<8192xui32>
    %50 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %51 = stablehlo.shift_left %48, %50 : tensor<8192xui32>
    %52 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %53 = stablehlo.shift_right_logical %48, %52 : tensor<8192xui32>
    %54 = stablehlo.or %51, %53 : tensor<8192xui32>
    %55 = stablehlo.xor %49, %54 : tensor<8192xui32>
    %56 = stablehlo.add %49, %55 : tensor<8192xui32>
    %57 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %58 = stablehlo.shift_left %55, %57 : tensor<8192xui32>
    %59 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %60 = stablehlo.shift_right_logical %55, %59 : tensor<8192xui32>
    %61 = stablehlo.or %58, %60 : tensor<8192xui32>
    %62 = stablehlo.xor %56, %61 : tensor<8192xui32>
    %63 = stablehlo.add %56, %62 : tensor<8192xui32>
    %64 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %65 = stablehlo.shift_left %62, %64 : tensor<8192xui32>
    %66 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %67 = stablehlo.shift_right_logical %62, %66 : tensor<8192xui32>
    %68 = stablehlo.or %65, %67 : tensor<8192xui32>
    %69 = stablehlo.xor %63, %68 : tensor<8192xui32>
    %70 = stablehlo.add %63, %69 : tensor<8192xui32>
    %71 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %72 = stablehlo.shift_left %69, %71 : tensor<8192xui32>
    %73 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %74 = stablehlo.shift_right_logical %69, %73 : tensor<8192xui32>
    %75 = stablehlo.or %72, %74 : tensor<8192xui32>
    %76 = stablehlo.xor %70, %75 : tensor<8192xui32>
    %77 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %78 = stablehlo.add %70, %77 : tensor<8192xui32>
    %79 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %80 = stablehlo.add %76, %79 : tensor<8192xui32>
    %81 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %82 = stablehlo.add %80, %81 : tensor<8192xui32>
    %83 = stablehlo.add %78, %82 : tensor<8192xui32>
    %84 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %85 = stablehlo.shift_left %82, %84 : tensor<8192xui32>
    %86 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %87 = stablehlo.shift_right_logical %82, %86 : tensor<8192xui32>
    %88 = stablehlo.or %85, %87 : tensor<8192xui32>
    %89 = stablehlo.xor %83, %88 : tensor<8192xui32>
    %90 = stablehlo.add %83, %89 : tensor<8192xui32>
    %91 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %92 = stablehlo.shift_left %89, %91 : tensor<8192xui32>
    %93 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %94 = stablehlo.shift_right_logical %89, %93 : tensor<8192xui32>
    %95 = stablehlo.or %92, %94 : tensor<8192xui32>
    %96 = stablehlo.xor %90, %95 : tensor<8192xui32>
    %97 = stablehlo.add %90, %96 : tensor<8192xui32>
    %98 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %99 = stablehlo.shift_left %96, %98 : tensor<8192xui32>
    %100 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %101 = stablehlo.shift_right_logical %96, %100 : tensor<8192xui32>
    %102 = stablehlo.or %99, %101 : tensor<8192xui32>
    %103 = stablehlo.xor %97, %102 : tensor<8192xui32>
    %104 = stablehlo.add %97, %103 : tensor<8192xui32>
    %105 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %106 = stablehlo.shift_left %103, %105 : tensor<8192xui32>
    %107 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %108 = stablehlo.shift_right_logical %103, %107 : tensor<8192xui32>
    %109 = stablehlo.or %106, %108 : tensor<8192xui32>
    %110 = stablehlo.xor %104, %109 : tensor<8192xui32>
    %111 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %112 = stablehlo.add %104, %111 : tensor<8192xui32>
    %113 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %114 = stablehlo.add %110, %113 : tensor<8192xui32>
    %115 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %116 = stablehlo.add %114, %115 : tensor<8192xui32>
    %117 = stablehlo.add %112, %116 : tensor<8192xui32>
    %118 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %119 = stablehlo.shift_left %116, %118 : tensor<8192xui32>
    %120 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %121 = stablehlo.shift_right_logical %116, %120 : tensor<8192xui32>
    %122 = stablehlo.or %119, %121 : tensor<8192xui32>
    %123 = stablehlo.xor %117, %122 : tensor<8192xui32>
    %124 = stablehlo.add %117, %123 : tensor<8192xui32>
    %125 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %126 = stablehlo.shift_left %123, %125 : tensor<8192xui32>
    %127 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %128 = stablehlo.shift_right_logical %123, %127 : tensor<8192xui32>
    %129 = stablehlo.or %126, %128 : tensor<8192xui32>
    %130 = stablehlo.xor %124, %129 : tensor<8192xui32>
    %131 = stablehlo.add %124, %130 : tensor<8192xui32>
    %132 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %133 = stablehlo.shift_left %130, %132 : tensor<8192xui32>
    %134 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %135 = stablehlo.shift_right_logical %130, %134 : tensor<8192xui32>
    %136 = stablehlo.or %133, %135 : tensor<8192xui32>
    %137 = stablehlo.xor %131, %136 : tensor<8192xui32>
    %138 = stablehlo.add %131, %137 : tensor<8192xui32>
    %139 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %140 = stablehlo.shift_left %137, %139 : tensor<8192xui32>
    %141 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %142 = stablehlo.shift_right_logical %137, %141 : tensor<8192xui32>
    %143 = stablehlo.or %140, %142 : tensor<8192xui32>
    %144 = stablehlo.xor %138, %143 : tensor<8192xui32>
    %145 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %146 = stablehlo.add %138, %145 : tensor<8192xui32>
    %147 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %148 = stablehlo.add %144, %147 : tensor<8192xui32>
    %149 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %150 = stablehlo.add %148, %149 : tensor<8192xui32>
    %151 = stablehlo.add %146, %150 : tensor<8192xui32>
    %152 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %153 = stablehlo.shift_left %150, %152 : tensor<8192xui32>
    %154 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %155 = stablehlo.shift_right_logical %150, %154 : tensor<8192xui32>
    %156 = stablehlo.or %153, %155 : tensor<8192xui32>
    %157 = stablehlo.xor %151, %156 : tensor<8192xui32>
    %158 = stablehlo.add %151, %157 : tensor<8192xui32>
    %159 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %160 = stablehlo.shift_left %157, %159 : tensor<8192xui32>
    %161 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %162 = stablehlo.shift_right_logical %157, %161 : tensor<8192xui32>
    %163 = stablehlo.or %160, %162 : tensor<8192xui32>
    %164 = stablehlo.xor %158, %163 : tensor<8192xui32>
    %165 = stablehlo.add %158, %164 : tensor<8192xui32>
    %166 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %167 = stablehlo.shift_left %164, %166 : tensor<8192xui32>
    %168 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %169 = stablehlo.shift_right_logical %164, %168 : tensor<8192xui32>
    %170 = stablehlo.or %167, %169 : tensor<8192xui32>
    %171 = stablehlo.xor %165, %170 : tensor<8192xui32>
    %172 = stablehlo.add %165, %171 : tensor<8192xui32>
    %173 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %174 = stablehlo.shift_left %171, %173 : tensor<8192xui32>
    %175 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %176 = stablehlo.shift_right_logical %171, %175 : tensor<8192xui32>
    %177 = stablehlo.or %174, %176 : tensor<8192xui32>
    %178 = stablehlo.xor %172, %177 : tensor<8192xui32>
    %179 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %180 = stablehlo.add %172, %179 : tensor<8192xui32>
    %181 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %182 = stablehlo.add %178, %181 : tensor<8192xui32>
    %183 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %184 = stablehlo.add %182, %183 : tensor<8192xui32>
    %185 = stablehlo.concatenate %180, %184, dim = 0 : (tensor<8192xui32>, tensor<8192xui32>) -> tensor<16384xui32>
    %186 = stablehlo.reshape %185 : (tensor<16384xui32>) -> tensor<128x128xui32>
    %187 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<128x128xui32>
    %188 = stablehlo.shift_right_logical %186, %187 : tensor<128x128xui32>
    %189 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<128x128xui32>
    %190 = stablehlo.or %188, %189 : tensor<128x128xui32>
    %191 = stablehlo.bitcast_convert %190 : (tensor<128x128xui32>) -> tensor<128x128xf32>
    %192 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    %193 = stablehlo.subtract %191, %192 : tensor<128x128xf32>
    %194 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<128x128xf32>
    %196 = stablehlo.multiply %193, %195 : tensor<128x128xf32>
    %197 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<128x128xf32>
    %198 = stablehlo.add %196, %197 : tensor<128x128xf32>
    %199 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<128x128xf32>
    %200 = stablehlo.maximum %199, %198 : tensor<128x128xf32>
    return %200 : tensor<128x128xf32>
  }
}
