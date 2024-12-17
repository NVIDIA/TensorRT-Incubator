// RUN: mlir-tensorrt-opt --allow-unregistered-dialect -convert-tensorrt-to-emitc -canonicalize %s -split-input-file | FileCheck %s
// RUN: mlir-tensorrt-opt --allow-unregistered-dialect -convert-tensorrt-to-emitc -canonicalize %s -split-input-file | mlir-tensorrt-translate -mlir-to-cpp | FileCheck %s --check-prefix=CPP

// CHECK-LABEL: @resnet50_builder(
// CHECK-LABEL: @resnet50_tester(

// CPP-LABEL: void resnet50_builder(
// CPP-LABEL: ::std::unique_ptr<::nvinfer1::IHostMemory> resnet50_tester(

func.func @resnet50(
    %arg0: tensor<?x3x224x224xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 3, 224, 224], opt = [5, 3, 224, 224], max = [10, 3, 224, 224]>}
  ) -> tensor<?x1000xf32>
      attributes {
        input_names = ["data"],
        output_names = ["resnetv24_dense0_fwd"]
      } {
  %cst_f32 = tensorrt.constant dense_resource<__elided__> : tensor<2048x1x1xf32>
  %cst_f32_0 = tensorrt.constant dense_resource<__elided__> : tensor<1024x1x1xf32>
  %cst_f32_1 = tensorrt.constant dense_resource<__elided__> : tensor<512x1x1xf32>
  %cst_f32_2 = tensorrt.constant dense_resource<__elided__> : tensor<256x1x1xf32>
  %cst_f32_3 = tensorrt.constant dense_resource<__elided__> : tensor<64x1x1xf32>
  %cst_f32_4 = tensorrt.constant dense<[[[0.114799649]], [[0.108629107]], [[0.100710414]]]> : tensor<3x1x1xf32>
  %cst_i32 = tensorrt.constant dense<[0, -1]> : tensor<2xi32>
  %cst_f32_5 = tensorrt.constant dense_resource<__elided__> : tensor<1000x2048xf32>
  %cst_f32_6 = tensorrt.constant dense_resource<__elided__> : tensor<1000xf32>
  %cst_f32_7 = tensorrt.constant dense<[[[0.802063524]], [[0.80319941]], [[0.796995401]]]> : tensor<3x1x1xf32>
  %0 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 3, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_7 : tensor<3x1x1xf32>) -> tensor<1x3x1x1xf32>
  %1 = tensorrt.element_wise <kPROD>(%arg0, %0 : tensor<?x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<?x3x224x224xf32>
  %2 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 3, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_4 : tensor<3x1x1xf32>) -> tensor<1x3x1x1xf32>
  %3 = tensorrt.element_wise <kSUM>(%1, %2 : tensor<?x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<?x3x224x224xf32>
  %4 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x3x7x7xf32>, post_padding = array<i64: 3, 3>, pre_padding = array<i64: 3, 3>, stride = array<i64: 2, 2>} in(%3 : tensor<?x3x224x224xf32>) -> tensor<?x64x112x112xf32>
  %5 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %4 : tensor<?x64x112x112xf32>
  %6 = tensorrt.pooling {poolingType = #tensorrt.pooling_type<kMAX>, postPadding = array<i64: 1, 1>, prePadding = array<i64: 1, 1>, stride = array<i64: 2, 2>, windowSize = array<i64: 3, 3>} ins(%5 : tensor<?x64x112x112xf32>) -> tensor<?x64x56x56xf32>
  %7 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 64, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_3 : tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
  %8 = tensorrt.element_wise <kPROD>(%6, %7 : tensor<?x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<?x64x56x56xf32>
  %9 = tensorrt.element_wise <kSUM>(%8, %7 : tensor<?x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<?x64x56x56xf32>
  %10 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %9 : tensor<?x64x56x56xf32>
  %11 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x64x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%10 : tensor<?x64x56x56xf32>) -> tensor<?x64x56x56xf32>
  %12 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %11 : tensor<?x64x56x56xf32>
  %13 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x64x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%12 : tensor<?x64x56x56xf32>) -> tensor<?x64x56x56xf32>
  %14 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %13 : tensor<?x64x56x56xf32>
  %15 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x64x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%14 : tensor<?x64x56x56xf32>) -> tensor<?x256x56x56xf32>
  %16 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x64x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%10 : tensor<?x64x56x56xf32>) -> tensor<?x256x56x56xf32>
  %17 = tensorrt.element_wise <kSUM>(%15, %16 : tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>
  %18 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 256, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_2 : tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
  %19 = tensorrt.element_wise <kPROD>(%17, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %20 = tensorrt.element_wise <kSUM>(%19, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %21 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %20 : tensor<?x256x56x56xf32>
  %22 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%21 : tensor<?x256x56x56xf32>) -> tensor<?x64x56x56xf32>
  %23 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %22 : tensor<?x64x56x56xf32>
  %24 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x64x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%23 : tensor<?x64x56x56xf32>) -> tensor<?x64x56x56xf32>
  %25 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %24 : tensor<?x64x56x56xf32>
  %26 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x64x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%25 : tensor<?x64x56x56xf32>) -> tensor<?x256x56x56xf32>
  %27 = tensorrt.element_wise <kSUM>(%26, %17 : tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>
  %28 = tensorrt.element_wise <kPROD>(%27, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %29 = tensorrt.element_wise <kSUM>(%28, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %30 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %29 : tensor<?x256x56x56xf32>
  %31 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%30 : tensor<?x256x56x56xf32>) -> tensor<?x64x56x56xf32>
  %32 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %31 : tensor<?x64x56x56xf32>
  %33 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x64x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%32 : tensor<?x64x56x56xf32>) -> tensor<?x64x56x56xf32>
  %34 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %33 : tensor<?x64x56x56xf32>
  %35 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x64x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%34 : tensor<?x64x56x56xf32>) -> tensor<?x256x56x56xf32>
  %36 = tensorrt.element_wise <kSUM>(%35, %27 : tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>
  %37 = tensorrt.element_wise <kPROD>(%36, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %38 = tensorrt.element_wise <kSUM>(%37, %18 : tensor<?x256x56x56xf32>, tensor<1x256x1x1xf32>) -> tensor<?x256x56x56xf32>
  %39 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %38 : tensor<?x256x56x56xf32>
  %40 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%39 : tensor<?x256x56x56xf32>) -> tensor<?x128x56x56xf32>
  %41 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %40 : tensor<?x128x56x56xf32>
  %42 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x128x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 2, 2>} in(%41 : tensor<?x128x56x56xf32>) -> tensor<?x128x28x28xf32>
  %43 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %42 : tensor<?x128x28x28xf32>
  %44 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x128x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%43 : tensor<?x128x28x28xf32>) -> tensor<?x512x28x28xf32>
  %45 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 2, 2>} in(%39 : tensor<?x256x56x56xf32>) -> tensor<?x512x28x28xf32>
  %46 = tensorrt.element_wise <kSUM>(%44, %45 : tensor<?x512x28x28xf32>, tensor<?x512x28x28xf32>) -> tensor<?x512x28x28xf32>
  %47 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 512, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_1 : tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
  %48 = tensorrt.element_wise <kPROD>(%46, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %49 = tensorrt.element_wise <kSUM>(%48, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %50 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %49 : tensor<?x512x28x28xf32>
  %51 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%50 : tensor<?x512x28x28xf32>) -> tensor<?x128x28x28xf32>
  %52 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %51 : tensor<?x128x28x28xf32>
  %53 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x128x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%52 : tensor<?x128x28x28xf32>) -> tensor<?x128x28x28xf32>
  %54 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %53 : tensor<?x128x28x28xf32>
  %55 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x128x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%54 : tensor<?x128x28x28xf32>) -> tensor<?x512x28x28xf32>
  %56 = tensorrt.element_wise <kSUM>(%55, %46 : tensor<?x512x28x28xf32>, tensor<?x512x28x28xf32>) -> tensor<?x512x28x28xf32>
  %57 = tensorrt.element_wise <kPROD>(%56, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %58 = tensorrt.element_wise <kSUM>(%57, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %59 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %58 : tensor<?x512x28x28xf32>
  %60 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%59 : tensor<?x512x28x28xf32>) -> tensor<?x128x28x28xf32>
  %61 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %60 : tensor<?x128x28x28xf32>
  %62 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x128x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%61 : tensor<?x128x28x28xf32>) -> tensor<?x128x28x28xf32>
  %63 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %62 : tensor<?x128x28x28xf32>
  %64 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x128x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%63 : tensor<?x128x28x28xf32>) -> tensor<?x512x28x28xf32>
  %65 = tensorrt.element_wise <kSUM>(%64, %56 : tensor<?x512x28x28xf32>, tensor<?x512x28x28xf32>) -> tensor<?x512x28x28xf32>
  %66 = tensorrt.element_wise <kPROD>(%65, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %67 = tensorrt.element_wise <kSUM>(%66, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %68 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %67 : tensor<?x512x28x28xf32>
  %69 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%68 : tensor<?x512x28x28xf32>) -> tensor<?x128x28x28xf32>
  %70 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %69 : tensor<?x128x28x28xf32>
  %71 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<128xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<128x128x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%70 : tensor<?x128x28x28xf32>) -> tensor<?x128x28x28xf32>
  %72 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %71 : tensor<?x128x28x28xf32>
  %73 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x128x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%72 : tensor<?x128x28x28xf32>) -> tensor<?x512x28x28xf32>
  %74 = tensorrt.element_wise <kSUM>(%73, %65 : tensor<?x512x28x28xf32>, tensor<?x512x28x28xf32>) -> tensor<?x512x28x28xf32>
  %75 = tensorrt.element_wise <kPROD>(%74, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %76 = tensorrt.element_wise <kSUM>(%75, %47 : tensor<?x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<?x512x28x28xf32>
  %77 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %76 : tensor<?x512x28x28xf32>
  %78 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%77 : tensor<?x512x28x28xf32>) -> tensor<?x256x28x28xf32>
  %79 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %78 : tensor<?x256x28x28xf32>
  %80 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 2, 2>} in(%79 : tensor<?x256x28x28xf32>) -> tensor<?x256x14x14xf32>
  %81 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %80 : tensor<?x256x14x14xf32>
  %82 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%81 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %83 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 2, 2>} in(%77 : tensor<?x512x28x28xf32>) -> tensor<?x1024x14x14xf32>
  %84 = tensorrt.element_wise <kSUM>(%82, %83 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %85 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 1024, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32_0 : tensor<1024x1x1xf32>) -> tensor<1x1024x1x1xf32>
  %86 = tensorrt.element_wise <kPROD>(%84, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %87 = tensorrt.element_wise <kSUM>(%86, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %88 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %87 : tensor<?x1024x14x14xf32>
  %89 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%88 : tensor<?x1024x14x14xf32>) -> tensor<?x256x14x14xf32>
  %90 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %89 : tensor<?x256x14x14xf32>
  %91 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%90 : tensor<?x256x14x14xf32>) -> tensor<?x256x14x14xf32>
  %92 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %91 : tensor<?x256x14x14xf32>
  %93 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%92 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %94 = tensorrt.element_wise <kSUM>(%93, %84 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %95 = tensorrt.element_wise <kPROD>(%94, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %96 = tensorrt.element_wise <kSUM>(%95, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %97 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %96 : tensor<?x1024x14x14xf32>
  %98 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%97 : tensor<?x1024x14x14xf32>) -> tensor<?x256x14x14xf32>
  %99 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %98 : tensor<?x256x14x14xf32>
  %100 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%99 : tensor<?x256x14x14xf32>) -> tensor<?x256x14x14xf32>
  %101 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %100 : tensor<?x256x14x14xf32>
  %102 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%101 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %103 = tensorrt.element_wise <kSUM>(%102, %94 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %104 = tensorrt.element_wise <kPROD>(%103, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %105 = tensorrt.element_wise <kSUM>(%104, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %106 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %105 : tensor<?x1024x14x14xf32>
  %107 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%106 : tensor<?x1024x14x14xf32>) -> tensor<?x256x14x14xf32>
  %108 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %107 : tensor<?x256x14x14xf32>
  %109 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%108 : tensor<?x256x14x14xf32>) -> tensor<?x256x14x14xf32>
  %110 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %109 : tensor<?x256x14x14xf32>
  %111 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%110 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %112 = tensorrt.element_wise <kSUM>(%111, %103 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %113 = tensorrt.element_wise <kPROD>(%112, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %114 = tensorrt.element_wise <kSUM>(%113, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %115 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %114 : tensor<?x1024x14x14xf32>
  %116 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%115 : tensor<?x1024x14x14xf32>) -> tensor<?x256x14x14xf32>
  %117 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %116 : tensor<?x256x14x14xf32>
  %118 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%117 : tensor<?x256x14x14xf32>) -> tensor<?x256x14x14xf32>
  %119 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %118 : tensor<?x256x14x14xf32>
  %120 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%119 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %121 = tensorrt.element_wise <kSUM>(%120, %112 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %122 = tensorrt.element_wise <kPROD>(%121, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %123 = tensorrt.element_wise <kSUM>(%122, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %124 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %123 : tensor<?x1024x14x14xf32>
  %125 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%124 : tensor<?x1024x14x14xf32>) -> tensor<?x256x14x14xf32>
  %126 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %125 : tensor<?x256x14x14xf32>
  %127 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<256xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<256x256x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%126 : tensor<?x256x14x14xf32>) -> tensor<?x256x14x14xf32>
  %128 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %127 : tensor<?x256x14x14xf32>
  %129 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<1024x256x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%128 : tensor<?x256x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %130 = tensorrt.element_wise <kSUM>(%129, %121 : tensor<?x1024x14x14xf32>, tensor<?x1024x14x14xf32>) -> tensor<?x1024x14x14xf32>
  %131 = tensorrt.element_wise <kPROD>(%130, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %132 = tensorrt.element_wise <kSUM>(%131, %85 : tensor<?x1024x14x14xf32>, tensor<1x1024x1x1xf32>) -> tensor<?x1024x14x14xf32>
  %133 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %132 : tensor<?x1024x14x14xf32>
  %134 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%133 : tensor<?x1024x14x14xf32>) -> tensor<?x512x14x14xf32>
  %135 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %134 : tensor<?x512x14x14xf32>
  %136 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x512x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 2, 2>} in(%135 : tensor<?x512x14x14xf32>) -> tensor<?x512x7x7xf32>
  %137 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %136 : tensor<?x512x7x7xf32>
  %138 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<2048x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%137 : tensor<?x512x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %139 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<2048x1024x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 2, 2>} in(%133 : tensor<?x1024x14x14xf32>) -> tensor<?x2048x7x7xf32>
  %140 = tensorrt.element_wise <kSUM>(%138, %139 : tensor<?x2048x7x7xf32>, tensor<?x2048x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %141 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 2048, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zeroIsPlaceholder = false} ins(%cst_f32 : tensor<2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
  %142 = tensorrt.element_wise <kPROD>(%140, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %143 = tensorrt.element_wise <kSUM>(%142, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %144 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %143 : tensor<?x2048x7x7xf32>
  %145 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x2048x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%144 : tensor<?x2048x7x7xf32>) -> tensor<?x512x7x7xf32>
  %146 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %145 : tensor<?x512x7x7xf32>
  %147 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x512x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%146 : tensor<?x512x7x7xf32>) -> tensor<?x512x7x7xf32>
  %148 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %147 : tensor<?x512x7x7xf32>
  %149 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<2048x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%148 : tensor<?x512x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %150 = tensorrt.element_wise <kSUM>(%149, %140 : tensor<?x2048x7x7xf32>, tensor<?x2048x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %151 = tensorrt.element_wise <kPROD>(%150, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %152 = tensorrt.element_wise <kSUM>(%151, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %153 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %152 : tensor<?x2048x7x7xf32>
  %154 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x2048x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%153 : tensor<?x2048x7x7xf32>) -> tensor<?x512x7x7xf32>
  %155 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %154 : tensor<?x512x7x7xf32>
  %156 = tensorrt.convolution {biasStatic = dense_resource<__elided__> : tensor<512xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<512x512x3x3xf32>, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 1, 1>} in(%155 : tensor<?x512x7x7xf32>) -> tensor<?x512x7x7xf32>
  %157 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %156 : tensor<?x512x7x7xf32>
  %158 = tensorrt.convolution {dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<2048x512x1x1xf32>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%157 : tensor<?x512x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %159 = tensorrt.element_wise <kSUM>(%158, %150 : tensor<?x2048x7x7xf32>, tensor<?x2048x7x7xf32>) -> tensor<?x2048x7x7xf32>
  %160 = tensorrt.element_wise <kPROD>(%159, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %161 = tensorrt.element_wise <kSUM>(%160, %141 : tensor<?x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<?x2048x7x7xf32>
  %162 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %161 : tensor<?x2048x7x7xf32>
  %163 = tensorrt.reduce <kAVG> %162 {keepDimensions = true, reduceAxes = array<i64: 2, 3>} : tensor<?x2048x7x7xf32> -> tensor<?x2048x1x1xf32>
  %164 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2, 3>, second_transpose = array<i64: 0, 1>, zeroIsPlaceholder = true} ins(%163, %cst_i32 : tensor<?x2048x1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %165 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%164, %cst_f32_5 : tensor<?x?xf32>, tensor<1000x2048xf32>) -> tensor<?x1000xf32>
  %166 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1000>, second_transpose = array<i64: 0, 1>, zeroIsPlaceholder = false} ins(%cst_f32_6 : tensor<1000xf32>) -> tensor<1x1000xf32>
  %167 = tensorrt.element_wise <kSUM>(%166, %165 : tensor<1x1000xf32>, tensor<?x1000xf32>) -> tensor<?x1000xf32>
  return %167 : tensor<?x1000xf32>
}

