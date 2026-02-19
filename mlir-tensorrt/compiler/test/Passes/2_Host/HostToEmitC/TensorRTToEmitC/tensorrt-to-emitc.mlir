// RUN: mlir-tensorrt-opt -convert-tensorrt-to-emitc -canonicalize %s -split-input-file | FileCheck %s
// RUN: mlir-tensorrt-opt -convert-tensorrt-to-emitc -canonicalize %s -split-input-file | mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

func.func @trt_slice_builder(%arg0: tensor<1024x1024xf32>)  -> tensor<128x128xf32> {
  %0 = tensorrt.slice %arg0[512, 512][128, 128][2, 2] : tensor<1024x1024xf32> to tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @trt_slice_builder
//       CHECK: %[[null:.+]] = "emitc.constant"() <{value = #emitc.opaque<"nullptr">}> : () -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//       CHECK: %[[input:.+]] = emitc.call_opaque "::nvinfer1::adaptor::networkAddInput"(%{{.+}}) {args = [0 : index, #emitc.opaque<"\22input_0\22">, #emitc.opaque<"::nvinfer1::DataType::kFLOAT">, #emitc.opaque<"::nvinfer1::Dims{2, {1024, 1024}}">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>) -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//       CHECK: %[[slice:.+]] = emitc.call_opaque "::nvinfer1::adaptor::networkAddSlice"(%{{.+}}, %[[input]], %[[null]], %[[null]], %[[null]], %[[null]]) {args = [0 : index, 1 : index, 2 : index, 3 : index, 4 : index, 5 : index, #emitc.opaque<"::nvinfer1::Dims{2, {512, 512}}">, #emitc.opaque<"::nvinfer1::Dims{2, {128, 128}}">, #emitc.opaque<"::nvinfer1::Dims{2, {2, 2}}">, #emitc.opaque<"::nvinfer1::SliceMode::kSTRICT_BOUNDS">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>) -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//       CHECK: emitc.call_opaque "::nvinfer1::adaptor::networkMarkOutput"(%{{.+}}, %[[slice]]) {args = [0 : index, 1 : index]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>) -> ()

// CPP: void trt_slice_builder_builder(::nvinfer1::INetworkDefinition* {{.+}}, std::unordered_map<const char*, std::vector<uint8_t>>& {{.+}})
// CPP: ::std::unique_ptr<::nvinfer1::IHostMemory> trt_slice_builder_tester()

// -----

#profile = #tensorrt.shape_profile<min=[1, 3, 224, 224], opt=[5, 3, 224, 224], max=[10, 3, 224, 224]>

func.func @trt_convolution(%arg0: tensor<?x3x224x224xf32> {tensorrt.shape_profile = #profile}) -> tensor<?x64x112x112xf32> {
  %0 = tensorrt.convolution {biasStatic = dense<[-4.31519293E-4, 0.0469331518, 0.182877243, 0.0603719354, 0.125377372, -2.59749097E-4, 0.104800701, 0.0492741279, -4.7041793E-4, -2.93487101E-4, 0.075063996, -3.07486393E-4, -4.12274268E-4, -3.09298775E-4, 0.0233091153, -3.54022079E-4, 0.20344013, 0.0376711711, -6.76479307E-4, 0.0423475467, 0.0442900509, 1.350410e-01, 3.141040e-02, 0.0345028304, 0.0279467013, -3.10788164E-4, -2.74391437E-4, 0.0616228282, -2.58856919E-4, 0.0513835512, 0.103114679, 0.0398080461, 0.0275381785, 0.26737985, -3.60830571E-4, 0.0606967248, 0.0541626811, 0.16662319, 0.185066834, -0.00142969133, -9.50477377E-4, 0.046104081, -2.99843668E-4, 0.0412311144, 0.0787721425, 0.0296669938, 0.0228572953, -0.055737935, 0.01954359, -4.58773051E-4, 0.0538752489, 0.0502701923, -2.86789058E-4, 0.0592686571, -2.84466834E-4, -3.50684219E-4, -0.00142672623, 0.10327334, -3.29065049E-4, 0.03248455, 0.0474017337, -2.75487808E-4, -6.91905792E-4, 0.0261605475]> : tensor<64xf32>, dilation = array<i64: 1, 1>, kernelStatic = dense_resource<__elided__> : tensor<64x3x7x7xf32>, post_padding = array<i64: 3, 3>, pre_padding = array<i64: 3, 3>, stride = array<i64: 2, 2>} in(%arg0 : tensor<?x3x224x224xf32>) -> tensor<?x64x112x112xf32>
  return %0 : tensor<?x64x112x112xf32>
}

// CHECK-LABEL: @trt_convolution_builder

// CPP: void trt_convolution_builder(::nvinfer1::INetworkDefinition* [[net:.+]], std::unordered_map<const char*, std::vector<uint8_t>>& [[weightsMap:.+]]) {
// CPP-DAG: ::nvinfer1::ITensor* [[null:.+]] = nullptr;
// CPP-DAG: ::nvinfer1::ITensor* [[input:.+]] = ::nvinfer1::adaptor::networkAddInput([[net]], "input_0", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{4, {-1, 3, 224, 224}});
// CPP-DAG: ::nvinfer1::Weights [[kernel:.+]] = nvinfer1::adaptor::trtSetWeightsSplat<float>([[weightsMap]], "c0", 9408, 1.00{{.+}}-01f);
// CPP-DAG: ::nvinfer1::Weights [[bias:.+]] = nvinfer1::adaptor::trtSetWeights<float>([[weightsMap]], "c1", {{{.+}}});
// CPP: ::nvinfer1::ITensor* [[conv:.+]] = ::nvinfer1::adaptor::networkAddConvolution([[net]], [[input]], [[null]], [[null]], [[kernel]], [[bias]], 64, ::nvinfer1::Dims{2, {7, 7}}, ::nvinfer1::Dims{2, {2, 2}}, ::nvinfer1::Dims{2, {3, 3}}, ::nvinfer1::Dims{2, {3, 3}}, 1, ::nvinfer1::Dims{2, {1, 1}});
// CPP: ::nvinfer1::adaptor::networkMarkOutput([[net]], [[conv]]);

// -----
#profile = #tensorrt.shape_profile<min=[1, 10, 20, 30], opt=[5, 10, 20, 30], max=[10, 10, 20, 30]>

func.func @trt_activation(%arg0: tensor<?x10x20x30xf32> {tensorrt.shape_profile = #profile}) -> tensor<?x10x20x30xf32> {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?x10x20x30xf32>
  return %0 : tensor<?x10x20x30xf32>
}

// CHECK-LABEL: @trt_activation_builder
// CHECK-LABEL: @trt_activation_tester

//   CPP-LABEL: void trt_activation_builder
//    CPP-SAME:  (::nvinfer1::INetworkDefinition* [[net:.+]], std::unordered_map<const char*, std::vector<uint8_t>>& [[weightsMap:.+]]) {
//         CPP:   ::nvinfer1::ITensor* [[input:.+]] = ::nvinfer1::adaptor::networkAddInput([[net]], "input_0", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{4, {-1, 10, 20, 30}});
//         CPP:   ::nvinfer1::ITensor* [[act:.+]] = ::nvinfer1::adaptor::networkAddActivation([[net]], [[input]], ::nvinfer1::ActivationType::kRELU);
//         CPP:   ::nvinfer1::adaptor::networkMarkOutput([[net]], [[act]]);
//   CPP-LABEL: trt_activation_tester

// -----

func.func @trt_reduce(%arg0: tensor<2x3x224x224xf32>) -> tensor<2x224x224xf32> {
  %0 = tensorrt.reduce <kMAX> %arg0 {
    reduceAxes = array<i64: 1>
  } : tensor<2x3x224x224xf32> -> tensor<2x224x224xf32>

  return %0 : tensor<2x224x224xf32>
}
// CHECK-LABEL: @trt_reduce_builder
// CHECK-LABEL: @trt_reduce_tester

// CPP-LABEL: void trt_reduce_builder(
//       CPP:   ::nvinfer1::adaptor::networkAddReduce({{.+}}, {{.+}}, false, 2, ::nvinfer1::ReduceOperation::kMAX);
// -----

#profile = #tensorrt.shape_profile<min=[1, 64, 112, 112], opt=[5, 64, 112, 112], max=[10, 64, 112, 112]>

func.func @trt_pooling(%arg0: tensor<?x64x112x112xf32> {tensorrt.shape_profile = #profile}) -> tensor<?x64x56x56xf32> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<?x64x112x112xf32>) -> tensor<?x64x56x56xf32>
  return %0 : tensor<?x64x56x56xf32>
}

// CHECK-LABEL: @trt_pooling_builder
// CHECK-LABEL: @trt_pooling_tester

// CPP-LABEL: void trt_pooling_builder(
//       CPP: ::nvinfer1::adaptor::networkAddPooling({{.+}}, {{.+}}, ::nvinfer1::PoolingType::kMAX, ::nvinfer1::Dims{2, {1, 1}}, ::nvinfer1::Dims{2, {1, 1}}, ::nvinfer1::Dims{2, {2, 2}}, ::nvinfer1::Dims{2, {3, 3}});

// -----

func.func @trt_shuffle(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 3, 1, 2>,
    reshape = array<i64: 2, 3, 224, 224>,
    second_transpose = array<i64: 0, 1, 2, 3>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>
  return %0 : tensor<2x3x224x224xf32>
}

// CHECK-LABEL: @trt_shuffle_builder
// CHECK-LABEL: @trt_shuffle_tester

// CPP-LABEL: void trt_shuffle_builder(
//       CPP:   ::nvinfer1::ITensor* [[null:.+]] = nullptr;
//       CPP:   ::nvinfer1::adaptor::networkAddShuffle({{.+}}, {{.+}}, [[null]], ::nvinfer1::Permutation{{\{\{}}0, 3, 1, 2}}, ::nvinfer1::Dims{4, {2, 3, 224, 224}}, ::nvinfer1::Permutation{{\{\{}}0, 1, 2, 3}}, true);

// -----

#profile = #tensorrt.shape_profile<min=[1, 10], opt=[5, 10], max=[10, 10]>

func.func @trt_element_wise(%arg0: tensor<?x10xf32> {tensorrt.shape_profile = #profile}, %arg1: tensor<?x10xf32> {tensorrt.shape_profile = #profile}) -> tensor<?x10xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// CHECK-LABEL: @trt_element_wise_builder
// CHECK-LABEL: @trt_element_wise_tester

// CPP-LABEL: trt_element_wise_builder(
//       CPP: ::nvinfer1::ITensor* [[input0:.+]] = ::nvinfer1::adaptor::networkAddInput(v1, "input_0", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{2, {-1, 10}});
//       CPP: ::nvinfer1::ITensor* [[input1:.+]] = ::nvinfer1::adaptor::networkAddInput(v1, "input_1", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{2, {-1, 10}});
//       CPP: ::nvinfer1::adaptor::networkAddElementWise({{.+}}, [[input0]], [[input1]], ::nvinfer1::ElementWiseOperation::kSUM);

// -----

// CHECK-LABEL: @trt_shape_builder
// CHECK-LABEL: @trt_shape_tester
//   CPP-LABEL: trt_shape_builder(
//         CPP:   ::nvinfer1::ITensor* [[t3:.+]] = ::nvinfer1::adaptor::networkAddInput({{.+}}, "input_0", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{2, {10, -1}});
//         CPP:   ::nvinfer1::ITensor* [[t4:.+]] = ::nvinfer1::adaptor::networkAddReduce({{.+}}, [[t3]], false, 3, ::nvinfer1::ReduceOperation::kSUM);
//         CPP:   ::nvinfer1::ITensor* [[t5:.+]] = ::nvinfer1::adaptor::networkAddShape({{.+}}, [[t4]]);

func.func @trt_shape(
    %arg0: tensor<10x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<
      min=[10, 128], opt=[10, 256], max=[10, 512]>
    }) -> tensor<0xi32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 0, 1>} : tensor<10x?xf32> -> tensor<f32>
  %1 = tensorrt.shape %0 : tensor<f32> -> tensor<0xi32>
  return %1 : tensor<0xi32>
}

// -----

func.func @trt_topk(%arg0: tensor<10x20xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 10 : i64,
    axis = 1 : i64
  } %arg0 : tensor<10x20xf32> -> tensor<10x10xf32>, tensor<10x10xi32>
  %2 = tensorrt.element_wise<kSUM>(%0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>
}

// CHECK-LABLEL: @trt_topk_builder
//        CHECK: %[[v0:.+]] = emitc.call_opaque "::nvinfer1::adaptor::networkAddInput"({{.*}}) {args = [0 : index, #emitc.opaque<"\22input_0\22">, #emitc.opaque<"::nvinfer1::DataType::kFLOAT">, #emitc.opaque<"::nvinfer1::Dims{2, {10, 20}}">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>) -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//        CHECK: %[[v1:.+]] = emitc.call_opaque "::nvinfer1::adaptor::networkAddInput"({{.*}}) {args = [0 : index, #emitc.opaque<"\22input_1\22">, #emitc.opaque<"::nvinfer1::DataType::kFLOAT">, #emitc.opaque<"::nvinfer1::Dims{2, {10, 10}}">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>) -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//        CHECK: %[[v2:.+]]:2 = emitc.call_opaque "::nvinfer1::adaptor::networkAddTopK"({{.*}}, %[[v0]]) {args = [0 : index, 1 : index, 10, 2 : ui32, #emitc.opaque<"::nvinfer1::TopKOperation::kMAX">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>) -> (!emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>)
//        CHECK: %[[v3:.+]] = emitc.call_opaque "::nvinfer1::adaptor::networkAddElementWise"({{.*}}, %[[v2]]#0, %[[v1]]) {args = [0 : index, 1 : index, 2 : index, #emitc.opaque<"::nvinfer1::ElementWiseOperation::kSUM">]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>) -> !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>
//        CHECK: emitc.call_opaque "::nvinfer1::adaptor::networkMarkOutput"({{.*}}, %[[v3]]) {args = [0 : index, 1 : index]} : (!emitc.ptr<!emitc.opaque<"::nvinfer1::INetworkDefinition">>, !emitc.ptr<!emitc.opaque<"::nvinfer1::ITensor">>) -> ()

//  CPP-LABEL: void trt_topk_builder
//   CPP-SAME: (::nvinfer1::INetworkDefinition* [[v1:.+]], std::unordered_map<const char*, std::vector<uint8_t>>& [[v2:.+]]) {
//    CPP-DAG: ::nvinfer1::ITensor* [[v3:.+]] = ::nvinfer1::adaptor::networkAddInput([[v1]], "input_0", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{2, {10, 20}});
//    CPP-DAG: ::nvinfer1::ITensor* [[v4:.+]] = ::nvinfer1::adaptor::networkAddInput([[v1]], "input_1", ::nvinfer1::DataType::kFLOAT, ::nvinfer1::Dims{2, {10, 10}});
//    CPP-DAG: ::nvinfer1::ITensor* [[v5:.+]];
//    CPP-DAG: ::nvinfer1::ITensor* [[v6:.+]];
//        CPP: std::tie([[v5]], [[v6]]) = ::nvinfer1::adaptor::networkAddTopK([[v1]], [[v3]], 10, 2, ::nvinfer1::TopKOperation::kMAX);
//        CPP: ::nvinfer1::ITensor* [[v7:.+]] = ::nvinfer1::adaptor::networkAddElementWise([[v1]], [[v5]], [[v4]], ::nvinfer1::ElementWiseOperation::kSUM);
//        CPP: ::nvinfer1::adaptor::networkMarkOutput([[v1]], [[v7]]);