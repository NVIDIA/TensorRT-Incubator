// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

func.func @test_opaque_quick_plugin_field_creation_using_registry(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.opaque_plugin {
    dso_path = "TensorRTTestPlugins.so",
    plugin_name = "TestQuickPlugin",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {
        i32_param = 10 : i32,
        i16_param = 20 : i16,
        i64_param = 33 : i64,
        i8_param  = 40 : i8,
        f32_param = 12345.6789 : f32,
        f64_param = 12345.6789 : f64,
        f32_elements_param = dense<[1.5, 2.5, 3.1]> : tensor<3xf32>,
        f16_elements_param = dense<[1.5, 2.5, 3.1]> : tensor<3xf16>,
        string_param = "some_data",
        shape_param = array<i64: 1, 2, 3>,
        shape_vec_param = [array<i64: 1 ,2 , 3>, array<i64: 4, 5, 6>]
    }} (%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: QuickPluginCreationRequest: kSTRICT_AOT
//      CHECK: Created TestQuickPlugin with 11 fields:
// CHECK-NEXT: field[0] name=i64_param, type=kINT64, length=1, data=[33]
// CHECK-NEXT: field[1] name=i32_param, type=kINT32, length=1, data=[10]
// CHECK-NEXT: field[2] name=i16_param, type=kINT16, length=1, data=[20]
// CHECK-NEXT: field[3] name=i8_param, type=kINT8, length=1, data=[40]
// CHECK-NEXT: field[4] name=f32_param, type=kFLOAT32, length=1, data=[12345.6787]
// CHECK-NEXT: field[5] name=f64_param, type=kFLOAT64, length=1, data=[12345.678900000001]
// CHECK-NEXT: field[6] name=string_param, type=kCHAR, length=9, data=[some_data]
// CHECK-NEXT: field[7] name=shape_param, type=kDIMS, length=1, data=[ Dims<1x2x3> ]
// CHECK-NEXT: field[8] name=shape_vec_param, type=kDIMS, length=2, data=[ Dims<1x2x3>  Dims<4x5x6> ]
// CHECK-NEXT: field[9] name=f32_elements_param, type=kFLOAT32, length=3, data=[1.5, 2.5, 3.0999999]
// CHECK-NEXT: field[10] name=f16_elements_param, type=kFLOAT16, length=3, data=[1.5, 2.5, 3.0996]
