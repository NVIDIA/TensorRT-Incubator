// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

func.func @test_opaque_v2_plugin_field_serialization(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.opaque_plugin {
    dso_path = "libTensorRTTestPlugins.so",
    plugin_name = "TestV2Plugin1",
    plugin_version = "0",
    plugin_namespace = "",
    creator_func = "getTestV2Plugin1Creator",
    creator_params = {
        i32_param = 10 : i32,
        i16_param = 20 : i16,
        i64_param = 31 : i64,
        i8_param  = 40 : i8,
        shape_param = array<i64: 1, 2, 3>
    }} (%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}


// CHECK-LABEL: Created TestV2Plugin1 with 4 fields:
// CHECK-NEXT: field[0] name=i32_param, type=kINT32, length=1, data=[10]
// CHECK-NEXT: field[1] name=i16_param, type=kINT16, length=1, data=[20]
// CHECK-NEXT: field[2] name=i8_param, type=kINT8, length=1, data=[40]
// CHECK-NEXT: field[3] name=shape_param, type=kDIMS, length=1, data=[ Dims<1x2x3> ]


// -----

func.func @test_opaque_v2_plugin_field_creation_using_registry_and_dso(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.opaque_plugin {
    dso_path = "libTensorRTTestPlugins.so",
    plugin_name = "TestV2Plugin1",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {
        i32_param = 10 : i32,
        i16_param = 20 : i16,
        i64_param = 32 : i64,
        i8_param  = 40 : i8,
        shape_param = array<i64: 1, 2, 3>
    }} (%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: Created TestV2Plugin1 with 4 fields:
// CHECK-NEXT: field[0] name=i32_param, type=kINT32, length=1, data=[10]
// CHECK-NEXT: field[1] name=i16_param, type=kINT16, length=1, data=[20]
// CHECK-NEXT: field[2] name=i8_param, type=kINT8, length=1, data=[40]
// CHECK-NEXT: field[3] name=shape_param, type=kDIMS, length=1, data=[ Dims<1x2x3> ]

// -----

func.func @test_opaque_v2_plugin_field_creation_using_registry(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.opaque_plugin {
    plugin_name = "TestV2Plugin1",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {
        i32_param = 10 : i32,
        i16_param = 20 : i16,
        i64_param = 33 : i64,
        i8_param  = 40 : i8,
        shape_param = array<i64: 1, 2, 3>
    }} (%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: Created TestV2Plugin1 with 4 fields:
// CHECK-NEXT: field[0] name=i32_param, type=kINT32, length=1, data=[10]
// CHECK-NEXT: field[1] name=i16_param, type=kINT16, length=1, data=[20]
// CHECK-NEXT: field[2] name=i8_param, type=kINT8, length=1, data=[40]
// CHECK-NEXT: field[3] name=shape_param, type=kDIMS, length=1, data=[ Dims<1x2x3> ]
