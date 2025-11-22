// REQUIRES: host-has-at-least-1-gpus
// RUN: executor-translate -mlir-to-runtime-executable %s | executor-runner -dump-data-segments -input-type=rtexe | FileCheck %s

executor.data_segment @dense_resource_f32 constant address_space <host> dense_resource<test.float32>  : tensor<1xf32>
executor.data_segment @dense_f32 constant  dense<[0x12345678, 0xABCDEF01]> : tensor<2xf32>
executor.data_segment @dense_i4  constant dense<[0x1, 0x2]> : tensor<2xi4>
executor.data_segment @bool_constant constant dense<[true, false, false, true]> : tensor<4xi1>
executor.data_segment @splat_bool_true constant dense<true> : tensor<4xi1>
executor.data_segment @splat_bool_false constant dense<false> : tensor<4xi1>
executor.data_segment @splat_i4 constant align 2 dense<0x3> : tensor<4xi4>
executor.data_segment @splat_i8 constant dense<0x3F> : tensor<4xi8>
executor.data_segment @splat_i16 constant dense<0x3FF> : tensor<4xi16>
executor.data_segment @splat_i32 constant dense<0x3FFF> : tensor<4xi32>
executor.data_segment @splat_i64 dense<0x010203FFFFFFF> : tensor<4xi64>
executor.data_segment @complex_f32 constant dense<[(0x12345678, 0x87654321), (0x87654321, 0x0)]> : tensor<2xcomplex<f32>>
executor.data_segment @uninit_f32 uninitialized dense<0.0> : tensor<128x2xf32>
executor.data_segment @uninit_f32_device uninitialized align 16 address_space<device> dense<0.0> : tensor<128x2xf32>


{-#
  dialect_resources: {
    builtin: {
      test.float32: "0x0400000012345678"
    }
  }
#-}

// CHECK-LABEL: DataSegment<dense_resource_f32, size=4, alignment=4, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x12, 0x34, 0x56, 0x78]
// CHECK-LABEL: DataSegment<dense_f32, size=8, alignment=4, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x78, 0x56, 0x34, 0x12, 0x1, 0xef, 0xcd, 0xab]
// CHECK-LABEL: DataSegment<dense_i4, size=2, alignment=1, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x1, 0x2]
// CHECK-LABEL: DataSegment<bool_constant, size=4, alignment=1, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x1, 0x0, 0x0, 0x1]
// CHECK-LABEL: DataSegment<splat_bool_true, size=4, alignment=1, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x1, 0x1, 0x1, 0x1]
// CHECK-LABEL: DataSegment<splat_bool_false, size=4, alignment=1, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x0, 0x0, 0x0, 0x0]
// CHECK-LABEL: DataSegment<splat_i4, size=4, alignment=2, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x3, 0x3, 0x3, 0x3]
// CHECK-LABEL: DataSegment<splat_i8, size=4, alignment=1, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x3f, 0x3f, 0x3f, 0x3f]
// CHECK-LABEL: DataSegment<splat_i16, size=8, alignment=2, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0xff, 0x3, 0xff, 0x3, 0xff, 0x3, 0xff, 0x3]
// CHECK-LABEL: DataSegment<splat_i32, size=16, alignment=4, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0xff, 0x3f, 0x0, 0x0, 0xff, 0x3f, 0x0, 0x0, 0xff, 0x3f, 0x0, 0x0, 0xff, 0x3f, 0x0, 0x0]
// CHECK-LABEL: DataSegment<splat_i64, size=32, alignment=4, constant=false, uninitialized=false, address_space=host>
// CHECK: Data: [0xff, 0xff, 0xff, 0x3f, 0x20, 0x10, 0x0, 0x0, 0xff, 0xff, 0xff, 0x3f, 0x20, 0x10, 0x0, 0x0, 0xff, 0xff, 0xff, 0x3f, 0x20, 0x10, 0x0, 0x0, 0xff, 0xff, 0xff, 0x3f, 0x20, 0x10, 0x0, 0x0]
// CHECK-LABEL: DataSegment<complex_f32, size=16, alignment=4, constant=true, uninitialized=false, address_space=host>
// CHECK: Data: [0x78, 0x56, 0x34, 0x12, 0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87, 0x0, 0x0, 0x0, 0x0]
// CHECK-LABEL: DataSegment<uninit_f32, size=1024, alignment=4, constant=false, uninitialized=true, address_space=host>
// CHECK-LABEL: DataSegment<uninit_f32_device, size=1024, alignment=16, constant=false, uninitialized=true, address_space=device>
