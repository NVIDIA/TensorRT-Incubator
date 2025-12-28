// RUN: rm -rf %t && mkdir -p %t
// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc --executor-serialize-artifacts="artifacts-directory=%t" %s | \
// RUN: tee %t/out.mlir | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP
// RUN: test -f %t/global_test/gv3.bin
// RUN: FileCheck --check-prefix=CHECK %s < %t/out.mlir

module @global_test {

memref.global @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>
memref.global @gv3 : memref<256xf32> = dense<0.0>


func.func @get_global() {
  %2 = memref.get_global @gv2 : memref<2x3xf32>
  %3 = memref.get_global @gv3 : memref<256xf32>
  return
}

}

//       CHECK:   emitc.global @gv2 : !emitc.array<2x3xf32> = dense<{{.*}}>
//       CHECK:   emitc.global @gv3 : !emitc.ptr<!emitc.opaque<"void">>
// CHECK-LABEL: func.func @get_global
//       CHECK:     %[[v0:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
//       CHECK:     %[[v1:.+]] = emitc.get_global @gv2 : !emitc.array<2x3xf32>
//       CHECK:     %[[v2:.+]] = emitc.call_opaque "mtrt::make_memref_descriptor"
//       CHECK:     %[[v3:.+]] = emitc.get_global @gv3
//       CHECK:     %[[v4:.+]] = emitc.load %[[v3]]
//       CHECK:     %[[v5:.+]] = emitc.call_opaque "mtrt::make_memref_descriptor"
//       CHECK:     return

// CHECK-LABEL: emitc.func @global_test_gv3_initialize
//       CHECK:     %[[c1:.+]] = "emitc.constant"() <{value = 1 : i32}> : () -> i32
//       CHECK:     %[[v0:.+]] = "emitc.constant"() <{value = #emitc.opaque<"\22global_test/gv3.bin\22">}>
//       CHECK:     %[[v1:.+]] = "emitc.constant"() <{value = 16 : i32}> : () -> i32
//       CHECK:     %[[v2:.+]] = get_global @gv3 :
//       CHECK:     %[[v3:.+]] = call_opaque "mtrt::constant_load_from_file"(%[[v0]], %[[v1]], %[[c1]])
//       CHECK:     assign %[[v3]] : !emitc.ptr<!emitc.opaque<"void">> to %[[v2]]
//       CHECK:     return
// CHECK-LABEL: emitc.func @global_test_gv3_destroy
//       CHECK:     %[[space:.+]] = "emitc.constant"
//       CHECK:     %[[v0:.+]] = get_global @gv3 :
//       CHECK:     %[[v1:.+]] = load %[[v0]] :
//       CHECK:     call_opaque "mtrt::constant_destroy"(%[[v1]], %[[space]])

// CPP: float gv2[2][3] = {
// CPP: void* gv3;
// CPP: void get_global() {
// CPP:   int32_t v1 = 0;
// CPP:   mtrt::RankedMemRef<2> v2 = mtrt::make_memref_descriptor<2>(gv2, gv2, v1, 2, 3, 3, 1);
// CPP:   void* v3 = gv3;
// CPP:   mtrt::RankedMemRef<1> v4 = mtrt::make_memref_descriptor<1>(v3, v3, v1, 256, 1);
// CPP:   return;

// CPP: void global_test_gv3_initialize() {
// CPP:   int32_t v1 = 1;
// CPP:   const char* v2 = "global_test/gv3.bin";
// CPP:   int32_t v3 = 16;
// CPP:   void* v4 = mtrt::constant_load_from_file(v2, v3, v1);
// CPP:   gv3 = v4;
// CPP:   return;

// CPP: void global_test_gv3_destroy() {
// CPP:   int32_t v1 = 1;
// CPP:   void* v2 = gv3;
// CPP:   mtrt::constant_destroy(v2, v1);
// CPP:   return;

// -----

func.func @extract_strided_metadata(
    %ref: memref<?x?xf32, strided<[?,?], offset: ?>>) {

  %base, %offset, %sizes:2, %strides:2 =
    memref.extract_strided_metadata %ref : memref<?x?xf32, strided<[?,?], offset: ?>>
    -> memref<f32>, index,
       index, index,
       index, index

  return
}

// CHECK-LABEL: func.func @extract_strided_metadata
//  CHECK-SAME: (%[[arg0:.+]]: !emitc.opaque<"mtrt::RankedMemRef<2>">) {
//   CHECK-DAG:     %[[v0:.+]] = "emitc.constant"() <{value = 1 : i32}> : () -> i32
//   CHECK-DAG:     %[[v1:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
//  CHECK-NEXT:     %[[v2:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_allocated_ptr"(%[[arg0]]
//  CHECK-NEXT:     %[[v3:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_aligned_ptr"(%[[arg0]])
//  CHECK-NEXT:     %[[v4:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_offset"(%[[arg0]])
//  CHECK-NEXT:     %[[v5:.+]] = emitc.call_opaque "mtrt::make_memref_descriptor"(%[[v2]], %[[v3]], %[[v1]])
//  CHECK-NEXT:     %[[v6:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_dim_size"(%[[arg0]], %[[v1]])
//  CHECK-NEXT:     %[[v7:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_dim_size"(%[[arg0]], %[[v0]])
//  CHECK-NEXT:     %[[v8:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_stride"(%[[arg0]], %[[v1]])
//  CHECK-NEXT:     %[[v9:.+]] = emitc.call_opaque "mtrt::memref_descriptor_get_stride"(%[[arg0]], %[[v0]])
//  CHECK-NEXT:     return

// CPP-LABEL: void extract_strided_metadata(mtrt::RankedMemRef<2> v1) {
//  CPP-NEXT:   int32_t v2 = 1;
//  CPP-NEXT:   int32_t v3 = 0;
//  CPP-NEXT:   void* v4 = mtrt::memref_descriptor_get_allocated_ptr(v1);
//  CPP-NEXT:   void* v5 = mtrt::memref_descriptor_get_aligned_ptr(v1);
//  CPP-NEXT:   int64_t v6 = mtrt::memref_descriptor_get_offset(v1);
//  CPP-NEXT:   mtrt::RankedMemRef<0> v7 = mtrt::make_memref_descriptor<0>(v4, v5, v3);
//  CPP-NEXT:   int64_t v8 = mtrt::memref_descriptor_get_dim_size(v1, v3);
//  CPP-NEXT:   int64_t v9 = mtrt::memref_descriptor_get_dim_size(v1, v2);
//  CPP-NEXT:   int64_t v10 = mtrt::memref_descriptor_get_stride(v1, v3);
//  CPP-NEXT:   int64_t v11 = mtrt::memref_descriptor_get_stride(v1, v2);
//  CPP-NEXT:   return;

// -----

func.func @zero_d_alloc() -> memref<f32> {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// CPP-LABEL: mtrt::RankedMemRef<0> zero_d_alloc() {
// CPP-NEXT:   int32_t v1 = 0;
// CPP-NEXT:   int32_t v2 = 16;
// CPP-NEXT:   int64_t v3 = 1;
// CPP-NEXT:   int32_t v4 = 4;
// CPP-NEXT:   int64_t v5 = v4 * v3;
// CPP-NEXT:   void* v6 = mtrt::host_aligned_alloc(v5, v2);
// CPP-NEXT:   mtrt::RankedMemRef<0> v7 = mtrt::make_memref_descriptor<0>(v6, v6, v1);
// CPP-NEXT:   return v7;

// -----

func.func @dealloc(%arg0: memref<f32>) {
  memref.dealloc %arg0 : memref<f32>
  return
}

// CPP-LABEL: void dealloc(mtrt::RankedMemRef<0> v1) {
// CPP-NEXT:   void* v2 = mtrt::memref_descriptor_get_allocated_ptr(v1);
// CPP-NEXT:   mtrt::host_free(v2);
// CPP-NEXT:   return;

// -----

func.func @static_load(%static : memref<10x42xf32>, %i : index, %j : index) -> f32 {
  %0 = memref.load %static[%i, %j] : memref<10x42xf32>
  return %0 : f32
}

// CPP-LABEL: float static_load(mtrt::RankedMemRef<2> v1, size_t v2, size_t v3)
// CPP-NEXT:   int64_t v4 = 42;
// CPP-NEXT:   int64_t v5 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   size_t v6 = v2 * v4;
// CPP-NEXT:   int64_t v7 = v5 + v6;
// CPP-NEXT:   int64_t v8 = v7 + v3;
// CPP-NEXT:   void* v9 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   float* v10 = (float*) v9;
// CPP-NEXT:   float v11 = v10[v8];
// CPP-NEXT:   return v11;

// -----

func.func @static_store(%static : memref<10x42xf32>, %i : index, %j : index, %val : f32) {
  memref.store %val, %static[%i, %j] : memref<10x42xf32>
  return
}

// CPP-LABEL: void static_store(mtrt::RankedMemRef<2> v1, size_t v2, size_t v3, float v4) {
// CPP-NEXT:   int64_t v5 = 42;
// CPP-NEXT:   int64_t v6 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   size_t v7 = v2 * v5;
// CPP-NEXT:   int64_t v8 = v6 + v7;
// CPP-NEXT:   int64_t v9 = v8 + v3;
// CPP-NEXT:   void* v10 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   float* v11 = (float*) v10;
// CPP-NEXT:   v11[v9] = v4;
// CPP-NEXT:   return;

// -----

func.func @zero_d_store(%arg0: memref<f32>, %arg1: f32) {
  memref.store %arg1, %arg0[] : memref<f32>
  return
}

// CPP-LABEL: void zero_d_store(mtrt::RankedMemRef<0> v1, float v2) {
// CPP-NEXT:   int64_t v3 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   void* v4 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   float* v5 = (float*) v4;
// CPP-NEXT:   v5[v3] = v2;
// CPP-NEXT:   return;

// -----

func.func @extract_aligned_pointer_as_index(%m: memref<?xf32>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<?xf32> -> index
  return %0: index
}

// CPP-LABEL: size_t extract_aligned_pointer_as_index(mtrt::RankedMemRef<1> v1)
// CPP-NEXT:   void* v2 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   size_t v3 = (size_t) v2;
// CPP-NEXT:   return v3;


// -----

func.func @alloc_of_index() -> memref<42xindex> {
  %0 = memref.alloc() : memref<42xindex>
  return %0 : memref<42xindex>
}


// CPP-LABEL: mtrt::RankedMemRef<1> alloc_of_index()
// CPP-DAG: int32_t [[v1:.+]] = 0;
// CPP-DAG: int32_t [[v2:.+]] = 16;
// CPP-DAG: int64_t [[v3:.+]] = 42;
// CPP-DAG: int64_t [[v4:.+]] = 1;
// CPP-DAG: int32_t [[v5:.+]] = 8;
// CPP-DAG: int64_t [[v6:.+]] = [[v5]] * [[v3]];
// CPP-DAG: void* [[v7:.+]] = mtrt::host_aligned_alloc([[v6]], [[v2]]);
// CPP-DAG: mtrt::RankedMemRef<1> [[v8:.+]] = mtrt::make_memref_descriptor<1>([[v7]], [[v7]], [[v1]], [[v3]], [[v4]]);
// CPP-DAG: return [[v8]];

// -----

func.func @memref_copy(%src: memref<10x20xf32>, %dst: memref<10x20xf32>) {
  memref.copy %src, %dst : memref<10x20xf32> to memref<10x20xf32>
  return
}

// CPP-LABEL: void memref_copy(mtrt::RankedMemRef<2> v1, mtrt::RankedMemRef<2> v2)
// CPP-DAG: void* [[srcAligned:.+]] = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-DAG: int8_t* [[srcI8:.+]] = (int8_t*) [[srcAligned]];
// CPP-DAG: int64_t [[srcOffset:.+]] = mtrt::memref_descriptor_get_offset(v1);
// CPP-DAG: void* [[dstAligned:.+]] = mtrt::memref_descriptor_get_aligned_ptr(v2);
// CPP-DAG: int8_t* [[dstI8:.+]] = (int8_t*) [[dstAligned]];
// CPP-DAG: int64_t [[dstOffset:.+]] = mtrt::memref_descriptor_get_offset(v2);
// CPP: std::memcpy(
// CPP: return;

// -----

// Test that 0-rank memref.global is converted to a size-1 array in C++.

memref.global @scalar_global : memref<f32> = dense<42.0>

func.func @get_scalar_global() -> memref<f32> {
  %0 = memref.get_global @scalar_global : memref<f32>
  return %0 : memref<f32>
}

// CHECK: emitc.global @scalar_global : !emitc.array<1xf32> = dense<4.200000e+01>
// CHECK-LABEL: func.func @get_scalar_global
// CHECK:   %[[v0:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
// CHECK:   %[[v1:.+]] = emitc.get_global @scalar_global : !emitc.array<1xf32>
// CHECK:   %[[v2:.+]] = emitc.call_opaque "mtrt::make_memref_descriptor"(%[[v1]], %[[v1]], %[[v0]])
// CHECK:   return %[[v2]]

// CPP: float scalar_global[1] = {4.2{{0*}}e+01f};
// CPP-LABEL: mtrt::RankedMemRef<0> get_scalar_global()
// CPP:   int32_t [[v0:.+]] = 0;
// CPP:   mtrt::RankedMemRef<0> [[v1:.+]] = mtrt::make_memref_descriptor<0>(scalar_global, scalar_global, [[v0]]);
// CPP:   return [[v1]];
