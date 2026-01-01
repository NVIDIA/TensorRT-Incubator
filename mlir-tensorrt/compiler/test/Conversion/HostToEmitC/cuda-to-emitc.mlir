// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc -executor-serialize-artifacts="artifacts-directory=%t create-manifest=true" %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

#host_pinned = #plan.memory_space<host_pinned>
func.func @zero_d_alloc() -> memref<1xf32, #host_pinned> {
  %0 = cuda.alloc() : memref<1xf32, #host_pinned>
  return %0 : memref<1xf32, #host_pinned>
}

// CPP-LABEL: mtrt::RankedMemRef<1> zero_d_alloc() {
// CPP:   void* [[ptr:.+]] = nullptr;
// CPP:   void** [[ptrAddr:.+]] = &[[ptr]];
// CPP:   int32_t [[st:.+]] = mtrt::cuda_alloc({{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ptrAddr]]);
// CPP:   mtrt::abort_on_error([[st]]);
// CPP:   mtrt::RankedMemRef<1> [[desc:.+]] = mtrt::make_memref_descriptor<1>({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}});
// CPP:   return [[desc]];

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<device>>
func.func @device_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CPP-LABEL: void device_free(CUstream v1, mtrt::RankedMemRef<3> v2)
// CPP-NEXT:   int8_t v3 = 0;
// CPP-NEXT:   void* v4 = mtrt::memref_descriptor_get_allocated_ptr(v2);
// CPP:        int32_t {{.*}} = mtrt::cuda_free(v1, v4, v3, v3);
// CPP:        mtrt::abort_on_error({{.*}});
// CPP-NEXT:   return;

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<host_pinned>>
func.func @free_host_pinned(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CPP-LABEL: void free_host_pinned(CUstream v1, mtrt::RankedMemRef<3> v2) {
// CPP-NEXT:   int8_t v3 = 1;
// CPP-NEXT:   int8_t v4 = 0;
// CPP-NEXT:   void* v5 = mtrt::memref_descriptor_get_allocated_ptr(v2);
// CPP:        int32_t {{.*}} = mtrt::cuda_free(v1, v5, v3, v4);
// CPP:        mtrt::abort_on_error({{.*}});

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<unified>>
func.func @free_unified(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CPP-LABEL: void free_unified(CUstream v1, mtrt::RankedMemRef<3> v2)
// CPP-NEXT: int8_t v3 = 0;
// CPP-NEXT: int8_t v4 = 1;
// CPP-NEXT: void* v5 = mtrt::memref_descriptor_get_allocated_ptr(v2);
// CPP:      int32_t {{.*}} = mtrt::cuda_free(v1, v5, v3, v4);
// CPP:      mtrt::abort_on_error({{.*}});
// CPP-NEXT: return;

// -----

#device_space = #plan.memory_space<device>
#host_space = #plan.memory_space<host>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @copy_d2d(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_d2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

//CPP-LABEL: void copy_d2d(mtrt::RankedMemRef<3> v1, mtrt::RankedMemRef<3> v2, CUstream v3)
// CPP-NEXT:   int64_t v4 = 4;
// CPP-NEXT:   int32_t v5 = 2;
// CPP-NEXT:   int32_t v6 = 1;
// CPP-NEXT:   int32_t v7 = 0;
// CPP-NEXT:   int64_t v8 = 1;
// CPP-NEXT:   int32_t v9 = 4;
// CPP-NEXT:   void* v10 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   int8_t* v11 = (int8_t*) v10;
// CPP-NEXT:   int64_t v12 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   int64_t v13 = v12 * v9;
// CPP-NEXT:   void* v14 = v11 + v13;
// CPP-NEXT:   void* v15 = mtrt::memref_descriptor_get_aligned_ptr(v2);
// CPP-NEXT:   int8_t* v16 = (int8_t*) v15;
// CPP-NEXT:   int64_t v17 = mtrt::memref_descriptor_get_offset(v2);
// CPP-NEXT:   int64_t v18 = v17 * v9;
// CPP-NEXT:   void* v19 = v16 + v18;
// CPP-NEXT:   int64_t v20 = mtrt::memref_descriptor_get_dim_size(v1, v7);
// CPP-NEXT:   int64_t v21 = v8 * v20;
// CPP-NEXT:   int64_t v22 = mtrt::memref_descriptor_get_dim_size(v1, v6);
// CPP-NEXT:   int64_t v23 = v21 * v22;
// CPP-NEXT:   int64_t v24 = mtrt::memref_descriptor_get_dim_size(v1, v5);
// CPP-NEXT:   int64_t v25 = v23 * v24;
// CPP-NEXT:   int64_t v26 = v25 * v4;
// CPP:        int32_t {{.*}} = mtrt::cuda_copy(v3, v14, v19, v26);
// CPP:        mtrt::abort_on_error({{.*}});
// CPP-NEXT:   return;

// -----

func.func @copy_d2h_offset(%arg0: memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>>,
                           %arg1: memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>>,
                           %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 :
    memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>>
    to memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>>
  return
}

// CPP-LABEL: void copy_d2h_offset(mtrt::RankedMemRef<2> v1, mtrt::RankedMemRef<2> v2, CUstream v3) {
// CPP-NEXT:   int64_t v4 = 4;
// CPP-NEXT:   int64_t v5 = 2048;
// CPP-NEXT:   int32_t v6 = 4;
// CPP-NEXT:   void* v7 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   int8_t* v8 = (int8_t*) v7;
// CPP-NEXT:   int64_t v9 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   int64_t v10 = v9 * v6;
// CPP-NEXT:   void* v11 = v8 + v10;
// CPP-NEXT:   void* v12 = mtrt::memref_descriptor_get_aligned_ptr(v2);
// CPP-NEXT:   int8_t* v13 = (int8_t*) v12;
// CPP-NEXT:   int64_t v14 = mtrt::memref_descriptor_get_offset(v2);
// CPP-NEXT:   int64_t v15 = v14 * v6;
// CPP-NEXT:   void* v16 = v13 + v15;
// CPP-NEXT:   int64_t v17 = v5 * v4;
// CPP:        int32_t {{.*}} = mtrt::cuda_copy(v3, v11, v16, v17);
// CPP:        mtrt::abort_on_error({{.*}});
// CPP-NEXT:   return;

// -----

!srcType = memref<6xf32, strided<[2], offset: 2>, #plan.memory_space<device>>
!dstType = memref<6xf32, strided<[2], offset: 4>, #plan.memory_space<host>>

func.func @copy_d2h_strided(%arg0: !srcType,
                           %arg1: !dstType, %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CPP-LABEL: void copy_d2h_strided(mtrt::RankedMemRef<1> v1, mtrt::RankedMemRef<1> v2, CUstream v3)
// CPP-NEXT:   int32_t [[elemSize:.+]] = 4;
// CPP-NEXT:   void* [[srcAligned:.+]] = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   int8_t* [[srcI8:.+]] = (int8_t*) [[srcAligned]];
// CPP-NEXT:   int64_t [[srcOffset:.+]] = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   int64_t [[srcByteOffset:.+]] = [[srcOffset]] * [[elemSize]];
// CPP-NEXT:   void* [[srcPtr:.+]] = [[srcI8]] + [[srcByteOffset]];
// CPP-NEXT:   void* [[dstAligned:.+]] = mtrt::memref_descriptor_get_aligned_ptr(v2);
// CPP-NEXT:   int8_t* [[dstI8:.+]] = (int8_t*) [[dstAligned]];
// CPP-NEXT:   int64_t [[dstOffset:.+]] = mtrt::memref_descriptor_get_offset(v2);
// CPP-NEXT:   int64_t [[dstByteOffset:.+]] = [[dstOffset]] * [[elemSize]];
// CPP-NEXT:   void* [[dstPtr:.+]] = [[dstI8]] + [[dstByteOffset]];
// CPP-NEXT:   mtrt::UnrankedMemRef [[srcUnranked:.+]] = mtrt::make_unranked_descriptor(1, v1);
// CPP-NEXT:   mtrt::UnrankedMemRef [[dstUnranked:.+]] = mtrt::make_unranked_descriptor(1, v2);
// CPP-NEXT:   mtrt::cuda_copy_strided(v3, [[srcPtr]], [[srcUnranked]], [[dstPtr]], [[dstUnranked]]);
// CPP-NEXT:   return;

// -----

!srcType = memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #plan.memory_space<device>>
!dstType = memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #plan.memory_space<host>>

func.func @memref_copy_contiguous_non_identity(%arg0: !srcType, %arg1: !dstType,
    %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CPP-LABEL: void memref_copy_contiguous_non_identity(mtrt::RankedMemRef<3> v1, mtrt::RankedMemRef<3> v2, CUstream v3) {
// CPP-NEXT:   int64_t v4 = 4;
// CPP-NEXT:   int64_t v5 = 32;
// CPP-NEXT:   int32_t v6 = 4;
// CPP-NEXT:   void* v7 = mtrt::memref_descriptor_get_aligned_ptr(v1);
// CPP-NEXT:   int8_t* v8 = (int8_t*) v7;
// CPP-NEXT:   int64_t v9 = mtrt::memref_descriptor_get_offset(v1);
// CPP-NEXT:   int64_t v10 = v9 * v6;
// CPP-NEXT:   void* v11 = v8 + v10;
// CPP-NEXT:   void* v12 = mtrt::memref_descriptor_get_aligned_ptr(v2);
// CPP-NEXT:   int8_t* v13 = (int8_t*) v12;
// CPP-NEXT:   int64_t v14 = mtrt::memref_descriptor_get_offset(v2);
// CPP-NEXT:   int64_t v15 = v14 * v6;
// CPP-NEXT:   void* v16 = v13 + v15;
// CPP-NEXT:   int64_t v17 = v5 * v4;
// CPP:        int32_t {{.*}} = mtrt::cuda_copy(v3, v11, v16, v17);
// CPP:        mtrt::abort_on_error({{.*}});

// -----

cuda.compiled_module @kernels dense<[0xFF,0x00]> : vector<2xi8>

func.func @test_get_func() -> !cuda.function {
  %func = cuda.get_function "kernel"from @kernels
  return %func: !cuda.function
}

// Verify cuda.compiled_module lowering creates a CUmodule global, initializes it
// from a staged PTX file, and resolves CUfunction symbols.
// CPP: static CUmodule unnamed_module_kernels_cumodule;
// CPP: static CUfunction unnamed_module_kernels_cumodule_kernel;

func.func @test_launch(%arg0: memref<4xf32>, %arg1: index, %arg2: i32, %arg3: f32) {
  %device = arith.constant 0 : i32
  %stream = cuda.get_global_stream device(%device)[0]
  %func = cuda.get_function "kernel"from @kernels
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  cuda.launch %func(%arg0, %arg1, %arg2, %arg3 : memref<4xf32>, index, i32, f32) with
    grid(%c1, %c1, %c1)
    block(%c1, %c1, %c1)
    smem(%c0) stream(%stream)
  return
}

// Verify cuda.launch lowering uses argv packing (void* argv[]) and calls the
// runtime wrapper.
// CPP-LABEL: void test_launch(
// CPP: void* {{.*}}[4];
// CPP: mtrt::cuda_launch_kernel(

// CPP-LABEL: int32_t unnamed_module_kernels_initialize() {
// CPP: const char* {{.*}} = "kernel";
// CPP: const char* {{.*}} = "unnamed_module/kernels.ptx";
// CPP: mtrt::cuda_module_create_from_ptx_file
// CPP: mtrt::cuda_module_get_func
// CPP-LABEL: void unnamed_module_kernels_destroy() {
// CPP: mtrt::cuda_module_destroy
// CPP: mtrt::abort_on_error
