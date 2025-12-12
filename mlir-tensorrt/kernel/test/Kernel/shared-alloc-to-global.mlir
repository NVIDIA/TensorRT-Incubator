// RUN:  kernel-opt %s -kernel-shared-alloc-to-global | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 + 32)>
#map2 = affine_map<()[s0] -> (s0 + 64)>
#map3 = affine_map<()[s0] -> (s0 + 96)>
gpu.module @kernels {
  func.func @kernel1(%arg0: memref<128xf32, strided<[?], offset: ?>>, %arg1: memref<128xf32, strided<[?], offset: ?>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = gpu.thread_id  x
    %1 = affine.apply #map()[%0]
    %2 = vector.transfer_read %arg0[%1], %cst {in_bounds = [true]} : memref<128xf32, strided<[?], offset: ?>>, vector<4xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32, #gpu.address_space<workgroup>>
    vector.transfer_write %2, %alloc[%1] {in_bounds = [true]} : vector<4xf32>, memref<128xf32, #gpu.address_space<workgroup>>
    %3 = affine.apply #map1()[%0]
    %4 = affine.apply #map2()[%0]
    %5 = affine.apply #map3()[%0]
    %6 = vector.transfer_read %alloc[%0], %cst {in_bounds = [true]} : memref<128xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %7 = vector.transfer_read %alloc[%3], %cst {in_bounds = [true]} : memref<128xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %8 = vector.transfer_read %alloc[%4], %cst {in_bounds = [true]} : memref<128xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %9 = vector.transfer_read %alloc[%5], %cst {in_bounds = [true]} : memref<128xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    vector.transfer_write %6, %arg1[%0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32, strided<[?], offset: ?>>
    vector.transfer_write %7, %arg1[%3] {in_bounds = [true]} : vector<1xf32>, memref<128xf32, strided<[?], offset: ?>>
    vector.transfer_write %8, %arg1[%4] {in_bounds = [true]} : vector<1xf32>, memref<128xf32, strided<[?], offset: ?>>
    vector.transfer_write %9, %arg1[%5] {in_bounds = [true]} : vector<1xf32>, memref<128xf32, strided<[?], offset: ?>>
    return
  }
}

//         CHECK: gpu.module @kernels
//         CHECK:     memref.global "private" @__shared_memory__
//    CHECK-SAME:        memref<128xf32, #gpu.address_space<workgroup>>
//    CHECK-SAME:        alignment = 64
//   CHECK-LABEL: @kernel1
//    CHECK-SAME: (%[[arg0:.+]]: memref<128xf32, strided<[?], offset: ?>>, %[[arg1:.+]]: memref<128xf32, strided<[?], offset: ?>>) {
//         CHECK:       %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//         CHECK:       %[[v0:.+]] = memref.get_global @__shared_memory__ : memref<128xf32, #gpu.address_space<workgroup>>
//         CHECK:      vector.transfer_write %{{.+}}, %[[v0]][
// CHECK-COUNT-4:      vector.transfer_read %[[v0]]
