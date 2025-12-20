// RUN: kernel-opt %s -split-input-file -pass-pipeline="builtin.module(convert-vector-to-scf,gpu.module(kernel-expand-memref-args,kernel-lower-to-nvvm,reconcile-unrealized-casts))" | FileCheck %s

gpu.module @kernels {

  memref.global @shared_memory : memref<128xf16, 3>

  func.func @simple_copy_kernel(%arg0: memref<128xf16>, %arg1: memref<128xf16>) attributes {gpu.kernel} {
    %0 = gpu.thread_id x
    %1 = gpu.block_id x

    %shm = memref.get_global @shared_memory : memref<128xf16, 3>

    // Assuming 32 threads and 2 blocks.
    // This tests affine lowering.
    %idx0 = affine.apply affine_map<()[s0, s1]->(s0*64+s1)> ()[%0, %1]
    %idx1 = affine.apply affine_map<()[s0, s1]->(s0*64+s1+32)> ()[%0, %1]

    // Global to shared
    %ldg0 = memref.load %arg0[%idx0] : memref<128xf16>
    %ldg1 = memref.load %arg0[%idx1] : memref<128xf16>
    memref.store %ldg0, %shm[%idx0] : memref<128xf16, 3>
    memref.store %ldg1, %shm[%idx1] : memref<128xf16, 3>
    // Do a synchronization just for fun.
    gpu.barrier
    // Shared to global
    %lds0 = memref.load %shm[%idx0] : memref<128xf16, 3>
    %lds1 = memref.load %shm[%idx1] : memref<128xf16, 3>
    memref.store %lds0, %arg1[%idx0] : memref<128xf16>
    memref.store %lds1, %arg1[%idx1] : memref<128xf16>

    return
  }
}


// CHECK-LABEL: llvm.func @simple_copy_kernel
//  CHECK-SAME: (%[[arg0:.+]]: !llvm.ptr, %[[arg1:.+]]: !llvm.ptr) attributes {nvvm.kernel} {

// -----

gpu.module @kernels {
  func.func @simple_vector_copy_kernel(%arg0: memref<128xf16>, %arg1: memref<128xf16>) attributes {gpu.kernel} {
    %0 = gpu.thread_id x
    %idx0 = affine.apply affine_map<()[s0]->(s0*8)> ()[%0]
    %cst = arith.constant 0.0 : f16
    %a = vector.transfer_read %arg0[%idx0], %cst {in_bounds = [true]} : memref<128xf16>, vector<8xf16>
    vector.transfer_write %a, %arg1[%idx0] {in_bounds = [true]} : vector<8xf16>, memref<128xf16>
    return
  }
}

// CHECK-LABEL: llvm.func @simple_vector_copy_kernel
//  CHECK-SAME: (%[[arg0:.+]]: !llvm.ptr, %[[arg1:.+]]: !llvm.ptr) attributes {nvvm.kernel} {
//   CHECK-DAG:   %[[v0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v1:.+]] = llvm.insertvalue %[[arg0]], %[[v0]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v2:.+]] = llvm.insertvalue %[[arg0]], %[[v1]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v3:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v4:.+]] = llvm.insertvalue %[[v3]], %[[v2]][2] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v5:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v6:.+]] = llvm.insertvalue %[[arg1]], %[[v5]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v7:.+]] = llvm.insertvalue %[[arg1]], %[[v6]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v8:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v9:.+]] = llvm.insertvalue %[[v8]], %[[v7]][2] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v10:.+]] = llvm.mlir.constant(8 : index) : i64
//   CHECK-DAG:   %[[v11:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:   %[[v12:.+]] = llvm.extractvalue %[[v9]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v13:.+]] = llvm.extractvalue %[[v9]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v14:.+]] = llvm.insertvalue %[[v12]], %[[v11]][0]
//   CHECK-DAG:   %[[v15:.+]] = llvm.insertvalue %[[v13]], %[[v14]][1]
//   CHECK-DAG:   %[[v16:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v17:.+]] = llvm.insertvalue %[[v16]], %[[v15]][2]
//   CHECK-DAG:   %[[v18:.+]] = llvm.mlir.constant(128 : index) : i64
//   CHECK-DAG:   %[[v19:.+]] = llvm.insertvalue %[[v18]], %[[v17]][3, 0]
//   CHECK-DAG:   %[[v20:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:   %[[v21:.+]] = llvm.insertvalue %[[v20]], %[[v19]][4, 0]
//   CHECK-DAG:   %[[v22:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:   %[[v23:.+]] = llvm.extractvalue %[[v4]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v24:.+]] = llvm.extractvalue %[[v4]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v25:.+]] = llvm.insertvalue %[[v23]], %[[v22]][0]
//   CHECK-DAG:   %[[v26:.+]] = llvm.insertvalue %[[v24]], %[[v25]][1]
//   CHECK-DAG:   %[[v27:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v28:.+]] = llvm.insertvalue %[[v27]], %[[v26]][2]
//   CHECK-DAG:   %[[v29:.+]] = llvm.mlir.constant(128 : index) : i64
//   CHECK-DAG:   %[[v30:.+]] = llvm.insertvalue %[[v29]], %[[v28]][3, 0]
//   CHECK-DAG:   %[[v31:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:   %[[v32:.+]] = llvm.insertvalue %[[v31]], %[[v30]][4, 0]
//   CHECK-DAG:   %[[v33:.+]] = nvvm.read.ptx.sreg.tid.x : i32
//   CHECK-DAG:   %[[v34:.+]] = llvm.sext %[[v33]] : i32 to i64
//   CHECK-DAG:   %[[v35:.+]] = llvm.mul %[[v34]], %[[v10]] overflow<nsw> : i64
//   CHECK-DAG:   %[[v36:.+]] = llvm.extractvalue %[[v32]][1]
//   CHECK-DAG:   %[[v37:.+]] = llvm.getelementptr %[[v36]][%[[v35]]]
//   CHECK-DAG:   %[[v38:.+]] = llvm.load %[[v37]] {alignment = 2 : i64} : !llvm.ptr -> vector<8xf16>
//   CHECK-DAG:   %[[v39:.+]] = llvm.extractvalue %[[v21]][1]
//   CHECK-DAG:   %[[v40:.+]] = llvm.getelementptr %[[v39]][%[[v35]]]
//   CHECK-DAG:   llvm.store %[[v38]], %[[v40]] {alignment = 2 : i64} : vector<8xf16>, !llvm.ptr
//   CHECK-DAG:   llvm.return

// -----

gpu.module @kernels {
  func.func @codegen_cluster_kernel(%arg0: memref<16xi32>) attributes {gpu.kernel} {
    %cst = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
    %c4 = arith.constant 4 : index
    %0 = gpu.block_id  x
    %1 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%0]
    %2 = arith.muli %0, %c4 : index
    %3 = vector.broadcast %2 : index to vector<4xindex>
    %4 = arith.addi %3, %cst : vector<4xindex>
    %5 = arith.index_cast %4 : vector<4xindex> to vector<4xi32>
    vector.transfer_write %5, %arg0[%1] {in_bounds = [true]} : vector<4xi32>, memref<16xi32>
    return
  }
}

// CHECK-LABEL: llvm.func @codegen_cluster_kernel
//  CHECK-SAME: (%[[arg0:.+]]: !llvm.ptr) attributes {nvvm.kernel} {
//   CHECK-DAG:   %[[v0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v1:.+]] = llvm.insertvalue %[[arg0]], %[[v0]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v2:.+]] = llvm.insertvalue %[[arg0]], %[[v1]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v3:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v4:.+]] = llvm.insertvalue %[[v3]], %[[v2]][2] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v5:.+]] = llvm.mlir.constant(4 : index) : i64
//   CHECK-DAG:   %[[v6:.+]] = llvm.mlir.constant(dense<[0, 1, 2, 3]> : vector<4xindex>) : vector<4xi64>
//   CHECK-DAG:   %[[v7:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:   %[[v8:.+]] = llvm.extractvalue %[[v4]][0] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v9:.+]] = llvm.extractvalue %[[v4]][1] : !llvm.struct<(ptr, ptr, i64)>
//   CHECK-DAG:   %[[v10:.+]] = llvm.insertvalue %[[v8]], %[[v7]][0]
//   CHECK-DAG:   %[[v11:.+]] = llvm.insertvalue %[[v9]], %[[v10]][1]
//   CHECK-DAG:   %[[v12:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:   %[[v13:.+]] = llvm.insertvalue %[[v12]], %[[v11]][2]
//   CHECK-DAG:   %[[v14:.+]] = llvm.mlir.constant(16 : index) : i64
//   CHECK-DAG:   %[[v15:.+]] = llvm.insertvalue %[[v14]], %[[v13]][3, 0]
//   CHECK-DAG:   %[[v16:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:   %[[v17:.+]] = llvm.insertvalue %[[v16]], %[[v15]][4, 0]
//   CHECK-DAG:   %[[v18:.+]] = nvvm.read.ptx.sreg.ctaid.x : i32
//   CHECK-DAG:   %[[v19:.+]] = llvm.sext %[[v18]] : i32 to i64
//   CHECK-DAG:   %[[v20:.+]] = llvm.mul %[[v19]], %[[v5]] overflow<nsw> : i64
//   CHECK-DAG:   %[[v21:.+]] = llvm.mul %[[v19]], %[[v5]] : i64
//   CHECK-DAG:   %[[v22:.+]] = llvm.mlir.poison : vector<4xi64>
//   CHECK-DAG:   %[[v23:.+]] = llvm.mlir.constant(0 : i32) : i32
//   CHECK-DAG:   %[[v24:.+]] = llvm.insertelement %[[v21]], %[[v22]][%[[v23]] : i32] : vector<4xi64>
//   CHECK-DAG:   %[[v25:.+]] = llvm.shufflevector %[[v24]], %[[v22]] [0, 0, 0, 0] : vector<4xi64>
//   CHECK-DAG:   %[[v26:.+]] = llvm.add %[[v25]], %[[v6]] : vector<4xi64>
//   CHECK-DAG:   %[[v27:.+]] = llvm.trunc %[[v26]] : vector<4xi64> to vector<4xi32>
//   CHECK-DAG:   %[[v28:.+]] = llvm.extractvalue %[[v17]][1]
//   CHECK-DAG:   %[[v29:.+]] = llvm.getelementptr %[[v28]][%[[v20]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
//   CHECK-DAG:   llvm.store %[[v27]], %[[v29]] {alignment = 4 : i64} : vector<4xi32>, !llvm.ptr
//   CHECK-DAG:   llvm.return

// -----

gpu.module @kernels {
  func.func @arith_minimumf_requires_workaround(%arg0: f32, %arg1: f32, %arg2: memref<f32>) attributes {gpu.kernel} {
    %0 = arith.minimumf %arg0, %arg1 : f32
    memref.store %0, %arg2[] : memref<f32>
    return
  }
}

// CHECK-LABEL: llvm.func @arith_minimumf_requires_workaround
//       CHECK:   llvm.fcmp
//  CHECK-NEXT:   llvm.select
//  CHECK-NEXT:   llvm.fcmp
//  CHECK-NEXT:   llvm.select

// -----

gpu.module @kernels {
  func.func @arith_maximumf_requires_workaround(
      %arg0: f32, %arg1: f32,
      %arg2: memref<f32>) attributes {gpu.kernel} {
    %0 = arith.maximumf %arg0, %arg1 : f32
    memref.store %0, %arg2[] : memref<f32>
    return
  }
}

// CHECK-LABEL: llvm.func @arith_maximumf_requires_workaround
//       CHECK:   llvm.fcmp
//  CHECK-NEXT:   llvm.select
//  CHECK-NEXT:   llvm.fcmp
//  CHECK-NEXT:   llvm.select

// -----

gpu.module @vector_strided_slice_check {
  func.func @vector_strided_slice_check(%arg2: memref<1x1x1x1xindex>) attributes {gpu.kernel} {
    %cst = arith.constant 3 : index
    %0 = vector.broadcast %cst : index to vector<1x1x8x2xindex>
    %1 = vector.extract_strided_slice %0 {
      offsets = [0, 0, 1, 0],
      sizes   = [1, 1, 1, 1],
      strides = [1, 1, 1, 1]
    } : vector<1x1x8x2xindex> to vector<1x1x1x1xindex>
    %cst1 = arith.constant 4 : index
    %2 = vector.broadcast %cst1 : index to vector<1x1x1x1xindex>
    %3 = arith.addi %1, %2 : vector<1x1x1x1xindex>

    %c0 = arith.constant 0 : index
    vector.transfer_write %3, %arg2[%c0, %c0, %c0, %c0]
       {in_bounds = [true, true, true, true]}
       : vector<1x1x1x1xindex>, memref<1x1x1x1xindex>
    return
  }
}

// CHECK-LABEL: llvm.func @vector_strided_slice_check
//   CHECK-NOT:   vector.extract_strided_slice
//   CHECK-NOT:   vector.broadcast
//   CHECK-NOT:   vector.transfer_write

// -----

// See comments regarding math-to-sqrt in LowerToNVVM.cpp.

gpu.module @test_math_vector_2d {
  func.func @test_math_vector_2d(%arg0: memref<2x2xf32>) attributes {gpu.kernel} {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst
      {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
    %1 = math.sqrt %0 : vector<2x2xf32>
    vector.transfer_write %1, %arg0[%c0, %c0]
       {in_bounds = [true, true]}
       : vector<2x2xf32>, memref<2x2xf32>
    return
  }
}

//   CHECK-LABEL: llvm.func @test_math_vector_2d
// CHECK-COUNT-4:  __nv_sqrtf

gpu.module @test_math_vector_1d {
  func.func @test_math_vector_1d(%arg0: memref<1xf32>) attributes {gpu.kernel} {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst
      {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
    %1 = math.sqrt %0 : vector<1xf32>
    vector.transfer_write %1, %arg0[%c0]
       {in_bounds = [true]}
       : vector<1xf32>, memref<1xf32>
    return
  }
}

// CHECK-LABEL: gpu.module @test_math_vector_1d
//       CHECK:  __nv_sqrtf
// CHECK-LABEL: llvm.func @test_math_vector_1d
//   CHECK-NOT:   llvm.intr.sqrt

// -----

gpu.module @test_scalar_vector_transfer_write {
  func.func @test_scalar_vector_transfer_write(%arg0: memref<128xf32>, %arg1: vector<f32>,
      %arg2: index) attributes {gpu.kernel} {
    %cst = arith.constant 0.0 : f32
    vector.transfer_write %arg1, %arg0[%arg2] : vector<f32>, memref<128xf32>
    return
  }
}

// CHECK-LABEL: llvm.func @test_scalar_vector_transfer
//       CHECK:  llvm.store

// -----

gpu.module @test_scalar_transfer_read {
  func.func @test_scalar_transfer_read(%arg0: memref<128xf32>, %arg1: memref<128xf32>,
      %arg2: index) attributes {gpu.kernel} {
    %cst = arith.constant 0.0 : f32
    %0 = vector.transfer_read %arg0[%arg2], %cst {in_bounds = [true]} : memref<128xf32>, vector<4xf32>
    %1 = vector.extract %0[2] : f32 from vector<4xf32>
    memref.store %1, %arg1[%arg2] : memref<128xf32>
    return
  }
}

// CHECK-LABEL: llvm.func @test_scalar_transfer_read
//       CHECK:  llvm.load
//       CHECK:  llvm.store

// -----

gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
  func.func @arith_extf_f8e4m3_f16(%arg0: f8E4M3FN) -> f16 attributes {gpu.kernel} {
    %0 = arith.extf %arg0 : f8E4M3FN to f16
    return %0 : f16
  }
}

// CHECK-LABEL: @arith_extf_f8e4m3_f16
//  CHECK-SAME: (%[[arg0:.+]]: i8) -> f16 attributes {nvvm.kernel}
//   CHECK-DAG: %[[v0:.+]] = llvm.zext %[[arg0]] : i8 to i16
//   CHECK-DAG: %[[v1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[v0]] : (i16) -> i32
//   CHECK-DAG: %[[v2:.+]] = llvm.trunc %[[v1]] : i32 to i16
//   CHECK-DAG: %[[v3:.+]] = llvm.bitcast %[[v2]] : i16 to f16
//   CHECK-DAG: llvm.return %[[v3]] : f16

// -----

gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
  func.func @arith_truncf_f16_f8e4m3(%arg0: f16) -> f8E4M3FN attributes {gpu.kernel} {
    %0 = arith.truncf %arg0 : f16 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

// CHECK-LABEL: @arith_truncf_f16_f8e4m3
//  CHECK-SAME: (%[[arg0:.+]]: f16) -> i8 attributes {nvvm.kernel}
//   CHECK-DAG: %[[v0:.+]] = llvm.bitcast %[[arg0]] : f16 to i16
//   CHECK-DAG: %[[v1:.+]] = llvm.zext %[[v0]] : i16 to i32
//   CHECK-DAG: %[[v2:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1;", "=h,r" %[[v1]] : (i32) -> i16
//   CHECK-DAG: %[[v3:.+]] = llvm.trunc %[[v2]] : i16 to i8
//   CHECK-DAG: llvm.return %[[v3]] : i8

// -----

gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
  func.func @arith_add_f8e4m3(%arg0: f8E4M3FN, %arg1: f8E4M3FN) -> f8E4M3FN attributes {gpu.kernel} {
    %0 = arith.extf %arg0 : f8E4M3FN to f16
    %1 = arith.extf %arg1 : f8E4M3FN to f16
    %2 = arith.addf %0, %1 : f16
    %3 = arith.truncf %2 : f16 to f8E4M3FN
    return %3 : f8E4M3FN
  }
}

// CHECK-LABEL: @arith_add_f8e4m3
//  CHECK-SAME: (%[[arg0:.+]]: i8, %[[arg1:.+]]: i8) -> i8 attributes {nvvm.kernel}
//   CHECK-DAG: %[[v0:.+]] = llvm.zext %[[arg0]] : i8 to i16
//   CHECK-DAG: %[[v1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[v0]] : (i16) -> i32
//   CHECK-DAG: %[[v2:.+]] = llvm.trunc %[[v1]] : i32 to i16
//   CHECK-DAG: %[[v3:.+]] = llvm.bitcast %[[v2]] : i16 to f16
//   CHECK-DAG: %[[v4:.+]] = llvm.zext %[[arg1]] : i8 to i16
//   CHECK-DAG: %[[v5:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[v4]] : (i16) -> i32
//   CHECK-DAG: %[[v6:.+]] = llvm.trunc %[[v5]] : i32 to i16
//   CHECK-DAG: %[[v7:.+]] = llvm.bitcast %[[v6]] : i16 to f16
//   CHECK-DAG: %[[v8:.+]] = llvm.fadd %[[v3]], %[[v7]] : f16
//   CHECK-DAG: %[[v9:.+]] = llvm.bitcast %[[v8]] : f16 to i16
//   CHECK-DAG: %[[v10:.+]] = llvm.zext %[[v9]] : i16 to i32
//   CHECK-DAG: %[[v11:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1;", "=h,r" %[[v10]] : (i32) -> i16
//   CHECK-DAG: %[[v12:.+]] = llvm.trunc %[[v11]] : i16 to i8
//   CHECK-DAG: llvm.return %[[v12]] : i8
