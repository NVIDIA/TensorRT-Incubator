// Tests NVVM translation for modules requiring libdevice.
// REQUIRES: cuda
// RUN: rm -rf %t || true
// RUN: kernel-opt -split-input-file %s -pass-pipeline="builtin.module(kernel-set-gpu-target{chip=sm_80},gpu.module(translate-nvvm-to-ptx{dump-ptx=%t}))" | FileCheck %s
// RUN: ls -lah %t | FileCheck %s --check-prefix=DUMP

gpu.module @kernels3 {
  llvm.func @__nv_cosf(f32) -> f32 attributes {nvvm.kernel}
  llvm.func @codegen_cluster_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) attributes {nvvm.kernel} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(4 : index) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %14 = nvvm.read.ptx.sreg.ctaid.x : i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = llvm.mul %15, %12  : i64
    %17 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %20 = llvm.mlir.poison : vector<4xf32>
    %21 = llvm.mlir.constant(0 : i64) : i64
    %22 = llvm.extractelement %19[%21 : i64] : vector<4xf32>
    %23 = llvm.call @__nv_cosf(%22) : (f32) -> f32
    %24 = llvm.insertelement %23, %20[%21 : i64] : vector<4xf32>
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.extractelement %19[%25 : i64] : vector<4xf32>
    %27 = llvm.call @__nv_cosf(%26) : (f32) -> f32
    %28 = llvm.insertelement %27, %24[%25 : i64] : vector<4xf32>
    %29 = llvm.mlir.constant(2 : i64) : i64
    %30 = llvm.extractelement %19[%29 : i64] : vector<4xf32>
    %31 = llvm.call @__nv_cosf(%30) : (f32) -> f32
    %32 = llvm.insertelement %31, %28[%29 : i64] : vector<4xf32>
    %33 = llvm.mlir.constant(3 : i64) : i64
    %34 = llvm.extractelement %19[%33 : i64] : vector<4xf32>
    %35 = llvm.call @__nv_cosf(%34) : (f32) -> f32
    %36 = llvm.insertelement %35, %32[%33 : i64] : vector<4xf32>
    %37 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %36, %38 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: gpu.module @kernels3
//       CHECK: kernel.ptx_data = dense_resource<[[BLOBKEY:.*]]> : tensor<{{[0-9]+}}xi8>
//       CHECK:  dialect_resources: {
//  CHECK-NEXT:    builtin: {
//  CHECK-NEXT:      [[BLOBKEY]]: "0x{{[0-9A-F]+}}"

// DUMP-DAG: no-symbol-name_kernels3.ptx
