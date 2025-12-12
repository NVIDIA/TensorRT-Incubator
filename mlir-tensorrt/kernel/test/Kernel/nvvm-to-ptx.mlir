// RUN: rm -rf %t || true
// RUN: kernel-opt -split-input-file %s -pass-pipeline="builtin.module(kernel-set-gpu-target{chip=sm_80},gpu.module(translate-nvvm-to-ptx{dump-ptx=%t}))" | FileCheck %s
// RUN: ls -lah %t | FileCheck %s --check-prefix=DUMP

builtin.module @outer_module {
gpu.module @kernels1 {
  llvm.mlir.global external @shared_memory() {addr_space = 3 : i32} : !llvm.array<128 x f16>
  llvm.func @simple_copy_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) {
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
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.mlir.constant(32 : index) : i64
    %14 = nvvm.read.ptx.sreg.tid.x : i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = nvvm.read.ptx.sreg.ctaid.x : i32
    %17 = llvm.sext %16 : i32 to i64
    %18 = llvm.mlir.constant(128 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.getelementptr %20[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mlir.addressof @shared_memory : !llvm.ptr<3>
    %25 = llvm.mlir.constant(3735928559 : index) : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr<3>
    %27 = llvm.mlir.poison : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %23, %28[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.insertvalue %30, %29[2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %18, %31[3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %19, %32[4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.mul %15, %12  : i64
    %35 = llvm.add %34, %17  : i64
    %36 = llvm.mul %15, %12  : i64
    %37 = llvm.add %36, %17  : i64
    %38 = llvm.add %37, %13  : i64
    %39 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.getelementptr %39[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %41 = llvm.load %40 : !llvm.ptr -> f16
    %42 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.getelementptr %42[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %44 = llvm.load %43 : !llvm.ptr -> f16
    %45 = llvm.extractvalue %33[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.getelementptr %45[%35] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    llvm.store %41, %46 : f16, !llvm.ptr<3>
    %47 = llvm.extractvalue %33[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.getelementptr %47[%38] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    llvm.store %44, %48 : f16, !llvm.ptr<3>
    nvvm.barrier0
    %49 = llvm.extractvalue %33[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.getelementptr %49[%35] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %51 = llvm.load %50 : !llvm.ptr<3> -> f16
    %52 = llvm.extractvalue %33[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.getelementptr %52[%38] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %54 = llvm.load %53 : !llvm.ptr<3> -> f16
    %55 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.getelementptr %55[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %51, %56 : f16, !llvm.ptr
    %57 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.getelementptr %57[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %54, %58 : f16, !llvm.ptr
    llvm.return
  }
}
}

// CHECK-LABEL: gpu.module @kernels1
//       CHECK: kernel.ptx_data = dense_resource<[[BLOBKEY:.*]]> : tensor<{{[0-9]+}}xi8>
//       CHECK:  dialect_resources: {
//  CHECK-NEXT:    builtin: {
//  CHECK-NEXT:      [[BLOBKEY]]: "0x{{[0-9A-F]+}}"

// DUMP-DAG: outer_module_kernels1.ptx

// -----

gpu.module @kernels2 {
  llvm.func @simple_vector_copy_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) {
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
    %12 = llvm.mlir.constant(8 : index) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %14 = nvvm.read.ptx.sreg.tid.x : i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = llvm.mul %15, %12  : i64
    %17 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %20 = llvm.load %18 {alignment = 2 : i64} : !llvm.ptr -> vector<8xf16>
    %21 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.getelementptr %21[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %20, %22 {alignment = 2 : i64} : vector<8xf16>, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: gpu.module @kernels2
//       CHECK: kernel.ptx_data = dense_resource<[[BLOBKEY:.*]]> : tensor<{{[0-9]+}}xi8>
//       CHECK:  dialect_resources: {
//  CHECK-NEXT:    builtin: {
//  CHECK-NEXT:      [[BLOBKEY]]: "0x{{[0-9A-F]+}}"

// DUMP-DAG: no-symbol-name_kernels2.ptx
