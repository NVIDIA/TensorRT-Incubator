// RUN: executor-opt %s -split-input-file -convert-scf-to-cf -convert-memref-to-executor="use-packed-memref-cconv=false index-bitwidth=32" \
// RUN:   -convert-executor-to-executor="use-packed-memref-cconv=false index-bitwidth=32" -executor-expand-ops \
// RUN:   -convert-std-to-executor="use-packed-memref-cconv=false index-bitwidth=32" -canonicalize | FileCheck %s
// RUN: executor-opt %s -split-input-file -convert-scf-to-cf -convert-memref-to-executor="index-bitwidth=32" \
// RUN:   -convert-executor-to-executor="index-bitwidth=32" -executor-expand-ops \
// RUN:   -convert-std-to-executor="index-bitwidth=32" -canonicalize | FileCheck %s --check-prefix=PACKED

func.func @alloc(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %0 = memref.alloc (%arg0, %arg1) : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

func.func @subview_load(%arg0: memref<5x4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<5x4xf32> -> memref<f32>, index, index, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [2], sizes: [1, 1], strides: [4, 1] : memref<f32> to memref<1x1xf32, strided<[4, 1], offset: 2>>
  %0 = memref.load %reinterpret_cast[%c0, %c0] : memref<1x1xf32, strided<[4, 1], offset: 2>>
  return %0 : f32
}

func.func @load_strided(%base: memref<?x?xf32, strided<[?,?], offset:?>>, %row: index, %col: index) -> f32 {
  %0 = memref.load %base[%row, %col] : memref<?x?xf32, strided<[?,?], offset:?>>
  return %0 : f32
}

func.func @copy(%dest: memref<?x?xf32>,
                 %src: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %src, %c0 : memref<?x?xf32>
  %dim1 = memref.dim %src, %c1 : memref<?x?xf32>
  scf.for %row = %c0 to %dim0 step %c1 {
    scf.for %col = %c0 to %dim1 step %c1 {
      %accum = arith.constant 0.1 : f32
      memref.store %accum, %src[%row, %col] : memref<?x?xf32>
      %el = memref.load %src[%row, %col] : memref<?x?xf32>
      memref.store %el, %dest[%row, %col] : memref<?x?xf32>
    }
  }
  return
}

func.func @main() -> index {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = func.call @alloc(%c64, %c128) : (index, index) -> memref<?x?xf32>
  %1 = func.call @alloc(%c64, %c128) : (index, index) -> memref<?x?xf32>
  func.call @copy(%1, %0)
    : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %c0 : index
}

// CHECK-LABEL: @alloc
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32> {
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c16_i32:.+]] = executor.constant 16 : i32
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v0:.+]] = executor.muli %[[arg1]], %[[arg0]] : i32
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c4_i32]] : i32
//       CHECK:     %[[v4:.+]] = executor.alloc %[[v1]] bytes
//       CHECK:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i32]], %[[arg0]], %[[arg1]], %[[arg1]], %[[c1_i32]]
//       CHECK:     return %[[v5]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>

// CHECK-LABEL: @subview_load
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32) -> f32 {
//       CHECK:     %[[c8_i32:.+]] = executor.constant 8 : i32
//       CHECK:     %[[v0:.+]] = executor.load %[[arg1]] + %[[c8_i32]]
//       CHECK:     return %[[v0]] : f32

// CHECK-LABEL: @load_strided
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32, %[[arg7:.+]]: i32, %[[arg8:.+]]: i32) -> f32 {
//       CHECK:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[v0:.+]] = executor.muli %[[arg7]], %[[arg5]] : i32
//       CHECK:     %[[v1:.+]] = executor.addi %[[arg2]], %[[v0]] : i32
//       CHECK:     %[[v2:.+]] = executor.muli %[[arg8]], %[[arg6]] : i32
//       CHECK:     %[[v3:.+]] = executor.addi %[[v1]], %[[v2]] : i32
//       CHECK:     %[[v4:.+]] = executor.muli %[[v3]], %[[c4_i32]] : i32
//       CHECK:     %[[v5:.+]] = executor.load %[[arg1]] + %[[v4]]
//       CHECK:     return %[[v5]] : f32
// CHECK-LABEL: @copy
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32, %[[arg7:.+]]: !executor.ptr<host>, %[[arg8:.+]]: !executor.ptr<host>, %[[arg9:.+]]: i32, %[[arg10:.+]]: i32, %[[arg11:.+]]: i32, %[[arg12:.+]]: i32, %[[arg13:.+]]: i32) {
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[fill:.+]] = executor.constant 1.000000e-01 : f32
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     cf.br ^bb1(%[[c0_i32]] : i32)
//       CHECK:   ^bb1(%[[v0]]: i32):  // 2 preds: ^bb0, ^bb4
//       CHECK:     %[[v1:.+]] = executor.icmp <slt> %[[v0]], %[[arg10]] : i32
//       CHECK:     cf.cond_br %[[v1]], ^bb2(%[[c0_i32]] : i32), ^bb5
//       CHECK:   ^bb2(%[[v2]]: i32):  // 2 preds: ^bb1, ^bb3
//       CHECK:     %[[v3:.+]] = executor.icmp <slt> %[[v2]], %[[arg11]] : i32
//       CHECK:     cf.cond_br %[[v3]], ^bb3, ^bb4
//       CHECK:   ^bb3:  // pred: ^bb2
//       CHECK:     %[[v4:.+]] = executor.muli %[[v0]], %[[arg12]] : i32
//       CHECK:     %[[v5:.+]] = executor.addi %[[v4]], %[[v2]] : i32
//       CHECK:     %[[v6:.+]] = executor.muli %[[v5]], %[[c4_i32]] : i32
//       CHECK:     executor.store %[[fill]] to %[[arg8]] + %[[v6]]
//       CHECK:     %[[v7:.+]] = executor.muli %[[v0]], %[[arg12]] : i32
//       CHECK:     %[[v8:.+]] = executor.addi %[[v7]], %[[v2]] : i32
//       CHECK:     %[[v9:.+]] = executor.muli %[[v8]], %[[c4_i32]] : i32
//       CHECK:     %[[v10:.+]] = executor.load %[[arg8]] + %[[v9]]
//       CHECK:     %[[v11:.+]] = executor.muli %[[v0]], %[[arg5]] : i32
//       CHECK:     %[[v12:.+]] = executor.addi %[[v11]], %[[v2]] : i32
//       CHECK:     %[[v13:.+]] = executor.muli %[[v12]], %[[c4_i32]] : i32
//       CHECK:     executor.store %[[v10]] to %[[arg1]] + %[[v13]]
//       CHECK:     %[[v14:.+]] = executor.addi %[[v2]], %[[c1_i32]] : i32
//       CHECK:     cf.br ^bb2(%[[v14]] : i32)
//       CHECK:   ^bb4:  // pred: ^bb2
//       CHECK:     %[[v15:.+]] = executor.addi %[[v0]], %[[c1_i32]] : i32
//       CHECK:     cf.br ^bb1(%[[v15]] : i32)
//       CHECK:   ^bb5:  // pred: ^bb1
//       CHECK:     return
// CHECK-LABEL: @main
//  CHECK-SAME: () -> i32 {
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c64_i32:.+]] = executor.constant 64 : i32
//   CHECK-DAG:     %[[c128_i32:.+]] = executor.constant 128 : i32
//       CHECK:     %[[v0:.+]] = call @alloc(%[[c64_i32]], %[[c128_i32]]) : (i32, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v1:.+]] = call @alloc(%[[c64_i32]], %[[c128_i32]]) : (i32, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v1]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v3:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v5:.+]] = executor.table.get %[[v1]][3] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v6:.+]] = executor.table.get %[[v1]][4] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v7:.+]] = executor.table.get %[[v1]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v8:.+]] = executor.table.get %[[v0]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v9:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v11:.+]] = executor.table.get %[[v0]][3] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v12:.+]] = executor.table.get %[[v0]][4] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v13:.+]] = executor.table.get %[[v0]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     call @copy(%[[v2]], %[[v3]], %[[c0_i32]], %[[v5]], %[[v6]], %[[v7]], %[[c1_i32]], %[[v8]], %[[v9]], %[[c0_i32]], %[[v11]], %[[v12]], %[[v13]], %[[c1_i32]])
//       CHECK:     return %[[c0_i32]] : i32


// -----

func.func @callee(%arg0: memref<10xf16>) -> memref<10xf16> {
  return %arg0 : memref<10xf16>
}

func.func @executor_f16_arg(%arg0: memref<10xf16>) -> memref<10xf16> {
  %0 = func.call @callee(%arg0) : (memref<10xf16>)->memref<10xf16>
  return %0 : memref<10xf16>
}

// CHECK-LABEL: @callee
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//       CHECK:     %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]] : !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     return %[[v0]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
// CHECK-LABEL: @executor_f16_arg
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c10_i32:.+]] = executor.constant 10 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[v0:.+]] = call @callee(%[[arg0]], %[[arg1]], %[[c0_i32]], %[[c10_i32]], %[[c1_i32]]) : (!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     return %[[v0]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>

// PACKED-LABEL: @callee
//  PACKED-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//       PACKED:     return %[[arg0]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
// PACKED-LABEL: @executor_f16_arg
//  PACKED-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//       PACKED:     %[[v0:.+]] = call @callee(%[[arg0]]) : (!executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       PACKED:     return %[[v0]]

// -----

func.func @add_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: add_int
//       CHECK: %[[v_0:.+]] = executor.addi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @add_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: add_float
//       CHECK: %[[v_0:.+]] = executor.addf %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @mul_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.muli %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: mul_int
//       CHECK: %[[v_0:.+]] = executor.muli %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @mul_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: mul_float
//       CHECK: %[[v_0:.+]] = executor.mulf %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @sub_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: sub_int
//       CHECK: %[[v_0:.+]] = executor.subi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @sub_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.subf %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: sub_float
//       CHECK: %[[v_0:.+]] = executor.subf %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @div_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.divsi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: div_int
//       CHECK: %[[v_0:.+]] = executor.sdivi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @div_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.divf %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: div_float
//       CHECK: %[[v_0:.+]] = executor.divf %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @max_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.maxsi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: max_int
//       CHECK: %[[v_0:.+]] = executor.smax %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @max_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.maximumf %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: max_float
//       CHECK: %[[v_0:.+]] = executor.fmax %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @shift_left(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.shli %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_left
//       CHECK: %[[v_0:.+]] = executor.shift_lefti %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @shift_right_logical(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.shrui %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_right_logical
//       CHECK: %[[v_0:.+]] = executor.shift_right_logicali %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @shift_right_arithmetic(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.shrsi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_right_arithmetic
//       CHECK: %[[v_0:.+]] = executor.shift_right_arithmetici %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitcast(%arg0: i32) -> f32 {
  %0 = arith.bitcast %arg0 : i32 to f32
  return %0 : f32
}
// CHECK-LABEL: bitcast
//       CHECK: %[[v_0:.+]] = executor.bitcast %[[arg0:.+]] : i32 to f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  %0 = executor.select %arg0, %arg1, %arg2 : i32
  return %0 : i32
}
// CHECK-LABEL: select
//       CHECK: %[[v_0:.+]] = executor.select %[[arg0:.+]], %[[arg1:.+]], %[[arg2:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @abs(%arg0: f32) -> f32 {
    %0 = math.absf %arg0 : f32
    return %0 : f32
}
// CHECK-LABEL: abs
//       CHECK: %[[v_0:.+]] = executor.absf %[[arg0:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @abs_int(%arg0: i32) -> i32 {
    %0 = math.absi %arg0 : i32
    return %0 : i32
}
// CHECK-LABEL: abs_int
//       CHECK: %[[v_0:.+]] = executor.absi %[[arg0:.+]] : i32
//       CHECK: return %[[v_0]] : i32


// -----

func.func @cmpi_ugt(%arg0: i32, %arg1: i32) -> i1 {
  %0 = arith.cmpi ugt, %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: @cmpi_ugt
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i1 {
//       CHECK:     %[[v0:.+]] = executor.icmp <ugt> %[[arg0]], %[[arg1]] : i32
//       CHECK:     return %[[v0]] : i1

// -----

func.func @cmpi_ult(%arg0: i32, %arg1: i32) -> i1 {
  %0 = arith.cmpi ult, %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: @cmpi_ult
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i1 {
//       CHECK:     %[[v0:.+]] = executor.icmp <ult> %[[arg0]], %[[arg1]] : i32
//       CHECK:     return %[[v0]] : i1

// -----

func.func @bitwise_or_bool(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.ori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: bitwise_or_bool
//       CHECK: %[[v_0:.+]] = executor.bitwise_ori %[[arg0:.+]], %[[arg1:.+]] : i1
//       CHECK: return %[[v_0]] : i1

// -----

func.func @bitwise_or_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.ori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_or_int
//       CHECK: %[[v_0:.+]] = executor.bitwise_ori %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitwise_and_bool(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.andi %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: bitwise_and_bool
//       CHECK: %[[v_0:.+]] = executor.bitwise_andi %[[arg0:.+]], %[[arg1:.+]] : i1
//       CHECK: return %[[v_0]] : i1

// -----

func.func @bitwise_and_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.andi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_and_int
//       CHECK: %[[v_0:.+]] = executor.bitwise_andi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitwise_xor_bool(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.xori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: bitwise_xor_bool
//       CHECK: %[[v_0:.+]] = executor.bitwise_xori %[[arg0:.+]], %[[arg1:.+]] : i1
//       CHECK: return %[[v_0]] : i1

// -----

func.func @bitwise_xor_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.xori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_xor_int
//       CHECK: %[[v_0:.+]] = executor.bitwise_xori %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @cast_si_to_fp(%arg0: i16) -> f32 {
  %0 = arith.sitofp %arg0 : i16 to f32
  return %0: f32
}
// CHECK-LABEL: cast_si_to_fp
//       CHECK: %[[v_0:.+]] = executor.sitofp %[[arg0:.+]] : i16 to f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @cast_fp_to_si(%arg0: f32) -> i32 {
  %0 = arith.fptosi %arg0 : f32 to i32
  return %0: i32
}
// CHECK-LABEL: cast_fp_to_si
//       CHECK: %[[v_0:.+]] = executor.fptosi %[[arg0:.+]] : f32 to i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @arith_fcmp_to_cmpf(%arg0: f32, %arg1: f32)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %1 = arith.cmpf false, %arg0, %arg1 : f32
  %2 = arith.cmpf oeq, %arg0, %arg1 : f32
  %3 = arith.cmpf ogt, %arg0, %arg1 : f32
  %4 = arith.cmpf oge, %arg0, %arg1 : f32
  %5 = arith.cmpf olt, %arg0, %arg1 : f32
  %6 = arith.cmpf ole, %arg0, %arg1 : f32
  %7 = arith.cmpf one, %arg0, %arg1 : f32
  %8 = arith.cmpf ord, %arg0, %arg1 : f32
  %9 = arith.cmpf ueq, %arg0, %arg1 : f32
  %10 = arith.cmpf ugt, %arg0, %arg1 : f32
  %11 = arith.cmpf uge, %arg0, %arg1 : f32
  %12 = arith.cmpf ult, %arg0, %arg1 : f32
  %13 = arith.cmpf ule, %arg0, %arg1 : f32
  %14 = arith.cmpf une, %arg0, %arg1 : f32
  %15 = arith.cmpf uno, %arg0, %arg1 : f32
  %16 = arith.cmpf true, %arg0, %arg1 : f32
  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16
    : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
// CHECK-LABEL: @arith_fcmp_to_cmpf
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32)
//       CHECK:     %[[v0:.+]] = executor.fcmp <_false> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v1:.+]] = executor.fcmp <oeq> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v2:.+]] = executor.fcmp <ogt> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v3:.+]] = executor.fcmp <oge> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v4:.+]] = executor.fcmp <olt> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v5:.+]] = executor.fcmp <ole> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v6:.+]] = executor.fcmp <one> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v7:.+]] = executor.fcmp <ord> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v8:.+]] = executor.fcmp <ueq> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v9:.+]] = executor.fcmp <ugt> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v10:.+]] = executor.fcmp <uge> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v11:.+]] = executor.fcmp <ult> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v12:.+]] = executor.fcmp <ule> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v13:.+]] = executor.fcmp <une> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v14:.+]] = executor.fcmp <uno> %[[arg0]], %[[arg1]] : f32
//       CHECK:     %[[v15:.+]] = executor.fcmp <_true> %[[arg0]], %[[arg1]] : f32
//       CHECK:     return %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], %[[v7]], %[[v8]], %[[v9]], %[[v10]], %[[v11]], %[[v12]], %[[v13]], %[[v14]], %[[v15]]

// -----


func.func @select(%arg0: i1, %arg1: index, %arg2: index, %arg3: memref<32xf32>, %arg4: memref<32xf32>, %arg5: memref<3x3xf16>, %arg6: memref<3x3xf16>) ->
    (index, memref<32xf32>, memref<3x3xf16>) {
  %0 = arith.select %arg0, %arg1, %arg2 : index
  %1 = arith.select %arg0, %arg3, %arg4 : memref<32xf32>
  %2 = arith.select %arg0, %arg5, %arg6 : memref<3x3xf16>
  return %0, %1, %2 : index, memref<32xf32>, memref<3x3xf16>
}

// CHECK-LABEL: @select
//       CHECK:     %[[v4:.+]] = executor.select
//       CHECK:     %[[v5:.+]] = executor.select
//       CHECK:     %[[v6:.+]] = executor.select
//       CHECK:     return %[[v4]], %[[v5]], %[[v6]] :
