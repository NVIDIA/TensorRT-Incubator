// RUN: executor-opt %s -split-input-file -convert-scf-to-cf -convert-memref-to-executor="index-bitwidth=32" \
// RUN:   -convert-executor-to-executor="index-bitwidth=32" -executor-expand-ops \
// RUN:   -convert-std-to-executor="index-bitwidth=32" -canonicalize -reconcile-unrealized-casts | FileCheck %s

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
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[arg1]], %[[arg0]] : i32
//   CHECK-DAG:     %[[v1:.+]] = executor.muli %[[v0]], %[[c4_i32]] : i32
//   CHECK-DAG:     %[[v4:.+]] = executor.alloc %[[v1]] bytes
//   CHECK-DAG:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i32]], %[[arg0]], %[[arg1]], %[[arg1]], %[[c1_i32]]
//       CHECK:     return %[[v5]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>

// CHECK-LABEL: @subview_load
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<{{.*}}>) -> f32 {
//   CHECK-DAG:     %[[c8_i32:.+]] = executor.constant 8 : i32
//   CHECK-DAG:     %[[ptr:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v0:.+]] = executor.load %[[ptr]] + %[[c8_i32]]
//       CHECK:     return %[[v0]] : f32


// CHECK-LABEL: func.func @load_strided
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32) -> f32 {
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[v0:.+]] = executor.table.get %[[arg0]][2] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = executor.table.get %[[arg0]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v2:.+]] = executor.muli %[[arg1]], %[[v1]] : i32
//   CHECK-DAG:     %[[v3:.+]] = executor.addi %[[v0]], %[[v2]] : i32
//   CHECK-DAG:     %[[v4:.+]] = executor.table.get %[[arg0]][6] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v5:.+]] = executor.muli %[[arg2]], %[[v4]] : i32
//   CHECK-DAG:     %[[v6:.+]] = executor.addi %[[v3]], %[[v5]] : i32
//   CHECK-DAG:     %[[v7:.+]] = executor.muli %[[v6]], %[[c4_i32]] : i32
//   CHECK-DAG:     %[[v8:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v9:.+]] = executor.load %[[v8]] + %[[v7]] : (!executor.ptr<host>, i32) -> f32
//       CHECK:     return %[[v9]] : f32

// CHECK-LABEL: func.func @copy
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>, %[[arg1:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>)
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[cst_f32:.+]] = executor.constant 1.000000e-01 : f32
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[v0:.+]] = executor.table.get %[[arg1]][3] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = executor.table.get %[[arg1]][4] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     cf.br ^bb1(%[[c0_i32]] : i32)
//       CHECK:   ^bb1(%[[v2]]: i32):  // 2 preds: ^bb0, ^bb4
//       CHECK:     %[[v3:.+]] = executor.icmp <slt> %[[v2]], %[[v0]] : i32
//       CHECK:     cf.cond_br %[[v3]], ^bb2(%[[c0_i32]] : i32), ^bb5
//       CHECK:   ^bb2(%[[v4]]: i32):  // 2 preds: ^bb1, ^bb3
//       CHECK:     %[[v5:.+]] = executor.icmp <slt> %[[v4]], %[[v1]] : i32
//       CHECK:     cf.cond_br %[[v5]], ^bb3, ^bb4
//       CHECK:   ^bb3:  // pred: ^bb2
//   CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[arg1]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v7:.+]] = executor.muli %[[v2]], %[[v6]] : i32
//   CHECK-DAG:     %[[v8:.+]] = executor.addi %[[v7]], %[[v4]] : i32
//   CHECK-DAG:     %[[v9:.+]] = executor.muli %[[v8]], %[[c4_i32]] : i32
//   CHECK-DAG:     %[[v10:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     executor.store %[[cst_f32]] to %[[v10]] + %[[v9]] : f32, !executor.ptr<host>, i32
//   CHECK-DAG:     %[[v11:.+]] = executor.table.get %[[arg1]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v12:.+]] = executor.muli %[[v2]], %[[v11]] : i32
//   CHECK-DAG:     %[[v13:.+]] = executor.addi %[[v12]], %[[v4]] : i32
//   CHECK-DAG:     %[[v14:.+]] = executor.muli %[[v13]], %[[c4_i32]] : i32
//   CHECK-DAG:     %[[v15:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v16:.+]] = executor.load %[[v15]] + %[[v14]] : (!executor.ptr<host>, i32) -> f32
//   CHECK-DAG:     %[[v17:.+]] = executor.table.get %[[arg0]][5] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v18:.+]] = executor.muli %[[v2]], %[[v17]] : i32
//   CHECK-DAG:     %[[v19:.+]] = executor.addi %[[v18]], %[[v4]] : i32
//   CHECK-DAG:     %[[v20:.+]] = executor.muli %[[v19]], %[[c4_i32]] : i32
//   CHECK-DAG:     %[[v21:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     executor.store %[[v16]] to %[[v21]] + %[[v20]] : f32, !executor.ptr<host>, i32
//       CHECK:     %[[v22:.+]] = executor.addi %[[v4]], %[[c1_i32]] : i32
//       CHECK:     cf.br ^bb2(%[[v22]] : i32)
//       CHECK:   ^bb4:  // pred: ^bb2
//       CHECK:     %[[v23:.+]] = executor.addi %[[v2]], %[[c1_i32]] : i32
//       CHECK:     cf.br ^bb1(%[[v23]] : i32)
//       CHECK:   ^bb5:  // pred: ^bb1
//       CHECK:     return

// CHECK-LABEL: func.func @main
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c64_i32:.+]] = executor.constant 64 : i32
//   CHECK-DAG:     %[[c128_i32:.+]] = executor.constant 128 : i32
//   CHECK-DAG:     %[[v0:.+]] = call @alloc(%[[c64_i32]], %[[c128_i32]]) : (i32, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = call @alloc(%[[c64_i32]], %[[c128_i32]]) : (i32, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     call @copy(%[[v1]], %[[v0]]) : (!executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>) -> ()
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
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//       CHECK:     return %[[arg0]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
// CHECK-LABEL: @executor_f16_arg
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> {
//       CHECK:     %[[v0:.+]] = call @callee(%[[arg0]]) : (!executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     return %[[v0]]

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

// -----

func.func @cf_assert(%arg0: i1) {
  cf.assert %arg0, "assert failed"
  return
}

// CHECK-LABEL: @cf_assert
//  CHECK-SAME: (%[[arg0:.+]]: i1)
//  CHECK-NEXT:     executor.assert %[[arg0]], "assert failed"
//  CHECK-NEXT:     return
