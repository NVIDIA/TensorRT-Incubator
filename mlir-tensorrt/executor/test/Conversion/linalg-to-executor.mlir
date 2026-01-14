// RUN: executor-opt %s -split-input-file -convert-linalg-to-executor -canonicalize | FileCheck %s

!scalar_type = i1
!memref_type = memref<?x!scalar_type>

func.func @fill_host_i1(%arg0: !memref_type, %arg1: !scalar_type) {
  linalg.fill ins(%arg1: !scalar_type) outs(%arg0 : !memref_type)
  return
}

// CHECK-LABEL: func.func @fill_host_i1
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xi1>, %[[arg1:.+]]: i1)
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//   CHECK-DAG:     %[[v1:.+]] = executor.table.get %[[v0]][1] :
//   CHECK-DAG:     %[[v2:.+]] = executor.table.get %[[v0]][3] :
//   CHECK-DAG:     %[[v3:.+]] = executor.getoffset[%[[v2]]] : (i64) -> i64, i1
//   CHECK-DAG:     %[[v4:.+]] = executor.zext %[[arg1]] : i1 to i8
//   CHECK-DAG:     executor.call @__memset_8(%[[v1]], %[[c0_i64]], %[[v3]], %[[v4]])
//   CHECK-DAG:     return

// -----

!scalar_type = f16
!memref_type = memref<?x!scalar_type>

func.func @fill_host_f16(%arg0: !memref_type, %arg1: !scalar_type) {
  linalg.fill ins(%arg1: !scalar_type) outs(%arg0 : !memref_type)
  return
}

// CHECK-LABEL: func.func @fill_host_f16
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xf16>, %[[arg1:.+]]: f16)
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//   CHECK-DAG:     %[[v1:.+]] = executor.table.get %[[v0]][1] :
//   CHECK-DAG:     %[[v2:.+]] = executor.table.get %[[v0]][3] :
//   CHECK-DAG:     %[[v3:.+]] = executor.getoffset[%[[v2]]] : (i64) -> i64, f16
//   CHECK-DAG:     %[[v4:.+]] = executor.bitcast %[[arg1]] : f16 to i16
//   CHECK-DAG:     executor.call @__memset_16(%[[v1]], %[[c0_i64]], %[[v3]], %[[v4]])
//   CHECK-DAG:     return

// -----

!scalar_type = f32
!memref_type = memref<?x!scalar_type>

func.func @fill_host_f32(%arg0: !memref_type, %arg1: !scalar_type) {
  linalg.fill ins(%arg1: !scalar_type) outs(%arg0 : !memref_type)
  return
}

// CHECK-LABEL: func.func @fill_host_f32
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: f32)
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//   CHECK-DAG:     %[[v1:.+]] = executor.table.get %[[v0]][1] :
//   CHECK-DAG:     %[[v2:.+]] = executor.table.get %[[v0]][3] :
//   CHECK-DAG:     %[[v3:.+]] = executor.getoffset[%[[v2]]] : (i64) -> i64, f32
//   CHECK-DAG:     %[[v4:.+]] = executor.bitcast %[[arg1]] : f32 to i32
//   CHECK-DAG:     executor.call @__memset_32(%[[v1]], %[[c0_i64]], %[[v3]], %[[v4]])
//   CHECK-DAG:     return

// -----

func.func @test_generic_fill_simple(%arg0: memref<f32, #executor.memory_type<host>>) {
  %cst = arith.constant 3.0 : f32
  linalg.generic {
    indexing_maps = [affine_map<() -> ()>],
    iterator_types = []
  } outs(%arg0 : memref<f32, #executor.memory_type<host>>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  return
}

// CHECK-LABEL: func.func @test_generic_fill_simple
//  CHECK-SAME: (%[[arg0:.+]]: memref<f32, #executor.memory_type<host>>)
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[cst:.+]] = arith.constant 3.000000e+00 : f32
//       CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//       CHECK:     %[[v1:.+]] = executor.table.get %[[v0]][1]
//       CHECK:     %[[v2:.+]] = executor.getoffset[1] : () -> i64, f32
//       CHECK:     %[[v3:.+]] = executor.bitcast %[[cst]] : f32 to i32
//       CHECK:     executor.call @__memset_32(%[[v1]], %[[c0_i64]], %[[v2]], %[[v3]])
//       CHECK:     return
