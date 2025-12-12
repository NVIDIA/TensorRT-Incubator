// RUN: kernel-opt %s -split-input-file -kernel-special-floats-type-conversion | FileCheck %s

gpu.module @float_type_conversions {
  func.func @test_1(%arg0: memref<f8E4M3FN>, %arg1: memref<f16>, %arg2: memref<f16>) {
    %0 = memref.load %arg0[] : memref<f8E4M3FN>
    %1 = memref.load %arg1[] : memref<f16>
    %2 = arith.extf %0 : f8E4M3FN to f16
    %3 = arith.addf %1, %2 : f16
    memref.store %3, %arg2[] : memref<f16>
    return
  }
}

//  CHECK-LABEL: @test_1
//   CHECK-SAME: (%[[arg0:.+]]: memref<i8>, %[[arg1:.+]]: memref<f16>, %[[arg2:.+]]: memref<f16>)
//    CHECK-DAG: %[[v0:.+]] = memref.load %[[arg0]][] : memref<i8>
//    CHECK-DAG: %[[v1:.+]] = arith.bitcast %[[v0]] : i8 to f8E4M3FN
//    CHECK-DAG: %[[v2:.+]] = memref.load %[[arg1]][] : memref<f16>
//    CHECK-DAG: %[[v3:.+]] = arith.extf %[[v1]] : f8E4M3FN to f16
//    CHECK-DAG: %[[v4:.+]] = arith.addf %[[v2]], %[[v3]] : f16
//    CHECK-DAG: memref.store %[[v4]], %[[arg2]][] : memref<f16>
//    CHECK-DAG: return

// -----

gpu.module @float_type_conversions {
  func.func @test_2(%arg0: f16) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : f16 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

//  CHECK-LABEL: @test_2
//   CHECK-SAME: (%[[arg0:.+]]: f16) -> i8
//    CHECK-DAG: %[[v0:.+]] = arith.truncf %[[arg0]] : f16 to f8E4M3FN
//    CHECK-DAG: %[[v1:.+]] = arith.bitcast %[[v0]] : f8E4M3FN to i8
//    CHECK-DAG: return %[[v1]] : i8

// -----

gpu.module @float_type_conversions {
  func.func @test_3(%arg0: f8E4M3FN) -> f8E4M3FN {
    %lb = arith.constant 0 : index
    %ub = arith.constant 5 : index
    %step = arith.constant 1 : index
    %sum_0 = arith.constant 1.35 : f8E4M3FN
    %sum = scf.for %iv=%lb to %ub step %step
        iter_args(%sum_iter = %sum_0) -> f8E4M3FN {
            %0 = arith.extf %arg0 : f8E4M3FN to f16
            %1 = arith.extf %sum_iter : f8E4M3FN to f16
            %2 = arith.addf %0, %1 : f16
            %3 = arith.truncf %2 : f16 to f8E4M3FN
            scf.yield %3 : f8E4M3FN
        }
    return %sum : f8E4M3FN
  }
}

// CHECK-LABEL: @test_3
//  CHECK-SAME: (%[[arg0:.+]]: i8) -> i8
//   CHECK-DAG: %[[v0:.+]] = arith.bitcast %[[arg0]] : i8 to f8E4M3FN
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[c5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[cst:.+]] = arith.constant 1.375000e+00 : f8E4M3FN
//   CHECK-DAG: %[[v1:.+]] = arith.bitcast %cst : f8E4M3FN to i8
//   CHECK-DAG: %[[v2:.+]] = scf.for %[[arg1]] = %[[c0]] to %[[c5]] step %[[c1]] iter_args(%[[arg2]] = %[[v1]]) -> (i8)
//   CHECK-DAG: %[[v3:.+]] = arith.bitcast %[[arg2]] : i8 to f8E4M3FN
//   CHECK-DAG: %[[v4:.+]] = arith.extf %[[v0]] : f8E4M3FN to f16
//   CHECK-DAG: %[[v5:.+]] = arith.extf %[[v3]] : f8E4M3FN to f16
//   CHECK-DAG: %[[v6:.+]] = arith.addf %[[v4]], %[[v5]] : f16
//   CHECK-DAG: %[[v7:.+]] = arith.truncf %[[v6]] : f16 to f8E4M3FN
//   CHECK-DAG: %[[v8:.+]] = arith.bitcast %[[v7]] : f8E4M3FN to i8
//   CHECK-DAG: scf.yield %[[v8]] : i8
//        CHECK: return %[[v2]] : i8

// -----

gpu.module @float_type_conversions {
  func.func @cast_to_f16(%arg0: f8E4M3FN) -> f16 {
    %0 = arith.extf %arg0 : f8E4M3FN to f16
    return %0 : f16
  }

  func.func @test_4(%arg0: f8E4M3FN) -> f16 {
    %c = arith.constant 1.2 : f16
    %0 = call @cast_to_f16(%arg0) : (f8E4M3FN) -> (f16)
    %1 = arith.addf %c, %0 : f16
    return %1 : f16
  }
}

// CHECK-LABEL: @cast_to_f16
//  CHECK-SAME: (%[[arg0:.+]]: i8) -> f16
//   CHECK-DAG: %[[v0:.+]] = arith.bitcast %[[arg0]] : i8 to f8E4M3FN
//   CHECK-DAG: %[[v1:.+]] = arith.extf %[[v0]] : f8E4M3FN to f16
//   CHECK-DAG: return %[[v1]] : f16
// CHECK-LABEL: @test_4
//  CHECK-SAME: (%[[arg1:.+]]: i8) -> f16
//   CHECK-DAG: %[[cst:.+]] = arith.constant 1.200200e+00 : f16
//   CHECK-DAG: %[[v0:.+]] = call @cast_to_f16(%[[arg1]]) : (i8) -> f16
//   CHECK-DAG: %[[v1:.+]] = arith.addf %[[cst]], %[[v0]] : f16
//   CHECK-DAG: return %[[v1]] : f16

// -----

gpu.module @float_type_conversions {
  func.func @test_f4_1(%arg0: f16) -> f4E2M1FN {
    %0 = arith.truncf %arg0 : f16 to f4E2M1FN
    return %0 : f4E2M1FN
  }
}

//  CHECK-LABEL: @test_f4_1
//   CHECK-SAME: (%[[arg0:.+]]: f16) -> i4
//    CHECK-DAG: %[[v0:.+]] = arith.truncf %[[arg0]] : f16 to f4E2M1FN
//    CHECK-DAG: %[[v1:.+]] = arith.bitcast %[[v0]] : f4E2M1FN to i4
//    CHECK-DAG: return %[[v1]] : i4

// -----

gpu.module @float_type_conversions {
  func.func @test_f4_2(%arg0: memref<f4E2M1FN>, %arg1: memref<f16>, %arg2: memref<f16>) {
    %0 = memref.load %arg0[] : memref<f4E2M1FN>
    %1 = memref.load %arg1[] : memref<f16>
    %2 = arith.extf %0 : f4E2M1FN to f16
    %3 = arith.addf %1, %2 : f16
    memref.store %3, %arg2[] : memref<f16>
    return
  }
}

//  CHECK-LABEL: @test_f4_2
//   CHECK-SAME: (%[[arg0:.+]]: memref<i4>, %[[arg1:.+]]: memref<f16>, %[[arg2:.+]]: memref<f16>)
//    CHECK-DAG: %[[v0:.+]] = memref.load %[[arg0]][] : memref<i4>
//    CHECK-DAG: %[[v1:.+]] = arith.bitcast %[[v0]] : i4 to f4E2M1FN
//    CHECK-DAG: %[[v2:.+]] = memref.load %[[arg1]][] : memref<f16>
//    CHECK-DAG: %[[v3:.+]] = arith.extf %[[v1]] : f4E2M1FN to f16
//    CHECK-DAG: %[[v4:.+]] = arith.addf %[[v2]], %[[v3]] : f16
//    CHECK-DAG: memref.store %[[v4]], %[[arg2]][] : memref<f16>
//    CHECK-DAG: return