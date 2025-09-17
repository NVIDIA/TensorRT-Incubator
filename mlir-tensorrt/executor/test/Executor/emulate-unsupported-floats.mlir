// RUN: executor-opt --executor-emulate-unsupported-floats -split-input-file %s | FileCheck %s

func.func @test_1(%arg0: f4E2M1FN, %arg1: f4E2M1FN) -> f4E2M1FN{
    %0 = executor.addf %arg0, %arg1 : f4E2M1FN
    return %0 : f4E2M1FN
}

// CHECK-LABEL: func.func @test_1
//  CHECK-SAME: (%[[ARG0:.*]]: f4E2M1FN, %[[ARG1:.*]]: f4E2M1FN) -> f4E2M1FN
//       CHECK: %[[EXT1:.*]] = executor.extf %[[ARG1]] : f4E2M1FN to f16
//       CHECK: %[[EXT0:.*]] = executor.extf %[[ARG0]] : f4E2M1FN to f16
//       CHECK: %[[ADD:.*]] = executor.addf %[[EXT0]], %[[EXT1]] : f16
//       CHECK: %[[TRUNC:.*]] = executor.truncf %[[ADD]] : f16 to f4E2M1FN
//       CHECK: return %[[TRUNC]] : f4E2M1FN

// -----

func.func @test_2(%arg0: f4E2M1FN) -> f4E2M1FN{
    %0 = executor.constant 1.5 : f4E2M1FN
    %1 = executor.subf %0, %arg0 : f4E2M1FN
    return %1 : f4E2M1FN
}

// CHECK-LABEL: func.func @test_2
//  CHECK-SAME: (%[[ARG0:.*]]: f4E2M1FN) -> f4E2M1FN
//       CHECK: %[[EXT0:.*]] = executor.extf %[[ARG0]] : f4E2M1FN to f16
//       CHECK: %[[CST:.*]] = executor.constant 1.500000e+00 : f16
//       CHECK: %[[SUB:.*]] = executor.subf %[[CST]], %[[EXT0]] : f16
//       CHECK: %[[TRUNC:.*]] = executor.truncf %[[SUB]] : f16 to f4E2M1FN
//       CHECK: return %[[TRUNC]] : f4E2M1FN

// -----

func.func @test_3_callee(%arg0: f4E2M1FN) -> f4E2M1FN{
    return %arg0 : f4E2M1FN
}

// CHECK-LABEL: func.func @test_3_callee
//  CHECK-SAME: (%[[ARG0:.*]]: f4E2M1FN) -> f4E2M1FN
//       CHECK: return %[[ARG0]] : f4E2M1FN

func.func @test_3(%arg0: f4E2M1FN) -> f4E2M1FN{
    %0 = func.call @test_3_callee(%arg0):(f4E2M1FN)->(f4E2M1FN)
    return %0: f4E2M1FN
}

// CHECK-LABEL: func.func @test_3
//  CHECK-SAME: (%[[ARG0:.*]]: f4E2M1FN) -> f4E2M1FN
//       CHECK: %[[CALL:.*]] = call @test_3_callee(%[[ARG0]]) : (f4E2M1FN) -> f4E2M1FN
//       CHECK: return %[[CALL]] : f4E2M1FN

// -----

func.func @test_4(%arg0: f4E2M1FN, %arg1: f4E2M1FN) -> f4E2M1FN{
    %0 = executor.extf %arg0 : f4E2M1FN to f16
    %1 = executor.extf %arg1 : f4E2M1FN to f16
    %2 = executor.addf %0, %1 : f16
    %3 = executor.truncf %2 : f16 to f4E2M1FN
    return %3 : f4E2M1FN
}

// CHECK-LABEL: func.func @test_4
//  CHECK-SAME: (%[[ARG0:.*]]: f4E2M1FN, %[[ARG1:.*]]: f4E2M1FN) -> f4E2M1FN
//       CHECK: %[[EXT0:.*]] = executor.extf %[[ARG0]] : f4E2M1FN to f16
//       CHECK: %[[EXT1:.*]] = executor.extf %[[ARG1]] : f4E2M1FN to f16
//       CHECK: %[[ADD:.*]] = executor.addf %[[EXT0]], %[[EXT1]] : f16
//       CHECK: %[[TRUNC:.*]] = executor.truncf %[[ADD]] : f16 to f4E2M1FN
//       CHECK: return %[[TRUNC]] : f4E2M1FN