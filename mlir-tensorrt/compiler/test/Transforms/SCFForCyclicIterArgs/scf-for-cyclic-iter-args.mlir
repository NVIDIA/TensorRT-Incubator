// RUN: mlir-tensorrt-opt %s -mtrt-scf-for-cyclic-iter-args -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func @cyclic_groups
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ITER:.*]] = %arg5) -> (i32) {
// CHECK:   %[[DIFF:.*]] = arith.subi %[[IV]], %[[C0]] : index
// CHECK:   %[[DIV:.*]] = arith.divui %[[DIFF]], %[[C1]] : index
// CHECK:   %[[C3_IN:.*]] = arith.constant 3 : index
// CHECK:   %[[REM3:.*]] = arith.remui %[[DIV]], %[[C3_IN]] : index
// CHECK:   %[[C1_IN:.*]] = arith.constant 1 : index
// CHECK:   %[[CMP1:.*]] = arith.cmpi eq, %[[REM3]], %[[C1_IN]] : index
// CHECK:   %[[SEL1:.*]] = arith.select %[[CMP1]], %arg1, %arg0 : i32
// CHECK:   %[[C2_IN:.*]] = arith.constant 2 : index
// CHECK:   %[[CMP2:.*]] = arith.cmpi eq, %[[REM3]], %[[C2_IN]] : index
// CHECK:   %[[SEL2:.*]] = arith.select %[[CMP2]], %arg2, %[[SEL1]] : i32
// CHECK:   arith.select %{{.*}}, %arg0, %arg2 : i32
// CHECK:   %[[REM2:.*]] = arith.remui %[[DIV]], %{{.*}} : index
// CHECK:   %[[CMP3:.*]] = arith.cmpi eq, %[[REM2]], %{{.*}} : index
// CHECK:   %[[SEL3:.*]] = arith.select %[[CMP3]], %arg4, %arg3 : i32
// CHECK:   %[[ADD:.*]] = arith.addi %[[SEL2]], %[[SEL3]] : i32
// CHECK:   %[[NEXT:.*]] = arith.addi %[[ITER]], %{{.*}} : i32
// CHECK:   scf.yield %[[NEXT]] : i32
// CHECK: }
// CHECK: %[[DIFF2:.*]] = arith.subi %[[C4]], %[[C0]] : index
// CHECK: %[[CEIL:.*]] = arith.ceildivui %[[DIFF2]], %[[C1]] : index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[HAS_ITERS:.*]] = arith.cmpi ugt, %[[C4]], %[[C0]] : index
// CHECK: %[[TRIP:.*]] = arith.select %[[HAS_ITERS]], %[[CEIL]], %[[ZERO]] : index
// CHECK: %[[C3_OUT:.*]] = arith.constant 3 : index
// CHECK: %[[REM_OUT:.*]] = arith.remui %[[TRIP]], %[[C3_OUT]] : index
// CHECK: %[[C2_OUT:.*]] = arith.constant 2 : index
// CHECK: %[[CMP4:.*]] = arith.cmpi eq, %[[REM_OUT]], %[[C2_OUT]] : index
// CHECK: %[[SEL4:.*]] = arith.select %[[CMP4]], %arg2, %{{.*}} : i32
// CHECK: arith.select %{{.*}}, %arg1, %{{.*}} : i32
// CHECK: %[[REM_OUT2:.*]] = arith.remui %[[TRIP]], %{{.*}} : index
// CHECK: return %[[SEL4]], %{{.*}}, %[[LOOP]] : i32, i32, i32
func.func @cyclic_groups(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32,
                         %arg4: i32, %arg5: i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %0:6 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%iter0 = %arg0, %iter1 = %arg1, %iter2 = %arg2,
                %iter3 = %arg3, %iter4 = %arg4, %iter5 = %arg5)
      -> (i32, i32, i32, i32, i32, i32) {
    %tmp = arith.addi %iter0, %iter3 : i32
    %next5 = arith.addi %iter5, %c1_i32 : i32
    scf.yield %iter1, %iter2, %iter0, %iter4, %iter3, %next5
      : i32, i32, i32, i32, i32, i32
  }
  return %0#0, %0#3, %0#5 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @full_cycle
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:   %[[DIFF:.*]] = arith.subi %[[IV]], %[[C0]] : index
// CHECK:   %[[DIV:.*]] = arith.divui %[[DIFF]], %[[C1]] : index
// CHECK:   %[[C2_IN:.*]] = arith.constant 2 : index
// CHECK:   %[[REM:.*]] = arith.remui %[[DIV]], %[[C2_IN]] : index
// CHECK:   %[[C1_IN:.*]] = arith.constant 1 : index
// CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[REM]], %[[C1_IN]] : index
// CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %arg1, %arg0 : i32
// CHECK: }
// CHECK: %[[DIFF2:.*]] = arith.subi %[[C2]], %[[C0]] : index
// CHECK: %[[CEIL:.*]] = arith.ceildivui %[[DIFF2]], %[[C1]] : index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[HAS:.*]] = arith.cmpi ugt, %[[C2]], %[[C0]] : index
// CHECK: %[[TRIP:.*]] = arith.select %[[HAS]], %[[CEIL]], %[[ZERO]] : index
// CHECK: %[[C2_OUT:.*]] = arith.constant 2 : index
// CHECK: %[[REM_OUT:.*]] = arith.remui %[[TRIP]], %[[C2_OUT]] : index
// CHECK: %[[C1_OUT:.*]] = arith.constant 1 : index
// CHECK: %[[CMP2:.*]] = arith.cmpi eq, %[[REM_OUT]], %[[C1_OUT]] : index
// CHECK: %[[SEL2:.*]] = arith.select %[[CMP2]], %arg1, %arg0 : i32
// CHECK: return %[[SEL2]], %{{.*}} : i32, i32
func.func @full_cycle(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0:2 = scf.for %i = %c0 to %c2 step %c1
      iter_args(%iter0 = %arg0, %iter1 = %arg1) -> (i32, i32) {
    scf.yield %iter1, %iter0 : i32, i32
  }
  return %0#0, %0#1 : i32, i32
}
