// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

func.func @callee(%arg0: memref<?xindex>) -> memref<?xindex> {
  return %arg0 : memref<?xindex>
}

func.func @caller(%arg0: memref<?xindex>) -> memref<?xindex> {
  %0 = call @callee(%arg0) : (memref<?xindex>) -> memref<?xindex>
  return %0 : memref<?xindex>
}

// CPP-LABEL:  mtrt::RankedMemRef<1> callee
//  CPP-SAME:    (mtrt::RankedMemRef<1> [[v1:.+]])
//       CPP:    return [[v1]];
//       CPP:   mtrt::RankedMemRef<1> caller
//  CPP-SAME:     (mtrt::RankedMemRef<1> [[v1:.+]])
//       CPP:    mtrt::RankedMemRef<1> [[v2:.+]] = callee([[v1]]);
//       CPP:    return [[v2]];

// -----

func.func @callee_multiple_return(%arg0: memref<?xindex>, %arg1: index) -> (memref<?xindex>, f32, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1.0 : f32
  return %arg0, %c1, %c0 : memref<?xindex>, f32, index
}

func.func @caller_multiple_return(%arg0: memref<?xindex>, %arg1: index) -> (memref<?xindex>, f32, index) {
  %0:3 = call @callee_multiple_return(%arg0, %arg1) : (memref<?xindex>, index) -> (memref<?xindex>, f32, index)
  return %0#0, %0#1, %0#2 : memref<?xindex>, f32, index
}

// CPP-LABEL: std::tuple<mtrt::RankedMemRef<1>, float, size_t> callee_multiple_return
//  CPP-SAME:  (mtrt::RankedMemRef<1> [[v1:.+]], size_t [[v2:.+]])
//   CPP-DAG:   size_t [[v3:.+]] = 0;
//   CPP-DAG:   float [[v4:.+]] = 1.000000000e+00f;
//   CPP-DAG:   return std::make_tuple([[v1]], [[v4]], [[v3]]);


// CPP-LABEL: std::tuple<mtrt::RankedMemRef<1>, float, size_t> caller_multiple_return
//  CPP-SAME:  (mtrt::RankedMemRef<1> [[v1:.+]], size_t [[v2:.+]])
//   CPP-DAG:   mtrt::RankedMemRef<1> [[v3:.+]];
//   CPP-DAG:   float [[v4:.+]];
//   CPP-DAG:   size_t [[v5:.+]];
//   CPP-DAG:   std::tie(v3, v4, v5) = callee_multiple_return(v1, v2);
//   CPP-DAG:   return std::make_tuple(v3, v4, v5);
