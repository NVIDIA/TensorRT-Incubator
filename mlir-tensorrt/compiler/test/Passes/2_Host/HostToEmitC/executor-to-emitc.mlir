// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc -canonicalize -form-expressions  %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s

// Test Executor ABI recv/send conversion to EmitC

func.func @abi_recv_memref_1(%ptr: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32>>}, %out: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>})
    attributes {executor.func_abi = (memref<10xf32>) -> (memref<10xf32>)} {
  %0 = executor.abi.recv %ptr : memref<10xf32>
  %true = arith.constant true
  executor.abi.send %0 to %out ownership(%true) : memref<10xf32>
  return
}

// CHECK-LABEL: void abi_recv_memref_1
// CHECK-SAME: (mtrt::RankedMemRef<1>* [[arg0:.*]], mtrt::RankedMemRef<1>* [[arg1:.*]]) {
// CHECK-DAG:   int32_t [[v3:.+]] = 0;
// CHECK-DAG:   mtrt::RankedMemRef<1> [[v5:.+]] = *[[arg0]];
//     CHECK:   [[arg1]]{{\[}}[[v3]]] = [[v5]];
//     CHECK:   return;

// -----

func.func @abi_send_f32(%val: f32, %ptr: !executor.ptr<host> {executor.abi = #executor.arg<byref, f32>})
    attributes {executor.func_abi = (f32) -> (f32)} {
  executor.abi.send %val to %ptr : f32
  return
}

// CHECK-LABEL: void abi_send_f32
// CHECK-SAME: (float [[arg0:.*]], float* [[arg1:.*]]) {
// CHECK-DAG:   int32_t [[v3:.+]] = 0;
// CHECK-NEXT:  [[arg1]]{{\[}}[[v3]]] = [[arg0]];
// CHECK-NEXT:  return;

// -----

func.func @abi_send_memref_2(%ptr_in: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<5x10xf32>>},
                            %ptr_out: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5x10xf32>>})
    attributes {executor.func_abi = (memref<5x10xf32>) -> (memref<5x10xf32>)} {
  %val = executor.abi.recv %ptr_in : memref<5x10xf32>

  %output_buffer = executor.abi.recv %ptr_out : memref<5x10xf32>
  executor.abi.send %output_buffer to %ptr_out : memref<5x10xf32>
  return
}

// CHECK-LABEL: void abi_send_memref_2
// CHECK-SAME: (mtrt::RankedMemRef<2>* [[arg0:.*]], mtrt::RankedMemRef<2>* [[arg1:.*]]) {
// CHECK-DAG:   mtrt::RankedMemRef<2> [[v1:.+]] = *[[arg0]];
// CHECK-DAG:   mtrt::RankedMemRef<2> [[v2:.+]] = *[[arg1]];
// CHECK-NEXT:   return;

// -----

// Test executor.print -> printf("<fmt>\n", args...)
func.func @executor_print_no_args() {
  executor.print "hello"()
  return
}

// CHECK-LABEL: void executor_print_no_args
// CHECK:       printf("hello\n");
// CHECK-NEXT:  return;

// -----

func.func @executor_print_with_args(%i: index, %f: f32) {
  executor.print "i=%ld f=%.1f"(%i, %f : index, f32)
  return
}

// CHECK-LABEL: void executor_print_with_args
// CHECK-SAME:  (size_t [[arg0:.*]], float [[arg1:.*]]) {
// CHECK:       printf("i=%ld f=%.1f\n", [[arg0]], [[arg1]]);
// CHECK-NEXT:  return;
