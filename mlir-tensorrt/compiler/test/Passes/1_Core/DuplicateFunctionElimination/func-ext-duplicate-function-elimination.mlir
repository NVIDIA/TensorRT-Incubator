// RUN: mlir-tensorrt-opt -func-ext-duplicate-function-elimination -split-input-file -allow-unregistered-dialect %s | FileCheck %s

func.func private @some_external_func(i32, i32) -> i32
func.func private @some_external_func2(i32, i32) -> i32

func.func @some_caller(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %0 = func.call @some_external_func(%arg0, %arg1) : (i32, i32) -> i32
  %1 = func.call @some_external_func2(%arg0, %arg1) : (i32, i32) -> i32
  return %0, %1 : i32, i32
}

//       CHECK: func.func private @some_external_func
//       CHECK: func.func private @some_external_func2
// CHECK-LABEL: func.func @some_caller
//       CHECK: call @some_external_func(
//       CHECK: call @some_external_func2(

// -----

func.func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

func.func @also_identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

builtin.module @sub_module {
  func.func @identity(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
  func.func @yet_another_identity(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}

func.func @user(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = call @identity(%arg0) {some_ref = @sub_module::@yet_another_identity} : (tensor<f32>) -> tensor<f32>
  %1 = call @also_identity(%0) {some_ref = @also_identity, nested_ref = @sub_module::@identity} : (tensor<f32>) -> tensor<f32>
  %2 = "unknown.call_op"(%0) {callee = @sub_module::@yet_another_identity, some_ref = @also_identity} : (tensor<f32>) -> (tensor<f32>)
  %3 = "unknown.call_op"(%0) {callee = @sub_module::@identity, some_ref = @also_identity} : (tensor<f32>) -> (tensor<f32>)
  return %3 : tensor<f32>
}

//       CHECK: func.func @identity(
//   CHECK-NOT: @also_identity
//       CHECK: module @sub_module
//  CHECK-NEXT:    func.func @identity
//   CHECK-NOT: @yet_another_identity
// CHECK-LABEL: func.func @user
//  CHECK-NEXT:     call @identity(%{{.+}}) {some_ref = @sub_module::@identity}
//  CHECK-NEXT:     call @identity(%{{.+}}) {nested_ref = @sub_module::@identity, some_ref = @identity}
//  CHECK-NEXT:     "unknown.call_op"(%{{.+}}) {callee = @sub_module::@identity, some_ref = @identity}
//  CHECK-NEXT:     "unknown.call_op"(%{{.+}}) {callee = @sub_module::@identity, some_ref = @identity}
//  CHECK-NEXT:     return
