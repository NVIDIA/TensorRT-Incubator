// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-scf | FileCheck %s

func.func @stablehlo_while_to_scf(){
    %init_i = stablehlo.constant dense<1> :tensor<i64>
    %init_sum = stablehlo.constant dense<0> :tensor<i64>
    %one = stablehlo.constant dense<1> :tensor<i64>
    %ten = stablehlo.constant dense<10> :tensor<i64>

    %results0, %results1 = "stablehlo.while"(%init_i, %init_sum) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %cond = "stablehlo.compare"(%arg0, %ten) {
        comparison_direction = #stablehlo<comparison_direction LT>
        } : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %new_sum = stablehlo.add %arg1, %one : tensor<i64>
        %new_i = stablehlo.add %arg0, %one : tensor<i64>
        stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    return
}

// CHECK-LABEL: @stablehlo_while_to_scf
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v4:.+]]:2 = scf.while (%[[arg0:.+]] = %[[v0]], %[[arg1:.+]] = %[[v1]]) : {{.*}} {
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.compare  LT, %[[arg0:.+]], %[[v3]] : {{.*}}
//  CHECK-NEXT: %[[extracted:.+]] = tensor.extract %[[v5]][] : {{.*}}
//  CHECK-NEXT: scf.condition(%[[extracted]]) %[[arg0]], %[[arg1]] : {{.*}}
//  CHECK-NEXT: } do {
//  CHECK-NEXT: ^bb0(%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}):
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.add %[[arg1]], %[[v2]] : {{.*}}
//  CHECK-NEXT: %[[v6:.+]] = stablehlo.add %[[arg0]], %[[v2]] : {{.*}}
//  CHECK-NEXT: scf.yield %[[v6]], %[[v5]] : {{.*}}
//  CHECK: return

// -----

func.func @if_ops_true_branch() {
  %pred = stablehlo.constant dense<true> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    %1 = stablehlo.constant dense<2> : tensor<2xi64>
    %3 = stablehlo.add %0, %1 : tensor<2xi64>
    stablehlo.return %3, %3 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return
}

// CHECK-LABEL: @if_ops_true_branch() {
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<true> : tensor<i1>
//  CHECK-NEXT: %[[extracted:.+]] = tensor.extract %[[v0]][] : tensor<i1>
//  CHECK-NEXT: %[[v1:.+]]:2 = scf.if %[[extracted]] -> {{.*}} {
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant {{.*}}
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.constant{{.*}}
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.add %[[v2]], %[[v3]] : {{.*}}
//  CHECK-NEXT: scf.yield %[[v4]], %[[v4]] : {{.*}}
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant {{.*}}
//  CHECK-NEXT: scf.yield %[[v2]], %[[v2]] : {{.*}}
//       CHECK: return

// -----

func.func @case_one_branch() {
  %index = stablehlo.constant dense<0> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return
}

// CHECK-LABEL: case_one_branch
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant
//  CHECK-NEXT: return

// -----

func.func @case_two_branches() {
  %index = stablehlo.constant dense<0> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = stablehlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  },{stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return
}

// CHECK-LABEL: case_two_branches
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.compare  EQ, %[[v0]], %[[v3]]
//  CHECK-NEXT: %[[extracted:.+]] = tensor.extract %[[v4]][]
//  CHECK-NEXT: %[[v5:.+]]:2 = scf.if %[[extracted]]
//  CHECK-NEXT: scf.yield %[[v1]], %[[v1]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT: scf.yield %[[v2]], %[[v2]]
//       CHECK: return

// -----

func.func @case_three_branches() {
  %index = stablehlo.constant dense<0> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = stablehlo.constant dense<1> : tensor<2xi64>
  %result_branch2 = stablehlo.constant dense<2> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  },{stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  },{stablehlo.return %result_branch2, %result_branch2 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return
}

// CHECK-LABEL: case_three_branches
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.compare  EQ, %[[v0]], %[[v4]]
//  CHECK-NEXT: %[[extracted:.+]] = tensor.extract %[[v5]][]
//  CHECK-NEXT: %[[v6:.+]]:2 = scf.if %[[extracted]]
//  CHECK-NEXT: scf.yield %[[v1]], %[[v1]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[v7:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v8:.+]] = stablehlo.compare  EQ, %[[v0]], %[[v7]]
//  CHECK-NEXT: %[[extracted_0:.+]] = tensor.extract %[[v8]][]
//  CHECK-NEXT: %[[v9:.+]]:2 = scf.if %[[extracted_0]]
//  CHECK-NEXT: scf.yield %[[v2]], %[[v2]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT: scf.yield %[[v3]], %[[v3]]
//       CHECK: scf.yield %[[v9]]#0, %[[v9]]#1
//       CHECK: return