// RUN: executor-translate %s -split-input-file -mlir-to-lua -debug-only=lua-allocation 2>&1 >/dev/null | FileCheck %s

func.func @single_block(%arg0: i32) -> i32 {
  %c1 = executor.constant 1 : i32
  %1 = executor.addi %arg0, %c1 : i32
  %2 = executor.subi %arg0, %c1 : i32
  %3 = executor.addi %1, %2 : i32
  return %3 : i32
}

// CHECK-LABEL: single_block
//  CHECK-NEXT: ^bb0:
//  CHECK-NEXT: |S    executor.constant
//  CHECK-NEXT: ||S   executor.addi
//  CHECK-NEXT: EE|S  executor.subi
//  CHECK-NEXT:   EES executor.addi
//  CHECK-NEXT:     E func.return

// -----

func.func @block_args(%arg0: i32) -> i32 {
  %c1 = executor.constant 1 : i32
  %0 = executor.addi %arg0, %arg0 : i32
  cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):
    %2 = executor.addi %1, %1 : i32
    %3 = executor.subi %1, %2 : i32
    return %3 : i32
}

// CHECK-LABEL: block_args
//  CHECK-NEXT: ^bb0:
//  CHECK-NEXT: |      executor.constant
//  CHECK-NEXT: E S    executor.addi
//  CHECK-NEXT:   ES   cf.br
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:    |S  executor.addi
//  CHECK-NEXT:    EES executor.subi
//  CHECK-NEXT:      E func.return

// -----

func.func @value_with_external_use(%arg0: i32) -> i32 {
  %c1 = executor.constant 1 : i32
  cf.br ^bb0
  ^bb0:
    %1 = executor.addi %arg0, %c1 : i32
    %2 = executor.subi %arg0, %c1 : i32
    cf.br ^bb1
  ^bb1:
    %3 = executor.addi %1, %2 : i32
    return %3 : i32
}

// CHECK-LABEL: value_with_external_use
//  CHECK-NEXT: ^bb0:
//  CHECK-NEXT: |S    executor.constant
//  CHECK-NEXT: ||    cf.br
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT: ||S   executor.addi
//  CHECK-NEXT: EE|S  executor.subi
//  CHECK-NEXT:   ||  cf.br
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:   EES executor.addi
//  CHECK-NEXT:     E func.return

// -----

func.func @for_loop(%arg0: i64, %arg1: i64, %arg2: i64) {
  %c0_i64 = executor.constant 0 : i64
  cf.br ^bb1(%arg0, %c0_i64 : i64, i64)
^bb1(%0: i64, %1: i64):  // 2 preds: ^bb0, ^bb2
  %2 = executor.icmp <slt> %0, %arg1 : i64
  cf.cond_br %2, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %3 = executor.addi %1, %0 : i64
  %4 = executor.addi %0, %arg2 : i64
  cf.br ^bb1(%4, %3 : i64, i64)
^bb3:  // pred: ^bb1
  return
}

// The block dominance tree is:
// bb0 -> bb1 -> bb3
//         `---> bb2
// The block numbering order is:
// bb0 -> bb1 -> bb3 -> bb2

// CHECK-LABEL: for_loop
//  CHECK-NEXT: ^bb0:
//  CHECK-NEXT: |||S      executor.constant
//  CHECK-NEXT: E||ESS    cf.br
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT:  || ||S   executor.icmp
//  CHECK-NEXT:  || ||E   cf.cond_br
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT:  || || S  executor.addi
//  CHECK-NEXT:  || || |S executor.addi
//  CHECK-NEXT:  EE EE EE cf.br
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT:  || ||    func.return

// -----

func.func @nested_loop(%arg0: i64, %arg1: i64, %arg2: i64) {
  %c1_i64 = executor.constant 1 : i64
  %c0_i64 = executor.constant 0 : i64
  cf.br ^bb1(%arg0, %c0_i64 : i64, i64)
^bb1(%0: i64, %1: i64):  // 2 preds: ^bb0, ^bb4
  %2 = executor.icmp <slt> %0, %arg1 : i64
  cf.cond_br %2, ^bb2(%arg0, %1 : i64, i64), ^bb5
^bb2(%3: i64, %4: i64):  // 2 preds: ^bb1, ^bb3
  %5 = executor.icmp <slt> %3, %arg1 : i64
  cf.cond_br %5, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  %6 = executor.addi %4, %c1_i64 : i64
  %7 = executor.addi %3, %arg2 : i64
  cf.br ^bb2(%7, %6 : i64, i64)
^bb4:  // pred: ^bb2
  %8 = executor.addi %0, %arg2 : i64
  cf.br ^bb1(%8, %4 : i64, i64)
^bb5:  // pred: ^bb1
  return
}

// The block dominance tree is:
// bb0 -> bb1 -> bb5
//         `---> bb2 -> bb4
//                `---> bb3
// The block numbering order is:
// bb0 -> bb1 -> bb5 -> bb2 -> bb4 -> bb3

// CHECK-LABEL: nested_loop
//  CHECK-NEXT: ^bb0:
//  CHECK-NEXT: |||S           executor.constant
//  CHECK-NEXT: ||||S          executor.constant
//  CHECK-NEXT: ||||ESS        cf.br
//  CHECK-NEXT: ^bb1:
//  CHECK-NEXT: |||| ||S       executor.icmp
//  CHECK-NEXT: |||| ||ESS     cf.cond_br
//  CHECK-NEXT: ^bb2:
//  CHECK-NEXT: |||| || ||S    executor.icmp
//  CHECK-NEXT: |||| || ||E    cf.cond_br
//  CHECK-NEXT: ^bb3:
//  CHECK-NEXT: |||| |  ||  S  executor.addi
//  CHECK-NEXT: |||| |  ||  |S executor.addi
//  CHECK-NEXT: EEEE E  EE  EE cf.br
//  CHECK-NEXT: ^bb4:
//  CHECK-NEXT: |||| || || S   executor.addi
//  CHECK-NEXT: |||| |E || E   cf.br
//  CHECK-NEXT: ^bb5:
//  CHECK-NEXT: |||| || ||     func.return
