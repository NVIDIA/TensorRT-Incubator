// RUN: executor-opt %s -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

// RUN: executor-opt %s -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable -lua-translation-block-arg-coalescing=false \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

// This test ensures correct handling of block arguments in the Lua translation.
// When lowering out of MLIR, block arguments are lowered to regular local
// registers/locals slots. During branching, the block arguments of the target
// block must be assigned from a list of source values. This assignment cannot
// be done naively since cycles may be present -- e.g. (l0, l1) = (l1, l0).
// The compiler should emit a sequence of sequential swaps, using a local temporary
// slot as necessary.

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64, !executor.ptr<host> = 64 : i64, !executor.ptr<device> = 64 : i64>, executor.global_init_func = @executor_init_globals} {
  func.func @block_args_allocation(%arg0: i64, %arg1: i64) {
    %0 = executor.icmp <slt> %arg0, %arg1 : i64
    cf.cond_br %0, ^bb1(%arg1, %arg0 : i64, i64), ^bb2
  ^bb1(%1: i64, %2: i64):  // 2 preds: ^bb0, ^bb4
    %3 = executor.addi %1, %2 : i64
    executor.print "block_args_allocation = %d"(%3 : i64)
    return
  ^bb2:  // pred: ^bb0
    %false = executor.constant false
    executor.assert %false, "path should not be taken"
    return
}
func.func @main() -> i64 {
    %c0_i64 = executor.constant 0 : i64
    %c1_i64 = executor.constant 1 : i64
    call @block_args_allocation(%c0_i64, %c1_i64) : (i64, i64) -> ()
    return %c0_i64 : i64
  }
  func.func private @executor_init_globals() {
    return
  }
}

// CHECK: block_args_allocation = 1
