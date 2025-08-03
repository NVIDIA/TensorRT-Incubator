// RUN: executor-opt %s -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

// This test ensures correct handling of block arguments in the Lua translation.
// Since Lua lacks the concept of block arguments, we simulate them by assigning
// values just before the `goto` that branches to the target block. As a result,
// the live ranges for block arguments must begin at the branch site, not at the
// destination block’s entry. The MLIR builtin liveness analysis does not
// capture this behavior, as it treats the block entry as the start of the live
// range.

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64, !executor.ptr<host> = 64 : i64, !executor.ptr<device> = 64 : i64>, executor.global_init_func = @executor_init_globals} {
  func.func @block_args_allocation(%arg0: i64, %arg1: i64) {
    %0 = executor.icmp <slt> %arg0, %arg1 : i64
    // At this point in translation, %arg0 and %arg1 are live and will be used
    // to assign to block arguments %1 and %2 in bb1. The live ranges of %1 and
    // %2 therefore should start from this branch site. Otherwise, %arg0 and
    // %arg1 can be allocated to locals l0 and l1, and since %arg0 and %arg1
    // have non-overlapping live ranges with %1 and %2, %1 and %2 can reuse the
    // locals, leading to the following error:
    //
    // ^bb0: do
    //   ...
    //   -- Both %arg0 and %0 are allocated to l0.
    //   -- Both %arg1 and %1 are allocated to l1.
    //   l0 = l1 -- assignment to %1 = 1
    //   l1 = l0 -- assignment to %2 = 1, error!
    //   goto ^bb1
    // end
    // ^bb1: do
    //
    // We can see that %2 gets a clobbered value.
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
