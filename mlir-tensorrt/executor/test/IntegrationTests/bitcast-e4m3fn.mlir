// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core

func.func private @e4m3fn_bitcast_to_i8(%arg0: f8E4M3FN)  -> i8 attributes {noinline} {
  %1 = executor.bitcast %arg0 : f8E4M3FN to i8
  return %1 : i8
}

func.func private @i8_bitcast_to_f8E4M3FN(%arg0: i8)  -> f8E4M3FN attributes {noinline} {
  %1 = executor.bitcast %arg0 : i8 to f8E4M3FN
  return %1 : f8E4M3FN
}

func.func private @e4m3fn_bitcast() attributes {noinline} {
  %0 = arith.constant 0x38 : i8
  %1 = call @i8_bitcast_to_f8E4M3FN(%0) : (i8) -> f8E4M3FN
  %2 = call @e4m3fn_bitcast_to_i8(%1) : (f8E4M3FN) -> i8
  %cmp = arith.cmpi eq, %0, %2 : i8
  executor.assert %cmp, "bitcast f8E4M3FN to i8 failed"

  %3 = arith.constant 2.0 : f8E4M3FN
  %4 = call @e4m3fn_bitcast_to_i8(%3) : (f8E4M3FN) -> i8
  %5 = call @i8_bitcast_to_f8E4M3FN(%4) : (i8) -> f8E4M3FN
  %cmp2 = executor.fcmp <oeq> %3, %5 : f8E4M3FN
  executor.assert %cmp2, "bitcast i8 to f8E4M3FN failed"
  return
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  func.call @e4m3fn_bitcast() : () -> ()
  return %c0 : i32
}
