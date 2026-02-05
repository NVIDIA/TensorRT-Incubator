// RUN: executor-opt %s \
// RUN:    -executor-lowering-pipeline | \
// RUN: executor-translate -mlir-to-runtime-executable |\
// RUN: executor-runner -input-type=rtexe -features=core | FileCheck %s

// RUN: executor-opt %s \
// RUN:    --executor-generate-abi-wrappers -inline -executor-lowering-pipeline | \
// RUN: executor-translate -mlir-to-runtime-executable |\
// RUN: executor-runner -input-type=rtexe -features=core | FileCheck %s

func.func @max_local_allocation_test(
  %arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32,
  %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32,
  %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32,
  %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32,
  %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32, %arg48: i32, %arg49: i32,
  %arg50: i32, %arg51: i32, %arg52: i32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: i32, %arg59: i32,
  %arg60: i32, %arg61: i32, %arg62: i32, %arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: i32, %arg69: i32,
  %arg70: i32, %arg71: i32, %arg72: i32, %arg73: i32, %arg74: i32, %arg75: i32, %arg76: i32, %arg77: i32, %arg78: i32, %arg79: i32,
  %arg80: i32, %arg81: i32, %arg82: i32, %arg83: i32, %arg84: i32, %arg85: i32, %arg86: i32, %arg87: i32, %arg88: i32, %arg89: i32,
  %arg90: i32, %arg91: i32, %arg92: i32, %arg93: i32, %arg94: i32, %arg95: i32, %arg96: i32, %arg97: i32, %arg98: i32, %arg99: i32,
  %arg100: i32, %arg101: i32, %arg102: i32, %arg103: i32, %arg104: i32, %arg105: i32, %arg106: i32, %arg107: i32, %arg108: i32, %arg109: i32,
  %arg110: i32, %arg111: i32, %arg112: i32, %arg113: i32, %arg114: i32, %arg115: i32, %arg116: i32, %arg117: i32, %arg118: i32, %arg119: i32,
  %arg120: i32, %arg121: i32, %arg122: i32, %arg123: i32, %arg124: i32, %arg125: i32, %arg126: i32, %arg127: i32, %arg128: i32, %arg129: i32,
  %arg130: i32, %arg131: i32, %arg132: i32, %arg133: i32, %arg134: i32, %arg135: i32, %arg136: i32, %arg137: i32, %arg138: i32, %arg139: i32,
  %arg140: i32, %arg141: i32, %arg142: i32, %arg143: i32, %arg144: i32, %arg145: i32, %arg146: i32, %arg147: i32, %arg148: i32, %arg149: i32,
  %arg150: i32, %arg151: i32, %arg152: i32, %arg153: i32, %arg154: i32, %arg155: i32, %arg156: i32, %arg157: i32, %arg158: i32, %arg159: i32,
  %arg160: i32, %arg161: i32, %arg162: i32, %arg163: i32, %arg164: i32, %arg165: i32, %arg166: i32, %arg167: i32, %arg168: i32, %arg169: i32,
  %arg170: i32, %arg171: i32, %arg172: i32, %arg173: i32, %arg174: i32, %arg175: i32, %arg176: i32, %arg177: i32, %arg178: i32, %arg179: i32,
  %arg180: i32, %arg181: i32, %arg182: i32, %arg183: i32, %arg184: i32, %arg185: i32, %arg186: i32, %arg187: i32, %arg188: i32, %arg189: i32,
  %arg190: i32, %arg191: i32, %arg192: i32, %arg193: i32, %arg194: i32, %arg195: i32, %arg196: i32, %arg197: i32, %arg198: i32) -> i32 {
  %0 = executor.addi %arg0, %arg11 : i32
  %1 = executor.addi %0, %arg22 : i32
  %2 = executor.addi %1, %arg33 : i32
  %3 = executor.addi %2, %arg44 : i32
  %4 = executor.addi %3, %arg55 : i32
  %5 = executor.addi %4, %arg66 : i32
  %6 = executor.addi %5, %arg77 : i32
  %7 = executor.addi %6, %arg88 : i32
  %8 = executor.addi %7, %arg99 : i32
  %9 = executor.addi %8, %arg110 : i32
  %10 = executor.addi %9, %arg121 : i32
  %11 = executor.addi %10, %arg132 : i32
  %12 = executor.addi %11, %arg143 : i32
  %13 = executor.addi %12, %arg154 : i32
  %14 = executor.addi %13, %arg165 : i32
  %15 = executor.addi %14, %arg176 : i32
  %16 = executor.addi %15, %arg187 : i32
  %17 = executor.addi %16, %arg198 : i32
  executor.print "max_local_allocation_test = %d"(%17 : i32)
  return %17 : i32
}

func.func @main() -> i32 {
  %c0 = executor.constant 0 : i32
  %c1 = executor.constant 1 : i32
  %c2 = executor.constant 2 : i32
  %c3 = executor.constant 3 : i32
  %c4 = executor.constant 4 : i32
  %c5 = executor.constant 5 : i32
  %c6 = executor.constant 6 : i32
  %c7 = executor.constant 7 : i32
  %c8 = executor.constant 8 : i32
  %c9 = executor.constant 9 : i32
  %0 = func.call @max_local_allocation_test(
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9,
    %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8) :
    (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
     i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
  return %0 : i32
}

// CHECK: max_local_allocation_test = 81
