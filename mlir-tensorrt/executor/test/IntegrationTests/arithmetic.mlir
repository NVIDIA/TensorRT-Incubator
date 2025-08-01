// RUN: rm %t.mlir || true
// RUN: executor-opt %s -executor-lowering-pipeline -o %t.mlir

// RUN: executor-translate -mlir-to-lua  %t.mlir \
// RUN:   | executor-runner -input-type=lua -features=core | FileCheck %s

// RUN: executor-translate -mlir-to-runtime-executable %t.mlir \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

func.func @test_addi(%arg0: i64, %arg1: i64) {
  %0 = executor.addi %arg0, %arg1 : i64
  executor.print "%d addi %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_addf(%arg0: f64, %arg1: f64) {
  %0 = executor.addf %arg0, %arg1 : f64
  executor.print "%f addf %f = %f"(%arg0, %arg1, %0 : f64, f64, f64)
  return
}

func.func @test_subi(%arg0: i64, %arg1: i64) {
  %0 = executor.subi %arg0, %arg1 : i64
  executor.print "%d subi %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_subf(%arg0: f64, %arg1: f64) {
  %0 = executor.subf %arg0, %arg1 : f64
  executor.print "%f subf %f = %f"(%arg0, %arg1, %0 : f64, f64, f64)
  return
}

func.func @test_muli(%arg0: i64, %arg1: i64) {
  %0 = executor.muli %arg0, %arg1 : i64
  executor.print "%d muli %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_mulf(%arg0: f64, %arg1: f64) {
  %0 = executor.mulf %arg0, %arg1 : f64
  executor.print "%f mulf %f = %f"(%arg0, %arg1, %0 : f64, f64, f64)
  return
}

func.func @test_mulf16(%arg0: f16, %arg1: f16) {
  %0 = executor.mulf %arg0, %arg1 : f16
  executor.print "%s mulf(f16) %s = %s"(%arg0, %arg1, %0 : f16, f16, f16)
  return
}

func.func @test_addf16(%arg0: f16, %arg1: f16) {
  %0 = executor.addf %arg0, %arg1 : f16
  executor.print "%s addf(f16) %s = %s"(%arg0, %arg1, %0 : f16, f16, f16)
  return
}

func.func @test_subf16(%arg0: f16, %arg1: f16) {
  %0 = executor.subf %arg0, %arg1 : f16
  executor.print "%s subf(f16) %s = %s"(%arg0, %arg1, %0 : f16, f16, f16)
  return
}

func.func @test_divf16(%arg0: f16, %arg1: f16) {
  %0 = executor.divf %arg0, %arg1 : f16
  executor.print "%s divf(f16) %s = %s"(%arg0, %arg1, %0 : f16, f16, f16)
  return
}

func.func @test_sdivi(%arg0: i64, %arg1: i64) {
  %0 = executor.sdivi %arg0, %arg1 : i64
  executor.print "%d sdivi %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_divf(%arg0: f64, %arg1: f64) {
  %0 = executor.divf %arg0, %arg1 : f64
  executor.print "%f divf %f = %f"(%arg0, %arg1, %0 : f64, f64, f64)
  return
}

func.func @test_sfloor_divi(%arg0: i64, %arg1: i64) {
  %0 = executor.sfloor_divi %arg0, %arg1 : i64
  executor.print "%d sfloor_divi %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_smax(%arg0: i64, %arg1: i64) {
  %0 = executor.smax %arg0, %arg1 : i64
  executor.print "%d smax %d = %f"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_fmax(%arg0: f64, %arg1: f64) {
  %0 = executor.fmax %arg0, %arg1 : f64
  executor.print "%f fmax %f = %f"(%arg0, %arg1, %0 : f64, f64, f64)
  return
}

func.func @test_shift_lefti(%arg0: i64, %arg1: i64) {
  %0 = executor.shift_lefti %arg0, %arg1 : i64
  executor.print "%d shift_left %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_shift_right_logicali(%arg0: i64, %arg1: i64) {
  %0 = executor.shift_right_logicali %arg0, %arg1 : i64
  executor.print "%d shift_right_logical %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_shift_right_arithmetici(%arg0: i64, %arg1: i64) {
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i64
  executor.print "%d shift_right_arithmetic %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_bitwise_and(%arg0: i64, %arg1: i64) {
  %0 = executor.bitwise_andi %arg0, %arg1 : i64
  executor.print "%d bitwise_and %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_bitwise_and_i1(%arg0: i1, %arg1: i1) {
  %0 = executor.bitwise_andi %arg0, %arg1 : i1
  executor.print "%d bitwise_and_i1 %d = %d"(%arg0, %arg1, %0 : i1, i1, i1)
  return
}

func.func @test_bitwise_or(%arg0: i64, %arg1: i64) {
  %0 = executor.bitwise_ori %arg0, %arg1 : i64
  executor.print "%d bitwise_or %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_bitwise_or_i1(%arg0: i1, %arg1: i1) {
  %0 = executor.bitwise_ori %arg0, %arg1 : i1
  executor.print "%d bitwise_or_i1 %d = %d"(%arg0, %arg1, %0 : i1, i1, i1)
  return
}

func.func @test_bitwise_xor(%arg0: i64, %arg1: i64) {
  %0 = executor.bitwise_xori %arg0, %arg1 : i64
  executor.print "%d bitwise_xor %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_bitwise_xor_i1(%arg0: i1, %arg1: i1) {
  %0 = executor.bitwise_xori %arg0, %arg1 : i1
  executor.print "%d bitwise_xor_i1 %d = %d"(%arg0, %arg1, %0 : i1, i1, i1)
  return
}

func.func @test_negf(%arg0: f64) {
  %0 = executor.negf %arg0 : f64
  executor.print "negative %f = %f"(%arg0, %0 : f64, f64)
  return
}

func.func @test_absi(%arg0: i64) {
  %0 = executor.absi %arg0 : i64
  executor.print "abs %d = %d"(%arg0, %0 : i64, i64)
  return
}

func.func @test_absf(%arg0: f64) {
  %0 = executor.absf %arg0 : f64
  executor.print "abs %f = %f"(%arg0, %0 : f64, f64)
  return
}

func.func @test_sqrt(%arg0: f64) {
  %0 = executor.sqrt %arg0 : f64
  executor.print "sqrt %d = %d"(%arg0, %0 : f64, f64)
  return
}

func.func @test_log1p(%arg0: f64) {
  %0 = executor.log1p %arg0 : f64
  executor.print "log1p %f = %f"(%arg0, %0 : f64, f64)
  return
}

func.func @test_srem(%arg0: i64, %arg1: i64) {
  %0 = executor.sremi %arg0, %arg1 : i64
  executor.print "%d srem %d = %d"(%arg0, %arg1, %0 : i64, i64, i64)
  return
}

func.func @test_select(%pred: i1, %arg0: i64, %arg1: i64) {
  %0 = executor.select %pred, %arg0, %arg1 : i64
  executor.print "select %d %d %d = %d"(%pred, %arg0, %arg1, %0 : i1, i64, i64, i64)
  return
}

func.func @test_selectf(%pred: i1, %arg0: f64, %arg1: f64) {
  %0 = executor.select %pred, %arg0, %arg1 : f64
  executor.print "select %d %d %d = %d"(%pred, %arg0, %arg1, %0 : i1, f64, f64, f64)
  return
}

func.func @test_bitcast_f32_i32(%arg0: f32) {
  %0 = executor.bitcast %arg0 : f32 to i32
  executor.print "bitcast f32 to i32 of %f = %d"(%arg0, %0 : f32, i32)
  return
}

func.func @test_bitcast_i32_f32(%arg0: i32) {
  %0 = executor.bitcast %arg0 : i32 to f32
  executor.print "bitcast i32 to f32 of %d = %f"(%arg0, %0 : i32, f32)
  return
}

func.func @test_bitcast_f64_i64(%arg0: f64) {
  %0 = executor.bitcast %arg0 : f64 to i64
  executor.print "bitcast f64 to i64 of %f = %d"(%arg0, %0 : f64, i64)
  return
}

func.func @test_bitcast_i64_f64(%arg0: i64) {
  %0 = executor.bitcast %arg0 : i64 to f64
  executor.print "bitcast i64 to f64 of %d = %f"(%arg0, %0 : i64, f64)
  return
}

func.func @test_sitofp(%arg0: i64, %arg1: i32, %arg2: i16,
    %arg3: i8) {
  %0 = executor.sitofp %arg0 : i64 to f64
  executor.print "sitofp i64 to f64 of %d = %f"(%arg0, %0 : i64, f64)
  %1 = executor.sitofp %arg0: i64 to f32
  executor.print "sitofp i64 to f32 of %d = %f"(%arg0, %1 : i64, f32)
  %2 = executor.sitofp %arg1 : i32 to f64
  executor.print "sitofp i32 to f64 of %d = %f"(%arg1, %2 : i32, f64)
  %3 = executor.sitofp %arg1: i32 to f32
  executor.print "sitofp i32 to f32 of %d = %f"(%arg1, %3 : i32, f32)
  %4 = executor.sitofp %arg2 : i16 to f64
  executor.print "sitofp i16 to f64 of %d = %f"(%arg2, %4 : i16, f64)
  %5 = executor.sitofp %arg2: i16 to f32
  executor.print "sitofp i16 to f32 of %d = %f"(%arg2, %5 : i16, f32)
  %6 = executor.sitofp %arg3 : i8 to f64
  executor.print "sitofp i8 to f64 of %d = %f"(%arg3, %6 : i8, f64)
  %7 = executor.sitofp %arg3: i8 to f32
  executor.print "sitofp i8 to f32 of %d = %f"(%arg3, %7 : i8, f32)
  return
}

func.func @test_fptosi(%arg0: f64, %arg1: f32) {
  %0 = executor.fptosi %arg0 : f64 to i64
  executor.print "fptosi f64 to i64 of %f = %d"(%arg0, %0 : f64, i64)
  %1 = executor.fptosi %arg0 : f64 to i32
  executor.print "fptosi f64 to i32 of %f = %d"(%arg0, %1 : f64, i32)
  %2 = executor.fptosi %arg0 : f64 to i16
  executor.print "fptosi f64 to i16 of %f = %d"(%arg0, %2 : f64, i16)
  %3 = executor.fptosi %arg0 : f64 to i8
  executor.print "fptosi f64 to i8 of %f = %d"(%arg0, %3 : f64, i8)
  %4 = executor.fptosi %arg1 : f32 to i64
  executor.print "fptosi f32 to i64 of %f = %d"(%arg1, %4 : f32, i64)
  %5 = executor.fptosi %arg1 : f32 to i32
  executor.print "fptosi f32 to i32 of %f = %d"(%arg1, %5 : f32, i32)
  %6 = executor.fptosi %arg1 : f32 to i16
  executor.print "fptosi f32 to i16 of %f = %d"(%arg1, %6 : f32, i16)
  %7 = executor.fptosi %arg1 : f32 to i8
  executor.print "fptosi f64 to i8 of %f = %d"(%arg1, %7 : f32, i8)
  return
}

func.func @test_icmp_if(%lhs: i64, %rhs: i64) {
  %c1 = arith.constant 1 : i1
  %ic0 = arith.constant 0 : i1
  %condEq = executor.icmp <eq> %lhs, %rhs: i64
  %eq = scf.if %condEq -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d eq %d = %q"(%lhs, %rhs, %eq : i64, i64, i1)

  %condNe = executor.icmp <ne> %lhs, %rhs: i64
  %ne = scf.if %condNe -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d ne %d = %q"(%lhs, %rhs, %ne : i64, i64, i1)

  %condSgt = executor.icmp <sgt> %lhs, %rhs: i64
  %sgt = scf.if %condSgt -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d sgt %d = %q"(%lhs, %rhs, %sgt : i64, i64, i1)

  %condSge = executor.icmp <sge> %lhs, %rhs: i64
  %sge = scf.if %condSge -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d sge %d = %q"(%lhs, %rhs, %sge : i64, i64, i1)

  return
}

func.func @test_icmp_ugt_i32(%lhs: i32, %rhs: i32) {
  %c1 = arith.constant 1 : i1
  %ic0 = arith.constant 0 : i1
  %cond = executor.icmp <ugt> %lhs, %rhs: i32
  %eq = scf.if %cond -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d ugt %d = %q"(%lhs, %rhs, %eq : i32, i32, i1)
  %cond1 = executor.icmp <ugt> %rhs, %lhs: i32
  %eq1 = scf.if %cond1 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d ugt %d = %q"(%rhs, %lhs, %eq1 : i32, i32, i1)
  return
}

func.func @test_icmp_ult_i32(%lhs: i32, %rhs: i32) {
  %c1 = arith.constant 1 : i1
  %ic0 = arith.constant 0 : i1
  %cond = executor.icmp <ult> %lhs, %rhs: i32
  %eq = scf.if %cond -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d ult %d = %q"(%lhs, %rhs, %eq : i32, i32, i1)
  %cond1 = executor.icmp <ult> %rhs, %lhs: i32
  %eq1 = scf.if %cond1 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%d ult %d = %q"(%rhs, %lhs, %eq1 : i32, i32, i1)
  return
}

func.func @test_cmpf(%arg0: f32, %arg1: f32)
      -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %0 = executor.fcmp <_false> %arg0, %arg1 : f32
  %1 = executor.fcmp <oeq> %arg0, %arg1 : f32
  %2 = executor.fcmp <ogt> %arg0, %arg1 : f32
  %3 = executor.fcmp <oge> %arg0, %arg1 : f32
  %4 = executor.fcmp <olt> %arg0, %arg1 : f32
  %5 = executor.fcmp <ole> %arg0, %arg1 : f32
  %6 = executor.fcmp <one> %arg0, %arg1 : f32
  %7 = executor.fcmp <ord> %arg0, %arg1 : f32
  %8 = executor.fcmp <ueq> %arg0, %arg1 : f32
  %9 = executor.fcmp <ugt> %arg0, %arg1 : f32
  %10 = executor.fcmp <uge> %arg0, %arg1 : f32
  %11 = executor.fcmp <ult> %arg0, %arg1 : f32
  %12 = executor.fcmp <ule> %arg0, %arg1 : f32
  %13 = executor.fcmp <une> %arg0, %arg1 : f32
  %14 = executor.fcmp <uno> %arg0, %arg1 : f32
  %15 = executor.fcmp <_true> %arg0, %arg1 : f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

func.func @print_test_cmpf(%arg0: f32, %arg1: f32) {
  %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 =
    func.call @test_cmpf(%arg0, %arg1) : (f32, f32) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)
  executor.print "comparing %f and %f"(%arg0, %arg1: f32, f32)
  executor.print "false=%d, oeq=%d, ogt=%d, oge=%d, olt=%d, ole=%d, one=%d, ord=%d, ueq=%d, ugt=%d, uge=%d, ult=%d, ule=%d, une=%d, uno=%d, true=%d"
    (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)
  return
}

func.func @print_test_zext_i1(%arg0: i1) {
  %0 = arith.extui %arg0 : i1 to i32
  executor.print "zext i1 %d to i32 %d"(%arg0, %0 : i1, i32)
  return
}

func.func @test_trunci_i64_to_i32(%arg0: i64) {
  %0 = executor.trunc %arg0: i64 to i32
  executor.print "trunci i64 0x%x to i32 = %d"(%arg0, %0 : i64, i32)
  return
}

func.func @test_trunci_i64_to_i1(%arg0: i64) {
  %0 = executor.trunc %arg0: i64 to i1
  executor.print "trunci i64 0x%x to i1 = %d"(%arg0, %0 : i64, i1)
  return
}

func.func @test_trunci_i32_to_i1(%arg0: i32) {
  %0 = executor.trunc %arg0: i32 to i1
  executor.print "trunci i32 0x%x to i1 = %d"(%arg0, %0 : i32, i1)
  return
}


func.func @test_trunci_i8_to_i1(%arg0: i8) {
  %0 = executor.trunc %arg0: i8 to i1
  executor.print "trunci i8 0x%x to i1 = 0x%x"(%arg0, %0 : i8, i1)
  return
}

func.func @print_test_zext_i8(%arg0: i8) {
  %0 = arith.extui %arg0 : i8 to i32
  executor.print "zext i8 %d to i32 %d"(%arg0, %0 : i8, i32)
  return
}

func.func @print_test_siext_i1(%arg0: i1) {
  %0 = arith.extsi %arg0 : i1 to i32
  executor.print "siext i1 %d to i32 %d"(%arg0, %0 : i1, i32)
  return
}

func.func @print_test_siext_i8(%arg0: i8) {
  %0 = arith.extsi %arg0 : i8 to i32
  executor.print "siext i8 %d to i32 %d"(%arg0, %0 : i8, i32)
  return
}


func.func @test_cmpi1() {
  %c0 = arith.constant 0 : i1
  %c1 = arith.constant 1 : i1

  %ne = arith.cmpi ne, %c0, %c1 : i1
  %eq = arith.cmpi eq, %c0, %c1 : i1

  executor.print "%d != %d = %d"(%c0, %c1, %ne: i1, i1, i1)
  executor.print "%d == %d = %d"(%c0, %c1, %eq: i1, i1, i1)

  return
}

// FP8 (f8E4M3) Tests

func.func @test_base_f8(%arg0: f8E4M3FN, %arg1: f8E4M3FN) {
  %0 = executor.addf %arg0, %arg1 : f8E4M3FN
  executor.print "%s add_f8 %s = %s"(%arg0, %arg1, %0 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  %1 = executor.subf %arg0, %arg1 : f8E4M3FN
  executor.print "%s sub_f8 %s = %s"(%arg0, %arg1, %1 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  %2 = executor.mulf %arg0, %arg1 : f8E4M3FN
  executor.print "%s mul_f8 %s = %s"(%arg0, %arg1, %2 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  %3 = executor.divf %arg0, %arg1 : f8E4M3FN
  executor.print "%s div_f8 %s = %s"(%arg0, %arg1, %3 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  %4 = executor.fmax %arg0, %arg1 : f8E4M3FN
  executor.print "%s fmax_f8 %s = %s"(%arg0, %arg1, %4 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  %5 = executor.fmin %arg0, %arg1 : f8E4M3FN
  executor.print "%s fmin_f8 %s = %s"(%arg0, %arg1, %5 : f8E4M3FN, f8E4M3FN, f8E4M3FN)
  return
}

func.func @test_nan_cast_f8(%arg0: f8E4M3FN, %arg2: f16, %arg3: f32){
  %0 = executor.extf %arg0: f8E4M3FN to f16
  executor.print "%s extf_f8E4M3FN_f16 = %s"(%arg0, %0 : f8E4M3FN, f16)
  %1 = executor.extf %arg0: f8E4M3FN to f32
  executor.print "%s extf_f8E4M3FN_f32 = %s"(%arg0, %1 : f8E4M3FN, f32)
  %2 = executor.truncf %arg2: f16 to f8E4M3FN
  executor.print "%s truncf_f16_f8E4M3FN = %s"(%arg2, %2 : f16, f8E4M3FN)
  %3 = executor.truncf %arg3: f32 to f8E4M3FN
  executor.print "%s truncf_f32_f8E4M3FN = %s"(%arg3, %3 : f32, f8E4M3FN)
  return
}

func.func @test_sitofp_f8(%arg0: i64, %arg1: i32, %arg2: i16,
    %arg3: i8) {
  %0 = executor.sitofp %arg0 : i64 to f8E4M3FN
  executor.print "sitofp i64 to f8E4M3FN of %d = %s"(%arg0, %0 : i64, f8E4M3FN)
  %2 = executor.sitofp %arg1 : i32 to f8E4M3FN
  executor.print "sitofp i32 to f8E4M3FN of %d = %s"(%arg1, %2 : i32, f8E4M3FN)
  %4 = executor.sitofp %arg2 : i16 to f8E4M3FN
  executor.print "sitofp i16 to f8E4M3FN of %d = %s"(%arg2, %4 : i16, f8E4M3FN)
  %6 = executor.sitofp %arg3 : i8 to f8E4M3FN
  executor.print "sitofp i8 to f8E4M3FN of %d = %s"(%arg3, %6 : i8, f8E4M3FN)
  return
}

func.func @test_fptosi_f8(%arg0: f8E4M3FN) {
  %0 = executor.fptosi %arg0 : f8E4M3FN to i64
  executor.print "fptosi f8E4M3FN to i64 of %s = %d"(%arg0, %0 : f8E4M3FN, i64)
  %1 = executor.fptosi %arg0 : f8E4M3FN to i32
  executor.print "fptosi f8E4M3FN to i32 of %s = %d"(%arg0, %1 : f8E4M3FN, i32)
  %2 = executor.fptosi %arg0 : f8E4M3FN to i16
  executor.print "fptosi f8E4M3FN to i16 of %s = %d"(%arg0, %2 : f8E4M3FN, i16)
  %3 = executor.fptosi %arg0 : f8E4M3FN to i8
  executor.print "fptosi f8E4M3FN to i8 of %s = %d"(%arg0, %3 : f8E4M3FN, i8)
  return
}

func.func @test_unary_f8(%arg0: f8E4M3FN){
  %0 = executor.negf %arg0 : f8E4M3FN
  executor.print "negative_f8 %s = %s"(%arg0, %0 : f8E4M3FN, f8E4M3FN)
  %1 = executor.sqrt %arg0 : f8E4M3FN
  executor.print "sqrt_f8 %s = %s"(%arg0, %1 : f8E4M3FN, f8E4M3FN)
  %2 = executor.log1p %arg0 : f8E4M3FN
  executor.print "log1p_f8 %s = %s"(%arg0, %2 : f8E4M3FN, f8E4M3FN)
  %3 = executor.log10 %arg0 : f8E4M3FN
  executor.print "log10_f8 %s = %s"(%arg0, %3 : f8E4M3FN, f8E4M3FN)
  %4 = executor.tan %arg0 : f8E4M3FN
  executor.print "tan_f8 %s = %s"(%arg0, %4 : f8E4M3FN, f8E4M3FN)
  %5 = executor.exp %arg0 : f8E4M3FN
  executor.print "exp_f8 %s = %s"(%arg0, %5 : f8E4M3FN, f8E4M3FN)
  return
}

func.func @test_compare_f8(%arg0: f8E4M3FN, %arg1: f8E4M3FN){
  %0 = executor.fcmp <_false> %arg0, %arg1 : f8E4M3FN
  %1 = executor.fcmp <oeq> %arg0, %arg1 : f8E4M3FN
  %2 = executor.fcmp <ogt> %arg0, %arg1 : f8E4M3FN
  %3 = executor.fcmp <oge> %arg0, %arg1 : f8E4M3FN
  %4 = executor.fcmp <olt> %arg0, %arg1 : f8E4M3FN
  %5 = executor.fcmp <ole> %arg0, %arg1 : f8E4M3FN
  %6 = executor.fcmp <one> %arg0, %arg1 : f8E4M3FN
  %7 = executor.fcmp <ord> %arg0, %arg1 : f8E4M3FN
  %8 = executor.fcmp <ueq> %arg0, %arg1 : f8E4M3FN
  %9 = executor.fcmp <ugt> %arg0, %arg1 : f8E4M3FN
  %10 = executor.fcmp <uge> %arg0, %arg1 : f8E4M3FN
  %11 = executor.fcmp <ult> %arg0, %arg1 : f8E4M3FN
  %12 = executor.fcmp <ule> %arg0, %arg1 : f8E4M3FN
  %13 = executor.fcmp <une> %arg0, %arg1 : f8E4M3FN
  %14 = executor.fcmp <uno> %arg0, %arg1 : f8E4M3FN
  %15 = executor.fcmp <_true> %arg0, %arg1 : f8E4M3FN
  executor.print "comparison results for %s and %s"(%arg0, %arg1: f8E4M3FN, f8E4M3FN)
  executor.print "false=%d, oeq=%d, ogt=%d, oge=%d, olt=%d, ole=%d, one=%d, ord=%d, ueq=%d, ugt=%d, uge=%d, ult=%d, ule=%d, une=%d, uno=%d, true=%d"
    (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)
  return
}


func.func @test_f8e4m3(){
  %f8_const_0 = executor.constant 2.32356 : f8E4M3FN
  %f8_const_1 = executor.constant 0.395264 : f8E4M3FN
  %f8_nan_1 = arith.constant 0x7F: f8E4M3FN
  %i64c_cast = executor.constant 9223372855707 : i64
  %i32c_cast = executor.constant 2143547678 : i32
  %i16c_cast = executor.constant 32667 : i16
  %i8c_cast = executor.constant 28 : i8
  %f32_nan = arith.constant 0x7FC00000 : f32
  %f16_nan = arith.constant 0x7FFF: f16
  func.call @test_base_f8(%f8_const_0, %f8_const_1 ) : (f8E4M3FN, f8E4M3FN)->()
  func.call @test_sitofp_f8(%i64c_cast, %i32c_cast, %i16c_cast, %i8c_cast): (i64, i32, i16, i8)->()
  func.call @test_fptosi_f8(%f8_const_0): (f8E4M3FN)->()
  func.call @test_unary_f8(%f8_const_0): (f8E4M3FN)->()
  func.call @test_compare_f8(%f8_const_0, %f8_const_1): (f8E4M3FN, f8E4M3FN)->()
  func.call @test_compare_f8(%f8_const_0, %f8_const_0): (f8E4M3FN, f8E4M3FN)->()
  func.call @test_compare_f8(%f8_const_0, %f8_nan_1): (f8E4M3FN, f8E4M3FN)->()
  func.call @test_nan_cast_f8(%f8_nan_1, %f16_nan, %f32_nan): (f8E4M3FN, f16, f32)->()
  return
}

// BF16 Tests

func.func @test_base_bf16(%arg0: bf16, %arg1: bf16) {
  %0 = executor.addf %arg0, %arg1 : bf16
  executor.print "%s add_bf16 %s = %s"(%arg0, %arg1, %0 : bf16, bf16, bf16)
  %1 = executor.subf %arg0, %arg1 : bf16
  executor.print "%s sub_bf16 %s = %s"(%arg0, %arg1, %1 : bf16, bf16, bf16)
  %2 = executor.mulf %arg0, %arg1 : bf16
  executor.print "%s mul_bf16 %s = %s"(%arg0, %arg1, %2 : bf16, bf16, bf16)
  %3 = executor.divf %arg0, %arg1 : bf16
  executor.print "%s div_bf16 %s = %s"(%arg0, %arg1, %3 : bf16, bf16, bf16)
  %4 = executor.fmax %arg0, %arg1 : bf16
  executor.print "%s fmax_bf16 %s = %s"(%arg0, %arg1, %4 : bf16, bf16, bf16)
  %5 = executor.fmin %arg0, %arg1 : bf16
  executor.print "%s fmin_bf16 %s = %s"(%arg0, %arg1, %5 : bf16, bf16, bf16)
  return
}

func.func @test_nan_cast_bf16(%arg0: bf16, %arg3: f32){
  %1 = executor.extf %arg0: bf16 to f32
  executor.print "%s extf_bf16_f32 = %s"(%arg0, %1 : bf16, f32)
  %3 = executor.truncf %arg3: f32 to bf16
  executor.print "%s truncf_f32_bf16 = %s"(%arg3, %3 : f32, bf16)
  return
}

func.func @test_sitofp_bf16(%arg0: i64, %arg1: i32, %arg2: i16,
    %arg3: i8) {
  %0 = executor.sitofp %arg0 : i64 to bf16
  executor.print "sitofp i64 to bf16 of %d = %s"(%arg0, %0 : i64, bf16)
  %2 = executor.sitofp %arg1 : i32 to bf16
  executor.print "sitofp i32 to bf16 of %d = %s"(%arg1, %2 : i32, bf16)
  %4 = executor.sitofp %arg2 : i16 to bf16
  executor.print "sitofp i16 to bf16 of %d = %s"(%arg2, %4 : i16, bf16)
  %6 = executor.sitofp %arg3 : i8 to bf16
  executor.print "sitofp i8 to bf16 of %d = %s"(%arg3, %6 : i8, bf16)
  return
}

func.func @test_fptosi_bf16(%arg0: bf16) {
  %0 = executor.fptosi %arg0 : bf16 to i64
  executor.print "fptosi bf16 to i64 of %s = %d"(%arg0, %0 : bf16, i64)
  %1 = executor.fptosi %arg0 : bf16 to i32
  executor.print "fptosi bf16 to i32 of %s = %d"(%arg0, %1 : bf16, i32)
  %2 = executor.fptosi %arg0 : bf16 to i16
  executor.print "fptosi bf16 to i16 of %s = %d"(%arg0, %2 : bf16, i16)
  %3 = executor.fptosi %arg0 : bf16 to i8
  executor.print "fptosi bf16 to i8 of %s = %d"(%arg0, %3 : bf16, i8)
  return
}

func.func @test_unary_bf16(%arg0: bf16){
  %0 = executor.negf %arg0 : bf16
  executor.print "negative_bf16 %s = %s"(%arg0, %0 : bf16, bf16)
  %1 = executor.sqrt %arg0 : bf16
  executor.print "sqrt_bf16 %s = %s"(%arg0, %1 : bf16, bf16)
  %2 = executor.log1p %arg0 : bf16
  executor.print "log1p_bf16 %s = %s"(%arg0, %2 : bf16, bf16)
  %3 = executor.log10 %arg0 : bf16
  executor.print "log10_bf16 %s = %s"(%arg0, %3 : bf16, bf16)
  %4 = executor.tan %arg0 : bf16
  executor.print "tan_bf16 %s = %s"(%arg0, %4 : bf16, bf16)
  %5 = executor.exp %arg0 : bf16
  executor.print "exp_bf16 %s = %s"(%arg0, %5 : bf16, bf16)
  return
}

func.func @test_compare_bf16(%arg0: bf16, %arg1: bf16){
  %0 = executor.fcmp <_false> %arg0, %arg1 : bf16
  %1 = executor.fcmp <oeq> %arg0, %arg1 : bf16
  %2 = executor.fcmp <ogt> %arg0, %arg1 : bf16
  %3 = executor.fcmp <oge> %arg0, %arg1 : bf16
  %4 = executor.fcmp <olt> %arg0, %arg1 : bf16
  %5 = executor.fcmp <ole> %arg0, %arg1 : bf16
  %6 = executor.fcmp <one> %arg0, %arg1 : bf16
  %7 = executor.fcmp <ord> %arg0, %arg1 : bf16
  %8 = executor.fcmp <ueq> %arg0, %arg1 : bf16
  %9 = executor.fcmp <ugt> %arg0, %arg1 : bf16
  %10 = executor.fcmp <uge> %arg0, %arg1 : bf16
  %11 = executor.fcmp <ult> %arg0, %arg1 : bf16
  %12 = executor.fcmp <ule> %arg0, %arg1 : bf16
  %13 = executor.fcmp <une> %arg0, %arg1 : bf16
  %14 = executor.fcmp <uno> %arg0, %arg1 : bf16
  %15 = executor.fcmp <_true> %arg0, %arg1 : bf16
  executor.print "comparison results for %s and %s"(%arg0, %arg1: bf16, bf16)
  executor.print "false=%d, oeq=%d, ogt=%d, oge=%d, olt=%d, ole=%d, one=%d, ord=%d, ueq=%d, ugt=%d, uge=%d, ult=%d, ule=%d, une=%d, uno=%d, true=%d"
    (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)
  return
}

func.func @test_bf16(){
  %bf16_const_0 = executor.constant 149.4356 : bf16
  %bf16_const_1 = executor.constant 93.395264 : bf16
  %bf16_nan_1 = arith.constant 0x7FFF: bf16
  %i64c_cast = executor.constant 9223372855707 : i64
  %i32c_cast = executor.constant 2143547678 : i32
  %i16c_cast = executor.constant 32667 : i16
  %i8c_cast = executor.constant 28 : i8
  %f32_nan = arith.constant 0x7FC00000 : f32
  func.call @test_base_bf16(%bf16_const_0, %bf16_const_1 ) : (bf16, bf16)->()
  func.call @test_sitofp_bf16(%i64c_cast, %i32c_cast, %i16c_cast, %i8c_cast): (i64, i32, i16, i8)->()
  func.call @test_fptosi_bf16(%bf16_const_0): (bf16)->()
  func.call @test_unary_bf16(%bf16_const_0): (bf16)->()
  func.call @test_compare_bf16(%bf16_const_0, %bf16_const_1): (bf16, bf16)->()
  func.call @test_compare_bf16(%bf16_const_0, %bf16_const_0): (bf16, bf16)->()
  func.call @test_compare_bf16(%bf16_const_0, %bf16_nan_1): (bf16, bf16)->()
  func.call @test_compare_bf16(%bf16_nan_1, %bf16_nan_1): (bf16, bf16)->()
  func.call @test_nan_cast_bf16(%bf16_nan_1, %f32_nan): (bf16, f32)->()
  return
}

// Test i4

func.func @test_base_i4(%arg0: i4, %arg1: i4, %arg2: i4){
  %0 = executor.addi %arg0, %arg1 : i4
  executor.print"%s addi %s = %s"(%arg0, %arg1, %0: i4, i4, i4)
  %1 = executor.subi %arg0, %arg1 : i4
  executor.print"%s subi %s = %s"(%arg0, %arg1, %1: i4, i4, i4)
  %2 = executor.muli %arg0, %arg1 : i4
  executor.print"%s muli %s = %s"(%arg0, %arg1, %2: i4, i4, i4)
  %3 = executor.sfloor_divi %arg0, %arg1 : i4
  executor.print"%s sfloor_divi %s = %s"(%arg0, %arg1, %3: i4, i4, i4)
  %4 = executor.sdivi %arg0, %arg1 : i4
  executor.print"%s sdivi %s = %s"(%arg0, %arg1, %4: i4, i4, i4)
  %5 = executor.sremi %arg0, %arg1 : i4
  executor.print"%s sremi %s = %s"(%arg0, %arg1, %5: i4, i4, i4)
  %6 = executor.absi %arg2 : i4
  executor.print"%s absi = %s"(%arg2, %6: i4, i4)
  return
}

func.func @test_compare_i4(%lhs: i4, %rhs: i4){
  %c1 = arith.constant 1 : i1
  %ic0 = arith.constant 0 : i1
  %condEq = executor.icmp <eq> %lhs, %rhs: i4
  %eq = scf.if %condEq -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s eq %s = %q"(%lhs, %rhs, %eq : i4, i4, i1)

  %condNe = executor.icmp <ne> %lhs, %rhs: i4
  %ne = scf.if %condNe -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s ne %s = %q"(%lhs, %rhs, %ne : i4, i4, i1)

  %condSgt = executor.icmp <sgt> %lhs, %rhs: i4
  %sgt = scf.if %condSgt -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s sgt %s = %q"(%lhs, %rhs, %sgt : i4, i4, i1)

  %condSge = executor.icmp <sge> %lhs, %rhs: i4
  %sge = scf.if %condSge -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s sge %s = %q"(%lhs, %rhs, %sge : i4, i4, i1)

  %condUgt1 = executor.icmp <ugt> %lhs, %rhs: i4
  %ugte1 = scf.if %condUgt1 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s ugt %s = %q"(%lhs, %rhs, %ugte1 : i4, i4, i1)

  %condUgt2 = executor.icmp <ugt> %rhs, %lhs: i4
  %ugte2 = scf.if %condUgt2 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s ugt %s = %q"(%rhs, %lhs, %ugte2 : i4, i4, i1)

  %condUlt1 = executor.icmp <ult> %lhs, %rhs: i4
  %ulte1 = scf.if %condUlt1 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s ult %s = %q"(%lhs, %rhs, %ulte1 : i4, i4, i1)

  %condUlt2 = executor.icmp <ult> %rhs, %lhs: i4
  %ulte2 = scf.if %condUlt2 -> i1{
    scf.yield %c1 : i1
  } else {
    scf.yield %ic0 : i1
  }
  executor.print "%s ult %s = %q"(%rhs, %lhs, %ulte2 : i4, i4, i1)
  return
}

func.func @test_bitwise_i4(%arg0: i4, %arg1: i4){
  %0 = executor.bitwise_andi %arg0, %arg1 : i4
  executor.print "%s bitwise_and %s = %s"(%arg0, %arg1, %0 : i4, i4, i4)
  %1 = executor.bitwise_ori %arg0, %arg1 : i4
  executor.print "%s bitwise_or %s = %s"(%arg0, %arg1, %1 : i4, i4, i4)
  %2 = executor.bitwise_xori %arg0, %arg1 : i4
  executor.print "%s bitwise_xor %s = %s"(%arg0, %arg1, %2 : i4, i4, i4)
  return
}

func.func @test_shift_i4(%arg0: i4, %arg1: i4) {
  %0 = executor.shift_lefti %arg0, %arg1 : i4
  executor.print "%s shift_left %s = %s"(%arg0, %arg1, %0 : i4, i4, i4)
  %1 = executor.shift_right_logicali %arg0, %arg1 : i4
  executor.print "%s shift_right_logical %s = %s"(%arg0, %arg1, %1 : i4, i4, i4)
  %2 = executor.shift_right_arithmetici %arg0, %arg1 : i4
  executor.print "%s shift_right_arithmetic %s = %s"(%arg0, %arg1, %2 : i4, i4, i4)
  return
}

func.func @test_sitofp_i4(%arg0: i4) {
  %0 = executor.sitofp %arg0 : i4 to f64
  executor.print "sitofp i4 to f64 of %s = %f"(%arg0, %0 : i4, f64)
  %1 = executor.sitofp %arg0 : i4 to f32
  executor.print "sitofp i4 to f32 of %s = %f"(%arg0, %1 : i4, f32)
  %2 = executor.sitofp %arg0 : i4 to f16
  executor.print "sitofp i4 to f16 of %s = %s"(%arg0, %2 : i4, f16)
  %3 = executor.sitofp %arg0 : i4 to bf16
  executor.print "sitofp i4 to bf16 of %s = %s"(%arg0, %3 : i4, bf16)
  %4 = executor.sitofp %arg0 : i4 to f8E4M3FN
  executor.print "sitofp i4 to fp8 of %s = %s"(%arg0, %4 : i4, f8E4M3FN)
  return
}

func.func @test_fptosi_i4(%c_f8: f8E4M3FN, %c_f16: f16, %c_bf16: bf16, %c_f32: f32, %c_f64: f64) {
  %0 = executor.fptosi %c_f8 : f8E4M3FN to i4
  executor.print "fptosi f8 to i4 of %s = %s"(%c_f8, %0 : f8E4M3FN, i4)
  %1 = executor.fptosi %c_f16 : f16 to i4
  executor.print "fptosi f16 to i4 of %s = %s"(%c_f16, %1 : f16, i4)
  %2 = executor.fptosi %c_bf16 : bf16 to i4
  executor.print "fptosi bf16 to i4 of %s = %s"(%c_bf16, %2 : bf16, i4)
  %3 = executor.fptosi %c_f32 : f32 to i4
  executor.print "fptosi f32 to i4 of %s = %s"(%c_f32, %3 : f32, i4)
  %4 = executor.fptosi %c_f64 : f64 to i4
  executor.print "fptosi f64 to i4 of %s = %s"(%c_f64, %4 : f64, i4)
  return
}

func.func @test_zext_i4(%arg0: i4){
  %0 = arith.extui %arg0 : i4 to i8
  executor.print "zext i4 %s to i8 %s"(%arg0, %0 : i4, i8)
  %1 = arith.extui %arg0 : i4 to i16
  executor.print "zext i4 %s to i16 %s"(%arg0, %1 : i4, i16)
  %2 = arith.extui %arg0 : i4 to i32
  executor.print "zext i4 %s to i32 %s"(%arg0, %2 : i4, i32)
  return
}

func.func @test_sext_i4(%arg0: i4){
  %0 = arith.extsi %arg0 : i4 to i8
  executor.print "sext i4 %s to i8 %s"(%arg0, %0 : i4, i8)
  %1 = arith.extsi %arg0 : i4 to i16
  executor.print "sext i4 %s to i16 %s"(%arg0, %1 : i4, i16)
  %2 = arith.extsi %arg0 : i4 to i32
  executor.print "sext i4 %s to i32 %s"(%arg0, %2 : i4, i32)
  return
}

func.func @test_ext_i4(%pos_i4: i4, %neg_i4 : i4){
  func.call @test_zext_i4(%pos_i4):(i4)->()
  func.call @test_zext_i4(%neg_i4):(i4)->()
  func.call @test_sext_i4(%pos_i4):(i4)->()
  func.call @test_sext_i4(%neg_i4):(i4)->()
  return
}

func.func @test_i4(){
  %c4 = executor.constant 4 : i4
  %c3 = executor.constant 3 : i4
  %cn5 = executor.constant -5 : i4
  func.call @test_base_i4(%c4, %c3, %cn5) : (i4, i4, i4)->()

  %cn4 = executor.constant -4: i4
  func.call @test_compare_i4(%c3, %cn4) : (i4, i4) -> ()

  %c7 = executor.constant 7: i4
  %c2 = executor.constant 2 : i4
  func.call @test_bitwise_i4(%c7, %c2) : (i4, i4) -> ()

  %cn3 = executor.constant -3 : i4
  func.call @test_shift_i4(%cn3, %c2) : (i4, i4) -> ()

  func.call @test_sitofp_i4(%cn4) : (i4) -> ()

  %c_f8 = executor.constant 4.35 : f8E4M3FN
  %c_f16 = executor.constant 4.456 : f16
  %c_bf16 = executor.constant 4.34567 : bf16
  %c_f32 = executor.constant 5.897 : f32
  %c_f64 = executor.constant 1.34567 : f64
  func.call @test_fptosi_i4(%c_f8, %c_f16, %c_bf16, %c_f32, %c_f64) : (f8E4M3FN, f16, bf16, f32, f64) -> ()

  func.call @test_ext_i4(%c3, %cn3) : (i4, i4) -> ()
  return
}

func.func @main() -> i64 {
  %ic0 = executor.constant 0 : i64
  %ic4 = executor.constant 4 : i64
  %icm4 = executor.constant -4 : i64
  %ic2 = executor.constant 2 : i64
  %icm2 = executor.constant -2 : i64
  %ic3 = executor.constant 3 : i64
  %icm3 = executor.constant -3 : i64
  %icm8 = executor.constant -8 : i64

  %fc0 = executor.constant 0.0 : f64
  %fc4 = executor.constant 4.0 : f64
  %fcm4 = executor.constant -4.0 : f64
  %fc2 = executor.constant 2.0 : f64
  %fcm2 = executor.constant -2.0 : f64
  %fc3 = executor.constant 3.0 : f64
  %fcm3 = executor.constant -3.0 : f64
  %fcm8 = executor.constant -8.0 : f64

  %f32c = executor.constant 29.99 : f32
  %i32c = executor.constant 396 : i32

  %c4_f16 = executor.constant 4.0 : f16
  %cm4_f16 = executor.constant -4.0 : f16

  %f32c_cast = executor.constant 16777216.0 : f32
  %f64c_cast = executor.constant 2147483647.0 : f64
  %i64c_cast = executor.constant 9223372855707 : i64
  %i32c_cast = executor.constant 2143547678 : i32
  %i16c_cast = executor.constant 32667 : i16
  %i8c_cast = executor.constant 28 : i8

  %ptrue = executor.constant 1 : i1
  %pfalse = executor.constant 0 : i1

  func.call @test_srem(%ic4, %ic2) : (i64, i64)->()
  func.call @test_srem(%ic4, %ic3) : (i64, i64)->()
  func.call @test_srem(%icm4, %ic2) : (i64, i64)->()
  func.call @test_srem(%icm4, %ic3) : (i64, i64)->()
  func.call @test_srem(%ic4, %icm2) : (i64, i64)->()
  func.call @test_srem(%ic4, %icm3) : (i64, i64)->()
  func.call @test_srem(%icm4, %icm2) : (i64, i64)->()
  func.call @test_srem(%icm4, %icm3) : (i64, i64)->()

  func.call @test_sdivi(%ic4, %ic2) : (i64, i64)->()
  func.call @test_sdivi(%ic4, %ic3) : (i64, i64)->()
  func.call @test_sdivi(%icm4, %icm2) : (i64, i64)->()
  func.call @test_sdivi(%icm4, %icm3) : (i64, i64)->()
  func.call @test_sdivi(%icm4, %icm8) : (i64, i64)->()
  func.call @test_sdivi(%icm4, %ic2) : (i64, i64)->()
  func.call @test_sdivi(%icm4, %ic3) : (i64, i64)->()
  func.call @test_sdivi(%ic4, %icm2) : (i64, i64)->()
  func.call @test_sdivi(%ic4, %icm3) : (i64, i64)->()

  func.call @test_divf(%fc4, %fc2) : (f64, f64)->()
  func.call @test_divf(%fc4, %fc3) : (f64, f64)->()
  func.call @test_divf(%fcm4, %fcm2) : (f64, f64)->()
  func.call @test_divf(%fcm4, %fcm3) : (f64, f64)->()
  func.call @test_divf(%fcm4, %fcm8) : (f64, f64)->()
  func.call @test_divf(%fcm4, %fc2) : (f64, f64)->()
  func.call @test_divf(%fcm4, %fc3) : (f64, f64)->()
  func.call @test_divf(%fc4, %fcm2) : (f64, f64)->()
  func.call @test_divf(%fc4, %fcm3) : (f64, f64)->()

  func.call @test_sfloor_divi(%ic4, %ic2) : (i64, i64)->()
  func.call @test_sfloor_divi(%ic4, %ic3) : (i64, i64)->()
  func.call @test_sfloor_divi(%icm4, %icm2) : (i64, i64)->()
  func.call @test_sfloor_divi(%icm4, %icm3) : (i64, i64)->()
  func.call @test_sfloor_divi(%icm4, %icm8) : (i64, i64)->()
  func.call @test_sfloor_divi(%icm4, %ic2) : (i64, i64)->()
  func.call @test_sfloor_divi(%icm4, %ic3) : (i64, i64)->()
  func.call @test_sfloor_divi(%ic4, %icm2) : (i64, i64)->()
  func.call @test_sfloor_divi(%ic4, %icm3) : (i64, i64)->()

  func.call @test_addi(%ic4, %ic3) : (i64, i64)->()
  func.call @test_addf(%fc4, %fcm4) : (f64, f64)->()
  func.call @test_subi(%ic4, %ic3) : (i64, i64)->()
  func.call @test_subf(%fc4, %fcm4) : (f64, f64)->()
  func.call @test_muli(%ic4, %ic3) : (i64, i64)->()
  func.call @test_mulf(%fc4, %fcm4) : (f64, f64)->()

  func.call @test_mulf16(%c4_f16, %cm4_f16) : (f16, f16)->()
  func.call @test_addf16(%c4_f16, %cm4_f16) : (f16, f16)->()
  func.call @test_subf16(%c4_f16, %cm4_f16) : (f16, f16)->()
  func.call @test_divf16(%c4_f16, %cm4_f16) : (f16, f16)->()

  func.call @test_smax(%ic4, %ic2) : (i64, i64)->()
  func.call @test_smax(%icm2, %ic3) : (i64, i64)->()
  func.call @test_smax(%icm3, %icm8) : (i64, i64)->()

  func.call @test_fmax(%fc4, %fc2) : (f64, f64)->()
  func.call @test_fmax(%fcm2, %fc3) : (f64, f64)->()
  func.call @test_fmax(%fcm3, %fcm8) : (f64, f64)->()

  func.call @test_shift_lefti(%ic4, %ic2) : (i64, i64)->()
  func.call @test_shift_right_logicali(%ic4, %ic2) : (i64, i64)->()
  func.call @test_shift_right_logicali(%icm4, %ic2) : (i64, i64)->()
  func.call @test_shift_right_arithmetici(%ic4, %ic2) : (i64, i64)->()
  func.call @test_shift_right_arithmetici(%icm4, %ic2) : (i64, i64)->()

  func.call @test_bitwise_and(%ic3, %ic2) : (i64, i64)->()
  func.call @test_bitwise_and(%icm4, %ic2) : (i64, i64)->()
  func.call @test_bitwise_and_i1(%ptrue, %pfalse) : (i1, i1)->()
  func.call @test_bitwise_or(%ic3, %ic2) : (i64, i64)->()
  func.call @test_bitwise_or(%icm4, %ic2) : (i64, i64)->()
  func.call @test_bitwise_or_i1(%ptrue, %pfalse) : (i1, i1)->()
  func.call @test_bitwise_xor(%ic3, %ic2) : (i64, i64)->()
  func.call @test_bitwise_xor(%icm4, %ic2) : (i64, i64)->()
  func.call @test_bitwise_xor_i1(%ptrue, %pfalse) : (i1, i1)->()

  func.call @test_negf(%fc4) : (f64)->()
  func.call @test_negf(%fcm4) : (f64)->()
  func.call @test_absi(%ic4) : (i64)->()
  func.call @test_absi(%icm4) : (i64)->()
  func.call @test_absf(%fc4) : (f64)->()
  func.call @test_absf(%fcm4) : (f64)->()

  func.call @test_sqrt(%fc4) : (f64)->()
  func.call @test_log1p(%fc4) : (f64)->()

  func.call @test_select(%ptrue, %ic4, %icm4) : (i1, i64, i64) -> ()
  func.call @test_select(%pfalse, %ic4, %icm4) : (i1, i64, i64) -> ()
  func.call @test_selectf(%ptrue, %fc3, %fcm8) : (i1, f64, f64) -> ()
  func.call @test_selectf(%pfalse, %fc3, %fcm8) : (i1, f64, f64) -> ()

  func.call @test_bitcast_f32_i32(%f32c) : (f32) -> ()
  func.call @test_bitcast_i32_f32(%i32c) : (i32) -> ()
  func.call @test_bitcast_f64_i64(%fc4) : (f64) -> ()
  func.call @test_bitcast_i64_f64(%ic4) : (i64) -> ()

  func.call @test_sitofp(%i64c_cast, %i32c_cast, %i16c_cast, %i8c_cast):
    (i64, i32, i16, i8)->()
  func.call @test_fptosi(%f64c_cast, %f32c_cast): (f64, f32)->()

  func.call @test_icmp_if(%ic4, %ic4) : (i64, i64)->()
  func.call @test_icmp_if(%ic4, %icm4) : (i64, i64)->()

  %cn1_i32 = executor.constant -1 : i32
  %c1_i32 = executor.constant 4 : i32
  func.call @test_icmp_ugt_i32(%cn1_i32, %c1_i32) : (i32, i32)->()
  func.call @test_icmp_ult_i32(%cn1_i32, %c1_i32) : (i32, i32)->()

  // cmpf tests

  %c42 = arith.constant 42. : f32
  %cpinf = arith.constant 0x7F800000 : f32
  func.call @print_test_cmpf(%c42, %cpinf) : (f32, f32) -> ()

  %cminf = arith.constant 0xFF800000 : f32
  func.call @print_test_cmpf(%c42, %cminf) : (f32, f32) -> ()

  %nan = arith.constant 0x7fc00000 : f32
  func.call @print_test_cmpf(%c42, %nan) : (f32, f32) -> ()

  func.call @print_test_cmpf(%c42, %c42) : (f32, f32) -> ()



  // zext  tests
  %c1b = arith.constant 1 : i1
  %c0b = arith.constant 0 : i1
  %c1_i8 = arith.constant 1 : i8
  %c-1_i8 = arith.constant -1 : i8
  %c15_i8 = arith.constant 15 : i8
  %c-15_i8 = arith.constant -15 : i8
  func.call @print_test_zext_i1(%c1b) : (i1)->()
  func.call @print_test_zext_i1(%c0b) : (i1)->()
  func.call @print_test_zext_i8(%c1_i8) : (i8)->()
  func.call @print_test_zext_i8(%c-1_i8) : (i8)->()
  func.call @print_test_zext_i8(%c15_i8) : (i8)->()
  func.call @print_test_zext_i8(%c-15_i8) : (i8)->()

  // siext  tests

  func.call @print_test_siext_i1(%c1b) : (i1)->()
  func.call @print_test_siext_i1(%c0b) : (i1)->()
  func.call @print_test_siext_i8(%c15_i8) : (i8)->()
  func.call @print_test_siext_i8(%c-15_i8) : (i8)->()

  // Comparison integer i1 tests
  func.call @test_cmpi1() : () -> ()

  // FP8 tests
  func.call @test_f8e4m3() : () -> ()

  // BF16 tests
  func.call @test_bf16() : () -> ()

  // INT4 tests
  func.call @test_i4():()->()

  // integer trunc tests
  executor.print "TEST: trunc i64 -> i32"()
  %c64_big = executor.constant 4294967295 : i64
  %c64_big1 = executor.constant 4294967296 : i64
  %c64_big2 = executor.constant -1 : i64
  func.call @test_trunci_i64_to_i32(%c64_big) : (i64) -> ()
  func.call @test_trunci_i64_to_i32(%c64_big1) : (i64) -> ()
  func.call @test_trunci_i64_to_i32(%c64_big2) : (i64) -> ()

  executor.print "TEST: trunc i64 -> i1"()
  %c64_3 = executor.constant 3 : i64
  %c64_2 = executor.constant 2 : i64
  %c64_1 = executor.constant 1 : i64
  %c64_0 = executor.constant 0 : i64
  func.call @test_trunci_i64_to_i1(%c64_3) : (i64) -> ()
  func.call @test_trunci_i64_to_i1(%c64_2) : (i64) -> ()
  func.call @test_trunci_i64_to_i1(%c64_1) : (i64) -> ()
  func.call @test_trunci_i64_to_i1(%c64_0) : (i64) -> ()

  executor.print "TEST: trunc i32 -> i1"()
  %c32_3 = executor.constant 3 : i32
  %c32_2 = executor.constant 2 : i32
  %c32_1 = executor.constant 1 : i32
  %c32_0 = executor.constant 0 : i32
  func.call @test_trunci_i32_to_i1(%c32_3) : (i32) -> ()
  func.call @test_trunci_i32_to_i1(%c32_2) : (i32) -> ()
  func.call @test_trunci_i32_to_i1(%c32_1) : (i32) -> ()
  func.call @test_trunci_i32_to_i1(%c32_0) : (i32) -> ()

  executor.print "TEST: trunc i8 -> i1"()
  %cn1_i8 = executor.constant -1 : i8
  func.call @test_trunci_i8_to_i1(%cn1_i8) : (i8) -> ()
  func.call @test_trunci_i8_to_i1(%c1_i8) : (i8) -> ()




  return %ic0 : i64
}

//      CHECK:  4 srem 2 = 0
// CHECK-NEXT:  4 srem 3 = 1
// CHECK-NEXT:  -4 srem 2 = 0
// CHECK-NEXT:  -4 srem 3 = 2
// CHECK-NEXT:  4 srem -2 = 0
// CHECK-NEXT:  4 srem -3 = -2
// CHECK-NEXT:  -4 srem -2 = 0
// CHECK-NEXT:  -4 srem -3 = -1
// CHECK-NEXT:  4 sdivi 2 = 2
// CHECK-NEXT:  4 sdivi 3 = 1
// CHECK-NEXT:  -4 sdivi -2 = 2
// CHECK-NEXT:  -4 sdivi -3 = 1
// CHECK-NEXT:  -4 sdivi -8 = 0
// CHECK-NEXT:  -4 sdivi 2 = -2
// CHECK-NEXT:  -4 sdivi 3 = -1
// CHECK-NEXT:  4 sdivi -2 = -2
// CHECK-NEXT:  4 sdivi -3 = -1
// CHECK-NEXT:  4.000000 divf 2.000000 = 2.000000
// CHECK-NEXT:  4.000000 divf 3.000000 = 1.333333
// CHECK-NEXT:  -4.000000 divf -2.000000 = 2.000000
// CHECK-NEXT:  -4.000000 divf -3.000000 = 1.333333
// CHECK-NEXT:  -4.000000 divf -8.000000 = 0.500000
// CHECK-NEXT:  -4.000000 divf 2.000000 = -2.000000
// CHECK-NEXT:  -4.000000 divf 3.000000 = -1.333333
// CHECK-NEXT:  4.000000 divf -2.000000 = -2.000000
// CHECK-NEXT:  4.000000 divf -3.000000 = -1.333333
// CHECK-NEXT:  4 sfloor_divi 2 = 2
// CHECK-NEXT:  4 sfloor_divi 3 = 1
// CHECK-NEXT:  -4 sfloor_divi -2 = 2
// CHECK-NEXT:  -4 sfloor_divi -3 = 1
// CHECK-NEXT:  -4 sfloor_divi -8 = 0
// CHECK-NEXT:  -4 sfloor_divi 2 = -2
// CHECK-NEXT:  -4 sfloor_divi 3 = -2
// CHECK-NEXT:  4 sfloor_divi -2 = -2
// CHECK-NEXT:  4 sfloor_divi -3 = -2
// CHECK-NEXT:  4 addi 3 = 7
// CHECK-NEXT:  4.000000 addf -4.000000 = 0.000000
// CHECK-NEXT:  4 subi 3 = 1
// CHECK-NEXT:  4.000000 subf -4.000000 = 8.000000
// CHECK-NEXT:  4 muli 3 = 12
// CHECK-NEXT:  4.000000 mulf -4.000000 = -16.000000
// CHECK-NEXT:  4 mulf(f16) -4 = -16
// CHECK-NEXT:  4 addf(f16) -4 = 0
// CHECK-NEXT:  4 subf(f16) -4 = 8
// CHECK-NEXT:  4 divf(f16) -4 = -1
// CHECK-NEXT:  4 smax 2 = 4.000000
// CHECK-NEXT:  -2 smax 3 = 3.000000
// CHECK-NEXT:  -3 smax -8 = -3.000000
// CHECK-NEXT:  4.000000 fmax 2.000000 = 4.000000
// CHECK-NEXT:  -2.000000 fmax 3.000000 = 3.000000
// CHECK-NEXT:  -3.000000 fmax -8.000000 = -3.000000
// CHECK-NEXT:  4 shift_left 2 = 16
// CHECK-NEXT:  4 shift_right_logical 2 = 1
// CHECK-NEXT:  -4 shift_right_logical 2 = 4611686018427387903
// CHECK-NEXT:  4 shift_right_arithmetic 2 = 1
// CHECK-NEXT:  -4 shift_right_arithmetic 2 = -1
// CHECK-NEXT:  3 bitwise_and 2 = 2
// CHECK-NEXT:  -4 bitwise_and 2 = 0
// CHECK-NEXT:  1 bitwise_and_i1 0 = 0
// CHECK-NEXT:  3 bitwise_or 2 = 3
// CHECK-NEXT:  -4 bitwise_or 2 = -2
// CHECK-NEXT:  1 bitwise_or_i1 0 = 1
// CHECK-NEXT:  3 bitwise_xor 2 = 1
// CHECK-NEXT:  -4 bitwise_xor 2 = -2
// CHECK-NEXT:  1 bitwise_xor_i1 0 = 1
// CHECK-NEXT:  negative 4.000000 = -4.000000
// CHECK-NEXT:  negative -4.000000 = 4.000000
// CHECK-NEXT:  abs 4 = 4
// CHECK-NEXT:  abs -4 = 4
// CHECK-NEXT:  abs 4.000000 = 4.000000
// CHECK-NEXT:  abs -4.000000 = 4.000000
// CHECK-NEXT:  sqrt 4 = 2
// CHECK-NEXT:  log1p 4.000000 = 1.609438
// CHECK-NEXT:  select 1 4 -4 = 4
// CHECK-NEXT:  select 0 4 -4 = -4
// CHECK-NEXT:  select 1 3 -8 = 3
// CHECK-NEXT:  select 0 3 -8 = -8
// CHECK-NEXT:  bitcast f32 to i32 of 29.990000 = 1106242437
// CHECK-NEXT:  bitcast i32 to f32 of 396 = 0.000000
// CHECK-NEXT:  bitcast f64 to i64 of 4.000000 = 4616189618054758400
// CHECK-NEXT:  bitcast i64 to f64 of 4 = 0.000000
// CHECK-NEXT:  sitofp i64 to f64 of 9223372855707 = 9223372855707.000000
// CHECK-NEXT:  sitofp i64 to f32 of 9223372855707 = 9223373062144.000000
// CHECK-NEXT:  sitofp i32 to f64 of 2143547678 = 2143547678.000000
// CHECK-NEXT:  sitofp i32 to f32 of 2143547678 = 2143547648.000000
// CHECK-NEXT:  sitofp i16 to f64 of 32667 = 32667.000000
// CHECK-NEXT:  sitofp i16 to f32 of 32667 = 32667.000000
// CHECK-NEXT:  sitofp i8 to f64 of 28 = 28.000000
// CHECK-NEXT:  sitofp i8 to f32 of 28 = 28.000000
// CHECK-NEXT:  fptosi f64 to i64 of 2147483647.{{0+}} = 2147483647
// CHECK-NEXT:  fptosi f64 to i32 of 2147483647.{{0+}} = 2147483647

// fptosi is undefined behavior in LLVM and C if the value falls outside
// the floating point min/max range, so don't match anything in particular
// in these cases.

// CHECK-NEXT:  fptosi f64 to i16 of 2147483647.{{0+}} = {{.*}}
// CHECK-NEXT:  fptosi f64 to i8 of 2147483647.{{0+}} = {{.*}}
// CHECK-NEXT:  fptosi f32 to i64 of 16777216.{{0+}} = 16777216
// CHECK-NEXT:  fptosi f32 to i32 of 16777216.{{0+}} = 16777216
// CHECK-NEXT:  fptosi f32 to i16 of 16777216.{{0+}} = {{.*}}
// CHECK-NEXT:  fptosi f64 to i8 of 16777216.{{0+}} = {{.*}}

// CHECK-NEXT:  4 eq 4 = 1
// CHECK-NEXT:  4 ne 4 = 0
// CHECK-NEXT:  4 sgt 4 = 0
// CHECK-NEXT:  4 sge 4 = 1
// CHECK-NEXT:  4 eq -4 = 0
// CHECK-NEXT:  4 ne -4 = 1
// CHECK-NEXT:  4 sgt -4 = 1
// CHECK-NEXT:  4 sge -4 = 1
// CHECK-NEXT:  -1 ugt 4 = 1
// CHECK-NEXT:  4 ugt -1 = 0
// CHECK-NEXT:  -1 ult 4 = 0
// CHECK-NEXT:  4 ult -1 = 1

// fcmp tests

// CHECK-LABEL: comparing 42.000000 and inf
//       CHECK: false=0, oeq=0, ogt=0, oge=0, olt=1, ole=1, one=1, ord=1, ueq=0, ugt=0, uge=0, ult=1, ule=1, une=1, uno=0, true=1
// CHECK-LABEL: comparing 42.000000 and -inf
//       CHECK: false=0, oeq=0, ogt=1, oge=1, olt=0, ole=0, one=1, ord=1, ueq=0, ugt=1, uge=1, ult=0, ule=0, une=1, uno=0, true=1
// CHECK-LABEL: comparing 42.000000 and nan
//       CHECK: false=0, oeq=0, ogt=0, oge=0, olt=0, ole=0, one=0, ord=0, ueq=1, ugt=1, uge=1, ult=1, ule=1, une=1, uno=1, true=1
// CHECK-LABEL: comparing 42.000000 and 42.000000
//       CHECK: false=0, oeq=1, ogt=0, oge=1, olt=0, ole=1, one=0, ord=1, ueq=1, ugt=0, uge=1, ult=0, ule=1, une=0, uno=0, true=1

//      CHECK: zext i1 1 to i32 1
// CHECK-NEXT: zext i1 0 to i32 0
// CHECK-NEXT: zext i8 1 to i32 1
// CHECK-NEXT: zext i8 -1 to i32 255
// CHECK-NEXT: zext i8 15 to i32 15
// CHECK-NEXT: zext i8 -15 to i32 241

//      CHECK: siext i1 1 to i32 1
// CHECK-NEXT: siext i1 0 to i32 0
// CHECK-NEXT: siext i8 15 to i32 15
// CHECK-NEXT: siext i8 -15 to i32 -15

// CHECK-LABEL: 0 != 1 = 1
//  CHECK-NEXT: 0 == 1 = 0

// FP8 Tests

//  CHECK-NEXT: 2.25 add_f8 0.40625 = 2.75
//  CHECK-NEXT: 2.25 sub_f8 0.40625 = 1.875
//  CHECK-NEXT: 2.25 mul_f8 0.40625 = 0.9375
//  CHECK-NEXT: 2.25 div_f8 0.40625 = 5.5
//  CHECK-NEXT: 2.25 fmax_f8 0.40625 = 2.25
//  CHECK-NEXT: 2.25 fmin_f8 0.40625 = 0.40625
//  CHECK-NEXT: sitofp i64 to f8E4M3FN of 9223372855707 = 448
//  CHECK-NEXT: sitofp i32 to f8E4M3FN of 2143547678 = 448
//  CHECK-NEXT: sitofp i16 to f8E4M3FN of 32667 = 448
//  CHECK-NEXT: sitofp i8 to f8E4M3FN of 28 = 28
//  CHECK-NEXT: fptosi f8E4M3FN to i64 of 2.25 = 2
//  CHECK-NEXT: fptosi f8E4M3FN to i32 of 2.25 = 2
//  CHECK-NEXT: fptosi f8E4M3FN to i16 of 2.25 = 2
//  CHECK-NEXT: fptosi f8E4M3FN to i8 of 2.25 = 2
//  CHECK-NEXT: negative_f8 2.25 = -2.25
//  CHECK-NEXT: sqrt_f8 2.25 = 1.5
//  CHECK-NEXT: log1p_f8 2.25 = 1.125
//  CHECK-NEXT: log10_f8 2.25 = 0.34375
//  CHECK-NEXT: tan_f8 2.25 = -1.25
//  CHECK-NEXT: exp_f8 2.25 = 9
// CHECK-LABEL: comparison results for 2.25 and 0.40625
//  CHECK-NEXT: false=0, oeq=0, ogt=1, oge=1, olt=0, ole=0, one=1, ord=1, ueq=0, ugt=1, uge=1, ult=0, ule=0, une=1, uno=0, true=1
// CHECK-LABEL: comparison results for 2.25 and 2.25
//  CHECK-NEXT: false=0, oeq=1, ogt=0, oge=1, olt=0, ole=1, one=0, ord=1, ueq=1, ugt=0, uge=1, ult=0, ule=1, une=0, uno=0, true=1
// CHECK-LABEL: comparison results for 2.25 and nan
//  CHECK-NEXT: false=0, oeq=0, ogt=0, oge=0, olt=0, ole=0, one=0, ord=0, ueq=1, ugt=1, uge=1, ult=1, ule=1, une=1, uno=1, true=1
// CHECK-NEXT: nan extf_f8E4M3FN_f16 = nan
// CHECK-NEXT: nan extf_f8E4M3FN_f32 = nan
// CHECK-NEXT: nan truncf_f16_f8E4M3FN = nan
// CHECK-NEXT: nan truncf_f32_f8E4M3FN = nan

// BF16 tests

//  CHECK-NEXT: 149 add_bf16 93.5 = 242
//  CHECK-NEXT: 149 sub_bf16 93.5 = 55.5
//  CHECK-NEXT: 149 mul_bf16 93.5 = 13952
//  CHECK-NEXT: 149 div_bf16 93.5 = 1.59375
//  CHECK-NEXT: 149 fmax_bf16 93.5 = 149
//  CHECK-NEXT: 149 fmin_bf16 93.5 = 93.5
//  CHECK-NEXT: sitofp i64 to bf16 of 9223372855707 = 9.20841e+12
//  CHECK-NEXT: sitofp i32 to bf16 of 2143547678 = 2.14748e+09
//  CHECK-NEXT: sitofp i16 to bf16 of 32667 = 32640
//  CHECK-NEXT: sitofp i8 to bf16 of 28 = 28
//  CHECK-NEXT: fptosi bf16 to i64 of 149 = 149
//  CHECK-NEXT: fptosi bf16 to i32 of 149 = 149
//  CHECK-NEXT: fptosi bf16 to i16 of 149 = 149
//  CHECK-NEXT: fptosi bf16 to i8 of 149 = 127
//  CHECK-NEXT: negative_bf16 149 = -149
//  CHECK-NEXT: sqrt_bf16 149 = 12.1875
//  CHECK-NEXT: log1p_bf16 149 = 5
//  CHECK-NEXT: log10_bf16 149 = 2.17188
//  CHECK-NEXT: tan_bf16 149 = 4.34375
//  CHECK-NEXT: exp_bf16 149 = inf
// CHECK-LABEL: comparison results for 149 and 93.5
//  CHECK-NEXT: false=0, oeq=0, ogt=1, oge=1, olt=0, ole=0, one=1, ord=1, ueq=0, ugt=1, uge=1, ult=0, ule=0, une=1, uno=0, true=1
// CHECK-LABEL: comparison results for 149 and 149
//  CHECK-NEXT: false=0, oeq=1, ogt=0, oge=1, olt=0, ole=1, one=0, ord=1, ueq=1, ugt=0, uge=1, ult=0, ule=1, une=0, uno=0, true=1
// CHECK-LABEL: comparison results for 149 and nan
//  CHECK-NEXT: false=0, oeq=0, ogt=0, oge=0, olt=0, ole=0, one=0, ord=0, ueq=1, ugt=1, uge=1, ult=1, ule=1, une=1, uno=1, true=1
// CHECK-LABEL: comparison results for nan and nan
//  CHECK-NEXT: false=0, oeq=0, ogt=0, oge=0, olt=0, ole=0, one=0, ord=0, ueq=1, ugt=1, uge=1, ult=1, ule=1, une=1, uno=1, true=1
//  CHECK-NEXT: nan extf_bf16_f32 = nan
//  CHECK-NEXT: nan truncf_f32_bf16 = nan

// INT4 Tests

//  CHECK-NEXT: 4 addi 3 = 7
//  CHECK-NEXT: 4 subi 3 = 1
//  CHECK-NEXT: 4 muli 3 = -4
//  CHECK-NEXT: 4 sfloor_divi 3 = 1
//  CHECK-NEXT: 4 sdivi 3 = 1
//  CHECK-NEXT: 4 sremi 3 = 1
//  CHECK-NEXT: -5 absi = 5
//  CHECK-NEXT: 3 eq -4 = 0
//  CHECK-NEXT: 3 ne -4 = 1
//  CHECK-NEXT: 3 sgt -4 = 1
//  CHECK-NEXT: 3 sge -4 = 1
//  CHECK-NEXT: 3 ugt -4 = 0
//  CHECK-NEXT: -4 ugt 3 = 1
//  CHECK-NEXT: 3 ult -4 = 1
//  CHECK-NEXT: -4 ult 3 = 0
//  CHECK-NEXT: 7 bitwise_and 2 = 2
//  CHECK-NEXT: 7 bitwise_or 2 = 7
//  CHECK-NEXT: 7 bitwise_xor 2 = 5
//  CHECK-NEXT: -3 shift_left 2 = 4
//  CHECK-NEXT: -3 shift_right_logical 2 = 3
//  CHECK-NEXT: -3 shift_right_arithmetic 2 = -1
//  CHECK-NEXT: sitofp i4 to f64 of -4 = -4.000000
//  CHECK-NEXT: sitofp i4 to f32 of -4 = -4.000000
//  CHECK-NEXT: sitofp i4 to f16 of -4 = -4
//  CHECK-NEXT: sitofp i4 to bf16 of -4 = -4
//  CHECK-NEXT: sitofp i4 to fp8 of -4 = -4
//  CHECK-NEXT: fptosi f8 to i4 of 4.5 = 4
//  CHECK-NEXT: fptosi f16 to i4 of 4.45703 = 4
//  CHECK-NEXT: fptosi bf16 to i4 of 4.34375 = 4
//  CHECK-NEXT: fptosi f32 to i4 of 5.89699984 = 5
//  CHECK-NEXT: fptosi f64 to i4 of 1.34567 = 1
//  CHECK-NEXT: zext i4 3 to i8 3
//  CHECK-NEXT: zext i4 3 to i16 3
//  CHECK-NEXT: zext i4 3 to i32 3
//  CHECK-NEXT: zext i4 -3 to i8 13
//  CHECK-NEXT: zext i4 -3 to i16 13
//  CHECK-NEXT: zext i4 -3 to i32 13
//  CHECK-NEXT: sext i4 3 to i8 3
//  CHECK-NEXT: sext i4 3 to i16 3
//  CHECK-NEXT: sext i4 3 to i32 3
//  CHECK-NEXT: sext i4 -3 to i8 -3
//  CHECK-NEXT: sext i4 -3 to i16 -3
//  CHECK-NEXT: sext i4 -3 to i32 -3

// CHECK-LABEL: TEST: trunc i64 -> i32
//  CHECK-NEXT: trunci i64 0xffffffff to i32 = -1
//  CHECK-NEXT: trunci i64 0x100000000 to i32 = 0
//  CHECK-NEXT: trunci i64 0xffffffffffffffff to i32 = -1

// CHECK-LABEL: TEST: trunc i64 -> i1
//  CHECK-NEXT: trunci i64 0x3 to i1 = 1
//  CHECK-NEXT: trunci i64 0x2 to i1 = 0
//  CHECK-NEXT: trunci i64 0x1 to i1 = 1
//  CHECK-NEXT: trunci i64 0x0 to i1 = 0

// CHECK-LABEL: TEST: trunc i32 -> i1
//  CHECK-NEXT: trunci i32 0x3 to i1 = 1
//  CHECK-NEXT: trunci i32 0x2 to i1 = 0
//  CHECK-NEXT: trunci i32 0x1 to i1 = 1
//  CHECK-NEXT: trunci i32 0x0 to i1 = 0

// CHECK-LABEL: TEST: trunc i8 -> i1
//  CHECK-NEXT: trunci i8 0xffffffffffffffff to i1 = 0x1
//  CHECK-NEXT: trunci i8 0x1 to i1 = 0x1
