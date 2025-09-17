// RUN: rm %t.mlir || true
// RUN: executor-opt %s -executor-lowering-pipeline -o %t.mlir

// RUN: executor-translate -mlir-to-lua  %t.mlir \
// RUN:   | executor-runner -input-type=lua -features=core | FileCheck %s

// RUN: executor-translate -mlir-to-runtime-executable %t.mlir \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

func.func @test_f4_representation(){
  executor.print "TEST F4E2M1FN REPRESENTATION"()
  %0 = executor.constant 0.0 : f16
  %f4_0 = executor.truncf %0 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%0, %f4_0 : f16, f4E2M1FN)
  %1 = executor.constant 0.5 : f16
  %f4_1 = executor.truncf %1 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%1, %f4_1 : f16, f4E2M1FN)
  %2 = executor.constant 1.0 : f16
  %f4_2 = executor.truncf %2 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%2, %f4_2 : f16, f4E2M1FN)
  %3 = executor.constant 1.5 : f16
  %f4_3 = executor.truncf %3 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%3, %f4_3 : f16, f4E2M1FN)
  %4 = executor.constant 2.0 : f16
  %f4_4 = executor.truncf %4 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%4, %f4_4 : f16, f4E2M1FN)
  %5 = executor.constant 3.0 : f16
  %f4_5 = executor.truncf %5 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%5, %f4_5 : f16, f4E2M1FN)
  %6 = executor.constant 4.0 : f16
  %f4_6 = executor.truncf %6 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%6, %f4_6 : f16, f4E2M1FN)
  %7 = executor.constant 6.0 : f16
  %f4_7 = executor.truncf %7 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%7, %f4_7 : f16, f4E2M1FN)
  %8 = executor.constant -0.5 : f16
  %f4_8 = executor.truncf %8 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%8, %f4_8 : f16, f4E2M1FN)
  %9 = executor.constant -1.0 : f16
  %f4_9 = executor.truncf %9 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%9, %f4_9 : f16, f4E2M1FN)
  %10 = executor.constant -1.5 : f16
  %f4_10 = executor.truncf %10 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%10, %f4_10 : f16, f4E2M1FN)
  %11 = executor.constant -2.0 : f16
  %f4_11 = executor.truncf %11 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%11, %f4_11 : f16, f4E2M1FN)
  %12 = executor.constant -3.0 : f16
  %f4_12 = executor.truncf %12 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%12, %f4_12 : f16, f4E2M1FN)
  %13 = executor.constant -4.0 : f16
  %f4_13 = executor.truncf %13 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%13, %f4_13 : f16, f4E2M1FN)
  %14 = executor.constant -6.0 : f16
  %f4_14 = executor.truncf %14 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%14, %f4_14 : f16, f4E2M1FN)
  return
}

func.func @test_f4_special_values(){
  executor.print "TEST F4E2M1FN SPECIAL VALUES"()
  %pos_inf = executor.constant 0x7C00 : f16
  %f4_pos_inf = executor.truncf %pos_inf : f16 to f4E2M1FN
  executor.print "f16 +inf = %s f4E2M1FN"(%f4_pos_inf : f4E2M1FN)
  %neg_inf = executor.constant 0xFC00 : f16
  %f4_neg_inf = executor.truncf %neg_inf : f16 to f4E2M1FN
  executor.print "f16 -inf = %s f4E2M1FN"(%f4_neg_inf : f4E2M1FN)
  %nan = executor.constant 0x7C01 : f16
  %f4_nan = executor.truncf %nan : f16 to f4E2M1FN
  executor.print "f16 NaN = %s f4E2M1FN"(%f4_nan : f4E2M1FN)
  %zero = executor.constant 0.0 : f16
  %f4_zero = executor.truncf %zero : f16 to f4E2M1FN
  executor.print "f16 zero = %s f4E2M1FN"(%f4_zero : f4E2M1FN)
  %neg_zero = executor.constant -0.0 : f16
  %f4_neg_zero = executor.truncf %neg_zero : f16 to f4E2M1FN
  executor.print "f16 neg zero = %s f4E2M1FN"(%f4_neg_zero : f4E2M1FN)
  return
}

func.func @test_f4_overflow(){
  executor.print "TEST F4E2M1FN OVERFLOW"()
  %c100 = executor.constant 100.0 : f16
  %f4_c100 = executor.truncf %c100 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%c100, %f4_c100 : f16, f4E2M1FN)
  %c100_neg = executor.constant -100.0 : f16
  %f4_c100_neg = executor.truncf %c100_neg : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%c100_neg, %f4_c100_neg : f16, f4E2M1FN)

  %c6p1 = executor.constant 6.1 : f16
  %f4_c6p1 = executor.truncf %c6p1 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%c6p1, %f4_c6p1 : f16, f4E2M1FN)
  %c6p1_neg = executor.constant -6.1 : f16
  %f4_c6p1_neg = executor.truncf %c6p1_neg : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%c6p1_neg, %f4_c6p1_neg : f16, f4E2M1FN)
  return
}

func.func @test_f4_rounding(){
  executor.print "TEST F4E2M1FN ROUNDING"()
  %0 = executor.constant 0.000000235 : f16
  %f4_0 = executor.truncf %0 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%0, %f4_0 : f16, f4E2M1FN)
  %neg_0 = executor.constant -0.000000235 : f16
  %f4_neg_0 = executor.truncf %neg_0 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_0, %f4_neg_0 : f16, f4E2M1FN)

  %1 = executor.constant 0.22 : f16
  %f4_1 = executor.truncf %1 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%1, %f4_1 : f16, f4E2M1FN)
  %neg_1 = executor.constant -0.22 : f16
  %f4_neg_1 = executor.truncf %neg_1 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_1, %f4_neg_1 : f16, f4E2M1FN)

  %2 = executor.constant 0.25 : f16
  %f4_2 = executor.truncf %2 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%2, %f4_2 : f16, f4E2M1FN)
  %neg_2 = executor.constant -0.25 : f16
  %f4_neg_2 = executor.truncf %neg_2 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_2, %f4_neg_2 : f16, f4E2M1FN)

  %3 = executor.constant 0.251 : f16
  %f4_3 = executor.truncf %3 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%3, %f4_3 : f16, f4E2M1FN)
  %neg_3 = executor.constant -0.251 : f16
  %f4_neg_3 = executor.truncf %neg_3 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_3, %f4_neg_3 : f16, f4E2M1FN)

  %4 = executor.constant 0.41 : f16
  %f4_4 = executor.truncf %4 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%4, %f4_4 : f16, f4E2M1FN)
  %neg_4 = executor.constant -0.41 : f16
  %f4_neg_4 = executor.truncf %neg_4 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_4, %f4_neg_4 : f16, f4E2M1FN)

  %5 = executor.constant 1.24 : f16
  %f5_5 = executor.truncf %5 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%5, %f5_5 : f16, f4E2M1FN)
  %neg_5 = executor.constant -1.24 : f16
  %f5_neg_5 = executor.truncf %neg_5 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_5, %f5_neg_5 : f16, f4E2M1FN)

  %6 = executor.constant 1.25 : f16
  %f6_6 = executor.truncf %6 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%6, %f6_6 : f16, f4E2M1FN)
  %neg_6 = executor.constant -1.25 : f16
  %f6_neg_6 = executor.truncf %neg_6 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_6, %f6_neg_6 : f16, f4E2M1FN)

  %7 = executor.constant 1.28 : f16
  %f7_7 = executor.truncf %7 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%7, %f7_7 : f16, f4E2M1FN)
  %neg_7 = executor.constant -1.28 : f16
  %f7_neg_7 = executor.truncf %neg_7 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_7, %f7_neg_7 : f16, f4E2M1FN)

  %8 = executor.constant 1.75 : f16
  %f8_8 = executor.truncf %8 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%8, %f8_8 : f16, f4E2M1FN)
  %neg_8 = executor.constant -1.75 : f16
  %f8_neg_8 = executor.truncf %neg_8 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_8, %f8_neg_8 : f16, f4E2M1FN)

  %9 = executor.constant 3.5 : f16
  %f9_9 = executor.truncf %9 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%9, %f9_9 : f16, f4E2M1FN)
  %neg_9 = executor.constant -3.5 : f16
  %f9_neg_9 = executor.truncf %neg_9 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_9, %f9_neg_9 : f16, f4E2M1FN)

  %10 = executor.constant 5.0 : f16
  %f10_10 = executor.truncf %10 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%10, %f10_10 : f16, f4E2M1FN)
  %neg_10 = executor.constant -5.0 : f16
  %f10_neg_10 = executor.truncf %neg_10 : f16 to f4E2M1FN
  executor.print "f16 %s = %s f4E2M1FN"(%neg_10, %f10_neg_10 : f16, f4E2M1FN)

  return
}

func.func @main() -> i64 {
    %r = executor.constant 0 : i64
    func.call @test_f4_representation() : () -> ()
    func.call @test_f4_special_values() : () -> ()
    func.call @test_f4_overflow():() -> ()
    func.call @test_f4_rounding():() -> ()
    return %r : i64
}

// CHECK-LABEL: TEST F4E2M1FN REPRESENTATION
// CHECK-NEXT: f16 0 = 0 f4E2M1FN
// CHECK-NEXT: f16 0.5 = 0.5 f4E2M1FN
// CHECK-NEXT: f16 1 = 1 f4E2M1FN
// CHECK-NEXT: f16 1.5 = 1.5 f4E2M1FN
// CHECK-NEXT: f16 2 = 2 f4E2M1FN
// CHECK-NEXT: f16 3 = 3 f4E2M1FN
// CHECK-NEXT: f16 4 = 4 f4E2M1FN
// CHECK-NEXT: f16 6 = 6 f4E2M1FN
// CHECK-NEXT: f16 -0.5 = -0.5 f4E2M1FN
// CHECK-NEXT: f16 -1 = -1 f4E2M1FN
// CHECK-NEXT: f16 -1.5 = -1.5 f4E2M1FN
// CHECK-NEXT: f16 -2 = -2 f4E2M1FN
// CHECK-NEXT: f16 -3 = -3 f4E2M1FN
// CHECK-NEXT: f16 -4 = -4 f4E2M1FN
// CHECK-NEXT: f16 -6 = -6 f4E2M1FN

// CHECK-LABEL: TEST F4E2M1FN SPECIAL VALUES
// CHECK-NEXT: f16 +inf = 6 f4E2M1FN
// CHECK-NEXT: f16 -inf = -6 f4E2M1FN
// CHECK-NEXT: f16 NaN = 0 f4E2M1FN
// CHECK-NEXT: f16 zero = 0 f4E2M1FN
// CHECK-NEXT: f16 neg zero = 0 f4E2M1FN

// CHECK-LABEL: TEST F4E2M1FN OVERFLOW
// CHECK-NEXT: f16 100 = 6 f4E2M1FN
// CHECK-NEXT: f16 -100 = -6 f4E2M1FN
// CHECK-NEXT: f16 6.10156 = 6 f4E2M1FN
// CHECK-NEXT: f16 -6.10156 = -6 f4E2M1FN

// CHECK-LABEL: TEST F4E2M1FN ROUNDING
// CHECK-NEXT: f16 2.38419e-07 = 0 f4E2M1FN
// CHECK-NEXT: f16 -2.38419e-07 = -0 f4E2M1FN
// CHECK-NEXT: f16 0.219971 = 0 f4E2M1FN
// CHECK-NEXT: f16 -0.219971 = -0 f4E2M1FN
// CHECK-NEXT: f16 0.25 = 0 f4E2M1FN
// CHECK-NEXT: f16 -0.25 = -0 f4E2M1FN
// CHECK-NEXT: f16 0.250977 = 0.5 f4E2M1FN
// CHECK-NEXT: f16 -0.250977 = -0.5 f4E2M1FN
// CHECK-NEXT: f16 0.409912 = 0.5 f4E2M1FN
// CHECK-NEXT: f16 -0.409912 = -0.5 f4E2M1FN
// CHECK-NEXT: f16 1.24023 = 1 f4E2M1FN
// CHECK-NEXT: f16 -1.24023 = -1 f4E2M1FN
// CHECK-NEXT: f16 1.25 = 1 f4E2M1FN
// CHECK-NEXT: f16 -1.25 = -1 f4E2M1FN
// CHECK-NEXT: f16 1.28027 = 1.5 f4E2M1FN
// CHECK-NEXT: f16 -1.28027 = -1.5 f4E2M1FN
// CHECK-NEXT: f16 1.75 = 2 f4E2M1FN
// CHECK-NEXT: f16 -1.75 = -2 f4E2M1FN
// CHECK-NEXT: f16 3.5 = 4 f4E2M1FN
// CHECK-NEXT: f16 -3.5 = -4 f4E2M1FN
// CHECK-NEXT: f16 5 = 4 f4E2M1FN
// CHECK-NEXT: f16 -5 = -4 f4E2M1FN