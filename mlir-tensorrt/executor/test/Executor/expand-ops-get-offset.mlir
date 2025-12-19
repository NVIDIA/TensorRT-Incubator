// RUN: executor-opt %s -split-input-file -executor-expand-ops -canonicalize -verify-diagnostics | FileCheck %s

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[1] : () -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     return %[[c4_i64]] : i64

// -----

!el_type = !executor.table<f32, f64>

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[1] : () -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     return %[[c16_i64]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f32>, !executor.table<i32, f32>>

func.func @lower_gep(%arg1: i64) -> i64 {
  %0 = executor.getoffset[%arg1, 1, 1] : (i64) -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//  CHECK-SAME: (%[[arg1:.+]]: i64) -> i64 {
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[c12_i64:.+]] = executor.constant 12 : i64
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[arg1]], %[[c16_i64]] : i64
//   CHECK-DAG:     %[[v3:.+]] = executor.addi %[[v0]], %[[c12_i64]] : i64
//   CHECK-DAG:     return %[[v3]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

func.func @lower_gep(%arg1: i64) -> i64 {
  %0 = executor.getoffset[%arg1, 1, 1] : (i64) -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//  CHECK-SAME: (%[[arg0:.+]]: i64) -> i64 {
//   CHECK-DAG:     %[[c32_i64:.+]] = executor.constant 32 : i64
//   CHECK-DAG:     %[[c24_i64:.+]] = executor.constant 24 : i64
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[arg0]], %[[c32_i64]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.addi %[[v0]], %[[c24_i64]] : i64
//   CHECK-DAG:     return %[[v2]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[0, 1, 1] : () -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c24_i64:.+]] = executor.constant 24 : i64
//   CHECK-DAG:     return %[[c24_i64]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i64 {
    // expected-error @+1 {{failed to legalize operation 'executor.getoffset' that was explicitly marked illegal}}
    %0 = executor.getoffset[0, 1, 1] : () -> i64, !el_type
    return %0 : i64
  }
}

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i32 {
    %0 = executor.getoffset[0, 1, 1] : () -> i32, !el_type
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c24:.+]] = executor.constant 24 : i32
//   CHECK-DAG:     return %[[c24]] : i32

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i32 {
    // expected-error @+1 {{failed to legalize operation 'executor.getoffset' that was explicitly marked illegal}}
    %0 = executor.getoffset[0x1FFFFFFFF, 1, 1] : () -> i32, !el_type
    return %0 : i32
  }
}

// -----

!el_type = !executor.table<i8, !executor.table<f32, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 64 : i64>
  >
} {
  func.func @lower_gep_aggregate() -> i64 {
    %0 = executor.getoffset[0, 1] : () -> i64, !el_type
    return %0 : i64
  }
}

// CHECK-LABEL: func.func @lower_gep_aggregate
//  CHECK-SAME: () -> i64 {
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     return %[[c4_i64]] : i64

// -----

func.func @fp4_to_f16(%arg0: f4E2M1FN) -> f16 {
  %0 = executor.extf %arg0 : f4E2M1FN to f16
  return %0 : f16
}

// CHECK-LABEL: func.func @fp4_to_f16
// CHECK-SAME: (%[[ARG0:.+]]: f4E2M1FN) -> f16
// CHECK-DAG: %[[C0_I4:.+]] = executor.constant 0 : i4
// CHECK-DAG: %[[C1_I4:.+]] = executor.constant 1 : i4
// CHECK-DAG: %[[C3_I4:.+]] = executor.constant 3 : i4
// CHECK-DAG: %[[C6_I4:.+]] = executor.constant 6 : i4
// CHECK-DAG: %[[C0_I16:.+]] = executor.constant 0 : i16
// CHECK-DAG: %[[C9_I16:.+]] = executor.constant 9 : i16
// CHECK-DAG: %[[C10_I16:.+]] = executor.constant 10 : i16
// CHECK-DAG: %[[C14_I16:.+]] = executor.constant 14 : i16
// CHECK-DAG: %[[C15_I16:.+]] = executor.constant 15 : i16
// CHECK: %[[BITCAST_I4:.+]] = executor.bitcast %[[ARG0]] : f4E2M1FN to i4
// CHECK: %[[SIGN_SHIFT:.+]] = executor.shift_right_logicali %[[BITCAST_I4]], %[[C3_I4]] : i4
// CHECK: %[[SIGN_EXT:.+]] = executor.zext %[[SIGN_SHIFT]] : i4 to i16
// CHECK: %[[SIGN_POS:.+]] = executor.shift_lefti %[[SIGN_EXT]], %[[C15_I16]] : i16
// CHECK: %[[EXP_MASK:.+]] = executor.bitwise_andi %[[BITCAST_I4]], %[[C6_I4]] : i4
// CHECK: %[[EXP:.+]] = executor.shift_right_logicali %[[EXP_MASK]], %[[C1_I4]] : i4
// CHECK: %[[MANT:.+]] = executor.bitwise_andi %[[BITCAST_I4]], %[[C1_I4]] : i4
// CHECK: %[[MANT_I16:.+]] = executor.zext %[[MANT]] : i4 to i16
// CHECK: %[[IS_EXP_ZERO:.+]] = executor.icmp <eq> %[[EXP]], %[[C0_I4]] : i4
// CHECK: %[[IS_MANT_ZERO:.+]] = executor.icmp <eq> %[[MANT]], %[[C0_I4]] : i4
// CHECK: %[[EXP_I16:.+]] = executor.zext %[[EXP]] : i4 to i16
// CHECK: %[[NORMAL_EXP:.+]] = executor.addi %[[EXP_I16]], %[[C14_I16]] : i16
// CHECK: %[[NORMAL_EXP_POS:.+]] = executor.shift_lefti %[[NORMAL_EXP]], %[[C10_I16]] : i16
// CHECK: %[[NORMAL_MANT_POS:.+]] = executor.shift_lefti %[[MANT_I16]], %[[C9_I16]] : i16
// CHECK: %[[SUBNORM_EXP:.+]] = executor.select %[[IS_MANT_ZERO]], %[[C0_I16]], %[[C14_I16]] : i16
// CHECK: %[[SUBNORM_EXP_POS:.+]] = executor.shift_lefti %[[SUBNORM_EXP]], %[[C10_I16]] : i16
// CHECK: %[[FINAL_EXP:.+]] = executor.select %[[IS_EXP_ZERO]], %[[SUBNORM_EXP_POS]], %[[NORMAL_EXP_POS]] : i16
// CHECK: %[[FINAL_MANT:.+]] = executor.select %[[IS_EXP_ZERO]], %[[C0_I16]], %[[NORMAL_MANT_POS]] : i16
// CHECK: %[[OR1:.+]] = executor.bitwise_ori %[[FINAL_EXP]], %[[SIGN_POS]] : i16
// CHECK: %[[OR2:.+]] = executor.bitwise_ori %[[OR1]], %[[FINAL_MANT]] : i16
// CHECK: %[[RESULT:.+]] = executor.bitcast %[[OR2]] : i16 to f16
// CHECK: return %[[RESULT]] : f16

// -----

func.func @f16_to_fp4(%arg0: f16) -> f4E2M1FN {
  %0 = executor.truncf %arg0 : f16 to f4E2M1FN
  return %0 : f4E2M1FN
}

// CHECK-LABEL: func.func @f16_to_fp4
// CHECK-SAME: (%[[ARG0:.+]]: f16) -> f4E2M1FN
// CHECK-DAG: %[[C8_I4:.+]] = executor.constant -8 : i4
// CHECK-DAG: %[[FALSE:.+]] = executor.constant false
// CHECK-DAG: %[[TRUE:.+]] = executor.constant true
// CHECK-DAG: %[[C0_I16:.+]] = executor.constant 0 : i16
// CHECK-DAG: %[[C10_I16:.+]] = executor.constant 10 : i16
// CHECK-DAG: %[[C13_I16:.+]] = executor.constant 13 : i16
// CHECK-DAG: %[[C14_I16:.+]] = executor.constant 14 : i16
// CHECK-DAG: %[[C15_I16:.+]] = executor.constant 15 : i16
// CHECK-DAG: %[[C16_I16:.+]] = executor.constant 16 : i16
// CHECK-DAG: %[[C17_I16:.+]] = executor.constant 17 : i16
// CHECK-DAG: %[[C31_I16:.+]] = executor.constant 31 : i16
// CHECK-DAG: %[[C256_I16:.+]] = executor.constant 256 : i16
// CHECK-DAG: %[[C512_I16:.+]] = executor.constant 512 : i16
// CHECK-DAG: %[[C768_I16:.+]] = executor.constant 768 : i16
// CHECK-DAG: %[[C1023_I16:.+]] = executor.constant 1023 : i16
// CHECK-DAG: %[[C31744_I16:.+]] = executor.constant 31744 : i16
// CHECK-DAG: %[[C0_I4:.+]] = executor.constant 0 : i4
// CHECK-DAG: %[[C1_I4:.+]] = executor.constant 1 : i4
// CHECK-DAG: %[[C2_I4:.+]] = executor.constant 2 : i4
// CHECK-DAG: %[[C3_I4:.+]] = executor.constant 3 : i4
// CHECK-DAG: %[[C4_I4:.+]] = executor.constant 4 : i4
// CHECK-DAG: %[[C5_I4:.+]] = executor.constant 5 : i4
// CHECK-DAG: %[[C6_I4:.+]] = executor.constant 6 : i4
// CHECK-DAG: %[[C7_I4:.+]] = executor.constant 7 : i4
// CHECK-DAG: %[[CNEG1_I4:.+]] = executor.constant -1 : i4
// CHECK: %[[BITCAST:.+]] = executor.bitcast %[[ARG0]] : f16 to i16
// CHECK: %[[SIGN_SHIFT:.+]] = executor.shift_right_logicali %[[BITCAST]], %[[C15_I16]] : i16
// CHECK: %[[IS_NEG:.+]] = executor.icmp <ne> %[[SIGN_SHIFT]], %[[C0_I16]] : i16
// CHECK: %[[EXP_BITS:.+]] = executor.bitwise_andi %[[BITCAST]], %[[C31744_I16]] : i16
// CHECK: %[[EXP:.+]] = executor.shift_right_logicali %[[EXP_BITS]], %[[C10_I16]] : i16
// CHECK: %[[MANT:.+]] = executor.bitwise_andi %[[BITCAST]], %[[C1023_I16]] : i16
// CHECK: %[[IS_EXP_31:.+]] = executor.icmp <eq> %[[EXP]], %[[C31_I16]] : i16
// CHECK: %[[IS_MANT_ZERO:.+]] = executor.icmp <eq> %[[MANT]], %[[C0_I16]] : i16
// CHECK: %[[SEL1:.+]] = executor.select %[[IS_MANT_ZERO]], %[[TRUE]], %[[FALSE]] : i1
// CHECK: %[[IS_INF:.+]] = executor.select %[[IS_EXP_31]], %[[SEL1]], %[[FALSE]] : i1
// CHECK: %[[SEL2:.+]] = executor.select %[[IS_MANT_ZERO]], %[[FALSE]], %[[TRUE]] : i1
// CHECK: %[[IS_NAN:.+]] = executor.select %[[IS_EXP_31]], %[[SEL2]], %[[FALSE]] : i1
// CHECK: %[[IS_EXP_ZERO:.+]] = executor.icmp <eq> %[[EXP]], %[[C0_I16]] : i16
// CHECK: %[[SEL3:.+]] = executor.select %[[IS_MANT_ZERO]], %[[TRUE]], %[[FALSE]] : i1
// CHECK: %[[IS_ZERO:.+]] = executor.select %[[IS_EXP_ZERO]], %[[SEL3]], %[[FALSE]] : i1
// CHECK: %[[IS_LT_13:.+]] = executor.icmp <ult> %[[EXP]], %[[C13_I16]] : i16
// CHECK: %[[IS_EXP_13:.+]] = executor.icmp <eq> %[[EXP]], %[[C13_I16]] : i16
// CHECK: %[[IS_MANT_ZERO2:.+]] = executor.icmp <eq> %[[MANT]], %[[C0_I16]] : i16
// CHECK: %[[IS_MANT_NONZERO:.+]] = executor.icmp <ne> %[[MANT]], %[[C0_I16]] : i16
// CHECK: %[[ROUND_13_ZERO:.+]] = executor.select %[[IS_EXP_13]], %[[IS_MANT_ZERO2]], %[[FALSE]] : i1
// CHECK: %[[ROUND_13_HALF:.+]] = executor.select %[[IS_EXP_13]], %[[IS_MANT_NONZERO]], %[[FALSE]] : i1
// CHECK: %[[IS_EXP_14:.+]] = executor.icmp <eq> %[[EXP]], %[[C14_I16]] : i16
// CHECK: %[[IS_MANT_LT_512:.+]] = executor.icmp <ult> %[[MANT]], %[[C512_I16]] : i16
// CHECK: %[[SEL4:.+]] = executor.select %[[IS_MANT_LT_512]], %[[FALSE]], %[[TRUE]] : i1
// CHECK: %[[ROUND_14_HALF:.+]] = executor.select %[[IS_EXP_14]], %[[IS_MANT_LT_512]], %[[FALSE]] : i1
// CHECK: %[[ROUND_14_ONE:.+]] = executor.select %[[IS_EXP_14]], %[[SEL4]], %[[FALSE]] : i1
// CHECK: %[[IS_EXP_15:.+]] = executor.icmp <eq> %[[EXP]], %[[C15_I16]] : i16
// CHECK: %[[IS_MANT_LT_256:.+]] = executor.icmp <ult> %[[MANT]], %[[C256_I16]] : i16
// CHECK: %[[IS_MANT_EQ_256:.+]] = executor.icmp <eq> %[[MANT]], %[[C256_I16]] : i16
// CHECK: %[[IS_MANT_LT_768:.+]] = executor.icmp <ult> %[[MANT]], %[[C768_I16]] : i16
// CHECK: %[[IS_MANT_EQ_768:.+]] = executor.icmp <eq> %[[MANT]], %[[C768_I16]] : i16
// CHECK: %[[IS_MANT_GT_256:.+]] = executor.icmp <ugt> %[[MANT]], %[[C256_I16]] : i16
// CHECK: %[[IS_MANT_GT_768:.+]] = executor.icmp <ugt> %[[MANT]], %[[C768_I16]] : i16
// CHECK: %[[SEL5:.+]] = executor.select %[[IS_MANT_EQ_256]], %[[TRUE]], %[[FALSE]] : i1
// CHECK: %[[MANT_LE_256:.+]] = executor.select %[[IS_MANT_LT_256]], %[[TRUE]], %[[SEL5]] : i1
// CHECK: %[[EXP15_TO_1:.+]] = executor.select %[[IS_EXP_15]], %[[MANT_LE_256]], %[[FALSE]] : i1
// CHECK: %[[MANT_BTW_256_768:.+]] = executor.select %[[IS_MANT_GT_256]], %[[IS_MANT_LT_768]], %[[FALSE]] : i1
// CHECK: %[[EXP15_TO_1P5:.+]] = executor.select %[[IS_EXP_15]], %[[MANT_BTW_256_768]], %[[FALSE]] : i1
// CHECK: %[[SEL6:.+]] = executor.select %[[IS_MANT_EQ_768]], %[[TRUE]], %[[FALSE]] : i1
// CHECK: %[[MANT_GE_768:.+]] = executor.select %[[IS_MANT_GT_768]], %[[TRUE]], %[[SEL6]] : i1
// CHECK: %[[EXP15_TO_2:.+]] = executor.select %[[IS_EXP_15]], %[[MANT_GE_768]], %[[FALSE]] : i1
// CHECK: %[[IS_EXP_16:.+]] = executor.icmp <eq> %[[EXP]], %[[C16_I16]] : i16
// CHECK: %[[EXP16_TO_2:.+]] = executor.select %[[IS_EXP_16]], %[[MANT_LE_256]], %[[FALSE]] : i1
// CHECK: %[[EXP16_TO_3:.+]] = executor.select %[[IS_EXP_16]], %[[MANT_BTW_256_768]], %[[FALSE]] : i1
// CHECK: %[[IS_MANT_LT_768_2:.+]] = executor.icmp <ult> %[[MANT]], %[[C768_I16]] : i16
// CHECK: %[[SEL7:.+]] = executor.select %[[IS_MANT_LT_768_2]], %[[FALSE]], %[[TRUE]] : i1
// CHECK: %[[EXP16_TO_4:.+]] = executor.select %[[IS_EXP_16]], %[[SEL7]], %[[FALSE]] : i1
// CHECK: %[[IS_EXP_17:.+]] = executor.icmp <eq> %[[EXP]], %[[C17_I16]] : i16
// CHECK: %[[EXP17_TO_4:.+]] = executor.select %[[IS_EXP_17]], %[[MANT_LE_256]], %[[FALSE]] : i1
// CHECK: %[[EXP17_TO_6:.+]] = executor.select %[[IS_EXP_17]], %[[IS_MANT_GT_256]], %[[FALSE]] : i1
// CHECK: %[[IS_GT_17:.+]] = executor.icmp <ugt> %[[EXP]], %[[C17_I16]] : i16
// CHECK: %[[SEL8:.+]] = executor.select %[[IS_LT_13]], %[[TRUE]], %[[ROUND_13_ZERO]] : i1
// CHECK: %[[SEL9:.+]] = executor.select %[[ROUND_13_HALF]], %[[TRUE]], %[[ROUND_14_HALF]] : i1
// CHECK: %[[SEL10:.+]] = executor.select %[[ROUND_14_ONE]], %[[TRUE]], %[[EXP15_TO_1]] : i1
// CHECK: %[[SEL11:.+]] = executor.select %[[EXP15_TO_2]], %[[TRUE]], %[[EXP16_TO_2]] : i1
// CHECK: %[[SEL12:.+]] = executor.select %[[EXP16_TO_4]], %[[TRUE]], %[[EXP17_TO_4]] : i1
// CHECK: %[[SEL13:.+]] = executor.select %[[EXP17_TO_6]], %[[TRUE]], %[[IS_GT_17]] : i1
// CHECK: %[[VAL1:.+]] = executor.select %[[SEL13]], %[[C7_I4]], %[[C0_I4]] : i4
// CHECK: %[[VAL2:.+]] = executor.select %[[SEL12]], %[[C6_I4]], %[[VAL1]] : i4
// CHECK: %[[VAL3:.+]] = executor.select %[[EXP16_TO_3]], %[[C5_I4]], %[[VAL2]] : i4
// CHECK: %[[VAL4:.+]] = executor.select %[[SEL11]], %[[C4_I4]], %[[VAL3]] : i4
// CHECK: %[[VAL5:.+]] = executor.select %[[EXP15_TO_1P5]], %[[C3_I4]], %[[VAL4]] : i4
// CHECK: %[[VAL6:.+]] = executor.select %[[SEL10]], %[[C2_I4]], %[[VAL5]] : i4
// CHECK: %[[VAL7:.+]] = executor.select %[[SEL9]], %[[C1_I4]], %[[VAL6]] : i4
// CHECK: %[[UNSIGNED_VAL:.+]] = executor.select %[[SEL8]], %[[C0_I4]], %[[VAL7]] : i4
// CHECK: %[[SIGN_BIT_OR:.+]] = executor.bitwise_ori %[[UNSIGNED_VAL]], %[[C8_I4]] : i4
// CHECK: %[[SIGNED_VAL:.+]] = executor.select %[[IS_NEG]], %[[SIGN_BIT_OR]], %[[UNSIGNED_VAL]] : i4
// CHECK: %[[NORMAL_VAL:.+]] = executor.select %[[IS_ZERO]], %[[C0_I4]], %[[SIGNED_VAL]] : i4
// CHECK: %[[SAT_VAL:.+]] = executor.select %[[IS_NEG]], %[[CNEG1_I4]], %[[C7_I4]] : i4
// CHECK: %[[INF_VAL:.+]] = executor.select %[[IS_INF]], %[[SAT_VAL]], %[[NORMAL_VAL]] : i4
// CHECK: %[[FINAL_VAL:.+]] = executor.select %[[IS_NAN]], %[[C0_I4]], %[[INF_VAL]] : i4
// CHECK: %[[RESULT:.+]] = executor.bitcast %[[FINAL_VAL]] : i4 to f4E2M1FN
// CHECK: return %[[RESULT]] : f4E2M1FN
