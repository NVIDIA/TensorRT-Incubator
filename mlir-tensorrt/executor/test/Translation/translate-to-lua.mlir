// RUN:  executor-opt %s -split-input-file -executor-lower-to-runtime-builtins | \
// RUN:   executor-translate -mlir-to-lua | FileCheck %s

// RUN:  executor-opt %s -split-input-file -executor-lower-to-runtime-builtins | \
// RUN:   executor-translate -mlir-to-runtime-executable

func.func @exec_addi(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.addi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_addi
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] + [[l1]];
//  CHECK-NEXT:     return [[l2]];

// -----

func.func @exec_addf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_addf
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] + [[l1]];
//  CHECK-NEXT:     return [[l2]];

// -----

func.func @exec_remf(
    %arg0: f64, %arg1: f64,
    %arg2: f32, %arg3: f32,
    %arg4: f16, %arg5: f16,
    %arg6: f8E4M3FN, %arg7: f8E4M3FN,
    %arg8: bf16, %arg9: bf16) ->
    (f64, f32, f16, f8E4M3FN, bf16) {
  %0 = executor.remf %arg0, %arg1 : f64
  %1 = executor.remf %arg2, %arg3 : f32
  %2 = executor.remf %arg4, %arg5 : f16
  %3 = executor.remf %arg6, %arg7 : f8E4M3FN
  %4 = executor.remf %arg8, %arg9 : bf16
  return %0, %1, %2, %3, %4 : f64, f32, f16, f8E4M3FN, bf16
}

// CHECK-LABEL: exec_remf
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]], [[l2:.+]], [[l3:.+]], [[l4:.+]], [[l5:.+]], [[l6:.+]], [[l7:.+]], [[l8:.+]], [[l9:.+]])
//  CHECK-NEXT:     local [[l10:.+]];
//  CHECK-NEXT:     [[l10]] = _remf_f64([[l0]], [[l1]]);
//  CHECK-NEXT:     [[l0]] = _remf_f32([[l2]], [[l3]]);
//  CHECK-NEXT:     [[l1]] = _remf_f16([[l4]], [[l5]]);
//  CHECK-NEXT:     [[l2]] = _remf_f8E4M3FN([[l6]], [[l7]]);
//  CHECK-NEXT:     [[l3]] = _remf_bf16([[l8]], [[l9]]);
//  CHECK-NEXT:     return [[l10]], [[l0]], [[l1]], [[l2]], [[l3]];
//  CHECK-NEXT: end

// -----

func.func @exec_subi(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_subi
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] - [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_subf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_subf
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] - [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdivi(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_sdivi
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _sdivi_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_divf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_divf
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _divf_f64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end
// -----

func.func @exec_muli(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_muli
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] * [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_mulf(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.mulf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: exec_mulf
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] * [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end
// -----

func.func @exec_srem(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sremi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_srem
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] % [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_andi_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_andi_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_andi_i1([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_ori_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_ori_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_ori_i1([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_xori_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_xori_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _bitwise_xori_i1([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_i32_f32(%arg0: i32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f32], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : i32 to f32
  return %0 : f32
}
// CHECK-LABEL: exec_bitcast_i32_f32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _bitcast_i32_f32([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_i64_f64(%arg0: i64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[i64],[f64], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : i64 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_bitcast_i64_f64
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _bitcast_i64_f64([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f32_i32(%arg0: f32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[i32], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f32 to i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitcast_f32_i32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _bitcast_f32_i32([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f64_i64(%arg0: f64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[f64],[i64], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f64 to i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitcast_f64_i64
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _bitcast_f64_i64([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f16_i16(%arg0: f16) -> i16
  attributes{executor.function_metadata=#executor.func_meta<[f16],[i16], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f16 to i16
  return %0 : i16
}
// CHECK-LABEL: exec_bitcast_f16_i16
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _bitcast_f16_i16([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @umin_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.umin %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: umin_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _umin_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @umax_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.umax %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: umax_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _umax_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @smax_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.smax %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: smax_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _smax_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @smax_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.smax %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: smax_i64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _smax_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @max_f32(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.fmax %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: max_f32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _fmax_f32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @max_f64(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.fmax %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: max_f64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _fmax_f64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_sdiv_i64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _sdivi_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: exec_sdiv_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _sdivi_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_f32(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: exec_sdiv_f32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _divf_f32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_f64(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_sdiv_f64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _divf_f64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT:   end
//  CHECK

// -----

func.func @shift_left_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_lefti %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_left_i64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _shift_lefti_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @shift_left_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_lefti %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_left_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _shift_lefti_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_logical_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_right_logicali %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_right_logical_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _shift_right_logicali_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_logical_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_right_logicali %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_right_logical_i64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:   local [[l2:.+]];
//  CHECK-NEXT:   [[l2]] = _shift_right_logicali_i64([[l0]], [[l1]]);
//  CHECK-NEXT:   return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_arithmetic_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_right_arithmetic_i64
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _shift_right_arithmetici_i64([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_arithmetic_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_right_arithmetic_i32
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = _shift_right_arithmetici_i32([[l0]], [[l1]]);
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sfloor_div(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sfloor_divi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_sfloor_div
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     local [[l2:.+]];
//  CHECK-NEXT:     [[l2]] = [[l0]] // [[l1]];
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @select(%arg0: i1, %arg1: i64, %arg2: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i1, i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.select %arg0, %arg1, %arg2 : i64
  return %0 : i64
}
// CHECK-LABEL: select
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]], [[l2:.+]])
//  CHECK-NEXT:    local [[l3:.+]];
//  CHECK-NEXT:    [[l3]] = _select([[l0]],[[l1]],[[l2]]);
//  CHECK-NEXT:    return [[l3]];
//  CHECK-NEXT: end

// -----

func.func @sqrt_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.sqrt %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: __check_for_function("_sqrt_f32");
// CHECK-LABEL: sqrt_f32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:    local [[l1:.+]];
//  CHECK-NEXT:    [[l1]] = _sqrt_f32([[l0]]);
//  CHECK-NEXT:    return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @abs_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.absf %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: abs_f32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:    local [[l1:.+]];
//  CHECK-NEXT:    [[l1]] = _absf_f32([[l0]]);
//  CHECK-NEXT:    return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @abs_i32(%arg0 : i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32],[i32], num_output_args = 0>}{
  %0 = executor.absi %arg0 : i32
  return %0: i32
}
// CHECK-LABEL: abs_i32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:    local [[l1:.+]];
//  CHECK-NEXT:    [[l1]] = _absi_i32([[l0]]);
//  CHECK-NEXT:    return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @log1p_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.log1p %arg0 : f32
  return %0: f32
}

// CHECK-LABEL: __check_for_function("_log1p_f32");
// CHECK-LABEL: log1p_f32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:    local [[l1:.+]];
//  CHECK-NEXT:    [[l1]] = _log1p_f32([[l0]]);
//  CHECK-NEXT:    return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @neg_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.negf %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: neg_f32
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:    local [[l1:.+]];
//  CHECK-NEXT:    [[l1]] = _negf_f32([[l0]]);
//  CHECK-NEXT:    return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_sitofp_i32_f16(%arg0: i32) -> f16
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f16], num_output_args = 0>}{
  %0 = executor.sitofp %arg0 : i32 to f16
  return %0 : f16
}
// CHECK-LABEL: exec_sitofp_i32_f16
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _sitofp_i32_f16([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_sitofp_i32_f64(%arg0: i32) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f64], num_output_args = 0>}{
  %0 = executor.sitofp %arg0 : i32 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_sitofp_i32_f64
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _sitofp_i32_f64([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_uitofp_i32_f64(%arg0: i32) -> f64 {
  %0 = executor.uitofp %arg0 : i32 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_uitofp_i32_f64
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _uitofp_i32_f64([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_fptosi_f32_i64(%arg0: f32) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[f32],[i64], num_output_args = 0>}{
  %0 = executor.fptosi %arg0 : f32 to i64
  return %0 : i64
}
// CHECK-LABEL: exec_fptosi_f32_i64
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _fptosi_f32_i64([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @exec_fptosi_f16_i8(%arg0: f16) -> i8
  attributes{executor.function_metadata=#executor.func_meta<[f16],[i8], num_output_args = 0>}{
  %0 = executor.fptosi %arg0 : f16 to i8
  return %0 : i8
}
// CHECK-LABEL: exec_fptosi_f16_i8
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]] = _fptosi_f16_i8([[l0]]);
//  CHECK-NEXT:     return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @cf_if_op(%arg0: i64, %arg1: i64) -> i64 attributes {executor.function_metadata = #executor.func_meta<[i64, i64],[i64], num_output_args = 0>} {
    %0 = executor.icmp <eq> %arg0, %arg1 : i64
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = executor.addi %arg0, %arg1 : i64
    cf.br ^bb3(%1 : i64)
  ^bb2:  // pred: ^bb0
    %2 = executor.subi %arg0, %arg1 : i64
    %3 = executor.muli %arg0, %2 : i64
    cf.br ^bb3(%3 : i64)
  ^bb3(%4: i64):  // 2 preds: ^bb1, ^bb2
    return %4 : i64
}

// CHECK-LABEL: cf_if_op
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:   local [[l2:.+]] = nil;
//  CHECK-NEXT:   [[l2]] = _icmp_eq_i64([[l0]], [[l1]]);
//  CHECK-NEXT:   if ([[l2]] == 1) or ([[l2]] == true) then
//  CHECK-NEXT:     goto label1;
//  CHECK-NEXT:   else
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     local [[l3:.+]];
//  CHECK-NEXT:     [[l3]] = [[l0]] + [[l1]];
//  CHECK-NEXT:     [[l2]] = [[l3]];
//  CHECK-NEXT:     goto label3;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     [[l2]] = [[l0]] - [[l1]];
//  CHECK-NEXT:     local [[l3:.+]];
//  CHECK-NEXT:     [[l3]] = [[l0]] * [[l2]];
//  CHECK-NEXT:     [[l2]] = [[l3]];
//  CHECK-NEXT:     goto label3;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @cf_cond_br_forward_entry(%arg0: i64, %arg1: i64) -> i64 attributes {executor.function_metadata = #executor.func_meta<[i64, i64],[i64], num_output_args = 0>} {
    %0 = executor.icmp <eq> %arg0, %arg1 : i64
    %c1 = executor.constant 1 : i64
    cf.cond_br %0, ^bb1(%arg0: i64), ^bb2(%arg1: i64)
  ^bb1(%arg2: i64):  // pred: ^bb0
    %1 = executor.addi %c1, %arg2 : i64
    cf.br ^bb3(%1 : i64)
  ^bb2(%arg3: i64):  // pred: ^bb0
    %2 = executor.subi %c1, %arg3 : i64
    cf.br ^bb3(%2 : i64)
  ^bb3(%4: i64):  // 2 preds: ^bb1, ^bb2
    return %4 : i64
}

// CHECK-LABEL: function cf_cond_br_forward_entry
// CHECK-SAME: ([[l0:.+]], [[l1:.+]])
// CHECK-NEXT:   local [[l4:.+]] = nil;
// CHECK-NEXT:   local [[l5:.+]] = nil;
// CHECK-NEXT:   local [[l2:.+]];
// CHECK-NEXT:   [[l2]] = _icmp_eq_i64([[l0]], [[l1]]);
// CHECK-NEXT:   local [[l3:.+]];
// CHECK-NEXT:   [[l3]] = 1;
// CHECK-NEXT:   if ([[l2]] == 1) or ([[l2]] == true) then
// CHECK-NEXT:     [[l4]] = [[l0]];
// CHECK-NEXT:     goto label1;
// CHECK-NEXT:   else
// CHECK-NEXT:     [[l5]] = [[l1]];
// CHECK-NEXT:     goto label2;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label1:: do
// CHECK-NEXT:     [[l0]] = [[l3]] + [[l4]];
// CHECK-NEXT:     [[l1]] = [[l0]];
// CHECK-NEXT:     goto label3;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label2:: do
// CHECK-NEXT:     [[l0]] = [[l3]] - [[l5]];
// CHECK-NEXT:     [[l1]] = [[l0]];
// CHECK-NEXT:     goto label3;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label3:: do
// CHECK-NEXT:     return [[l1]];
// CHECK-NEXT: end

// -----

func.func @cf_switch_op(%arg0: i64) -> i64 {
  %c0 = executor.constant 0 : i64
  %c1 = executor.constant 1 : i64
  %c2 = executor.constant 2 : i64
  cf.switch %arg0 : i64, [
    default: ^bb1(%c0 : i64),
    1: ^bb2(%c1 : i64),
    2: ^bb3(%c2 : i64)
  ]
  ^bb1(%arg1: i64):
    %1 = executor.addi %arg1, %c1 : i64
    cf.br ^bb3(%1 : i64)
  ^bb2(%arg2: i64):
    %2 = executor.addi %arg2, %c2 : i64
    cf.br ^bb3(%2 : i64)
  ^bb3(%arg3: i64):
    return %arg3 : i64
}

// CHECK-LABEL: cf_switch_op
//  CHECK-SAME: ([[l0:.+]])
//  CHECK-NEXT:   local [[l4:.+]] = nil;
//  CHECK-NEXT:   local [[l5:.+]] = nil;
//  CHECK-NEXT:   local [[l6:.+]] = nil;
//  CHECK-NEXT:   local [[l1:.+]];
//  CHECK-NEXT:   [[l1]] = 0;
//  CHECK-NEXT:   local [[l2:.+]];
//  CHECK-NEXT:   [[l2]] = 1;
//  CHECK-NEXT:   local [[l3:.+]];
//  CHECK-NEXT:   [[l3]] = 2;
//  CHECK-NEXT:   if ([[l0]] == 1) then
//  CHECK-NEXT:     [[l5]] = [[l2]];
//  CHECK-NEXT:     goto label1;
//  CHECK-NEXT:   elseif ([[l0]] == 2) then
//  CHECK-NEXT:     [[l6]] = [[l3]];
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   else
//  CHECK-NEXT:     [[l4]] = [[l1]];
//  CHECK-NEXT:     goto label3;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     [[l0]] = [[l4]] + [[l2]];
//  CHECK-NEXT:     [[l6]] = [[l0]];
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     [[l0]] = [[l5]] + [[l3]];
//  CHECK-NEXT:     [[l6]] = [[l0]];
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     return [[l6]];
//  CHECK-NEXT:   end
//  CHECK-NEXT: end

// -----

// This test contains a nested loop structure. The cond_br in bb3 jumps to bb1 and
// must assign the block argument arg2 (bb2). The assignment should only occur if
// the branch condition is false, otherwise arg2(bb2) referenced in bb4 will be
// wrong.

func.func @block_arg_handling(%arg0: i64, %arg1: i64) {
  %0 = executor.addi %arg0, %arg1 : i64
  %c0 = executor.constant 0 : i64
  %c1 = executor.constant 1 : i64
  %c5 = executor.constant 5 : i64
  cf.br ^bb1(%0: i64)
^bb1(%arg2: i64):
  %1 = executor.icmp <slt> %arg2, %c5 : i64
  cf.cond_br %1, ^bb2, ^bb5
^bb2:
  %2 = executor.addi %arg2, %c1 : i64
  cf.br ^bb3(%c0: i64)
^bb3(%arg3: i64):
  %3 = executor.icmp <sle> %arg3, %arg2 : i64
  cf.cond_br %3, ^bb4, ^bb1(%2: i64)
^bb4:
  executor.print "%arg2 = %d"(%arg2 : i64)
  %4 = executor.addi %arg2, %c1 : i64
  cf.br ^bb3(%4: i64)
^bb5:
  return
}

// CHECK-LABEL: function block_arg_handling
//  CHECK-SAME:  ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:   local [[l4:.+]] = nil;
//  CHECK-NEXT:   local [[l2:.+]] = nil;
//  CHECK-NEXT:   local [[l5:.+]] = nil;
//  CHECK-NEXT:   [[l2]] = [[l0]] + [[l1]];
//  CHECK-NEXT:   [[l0]] = 0;
//  CHECK-NEXT:   [[l1]] = 1;
//  CHECK-NEXT:   local [[l3:.+]];
//  CHECK-NEXT:   [[l3]] = 5;
//  CHECK-NEXT:   [[l4]] = [[l2]];
//  CHECK-NEXT:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     [[l2]] = _icmp_slt_i64([[l4]], [[l3]]);
//  CHECK-NEXT:     if ([[l2]] == 1) or ([[l2]] == true) then
//  CHECK-NEXT:       goto label2;
//  CHECK-NEXT:     else
//  CHECK-NEXT:       goto label3;
//  CHECK-NEXT:     end
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     [[l2]] = [[l4]] + [[l1]];
//  CHECK-NEXT:     [[l5]] = [[l0]];
//  CHECK-NEXT:     goto label4;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label4:: do
//  CHECK-NEXT:     local [[l6:.+]];
//  CHECK-NEXT:     [[l6]] = _icmp_sle_i64([[l5]], [[l4]]);
//  CHECK-NEXT:     if ([[l6]] == 1) or ([[l6]] == true) then
//  CHECK-NEXT:       goto label5;
//  CHECK-NEXT:     else
//  CHECK-NEXT:       [[l4]] = [[l2]];
//  CHECK-NEXT:       goto label1;
//  CHECK-NEXT:     end
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label5:: do
//  CHECK-NEXT:     print(string.format("%arg2 = %d", [[l4]]));
//  CHECK-NEXT:     local [[l6:.+]];
//  CHECK-NEXT:     [[l6]] = [[l4]] + [[l1]];
//  CHECK-NEXT:     [[l5]] = [[l6]];
//  CHECK-NEXT:     goto label4;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     return;
//  CHECK-NEXT:   end
//  CHECK-NEXT: end

// -----

func.func @executor_ops(%arg0: i64, %arg1: i64) -> i64 attributes {executor.function_metadata = #executor.func_meta<[i64, i64],[i64], num_output_args = 0>} {
  %c0 = executor.constant 0 : i64
  %c1 = executor.constant 1 : i64
  %0 = executor.addi %arg0, %arg1 : i64
  %1 = executor.subi %0, %c1 : i64
  %2 = executor.muli %0, %1 : i64
  return %2 : i64
}

// CHECK-LABEL: executor_ops
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:   local [[l2:.+]];
//  CHECK-NEXT:   [[l2]] = 0;
//  CHECK-NEXT:   [[l2]] = 1;
//  CHECK-NEXT:   local [[l3:.+]];
//  CHECK-NEXT:   [[l3]] = [[l0]] + [[l1]];
//  CHECK-NEXT:   [[l0]] = [[l3]] - [[l2]];
//  CHECK-NEXT:   [[l1]] = [[l3]] * [[l0]];
//  CHECK-NEXT:   return [[l1]];
//  CHECK-NEXT: end

// -----

func.func @cf_for_op(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {executor.function_metadata = #executor.func_meta<[i64, i64, i64],[i64], num_output_args = 0>} {
  %c0 = executor.constant 0 : i64
  cf.br ^bb1(%arg0, %c0 : i64, i64)
^bb1(%0: i64, %1: i64):  // 2 preds: ^bb0, ^bb2
  %2 = executor.icmp <slt> %0, %arg1 : i64
  cf.cond_br %2, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %3 = executor.addi %1, %0 : i64
  %4 = executor.addi %0, %arg2 : i64
  cf.br ^bb1(%4, %3 : i64, i64)
^bb3:  // pred: ^bb1
  return %1 : i64
}

// CHECK-LABEL: cf_for_op
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]], [[l2:.+]])
//  CHECK-NEXT:   local [[l4:.+]] = nil;
//  CHECK-NEXT:   local [[l5:.+]] = nil;
//  CHECK-NEXT:   local [[l3:.+]];
//  CHECK-NEXT:   [[l3]] = 0;
//  CHECK-NEXT:   [[l4]] = [[l0]];
//  CHECK-NEXT:   [[l5]] = [[l3]];
//  CHECK-NEXT:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     [[l0]] = _icmp_slt_i64([[l4]], [[l1]]);
//  CHECK-NEXT:     if ([[l0]] == 1) or ([[l0]] == true) then
//  CHECK-NEXT:       goto label2;
//  CHECK-NEXT:     else
//  CHECK-NEXT:       goto label3;
//  CHECK-NEXT:     end
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     [[l0]] = [[l5]] + [[l4]];
//  CHECK-NEXT:     [[l3]] = [[l4]] + [[l2]];
//  CHECK-NEXT:     [[l4]] = [[l3]];
//  CHECK-NEXT:     [[l5]] = [[l0]];
//  CHECK-NEXT:     goto label1;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     return [[l5]];
//  CHECK-NEXT:   end
//  CHECK-NEXT: end

// -----

func.func @executor_print_op(%arg0: i64, %arg1: f32) attributes {executor.function_metadata = #executor.func_meta<[i64, f32],[], num_output_args = 0>} {
  executor.print (%arg0, %arg1 : i64, f32)
  executor.print "hello %d, %f"( %arg0, %arg1 : i64, f32 )
  executor.print "hello world"()
  return
}

// CHECK-LABEL: executor_print_op
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]])
//  CHECK-NEXT:     print([[l0]], [[l1]]);
//  CHECK-NEXT:     print(string.format("hello %d, %f", [[l0]], [[l1]]));
//  CHECK-NEXT:     print(string.format("hello world"));
//  CHECK-NEXT:     return;
//  CHECK-NEXT: end

// -----

!descriptor2d = !executor.table<
    i64,
    i64,
    i32,
    i32, i32,
    i32, i32>

func.func @executor_aggregates(%arg0: i64, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> (!descriptor2d, !descriptor2d, i32, !descriptor2d)
    attributes {executor.function_metadata = #executor.func_meta<[memref<2xi32>],[memref<2xi32>, memref<2xi32>, i32, memref<2xi32>], num_output_args = 0>} {
  %0 = executor.table.create : !descriptor2d
  %1 = executor.table.create (%arg0, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : i64, i64, i32, i32, i32, i32, i32): !descriptor2d
  %2 = executor.table.get %1[2] : !descriptor2d
  %c0 = executor.constant 0 : i32
  %3 = executor.table.set %c0 into %1[2]  : i32, !descriptor2d
  return %0, %1, %2, %3 : !descriptor2d, !descriptor2d, i32, !descriptor2d
}

// CHECK-LABEL: executor_aggregates
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]], [[l2:.+]], [[l3:.+]], [[l4:.+]], [[l5:.+]])
//  CHECK-NEXT:   local [[l6:.+]];
//  CHECK-NEXT:   [[l6]] = {};
//  CHECK-NEXT:   local [[l7:.+]];
//  CHECK-NEXT:   [[l7]] = {[[l0]], [[l0]], [[l1]], [[l2]], [[l3]], [[l4]], [[l5]]};
//  CHECK-NEXT:   [[l0]] = [[l7]][3];
//  CHECK-NEXT:   [[l1]] = 0;
//  CHECK-NEXT:   [[l2]] = {};
//  CHECK-NEXT:   for j,x in ipairs([[l7]]) do [[l2]][j] = x end;
//  CHECK-NEXT:   [[l2]][3] = [[l1]];
//  CHECK-NEXT:   return [[l6]], [[l7]], [[l0]], [[l2]];
//  CHECK-NEXT: end

// -----

!descriptor2d = !executor.table<i64, i64, i64, i64>

func.func @executor_dynamic_extract(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %index: i32) -> (!descriptor2d, i64)
    attributes {executor.function_metadata = #executor.func_meta<[i64, i64, i64, i64, i32],[memref<4xi32>, i64], num_output_args = 0>} {
  %0 = executor.table.create (%arg0, %arg1, %arg2, %arg3 : i64, i64, i64, i64): !descriptor2d
  %1 = executor.table.dynamic_get %0[%index] : (!descriptor2d, i32) -> i64
  return %0, %1 : !descriptor2d, i64
}

// CHECK-LABEL: executor_dynamic_extract
//  CHECK-SAME: ([[l0:.+]], [[l1:.+]], [[l2:.+]], [[l3:.+]], [[l4:.+]])
//  CHECK-NEXT:   local [[l5:.+]];
//  CHECK-NEXT:   [[l5]] = {[[l0]], [[l1]], [[l2]], [[l3]]};
//  CHECK-NEXT:   [[l0]] = [[l5]][[[l4]] + 1];
//  CHECK-NEXT:   return [[l5]], [[l0]];
//  CHECK-NEXT: end

// -----

func.func @const_literal() attributes {executor.function_metadata = #executor.func_meta<[],[], num_output_args = 0>} {
  %0 = executor.str_literal "function_1"
  executor.print "%s"(%0 : !executor.str_literal)
  return
}

// CHECK-LABEL: function const_literal
//  CHECK-NEXT:   local [[l0:.+]];
//  CHECK-NEXT:   [[l0]] = "function_1";
//  CHECK-NEXT:   print(string.format("%s", [[l0]]));
//  CHECK-NEXT:   return;
//  CHECK-NEXT: end

// -----
#metadata = #executor.func_meta<[i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,
i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32,i32],[i32], num_output_args = 0>

func.func @return_two_vals(%arg0: i32, %arg1: i32) -> (i32, i32) attributes {
  executor.function_metadata = #executor.func_meta<[i32, i32], [i32, i32], num_output_args = 0>
} {
  return %arg0, %arg1 : i32, i32
}

func.func @max_local_allocation_test(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32,
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
  %arg190: i32, %arg191: i32, %arg192: i32, %arg193: i32, %arg194: i32, %arg195: i32, %arg196: i32, %arg197: i32, %arg198: i32) -> i32
   attributes {executor.function_metadata = #metadata} {
  %3, %4 = func.call @return_two_vals(%arg0, %arg0) : (i32, i32) -> (i32, i32)
  %5, %6 = func.call @return_two_vals(%3, %4) : (i32, i32) -> (i32, i32)

  return %5 : i32
}

func.func @other_func(%arg0: i32) -> i32 {
  %0 = executor.addi %arg0, %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: function max_local_allocation_test
//  CHECK-SAME:  ([[arg0:l[0-9]+]],
//  CHECK-NEXT:      local [[locals:.+]] = {};
//  CHECK-NEXT:      [[locals]].[[t0:.+]], [[locals]].[[t1:.+]] = return_two_vals([[arg0]], [[arg0]]);
//  CHECK-NEXT:      [[l0:l[0-9]+]], [[l1:l[0-9]+]] = return_two_vals([[locals]].[[t0]], [[locals]].[[t1]]);
//  CHECK-NEXT:      return [[l0]];
//  CHECK-NEXT: end

// -----

func.func @cmpi_ugt(%arg0: i32, %arg1: i32) -> i1
    attributes {executor.function_metadata = #executor.func_meta<[i32, i32],[i1], num_output_args = 0>} {
  %0 = executor.icmp <ugt> %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: function cmpi_ugt ({{.+}}, {{.+}})
//  CHECK-NEXT:   local [[l2:.+]];
//  CHECK-NEXT:   [[l2]] = _icmp_ugt_i32({{.+}}, {{.+}});
//  CHECK-NEXT:   return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @cmpi_ult(%arg0: i32, %arg1: i32) -> i1
    attributes {executor.function_metadata = #executor.func_meta<[i32, i32],[i1], num_output_args = 0>} {
  %0 = executor.icmp <ult> %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: function cmpi_ult ({{.+}}, {{.+}})
//  CHECK-NEXT:   local [[l2:.+]];
//  CHECK-NEXT:   [[l2]] = _icmp_ult_i32({{.+}}, {{.+}});
//  CHECK-NEXT:   return [[l2]];
//  CHECK-NEXT: end

// -----

func.func @test_assert(%arg0: i1) {
  executor.assert %arg0, "assertion message"
  return
}

// CHECK-LABEL: function test_assert
//  CHECK-SAME:    ([[l0:.*]])
//  CHECK-NEXT:   assert(([[l0]] == 1) or ([[l0]] == true), "assertion message");
//  CHECK-NEXT:   return;
//  CHECK-NEXT: end

// -----

func.func @coro(%arg0: i32) -> i32 {
  executor.coro_yield %arg0 : i32
  return %arg0 : i32
}

func.func @coro_await() -> (i32) {
  %c0 = executor.constant 0 : i32
  %coro = executor.coro_create @coro : (i32) -> i32
  %0:2 = executor.coro_await %coro (%c0 :  i32) : (i32) -> i32
  %1:2 = executor.coro_await %coro () : (i32) -> i32
  return %1#1 : i32
}

//  CHECK-LABEL: function coro
//   CHECK-SAME:  ([[l0:.+]])
//   CHECK-NEXT:   coroutine.yield([[l0]]);
//   CHECK-NEXT:   return [[l0]];
//   CHECK-NEXT: end
// CHECK-LABEL: function coro_await ()
//   CHECK-NEXT:  local [[l0:.+]];
//   CHECK-NEXT:  [[l0]] = 0;
//   CHECK-NEXT:  local [[l1:.+]];
//   CHECK-NEXT:  [[l1]] = coroutine.create(coro);
//   CHECK-NEXT:  local [[l2:.+]];
//   CHECK-NEXT:  local [[l3:.+]];
//   CHECK-NEXT:  [[l2]], [[l3]] = coroutine.resume([[l1]], [[l0]]);
//   CHECK-NEXT:  [[l0]], [[l2]] = coroutine.resume([[l1]]);
//   CHECK-NEXT:  return [[l2]];
//   CHECK-NEXT: end

// -----

// This test ensures that an operation with both local and outer-scope results
// can be emitted correctly.

func.func @coro(%arg0: i32) -> i32 {
  executor.coro_yield %arg0 : i32
  return %arg0 : i32
}

func.func @test_mixed_scoped_results() -> (i32) {
  %c0 = executor.constant 0 : i32
  cf.br ^bb0
^bb0:
  %coro = executor.coro_create @coro : (i32) -> i32
  %0:2 = executor.coro_await %coro () : (i32) -> i32
  cf.br ^bb1
^bb1:
  return %0#1 : i32
}


// CHECK-LABEL: function coro
// CHECK-LABEL: function test_mixed_scoped_results ()
//  CHECK-NEXT:   local [[l2:.+]] = nil;
//  CHECK-NEXT:   local [[l0:.+]];
//  CHECK-NEXT:   [[l0]] = 0;
//  CHECK-NEXT:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     [[l0]] = coroutine.create(coro);
//  CHECK-NEXT:     local [[l1:.+]];
//  CHECK-NEXT:     [[l1]], [[l2]] = coroutine.resume([[l0]]);
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:    ::label2:: do
//  CHECK-NEXT:      return [[l2]];
//  CHECK-NEXT:    end
//  CHECK-NEXT: end

// -----

func.func @test_values_with_external_use(%arg0: i32) -> i32 {
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

//  CHECK-LABEL: function test_values_with_external_use
//   CHECK-SAME:  ([[l0:.+]])
//   CHECK-NEXT:   local [[l2:.+]] = nil;
//   CHECK-NEXT:   local [[l3:.+]] = nil;
//   CHECK-NEXT:   local [[l1:.+]];
//   CHECK-NEXT:   [[l1]] = 1;
//   CHECK-NEXT:   goto label1;
//   CHECK-NEXT:   ::label1:: do
//   CHECK-NEXT:     [[l2]] = [[l0]] + [[l1]];
//   CHECK-NEXT:     [[l3]] = [[l0]] - [[l1]];
//   CHECK-NEXT:     goto label2;
//   CHECK-NEXT:   end
//   CHECK-NEXT:   ::label2:: do
//   CHECK-NEXT:     [[l0]] = [[l2]] + [[l3]];
//   CHECK-NEXT:     return [[l0]];
//   CHECK-NEXT:   end
//   CHECK-NEXT: end
