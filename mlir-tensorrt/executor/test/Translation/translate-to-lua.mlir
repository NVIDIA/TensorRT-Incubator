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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] + [[v2]];
//  CHECK-NEXT:     return [[v3]];
// -----

func.func @exec_addf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_addf
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] + [[v2]];
//  CHECK-NEXT:     return [[v3]];

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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]], [[v3:.+]], [[v4:.+]], [[v5:.+]], [[v6:.+]], [[v7:.+]], [[v8:.+]], [[v9:.+]], [[v10:.+]])
//  CHECK-NEXT:     local [[v11:.+]] <const> = _remf_f64([[v1]], [[v2]]);
//  CHECK-NEXT:     local [[v12:.+]] <const> = _remf_f32([[v3]], [[v4]]);
//  CHECK-NEXT:     local [[v13:.+]] <const> = _remf_f16([[v5]], [[v6]]);
//  CHECK-NEXT:     local [[v14:.+]] <const> = _remf_f8E4M3FN([[v7]], [[v8]]);
//  CHECK-NEXT:     local [[v15:.+]] <const> = _remf_bf16([[v9]], [[v10]]);
//  CHECK-NEXT:     return [[v11]], [[v12]], [[v13]], [[v14]], [[v15]];
//  CHECK-NEXT: end

// -----

func.func @exec_subi(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_subi
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] - [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_subf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_subf
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] - [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdivi(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_sdivi
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _sdivi_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_divf(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_divf
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _divf_f64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end
// -----

func.func @exec_muli(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_muli
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] * [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_mulf(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.mulf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: exec_mulf
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] * [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end
// -----

func.func @exec_srem(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sremi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_srem
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] % [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_andi_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_andi_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_and(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_andi %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_and
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_andi_i1([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_ori_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_ori_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_or(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_ori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_or
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_ori_i1([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_xori_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_xori_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitwise_xor(%arg0: i1, %arg1: i1) -> i1
  attributes{executor.function_metadata=#executor.func_meta<[i1, i1],[i1], num_output_args = 0>}{
  %0 = executor.bitwise_xori %arg0, %arg1 : i1
  return %0 : i1
}
// CHECK-LABEL: exec_bitwise_xor
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _bitwise_xori_i1([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_i32_f32(%arg0: i32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f32], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : i32 to f32
  return %0 : f32
}
// CHECK-LABEL: exec_bitcast_i32_f32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _bitcast_i32_f32([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_i64_f64(%arg0: i64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[i64],[f64], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : i64 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_bitcast_i64_f64
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _bitcast_i64_f64([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f32_i32(%arg0: f32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[i32], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f32 to i32
  return %0 : i32
}
// CHECK-LABEL: exec_bitcast_f32_i32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _bitcast_f32_i32([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f64_i64(%arg0: f64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[f64],[i64], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f64 to i64
  return %0 : i64
}
// CHECK-LABEL: exec_bitcast_f64_i64
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _bitcast_f64_i64([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_bitcast_f16_i16(%arg0: f16) -> i16
  attributes{executor.function_metadata=#executor.func_meta<[f16],[i16], num_output_args = 0>}{
  %0 = executor.bitcast %arg0 : f16 to i16
  return %0 : i16
}
// CHECK-LABEL: exec_bitcast_f16_i16
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _bitcast_f16_i16([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @max_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.smax %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: __check_for_function("_smax_i32")
// CHECK-LABEL: max_i32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _smax_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @max_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.smax %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: __check_for_function("_smax_i64")
// CHECK-LABEL: max_i64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _smax_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @max_f32(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.fmax %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: __check_for_function("_fmax_f32")
// CHECK-LABEL: max_f32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _fmax_f32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @max_f64(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.fmax %arg0, %arg1 : f64
  return %0 : f64
}
// CHECK-LABEL: __check_for_function("_fmax_f64")
// CHECK-LABEL: max_f64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _fmax_f64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i64
  return %0 : i64
}
// CHECK-LABEL: exec_sdiv_i64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _sdivi_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.sdivi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: exec_sdiv_i32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _sdivi_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_f32(%arg0: f32, %arg1: f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32, f32],[f32], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: exec_sdiv_f32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _divf_f32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sdiv_f64(%arg0: f64, %arg1: f64) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[f64, f64],[f64], num_output_args = 0>}{
  %0 = executor.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: exec_sdiv_f64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _divf_f64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT:   end
//  CHECK

// -----

func.func @shift_left_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_lefti %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_left_i64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _shift_lefti_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @shift_left_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_lefti %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_left_i32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _shift_lefti_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_logical_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_right_logicali %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_right_logical_i32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _shift_right_logicali_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_logical_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_right_logicali %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_right_logical_i64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:   local [[v3:.+]] <const> = _shift_right_logicali_i64([[v1]], [[v2]]);
//  CHECK-NEXT:   return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_arithmetic_i64(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: shift_right_arithmetic_i64
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _shift_right_arithmetici_i64([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @shift_right_arithmetic_i32(%arg0: i32, %arg1: i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>}{
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: shift_right_arithmetic_i32
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = _shift_right_arithmetici_i32([[v1]], [[v2]]);
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @exec_sfloor_div(%arg0: i64, %arg1: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.sfloor_divi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: exec_sfloor_div
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = [[v1]] // [[v2]];
//  CHECK-NEXT:     return [[v3]];
//  CHECK-NEXT: end

// -----

func.func @select(%arg0: i1, %arg1: i64, %arg2: i64) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[i1, i64, i64],[i64], num_output_args = 0>}{
  %0 = executor.select %arg0, %arg1, %arg2 : i64
  return %0 : i64
}
// CHECK-LABEL: select
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]], [[v3:.+]])
//  CHECK-NEXT:    local [[v4:.+]] <const> = _select([[v1]],[[v2]],[[v3]]);
//  CHECK-NEXT:    return [[v4]];
//  CHECK-NEXT: end

// -----

func.func @sqrt_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.sqrt %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: __check_for_function("_sqrt_f32");
// CHECK-LABEL: sqrt_f32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:    local [[v2:.+]] <const> = _sqrt_f32([[v1]]);
//  CHECK-NEXT:    return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @abs_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.absf %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: abs_f32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:    local [[v2:.+]] <const> = _absf_f32([[v1]]);
//  CHECK-NEXT:    return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @abs_i32(%arg0 : i32) -> i32
  attributes{executor.function_metadata=#executor.func_meta<[i32],[i32], num_output_args = 0>}{
  %0 = executor.absi %arg0 : i32
  return %0: i32
}
// CHECK-LABEL: abs_i32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:    local [[v2:.+]] <const> = _absi_i32([[v1]]);
//  CHECK-NEXT:    return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @log1p_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.log1p %arg0 : f32
  return %0: f32
}

// CHECK-LABEL: __check_for_function("_log1p_f32");
// CHECK-LABEL: log1p_f32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:    local [[v2:.+]] <const> = _log1p_f32([[v1]]);
//  CHECK-NEXT:    return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @neg_f32(%arg0 : f32) -> f32
  attributes{executor.function_metadata=#executor.func_meta<[f32],[f32], num_output_args = 0>}{
  %0 = executor.negf %arg0 : f32
  return %0: f32
}
// CHECK-LABEL: neg_f32
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:    local [[v2:.+]] <const> = _negf_f32([[v1]]);
//  CHECK-NEXT:    return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sitofp_i32_f16(%arg0: i32) -> f16
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f16], num_output_args = 0>}{
  %0 = executor.sitofp %arg0 : i32 to f16
  return %0 : f16
}
// CHECK-LABEL: exec_sitofp_i32_f16
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _sitofp_i32_f16([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_sitofp_i32_f64(%arg0: i32) -> f64
  attributes{executor.function_metadata=#executor.func_meta<[i32],[f64], num_output_args = 0>}{
  %0 = executor.sitofp %arg0 : i32 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_sitofp_i32_f64
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _sitofp_i32_f64([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_uitofp_i32_f64(%arg0: i32) -> f64 {
  %0 = executor.uitofp %arg0 : i32 to f64
  return %0 : f64
}
// CHECK-LABEL: exec_uitofp_i32_f64
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _uitofp_i32_f64([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_fptosi_f32_i64(%arg0: f32) -> i64
  attributes{executor.function_metadata=#executor.func_meta<[f32],[i64], num_output_args = 0>}{
  %0 = executor.fptosi %arg0 : f32 to i64
  return %0 : i64
}
// CHECK-LABEL: exec_fptosi_f32_i64
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _fptosi_f32_i64([[v1]]);
//  CHECK-NEXT:     return [[v2]];
//  CHECK-NEXT: end

// -----

func.func @exec_fptosi_f16_i8(%arg0: f16) -> i8
  attributes{executor.function_metadata=#executor.func_meta<[f16],[i8], num_output_args = 0>}{
  %0 = executor.fptosi %arg0 : f16 to i8
  return %0 : i8
}
// CHECK-LABEL: exec_fptosi_f16_i8
//  CHECK-SAME: ([[v1:.+]])
//  CHECK-NEXT:     local [[v2:.+]] <const> = _fptosi_f16_i8([[v1]]);
//  CHECK-NEXT:     return [[v2]];
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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:   local [[barg2:.+]] = nil
//  CHECK-NEXT:   local [[v3:.+]] <const> = _icmp_eq_i64([[v1]], [[v2]])
//  CHECK-NEXT:   if ([[v3]] == 1) or ([[v3]] == true) then
//  CHECK-NEXT:     goto label1;
//  CHECK-NEXT:   else
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     local [[v4:.+]] <const> = [[v1]] + [[v2]];
//  CHECK-NEXT:     [[barg2:.+]] = [[v4]];
//  CHECK-NEXT:     goto label3;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     local [[v6:.+]] <const> = [[v1]] - [[v2]];
//  CHECK-NEXT:     local [[v7:.+]] <const> = [[v1]] * [[v6]];
//  CHECK-NEXT:     [[barg2:.+]] = [[v7]];
//  CHECK-NEXT:     goto label3;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     return[[barg2]];
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
// CHECK-SAME: ([[arg0:.+]], [[arg1:.+]])
// CHECK-NEXT:   local [[barg2:barg.+]] = nil
// CHECK-NEXT:   local [[barg3:barg.+]] = nil
// CHECK-NEXT:   local [[barg4:barg.+]] = nil
// CHECK-NEXT:   local [[l2:.+]] <const>  = _icmp_eq_i64([[arg0]], [[arg1]]);
// CHECK-NEXT:   local [[l3:.+]] <const>  = 1;
// CHECK-NEXT:   if ([[l2]] == 1) or ([[l2]] == true) then
// CHECK-NEXT:     [[barg2]] = [[arg0]]
// CHECK-NEXT:     goto label1;
// CHECK-NEXT:   else
// CHECK-NEXT:     [[barg3]] = [[arg1]]
// CHECK-NEXT:     goto label2;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label1:: do
// CHECK-NEXT:     local [[l6:.+]] <const>  = [[l3]] + [[barg2]];
// CHECK-NEXT:     [[barg4]] = [[l6]];
// CHECK-NEXT:     goto label3;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label2:: do
// CHECK-NEXT:     local [[l6:.+]] <const>  = [[l3]] - [[barg3]];
// CHECK-NEXT:     [[barg4]] = [[l6]];
// CHECK-NEXT:     goto label3;
// CHECK-NEXT:   end
// CHECK-NEXT:   ::label3:: do
// CHECK-NEXT:     return [[barg4]];
// CHECK-NEXT:   end
// CHECK-NEXT: end

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
//  CHECK-SAME:  ([[arg0:.+]], [[arg1:.+]])
//   CHECK-DAG:   local [[barg2:barg2]] = nil;
//   CHECK-DAG:   local [[l3:.+]] = nil;
//   CHECK-DAG:   local [[barg4:barg4]] = nil;
//   CHECK-DAG:   local [[l5:.+]] <const>  = [[arg0]] + [[arg1]];
//   CHECK-DAG:   local [[l6:.+]] <const>  = 0;
//   CHECK-DAG:   local [[l7:.+]] <const>  = 1;
//   CHECK-DAG:   local [[l8:.+]] <const>  = 5;
//       CHECK:   [[barg2]] = [[l5]];
//       CHECK:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     local [[l9:.+]] <const>  = _icmp_slt_i64([[barg2]], [[l8]]);
//  CHECK-NEXT:     if ([[l9]] == 1) or ([[l9]] == true) then
//  CHECK-NEXT:       goto label2;
//  CHECK-NEXT:     else
//  CHECK-NEXT:       goto label3;
//  CHECK-NEXT:     end
//       CHECK:   end
//       CHECK:   ::label2:: do
//  CHECK-NEXT:     [[l3]] = [[barg2]] + [[l7]];
//  CHECK-NEXT:     [[barg4]] = [[l6]];
//  CHECK-NEXT:     goto label4;
//       CHECK:   end
//       CHECK:   ::label4:: do
//       CHECK:     local [[l9:.+]] <const>  = _icmp_sle_i64([[barg4]], [[barg2]]);
//       CHECK:     if ([[l9]] == 1) or ([[l9]] == true) then
//  CHECK-NEXT:       goto label5;
//       CHECK:     else
//  CHECK-NEXT:      [[barg2]] = [[l3]];
//  CHECK-NEXT:      goto label1;
//       CHECK:     end
//       CHECK:   end
//       CHECK:   ::label5:: do
//   CHECK-DAG:     print(string.format("%arg2 = %d", [[barg2]]));
//   CHECK-DAG:     local [[l9:.+]] <const>  = [[barg2]] + [[l7]];
//  CHECK-NEXT:     [[barg4]] = [[l9]];
//  CHECK-NEXT:     goto label4;
//       CHECK:   end
//       CHECK:   ::label3:: do
//       CHECK:     return;
//       CHECK:   end
//       CHECK: end

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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     local [[v3:.+]] <const> = 0;
//  CHECK-NEXT:     local [[v4:.+]] <const> = 1;
//  CHECK-NEXT:     local [[v5:.+]] <const> = [[v1]] + [[v2]];
//  CHECK-NEXT:     local [[v6:.+]] <const> = [[v5]] - [[v4]];
//  CHECK-NEXT:     local [[v7:.+]] <const> = [[v5]] * [[v6]];
//  CHECK-NEXT:     return [[v7]];
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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]], [[v3:.+]])
//  CHECK-NEXT:   local [[v5:barg.+]] = nil
//  CHECK-NEXT:   local [[v6:barg.+]] = nil
//  CHECK-NEXT:   local [[v4:.+]] <const> = 0;
//  CHECK-NEXT:   [[v5]] = [[v1]];
//  CHECK-NEXT:   [[v6]] = [[v4]];
//  CHECK-NEXT:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     local [[v7:.+]] <const> = _icmp_slt_i64([[v5]], [[v2]])
//  CHECK-NEXT:     if ([[v7]] == 1) or ([[v7]] == true) then
//  CHECK-NEXT:       goto label2;
//  CHECK-NEXT:     else
//  CHECK-NEXT:       goto label3;
//  CHECK-NEXT:     end
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     local [[v8:.+]] <const> = [[v6]] + [[v5]];
//  CHECK-NEXT:     local [[v9:.+]] <const> = [[v5]] + [[v3]];
//  CHECK-NEXT:     [[v5]] = [[v9]];
//  CHECK-NEXT:     [[v6]] = [[v8]];
//  CHECK-NEXT:     goto label1;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label3:: do
//  CHECK-NEXT:     return [[v6]];
//  CHECK-NEXT: end

// -----

func.func @executor_print_op(%arg0: i64, %arg1: f32) attributes {executor.function_metadata = #executor.func_meta<[i64, f32],[], num_output_args = 0>} {
  executor.print (%arg0, %arg1 : i64, f32)
  executor.print "hello %d, %f"( %arg0, %arg1 : i64, f32 )
  executor.print "hello world"()
  return
}

// CHECK-LABEL: executor_print_op
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]])
//  CHECK-NEXT:     print([[v1]], [[v2]]);
//  CHECK-NEXT:     print(string.format("hello %d, %f", [[v1]], [[v2]]));
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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]], [[v3:.+]], [[v4:.+]], [[v5:.+]], [[v6:.+]])
//  CHECK-NEXT:     local [[v7:.+]] <const> = {};
//  CHECK-NEXT:     local [[v8:.+]] <const> = {[[v1]], [[v1]], [[v2]], [[v3]], [[v4]], [[v5]], [[v6]]};
//  CHECK-NEXT:     local [[v9:.+]] <const> = [[v8]][3];
//  CHECK-NEXT:     local [[v10:.+]] <const> = 0;
//  CHECK-NEXT:     local [[v11:.+]] <const> = {};
//  CHECK-NEXT:     for j,x in ipairs([[v8]]) do [[v11]][j] = x end;
//  CHECK-NEXT:     [[v11]][3] = [[v10]];
//  CHECK-NEXT:     return [[v7]], [[v8]], [[v9]], [[v11]];
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
//  CHECK-SAME: ([[v1:.+]], [[v2:.+]], [[v3:.+]], [[v4:.+]], [[v5:.+]])
//  CHECK-NEXT:   local [[v6:.+]] <const> = {[[v1]], [[v2]], [[v3]], [[v4]]};
//  CHECK-NEXT:   local [[v7:.+]] <const> = [[v6]][[[v5]] + 1];
//  CHECK-NEXT:   return [[v6]], [[v7]];
//  CHECK-NEXT: end

// -----

func.func @const_literal() attributes {executor.function_metadata = #executor.func_meta<[],[], num_output_args = 0>} {
  %0 = executor.str_literal "function_1"
  executor.print "%s"(%0 : !executor.str_literal)
  return
}

// CHECK-LABEL: function const_literal
//  CHECK-NEXT:   local [[v1:.+]] <const> = "function_1";
//  CHECK-NEXT:   print(string.format("%s", [[v1]]));
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
  %arg190: i32, %arg191: i32, %arg192: i32) -> i32
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
//  CHECK-SAME:  ([[arg0:arg[0-9]+]],
//  CHECK-NEXT:      local [[v194:l.+]], [[v195:l.+]] <const> = return_two_vals([[arg0]], [[arg0]]);
//  CHECK-NEXT:      local [[ret:l.+]], [[v197:l.+]] <const> = return_two_vals([[v194]], [[v195]]);
//  CHECK-NEXT:      return [[ret]];
//  CHECK-NEXT: end

// -----

func.func @cmpi_ugt(%arg0: i32, %arg1: i32) -> i1
    attributes {executor.function_metadata = #executor.func_meta<[i32, i32],[i1], num_output_args = 0>} {
  %0 = executor.icmp <ugt> %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: function cmpi_ugt ({{.+}}, {{.+}})
//       CHECK:     _icmp_ugt_i32({{.+}}, {{.+}});

// -----

func.func @cmpi_ult(%arg0: i32, %arg1: i32) -> i1
    attributes {executor.function_metadata = #executor.func_meta<[i32, i32],[i1], num_output_args = 0>} {
  %0 = executor.icmp <ult> %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: function cmpi_ult ({{.+}}, {{.+}})
//       CHECK:     _icmp_ult_i32({{.+}}, {{.+}});

// -----

func.func @test_assert(%arg0: i1) {
  executor.assert %arg0, "assertion message"
  return
}

// CHECK-LABEL: function test_assert
//  CHECK-SAME:     ([[v1:.*]])
//       CHECK:   assert(([[v1]] == 1) or ([[v1]] == true), "assertion message");

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
//   CHECK-SAME:  ([[v1:.+]])
//   CHECK-NEXT:   coroutine.yield([[v1]]);
//   CHECK-NEXT:   return [[v1]];
//   CHECK-NEXT: end
// CHECK-LABEL: function coro_await ()
//   CHECK-NEXT:  local [[v2:.+]] <const> = 0;
//   CHECK-NEXT:  local [[v3:.+]] <const> = coroutine.create(coro);
//   CHECK-NEXT:  local [[v4:.+]], [[v5:.+]] <const> = coroutine.resume([[v3]], [[v2]]);
//   CHECK-NEXT:  local [[v6:.+]], [[v7:.+]] <const> = coroutine.resume([[v3]]);
//   CHECK-NEXT:  return [[v7]];
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
//  CHECK-NEXT:   local [[l0:.+]] = nil;
//  CHECK-NEXT:   local [[l1:.+]] <const> = 0;
//  CHECK-NEXT:   goto label1;
//  CHECK-NEXT:   ::label1:: do
//  CHECK-NEXT:     local [[l2:.+]] <const> = coroutine.create(coro);
//  CHECK-NEXT:     local [[l3:.+]] <const>;
//  CHECK-NEXT:     [[l3]], [[l0]] = coroutine.resume([[l2]]);
//  CHECK-NEXT:     goto label2;
//  CHECK-NEXT:   end
//  CHECK-NEXT:   ::label2:: do
//  CHECK-NEXT:     return [[l0]];
//  CHECK-NEXT:   end
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
//   CHECK-SAME:  ([[v1:.+]])
//   CHECK-NEXT:   local [[v2:.+]] = nil;
//   CHECK-NEXT:   local [[v3:.+]] = nil;
//   CHECK-NEXT:   local [[v4:.+]] <const> = 1;
//   CHECK-NEXT:   goto label1;
//   CHECK-NEXT:   ::label1:: do
//   CHECK-NEXT:     [[v2]] = [[v1]] + [[v4]];
//   CHECK-NEXT:     [[v3]] = [[v1]] - [[v4]];
//   CHECK-NEXT:     goto label2;
//   CHECK-NEXT:   end
//   CHECK-NEXT:   ::label2:: do
//   CHECK-NEXT:     local [[v5:.+]] <const> = [[v2]] + [[v3]];
//   CHECK-NEXT:     return [[v5]];
//   CHECK-NEXT:   end
//   CHECK-NEXT: end
