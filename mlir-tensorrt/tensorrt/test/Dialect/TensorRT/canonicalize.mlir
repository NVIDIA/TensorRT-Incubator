// RUN: tensorrt-opt %s -split-input-file -canonicalize | FileCheck %s
// The below command will fail if patterns fail to converge:
// RUN: tensorrt-opt %s -split-input-file -canonicalize=test-convergence

func.func @expand_rank_simplify(%arg0: tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<10x10xf32> to tensor<10x10x1xf32>
  %1 = tensorrt.expand_rank %0 : tensor<10x10x1xf32> to tensor<1x10x10x1xf32>
  return %1 : tensor<1x10x10x1xf32>
}

// CHECK-LABEL: @expand_rank_simplify
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:  %[[result:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<10x10xf32> to tensor<1x10x10x1xf32>
//  CHECK-NEXT:  return %[[result:.+]]

// -----

func.func @collapse_rank_simplify(%arg0: tensor<1x10x10x1xf32>) -> tensor<10x10xf32> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<1x10x10x1xf32> to tensor<10x10x1xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<10x10x1xf32> to tensor<10x10xf32>
  return %1 : tensor<10x10xf32>
}

// CHECK-LABEL: @collapse_rank_simplify
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:  %[[result:.+]] = tensorrt.collapse_rank %[[arg0]] : tensor<1x10x10x1xf32> to tensor<10x10xf32>
//  CHECK-NEXT:  return %[[result:.+]]

// -----

func.func @collapse_expand_compose(%arg0: tensor<10x1xf32>) -> tensor<1x10x1xf32> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<10x1xf32> to tensor<10xf32>
  %1 = tensorrt.expand_rank %0 : tensor<10xf32> to tensor<1x10x1xf32>
  return %1 : tensor<1x10x1xf32>
}

// CHECK-LABEL: @collapse_expand_compose
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:  %[[result:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<10x1xf32> to tensor<1x10x1xf32>
//  CHECK-NEXT:  return %[[result:.+]]


// -----

func.func @expand_collapse_compose(%arg0: tensor<10xf32>) -> tensor<10x1xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<10xf32> to tensor<1x10x1xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x10x1xf32> to tensor<10x1xf32>
  return %1 : tensor<10x1xf32>
}

// CHECK-LABEL: @expand_collapse_compose
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<{{.+}}>)
//  CHECK-NEXT:  %[[result:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<10xf32> to tensor<10x1xf32>
//  CHECK-NEXT:  return %[[result:.+]]

// -----

func.func @fold_expand_of_collapse_rank(%arg0: tensor<1x12x8x1xf16>) -> tensor<1x12x8x1xf16> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<1x12x8x1xf16> to tensor<1x12x8xf16>
  %1 = tensorrt.expand_rank %0 : tensor<1x12x8xf16> to tensor<1x12x8x1xf16>
  return %1 : tensor<1x12x8x1xf16>
}

// CHECK-LABEL: @fold_expand_of_collapse_rank
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x12x8x1xf16>)
//       CHECK:     return %[[arg0]]

// -----

func.func @fold_collapse_of_expand_rank(%arg0: tensor<1x12x8xf16>) -> tensor<1x12x8xf16> {
  %0 = tensorrt.expand_rank %arg0 : tensor<1x12x8xf16> to tensor<1x12x8x1xf16>
  %1 = tensorrt.collapse_rank %0 : tensor<1x12x8x1xf16> to tensor<1x12x8xf16>
  return %1 : tensor<1x12x8xf16>
}

// CHECK-LABEL: @fold_collapse_of_expand_rank
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x12x8xf16>)
//       CHECK:     return %[[arg0]]

// -----

func.func @reshape_reshape(%arg0: tensor<10xf32>) -> tensor<1x2x5xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<10xf32> to tensor<2x5xf32>
  %1 = tensorrt.reshape %0 : tensor<2x5xf32> to tensor<1x2x5xf32>
  return %1 : tensor<1x2x5xf32>
}

// CHECK-LABEL: @reshape_reshape(
//  CHECK-NEXT: tensorrt.reshape %{{.+}} : tensor<10xf32> to tensor<1x2x5xf32>
//  CHECK-NEXT: return %{{.+}}

// -----

func.func @reshape_noop_fold(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<10xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @reshape_noop_fold(
//  CHECK-SAME:  %[[arg0:.+]]: tensor
//  CHECK-NEXT: return %[[arg0]]

// -----

func.func @reshape_noop_dont_fold_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<2xi32>) : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @reshape_noop_dont_fold_dynamic
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]:
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.reshape %[[arg0]] shape(%[[arg1]]
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @reshape_to_expand_rank(%arg0: tensor<10xf32>) -> tensor<1x10xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<10xf32> to tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @reshape_to_expand_rank(
//  CHECK-NEXT: tensorrt.expand_rank
//  CHECK-NEXT: return %{{.+}}

// -----

func.func @reshape_to_collapse_rank(%arg0: tensor<1x10xf32>) -> tensor<10xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<1x10xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @reshape_to_collapse_rank(
//  CHECK-NEXT: tensorrt.collapse_rank
//  CHECK-NEXT: return %{{.+}}

// -----

func.func @reshape_to_expand_rank_negative(%arg0: tensor<1x10xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<3xi32>) : tensor<1x10xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @reshape_to_expand_rank_negative(
//   CHECK-NOT:   tensorrt.expand_rank
//       CHECK:   tensorrt.reshape %{{.+}} shape(%{{.+}}: tensor<3xi32>) : tensor<1x10xf32> to tensor<?x?x?xf32>

// -----

func.func @shape_of_static_tensor(%arg0: tensor<10x20xf32>) -> tensor<2xi32> {
  %0 = tensorrt.shape %arg0 : tensor<10x20xf32> -> tensor<2xi32>
  return %0 :  tensor<2xi32>
}

// CHECK-LABEL: @shape_of_static_tensor
//  CHECK-NEXT:   %[[const:.+]] = tensorrt.constant
//  CHECK-SAME:     dense<[10, 20]> : tensor<2xi32>
//  CHECK-NEXT:   return %[[const]]

// -----

func.func @simplify_dynamic_shuffle_reshape(%arg0: tensor<1x2x3xf32>) -> tensor<6xf32> {
  %0 = tensorrt.constant dense<[6]> : tensor<1xi32>
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2>,
    reshape = array<i64: 6>,
    second_transpose = array<i64: 0>
  } ins(%arg0, %0: tensor<1x2x3xf32>, tensor<1xi32>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

// CHECK-LABEL: @simplify_dynamic_shuffle_reshape
//       CHECK:   tensorrt.shuffle
//  CHECK-SAME:    first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME:    reshape = array<i64: 6>
//  CHECK-SAME:    second_transpose = array<i64: 0>
//  CHECK-sAME:    ins(%{{.+}} : tensor<1x2x3xf32>) -> tensor<6xf32>

// -----

func.func @simplify_dynamic_shuffle_reshape_negative(
    %arg0: tensor<?x2x3xf32>, %arg1: tensor<1xi32>) -> tensor<?xf32> {
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2>,
    second_transpose = array<i64: 0>
  } ins(%arg0, %arg1: tensor<?x2x3xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: @simplify_dynamic_shuffle_reshape_negative
//       CHECK:   tensorrt.shuffle
//  CHECK-SAME:    first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME:    second_transpose = array<i64: 0>
//  CHECK-sAME:    ins(%{{.+}}, %{{.+}} : tensor<?x2x3xf32>, tensor<1xi32>) -> tensor<?xf32>

// -----

func.func @trt_shuffle_simplify_zero_is_placeholder(%arg0: tensor<10x10x1xf32>) -> tensor<10x10xf32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 0, 0>,
        second_transpose = array<i64: 0, 1>,
        zero_is_placeholder = true
    } ins(%arg0 : tensor<10x10x1xf32>) -> tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_shuffle_simplify_zero_is_placeholder
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME: reshape = array<i64: 10, 10>
//  CHECK-SAME: second_transpose = array<i64: 0, 1>,
//  CHECK-SAME: zero_is_placeholder = false
//  CHECK-SAME: ins(%{{.+}} : tensor<10x10x1xf32>) -> tensor<10x10xf32>

// -----

func.func @trt_shuffle_simplify_zero_is_placeholder2(%arg0: tensor<10x10x1xf32>) -> tensor<10x10xf32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 10, 10>,
        second_transpose = array<i64: 0, 1>,
        zero_is_placeholder = true
    } ins(%arg0 : tensor<10x10x1xf32>) -> tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_shuffle_simplify_zero_is_placeholder2
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME: reshape = array<i64: 10, 10>
//  CHECK-SAME: second_transpose = array<i64: 0, 1>,
//  CHECK-SAME: zero_is_placeholder = false
//  CHECK-SAME: ins(%{{.+}} : tensor<10x10x1xf32>) -> tensor<10x10xf32>

// -----

// zero is placeholder: static reshape but dynamic result

func.func @trt_reshape_simplify_dynamic_negative(%arg0: tensor<?x?x1xf32>) -> tensor<?x?xf32> {
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2>,
    reshape = array<i64: 0, 0>,
    second_transpose = array<i64: 0, 1>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<?x?x1xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_reshape_simplify_dynamic_negative
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME: reshape = array<i64: 0, 0>
//  CHECK-SAME: second_transpose = array<i64: 0, 1>
//  CHECK-SAME: ins(%{{.+}} : tensor<?x?x1xf32>) -> tensor<?x?xf32>

// -----

func.func @simplify_sequential_shuffle_static(%arg0: tensor<1x10xf32>) -> tensor<1x10x1x1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    reshape = array<i64: 10, 1>,
    second_transpose = array<i64: 0, 1>,
    zero_is_placeholder = false
  } ins(%arg0 : tensor<1x10xf32>) -> tensor<10x1xf32>
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    reshape = array<i64: 1, 10, 1, 1>,
    second_transpose = array<i64: 0, 1, 2, 3>,
    zero_is_placeholder = false
  } ins(%0 : tensor<10x1xf32>) -> tensor<1x10x1x1xf32>
  return %1 : tensor<1x10x1x1xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle_static(
//  CHECK-SAME:    %[[arg0:.+]]: tensor<1x10xf32>) -> tensor<1x10x1x1xf32> {
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle {
//  CHECK-SAME:      first_transpose = array<i64: 1, 0>
//  CHECK-SAME:      reshape = array<i64: 1, 10, 1, 1>
//  CHECK-SAME:      second_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME:      zero_is_placeholder = false
//  CHECK-SAME:     ins(%[[arg0]] : tensor<1x10xf32>) -> tensor<1x10x1x1xf32>
//       CHECK:   return %[[v0]] : tensor<1x10x1x1xf32>

// -----

// Dynamic reshape can be eliminated in this case.
func.func @simplify_sequential_shuffle(%arg0: tensor<1x10xf32>, %arg1: tensor<2xi32>) -> tensor<1x10x1x1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    second_transpose = array<i64: 0, 1>,
    zero_is_placeholder = false
  } ins(%arg0, %arg1: tensor<1x10xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    reshape = array<i64: 1, 10, 1, 1>,
    second_transpose = array<i64: 0, 1, 2, 3>,
    zero_is_placeholder = false
  } ins(%0 : tensor<?x?xf32>) -> tensor<1x10x1x1xf32>
  return %1 : tensor<1x10x1x1xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle(
//  CHECK-SAME:    %[[arg0:.+]]: tensor<1x10xf32>, %[[arg1:.+]]: tensor<2xi32>
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle {
//  CHECK-SAME:      first_transpose = array<i64: 1, 0>
//  CHECK-SAME:      reshape = array<i64: 1, 10, 1, 1>
//  CHECK-SAME:      second_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME:      zero_is_placeholder = false
//  CHECK-SAME:     ins(%[[arg0]] : tensor<1x10xf32>) -> tensor<1x10x1x1xf32>
//       CHECK:   return %[[v0]] : tensor<1x10x1x1xf32>

// -----

// Preserve dynamic reshape and "zero is placeholder".
func.func @simplify_sequential_shuffle(%arg0: tensor<1x10xf32>, %arg1: tensor<4xi32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    second_transpose = array<i64: 0, 1>,
    reshape = array<i64: 10, 1>,
    zero_is_placeholder = false
  } ins(%arg0: tensor<1x10xf32>) -> tensor<10x1xf32>
  %1 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    second_transpose = array<i64: 0, 1, 2, 3>,
    zero_is_placeholder = true
  } ins(%0, %arg1 : tensor<10x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle(
//  CHECK-SAME:    %[[arg0:.+]]: tensor<1x10xf32>, %[[arg1:.+]]: tensor<4xi32>)
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle
//  CHECK-SAME:     first_transpose = array<i64: 1, 0>
//  CHECK-SAME:     second_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME:     ins(%[[arg0]], %[[arg1]] : tensor<1x10xf32>, tensor<4xi32>)
//       CHECK:   return %[[v0]] : tensor<?x?x?x?xf32>

// -----

// inner scalar case
func.func @simplify_sequential_shuffle(%arg0: tensor<1x1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    second_transpose = array<i64>,
    reshape = array<i64>,
    zero_is_placeholder = false
  } ins(%arg0: tensor<1x1xf32>) -> tensor<f32>
  %1 = tensorrt.shuffle {
    first_transpose = array<i64>,
    second_transpose = array<i64: 0>,
    reshape = array<i64: 1>,
    zero_is_placeholder = false
  } ins(%0: tensor<f32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle(
//  CHECK-SAME:    %[[arg0:.+]]: tensor<1x1xf32>) -> tensor<1xf32> {
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle {
//  CHECK-SAME:      first_transpose = array<i64: 1, 0>
//  CHECK-SAME:      reshape = array<i64: 1>
//  CHECK-SAME:      second_transpose = array<i64: 0>
//  CHECK-SAME:      zero_is_placeholder = false
//  CHECK-SAME:     ins(%[[arg0]] : tensor<1x1xf32>) -> tensor<1xf32>
//       CHECK:   return %[[v0]] : tensor<1xf32>

// -----

func.func @simplify_sequential_shuffle_static_case2(%arg0: tensor<16x197x768xf32>) -> tensor<12x16x64x197xf32> {
    %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 16, 197, 12, 64>,
        second_transpose = array<i64: 0, 2, 1, 3>,
        zero_is_placeholder = false}
        ins(%arg0 : tensor<16x197x768xf32>) -> tensor<16x12x197x64xf32>
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 1, 2, 3, 0>,
        second_transpose = array<i64: 0, 3, 2, 1>,
        zero_is_placeholder = false}
        ins(%0 : tensor<16x12x197x64xf32>) -> tensor<12x16x64x197xf32>
    return %1 : tensor<12x16x64x197xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle_static_case2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x197x768xf32>) -> tensor<12x16x64x197xf32>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>,
//  CHECK-SAME: reshape = array<i64: 16, 197, 12, 64>,
//  CHECK-SAME: second_transpose = array<i64: 2, 0, 3, 1>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%[[arg0]] : tensor<16x197x768xf32>) -> tensor<12x16x64x197xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<12x16x64x197xf32>

// -----

func.func @simplify_sequential_shuffle_dynamic_case2(%arg0: tensor<16x197x768xf32>,
%arg1: tensor<4xi32>) -> tensor<?x?x?x?xf32> {
    %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        second_transpose = array<i64: 0, 1, 2, 3>,
        zero_is_placeholder = false}
        ins(%arg0, %arg1 : tensor<16x197x768xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 2, 1, 3>,
        second_transpose = array<i64: 0, 1, 2, 3>,
        zero_is_placeholder = false}
        ins(%0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle_dynamic_case2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x197x768xf32>, %[[arg1:.+]]: tensor<4xi32>) -> tensor<?x?x?x?xf32>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>,
//  CHECK-SAME: second_transpose = array<i64: 0, 2, 1, 3>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<16x197x768xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<?x?x?x?xf32>

// -----

func.func @simplify_sequential_shuffle_case2_negative(%arg0: tensor<16x197x768xf32>) -> tensor<12x197x1024xf32> {
    %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 16, 197, 12, 64>,
        second_transpose = array<i64: 0, 2, 1, 3>,
        zero_is_placeholder = false}
        ins(%arg0 : tensor<16x197x768xf32>) -> tensor<16x12x197x64xf32>
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 1, 2, 3, 0>,
        reshape = array<i64: 12, 197, 1024>,
        second_transpose = array<i64: 0, 1, 2>,
        zero_is_placeholder = false}
        ins(%0 : tensor<16x12x197x64xf32>) -> tensor<12x197x1024xf32>
    return %1 : tensor<12x197x1024xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle_case2_negative
//       CHECK: tensorrt.shuffle
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-NEXT: return

// -----

func.func @simplify_sequential_shuffle_case2_dynamic_negative(%arg0: tensor<16x197x768xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
    %0 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 16, 197, 12, 64>,
        second_transpose = array<i64: 0, 2, 1, 3>,
        zero_is_placeholder = false}
        ins(%arg0 : tensor<16x197x768xf32>) -> tensor<16x12x197x64xf32>
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 1, 2, 3, 0>,
        second_transpose = array<i64: 0, 1, 2>,
        zero_is_placeholder = false}
        ins(%0, %arg1 : tensor<16x12x197x64xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
    return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @simplify_sequential_shuffle_case2_dynamic_negative
//       CHECK: tensorrt.shuffle
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-NEXT: return

// -----

func.func @fold_shuffle(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    reshape = array<i64: 1, 2>,
    second_transpose = array<i64: 0, 1>,
    zero_is_placeholder = false
  } ins(%arg0 : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: @fold_shuffle
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x2xf32>)
//  CHECK-NEXT:  return %[[arg0]]

func.func @fold_shuffle_with_all_dim_unity_except_one(%arg0: tensor<1x2048x1x1xf16>) -> tensor<1x2048x1x1xf16>{
    %0 = tensorrt.shuffle {
      first_transpose = array<i64: 0, 2, 3, 1>,
      reshape = array<i64: 1, 2048, 1, 1>,
      second_transpose = array<i64: 0, 1, 2, 3>,
      zero_is_placeholder = false}
      ins(%arg0 : tensor<1x2048x1x1xf16>) -> tensor<1x2048x1x1xf16>
    return %0: tensor<1x2048x1x1xf16>
}

// CHECK-LABEL: @fold_shuffle_with_all_dim_unity_except_one
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2048x1x1xf16>)
//  CHECK-NEXT: return %[[arg0]]

// -----

func.func @fold_shuffle_inverse(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    reshape = array<i64: 2, 1>,
    second_transpose = array<i64: 1, 0>,
    zero_is_placeholder = false
  } ins(%arg0 : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: @fold_shuffle_inverse
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x2xf32>)
//  CHECK-NEXT:  return %[[arg0]]

// -----

func.func @shuffle_dont_fold(%arg0: tensor<1x2xf32>) -> tensor<2x1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    reshape = array<i64: 1, 2>,
    second_transpose = array<i64: 1, 0>,
    zero_is_placeholder = false
  } ins(%arg0 : tensor<1x2xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: @shuffle_dont_fold(
//  CHECK-SAME:  %[[arg0:.+]]: tensor<1x2xf32>)
//       CHECK:  %[[v0:.+]] = tensorrt.shuffle
//  CHECK-SAME:    first_transpose = array<i64: 1, 0>,
//  CHECK-SAME:    reshape = array<i64: 1, 2>,
//  CHECK-SAME:    second_transpose = array<i64: 1, 0>
//  CHECK-SAME:    zero_is_placeholder = false
//  CHECK-SAME:    ins(%[[arg0]] : tensor<1x2xf32>) -> tensor<2x1xf32>
//       CHECK:  return %[[v0]] : tensor<2x1xf32>

// -----

func.func @shuffle_dont_fold(%arg0: tensor<1x2xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 1, 0>,
    second_transpose = array<i64: 1, 0>,
    zero_is_placeholder = false
  } ins(%arg0, %arg1 : tensor<1x2xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @shuffle_dont_fold(
//  CHECK-SAME:    %[[arg0:.+]]: tensor<1x2xf32>, %[[arg1:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:    %[[v0:.+]] = tensorrt.shuffle
//  CHECK-SAME:     first_transpose = array<i64: 1, 0>
//  CHECK-SAME:     second_transpose = array<i64: 1, 0>
//  CHECK-SAME:     zero_is_placeholder = false}
//  CHECK-SAME:     ins(%[[arg0]], %[[arg1]] : tensor<1x2xf32>, tensor<2xi32>) -> tensor<?x?xf32>
//       CHECK:    return %[[v0]] : tensor<?x?xf32>

// -----

func.func @broadcast_fold(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x10xf32> to tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @broadcast_fold
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x10xf32>)
//  CHECK-NEXT:  return %[[arg0]]

// -----

func.func @broadcast_no_fold(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1, 0> : tensor<2x2xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @broadcast_no_fold
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<2x2xf32>)
//  CHECK-NEXT:  %[[v0:.+]] = tensorrt.broadcast %[[arg0]]
//  CHECK-SAME:   broadcast_dims<1, 0>
//  CHECK-SAME:    : tensor<2x2xf32> to tensor<2x2xf32>
//  CHECK-NEXT:  return %[[v0]]

// -----

func.func @broadcast_no_fold(%arg0: tensor<?x2xf32>, %shape: tensor<2xi32>) -> tensor<?x2xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%shape: tensor<2xi32>) : tensor<?x2xf32> to tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: @broadcast_no_fold
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x2xf32>, %[[arg1:.+]]: tensor<2xi32>
//  CHECK-NEXT:  %[[v0:.+]] = tensorrt.broadcast %[[arg0]]
//  CHECK-SAME:   broadcast_dims<0, 1>
//  CHECK-SAME:   shape(%[[arg1]] : tensor<2xi32>)
//  CHECK-SAME:    : tensor<?x2xf32> to tensor<?x2xf32>
//  CHECK-NEXT:  return %[[v0]]


// -----

func.func @transpose_fold(%arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tensorrt.transpose {permutation = affine_map<(d0)->(d0)>} %arg1 : tensor<1xf32> to tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @transpose_fold
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32>) -> tensor<1xf32> {
//       CHECK:     return %[[arg0]] : tensor<1xf32>

// -----

func.func @transpose_fold_scalar(%arg1: tensor<f32>) -> tensor<f32> {
  %0 = tensorrt.transpose {permutation = affine_map<()->()>} %arg1 : tensor<f32> to tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @transpose_fold_scalar
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>) -> tensor<f32> {
//       CHECK:     return %[[arg0]] : tensor<f32>

// -----

func.func @transpose_sequential(%arg1: tensor<10x2x1xf32>) -> tensor<1x10x2xf32> {
  %0 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d0, d2, d1)>
  } %arg1 : tensor<10x2x1xf32> to tensor<10x1x2xf32>
  %1 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>
  } %0 : tensor<10x1x2xf32> to tensor<1x10x2xf32>
  return %1 : tensor<1x10x2xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// CHECK-LABEL: @transpose_sequential
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x2x1xf32>) -> tensor<1x10x2xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<10x2x1xf32> to tensor<1x10x2xf32>
//       CHECK:     return %[[v0]] : tensor<1x10x2xf32>

// -----

func.func @transpose_sequential_identity(%arg1: tensor<10x2x1xf32>) -> tensor<10x2x1xf32> {
  %0 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d0, d2, d1)>
  } %arg1 : tensor<10x2x1xf32> to tensor<10x1x2xf32>
  %1 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d0, d2, d1)>
  } %0 : tensor<10x1x2xf32> to tensor<10x2x1xf32>
  return %1 : tensor<10x2x1xf32>
}

// CHECK-LABEL: @transpose_sequential_identity
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x2x1xf32>) -> tensor<10x2x1xf32> {
//       CHECK:     return %[[arg0]] : tensor<10x2x1xf32>

// -----

func.func @expand_rank_const() -> tensor<5x2xf32> {
  %0 = tensorrt.constant dense<0.0> : tensor<10xf32>
  %1 = tensorrt.expand_rank %0 : tensor<10xf32> to tensor<1x10x1xf32>
  %2 = tensorrt.collapse_rank %1 : tensor<1x10x1xf32> to tensor<10x1xf32>
  %3 = tensorrt.reshape %2 : tensor<10x1xf32> to tensor<5x2xf32>
  return %3 : tensor<5x2xf32>
}

// CHECK-LABEL: @expand_rank_const()
//       CHECK:   %[[cst_f32:.+]] = tensorrt.constant dense{{.*}} : tensor<5x2xf32>
//       CHECK:   return %[[cst_f32]]

// -----

func.func @expand_rank_const_elided() -> tensor<1x10x1xf32> {
  %0 = tensorrt.constant dense_resource<__elided__> : tensor<10xf32>
  %1 = tensorrt.expand_rank %0 : tensor<10xf32> to tensor<1x10x1xf32>
  return %1 : tensor<1x10x1xf32>
}

// CHECK-LABEL: @expand_rank_const_elided()
//       CHECK:   %[[cst_f32:.+]] = tensorrt.constant dense{{.*}} : tensor<1x10x1xf32>
//       CHECK:   return %[[cst_f32]]

// -----

func.func @identity_fold(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32>{
    %0 = tensorrt.identity %arg0 : tensor<2x3xf32> to tensor<2x3xf32>
    %1 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kABS>
    } %0 : tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: @identity_fold
//   CHECK-NOT: tensorrt.identity
//       CHECK: %[[r:.+]] = tensorrt.unary {
//       CHECK: unaryOperation = #tensorrt.unary_operation<kABS>
//       CHECK: } %[[i:.+]] : tensor<2x3xf32>

// -----

func.func @unary_neg_fold() -> tensor<10xf32> {
  %0 = tensorrt.constant dense<1.0> : tensor<10xf32>
  %1 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %0 : tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @unary_neg_fold(
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<-1.0{{.*}}> : tensor<10xf32>
//       CHECK:     return %[[cst_f32]] : tensor<10xf32>

// -----

func.func @conv_bias_sub_to_add(%arg0: tensor<16x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<16x64x112x112xf32> {
  %0 = tensorrt.convolution {dilation = array<i64: 1, 1>, post_padding = array<i64: 3, 3>,
        pre_padding = array<i64: 3, 3>, stride = array<i64: 2, 2>}
        in(%arg0 : tensor<16x3x224x224xf32>)
        kernel(%arg1 : tensor<64x3x7x7xf32>) -> tensor<16x64x112x112xf32>
  %cst_f32_1 = tensorrt.constant dense<1.0> : tensor<1x64x1x1xf32>
  %1 = tensorrt.element_wise <kSUB>(%0, %cst_f32_1 : tensor<16x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<16x64x112x112xf32>
  return %1 : tensor<16x64x112x112xf32>
}

// CHECK-LABEL: @conv_bias_sub_to_add
//       CHECK:     %[[v0:.+]] = tensorrt.convolution
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[cst_f32]] : tensor<16x64x112x112xf32>, tensor<1x1x1x1xf32>) -> tensor<16x64x112x112xf32>
//       CHECK:     return %[[v1]] : tensor<16x64x112x112xf32>

// -----

func.func @sum_neg_to_sub(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %arg1 : tensor<1024xf32>
  %1 = tensorrt.element_wise <kSUM>(%arg0, %0 : tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// CHECK-LABEL: @sum_neg_to_sub
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>)
//       CHECK:  tensorrt.element_wise <kSUB>(%[[arg0]], %[[arg1]] :

// -----

func.func @max_to_relu(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tensorrt.constant dense<0.0> : tensor<1xf32>
  %0 = tensorrt.element_wise <kMAX>(%arg0, %c0 : tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @max_to_relu
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %[[arg0]]
//       CHECK:     return %[[v0]]

// -----

func.func @max_to_relu2(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tensorrt.constant dense<0.0> : tensor<10xf32>
  %0 = tensorrt.element_wise <kMAX>(%arg0, %c0 : tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @max_to_relu2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %[[arg0]]
//       CHECK:     return %[[v0]]

// -----

func.func @max_to_relu_negative(%arg0: tensor<1xf32>) -> tensor<10xf32> {
  %c0 = tensorrt.constant dense<0.0> : tensor<10xf32>
  %0 = tensorrt.element_wise <kMAX>(%arg0, %c0 : tensor<1xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @max_to_relu_negative
//       CHECK:   %[[v0:.+]] = tensorrt.element_wise <kMAX>
//       CHECK:   return %[[v0]]

// -----

func.func @min_max_clip(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %cn1 = tensorrt.constant dense<-1.0> : tensor<1xf32>
  %c1 = tensorrt.constant dense<1.0> : tensor<1xf32>
  %0 = tensorrt.element_wise <kMAX>(%arg0, %cn1 : tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  %1 = tensorrt.element_wise <kMIN>(%0, %c1 : tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @min_max_clip(
//       CHECK:     %[[v0:.+]] = tensorrt.activation
//  CHECK-SAME:       activationType = #tensorrt.activation_type<kCLIP>
//  CHECK-SAME:       alpha = -1.{{0+}}e+{{0+}} : f32
//  CHECK-SAME:         beta = 1.{{0+}}e+{{0+}} : f32
//       CHECK:     return %[[v0]]

// -----

func.func @min_max_clip_negative(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %cn1 = tensorrt.constant dense<-1.0> : tensor<1xf32>
  %c1 = tensorrt.constant dense<1.0> : tensor<1xf32>
  // Don't rewrite to clip here since  if f(x) = min(max(x, 1), 0),
  // then f(0) = 0 != max(min(0, 0), 1) = 1
  %0 = tensorrt.element_wise <kMAX>(%arg0, %c1 : tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  %1 = tensorrt.element_wise <kMIN>(%0, %cn1 : tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @min_max_clip_negative(
//   CHECK-NOT:   tensorrt.activation

// -----

func.func @min_max_clip_f16(%arg0: tensor<512x512xf16>) -> tensor<512x512xf16> {
  %cst_f16 = tensorrt.constant dense<4.200000e+01> : tensor<1x1xf16>
  %cst_f16_0 = tensorrt.constant dense<-4.200000e+01> : tensor<1x1xf16>
  %0 = tensorrt.element_wise <kMAX>(%arg0, %cst_f16_0 : tensor<512x512xf16>, tensor<1x1xf16>) -> tensor<512x512xf16>
  %1 = tensorrt.element_wise <kMIN>(%0, %cst_f16 : tensor<512x512xf16>, tensor<1x1xf16>) -> tensor<512x512xf16>
  return %1 : tensor<512x512xf16>
}

// CHECK-LABEL: @min_max_clip_f16
//       CHECK:   tensorrt.activation
//  CHECK-SAME:     #tensorrt.activation_type<kCLIP>
//  CHECK-SAME:     alpha = -4.200000e+01
//  CHECK-SAME:     beta = 4.200000e+01

// -----

func.func @conv_weights_to_static(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x64x128x128xf32> {
  %kernel = tensorrt.constant dense<0.1> : tensor<64x32x3x3xf32>
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    biasStatic = dense<0.1>:tensor<64xf32>
  } in (%arg0: tensor<1x32x128x128xf32>)
    kernel (%kernel: tensor<64x32x3x3xf32>) -> tensor<1x64x128x128xf32>
  return %0 : tensor<1x64x128x128xf32>
}

// CHECK-LABEL: @conv_weights_to_static
//   CHECK-NOT:   tensorrt.constant
//       CHECK:   %[[v0:.+]] = tensorrt.convolution
//  CHECK-SAME:     kernelStatic = dense<1.000000e-01> : tensor<64x32x3x3xf32>
//       CHECK:   return %[[v0]] : tensor<1x64x128x128xf32>

// -----

func.func @conv_mixed_precision_rewrite(%arg0: tensor<16x128x28x28xf16>) -> tensor<16x512x28x28xf16> {
  %cst = tensorrt.constant dense<0.0> : tensor<512x128x1x1xf16>
  %cst0 = tensorrt.constant dense<0.1> : tensor<1x512x1x1xf32>
  %cst1 = tensorrt.constant dense<0.2> : tensor<1x512x1x1xf32>
  %cst2 = tensorrt.constant dense<0.3> : tensor<1x512x1x1xf32>
  %2 = tensorrt.convolution {dilation = array<i64: 1, 1>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%arg0 : tensor<16x128x28x28xf16>) kernel(%cst : tensor<512x128x1x1xf16>) -> tensor<16x512x28x28xf16>
  %3 = tensorrt.identity %2 : tensor<16x512x28x28xf16> to tensor<16x512x28x28xf32>
  %4 = tensorrt.element_wise <kSUM>(%3, %cst0 : tensor<16x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<16x512x28x28xf32>
  %5 = tensorrt.element_wise <kPROD>(%4, %cst1 : tensor<16x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<16x512x28x28xf32>
  %6 = tensorrt.element_wise <kSUM>(%5, %cst2 : tensor<16x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<16x512x28x28xf32>
  %7 = tensorrt.identity %6 : tensor<16x512x28x28xf32> to tensor<16x512x28x28xf16>
  return %7 : tensor<16x512x28x28xf16>
}

// CHECK-LABEL: @conv_mixed_precision_rewrite
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x128x28x28xf16>)
//       CHECK:     %[[cst_f16:.+]] = tensorrt.constant dense<{{.+}}> : tensor<1x1x1x1xf16>
//       CHECK:     %[[cst_f16_0:.+]] = tensorrt.constant dense<{{.+}}> : tensor<1x1x1x1xf16>
//       CHECK:     %[[cst_f16_1:.+]] = tensorrt.constant dense<{{.+}}> : tensor<1x1x1x1xf16>
//       CHECK:     %[[v0:.+]] = tensorrt.convolution {{.+}} in(%[[arg0]] : tensor<16x128x28x28xf16>) -> tensor<16x512x28x28xf16>
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[cst_f16_1]] : tensor<16x512x28x28xf16>, tensor<1x1x1x1xf16>) -> tensor<16x512x28x28xf16>
//       CHECK:     %[[v2:.+]] = tensorrt.element_wise <kPROD>(%[[v1]], %[[cst_f16_0]] : tensor<16x512x28x28xf16>, tensor<1x1x1x1xf16>) -> tensor<16x512x28x28xf16>
//       CHECK:     %[[v3:.+]] = tensorrt.element_wise <kSUM>(%[[v2]], %[[cst_f16]] : tensor<16x512x28x28xf16>, tensor<1x1x1x1xf16>) -> tensor<16x512x28x28xf16>
//       CHECK:     return %[[v3]] : tensor<16x512x28x28xf16>

// -----

func.func @reduce_expand_rank_rewrite(%arg0: tensor<8x12x197x197xf32>) -> tensor<8x12x197x1xf32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 3>} : tensor<8x12x197x197xf32> -> tensor<8x12x197xf32>
  %1 = tensorrt.expand_rank %0 : tensor<8x12x197xf32> to tensor<8x12x197x1xf32>
  return %1 : tensor<8x12x197x1xf32>
}

// CHECK-LABEL: @reduce_expand_rank_rewrite
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8x12x197x197xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.reduce <kSUM> %[[arg0]]
//  CHECK-SAME:        keepDimensions = true
//       CHECK:     return %[[v0]] :

// -----

func.func @reduce_expand_rank_negative(%arg0: tensor<8x12x197x197xf32>) -> tensor<8x12x1x197xf32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 3>} : tensor<8x12x197x197xf32> -> tensor<8x12x197xf32>
  %1 = tensorrt.expand_rank %0 : tensor<8x12x197xf32> to tensor<8x12x1x197xf32>
  return %1 : tensor<8x12x1x197xf32>
}

// CHECK-LABEL: @reduce_expand_rank_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8x12x197x197xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.reduce <kSUM> %[[arg0]]
//       CHECK:     %[[v1:.+]] = tensorrt.expand_rank %[[v0]] : tensor<8x12x197xf32> to tensor<8x12x1x197xf32>
//       CHECK:     return %[[v1]] :

// -----

func.func @reduce_sum_div_to_mean(%arg0: tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 3>} : tensor<8x12x197x197xf32> -> tensor<8x12x197xf32>
  %cst = tensorrt.constant dense<197.0> : tensor<1x1x1xf32>
  %1 = tensorrt.element_wise <kDIV>(%0, %cst : tensor<8x12x197xf32>, tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
  return %1 : tensor<8x12x197xf32>
}

// CHECK-LABEL: @reduce_sum_div_to_mean(
//  CHECK-SAME: %[[arg0:.+]]: tensor<8x12x197x197xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.reduce <kAVG> %[[arg0]]
//       CHECK:     return %[[v0]] : tensor<8x12x197xf32>

// -----

func.func @reduce_sum_div_to_mean_negative(%arg0: tensor<8x12x197x197xf32>) -> tensor<8x12x197xf32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 3>} : tensor<8x12x197x197xf32> -> tensor<8x12x197xf32>
  %cst = tensorrt.constant dense<197.1> : tensor<1x1x1xf32>
  %1 = tensorrt.element_wise <kDIV>(%0, %cst : tensor<8x12x197xf32>, tensor<1x1x1xf32>) -> tensor<8x12x197xf32>
  return %1 : tensor<8x12x197xf32>
}

// CHECK-LABEL: @reduce_sum_div_to_mean_negative
//   CHECK-NOT:     tensorrt.reduce <kAVG>

// -----

func.func @softmax_raiser(%arg0: tensor<4x12x197x197xf32>) -> tensor<4x12x197x197xf32> {
  %1 = tensorrt.reduce <kMAX> %arg0 {keepDimensions = true, reduceAxes = array<i64: 3>} : tensor<4x12x197x197xf32> -> tensor<4x12x197x1xf32>
  %2 = tensorrt.element_wise <kSUB>(%arg0, %1 : tensor<4x12x197x197xf32>, tensor<4x12x197x1xf32>) -> tensor<4x12x197x197xf32>
  %3 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kEXP>} %2 : tensor<4x12x197x197xf32>
  %4 = tensorrt.reduce <kSUM> %3 {keepDimensions = true, reduceAxes = array<i64: 3>} : tensor<4x12x197x197xf32> -> tensor<4x12x197x1xf32>
  %5 = tensorrt.element_wise <kDIV>(%3, %4 : tensor<4x12x197x197xf32>, tensor<4x12x197x1xf32>) -> tensor<4x12x197x197xf32>
  return %5 : tensor<4x12x197x197xf32>
}

// CHECK-LABEL: @softmax_raiser
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x12x197x197xf32>)
//       CHECK:     %[[v0:.+]] = tensorrt.softmax {axis = 3 : i64} %[[arg0]] : tensor<4x12x197x197xf32>
//       CHECK:     return %[[v0]] : tensor<4x12x197x197xf32>

// -----

// When the reduction dimensions don't match up on the two tensorrt.reduce ops below,
// the operation does not implement a softmax.

func.func @softmax_raiser_negative(%arg0: tensor<4x12x1x197xf32>) -> tensor<4x12x1x197xf32> {
  %1 = tensorrt.reduce <kMAX> %arg0 {keepDimensions = true, reduceAxes = array<i64: 3>} : tensor<4x12x1x197xf32> -> tensor<4x12x1x1xf32>
  %2 = tensorrt.element_wise <kSUB>(%arg0, %1 : tensor<4x12x1x197xf32>, tensor<4x12x1x1xf32>) -> tensor<4x12x1x197xf32>
  %3 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kEXP>} %2 : tensor<4x12x1x197xf32>
  %4 = tensorrt.reduce <kSUM> %3 {keepDimensions = true, reduceAxes = array<i64: 2>} : tensor<4x12x1x197xf32> -> tensor<4x12x1x197xf32>
  %5 = tensorrt.element_wise <kDIV>(%3, %4 : tensor<4x12x1x197xf32>, tensor<4x12x1x197xf32>) -> tensor<4x12x1x197xf32>
  return %5 : tensor<4x12x1x197xf32>
}

// CHECK-LABEL: @softmax_raiser_negative
//   CHECK-NOT:   tensorrt.softmax

// -----

func.func @softmax_raiser_f16(%arg0: tensor<4x12x197x197xf16>) -> tensor<4x12x197x197xf16> {
  %1 = tensorrt.reduce <kMAX> %arg0 {keepDimensions = true, reduceAxes = array<i64: 3>} : tensor<4x12x197x197xf16> -> tensor<4x12x197x1xf16>
  %2 = tensorrt.element_wise <kSUB>(%arg0, %1 : tensor<4x12x197x197xf16>, tensor<4x12x197x1xf16>) -> tensor<4x12x197x197xf16>
  %3 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kEXP>} %2 : tensor<4x12x197x197xf16>
  %4 = tensorrt.identity %3 : tensor<4x12x197x197xf16> to tensor<4x12x197x197xf32>
  %5 = tensorrt.reduce <kSUM> %4 {keepDimensions = true, reduceAxes = array<i64: 3>} : tensor<4x12x197x197xf32> -> tensor<4x12x197x1xf32>
  %6 = tensorrt.identity %5 : tensor<4x12x197x1xf32> to tensor<4x12x197x1xf16>
  %7 = tensorrt.element_wise <kDIV>(%3, %6 : tensor<4x12x197x197xf16>, tensor<4x12x197x1xf16>) -> tensor<4x12x197x197xf16>
  return %7 : tensor<4x12x197x197xf16>
}

// CHECK-LABEL: @softmax_raiser_f16
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x12x197x197xf16>)
//       CHECK:     %[[v0:.+]] = tensorrt.softmax {axis = 3 : i64} %[[arg0]] : tensor<4x12x197x197xf16>
//       CHECK:     return %[[v0]] : tensor<4x12x197x197xf16>

// -----

func.func @concat_folder_drop_zero_extent() -> tensor<2x4x8xi32> {
  %0 = tensorrt.constant dense<0> : tensor<2x0x8xi32>
  %1 = tensorrt.constant dense<0> : tensor<2x4x8xi32>
  %2 = tensorrt.concatenation {axis = 1 : i32} ins( %0, %1 : tensor<2x0x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
  return %2 : tensor<2x4x8xi32>
}

// CHECK-LABEL: @concat_folder_drop_zero_extent
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<2x4x8xi32>
//       CHECK:     return %[[cst_i32]]

// -----

func.func @concat_folder_drop_zero_extent2(%arg0: tensor<2x?x8xi32>, %arg1: tensor<2x?x8xi32>) -> tensor<2x4x8xi32> {
  %2 = tensorrt.concatenation {axis = 1 : i32} ins( %arg0, %arg1 : tensor<2x?x8xi32>, tensor<2x?x8xi32>) -> tensor<2x4x8xi32>
  return %2 : tensor<2x4x8xi32>
}

// verify that dynamic tensors are not dropped.

// CHECK-LABEL: @concat_folder_drop_zero_extent2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x?x8xi32>, %[[arg1:.+]]: tensor<2x?x8xi32>)
//       CHECK:     %[[v0:.+]] = tensorrt.concatenation {axis = 1 : i32} ins(%[[arg0]], %[[arg1]] : tensor<2x?x8xi32>, tensor<2x?x8xi32>)
//       CHECK:     return %[[v0]]

// -----

func.func @concat_const_folder() -> tensor<4xi32> {
  %0 = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
  %1 = tensorrt.constant dense<[3, 4]> : tensor<2xi32>
  %2 = tensorrt.concatenation {axis = 0 : i32} ins( %0, %1 : tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: @concat_const_folder
//       CHECK:   %[[cst_i32:.+]] = tensorrt.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
//       CHECK:   return %[[cst_i32]] : tensor<4xi32>

// -----

func.func @concat_folder_unsupported() -> (tensor<4xi32>, tensor<4x2xi32>, tensor<4xf32>) {
  %0 = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
  %1 = tensorrt.constant dense_resource<__elided__> : tensor<2xi32>
  %2 = tensorrt.concatenation {axis = 0 : i32} ins( %0, %1 : tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  %3 = tensorrt.constant dense<0> : tensor<2x2xi32>
  %4 = tensorrt.concatenation {axis = 0 : i32} ins( %3, %3 : tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  %5 = tensorrt.constant dense<0.0> : tensor<2xf32>
  %6 = tensorrt.concatenation {axis = 0 : i32} ins( %5, %5 : tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  return %2, %4, %6 : tensor<4xi32>, tensor<4x2xi32>, tensor<4xf32>
}

//   CHECK-LABEL: @concat_folder_unsupported
// CHECK-COUNT-3:   tensorrt.concatenation

// -----

func.func @expand_rank_folder1() -> tensor<1x4xf32> {
  %0 = tensorrt.constant dense<0.> : tensor<4xf32>
  %1 = tensorrt.expand_rank %0 : tensor<4xf32> to tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL: @expand_rank_folder1
//       CHECK:   %[[cst_f32:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<1x4xf32>
//       CHECK:   return %[[cst_f32]]

// -----

func.func @expand_rank_folder2() -> tensor<1x4xf32> {
  %0 = tensorrt.constant dense_resource<__elided__> : tensor<4xf32>
  %1 = tensorrt.expand_rank %0 : tensor<4xf32> to tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL: @expand_rank_folder2
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<1x4xf32>
//       CHECK:     return %[[cst_f32]]

// -----

func.func @ewise_sum_const_fold() -> tensor<2xi32> {
  %0 = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
  %1 = tensorrt.constant dense<[3, 4]> : tensor<2xi32>
  %2 = tensorrt.element_wise <kSUM>(%0, %1 : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}

// CHECK-LABEL: @ewise_sum_const_fold
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<[4, 6]>
//       CHECK:     return %[[cst_i32]]


// -----

func.func @ewise_sub_const_fold() -> tensor<2xi32> {
  %0 = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
  %1 = tensorrt.constant dense<[3, 4]> : tensor<2xi32>
  %2 = tensorrt.element_wise <kSUB>(%0, %1 : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}

// CHECK-LABEL: @ewise_sub_const_fold
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<-2> : tensor<2xi32>
//       CHECK:     return %[[cst_i32]]

// -----

func.func @ewise_const_fold_unsupported() -> tensor<2xf32> {
  %0 = tensorrt.constant dense<1.0> : tensor<2xf32>
  %1 = tensorrt.constant dense<2.0> : tensor<2xf32>
  %2 = tensorrt.element_wise <kSUM>(%0, %1 : tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: @ewise_const_fold_unsupported
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise
//       CHECK:     return %[[v0]]

// -----

func.func @push_transpose_out(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32> {
    %1 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} %arg1 : tensor<1x197x12x64xf32> to tensor<1x12x197x64xf32>
    %2 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kTRANSPOSE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%1, %arg0 : tensor<1x12x197x64xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x64x197xf32>
    return %2: tensor<1x12x64x197xf32>
}

//       CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @push_transpose_out
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x12x197x197xf32>, %[[arg1:.+]]: tensor<1x197x12x64xf32>)
//       CHECK: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$MAP0]]} %[[arg1]] : tensor<1x197x12x64xf32> to tensor<1x12x64x197xf32>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[v0]], %[[arg0]] : {{.*}}) -> tensor<1x12x64x197xf32>
//  CHECK-NEXT: return %[[v1]] : tensor<1x12x64x197xf32>

// -----

func.func @pull_transpose_in(%arg0: tensor<1x100x200xf16>, %arg1: tensor<1x600x200xf16>) -> tensor<1x100x600xf16> {
    %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} %arg1 : tensor<1x600x200xf16> to tensor<1x200x600xf16>
    %2 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%arg0, %0 : tensor<1x100x200xf16>, tensor<1x200x600xf16>) -> tensor<1x100x600xf16>
    return %2 : tensor<1x100x600xf16>
}

// CHECK-LABEL: @pull_transpose_in
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x100x200xf16>, %[[arg1:.+]]: tensor<1x600x200xf16>)
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%arg0, %arg1 : {{.*}}) -> tensor<1x100x600xf16>
//  CHECK-NEXT: return %[[v0]] : tensor<1x100x600xf16>

// -----

func.func @push_transpose_out_explicit(%arg0: tensor<1x12x197x64xf32>, %arg1: tensor<1x12x197x64xf32>, %arg2: tensor<1x12x197x197xf32>) -> (tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>) {
    %0 = tensorrt.element_wise <kDIV>(%arg0, %arg1 : tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = tensorrt.matrix_multiply
    {op0 = #tensorrt.matrix_operation<kTRANSPOSE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
    ins(%0, %arg2 : tensor<1x12x197x64xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x64x197xf32>
    %2 = tensorrt.element_wise <kSUM>(%0, %arg0 : tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    return %2, %1: tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>
}

//       CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @push_transpose_out_explicit
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x12x197x64xf32>, %[[arg1:.+]]: tensor<1x12x197x64xf32>, %[[arg2:.+]]: tensor<1x12x197x197xf32>)
//       CHECK: %[[v0:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$MAP0]]} %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[v1]], %[[arg2]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[arg0]] : {{.*}})
//  CHECK-NEXT: return %[[v3]], %[[v2]] : tensor<1x12x197x64xf32>, tensor<1x12x64x197xf32>

// -----

func.func @remove_prod_div_pair(%arg0: tensor<2048x1xf16>) -> tensor<2048x1xf16>{
    %cst_f16 = tensorrt.constant dense<3.0> : tensor<1x1xf16>
    %cst_f16_2 = tensorrt.constant dense<3.0> : tensor<2048x1xf16>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f16_2 : tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %3 = tensorrt.element_wise <kSUB>(%1, %2: tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    return %3 : tensor<2048x1xf16>
}

// CHECK-LABEL: @remove_prod_div_pair
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf16>)
//  CHECK-NEXT: %[[cst_f16:.+]] = tensorrt.constant dense<3.000000e+00> : tensor<1x1xf16>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_f16]] : {{.*}}) -> tensor<2048x1xf16>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kSUB>(%[[arg0]], %[[v0]] : {{.*}}) -> tensor<2048x1xf16>
//  CHECK-NEXT: return %[[v1]] : tensor<2048x1xf16>

// -----

func.func @remove_prod_div_pair_prod_use(%arg0: tensor<2048x1xf16>) -> tensor<2048x1xf16>{
    %cst_f16 = tensorrt.constant dense<3.0> : tensor<1x1xf16>
    %cst_f16_2 = tensorrt.constant dense<3.0> : tensor<2048x1xf16>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f16_2 : tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %3 = tensorrt.element_wise <kSUB>(%0, %2: tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    return %3 : tensor<2048x1xf16>
}

// CHECK-LABEL: @remove_prod_div_pair_prod_use
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf16>)
//  CHECK-NEXT: %[[cst_f16:.+]] = tensorrt.constant dense<3.000000e+00> : tensor<1x1xf16>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[cst_f16]] : {{.*}}) -> tensor<2048x1xf16>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_f16]] : {{.*}}) -> tensor<2048x1xf16>
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUB>(%[[v0]], %[[v1]] : {{.*}}) -> tensor<2048x1xf16>
//  CHECK-NEXT: return %[[v2]] : tensor<2048x1xf16>

// -----

func.func @remove_prod_div_pair_negative(%arg0: tensor<2048x1xf16>) -> tensor<2048x1xf16>{
    %cst_f16 = tensorrt.constant dense<3.456> : tensor<1x1xf16>
    %cst_f16_2 = tensorrt.constant dense<3.45689> : tensor<2048x1xf16>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f16_2 : tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f16 : tensor<2048x1xf16>, tensor<1x1xf16>) -> tensor<2048x1xf16>
    %3 = tensorrt.element_wise <kSUB>(%1, %2: tensor<2048x1xf16>, tensor<2048x1xf16>) -> tensor<2048x1xf16>
    return %3 : tensor<2048x1xf16>
}

// CHECK-LABEL: @remove_prod_div_pair_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf16>)
//  CHECK-NEXT: %[[cst_f16:.+]] = tensorrt.constant dense<3.457030e+00>
//  CHECK-NEXT: %[[cst_f16_0:.+]] = tensorrt.constant dense<3.455080e+00>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[cst_f16_0]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[cst_f16]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[cst_f16_0]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUB>(%[[v1]], %[[v2]] : {{.*}})
//  CHECK-NEXT: return %[[v3]] : tensor<2048x1xf16>


// -----

func.func @remove_prod_div_pair_negative_pos_inf(%arg0: tensor<2048x1xf32>) -> tensor<2048x1xf32>{
    %cst_f32 = tensorrt.constant dense<0x7f800000> : tensor<1x1xf32>
    %cst_f32_2 = tensorrt.constant dense<0x7f800000> : tensor<2048x1xf32>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f32_2 : tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %3 = tensorrt.element_wise <kSUB>(%1, %2: tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    return %3 : tensor<2048x1xf32>
}

// CHECK-LABEL: @remove_prod_div_pair_negative_pos_inf
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf32>)
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant dense<0x7F800000>
//  CHECK-NEXT: %[[cst_f32_0:.+]] = tensorrt.constant dense<0x7F800000>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[cst_f32_0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUB>(%[[v1]], %[[v2]] : {{.*}})
//  CHECK-NEXT: return %[[v3]] : tensor<2048x1xf32>

// -----

func.func @remove_prod_div_pair_negative_neg_inf(%arg0: tensor<2048x1xf32>) -> tensor<2048x1xf32>{
    %cst_f32 = tensorrt.constant dense<0xff800000> : tensor<1x1xf32>
    %cst_f32_2 = tensorrt.constant dense<0xff800000> : tensor<2048x1xf32>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f32_2 : tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %3 = tensorrt.element_wise <kSUB>(%1, %2: tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    return %3 : tensor<2048x1xf32>
}

// CHECK-LABEL: @remove_prod_div_pair_negative_neg_inf
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf32>)
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant dense<0xFF800000>
//  CHECK-NEXT: %[[cst_f32_0:.+]] = tensorrt.constant dense<0xFF800000>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[cst_f32_0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUB>(%[[v1]], %[[v2]] : {{.*}})
//  CHECK-NEXT: return %[[v3]] : tensor<2048x1xf32>

// -----

func.func @remove_prod_div_pair_negative_nan(%arg0: tensor<2048x1xf32>) -> tensor<2048x1xf32>{
    %cst_f32 = tensorrt.constant dense<0x7fc00000> : tensor<1x1xf32>
    %cst_f32_2 = tensorrt.constant dense<0x7fc00000> : tensor<2048x1xf32>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %1 = tensorrt.element_wise <kDIV>(%0, %cst_f32_2 : tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    %2 = tensorrt.element_wise <kSUM>(%1, %cst_f32 : tensor<2048x1xf32>, tensor<1x1xf32>) -> tensor<2048x1xf32>
    %3 = tensorrt.element_wise <kSUB>(%1, %2: tensor<2048x1xf32>, tensor<2048x1xf32>) -> tensor<2048x1xf32>
    return %3 : tensor<2048x1xf32>
}

// CHECK-LABEL: @remove_prod_div_pair_negative_nan
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048x1xf32>)
//  CHECK-NEXT: %[[cst_f32:.+]] = tensorrt.constant dense<0x7FC00000>
//  CHECK-NEXT: %[[cst_f32_0:.+]] = tensorrt.constant dense<0x7FC00000>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.element_wise <kPROD>(%[[arg0]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[v0]], %[[cst_f32_0]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[cst_f32]] : {{.*}})
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUB>(%[[v1]], %[[v2]] : {{.*}})
//  CHECK-NEXT: return %[[v3]] : tensor<2048x1xf32>

// -----

func.func @raise_layer_normalization_op(%arg0: tensor<1x197x768xf32>) -> tensor<197x768xf32> {
    %eps = tensorrt.constant dense<0.0001> : tensor<1x197x1xf32>
    %scale = tensorrt.constant dense<1.0> : tensor<1x1x768xf32>
    %bias = tensorrt.constant dense<0.0> : tensor<1x1x768xf32>
    %6 = tensorrt.element_wise <kPROD>(%arg0, %arg0 : tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %7 = tensorrt.reduce <kAVG> %arg0 {reduceAxes = array<i64: 2>} : tensor<1x197x768xf32> -> tensor<1x197xf32>
    %8 = tensorrt.reduce <kAVG> %6 {reduceAxes = array<i64: 2>} : tensor<1x197x768xf32> -> tensor<1x197xf32>
    %9 = tensorrt.element_wise <kPROD>(%7, %7 : tensor<1x197xf32>, tensor<1x197xf32>) -> tensor<1x197xf32>
    %10 = tensorrt.element_wise <kSUB>(%8, %9 : tensor<1x197xf32>, tensor<1x197xf32>) -> tensor<1x197xf32>
    %11 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %10 : tensor<1x197xf32>
    %12 = tensorrt.expand_rank %7 : tensor<1x197xf32> to tensor<1x197x1xf32>
    %13 = tensorrt.expand_rank %11 : tensor<1x197xf32> to tensor<1x197x1xf32>
    %14 = tensorrt.element_wise <kSUB>(%arg0, %12 : tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %15 = tensorrt.element_wise <kSUM>(%13, %eps : tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %16 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %15 : tensor<1x197x1xf32>
    %17 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %16 : tensor<1x197x1xf32>
    %18 = tensorrt.element_wise <kPROD>(%17, %scale : tensor<1x197x1xf32>, tensor<1x1x768xf32>) -> tensor<1x197x768xf32>
    %19 = tensorrt.element_wise <kPROD>(%14, %18 : tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %20 = tensorrt.element_wise <kSUM>(%19, %bias : tensor<1x197x768xf32>, tensor<1x1x768xf32>) -> tensor<1x197x768xf32>
    %21 = tensorrt.collapse_rank %20 : tensor<1x197x768xf32> to tensor<197x768xf32>
    return %21: tensor<197x768xf32>
}

// CHECK-LABEL: @raise_layer_normalization_op
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x197x768xf32>) -> tensor<197x768xf32>
//  CHECK-NEXT: %[[c0:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<1x1x768xf32>
//  CHECK-NEXT: %[[c1:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<1x1x768xf32>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.normalization {axis = array<i64: 2>, eps = 9.99999974E-5 : f32}(%[[arg0]] : tensor<1x197x768xf32>, %[[c0]] : tensor<1x1x768xf32>, %[[c1]] : tensor<1x1x768xf32>)
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : {{.*}}
//  CHECK-NEXT: return %[[v1]] : tensor<197x768xf32>

// -----

func.func @raise_layer_normalization_op_v2(%arg0: tensor<8x128x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<8x128x1024xf32> {
    %cst_f32 = tensorrt.constant dense<9.99999974E-6> : tensor<8x128x1xf32>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %arg0 : tensor<8x128x1024xf32>, tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %1 = tensorrt.reduce <kAVG> %arg0 {reduceAxes = array<i64: 2>} : tensor<8x128x1024xf32> -> tensor<8x128xf32>
    %2 = tensorrt.reduce <kAVG> %0 {reduceAxes = array<i64: 2>} : tensor<8x128x1024xf32> -> tensor<8x128xf32>
    %3 = tensorrt.element_wise <kPROD>(%1, %1 : tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %4 = tensorrt.element_wise <kSUB>(%2, %3 : tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %5 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %4 : tensor<8x128xf32>
    %6 = tensorrt.expand_rank %1 : tensor<8x128xf32> to tensor<8x128x1xf32>
    %7 = tensorrt.expand_rank %5 : tensor<8x128xf32> to tensor<8x128x1xf32>
    %8 = tensorrt.element_wise <kSUB>(%arg0, %6 : tensor<8x128x1024xf32>, tensor<8x128x1xf32>) -> tensor<8x128x1024xf32>
    %9 = tensorrt.element_wise <kSUM>(%7, %cst_f32 : tensor<8x128x1xf32>, tensor<8x128x1xf32>) -> tensor<8x128x1xf32>
    %10 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %9 : tensor<8x128x1xf32>
    %11 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %10 : tensor<8x128x1xf32>
    %12 = tensorrt.expand_rank %arg1 : tensor<1024xf32> to tensor<1x1x1024xf32>
    %13 = tensorrt.element_wise <kPROD>(%11, %12 : tensor<8x128x1xf32>, tensor<1x1x1024xf32>) -> tensor<8x128x1024xf32>
    %14 = tensorrt.element_wise <kPROD>(%8, %13 : tensor<8x128x1024xf32>, tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %15 = tensorrt.expand_rank %arg2 : tensor<1024xf32> to tensor<1x1x1024xf32>
    %16 = tensorrt.element_wise <kSUM>(%14, %15 : tensor<8x128x1024xf32>, tensor<1x1x1024xf32>) -> tensor<8x128x1024xf32>
    return %16 : tensor<8x128x1024xf32>
}

// CHECK-LABEL: @raise_layer_normalization_op_v2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8x128x1024xf32>, %[[arg1:.+]]: tensor<1024xf32>, %[[arg2:.+]]: tensor<1024xf32>)
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.expand_rank %[[arg2]]
//  CHECK-NEXT: %2 = tensorrt.normalization {axis = array<i64: 2>, eps = 9.99999974E-6 : f32}(%[[arg0]] : tensor<8x128x1024xf32>, %[[v0]] : tensor<1x1x1024xf32>, %[[v1]] : tensor<1x1x1024xf32>)
//  CHECK-NEXT: return %[[v2]] : tensor<8x128x1024xf32>

// -----

func.func @raise_layer_normalization_op_v3(%arg0: tensor<8x128x1024xf32>, %arg1: tensor<1x1x1024xf32>, %arg2: tensor<1x1x1024xf32>) -> tensor<8x128x1024xf32> {
    %cst_f32 = tensorrt.constant dense<9.99999974E-6> : tensor<8x128x1xf32>
    %0 = tensorrt.element_wise <kPROD>(%arg0, %arg0 : tensor<8x128x1024xf32>, tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %1 = tensorrt.reduce <kAVG> %arg0 {reduceAxes = array<i64: 2>} : tensor<8x128x1024xf32> -> tensor<8x128xf32>
    %2 = tensorrt.reduce <kAVG> %0 {reduceAxes = array<i64: 2>} : tensor<8x128x1024xf32> -> tensor<8x128xf32>
    %3 = tensorrt.element_wise <kPROD>(%1, %1 : tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %4 = tensorrt.element_wise <kSUB>(%2, %3 : tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
    %5 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %4 : tensor<8x128xf32>
    %6 = tensorrt.expand_rank %1 : tensor<8x128xf32> to tensor<8x128x1xf32>
    %7 = tensorrt.expand_rank %5 : tensor<8x128xf32> to tensor<8x128x1xf32>
    %8 = tensorrt.element_wise <kSUB>(%arg0, %6 : tensor<8x128x1024xf32>, tensor<8x128x1xf32>) -> tensor<8x128x1024xf32>
    %9 = tensorrt.element_wise <kSUM>(%7, %cst_f32 : tensor<8x128x1xf32>, tensor<8x128x1xf32>) -> tensor<8x128x1xf32>
    %10 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %9 : tensor<8x128x1xf32>
    %11 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %10 : tensor<8x128x1xf32>
    %13 = tensorrt.element_wise <kPROD>(%11, %arg1 : tensor<8x128x1xf32>, tensor<1x1x1024xf32>) -> tensor<8x128x1024xf32>
    %14 = tensorrt.element_wise <kPROD>(%8, %13 : tensor<8x128x1024xf32>, tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %16 = tensorrt.element_wise <kSUM>(%14, %arg2 : tensor<8x128x1024xf32>, tensor<1x1x1024xf32>) -> tensor<8x128x1024xf32>
    return %16 : tensor<8x128x1024xf32>
}

// CHECK-LABEL: @raise_layer_normalization_op_v3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8x128x1024xf32>, %[[arg1:.+]]: tensor<1x1x1024xf32>, %[[arg2:.+]]: tensor<1x1x1024xf32>)
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.normalization {axis = array<i64: 2>, eps = 9.99999974E-6 : f32}(%[[arg0]] : tensor<8x128x1024xf32>, %[[arg1]] : tensor<1x1x1024xf32>, %[[arg2]] : tensor<1x1x1024xf32>)
//  CHECK-NEXT: return %[[v0]] : tensor<8x128x1024xf32>

// -----

func.func @layer_normalization_different_scale_bias_shape(%arg0: tensor<1x197x768xf32>) -> tensor<197x768xf32> {
    %eps = tensorrt.constant dense<0.0001> : tensor<1x197x1xf32>
    %scale = tensorrt.constant dense<1.0> : tensor<1x1x768xf32>
    %bias = tensorrt.constant dense<0.0> : tensor<1x197x768xf32>
    %6 = tensorrt.element_wise <kPROD>(%arg0, %arg0 : tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %7 = tensorrt.reduce <kAVG> %arg0 {reduceAxes = array<i64: 2>} : tensor<1x197x768xf32> -> tensor<1x197xf32>
    %8 = tensorrt.reduce <kAVG> %6 {reduceAxes = array<i64: 2>} : tensor<1x197x768xf32> -> tensor<1x197xf32>
    %9 = tensorrt.element_wise <kPROD>(%7, %7 : tensor<1x197xf32>, tensor<1x197xf32>) -> tensor<1x197xf32>
    %10 = tensorrt.element_wise <kSUB>(%8, %9 : tensor<1x197xf32>, tensor<1x197xf32>) -> tensor<1x197xf32>
    %11 = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>} %10 : tensor<1x197xf32>
    %12 = tensorrt.expand_rank %7 : tensor<1x197xf32> to tensor<1x197x1xf32>
    %13 = tensorrt.expand_rank %11 : tensor<1x197xf32> to tensor<1x197x1xf32>
    %14 = tensorrt.element_wise <kSUB>(%arg0, %12 : tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
    %15 = tensorrt.element_wise <kSUM>(%13, %eps : tensor<1x197x1xf32>, tensor<1x197x1xf32>) -> tensor<1x197x1xf32>
    %16 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %15 : tensor<1x197x1xf32>
    %17 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %16 : tensor<1x197x1xf32>
    %18 = tensorrt.element_wise <kPROD>(%17, %scale : tensor<1x197x1xf32>, tensor<1x1x768xf32>) -> tensor<1x197x768xf32>
    %19 = tensorrt.element_wise <kPROD>(%14, %18 : tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %20 = tensorrt.element_wise <kSUM>(%19, %bias : tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %21 = tensorrt.collapse_rank %20 : tensor<1x197x768xf32> to tensor<197x768xf32>
    return %21: tensor<197x768xf32>
}

// CHECK-LABEL: @layer_normalization_different_scale_bias_shape
//   CHECK-NOT: tensorrt.normalization
//       CHECK: tensorrt.collapse_rank


// -----

func.func @ewise_constant_splat_broadcast_rhs(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %cst = tensorrt.constant dense<0.1> : tensor<10x1xf32>
  %0 = tensorrt.element_wise <kSUM>(%arg0, %cst: tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @ewise_constant_splat_broadcast_rhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x10xf32>)
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<{{.+}}> : tensor<1x1xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_f32]]
//       CHECK:     return %[[v0]] : tensor<10x10xf32>

// -----

func.func @ewise_constant_splat_broadcast_lhs(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %cst = tensorrt.constant dense<0.1> : tensor<10x1xf32>
  %0 = tensorrt.element_wise <kSUM>(%cst, %arg0: tensor<10x1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @ewise_constant_splat_broadcast_lhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x10xf32>)
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<{{.+}}> : tensor<1x1xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[cst_f32]], %[[arg0]]
//       CHECK:     return %[[v0]]

// -----

func.func @ewise_constant_splat_broadcast_negative(%arg0: tensor<1x10xf32>) -> tensor<10x10xf32> {
  %cst = tensorrt.constant dense<0.1> : tensor<10x1xf32>
  %0 = tensorrt.element_wise <kSUM>(%arg0, %cst: tensor<1x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @ewise_constant_splat_broadcast_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x10xf32>)
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<{{.+}}> : tensor<10x1xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_f32]]
//       CHECK:     return %[[v0]] : tensor<10x10xf32>

// -----

func.func @swap_mha_matmul_operands(%qk_t: tensor<1x12x197x197xf16>, %v: tensor<1x12x64x197xf16>) -> tensor<1x12x64x197xf16>{
  %0 = tensorrt.softmax {axis = 3 : i64} %qk_t : tensor<1x12x197x197xf16>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
  ins(%v, %0 : tensor<1x12x64x197xf16>, tensor<1x12x197x197xf16>) -> tensor<1x12x64x197xf16>
  return %1: tensor<1x12x64x197xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @swap_mha_matmul_operands
//  CHECK-SAME: (%[[qk_t:.+]]: {{.*}}, %[[v:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.softmax
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[v]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]}
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @swap_mha_matmul_operands_2(%qk_t: tensor<1x2x3x4xf16>, %v: tensor<1x2x8x4xf16>) -> tensor<1x2x8x3xf16>{
  %0 = tensorrt.softmax {axis = 3 : i64} %qk_t : tensor<1x2x3x4xf16>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
  ins(%v, %0 : tensor<1x2x8x4xf16>, tensor<1x2x3x4xf16>) -> tensor<1x2x8x3xf16>
  return %1: tensor<1x2x8x3xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @swap_mha_matmul_operands_2
//  CHECK-SAME: (%[[qk_t:.+]]: {{.*}}, %[[v:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.softmax
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[v]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]}
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @swap_mha_matmul_operands_3(%qk_t: tensor<1x2x3x4xf16>, %v: tensor<1x2x8x4xf16>, %arg: tensor<1x2x8x3xf16>) -> (tensor<1x2x8x3xf16>, tensor<1x2x8x3xf16>) {
  %0 = tensorrt.softmax {axis = 3 : i64} %qk_t : tensor<1x2x3x4xf16>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
  ins(%v, %0 : tensor<1x2x8x4xf16>, tensor<1x2x3x4xf16>) -> tensor<1x2x8x3xf16>
  %2 = tensorrt.element_wise <kSUM> (%1, %arg: tensor<1x2x8x3xf16>, tensor<1x2x8x3xf16>) -> tensor<1x2x8x3xf16>
  return %1, %2: tensor<1x2x8x3xf16>, tensor<1x2x8x3xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @swap_mha_matmul_operands_3
//  CHECK-SAME: (%[[qk_t:.+]]: {{.*}}, %[[v:.+]]: {{.*}}, %[[arg:.+]]: {{.*}})
//       CHECK: %[[v0:.+]] = tensorrt.softmax
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[v]] : {{.*}})
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]}
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.element_wise <kSUM>(%[[v2]], %[[arg]] : {{.*}})
//  CHECK-NEXT: return %[[v2]], %[[v3]]

// -----

func.func @swap_mha_matmul_operands_neg(%qk_t: tensor<1x12x197x197xf16>, %v: tensor<1x12x197x64xf16>) -> tensor<1x12x197x64xf16>{
  %0 = tensorrt.softmax {axis = 3 : i64} %qk_t : tensor<1x12x197x197xf16>
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
  ins(%0, %v : tensor<1x12x197x197xf16>, tensor<1x12x197x64xf16>) -> tensor<1x12x197x64xf16>
  return %1: tensor<1x12x197x64xf16>
}

// CHECK-LABEL: @swap_mha_matmul_operands_neg
//  CHECK-SAME: (%[[qk_t:.+]]: tensor<1x12x197x197xf16>, %[[v:.+]]: tensor<1x12x197x64xf16>)
//       CHECK: %[[v0:.+]] = tensorrt.softmax
//       CHECK: %[[v1:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[v]] : {{.*}})
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @swap_mha_matmul_operands_neg_2(%arg0: tensor<1x12x197x64xf16>, %arg1: tensor<1x12x197x64xf16>) -> tensor<1x12x197x197xf16>{
  %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
  ins(%arg0, %arg1 : tensor<1x12x197x64xf16>, tensor<1x12x197x64xf16>) -> tensor<1x12x197x197xf16>
  return %1: tensor<1x12x197x197xf16>
}

// CHECK-LABEL: @swap_mha_matmul_operands_neg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x12x197x64xf16>, %[[arg1:.+]]: tensor<1x12x197x64xf16>)
//       CHECK: %[[v0:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg0]], %[[arg1]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]
