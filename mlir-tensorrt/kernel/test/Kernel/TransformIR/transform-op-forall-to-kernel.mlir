// RUN: kernel-opt %s -split-input-file -transform-interpreter -cse | FileCheck %s

func.func @test_basic(%arg0: tensor<1024x1024xf32>,
                %arg1: tensor<1024x1024xf32>,
                %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg2) -> tensor<1024x1024xf32> {
    %row = affine.apply affine_map<()[s0]->(s0 * 128)>()[%i]
    %col = affine.apply affine_map<()[s0]->(s0 * 128)>()[%j]
    %a = tensor.extract_slice %arg0[%row, %c0][128, 1024][1, 1] : tensor<1024x1024xf32> to tensor<128x1024xf32>
    %b = tensor.extract_slice %arg1[%c0, %col][1024, 128][1, 1] : tensor<1024x1024xf32> to tensor<1024x128xf32>
    %c = tensor.extract_slice %out[%row, %col][128, 128][1, 1] : tensor<1024x1024xf32> to tensor<128x128xf32>
    %d = linalg.matmul ins(%a, %b : tensor<128x1024xf32>, tensor<1024x128xf32>) outs(%c : tensor<128x128xf32>)
      -> tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %d into %out[%row, %col][128, 128][1, 1] : tensor<128x128xf32> into tensor<1024x1024xf32>
    }
  }
  return %0 : tensor<1024x1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(128)
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-LABEL: func.func @test_basic(
//  CHECK-SAME: %[[arg0:.+]]: tensor<1024x1024xf32>, %[[arg1:.+]]: tensor<1024x1024xf32>, %[[arg2:.+]]: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
//   CHECK-DAG:     %[[c8:.+]] = arith.constant 8 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c128:.+]] = arith.constant 128 : index
//       CHECK:     %[[v0:.+]] = kernel.call @kernels::@test_basic_kernel grid[%[[c8]], %[[c8]], %[[c1]]] block[%[[c128]]] (%[[arg0]], %[[arg1]]) outs(%[[arg2]])
//       CHECK:     return %[[v0]]
// CHECK-LABEL: func.func @test_basic_kernel
//  CHECK-SAME: (%[[arg0m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg1m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg2m:.+]]: memref<1024x1024xf32, {{.*}}>)
//   CHECK-DAG:       %[[arg0:.+]] = bufferization.to_tensor %[[arg0m]] restrict :
//   CHECK-DAG:       %[[arg1:.+]] = bufferization.to_tensor %[[arg1m]] restrict :
//   CHECK-DAG:       %[[arg2:.+]] = bufferization.to_tensor %[[arg2m]] restrict writable :
//   CHECK-DAG:       %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[bx:.+]] = gpu.block_id  x
//   CHECK-DAG:       %[[by:.+]] = gpu.block_id  y
//   CHECK-DAG:       %[[o1:.+]] = affine.apply #[[$map]]()[%[[bx]]]
//   CHECK-DAG:       %[[o2:.+]] = affine.apply #[[$map]]()[%[[by]]]
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[o1]], %[[c0]]] [128, 1024] [1, 1] :
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[c0]], %[[o2]]] [1024, 128] [1, 1] :
//       CHECK:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg2]][%[[o1]], %[[o2]]] [128, 128] [1, 1] :
//       CHECK:       %[[v5:.+]] = linalg.matmul ins(%[[extracted_slice]], %[[extracted_slice_0]] : {{.*}}) outs(%[[extracted_slice_1]] :
//   CHECK-DAG:       %[[v8:.+]] = bufferization.to_buffer %[[arg2]] :
//   CHECK-DAG:       %[[subview:.+]] = memref.subview %[[v8]][%[[o1]], %[[o2]]] [128, 128] [1, 1] :
//   CHECK-DAG:       %[[v9:.+]] = bufferization.to_buffer %[[v5]] : tensor<128x128xf32> to memref<128x128xf32, {{.*}}>
//   CHECK-DAG:       memref.copy %[[v9]], %[[subview]] : memref<128x128xf32, {{.*}}> to memref<128x128xf32, {{.*}}>
//       CHECK:       return

// -----

// CHECK-LABEL: @test_rank_reduction
// CHECK-SAME: %[[arg0:.+]]: tensor<1024xf32>, %[[arg1:.+]]: tensor<1024xf32>, %[[arg2:.+]]: tensor<1024x1xf32>, %[[arg3:.+]]: tensor<1024x1xf32>)
func.func @test_rank_reduction(
                %arg0: tensor<1024xf32>,
                %arg1: tensor<1024xf32>,
                %arg2: tensor<1024x1xf32>,
                %arg3: tensor<1024x1xf32>) -> (tensor<1024x1xf32>, tensor<1024x1xf32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index

  // CHECK: %[[a_tensor:.+]] = bufferization.to_tensor %[[arg0]]
  // CHECK: %[[out_tensor1:.+]] = bufferization.to_tensor %[[arg2]]
  // CHECK: %[[out_tensor2:.+]] = bufferization.to_tensor %[[arg3]]
  // CHECK: %[[offset:.+]] = affine.apply
  // CHECK: %[[a_tile:.+]] = tensor.extract_slice %[[a_tensor]]
  // CHECK: %[[b1_tile:.+]] = tensor.expand_shape

  %0:2 = scf.forall (%i) in (8) shared_outs(%out1 = %arg2, %out2 = %arg3) -> (tensor<1024x1xf32>, tensor<1024x1xf32>) {
    %offset = affine.apply affine_map<()[s0]->(s0 * 128)>()[%i]
    %a = tensor.extract_slice %arg0[%offset][128][1] : tensor<1024xf32> to tensor<128xf32>
    %b = tensor.extract_slice %arg1[%offset][128][1] : tensor<1024xf32> to tensor<128xf32>
    %b1 = tensor.expand_shape %b [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    scf.forall.in_parallel {
      // CHECK-DAG: %[[src:.+]] = bufferization.to_buffer %[[out_tensor1]] :
      // CHECK-DAG: %[[dest:.+]] = bufferization.to_buffer %[[a_tile]] :
      // CHECK-DAG: %[[subview:.+]] = memref.subview %[[src]][%[[offset]], 0] [128, 1] [1, 1] : memref<1024x1xf32, {{.*}}> to memref<128xf32, {{.*}}>
      //     CHECK: memref.copy %[[dest]], %[[subview]] : memref<128xf32, {{.*}}> to memref<128xf32, {{.*}}>
      tensor.parallel_insert_slice %a into %out1[%offset, 0][128, 1][1, 1] : tensor<128xf32> into tensor<1024x1xf32>
      // CHECK-DAG: %[[dest:.+]] = bufferization.to_buffer %[[out_tensor2]] :
      // CHECK-DAG: %[[src:.+]] = bufferization.to_buffer %[[b1_tile]] :
      // CHECK-DAG: %[[dest_sv:.+]] = memref.subview %[[dest]][%[[offset]], 0] [128, 1] [1, 1] : {{.*}} to memref<128x1xf32, {{.*}}>
      //     CHECK: memref.copy %[[src]], %[[dest_sv]] : memref<128x1xf32, {{.*}}> to memref<128x1xf32, {{.*}}>
      tensor.parallel_insert_slice %b1 into %out2[%offset, 0][128, 1][1, 1] : tensor<128x1xf32> into tensor<1024x1xf32>
    }
  }
  return %0#0, %0#1 : tensor<1024x1xf32>, tensor<1024x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(128)
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @multiple_kernels(%arg0: tensor<1024x1024xf32>,
                            %arg1: tensor<1024x1024xf32>,
                            %arg2: tensor<1024x1024xf32>,
                            %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg3) -> tensor<1024x1024xf32> {
    %row = affine.apply affine_map<()[s0]->(s0 * 128)>()[%i]
    %col = affine.apply affine_map<()[s0]->(s0 * 128)>()[%j]
    %a = tensor.extract_slice %arg0[%row, %c0][128, 1024][1, 1] : tensor<1024x1024xf32> to tensor<128x1024xf32>
    %b = tensor.extract_slice %arg1[%c0, %col][1024, 128][1, 1] : tensor<1024x1024xf32> to tensor<1024x128xf32>
    %c = tensor.extract_slice %out[%row, %col][128, 128][1, 1] : tensor<1024x1024xf32> to tensor<128x128xf32>
    %d = linalg.matmul ins(%a, %b : tensor<128x1024xf32>, tensor<1024x128xf32>) outs(%c : tensor<128x128xf32>)
      -> tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %d into %out[%row, %col][128, 128][1, 1] : tensor<128x128xf32> into tensor<1024x1024xf32>
    }
  }
  %1 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg3) -> tensor<1024x1024xf32> {
    %row = affine.apply affine_map<()[s0]->(s0 * 128)>()[%i]
    %col = affine.apply affine_map<()[s0]->(s0 * 128)>()[%j]
    %a = tensor.extract_slice %0[%row, %c0][128, 1024][1, 1] : tensor<1024x1024xf32> to tensor<128x1024xf32>
    %b = tensor.extract_slice %arg2[%c0, %col][1024, 128][1, 1] : tensor<1024x1024xf32> to tensor<1024x128xf32>
    %c = tensor.extract_slice %out[%row, %col][128, 128][1, 1] : tensor<1024x1024xf32> to tensor<128x128xf32>
    %d = linalg.matmul ins(%a, %b : tensor<128x1024xf32>, tensor<1024x128xf32>) outs(%c : tensor<128x128xf32>)
      -> tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %d into %out[%row, %col][128, 128][1, 1] : tensor<128x128xf32> into tensor<1024x1024xf32>
    }
  }
  return %1 : tensor<1024x1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(64)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-LABEL: func.func @multiple_kernels(
//  CHECK-SAME: %[[arg0:.+]]: tensor<1024x1024xf32>, %[[arg1:.+]]: tensor<1024x1024xf32>, %[[arg2:.+]]: tensor<1024x1024xf32>, %[[arg3:.+]]: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
//   CHECK-DAG:     %[[c8:.+]] = arith.constant 8 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c64:.+]] = arith.constant 64 : index
//       CHECK:     %[[v0:.+]] = kernel.call @kernels::@multiple_kernels_kernel grid[%[[c8]], %[[c8]], %[[c1]]] block[%[[c64]]] (%[[arg0]], %[[arg1]]) outs(%[[arg3]])
//       CHECK:     %[[v1:.+]] = kernel.call @kernels::@multiple_kernels_kernel_0 grid[%[[c8]], %[[c8]], %[[c1]]] block[%[[c64]]] (%[[arg2]], %[[v0]]) outs(%[[arg3]])
//       CHECK:     return %[[v1]] : tensor<1024x1024xf32>
//       CHECK:   gpu.module @kernels
// CHECK-LABEL: func.func @multiple_kernels_kernel
//  CHECK-SAME: (%[[arg0m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg1m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg2m:.+]]: memref<1024x1024xf32, {{.*}}>)
//  CHECK-SAME:         gpu.known_grid_size = array<i32: 8, 8, 1>
//  CHECK-SAME:         kernel.num_threads = 64 : i64
//   CHECK-DAG:       %[[arg0:.+]] = bufferization.to_tensor %[[arg0m]] restrict :
//   CHECK-DAG:       %[[arg1:.+]] = bufferization.to_tensor %[[arg1m]] restrict :
//   CHECK-DAG:       %[[arg2:.+]] = bufferization.to_tensor %[[arg2m]] restrict writable :
//   CHECK-DAG:       %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[v0:.+]] = gpu.block_id  x
//   CHECK-DAG:       %[[v1:.+]] = gpu.block_id  y
//   CHECK-DAG:       %[[o1:.+]] = affine.apply #[[$map]]()[%[[v0]]]
//   CHECK-DAG:       %[[o2:.+]] = affine.apply #[[$map]]()[%[[v1]]]
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[o1]], %[[c0]]] [128, 1024] [1, 1] :
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[c0]], %[[o2]]] [1024, 128] [1, 1] :
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg2]][%[[o1]], %[[o2]]] [128, 128] [1, 1] :
//   CHECK-DAG:       %[[v5:.+]] = linalg.matmul ins(%[[extracted_slice]], %[[extracted_slice_0]] : tensor<128x1024xf32>, tensor<1024x128xf32>) outs(%[[extracted_slice_1]] : tensor<128x128xf32>) -> tensor<128x128xf32>
//   CHECK-DAG:       %[[v8:.+]] = bufferization.to_buffer %[[arg2]]
//   CHECK-DAG:       %[[subview:.+]] = memref.subview %[[v8]][%[[o1]], %[[o2]]] [128, 128] [1, 1] : memref<1024x1024xf32, {{.*}}
//   CHECK-DAG:       %[[v9:.+]] = bufferization.to_buffer %[[v5]] : tensor<128x128xf32> to memref<128x128xf32, {{.*}}>
//       CHECK:       memref.copy %[[v9]], %[[subview]]


// CHECK-LABEL: func.func @multiple_kernels_kernel_0
//  CHECK-SAME: (%[[arg0m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg1m:.+]]: memref<1024x1024xf32, {{.*}}>, %[[arg2m:.+]]: memref<1024x1024xf32, {{.*}}>)


// -----

func.func @nested_forall(%arg0: tensor<?xf32>,
                            %arg1: tensor<?xf32>,
                            %arg2: index,
                            %arg3: index) -> tensor<?xf32> {
  %0 = scf.forall (%i) in (%arg2) shared_outs (%out = %arg1) -> tensor<?xf32> {
    %offt = affine.apply affine_map<(d0)->(d0 * 64)>(%i)
    %1 = tensor.extract_slice %out[%offt][64][1] : tensor<?xf32> to tensor<64xf32>
    %2 = scf.forall (%j) in (%arg3) shared_outs (%out1 = %1) -> tensor<64xf32> {
      %offt1 = affine.apply affine_map<(d0, d1)->(d0 * 64 + d1 * 32)>(%i, %j)
      %offt2 = affine.apply affine_map<(d0)->(d0 * 32)>(%j)
      %a = tensor.extract_slice %arg0[%offt1][32][1] : tensor<?xf32> to tensor<32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %a into %out1[%offt2][32][1] : tensor<32xf32> into tensor<64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %out[%offt][64][1] : tensor<64xf32> into tensor<?xf32>
    }
  }
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(128) {
      gpu_target = #nvvm.target<chip = "sm_80">
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0) -> (d0 * 64)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1 * 32)>
//       CHECK: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: func.func @nested_forall(
//  CHECK-SAME: %[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xf32>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) -> tensor<?xf32> {
//       CHECK:     %[[c128:.+]] = arith.constant 128 : index
//       CHECK:     %[[v0:.+]] = kernel.call @kernels::@nested_forall_kernel grid[%{{.*}}] block[%[[c128]]] (%[[arg0]], %[[arg3]]) outs(%[[arg1]])


// CHECK-LABEL: func.func @nested_forall_kernel
//  CHECK-SAME: (%[[arg0m:.+]]: memref<?xf32, {{.*}}>, %[[arg1:.+]]: index, %[[arg2m:.+]]: memref<?xf32, {{.*}}>)
//  CHECK-SAME: kernel.num_threads = 128 : i64
//       CHECK:       %[[arg0:.+]] = bufferization.to_tensor %[[arg0m]] restrict :
//       CHECK:       %[[arg2:.+]] = bufferization.to_tensor %[[arg2m]] restrict writable :
//       CHECK:       %[[v0:.+]] = gpu.block_id  x
//       CHECK:       %[[v1:.+]] = affine.apply #[[$map]](%[[v0]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[v1]]] [64] [1] : tensor<?xf32> to tensor<64xf32>
//       CHECK:       %[[v2:.+]] = scf.forall (%[[arg3:.+]]) in (%[[arg1]]) shared_outs(%[[arg4:.+]] = %[[extracted_slice]]) -> (tensor<64xf32>) {
//       CHECK:         %[[v3:.+]] = affine.apply #[[$map1]](%[[v0]], %[[arg3]])
//       CHECK:         %[[v4:.+]] = affine.apply #[[$map2]](%[[arg3]])
//       CHECK:         %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg0]][%[[v3]]] [32] [1] : tensor<?xf32> to tensor<32xf32>
//       CHECK:         scf.forall.in_parallel
//       CHECK:           tensor.parallel_insert_slice %[[extracted_slice_0]] into %[[arg4]][%[[v4]]] [32] [1] : tensor<32xf32> into tensor<64xf32>
//   CHECK-DAG:       %[[v4:.+]] = bufferization.to_buffer %[[arg2]] : tensor<?xf32> to memref<?xf32, {{.*}}>
//   CHECK-DAG:       %[[subview:.+]] = memref.subview %[[v4]][%[[v1]]] [64] [1] : memref<?xf32, {{.*}}> to memref<64xf32, {{.*}}>
//   CHECK-DAG:       %[[v5:.+]] = bufferization.to_buffer %[[v2]] : tensor<64xf32> to memref<64xf32, {{.*}}>
//       CHECK:       memref.copy %[[v5]], %[[subview]]



// -----

!tensor_type = tensor<8x8xf32>

// Test the 'reuse_existing_gpu_module' and 'extra_module_attrs' attributes.

func.func @create_multiple_gpu_modules(
                            %arg0: !tensor_type,
                            %arg1: !tensor_type,
                            %arg2: !tensor_type,
                            %arg3: !tensor_type) -> !tensor_type {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg3) -> !tensor_type {
    %lhs = tensor.extract %arg0[%i, %j] : !tensor_type
    %rhs = tensor.extract %arg1[%i, %j] : !tensor_type
    %res_scalar = arith.addf %lhs, %rhs : f32
    %res_tensor = tensor.from_elements %res_scalar : tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %res_tensor into %out[%i, %j][1, 1][1, 1] : tensor<1x1xf32> into !tensor_type
    }
  }
  %1 = scf.forall (%i, %j) in (%c8, %c8) shared_outs(%out = %arg3) -> !tensor_type {
    %lhs = tensor.extract %0[%i, %j] : !tensor_type
    %rhs = tensor.extract %arg2[%i, %j] : !tensor_type
    %res_scalar = arith.addf %lhs, %rhs : f32
    %res_tensor = tensor.from_elements %res_scalar : tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %res_tensor into %out[%i, %j][1, 1][1, 1] : tensor<1x1xf32> into !tensor_type
    }
  }
  return %1 : !tensor_type
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(64) {
      reuse_existing_gpu_module = false,
      extra_module_attrs = {
        targets = [#nvvm.target<chip = "sm_80">]
      }
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @create_multiple_gpu_modules(
//       CHECK:   gpu.module @create_multiple_gpu_modules_kernel [#nvvm.target<chip = "sm_80">]
//       CHECK:   gpu.module @create_multiple_gpu_modules_kernel_0 [#nvvm.target<chip = "sm_80">]

// -----

func.func @complex_fill(%arg0: tensor<?xcomplex<f32>>) -> tensor<?xcomplex<f32>> {
  %cst = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %0 = scf.forall (%i) in (8) shared_outs(%out = %arg0) -> tensor<?xcomplex<f32>> {
    %1 = tensor.insert %cst into %out[%i] : tensor<?xcomplex<f32>>
    %c0 = arith.constant 0 : index
    %size = tensor.dim %out, %c0 : tensor<?xcomplex<f32>>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %out[%c0][%size][1] : tensor<?xcomplex<f32>> into tensor<?xcomplex<f32>>
    }
  }
  return %0 : tensor<?xcomplex<f32>>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(64) {
      reuse_existing_gpu_module = false
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// Just verify that the constant constant is cloned into the kernel function:
// CHECK-LABEL: func.func @complex_fill(
//   CHECK-NOT:  complex.constant
//       CHECK:  kernel.call @complex_fill_kernel::@kernel
//       CHECK: gpu.module @complex_fill_kernel
// CHECK-LABEL: func.func @kernel
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xcomplex<f32>, strided<[?], offset: ?>>)
//       CHECK:       %[[cst:.+]] = complex.constant [1.000000e+00 : f32, 2.000000e+00 : f32] : complex<f32>
//       CHECK:       tensor.insert %[[cst]] into


// -----

!tensor_type = tensor<?x?x?x?x?x?xf32>
!slice_type = tensor<2x2x2x2x2x2xf32>

func.func @mixed_dynamic_and_static(%arg0: !tensor_type, %arg1: index, %arg2: index) -> !tensor_type {
  %0 = scf.forall (%i0, %i1, %i2, %i3, %i4, %i5) in (%arg1, 1024, %arg2, 1024, 1024, 1024) shared_outs(%out = %arg0) -> !tensor_type {
    %slice = tensor.extract_slice %out[%i0, %i1, %i2, %i3, %i4, %i5][2, 2, 2, 2, 2, 2][1, 1, 1, 1, 1, 1] : !tensor_type to !slice_type
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i0, %i1, %i2, %i3, %i4, %i5][2, 2, 2, 2, 2, 2][1, 1, 1, 1, 1, 1] : !slice_type into !tensor_type
    }
  }
  return %0 : !tensor_type
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %call_op, %kern_module_op, %kernel_func_op = transform.kernel.forall_to_kernel %forall_op threads(64) {
      reuse_existing_gpu_module = false
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 1024)>
//       CHECK: #[[$map1:.+]] = affine_map<()[s0] -> (s0 floordiv 1024)>
//       CHECK: #[[$map2:.+]] = affine_map<()[s0] -> (s0 mod 1024)>

// CHECK-LABEL: func.func @mixed_dynamic_and_static
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?x?x?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
//   CHECK-DAG:     %[[v0:.+]] = affine.apply #[[$map]]()[%[[arg1]]]
//   CHECK-DAG:     %[[c1048576:.+]] = arith.constant 1048576 : index
//   CHECK-DAG:     %[[v1:.+]] = affine.apply #[[$map]]()[%[[arg2]]]
//   CHECK-DAG:     %[[c64:.+]] = arith.constant 64 : index
//   CHECK-DAG:     kernel.call @mixed_dynamic_and_static_kernel::@kernel grid[%[[v0]], %[[c1048576]], %[[v1]]] block[%[[c64]]]
// CHECK-LABEL: func.func @kernel
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?x?x?x?x?xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>>)
//   CHECK-DAG:       %[[v0:.+]] = bufferization.to_tensor %[[arg0]]
//   CHECK-DAG:       %[[block_id_x:.+]] = gpu.block_id  x
//   CHECK-DAG:       %[[v1:.+]] = affine.apply #[[$map1]]()[%[[block_id_x]]]
//   CHECK-DAG:       %[[v2:.+]] = affine.apply #[[$map2]]()[%[[block_id_x]]]
//   CHECK-DAG:       %[[block_id_y:.+]] = gpu.block_id  y
//   CHECK-DAG:       %[[v3:.+]] = affine.apply #[[$map1]]()[%[[block_id_y]]]
//   CHECK-DAG:       %[[v4:.+]] = affine.apply #[[$map2]]()[%[[block_id_y]]]
//   CHECK-DAG:       %[[block_id_z:.+]] = gpu.block_id  z
//   CHECK-DAG:       %[[v5:.+]] = affine.apply #[[$map1]]()[%[[block_id_z]]]
//   CHECK-DAG:       %[[v6:.+]] = affine.apply #[[$map2]]()[%[[block_id_z]]]
//   CHECK-DAG:       tensor.extract_slice %[[v0]][%[[v1]], %[[v3]], %[[v5]], %[[v2]], %[[v4]], %[[v6]]]
