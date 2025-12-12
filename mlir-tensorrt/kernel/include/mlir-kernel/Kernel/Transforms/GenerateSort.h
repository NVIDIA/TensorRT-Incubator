//===- GenerateSort.h -----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// GPU merge sort kernel generation utilities.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_GENERATE_SORT_H
#define MLIR_KERNEL_GENERATE_SORT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>

namespace mlir {
class MLIRContext;
class ModuleOp;
class Type;
class Value;
class OpBuilder;
class Location;
template <typename OpTy>
class OwningOpRef;
namespace gpu {
class GPUModuleOp;
}

namespace kernel {

/// Result of creating a merge sort kernel module
struct MergeSortKernelResult {
  /// Main dispatch function
  func::FuncOp dispatchFunc;
  /// Block sort kernel function
  func::FuncOp blockSortFunc;
  /// Partition kernel function
  func::FuncOp partitionFunc;
  /// Merge kernel function
  func::FuncOp mergeFunc;
};

/// Configuration for merge sort kernel generation
struct MergeSortConfig {
  /// Number of threads per block (must be power of 2)
  int64_t blockThreads = 128;

  /// Nominal number of items processed per thread (for 4-byte types).
  /// This value is scaled based on actual type size to maintain constant
  /// register pressure. See getActualItemsPerThread().
  ///
  /// Default of 4 is empirically optimized for Hopper/Blackwell GPUs with
  /// MLIR codegen. Smaller tiles provide better occupancy and memory
  /// coalescing.
  int64_t itemsPerThread = 4;

  /// Whether to generate stable sort (preserves relative order of equal
  /// elements)
  bool stable = false;

  /// Whether this is a keys-only sort (no values)
  bool keysOnly = true;

  /// Get actual items per thread based on type size.
  /// Scales the nominal itemsPerThread value to maintain constant register
  /// pressure across different data types, following CUB's
  /// Nominal4BItemsToItems strategy.
  ///
  /// For example, with itemsPerThread=4 (nominal for 4-byte types):
  ///   - i8  (1 byte):  min(4, 4*4/1) = 4 items/thread
  ///   - i32 (4 bytes): min(4, 4*4/4) = 4 items/thread
  ///   - i64 (8 bytes): min(4, 4*4/8) = 2 items/thread (scaled down)
  ///
  /// \param type The key type being sorted
  /// \return Scaled items per thread for the given type
  int64_t getActualItemsPerThread(mlir::Type type) const;
};

/// Generates GPU merge sort kernels following the CUB merge sort algorithm.
/// The implementation creates three main kernels:
/// 1. Block sort kernel - sorts data within each thread block
/// 2. Partition kernel - computes merge path partitions for parallel merging
/// 3. Merge kernel - merges sorted segments in parallel
class MergeSortKernelGenerator {
public:
  /// Creates a complete merge sort kernel module with dispatch and device
  /// kernels.
  ///
  /// \param ctx MLIR context
  /// \param keyType Type of the keys to sort (e.g., i32, f32)
  /// \param valueType Type of the values (optional, for key-value pairs)
  /// \param config Configuration parameters for the sort
  /// \return MergeSortKernelResult containing the generated module and
  /// functions
  static FailureOr<MergeSortKernelResult> createMergeSortKernels(
      OpBuilder &builder, Location loc, Type keyType, Type valueType,
      ModuleOp module, gpu::GPUModuleOp gpuModule,
      SymbolTableCollection &symbolTables, const MergeSortConfig &config = {});

  /// Helper to create the GPU module and basic setup
  static FailureOr<gpu::GPUModuleOp> createGPUModule(ModuleOp parentModule,
                                                     StringRef moduleName);

  /// Creates the block sort kernel that sorts data within each thread block
  static FailureOr<func::FuncOp>
  createBlockSortKernel(OpBuilder &builder, Location loc,
                        gpu::GPUModuleOp gpuModule, Type keyType,
                        Type valueType, const MergeSortConfig &config);

  /// Creates the partition kernel for merge path computation
  static FailureOr<func::FuncOp>
  createPartitionKernel(OpBuilder &builder, Location loc,
                        gpu::GPUModuleOp gpuModule, Type keyType,
                        Type valueType, const MergeSortConfig &config);

  /// Creates the merge kernel for parallel segment merging
  static FailureOr<func::FuncOp>
  createMergeKernel(OpBuilder &builder, Location loc,
                    gpu::GPUModuleOp gpuModule, Type keyType, Type valueType,
                    const MergeSortConfig &config);

  /// Creates the main dispatch function
  static FailureOr<func::FuncOp>
  createDispatchFunction(OpBuilder &builder, Location loc, ModuleOp module,
                         gpu::GPUModuleOp gpuModule, func::FuncOp blockSortFunc,
                         func::FuncOp partitionFunc, func::FuncOp mergeFunc,
                         Type keyType, Type valueType,
                         SymbolTableCollection &symbolTables,
                         const MergeSortConfig &config);
};

} // namespace kernel
} // namespace mlir

#endif // MLIR_KERNEL_GENERATE_SORT_H
