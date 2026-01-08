//===- CUDAUtils.h ----------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
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
/// Utility functions for the CUDA dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_UTILS_CUDAUTILS_H
#define MLIR_TENSORRT_DIALECT_CUDA_UTILS_CUDAUTILS_H

#include "mlir/IR/Block.h"
namespace mlir {
class Operation;
class Value;
class Location;
class RewriterBase;
class PatternRewriter;

namespace cuda {

/// Create a default stream (stream 0) on device 0. This creates:
/// - A constant 0 for the device index
/// - A cuda.get_program_device operation
/// - A cuda.stream.create operation with index 0
Value createDefaultStream0(RewriterBase &rewriter, Location loc);

/// Go over the operations in Block (containing anchor) from the first operation
/// in the Block to the point before `anchor`. If we find a `cuda.stream.create`
/// operation matching the pattern produced by `createDefaultStream0`, return
/// the result of that operation. Otherwise, call createDefaultStream0 to create
/// a new stream at the beginning of the block.
Value getOrCreateDefaultStream0(RewriterBase &rewriter, Operation *anchor);

/// Go over the operations in Block (containing anchor point) from the first
/// operation in the Block to the point before `anchor point`. If we find a
/// `cuda.stream.create` operation matching the pattern produced by
/// `createDefaultStream0`, return the result of that operation. Otherwise, call
/// createDefaultStream0 to create a new stream at the beginning of the block.
Value getOrCreateDefaultStream0(RewriterBase &rewriter, Location loc,
                                Block::iterator anchorPoint);

} // namespace cuda
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_CUDA_UTILS_CUDAUTILS_H
