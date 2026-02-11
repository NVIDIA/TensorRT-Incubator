//===- StablehloBuilder.h ---------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Declares convenience utilities for building Stable HLO operations.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class RewriterBase;
class Value;
class ValueRange;
class RankedTensorType;
class AffineMap;

namespace stablehlo {

class BroadcastInDimOp;
class ConcatenateOp;
class ReshapeOp;
class SliceOp;
class TransposeOp;

/// Helper wrapper around a RewriterBase for creating Stable HLO operations.
class StablehloBuilder {
public:
  explicit StablehloBuilder(RewriterBase &rewriter) : rewriter(rewriter) {}

  BroadcastInDimOp broadcastInDim(Value v, RankedTensorType newType,
                                  llvm::ArrayRef<int64_t> broadcastDims);
  ConcatenateOp concat(ValueRange v, int64_t dim);
  ReshapeOp expandDims(Value v, int64_t idx);
  SliceOp indexSelect(Value v, int64_t dim, int64_t idx);
  ReshapeOp reshape(Value v, llvm::ArrayRef<int64_t> shape);
  ReshapeOp squeezeDims(Value v, int64_t idx);
  TransposeOp transpose(Value v, llvm::ArrayRef<int64_t> perm);
  TransposeOp transpose(Value v, AffineMap perm);

private:
  RewriterBase &rewriter;
};

} // namespace stablehlo
} // namespace mlir
