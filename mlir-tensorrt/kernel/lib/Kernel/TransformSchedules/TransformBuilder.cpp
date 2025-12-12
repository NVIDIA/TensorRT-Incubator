//===- TransformBuilder.cpp -----------------------------------------------===//
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
/// Definitions for transform IR builder utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/TransformSchedules/TransformBuilder.h"

using namespace mlir;
using namespace mlir::kernel;

Value TransformIRBuilder::operandHandle(Value consumerHandle,
                                        int64_t operandIndex) {
  Value operandValueHandle = create<transform::GetOperandOp>(
                                 /*type=*/anyValueType,
                                 /*target=*/consumerHandle,
                                 /*raw_position_list=*/
                                 ArrayRef<int64_t>{operandIndex},
                                 /*is_inverted=*/false,
                                 /*is_all=*/false)
                                 ->getResult(0);
  return create<transform::GetDefiningOp>(anyOpType, operandValueHandle);
}

Value TransformIRBuilder::fuseProducersIntoContainingOp(
    Value containingOpHandle, MutableArrayRef<Value> producerHandles) {
  for (auto [idx, producerHandle] : llvm::enumerate(producerHandles)) {
    if (!producerHandle)
      continue;
    auto fuseOp = create<transform::FuseIntoContainingOp>(
        /*producer_op=*/producerHandle,
        /*containing_op=*/containingOpHandle);
    containingOpHandle = fuseOp.getNewContainingOp();
    producerHandle = fuseOp.getFusedOp();
  }
  return containingOpHandle;
}

void TransformIRBuilder::cse(Value funcHandle) {
  create<transform::ApplyCommonSubexpressionEliminationOp>(funcHandle);
}

/// Get the parent `scf.forall` op.
Value TransformIRBuilder::getParentForallOp(Value opHandle) {
  return create<transform::GetParentOp>(transform::AnyOpType::get(getContext()),
                                        opHandle, false, false,
                                        getStringAttr("scf.forall"), false);
}

transform::TileUsingForallOp
TransformIRBuilder::tileToForall(Value opHandle, ArrayRef<int64_t> numThreads,
                                 transform::NumThreadsSpec dispatch) {
  return create<transform::TileUsingForallOp>(opHandle, numThreads, dispatch);
}
transform::TileUsingForallOp
TransformIRBuilder::tileToForall(Value opHandle, ArrayRef<int64_t> tileShape) {
  return create<transform::TileUsingForallOp>(opHandle, tileShape,
                                              transform::TileSizesSpec{});
}

transform::SequenceOp TransformIRBuilder::sequence(
    Value funcHandle,
    std::function<void(TransformIRBuilder &, BlockArgument)> bodyBuilder) {
  return create<transform::SequenceOp>(
      anyOpType, transform::FailurePropagationMode::Propagate, funcHandle,
      [&](OpBuilder &b_, Location loc, BlockArgument funcHandle) {
        TransformIRBuilder b(loc, b_);
        bodyBuilder(b, funcHandle);
      });
}
