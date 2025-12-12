//===- VectorTransforms.cpp -----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Definitions for vector-dialect related transformation utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

LogicalResult kernel::replaceVectorTransferReadWithConstant(
    RewriterBase &rewriter, vector::TransferReadOp transferReadOp) {
  auto producer =
      transferReadOp.getSource().getDefiningOp<vector::TransferWriteOp>();

  SplatElementsAttr constAttr;
  if (!producer || !matchPattern(producer.getVector(), m_Constant(&constAttr)))
    return failure();

  if (!llvm::all_of(producer.getInBounds(),
                    [](Attribute inBounds) {
                      return isa<BoolAttr>(inBounds) &&
                             cast<BoolAttr>(inBounds).getValue() == true;
                    }) ||
      !llvm::all_of(producer.getIndices(),
                    [](Value v) { return matchPattern(v, m_Zero()); }) ||
      producer.getVector().getType().getShape() !=
          cast<RankedTensorType>(producer->getResult(0).getType()).getShape())
    return failure();

  rewriter.replaceOpWithNewOp<arith::ConstantOp>(
      transferReadOp, transferReadOp.getType(),
      DenseElementsAttr::get(transferReadOp.getType(),
                             constAttr.getSplatValue<Attribute>()));
  return success();
}
