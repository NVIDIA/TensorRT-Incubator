//===- CUDAUtils.cpp ------------------------------------------------------===//
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
/// Implementation of utility functions for the CUDA dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/Utils/CUDAUtils.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::cuda;

Value cuda::createDefaultStream0(RewriterBase &rewriter, Location loc) {
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  Value device = rewriter.create<cuda::GetProgramDeviceOp>(loc, zero);
  return rewriter.create<cuda::GetGlobalStreamOp>(loc, device, /*index=*/0);
}

Value cuda::getOrCreateDefaultStream0(RewriterBase &rewriter,
                                      Operation *anchor) {
  return getOrCreateDefaultStream0(rewriter, anchor->getLoc(),
                                   anchor->getIterator());
}

Value cuda::getOrCreateDefaultStream0(RewriterBase &rewriter, Location loc,
                                      Block::iterator anchorPoint) {
  Block &block = *anchorPoint->getBlock();

  for (Operation &op : block) {
    if (op.getIterator() == anchorPoint)
      break;
    auto globalStreamOp = dyn_cast<cuda::GetGlobalStreamOp>(&op);
    if (!globalStreamOp)
      continue;
    if (globalStreamOp.getIndex() != 0 ||
        !matchPattern(globalStreamOp.getDevice(),
                      m_Op<cuda::GetProgramDeviceOp>(m_Zero())))
      continue;

    return op.getResult(0);
  }

  // No matching stream found, create a new one at the beginning of the block
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&block);
  return createDefaultStream0(rewriter, loc);
}
