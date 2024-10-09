//===- RegionUtils.cpp ----------------------------------------------------===//
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
#include "mlir-executor/Utils/RegionUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::tensorrt;

SmallVector<Value>
tensorrt::sinkConstantsAndGetUsedValuesDefinedAbove(RewriterBase &rewriter,
                                                    Region &body) {
  assert(body.getBlocks().size() == 1 && "expected a single-block region");
  Block *bodyBlock = &body.getBlocks().front();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(bodyBlock);
  SetVector<Value> usedDefinedAbove;
  getUsedValuesDefinedAbove(body, usedDefinedAbove);

  llvm::SmallSetVector<Value, 8> finalUsedDefinedAbove(usedDefinedAbove.begin(),
                                                       usedDefinedAbove.end());
  auto cloneableValues = llvm::make_filter_range(usedDefinedAbove, [](Value v) {
    Type t = v.getType();
    Operation *defOp = v.getDefiningOp();
    return defOp && defOp->hasTrait<OpTrait::ConstantLike>() &&
           (t.isIntOrFloat() || isa<VectorType>(t) || isa<IndexType>(t));
  });

  for (Value v : cloneableValues) {
    Operation *clone = rewriter.clone(*v.getDefiningOp());
    rewriter.replaceOpUsesWithinBlock(v.getDefiningOp(), clone->getResults(),
                                      bodyBlock);
    finalUsedDefinedAbove.remove(v);
  }
  return SmallVector<Value>(finalUsedDefinedAbove.begin(),
                            finalUsedDefinedAbove.end());
}
