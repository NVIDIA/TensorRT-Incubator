//===- ModuleUtils.cpp ----------------------------------------------------===//
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
/// Utilities for querying information about or manipulating module-like ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;

ModuleLikeOp::ModuleLikeOp(Operation *op) : op(nullptr) {
  if (op->hasTrait<OpTrait::SymbolTable>() &&
      op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
      op->hasTrait<OpTrait::OneRegion>())
    this->op = op;
}

StringRef ModuleLikeOp::getSymbolName() const {
  assert(*this && "expected valid op");
  auto name = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  if (!name)
    return "unknown-symbol-name";
  return name.strref();
}

LogicalResult mlir::getFuncOpsOrderedByCalls(
    ModuleLikeOp moduleOp, SmallVectorImpl<func::FuncOp> &orderedFuncOps,
    SmallVectorImpl<func::FuncOp> &remainingFuncOps,
    const std::function<bool(func::FuncOp)> &filter) {

  // Call graph doesn't give information about external callables, so enqueue
  // all of those first.
  for (auto func : moduleOp.getOps<func::FuncOp>()) {
    if (func.isDeclaration())
      orderedFuncOps.push_back(func);
  }

  const mlir::CallGraph callgraph(moduleOp);
  for (auto &scc : llvm::make_range(llvm::scc_begin(&callgraph),
                                    llvm::scc_end(&callgraph))) {
    if (scc.size() == 1) {
      if ((*scc.front()).isExternal()) {

        continue;
      }
      auto func = dyn_cast<func::FuncOp>(
          scc.front()->getCallableRegion()->getParentOp());
      if (!func || (filter && !filter(func)))
        continue;
      orderedFuncOps.push_back(func);
      continue;
    }

    for (auto &node : scc) {
      if (node->isExternal())
        continue;
      auto func = dyn_cast<func::FuncOp>(
          scc.front()->getCallableRegion()->getParentOp());

      // Exclude nested symbol tables.
      if (!func || (func && !filter(func)))
        continue;
      remainingFuncOps.push_back(func);
    }
  }

  return success();
}
