//===- Utils.cpp ----------------------------------------------------------===//
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
#include "mlir-executor/Executor/Utils/Utils.h"

using namespace mlir;
using namespace mlir::executor;

DataSegmentOp executor::getOrCreateConstantResourceDeclaration(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name,
    ElementsAttr data) {
  auto op = module.lookupSymbol<DataSegmentOp>(name);
  if (op && op.getValueAttr() == data)
    return op;
  auto resourceOp =
      executor::DataSegmentOp::create(loc, name, data, /*constant=*/true,
                                      /*uninitialized=*/false, IntegerAttr{});
  SymbolTable(module).insert(resourceOp);
  return resourceOp;
}

DataSegmentOp executor::createConstantResourceDeclaration(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name,
    ElementsAttr data, bool constant, bool uninitialized) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(module.getBody());
  auto resourceOp = executor::DataSegmentOp::create(
      loc, name, data, constant, uninitialized, IntegerAttr{});
  b.insert(resourceOp);
  return resourceOp;
}

GlobalOp executor::getOrCreateGlobalOp(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name, Type type,
    bool constant,
    std::function<void(OpBuilder &, Location)> initRegionBuilder) {
  for (auto globalOp : module.getOps<executor::GlobalOp>()) {
    if (globalOp.getSymName() == name)
      return globalOp;
  }
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(module.getBody());
  auto globalOp = b.create<executor::GlobalOp>(loc, name, type,
                                               initRegionBuilder, constant);
  return globalOp;
}

GlobalOp executor::createUniqueGlobalOp(
    Location loc, ModuleOp module, StringRef name, Type type, bool constant,
    std::function<void(OpBuilder &, Location)> initRegionBuilder) {
  OpBuilder b(loc->getContext());
  auto globalOp = b.create<executor::GlobalOp>(loc, name, type,
                                               initRegionBuilder, constant);
  SymbolTable symbolTable(module);

  // Insert after the last global op, or at the beginning if there are no global
  // ops.
  auto insertPt = Block::iterator(module.getBody()->front());
  for (auto globalOp : module.getOps<GlobalOp>())
    insertPt = Block::iterator(globalOp);
  symbolTable.insert(globalOp, insertPt);
  return globalOp;
}

SmallString<16> executor::getUniqueSymbolName(ModuleOp moduleOp,
                                              StringRef prefix) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (prefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));
  return stringConstName;
}

DataSegmentOp executor::getOrCreateStringConstant(OpBuilder &b, Location loc,
                                                  ModuleOp moduleOp,
                                                  StringRef symbolName,
                                                  StringRef str,
                                                  uint64_t alignment) {
  llvm::SmallString<20> nullTermStr(str);
  nullTermStr.push_back('\0'); // Null terminate for C
  StringAttr attr = b.getStringAttr(nullTermStr);
  // Try to find existing global.
  for (auto globalOp : moduleOp.getOps<DataSegmentOp>())
    if (globalOp.getAddressSpace() == executor::MemoryType::host &&
        globalOp.getConstant() && globalOp.getValueAttr() == attr &&
        globalOp.getAlignment().value_or(0) == alignment)
      return globalOp;
  // Not found: create new global.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(moduleOp.getBody());
  SmallString<16> name = getUniqueSymbolName(moduleOp, symbolName);
  auto resourceOp = executor::DataSegmentOp::create(
      loc, name, attr, /*constant=*/true, /*uninitialized=*/false,
      /*alignment=*/b.getI64IntegerAttr(alignment));
  b.insert(resourceOp);
  return resourceOp;
}
