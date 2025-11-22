//===- Utils.h ------------------------------------------------------------===//
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
/// Utilities for the Executor dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_UTILS_UTILS_H
#define MLIR_TENSORRT_DIALECT_EXECUTOR_UTILS_UTILS_H

#include "mlir-executor/Executor/IR/Executor.h"

namespace mlir {
class RewriterBase;
namespace executor {

/// Within the given `module`, check if a DataSegmentOp with the given
/// containing `data` and `name` exists. If it does, return that
/// DataSegmentOp. Otherwise, create the resource op with the given name,
/// insert it into the top of the module, and return it.
DataSegmentOp getOrCreateConstantResourceDeclaration(OpBuilder &b, Location loc,
                                                     ModuleOp module,
                                                     StringRef name,
                                                     ElementsAttr data);

/// Construct a new DataSegmentOp with the given name, data, and insert it into
/// the top of the module.
DataSegmentOp createConstantResourceDeclaration(OpBuilder &b, Location loc,
                                                ModuleOp module, StringRef name,
                                                ElementsAttr data,
                                                bool constant = true,
                                                bool uninitialized = false);

/// Within the given `module`, check if GlobalOp with the given name exists. If
/// it does, return that GlobalOp. Otherwise, insert it into the module and
/// return it.
GlobalOp getOrCreateGlobalOp(
    OpBuilder &b, Location loc, ModuleOp module, StringRef name, Type type,
    bool constant,
    std::function<void(OpBuilder &, Location)> initRegionBuidler);

/// Within the given `module`, insert the global into the module and return it.
/// The name may be updated to create a unqiue global, so the final symbol name
/// of the returned global may be different than `name`.
GlobalOp createUniqueGlobalOp(
    Location loc, ModuleOp module, StringRef name, Type type, bool constant,
    std::function<void(OpBuilder &, Location)> initRegionBuidler);

SmallString<16> getUniqueSymbolName(ModuleOp moduleOp, StringRef prefix);

DataSegmentOp getOrCreateStringConstant(OpBuilder &b, Location loc,
                                        ModuleOp moduleOp, StringRef namePrefix,
                                        StringRef str, uint64_t alignment = 4);

} // namespace executor
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_UTILS_UTILS_H
