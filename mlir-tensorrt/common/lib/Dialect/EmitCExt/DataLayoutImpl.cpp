//===- DataLayoutImpl.cpp -------------------------------------------------===//
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
/// This file contains the implementation for the DataLayout extensions to
/// the EmitC dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Dialect/EmitCExt/IR/DataLayoutImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {
/// Add DataLayoutTypeInterface to the `!emitc.size_t` type. We map all queries
/// to the corresponding property of the built-in `index` type.
struct SizeTDataLayoutInterface
    : public DataLayoutTypeInterface::ExternalModel<SizeTDataLayoutInterface,
                                                    emitc::SizeTType> {
  llvm::TypeSize getTypeSize(Type type, const DataLayout &dataLayout,
                             DataLayoutEntryListRef params) const {
    return dataLayout.getTypeSize(IndexType::get(type.getContext()));
  }
  llvm::TypeSize getTypeSizeInBits(Type type, const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
    return dataLayout.getTypeSizeInBits(IndexType::get(type.getContext()));
  }
  uint64_t getABIAlignment(Type type, const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const {
    return dataLayout.getTypeABIAlignment(IndexType::get(type.getContext()));
  }
  uint64_t getPreferredAlignment(Type type, const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const {
    return dataLayout.getTypePreferredAlignment(
        IndexType::get(type.getContext()));
  }
};
} // namespace

namespace mlir::emitc_ext {
void registerDataLayoutInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, emitc::EmitCDialect *emitcdialect) {
        emitc::SizeTType::attachInterface<SizeTDataLayoutInterface>(*ctx);
      });
}
} // namespace mlir::emitc_ext