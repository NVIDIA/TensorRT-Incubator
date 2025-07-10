//===- OwningModuleRef.cpp ------------------------------------------------===//
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
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Utils/OwningModuleRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace mlir;

void mlir::detail::releaseReferencedResourceElementsAttrsBlobs(
    mlir::Operation *op) {
  // Release the MLIR resource blobs referenced in the module.
  // Currently we only use ResourceElementsAttr variants, which are owned by the
  // Builtin dialect. All their storage is managed by builtin dialect's
  // ResourceBloblManager. Unfortunately currently we don't have a way to
  // iterate the keys until a more recent upstream change, so we need to go
  // through and check all ops/attributes.
  auto &builtinDialectResourceBlobManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(op->getContext());

  op->walk([&](mlir::Operation *op) {
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      if (auto resourceAttr = llvm::dyn_cast<mlir::DenseResourceElementsAttr>(
              attr.getValue())) {
        auto handle = resourceAttr.getRawHandle();
        if (handle.getBlob() && !handle.getBlob()->getData().empty()) {
          llvm::StringRef blobName = handle.getKey();
          builtinDialectResourceBlobManager.update(blobName,
                                                   mlir::AsmResourceBlob());
        }
      }
    }
  });
}

OwningModuleRef::OwningModuleRef(std::nullptr_t)
    : op(nullptr), beforeDestroyHook(nullptr), afterDestroyHook(nullptr) {}

OwningModuleRef::OwningModuleRef(ModuleOp op)
    : op(op),
      beforeDestroyHook(detail::releaseReferencedResourceElementsAttrsBlobs),
      afterDestroyHook(nullptr) {}

OwningModuleRef::OwningModuleRef(OwningModuleRef &&other)
    : op(other.release()), beforeDestroyHook(other.beforeDestroyHook),
      afterDestroyHook(other.afterDestroyHook) {}

OwningModuleRef::~OwningModuleRef() { eraseOp(); }

void OwningModuleRef::eraseOp() {
  if (!op)
    return;
  if (beforeDestroyHook)
    beforeDestroyHook(op);
  op->erase();
  if (afterDestroyHook)
    afterDestroyHook();
}

/// Assign from another op reference.
OwningModuleRef &OwningModuleRef::operator=(OwningModuleRef &&other) {
  eraseOp();
  op = other.release();
  return *this;
}

/// Allow accessing the internal op.
ModuleOp OwningModuleRef::get() const { return op; }
ModuleOp OwningModuleRef::operator*() const { return op; }
ModuleOp OwningModuleRef::operator->() { return op; }

/// Release the referenced op.
ModuleOp OwningModuleRef::release() {
  ModuleOp released(nullptr);
  std::swap(released, op);
  return released;
}
