//===- Interfaces.cpp -----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Kernel interface definitions.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"

#define DEBUG_TYPE "kernel-interfaces"
#define DBGS()                                                                 \
  (llvm::dbgs() << __FILE__ << ":" << __LINE__ << " [" DEBUG_TYPE "] ")

using namespace mlir;
using namespace mlir::kernel;

//===----------------------------------------------------------------------===//
// PointerLikeTypeInterface
//===----------------------------------------------------------------------===//

void mlir::kernel::registerExternalDialectPtrTypeInterfaceImpls(
    DialectRegistry &registry) {
  // LLVM pointer type: opaque pointer-like (no pointee information).
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *) {
    struct LLVMPointerLikeModel
        : public mlir::kernel::PointerLikeTypeInterface::ExternalModel<
              LLVMPointerLikeModel, LLVM::LLVMPointerType> {
      Type getPointeeType(Type) const { return Type(); }
    };
    LLVM::LLVMPointerType::attachInterface<LLVMPointerLikeModel>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, mlir::ptr::PtrDialect *) {
    struct PtrDialectPointerLikeModel
        : public mlir::kernel::PointerLikeTypeInterface::ExternalModel<
              PtrDialectPointerLikeModel, mlir::ptr::PtrType> {
      Type getPointeeType(Type) const { return Type(); }
    };
    mlir::ptr::PtrType::attachInterface<PtrDialectPointerLikeModel>(*ctx);
  });
}

//===----------------------------------------------------------------------===//
// TableGen'd interface definition.
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/AttrInterfaces.cpp.inc"
#include "mlir-kernel/Kernel/IR/OpInterfaces.cpp.inc"
#include "mlir-kernel/Kernel/IR/TypeInterfaces.cpp.inc"
