//===- ApplyTransforms.cpp ------------------------------------------------===//
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
// Implementation of pass to apply Transform IR to each function in the module.
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_APPLYTRANSFORMSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

namespace {
//===----------------------------------------------------------------------===//
// Apply Transforms Pass
//===----------------------------------------------------------------------===//
struct ApplyTransformsPass
    : public kernel::impl::ApplyTransformsPassBase<ApplyTransformsPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    // Check that the top level target operation has the SymbolTable trait
    if (!rootOp->hasTrait<OpTrait::SymbolTable>()) {
      rootOp->emitError()
          << "pass " << getPassName()
          << " requires root operation to have SymbolTable trait";
      return signalPassFailure();
    }

    // Create dictionaries from func names to transforms
    llvm::StringMap<transform::SequenceOp> transforms;
    rootOp->walk([&](transform::SequenceOp transformOp) {
      if (transformOp->getParentOp() != rootOp)
        return;
      if (auto def = transformOp->getAttrOfType<SymbolRefAttr>(
              KernelDialect::getDialectNamespace().str() + ".target_func"))
        transforms[def.getRootReference().str()] = transformOp;
    });

    // Lookup the func and apply the corresponding transformation to each func
    for (const auto &[funcName, transformOp] : transforms) {
      if (auto funcOp = dyn_cast_or_null<func::FuncOp>(
              SymbolTable::lookupSymbolIn(rootOp, funcName))) {
        // We found the function symbol, now we can apply the transform.
        auto transformed = transform::applyTransforms(funcOp, transformOp);

        if (failed(transformed))
          return signalPassFailure();

        // Delete the whole transform.sequence op
        transformOp->erase();

      } else {
        emitError(transformOp->getLoc()) << "function for transform schedule "
                                         << funcName << " is not found";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
