//===- PopulateDefaultBackendMetadata.cpp
//----------------------------------===//
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
/// Implementation of the `plan-populate-default-backend-metadata` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::plan {
#define GEN_PASS_DEF_POPULATEDEFAULTBACKENDMETADATAPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {
class PopulateDefaultBackendMetadataPass
    : public plan::impl::PopulateDefaultBackendMetadataPassBase<
          PopulateDefaultBackendMetadataPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(plan::PlanDialect::kBackendsAttrName))
      return;

    std::string backendsStr =
        llvm::formatv("[{0}]", llvm::iterator_range(backends));

    Attribute configArrayAttr =
        mlir::parseAttribute(backendsStr, module->getContext());
    if (!configArrayAttr || !isa<ArrayAttr>(configArrayAttr)) {
      emitError(module->getLoc(),
                "failed to parse config as an array attribute: " + backendsStr);
      return signalPassFailure();
    }
    module->setAttr(plan::PlanDialect::kBackendsAttrName, configArrayAttr);
  }
};
} // namespace
