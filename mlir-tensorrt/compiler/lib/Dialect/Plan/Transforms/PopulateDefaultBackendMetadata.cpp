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
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"

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

  plan::MemorySpaceAttr defaultMemorySpace;
  SmallVector<plan::CompilerBackendAttrInterface> parsedBackends;

  LogicalResult parseBackends(MLIRContext *ctx) {
    for (const std::string &backend : backends) {
      Attribute parsed = mlir::parseAttribute(backend, ctx);
      if (!parsed) {
        return emitError(UnknownLoc::get(ctx))
               << "pass " << getArgument() << " failed to parse "
               << " backend attribute \"" << backend << "\"";
      }
      auto backendAttr = dyn_cast<plan::CompilerBackendAttrInterface>(parsed);
      if (!backendAttr) {
        return emitError(UnknownLoc::get(ctx))
               << "pass " << getArgument() << " parsed "
               << " backend attribute \"" << backend << "\" but it is not a "
               << "CompilerBackendAttrInterface attribute";
      }
      parsedBackends.push_back(backendAttr);
    }
    return success();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    if (!backends.empty() && failed(parseBackends(ctx)))
      return failure();

    if (defaultMemorySpaceString.empty())
      return success();

    // Try to parse it using the enum shorthand first.
    if (std::optional<plan::MemorySpace> parsedMemorySpace =
            plan::symbolizeMemorySpace(defaultMemorySpaceString)) {
      defaultMemorySpace =
          plan::MemorySpaceAttr::get(ctx, parsedMemorySpace.value());
      return success();
    }

    Attribute parsed = mlir::parseAttribute(defaultMemorySpaceString, ctx);
    if (!parsed) {
      return emitError(UnknownLoc::get(ctx))
             << "pass " << getArgument() << " failed to parse "
             << " default-memory-space option \"" << defaultMemorySpaceString
             << "\"";
    }
    defaultMemorySpace = dyn_cast<plan::MemorySpaceAttr>(parsed);
    if (!defaultMemorySpace) {
      return emitError(UnknownLoc::get(ctx))
             << "pass " << getArgument()
             << " parsed default-memory-space option \""
             << defaultMemorySpaceString
             << "\" but it is not a plan.memory_space attribute";
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(plan::PlanDialect::kBackendsAttrName))
      return;

    auto existingMemorySpaceConstraint =
        module->getAttrOfType<plan::MemorySpaceAttr>(
            plan::PlanDialect::kMemorySpaceConstraintAttrName);
    if (defaultMemorySpace) {
      if (existingMemorySpaceConstraint &&
          existingMemorySpaceConstraint != defaultMemorySpace) {
        emitRemark(module->getLoc())
            << "compiler received option default-memory-space=\""
            << defaultMemorySpaceString
            << "\", but the module has an existing constraint \""
            << existingMemorySpaceConstraint << "\" which takes precedence";
      } else if (!existingMemorySpaceConstraint) {
        module->setAttr(plan::PlanDialect::kMemorySpaceConstraintAttrName,
                        defaultMemorySpace);
      }
    }

    if (!parsedBackends.empty()) {
      module->setAttr(
          plan::PlanDialect::kBackendsAttrName,
          ArrayAttr::get(module->getContext(),
                         llvm::to_vector_of<Attribute>(parsedBackends)));
    }
  }
};
} // namespace
