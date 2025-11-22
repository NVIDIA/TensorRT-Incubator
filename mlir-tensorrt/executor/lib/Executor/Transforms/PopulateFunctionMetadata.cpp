//===- PopulateFunctionMetadata.cpp ---------------------------------------===//
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
/// Implementation for `executor-populate-function-metadata`.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORPOPULATEFUNCTIONMETADATAPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

/// Creates metadata for a function. Pass named `plan-alloc-tensors` which
/// runs before this changes functions that are not explicitly marked as private
/// to be DPS. During this process, result arguments (DPS init) with
/// `ExecutorDialect::kResultArgAttrName` attribute are added for each returned
/// result value. Only functions that are not explicitly private are assigned
/// metadata, since this is used to generate the executable metadata for
/// public-facing functions.
static LogicalResult createFunctionMetadata(FunctionOpInterface funcOp) {
  SmallVector<Type> argTypes;
  SmallVector<mlir::Attribute> argAttr;
  int64_t numOutputArgs = 0;
  llvm::SmallSetVector<uint32_t, 4> seenArgs;
  for (BlockArgument arg : funcOp.getArguments()) {
    if (auto resultSlot = funcOp.getArgAttrOfType<IntegerAttr>(
            arg.getArgNumber(), ExecutorDialect::kResultArgAttrName)) {
      if (seenArgs.contains(resultSlot.getInt())) {
        return funcOp.emitError()
               << "result slot " << resultSlot.getInt() << " is already used";
      }
      if (!seenArgs.getArrayRef().empty() &&
          seenArgs.getArrayRef().back() >= resultSlot.getInt()) {
        return funcOp.emitError() << "malformed result slot attributes";
      }
      seenArgs.insert(resultSlot.getInt());
      numOutputArgs++;
    }
    argTypes.push_back(arg.getType());
    argAttr.push_back(executor::getFuncArgsBounds(funcOp, arg.getArgNumber()));
  }

  SmallVector<Type> resultTypes;
  SmallVector<mlir::Attribute> resAttr;
  FunctionType funcType = cast<FunctionType>(funcOp.getFunctionType());
  for (auto [idx, type] : llvm::enumerate(funcType.getResults())) {
    resultTypes.push_back(type);
    resAttr.push_back(executor::getFuncResultBounds(funcOp, idx));
  }

  auto shapeSymAttr = funcOp->getAttrOfType<FlatSymbolRefAttr>(
      executor::ExecutorDialect::kShapeFuncAttrName);
  auto metadataAttr = executor::FunctionMetadataAttr::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(funcOp.getLoc()),
      funcOp.getContext(), ArrayRef<Type>(argTypes),
      ArrayRef<Type>(resultTypes), numOutputArgs, ArrayRef<Attribute>(argAttr),
      ArrayRef<Attribute>(resAttr), shapeSymAttr, CallingConvention::unpacked);

  if (!metadataAttr)
    return funcOp.emitError()
           << "the #executor.function_metadata attribute (" << metadataAttr
           << ") is not compatible with the function type (" << funcType << ")";

  funcOp->setAttr(ExecutorDialect::kFunctionMetadataAttrName, metadataAttr);

  return success();
}

namespace {
class ExecutorPopulateFunctionMetadataPass
    : public executor::impl::ExecutorPopulateFunctionMetadataPassBase<
          ExecutorPopulateFunctionMetadataPass> {
  using Base::Base;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.isPrivate() || !isa<FunctionType>(funcOp.getFunctionType()))
      return;

    if (failed(createFunctionMetadata(funcOp))) {
      emitError(funcOp.getLoc())
          << "failed to create Executor function metadata for function "
          << funcOp.getName() << " of with function type "
          << funcOp.getFunctionType();
      return signalPassFailure();
    }
  }
};
} // namespace
