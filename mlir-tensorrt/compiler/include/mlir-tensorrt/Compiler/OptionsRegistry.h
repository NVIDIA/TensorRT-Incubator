//===- OptionsRegistry.h -----------------------------------------*- C++-*-===//
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
/// This is the top-level C++ interface for clients that wish to compile
/// MLIR programs from a supported input (e.g. StableHLO) into a supported
/// target (e.g. TensorRT engine). This file contains just the declarations for
/// the CompilerClient object, see below for details.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_OPTIONS_REGISTRY
#define MLIR_TENSORRT_COMPILER_OPTIONS_REGISTRY

#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <functional>

namespace mlirtrt::compiler {

using OptionsConstructorFuncT =
    std::function<StatusOr<std::unique_ptr<mlir::OptionsContext>>(
        mlir::MLIRContext *, llvm::ArrayRef<llvm::StringRef>)>;

/// Registers an options creation function for a specific options type.
void registerOption(llvm::StringRef optionsType, OptionsConstructorFuncT func);

/// Creates an options instance for the specified options type using a creation
/// function that was previously registered.
StatusOr<std::unique_ptr<mlir::OptionsContext>>
createOptions(mlir::MLIRContext *client, llvm::StringRef optionsType,
              llvm::ArrayRef<llvm::StringRef> args);

/// Helper to build callbacks that can create options.
template <typename OptionsT, typename TaskT>
StatusOr<std::unique_ptr<OptionsT>>
optionsCreateFromArgs(mlir::MLIRContext *context,
                      llvm::ArrayRef<llvm::StringRef> args) {
  // Load available extensions.
  mlir::plan::PlanDialect *planDialect =
      context->getLoadedDialect<mlir::plan::PlanDialect>();
  compiler::TaskExtensionRegistry extensions =
      planDialect->extensionConstructors.getExtensionRegistryForTask<TaskT>();

  auto result = std::make_unique<OptionsT>(std::move(extensions));

  std::string err;
  if (failed(result->parse(args, err))) {
    return getInternalErrorStatus(
        "failed to parse options string \"{0:$[ ]}\" due to error {1}",
        llvm::iterator_range(args), err);
  }

  llvm::Error finalizeStatus = result->finalize();

  std::optional<std::string> errMsg{};
  llvm::handleAllErrors(
      std::move(finalizeStatus),
      [&errMsg](const llvm::StringError &err) { errMsg = err.getMessage(); });

  if (errMsg)
    return getInternalErrorStatus("failed to initialize options: {0}", *errMsg);

  return result;
}
} // namespace mlirtrt::compiler

#endif
