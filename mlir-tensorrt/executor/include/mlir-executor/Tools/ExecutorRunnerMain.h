//===- ExecutorRunnerMain.h -------------------------------------*- C++ -*-===//
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
/// Functions for creating a runner entrypoint.
///
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_EXECUTOR_TOOLS_EXECUTORRUNNERMAIN
#define INCLUDE_MLIR_EXECUTOR_TOOLS_EXECUTORRUNNERMAIN

#include "mlir/Support/LogicalResult.h"
#include <cstdlib>
#include <functional>

namespace mlir::executor {
/// Defines the possible types of input data.
enum InputType {
  /// The input type was unspecified and should be inferrred from the filename.
  Unspecified,
  /// A Lua text source (*.lua if a file).
  Lua,
  /// An Executor runtime executable (*.edb if a file).
  ExecutorRuntimeExecutable
};

/// Implementation for tools like `executor-runner`.
LogicalResult ExecutorRunnerMain(int argc, char **argv,
                                 std::function<void()> postInitCallback = {});

/// Helper wrapper to return the result of ExecutorRunnerMain directly from
/// main.
inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace mlir::executor

#endif // INCLUDE_MLIR_EXECUTOR_TOOLS_EXECUTORRUNNERMAIN
