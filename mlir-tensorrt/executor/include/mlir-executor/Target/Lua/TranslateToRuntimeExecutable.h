//===- TranslateToRuntimeExecutable.h ---------------------------*- C++ -*-===//
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
/// Declarations for translation `mlir-to-runtime-executable`.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TARGET_LUA_TRANSLATETORUNTIMEEXECUTABLE
#define MLIR_TENSORRT_TARGET_LUA_TRANSLATETORUNTIMEEXECUTABLE

#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// Translate the given op to an Executor runtime executable.
LogicalResult translateToRuntimeExecutable(Operation *op, raw_ostream &os);

/// Translate the given module to a Executor runtime executable, which is
/// returned as a serialized flatbuffer.
FailureOr<std::unique_ptr<mtrt::ExecutableStorage>>
translateToRuntimeExecutable(Operation *op);

/// Register the `-mlir-to-executable` translation in MLIR translation registry.
void registerToRuntimeExecutableTranslation();

} // namespace mlir

#endif // MLIR_TENSORRT_TARGET_LUA_TRANSLATETORUNTIMEEXECUTABLE
