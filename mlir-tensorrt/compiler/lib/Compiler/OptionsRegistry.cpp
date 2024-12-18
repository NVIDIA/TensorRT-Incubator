//===- OptionsRegistry.cpp-------------------------------------------------===//
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
#include "mlir-tensorrt/Compiler/OptionsRegistry.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlirtrt::compiler;

static llvm::ManagedStatic<llvm::StringMap<OptionsConstructorFuncT>> registry{};

void mlirtrt::compiler::registerOption(llvm::StringRef optionsType,
                                       OptionsConstructorFuncT func) {
  (*registry)[optionsType] = std::move(func);
}

mlirtrt::StatusOr<std::unique_ptr<mlir::OptionsContext>>
mlirtrt::compiler::createOptions(mlir::MLIRContext *ctx,
                                 llvm::StringRef optionsType,
                                 llvm::ArrayRef<llvm::StringRef> args) {
  if (!registry->contains(optionsType))
    return getInvalidArgStatus(
        "{0} is not a valid option type. Valid options were: {1:$[ ]}",
        optionsType, llvm::iterator_range(registry->keys()));
  return (*registry)[optionsType](ctx, args);
}
