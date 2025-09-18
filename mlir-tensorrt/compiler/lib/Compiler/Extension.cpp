//===- Extension.cpp ------------------------------------------------------===//
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
/// Compiler extension mechanism definitions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Extension.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"

#define DEBUG_TYPE "compiler-extensions"
#define DBGV(fmt, ...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("[" DEBUG_TYPE "] " fmt "\n",       \
                                           __VA_ARGS__))

using namespace mtrt;
using namespace mtrt::compiler;

static llvm::ManagedStatic<ExtensionConstructorRegistry>
    globalTaskExtensionRegistry;

TaskExtensionBase::~TaskExtensionBase() {}

void ExtensionList::loadExtensions(CompilationTaskOptionsBase &task) {
  for (auto &[extensionName, builder] : builders) {
    if (extensions.contains(extensionName))
      continue;
    DBGV("Loading extension: {0}", extensionName);
    extensions.insert(std::make_pair(extensionName, builder(task)));
  }
}

void ExtensionConstructorRegistry::addExtension(llvm::StringRef taskName,
                                                llvm::StringRef extensionName,
                                                ConstructorFunc constructor) {
  llvm::sys::ScopedLock lock(registryMutex);
  if (!constructors.contains(taskName)) {
    ExtensionList::ExtensionBuilders inner = {
        {extensionName, std::move(constructor)}};
    constructors.insert(std::make_pair(taskName, std::move(inner)));
    return;
  }
  constructors[taskName].insert(
      std::make_pair(extensionName, std::move(constructor)));
}

ExtensionList ExtensionConstructorRegistry::getExtensionsForTask(
    llvm::StringRef taskName) const {
  llvm::sys::ScopedLock lock(registryMutex);
  if (!constructors.contains(taskName))
    return ExtensionList();
  return ExtensionList(constructors.lookup(taskName));
}

void compiler::registerExtension(llvm::StringRef taskName,
                                 llvm::StringRef extensionName,
                                 ExtensionList::ConstructorFunc constructor) {
  globalTaskExtensionRegistry->addExtension(taskName, extensionName,
                                            constructor);
}

ExtensionList compiler::getExtensionsForTask(llvm::StringRef taskName) {
  return globalTaskExtensionRegistry->getExtensionsForTask(taskName);
}
