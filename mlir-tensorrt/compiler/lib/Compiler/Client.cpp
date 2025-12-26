//===- Client.cpp ---------------------------------------------------------===//
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
/// Implementation for the CompilerClient.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mtrt;
using namespace mtrt::compiler;
using namespace mlir;

#define DEBUG_TYPE "compiler-api"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

static llvm::ManagedStatic<llvm::StringMap<TaskRegistration>> taskRegistry{};

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<CompilerClient>>
CompilerClient::create(MLIRContext *context) {
  return std::unique_ptr<CompilerClient>(new CompilerClient(context));
}

CompilerClient::CompilerClient(mlir::MLIRContext *context) : context(context) {}

StatusOr<PipelineBase *>
CompilerClient::getPipeline(llvm::StringRef mnemonic,
                            llvm::ArrayRef<llvm::StringRef> options,
                            bool enableDebugOptions) {
  if (!taskRegistry.isConstructed())
    return getInvalidArgStatus("no pipeline registered with name {0}",
                               mnemonic);
  auto it = taskRegistry->find(mnemonic);
  if (it == taskRegistry->end())
    return getInvalidArgStatus("no pipeline registered with name {0}",
                               mnemonic);
  return it->second.registryFunc(*this, options, enableDebugOptions);
}

void CompilerClient::updateCachedPipeline(llvm::StringRef taskName,
                                          const llvm::hash_code &hash,
                                          std::unique_ptr<PipelineBase> task) {
  cachedPassManagers[taskName][hash] = std::move(task);
}

PipelineBase *
CompilerClient::lookupCachedPipeline(llvm::StringRef taskName,
                                     const llvm::hash_code &optionsHash) const {
  auto it = cachedPassManagers.find(taskName);
  if (it == cachedPassManagers.end())
    return nullptr;
  auto it2 = it->second.find(optionsHash);
  if (it2 == it->second.end())
    return nullptr;
  return it2->second.get();
}

void compiler::detail::registerPipeline(llvm::StringRef taskName,
                                        TaskRegistryFunction func) {
  if (taskRegistry->contains(taskName))
    llvm::report_fatal_error("detected double registration of pipeline \"" +
                             taskName + "\"");

  taskRegistry->insert({taskName, TaskRegistration{std::move(func)}});
}

//===----------------------------------------------------------------------===//
// Task Lookup Utilities
//===----------------------------------------------------------------------===//

llvm::SmallVector<llvm::StringRef> compiler::getRegisteredPipelineNames() {
  llvm::SmallVector<llvm::StringRef> result;
  for (const auto &[name, registration] : *taskRegistry)
    result.push_back(name);
  return result;
}

void compiler::printPipelineHelp(mlir::MLIRContext *ctx,
                                 llvm::StringRef mnemonic) {
  StatusOr<std::unique_ptr<CompilerClient>> client =
      compiler::CompilerClient::create(ctx);
  mtrt::cantFail(client);
  StatusOr<PipelineBase *> pipeline = (*client)->getPipeline(mnemonic, {});
  mtrt::cantFail(pipeline);
  assert(*pipeline != nullptr && "expected a valid pipeline");
  (*pipeline)->getOptions().printHelp(0, 70);
}
