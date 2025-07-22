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
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#define DEBUG_TYPE "compiler-api"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

static llvm::ManagedStatic<llvm::DenseMap<mlir::TypeID, TaskRegistration>>
    taskRegistry{};

/// Global registry for mapping task mnemonics to type IDs.
static llvm::ManagedStatic<llvm::StringMap<mlir::TypeID>> taskNameRegistry;

//===----------------------------------------------------------------------===//
// CompilationTask
//===----------------------------------------------------------------------===//
void CompilationTaskBase::setupPassManagerInstrumentation(
    const DebugOptions *options) {
  // TODO: add API in upstream to detect whether this PM already has
  // instrumentation attached.
  if (options) {
    options->applyToPassManager(*this);
    return;
  }
  // Populate from global CL options.
  // TODO: we may want to consider making this a non-error.
  if (failed(applyPassManagerCLOptions(*this)))
    llvm::report_fatal_error("failed to populate pass manager "
                             "instrumentation from global CL options");
  applyDefaultTimingPassManagerCLOptions(*this);
}

CompilationTaskBase::CompilationTaskBase(
    MLIRContext *context, mlir::TypeID typeID,
    std::unique_ptr<CompilationTaskOptionsBase> options)
    : mlir::PassManager(context, mlir::ModuleOp::getOperationName()),
      typeID(typeID), taskOptions(std::move(options)) {
  setupPassManagerInstrumentation(taskOptions->getDebugOptions());
}

CompilationTaskBase::~CompilationTaskBase() {}

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<CompilerClient>>
CompilerClient::create(MLIRContext *context) {
  return std::unique_ptr<CompilerClient>(new CompilerClient(context));
}

CompilerClient::CompilerClient(mlir::MLIRContext *context) : context(context) {}

static StatusOr<CompilationTaskBase *>
lookupAndBuildTask(CompilerClient &client, ArrayRef<StringRef> options,
                   mlir::TypeID taskID, bool enableDebugOptions) {
  if (!taskRegistry.isConstructed())
    llvm::report_fatal_error("no such task registered");
  auto it = taskRegistry->find(taskID);
  if (it == taskRegistry->end())
    llvm::report_fatal_error("no such task registered");
  return it->second.registryFunc(client, options, enableDebugOptions);
}

static StatusOr<CompilationTaskBase *>
lookupAndBuildTask(CompilerClient &client, ArrayRef<StringRef> options,
                   StringRef mnemonic, bool enableDebugOptions) {
  if (!taskNameRegistry.isConstructed())
    return getInvalidArgStatus("no compilation task registered with name {0}",
                               mnemonic);
  auto it = taskNameRegistry->find(mnemonic);
  if (it == taskNameRegistry->end())
    return getInvalidArgStatus("no compilation task registered with name {0}",
                               mnemonic);
  return lookupAndBuildTask(client, options, it->second, enableDebugOptions);
}

StatusOr<CompilationTaskBase *>
CompilerClient::getCompilationTask(mlir::TypeID taskID,
                                   llvm::ArrayRef<llvm::StringRef> options,
                                   bool enableDebugOptions) {
  return lookupAndBuildTask(*this, options, taskID, enableDebugOptions);
}

StatusOr<CompilationTaskBase *>
CompilerClient::getCompilationTask(llvm::StringRef mnemonic,
                                   llvm::ArrayRef<StringRef> options,
                                   bool enableDebugOptions) {
  return lookupAndBuildTask(*this, options, mnemonic, enableDebugOptions);
}

void compiler::detail::registerCompilationTask(llvm::StringRef mnemonic,
                                               mlir::TypeID typeID,
                                               TaskRegistryFunction func) {
  if (taskNameRegistry->contains(mnemonic) || taskRegistry->contains(typeID))
    llvm::report_fatal_error(
        "detected double registration of compilation task \"" + mnemonic +
        "\"");
  taskNameRegistry->insert({mnemonic, typeID});
  taskRegistry->insert(
      std::make_pair(typeID, TaskRegistration{std::move(func)}));
}

//===----------------------------------------------------------------------===//
// Task Lookup Utilities
//===----------------------------------------------------------------------===//

llvm::SmallVector<llvm::StringRef>
compiler::getRegisteredCompilationTaskNames() {
  llvm::SmallVector<llvm::StringRef> result;
  for (const auto &[name, id] : *taskNameRegistry)
    result.push_back(name);
  return result;
}

void compiler::printCompilationTaskHelpInfo(mlir::MLIRContext *ctx,
                                            llvm::StringRef mnemonic) {
  StatusOr<std::unique_ptr<CompilerClient>> client =
      compiler::CompilerClient::create(ctx);
  if (!client.isOk())
    llvm::report_fatal_error(client.getString().c_str());
  StatusOr<CompilationTaskBase *> task =
      lookupAndBuildTask(**client, {}, mnemonic, /*enableDebugOptions=*/false);
  if (!task.isOk())
    llvm::report_fatal_error(task.getString().c_str());
  (*task)->getTaskOptions().printHelp(0, 70);
}

StatusOr<CompilationTaskBase *> compiler::buildTask(mlir::MLIRContext *ctx,
                                                    llvm::StringRef mnemonic,
                                                    llvm::StringRef options) {
  StatusOr<std::unique_ptr<CompilerClient>> client =
      compiler::CompilerClient::create(ctx);
  if (!client.isOk())
    return client.getStatus();
  return lookupAndBuildTask(**client, {}, mnemonic,
                            /*enableDebugOptions=*/false);
}