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
#include "mlir-executor/Support/Status.h"
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
CompilationTaskBase::CompilationTaskBase(MLIRContext *context,
                                         mlir::TypeID typeID)
    : mlir::PassManager(context, mlir::ModuleOp::getOperationName()),
      typeID(typeID) {}

CompilationTaskBase::~CompilationTaskBase() {}

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<CompilerClient>>
CompilerClient::create(MLIRContext *context) {
  context->disableMultithreading();
  return std::unique_ptr<CompilerClient>(new CompilerClient(context));
}

CompilerClient::CompilerClient(mlir::MLIRContext *context) : context(context) {}

void CompilerClient::setupPassManagerLogging(mlir::PassManager &pm,
                                             const DebugOptions &options) {
  pm.enableVerifier(true);
  if (!options.dumpIRPath.empty()) {
    // Enable IR printing after passes run. We can expand the debug options to
    // match MLIR's global CL options if needed in the future.
    pm.enableIRPrintingToFileTree(
        [](mlir::Pass *, mlir::Operation *) { return true; },
        [](mlir::Pass *, mlir::Operation *) { return false; }, true, false,
        false, options.dumpIRPath,
        mlir::OpPrintingFlags().elideLargeElementsAttrs(32));
  }
}

StatusOr<CompilationTaskBase *>
CompilerClient::getCompilationTask(mlir::TypeID taskID,
                                   llvm::ArrayRef<llvm::StringRef> options) {
  auto it = taskRegistry->find(taskID);
  if (it == taskRegistry->end())
    llvm::report_fatal_error("no such task registered");
  return it->second.registryFunc(*this, options);
}

StatusOr<CompilationTaskBase *>
CompilerClient::getCompilationTask(llvm::StringRef mnemonic,
                                   llvm::ArrayRef<StringRef> options) {
  auto it = taskNameRegistry->find(mnemonic);
  if (it == taskNameRegistry->end())
    return getInvalidArgStatus("no compilation task registered with name {0}",
                               mnemonic);

  return getCompilationTask(taskNameRegistry->lookup(mnemonic), options);
}

void compiler::registerCompilationTask(llvm::StringRef mnemonic,
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
