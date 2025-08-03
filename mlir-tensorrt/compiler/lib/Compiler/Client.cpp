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
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#define DEBUG_TYPE "compiler-api"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]")

static llvm::ManagedStatic<llvm::StringMap<TaskRegistration>> taskRegistry{};

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
    llvm::StringRef taskName, MLIRContext *context,
    std::unique_ptr<CompilationTaskOptionsBase> options)
    : mlir::PassManager(context, mlir::ModuleOp::getOperationName()),
      name(taskName), taskOptions(std::move(options)),
      extensions(compiler::getExtensionsForTask(taskName)) {}

CompilationTaskBase::~CompilationTaskBase() {}

/// Create all directories in `path`, ignoring those that already exist.
/// Failure to create a directory results in emitting a diagnostic and
/// returning failure.
static llvm::Error createDirectories(StringRef path) {
  if (path.empty())
    return llvm::Error::success();
  if (std::error_code EC =
          llvm::sys::fs::create_directories(path, /*IgnoreExisting=*/true))
    return llvm::createStringError(
        EC, "failed to create directories for path \"" + path + "\"");
  return llvm::Error::success();
}

Status CompilationTaskBase::initialize(
    llvm::ArrayRef<llvm::StringRef> options,
    std::optional<llvm::StringRef> overrideOutputPath) {
  if (initialized)
    return getInternalErrorStatus(
        "attempted to reinitialize a compilation task");

  assert(this->getPasses().empty() && "expected empty pass manager");

  // Load all extensions so that there OptionsProvider(s) are connected to the
  // CompilationTaskOptions.
  extensions.loadExtensions(getTaskOptions());

  // Update option values based on the provided command line strings.
  std::string err;
  if (failed(getTaskOptions().parse(options, err)))
    return getInvalidArgStatus("failed to parse options \"{0:$[ ]}\": {1}",
                               llvm::iterator_range(options), err);

  // Run the onOptionsParsed hook for all extensions.
  for (const auto &extension : extensions)
    extension.getValue()->onOptionsParsed();

  // Now that options are parsed, we can setup instrumentation.
  setupPassManagerInstrumentation(getTaskOptions().getDebugOptions());

  // If an output directory is provided, then update the corresponding options
  // value. The option is used to populate pass manager options.
  // TODO: eliminate this when we have better artifact management.
  if (overrideOutputPath) {
    getTaskOptions().artifactsDirectory = overrideOutputPath->str();
    if (llvm::Error err =
            createDirectories(getTaskOptions().artifactsDirectory))
      return getInternalErrorStatus("{0}", llvm::fmt_consume(std::move(err)));
  }

  // Now we can populate the pass manager.
  populatePassManager();

  initialized = true;

  return getOkStatus();
}

/// When the output path is a directory or empty, this function is used to
/// create the output file name with an extension guessed from the host target.
static std::string
createMainOutputFileName(llvm::StringRef initialName, HostTarget hostTarget,
                         std::optional<llvm::StringRef> overrideExtension) {
  initialName = initialName.trim();
  if (!initialName.empty() &&
      (initialName == "-" || !llvm::sys::fs::is_directory(initialName)))
    return initialName.str();

  llvm::SmallString<128> result = initialName;
  std::string name = "output";
  if (overrideExtension)
    name += overrideExtension->str();
  else if (hostTarget == HostTarget::EmitC)
    name += ".cpp";
  else if (hostTarget == HostTarget::LLVM)
    name += ".mlir";
  else if (hostTarget == HostTarget::Executor)
    name += ".rtexe";
  else
    llvm_unreachable("unknown output kind");

  llvm::sys::path::append(result, name);
  return result.str().str();
}

std::unique_ptr<llvm::ToolOutputFile> CompilationTaskBase::openOutputFile(
    llvm::StringRef outputFileName, std::string &errorMessage,
    std::optional<llvm::StringRef> overrideExtension) {
  if (!initialized) {
    errorMessage = "attempted to open an output file before initialization";
    return nullptr;
  }
  std::string processedOutputName = createMainOutputFileName(
      outputFileName, taskOptions->hostTarget, overrideExtension);
  if (outputFileName != "-" && !taskOptions->artifactsDirectory.empty() &&
      // Only try to combine if we are sure that the artifacts directory is
      // present. Sometimes it is unused. If it is populated, then it should
      // have been created in the 'initialize' method.
      llvm::sys::fs::is_directory(taskOptions->artifactsDirectory) &&
      // This is an absolute path - don't combine.
      !llvm::sys::path::is_absolute(processedOutputName) &&
      // This is relative, but it has a parent path - don't combine.
      !llvm::sys::path::has_parent_path(processedOutputName)) {
    llvm::SmallString<128> path;
    llvm::sys::path::append(path, taskOptions->artifactsDirectory,
                            processedOutputName);
    return mlir::openOutputFile(path, &errorMessage);
  }
  return mlir::openOutputFile(processedOutputName, &errorMessage);
}

LogicalResult
CompilationTaskBase::translateToTargetFormat(mlir::ModuleOp module,
                                             llvm::raw_ostream &os) {
  const HostTarget hostTarget = getTaskOptions().hostTarget;
  if (hostTarget == HostTarget::EmitC) {
    if (failed(emitc::translateToCpp(module, os)))
      return failure();
    return success();
  }

  if (hostTarget == HostTarget::LLVM) {
    module->print(os);
    return success();
  }

  if (hostTarget == HostTarget::Executor) {
    if (failed(mlir::translateToRuntimeExecutable(module, os)))
      return failure();
    return success();
  }

  return emitError(module->getLoc()) << "unknown host target format";
}

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<CompilerClient>>
CompilerClient::create(MLIRContext *context) {
  return std::unique_ptr<CompilerClient>(new CompilerClient(context));
}

CompilerClient::CompilerClient(mlir::MLIRContext *context) : context(context) {}

StatusOr<CompilationTaskBase *> CompilerClient::getCompilationTask(
    llvm::StringRef mnemonic, llvm::ArrayRef<llvm::StringRef> options,
    std::optional<llvm::StringRef> overrideArtifactsDir,
    bool enableDebugOptions) {
  if (!taskRegistry.isConstructed())
    return getInvalidArgStatus("no compilation task registered with name {0}",
                               mnemonic);
  auto it = taskRegistry->find(mnemonic);
  if (it == taskRegistry->end())
    return getInvalidArgStatus("no compilation task registered with name {0}",
                               mnemonic);
  return it->second.registryFunc(*this, options, overrideArtifactsDir,
                                 enableDebugOptions);
}

void CompilerClient::updateCachedCompilationTask(
    llvm::StringRef taskName, const llvm::hash_code &hash,
    std::unique_ptr<CompilationTaskBase> task) {
  cachedPassManagers[taskName][hash] = std::move(task);
}

CompilationTaskBase *CompilerClient::lookupCachedCompilationTask(
    llvm::StringRef taskName, const llvm::hash_code &optionsHash) const {
  auto it = cachedPassManagers.find(taskName);
  if (it == cachedPassManagers.end())
    return nullptr;
  auto it2 = it->second.find(optionsHash);
  if (it2 == it->second.end())
    return nullptr;
  return it2->second.get();
}

void compiler::detail::registerCompilationTask(llvm::StringRef taskName,
                                               TaskRegistryFunction func) {
  if (taskRegistry->contains(taskName))
    llvm::report_fatal_error(
        "detected double registration of compilation task \"" + taskName +
        "\"");

  taskRegistry->insert({taskName, TaskRegistration{std::move(func)}});
}

//===----------------------------------------------------------------------===//
// Task Lookup Utilities
//===----------------------------------------------------------------------===//

llvm::SmallVector<llvm::StringRef>
compiler::getRegisteredCompilationTaskNames() {
  llvm::SmallVector<llvm::StringRef> result;
  for (const auto &[name, registration] : *taskRegistry)
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
      (*client)->getCompilationTask(mnemonic, {});
  if (!task.isOk())
    llvm::report_fatal_error(task.getString().c_str());
  assert(*task != nullptr && "expected a valid task");
  (*task)->getTaskOptions().printHelp(0, 70);
}
