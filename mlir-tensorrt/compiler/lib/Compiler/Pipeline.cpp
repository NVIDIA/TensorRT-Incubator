//===- Pipeline.cpp -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation for the Pipeline.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//
void PipelineBase::setupPassManagerInstrumentation(
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

PipelineBase::PipelineBase(llvm::StringRef taskName, MLIRContext *context,
                           std::unique_ptr<PipelineOptionsBase> options)
    : mlir::PassManager(context, mlir::ModuleOp::getOperationName()),
      name(taskName), options(std::move(options)),
      extensions(compiler::getExtensionsForTask(taskName)) {}

PipelineBase::~PipelineBase() {}

Status PipelineBase::initialize(llvm::ArrayRef<llvm::StringRef> options) {
  if (initialized)
    return getInternalErrorStatus("attempted to reinitialize a pipeline");

  assert(this->getPasses().empty() && "expected empty pass manager");

  // Load all extensions so that their OptionsProvider(s) are connected to the
  // PipelineOptions.
  extensions.loadExtensions(getOptions());

  // Update option values based on the provided command line strings.
  std::string err;
  if (failed(getOptions().parse(options, err)))
    return getInvalidArgStatus("failed to parse options \"{0:$[ ]}\": {1}",
                               llvm::iterator_range(options), err);

  // Run the onOptionsParsed hook for all extensions.
  for (const auto &extension : extensions)
    extension.getValue()->onOptionsParsed();

  // Now that options are parsed, we can setup instrumentation.
  setupPassManagerInstrumentation(getOptions().getDebugOptions());

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

std::unique_ptr<llvm::ToolOutputFile>
PipelineBase::openOutputFile(llvm::StringRef outputFileName,
                             std::string &errorMessage,
                             std::optional<llvm::StringRef> overrideExtension) {
  if (!initialized) {
    errorMessage = "attempted to open an output file before initialization";
    return nullptr;
  }
  std::string processedOutputName = createMainOutputFileName(
      outputFileName, options->hostTarget, overrideExtension);
  if (outputFileName != "-" && !options->artifactsDirectory.empty() &&
      // Only try to combine if we are sure that the artifacts directory is
      // present. Sometimes it is unused. Note: the artifacts directory is not
      // automatically created by the Pipeline; it must already exist.
      llvm::sys::fs::is_directory(options->artifactsDirectory) &&
      // This is an absolute path - don't combine.
      !llvm::sys::path::is_absolute(processedOutputName) &&
      // This is relative, but it has a parent path - don't combine.
      !llvm::sys::path::has_parent_path(processedOutputName)) {
    llvm::SmallString<128> path;
    llvm::sys::path::append(path, options->artifactsDirectory,
                            processedOutputName);
    return mlir::openOutputFile(path, &errorMessage);
  }
  return mlir::openOutputFile(processedOutputName, &errorMessage);
}

LogicalResult PipelineBase::translateToTargetFormat(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  const HostTarget hostTarget = getOptions().hostTarget;
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
