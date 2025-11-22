//===- Client.h -------------------------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_COMPILER_CLIENT
#define MLIR_TENSORRT_COMPILER_CLIENT

#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>

namespace mtrt::compiler {

//===----------------------------------------------------------------------===//
// CompilationTask
//===----------------------------------------------------------------------===//

/// Base class for all "compilation tasks". A "compilation task" is like a MLIR
/// PassManager, but it has some additional functionality. Primarily, it is
/// associated with a CompilationOptions object which should capture all options
/// that may modify the behavior of the compiler pipeline and a list of
/// extensions. Each concrete subclass type is associated with a unique name
/// (static string literal ID).
///
/// Extensions are loaded from a global registry and will update the owned
/// options object. It is up to the derived class to actually invoke the
/// extension hooks for populating passes inside `populatePassManager`.
///
/// A CompilationTaskBase should never be constructed directly. Instead, tasks
/// are constructed in a "default" state through the CompilerClient API. They
/// are then lazily initialized using the `initialize` method, which loads
/// extensions and populates the passes.
///
/// Concrete subclasses should implement `populatePassManager`. Subclasses
/// should be implemented by deriving from the `CompilationTask` CRTP template
/// rather than directly inheriting from this class.
///
/// The additional methods `openOutputFile` and `translateToTargetFormat` are
/// provided to simplify implementation of the final compilation steps.
class CompilationTaskBase : public mlir::PassManager {
public:
  virtual ~CompilationTaskBase();

  /// Initialize the CompilationTask using the given command line options.
  /// This should be called after all extensions have been constructed.
  /// It can only be called once, otherwise it will return an error.
  /// The `overrideArtifactsDir` parameter can be used to specify an artifacts
  /// directory and will override the `--artifacts-dir` flag that may also
  /// be provided through `options`. The purpose of this is to simplify the
  /// implementation of tools like `mlir-tensorrt-compiler`.
  Status initialize(llvm::ArrayRef<llvm::StringRef> options,
                    std::optional<llvm::StringRef> overrideArtifactsDir);

  /// Return the options for the task.
  const CompilationTaskOptionsBase &getTaskOptions() const {
    return *taskOptions;
  }

  /// Return the (mutable) options for the task.
  CompilationTaskOptionsBase &getTaskOptions() { return *taskOptions; }

  /// Open an output file with the given name. If the file cannot be opened,
  /// then the error message is passed through `errorMessage`. If
  /// `outputFileName` is `-`, then the output file corresponds to stdout. If
  /// the `outputFileName` is an absolute path, then it that file is opened or
  /// created. If it is a relative path, then it is appended to the artifacts
  /// directory path if the artifacts directory is set and exists. Otherwise,
  /// the output file is opened relative to the current directory. Any existing
  /// file is overwritten.
  std::unique_ptr<llvm::ToolOutputFile>
  openOutputFile(llvm::StringRef outputFileName, std::string &errorMessage,
                 std::optional<llvm::StringRef> overrideExtension = {});

  /// Translate to the final target format.
  mlir::LogicalResult translateToTargetFormat(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

protected:
  CompilationTaskBase(llvm::StringRef taskName, mlir::MLIRContext *context,
                      std::unique_ptr<CompilationTaskOptionsBase> options);

  /// Populate the pass manager with the appropriate passes. This should be
  /// implemented by concrete subclasses. The pass manager is empty when this
  /// method is called, and it will only be called once.
  virtual void populatePassManager() = 0;

  /// Populate pass manager instrumentation (e.g. dumping IR after passes,
  /// timing, debug actions, etc) based on the given options. If the
  /// DebugOptions is nullptr, then the instrumentation and timing are populated
  /// from global CL options.
  void setupPassManagerInstrumentation(const DebugOptions *options);

  const llvm::StringRef name;

  /// Options for the task.
  std::unique_ptr<CompilationTaskOptionsBase> taskOptions;

  /// The Extensions associated with this task.
  ExtensionList extensions;

  /// A flag to indicate whether the task has been fully initialized.
  bool initialized{false};
};

/// CRTP base class for compilation tasks.
template <typename DerivedTaskT, typename OptionsT>
class CompilationTask : public CompilationTaskBase {
public:
  using OptionsType = OptionsT;

  CompilationTask(mlir::MLIRContext *context, std::unique_ptr<OptionsT> options)
      : CompilationTaskBase(DerivedTaskT::getName(), context,
                            std::move(options)) {}

  const OptionsT &getOptions() {
    return static_cast<const OptionsT &>(*this->taskOptions);
  }

  using Base = CompilationTask;
};

/// Phase describes the overall phases of the compilation pipeline. Not all
/// phases are applicable to all tasks.
enum class Phase {
  ConstantFolding,
  PreClustering,
  PostClustering,
  PreBufferization,
  PostBufferization,
  ExecutorLowering
};

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

/// C++ users of the MLIR-TensorRT Compiler API should create a CompilerClient
/// once for each MLIRContext that is in use. Currently it is recommended
/// to instantiate different clients for different MLIRContexts and to only
/// associate a single client with a particular MLIRContext.
///
/// The MLIRContext is only referenced, so the lifetime must outlive the Client
/// object.
///
/// The CompilerClient provides an API to construct CompilationTasks from
/// the mnemonic and pass pipeline options. The CompilerClient assumes ownership
/// of the pipeline and subsequent queries using the same options will return
/// the same CompilationTaskBase object.
///
/// TODO: We should remove the caching mechanism; that should be the
/// responsibility of the downstream user.
class CompilerClient {
public:
  static StatusOr<std::unique_ptr<CompilerClient>>
  create(mlir::MLIRContext *context);

  ~CompilerClient() = default;

  /// Create or retrieve from the cache a compilation task of the specified
  /// type and options. If an existing compilation task is not in the cache,
  /// then it is constructed using the registered construction function and
  /// inserted into the cache.
  StatusOr<CompilationTaskBase *>
  getCompilationTask(llvm::StringRef taskName,
                     llvm::ArrayRef<llvm::StringRef> options,
                     std::optional<llvm::StringRef> overrideArtifactsDir = {},
                     bool enableDebugOptions = false);

  /// Insert a compilation task of type T with options hash `hash` into the
  /// cache.
  void updateCachedCompilationTask(llvm::StringRef taskName,
                                   const llvm::hash_code &hash,
                                   std::unique_ptr<CompilationTaskBase> task);

  /// Check whether a CompilationTask with the specified task type and whose
  /// options have the given hash is in the cache. If so, return it; otherwise
  /// returns nullptr.
  CompilationTaskBase *
  lookupCachedCompilationTask(llvm::StringRef taskName,
                              const llvm::hash_code &optionsHash) const;

  /// Return the MLIRContext associated with the client.
  mlir::MLIRContext *getContext() const { return context; }

protected:
  CompilerClient(mlir::MLIRContext *context);

  /// The MLIRContext in use by this client.
  mlir::MLIRContext *context;

  /// A registry of pass managers for specific kinds of tasks. The map is
  /// indexed by the task name and the hash of the options
  /// used to create the PM.
  llvm::StringMap<
      llvm::DenseMap<llvm::hash_code, std::unique_ptr<CompilationTaskBase>>>
      cachedPassManagers;
};

/// A registry function that adds passes to the given pass manager. This should
/// also parse options and return success() if parsing succeeded.
/// `errorHandler` is a functor used to emit errors during parsing.
/// parameter corresponds to the raw location within the pipeline string. This
/// should always return failure.
using TaskRegistryFunction = std::function<StatusOr<CompilationTaskBase *>(
    CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options,
    std::optional<llvm::StringRef> overrideArtifactsDir,
    bool enableDebugOptions)>;

struct TaskRegistration {
  TaskRegistryFunction registryFunc;
};

///===----------------------------------------------------------------------===//
// Task Lookup Utilities
//===----------------------------------------------------------------------===//

/// Returns a list of registered compilation task names.
llvm::SmallVector<llvm::StringRef> getRegisteredCompilationTaskNames();

/// For the given task, prints a CLI "--help"-type description to stdout
/// that describes each option associated with the task. If the task is not
/// registered, a fatal error is issued.
void printCompilationTaskHelpInfo(mlir::MLIRContext *ctx,
                                  llvm::StringRef mnemonic);

//===----------------------------------------------------------------------===//
// Task Registration Utilities
//===----------------------------------------------------------------------===//

namespace detail {
/// Register task given the mnemonic and the options registration function.
void registerCompilationTask(llvm::StringRef mnemonic,
                             TaskRegistryFunction func);
} // namespace detail

/// Register a task by providing an explicit registration function for the given
/// options type.
template <typename T>
void registerCompilationTask(TaskRegistryFunction func) {
  return detail::registerCompilationTask(T::getName(), std::move(func));
}

/// This helper provides a convenience registration wrapper for most tasks whose
/// options can be constructed from a single boolean (`enableDebugOptions`) and
/// do not have associated extensions.
template <typename T, typename OptionsType>
void registerCompilationTaskWithNoExtensions(llvm::StringRef mnemonic) {
  registerCompilationTask<T>(
      [](CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options,
         std::optional<llvm::StringRef> overrideArtifactsDir,
         bool enableDebugOptions) -> StatusOr<CompilationTaskBase *> {
        // Hash the flags directly.
        llvm::hash_code hashCode =
            llvm::hash_combine_range(options.begin(), options.end());

        // Check the cache.
        CompilationTaskBase *cached =
            client.lookupCachedCompilationTask(T::getName(), hashCode);
        if (cached)
          return cached;

        /// Construct "uninitialized" options, which are just default options
        /// prior to updating the value from the command line flags.
        auto uninitOptions = std::make_unique<OptionsType>(enableDebugOptions);

        // No cached task, so construct a new task.
        auto newPM =
            std::make_unique<T>(client.getContext(), std::move(uninitOptions));

        // Invoke the initialization.
        if (Status s = newPM->initialize(options, overrideArtifactsDir);
            !s.isOk())
          return s;

        // Give ownership to the client.
        auto ptr = newPM.get();
        client.updateCachedCompilationTask(T::getName(), hashCode,
                                           std::move(newPM));
        return ptr;
      });
}

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_CLIENT
