//===- Extension.h ----------------------------------------------*- C++ -*-===//
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
/// Declarations for the compiler extension mechanism. An extension is best
/// thought of as an "options extension". It should contains some data fields
/// which are filled out by an OptionsContext associated with a particular
/// extesible compilation task (e.g. 'StableHloToExecutableOptions'). The option
/// data fields are filled out when the OptionsContext parses the command
/// arguments. Besides the data fields, an extension provides hooks that
/// populate the PasManager at specific points (e.g. pre|post-clustering,
/// pre-post-bufferization, etc).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_EXTENSION
#define MLIR_TENSORRT_COMPILER_EXTENSION

#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Mutex.h"

namespace mlirtrt::compiler {

enum class Phase;

///===---------------------------------------------------------------------===//
// Task Extension
//===----------------------------------------------------------------------===//

/// Base class for all Compiler extensions.
class TaskExtensionBase {
public:
  /// Construct a new task extension instance.
  ///
  /// - taskName: The name of the compilation task this extension is
  ///   attached to.
  /// - extensionName: The unique name of this extension type.
  /// - ctx: The options context used by the task; extensions read their options
  ///   from this context.
  TaskExtensionBase(llvm::StringRef taskName, llvm::StringRef extensionName,
                    CompilationTaskOptionsBase &ctx)
      : taskName(taskName), extensionName(extensionName), ctx(ctx) {}

  /// Virtual destructor to allow proper cleanup via base pointer.
  virtual ~TaskExtensionBase();

  /// Hook invoked when the options have been parsed/finalized. Extensions can
  /// use this to perform changes to internal data based on the final options.
  virtual void onOptionsParsed() {}

  virtual void populatePasses(mlir::OpPassManager &pm, Phase phase) const = 0;

  /// Hook invoked immediately prior to running the MLIR compilation pipeline.
  /// The hook is given the module to be compiled.
  /// This should only be used to emit diagnostics and potentially abort
  /// compilation if for some reason the extension believes compilation of the
  /// module is not possible. Modifications to the module itself should be done
  /// through passes.
  virtual mlir::LogicalResult onBeforePipelineRun(mlir::ModuleOp module) const {
    return mlir::success();
  }

  /// Convenience alias to declare scalar options within extensions.
  template <typename T, typename... Mods>
  using Option = mlir::detail::PassOptions::Option<T, Mods...>;
  /// Convenience alias to declare list options within extensions.
  template <typename T, typename... Mods>
  using ListOption = mlir::detail::PassOptions::ListOption<T, Mods...>;

protected:
  /// Name of the compilation task this extension is associated with.
  llvm::StringRef taskName;
  /// Unique name of the extension type.
  llvm::StringRef extensionName;
  /// Reference to the options context driving this compilation task.
  CompilationTaskOptionsBase &ctx;
};

template <typename ExtensionType, typename TaskType>
class Extension : public TaskExtensionBase {
public:
  Extension(CompilationTaskOptionsBase &options)
      : TaskExtensionBase(TaskType::getName(), ExtensionType::getName(),
                          options) {}

  using OptionsType = TaskType::OptionsType;

  const OptionsType &getOptions() const {
    return static_cast<const OptionsType &>(this->ctx);
  }
};

/// An ExtensionList contains a set of extension constructor functions and a set
/// of constructed extension instances. This object is constructed from an
/// ExtensionRegistry and populated with constructors for all extensions
/// registered against a particular task. The extensions are lazily constructed
/// using "loadExtensions".
class ExtensionList {
public:
  /// Type of a factory function that constructs an extension given a task
  /// options context.
  using ConstructorFunc = std::function<std::unique_ptr<TaskExtensionBase>(
      CompilationTaskOptionsBase &)>;
  /// Mapping from extension name to constructed extension instance.
  using CompilerExtensionModules =
      llvm::StringMap<std::unique_ptr<TaskExtensionBase>>;
  /// Mapping from extension name to its constructor function.
  using ExtensionBuilders = llvm::StringMap<ConstructorFunc>;

  ExtensionList() = default;
  ExtensionList(ExtensionBuilders builders) : builders(std::move(builders)) {}

  /// Begin iterator over constructed extensions.
  CompilerExtensionModules::const_iterator begin() const {
    return extensions.begin();
  }
  /// End iterator over constructed extensions.
  CompilerExtensionModules::const_iterator end() const {
    return extensions.end();
  }

  /// Load all extensions that are not already loaded.
  void loadExtensions(CompilationTaskOptionsBase &task);

private:
  /// Storage for constructed extension instances.
  CompilerExtensionModules extensions;
  /// Storage for registered extension constructor functions.
  ExtensionBuilders builders;
};

/// An ExtensionConstructorRegistry is a mapping from CompilationTask kind to
/// constructor functions for each known TaskExtension associated with that
/// CompilationTask kind.
class ExtensionConstructorRegistry {
public:
  using ConstructorFunc = ExtensionList::ConstructorFunc;

  /// Register a constructor for an extension identified by task and extension
  /// names.
  void addExtension(llvm::StringRef taskName, llvm::StringRef extensionName,
                    ConstructorFunc constructor);

  /// Retrieve the registry of extensions that apply to the given task name.
  ExtensionList getExtensionsForTask(llvm::StringRef taskName) const;

private:
  /// Mapping: task name -> (extension name -> constructor function)
  llvm::StringMap<ExtensionList::ExtensionBuilders> constructors;

  /// Mutex to protect the registry from concurrent access.
  mutable llvm::sys::Mutex registryMutex;
};

/// Register an extension constructor for a given task and extension name with
/// the global Extension registry.
void registerExtension(llvm::StringRef taskName, llvm::StringRef extensionName,
                       ExtensionList::ConstructorFunc constructor);

/// Register an extension type for a specific task type using their static
/// names with the global Extension registry.
template <typename TaskType, typename ExtensionType>
void registerExtension() {
  registerExtension(TaskType::getName(), ExtensionType::getName(),
                    [](CompilationTaskOptionsBase &ctx)
                        -> std::unique_ptr<TaskExtensionBase> {
                      return std::make_unique<ExtensionType>(ctx);
                    });
}

/// Return a new registry of constructors for all extensions associated with the
/// given task name in the global Extension registry.
ExtensionList getExtensionsForTask(llvm::StringRef taskName);

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_EXTENSION
