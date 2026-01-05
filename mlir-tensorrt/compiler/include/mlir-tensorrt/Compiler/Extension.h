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
/// which are filled out by an OptionsContext associated with a pipeline. The
/// option data fields are filled out when the pipeline options parses the
/// command arguments. Besides the data fields, an extension provides hooks that
/// populate the PassManager at specific points (e.g. pre|post-clustering,
/// pre|post-bufferization, etc).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_EXTENSION
#define MLIR_TENSORRT_COMPILER_EXTENSION

#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Mutex.h"

namespace mtrt::compiler {

enum class ExtensionPoint;

///===---------------------------------------------------------------------===//
// ExtensionBase
//===----------------------------------------------------------------------===//

/// Base class for all Compiler extensions.
class ExtensionBase {
public:
  /// Construct a new extension instance.
  ///
  /// - extensionName: The unique name of this extension type.
  /// - ctx: The options context used by the pipeline; extensions read their
  ///   options from this context.
  ExtensionBase(llvm::StringRef extensionName, mlir::CLOptionScope &ctx)
      : extensionName(extensionName), ctx(ctx) {}

  /// Virtual destructor to allow proper cleanup via base pointer.
  virtual ~ExtensionBase();

  virtual void populatePasses(mlir::OpPassManager &pm,
                              ExtensionPoint point) const = 0;

  /// Convenience alias to declare scalar options within extensions.
  template <typename T, typename... Mods>
  using Option = mlir::CLOptionScope::Option<T, Mods...>;
  /// Convenience alias to declare list options within extensions.
  template <typename T, typename... Mods>
  using ListOption = mlir::CLOptionScope::ListOption<T, Mods...>;

protected:
  /// Unique name of the extension type.
  llvm::StringRef extensionName;
  /// Reference to the options context driving the compilation pipeline.
  mlir::CLOptionScope &ctx;
};

template <typename ExtensionType, typename BaseOptionsType>
class Extension : public ExtensionBase {
public:
  Extension(mlir::CLOptionScope &options)
      : ExtensionBase(ExtensionType::getName(), options) {}

  const BaseOptionsType &getOptions() const {
    return static_cast<const BaseOptionsType &>(this->ctx);
  }
};

/// An ExtensionList contains a set of extension constructor functions and a set
/// of constructed extension instances. This object is constructed from an
/// ExtensionRegistry and populated with constructors for all extensions
/// registered with the global registry. The extensions are lazily constructed
/// using "loadExtensions".
class ExtensionList {
public:
  /// Type of a factory function that constructs an extension given a task
  /// options context.
  using ConstructorFunc =
      std::function<std::unique_ptr<ExtensionBase>(mlir::CLOptionScope &)>;
  /// Mapping from extension name to constructed extension instance.
  using CompilerExtensionModules =
      llvm::StringMap<std::unique_ptr<ExtensionBase>>;
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
  void loadExtensions(mlir::CLOptionScope &task);

private:
  /// Storage for constructed extension instances.
  CompilerExtensionModules extensions;
  /// Storage for registered extension constructor functions.
  ExtensionBuilders builders;
};

/// An ExtensionConstructorRegistry maps extension names to constructor
/// functions for each known TaskExtension.
class ExtensionConstructorRegistry {
public:
  using ConstructorFunc = ExtensionList::ConstructorFunc;

  /// Register a constructor for an extension identified by extension name.
  void addExtension(llvm::StringRef extensionName, ConstructorFunc constructor);

  /// Retrieve the registry of all registered extensions.
  ExtensionList getAllExtensions() const;

private:
  /// Mapping: extension name -> constructor function.
  ExtensionList::ExtensionBuilders constructors;

  /// Mutex to protect the registry from concurrent access.
  mutable llvm::sys::Mutex registryMutex;
};

/// Register an extension constructor for an extension name with the global
/// Extension registry.
void registerExtension(llvm::StringRef extensionName,
                       ExtensionList::ConstructorFunc constructor);

/// Register an extension type using its static name with the global Extension
/// registry.
template <typename ExtensionType>
void registerExtension() {
  registerExtension(
      ExtensionType::getName(),
      [](mlir::CLOptionScope &ctx) -> std::unique_ptr<ExtensionBase> {
        return std::make_unique<ExtensionType>(ctx);
      });
}

/// Return a new registry of constructors for all registered extensions in the
/// global Extension registry.
ExtensionList getAllExtensions();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_EXTENSION
