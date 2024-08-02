//===- Extension.h ----------------------------------------------*- C++ -*-===//
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

#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir/Support/TypeID.h"

namespace mlirtrt::compiler {

///===---------------------------------------------------------------------===//
// Task Extension
//===----------------------------------------------------------------------===//

/// Base class for all Compiler extensions. An extension must be given a TypeID
/// through the macros 'MLIR_(DECLARE|DEFINE)_EXPLICIT_TYPE_ID'.
/// Tasks Extensions have two TypeIDs: 1) their own unique TypeID and 2)
/// the type ID of the task to which they are associated.
class TaskExtensionBase {
public:
  TaskExtensionBase(mlir::TypeID typeID, mlir::TypeID taskID)
      : typeID(typeID), taskID(taskID) {}

  virtual ~TaskExtensionBase();

  mlir::TypeID getTypeID() const { return typeID; }

  /// Return the TypeID for the CompilationTask this extension is associated
  /// with.
  mlir::TypeID getTaskID() const { return taskID; }

  /// Retrieve the human-readable name for the extension. This is used for
  /// logging/debugging.
  virtual llvm::StringRef getName() const = 0;

  /// Hook invoked when compilation of a module has finished. This can be used
  /// to update persistent resources on disk (e.g. cache databases, statistic
  /// info, etc).
  virtual Status onCompilationFinished() { return Status::getOk(); }

  /// Allows the extension to hook into the option parsing infrastructure.
  virtual void addToOptions(mlir::OptionsContext &context) = 0;

protected:
  /// Whether this extension is disabled. Should default to false and be
  /// associated with a flag `--disable-[name]-extension`.
  bool disabled{false};

private:
  mlir::TypeID typeID;

  /// TypeID for the task this extension is associated with.
  mlir::TypeID taskID;
};

/// An extension registry is just a list of extensions, associated with one
/// particular task.
class TaskExtensionRegistry {
public:
  using CompilerExtensionModules =
      llvm::DenseMap<mlir::TypeID, std::unique_ptr<TaskExtensionBase>>;

  /// Return an instance of the specified extension if it is in the registry,
  /// otherwise nulloptr.
  template <typename T>
  T *getExtension() {
    return llvm::dyn_cast_if_present<T>(
        extensions[mlir::TypeID::get<T>()].get());
  }

  /// Convenience method for creating a specific instance of the specified
  /// extension type if it is not already in the registry. Must be
  /// default-constructable.
  template <typename T>
  T *getOrCreateExtension() {
    if (!extensions.contains(mlir::TypeID::get<T>()))
      extensions[mlir::TypeID::get<T>()] = std::make_unique<T>();
    return getExtension<T>();
  }

  CompilerExtensionModules::const_iterator begin() const {
    return extensions.begin();
  }
  CompilerExtensionModules::const_iterator end() const {
    return extensions.end();
  }

  CompilerExtensionModules extensions;
};

/// An ExtensionConstructorRegistry is a mapping from CompilationTask kind to
/// constructor functionss for each known TaskExtension associated with that
/// CompilationTask kind.
class ExtensionConstructorRegistry {
public:
  using ConstructorFunc = std::function<std::unique_ptr<TaskExtensionBase>()>;

  /// Invoke the constructors for all extensions associated with a given task
  /// and return the newly created extensions as a TaskExtensionRegistry.
  template <typename Task>
  TaskExtensionRegistry getExtensionRegistryForTask() const {
    const mlir::TypeID taskID = mlir::TypeID::get<Task>();
    TaskExtensionRegistry registry;
    if (!constructors.contains(taskID))
      return registry;
    for (auto [key, extConst] : constructors.lookup(taskID))
      registry.extensions.insert(std::make_pair(key, extConst()));
    return registry;
  }

  template <typename Task, typename ConcreteExtensionType>
  mlir::LogicalResult addCheckedExtensionConstructor() {
    auto constructor = []() -> std::unique_ptr<TaskExtensionBase> {
      return std::make_unique<ConcreteExtensionType>();
    };
    auto taskID = mlir::TypeID::get<Task>();
    auto extID = mlir::TypeID::get<ConcreteExtensionType>();
    if (!constructors.contains(taskID)) {
      llvm::SmallDenseMap<mlir::TypeID, ConstructorFunc> inner = {
          {mlir::TypeID::get<ConcreteExtensionType>(), std::move(constructor)}};
      constructors.insert(
          std::make_pair(mlir::TypeID::get<Task>(), std::move(inner)));
      return mlir::success();
    }
    if (constructors[taskID].contains(extID))
      return mlir::failure();
    constructors[taskID].insert(std::make_pair(extID, std::move(constructor)));
    return mlir::success();
  }

private:
  llvm::SmallDenseMap<
      mlir::TypeID,
      llvm::SmallDenseMap<mlir::TypeID,
                          std::function<std::unique_ptr<TaskExtensionBase>()>>>
      constructors;
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_EXTENSION
