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

#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// CompilationTask
//===----------------------------------------------------------------------===//

/// Base class for all compilation tasks. CompilationTasks have associated
/// unique TypeID. A CompilationTask is also a PassManager and has all the same
/// methods. However, the CompilationTask's constructor is meant to construct to
/// implement a function that maps a specific Options instance to a set of
/// Passes that accomplish the task with the options. After construction,
/// CompilationTask should be considered "frozen".
/// TODO: consider moving this upstream so that we can properly implement the
/// "freezing" API and disallow further modifications to the pipeline.
class CompilationTaskBase : public mlir::PassManager {
public:
  CompilationTaskBase(mlir::MLIRContext *context, mlir::TypeID typeID);

  virtual ~CompilationTaskBase();

  mlir::TypeID getTypeID() const { return typeID; }

private:
  mlir::TypeID typeID;
};

/// CRTP base class for compilation tasks. The derived classes must define
/// `populatePassManager` and use appropriate macros to define their unique
/// TypeID.
template <typename DerivedTaskT, typename OptionsT>
class CompilationTask : public CompilationTaskBase {
public:
  CompilationTask(mlir::MLIRContext *context, const OptionsT &options)
      : CompilationTaskBase(context, mlir::TypeID::get<DerivedTaskT>()) {
    DerivedTaskT::populatePassManager(*this, options);
  }

  using Base = CompilationTask;

  static bool classof(const CompilationTaskBase *base) {
    return base->getTypeID() == mlir::TypeID::get<DerivedTaskT>();
  }
};

//===----------------------------------------------------------------------===//
// CompilerClient
//===----------------------------------------------------------------------===//

/// C++ users of the MLIR-TensorRT Compiler API should create a CompilerClient
/// once for each process or thread that will be performing concurrent
/// compilation work. The CompilerClient holds long-lived resources such as the
/// MLIRContext and a TensorRT builder cache. Clients don't share resources
/// since concurrent access could create issues, and currently we prefer to
/// avoid cross-client locks. This means that separate clients have separate
/// TensorRT builder caches and should be initialized with unique paths if the
/// builder caches are being persisted to disk.
class CompilerClient {
public:
  static StatusOr<std::unique_ptr<CompilerClient>>
  create(mlir::MLIRContext *context);

  ~CompilerClient() = default;

  /// Create or retrieve a cached PassManager of the given derived type using
  /// the provided options. PassManagers are cached by type and a hash of the
  /// string representation of the options.
  /// This function should only be called if the options have a valid hash.
  template <typename CompilationTaskType, typename OptionsType>
  mlir::PassManager &getOrCreatePassManager(const OptionsType &options) {
    std::optional<llvm::hash_code> hash = options.getHash();
    if (!hash)
      llvm::report_fatal_error("attempted to lookup a PassManager from a cache "
                               "with an un-hashable options key");

    auto key =
        std::make_pair(mlir::TypeID::get<CompilationTaskType>(), hash.value());
    auto it = cachedPassManagers.find(key);
    if (it == cachedPassManagers.end()) {
      auto pm = std::make_unique<CompilationTaskType>(context, options);
      setupPassManagerLogging(*pm, options.debugOptions);
      auto *ptr = pm.get();
      cachedPassManagers[key] = std::move(pm);
      return *ptr;
    }
    return *it->second;
  }

  /// Return the MLIRContext associated with the client.
  mlir::MLIRContext *getContext() const { return context; }

  /// Helper for setting the correct logging options on cached PassManagers.
  static void setupPassManagerLogging(mlir::PassManager &pm,
                                      const DebugOptions &options);

protected:
  CompilerClient(mlir::MLIRContext *context);

  /// The MLIRContext in use by this client.
  mlir::MLIRContext *context;

  /// Key pair of [PassManager Kind, Options Hash].
  using PassManagerKey = std::pair<mlir::TypeID, llvm::hash_code>;

  /// A registry of pass managers for specific kinds of tasks. The map is
  /// indexed by the TypeID of the PassManager kind and the hash of the options
  /// used to create the PM.
  llvm::DenseMap<PassManagerKey, std::unique_ptr<CompilationTaskBase>>
      cachedPassManagers;
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_CLIENT
