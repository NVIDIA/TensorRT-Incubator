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
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir-tensorrt/Compiler/OptionsRegistry.h"
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

  /// Create or retrieve from the cache a compilation task of the specified
  /// type and options. If an existing compilation task is not in the cache,
  /// then it is constructed using the registered construction function and
  /// inserted into the cache.
  StatusOr<CompilationTaskBase *>
  getCompilationTask(mlir::TypeID taskID,
                     llvm::ArrayRef<llvm::StringRef> options);

  /// Create or retrieve from the cache a compilation task of the specified
  /// type ID and options. If an existing compilation task is not in the cache,
  /// then it is constructed using the registered construction function and
  /// inserted into the cache.
  StatusOr<CompilationTaskBase *>
  getCompilationTask(mlir::TypeID taskID, llvm::ArrayRef<std::string> options) {
    return getCompilationTask(
        taskID, llvm::map_to_vector(options, [](const std::string &x) {
          return llvm::StringRef(x);
        }));
  }

  StatusOr<CompilationTaskBase *>
  getCompilationTask(llvm::StringRef mnemonic,
                     llvm::ArrayRef<llvm::StringRef> options);

  /// Create or retrieve from the cache a compilation task of the specified
  /// type and options. If an existing compilation task is not in the cache,
  /// then it is constructed using the registered construction function and
  /// inserted into the cache.
  template <typename T, typename... Args>
  StatusOr<CompilationTaskBase *> getCompilationTask(Args &&...args) {
    return getCompilationTask(mlir::TypeID::get<T>(),
                              std::forward<Args>(args)...);
  }

  /// Insert a compilation task of type T with options hash `hash` into the
  /// cache.
  template <typename T>
  void updateCachedCompilationTask(const llvm::hash_code &hash,
                                   std::unique_ptr<CompilationTaskBase> task) {
    cachedPassManagers[std::make_pair(mlir::TypeID::get<T>(), hash)] =
        std::move(task);
  }

  /// Check whether a CompilationTask with the specified typeID and whose
  /// options have the given hash is in the cache. If so, return it; otherwise
  /// returns nullptr.
  CompilationTaskBase *
  lookupCachedCompilationTask(mlir::TypeID taskID,
                              const llvm::hash_code &optionsHash) {
    auto key = std::make_pair(taskID, optionsHash);
    auto it = cachedPassManagers.find(key);
    if (it == cachedPassManagers.end())
      return nullptr;
    return it->second.get();
  }

  /// Check whether a CompilationTask with the specified type T and whose
  /// options have the given hash is in the cache. If so, return it; otherwise
  /// returns nullptr.
  template <typename T>
  CompilationTaskBase *
  lookupCachedCompilationTask(const llvm::hash_code &optionsHash) {
    return lookupCachedCompilationTask(mlir::TypeID::get<T>(), optionsHash);
  }

  /// Return the MLIRContext associated with the client.
  mlir::MLIRContext *getContext() const { return context; }

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

/// A registry function that adds passes to the given pass manager. This should
/// also parse options and return success() if parsing succeeded.
/// `errorHandler` is a functor used to emit errors during parsing.
/// parameter corresponds to the raw location within the pipeline string. This
/// should always return failure.
using TaskRegistryFunction = std::function<StatusOr<CompilationTaskBase *>(
    CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options)>;

struct TaskRegistration {
  TaskRegistryFunction registryFunc;
};

void registerCompilationTask(llvm::StringRef mnemonic, mlir::TypeID typeID,
                             TaskRegistryFunction func);

template <typename T>
void registerCompilationTask(llvm::StringRef mnemonic,
                             TaskRegistryFunction func) {
  return registerCompilationTask(mnemonic, mlir::TypeID::get<T>(),
                                 std::move(func));
}

template <typename T, typename OptionsType>
void registerCompilationTaskWithNoExtensions(llvm::StringRef mnemonic) {
  registerCompilationTask<T>(
      mnemonic,
      [](CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options)
          -> StatusOr<CompilationTaskBase *> {
        OptionsType result;
        std::string err;
        if (failed(result.parse(options, err)))
          return getInvalidArgStatus(
              "failed to parse options string \"{0:$[ ]}\" due to error {1}",
              llvm::iterator_range(options), err);

        llvm::Error finalizeStatus = result.finalize();
        std::optional<std::string> errMsg{};
        llvm::handleAllErrors(std::move(finalizeStatus),
                              [&errMsg](const llvm::StringError &err) {
                                errMsg = err.getMessage();
                              });

        if (errMsg)
          return getInvalidArgStatus("failed to parse options due to error {0}",
                                     errMsg);

        std::optional<llvm::hash_code> hashCode = result.getHash();
        if (!hashCode)
          return getInvalidArgStatus("failed to hash options");

        CompilationTaskBase *cached =
            client.lookupCachedCompilationTask<T>(*hashCode);
        if (cached)
          return cached;

        auto newPM = std::make_unique<T>(client.getContext(), result);
        auto ptr = newPM.get();
        client.updateCachedCompilationTask<T>(*hashCode, std::move(newPM));
        return ptr;
      });
}

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_CLIENT
