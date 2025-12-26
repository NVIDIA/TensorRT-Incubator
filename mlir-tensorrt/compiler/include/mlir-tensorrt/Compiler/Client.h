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
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mtrt::compiler {

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
/// The CompilerClient provides an API to construct Pipelines from
/// the mnemonic and pass pipeline options. The CompilerClient assumes ownership
/// of the pipeline and subsequent queries using the same options will return
/// the same PipelineBase object.
///
/// TODO: We should remove the caching mechanism; that should be the
/// responsibility of the downstream user.
class CompilerClient {
public:
  static StatusOr<std::unique_ptr<CompilerClient>>
  create(mlir::MLIRContext *context);

  ~CompilerClient() = default;

  /// Create or retrieve from the cache a pipeline of the specified
  /// type and options. If an existing pipeline is not in the cache,
  /// then it is constructed using the registered construction function and
  /// inserted into the cache.
  StatusOr<PipelineBase *> getPipeline(llvm::StringRef taskName,
                                       llvm::ArrayRef<llvm::StringRef> options,
                                       bool enableDebugOptions = false);

  /// Insert a pipeline of type T with options hash `hash` into the
  /// cache.
  void updateCachedPipeline(llvm::StringRef taskName,
                            const llvm::hash_code &hash,
                            std::unique_ptr<PipelineBase> task);

  /// Check whether a Pipeline with the specified task type and whose
  /// options have the given hash is in the cache. If so, return it; otherwise
  /// returns nullptr.
  PipelineBase *lookupCachedPipeline(llvm::StringRef taskName,
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
      llvm::DenseMap<llvm::hash_code, std::unique_ptr<PipelineBase>>>
      cachedPassManagers;
};

/// A registry function that adds passes to the given pass manager. This should
/// also parse options and return success() if parsing succeeded.
/// `errorHandler` is a functor used to emit errors during parsing.
/// parameter corresponds to the raw location within the pipeline string. This
/// should always return failure.
using TaskRegistryFunction = std::function<StatusOr<PipelineBase *>(
    CompilerClient &client, llvm::ArrayRef<llvm::StringRef> options,
    bool enableDebugOptions)>;

struct TaskRegistration {
  TaskRegistryFunction registryFunc;
};

///===----------------------------------------------------------------------===//
// Task Lookup Utilities
//===----------------------------------------------------------------------===//

/// Returns a list of registered pipeline task names.
llvm::SmallVector<llvm::StringRef> getRegisteredPipelineNames();

/// For the given pipeline task, prints a CLI "--help"-type description to
/// stdout that describes each option associated with the pipeline. If the
/// pipeline is not registered, a fatal error is issued.
void printPipelineHelp(mlir::MLIRContext *ctx, llvm::StringRef mnemonic);

//===----------------------------------------------------------------------===//
// Task Registration Utilities
//===----------------------------------------------------------------------===//

namespace detail {
/// Register pipeline task given the mnemonic and the options registration
/// function.
void registerPipeline(llvm::StringRef mnemonic, TaskRegistryFunction func);
} // namespace detail

/// Register a pipeline task by providing an explicit registration function for
/// the given options type.
template <typename T>
void registerPipeline(TaskRegistryFunction func) {
  return detail::registerPipeline(T::getName(), std::move(func));
}

/// This helper provides a convenience registration wrapper for most pipeline
/// tasks whose options can be constructed from a single boolean
/// (`enableDebugOptions`) and do not have associated extensions.
template <typename T, typename OptionsType>
void registerPipelineWithNoExtensions(llvm::StringRef mnemonic) {
  registerPipeline<T>([](CompilerClient &client,
                         llvm::ArrayRef<llvm::StringRef> options,
                         bool enableDebugOptions) -> StatusOr<PipelineBase *> {
    // Hash the flags directly.
    llvm::hash_code hashCode =
        llvm::hash_combine_range(options.begin(), options.end());

    // Check the cache.
    PipelineBase *cached = client.lookupCachedPipeline(T::getName(), hashCode);
    if (cached)
      return cached;

    /// Construct "uninitialized" options, which are just default options
    /// prior to updating the value from the command line flags.
    auto uninitOptions = std::make_unique<OptionsType>(enableDebugOptions);

    // No cached pipeline, so construct a new pipeline.
    auto newPM =
        std::make_unique<T>(client.getContext(), std::move(uninitOptions));

    // Invoke the initialization.
    if (Status s = newPM->initialize(options); !s.isOk())
      return s;

    // Give ownership to the client.
    auto ptr = newPM.get();
    client.updateCachedPipeline(T::getName(), hashCode, std::move(newPM));
    return ptr;
  });
}

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_CLIENT
