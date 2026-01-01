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
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/Hashing.h"
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
/// The CompilerClient provides an API to construct `Pipeline` instances from a
/// populated `MainOptions` object. The CompilerClient assumes ownership of the
/// pipeline and subsequent queries using the same options will return the same
/// cached `Pipeline` object.
///
/// TODO: We should remove the caching mechanism; that should be the
/// responsibility of the downstream user.
class CompilerClient {
public:
  static StatusOr<std::unique_ptr<CompilerClient>>
  create(mlir::MLIRContext *context);

  ~CompilerClient() = default;

  /// Insert a pipeline with options hash `hash` into the cache.
  void updateCachedPipeline(const llvm::hash_code &hash,
                            std::unique_ptr<Pipeline> pipeline);

  /// Check whether a Pipeline whose options have the given hash is in the
  /// cache. If so, return it; otherwise returns nullptr.
  Pipeline *lookupCachedPipeline(const llvm::hash_code &optionsHash) const;

  /// Return the MLIRContext associated with the client.
  mlir::MLIRContext *getContext() const { return context; }

  /// Get or create a pipeline for the given options.
  StatusOr<Pipeline *>
  getOrCreatePipeline(llvm::IntrusiveRefCntPtr<MainOptions> options);

protected:
  CompilerClient(mlir::MLIRContext *context);

  /// The MLIRContext in use by this client.
  mlir::MLIRContext *context;

  /// A registry of cached pipelines keyed by an options hash.
  llvm::DenseMap<llvm::hash_code, std::unique_ptr<Pipeline>> cachedPassManagers;
};

///===----------------------------------------------------------------------===//
// Pipeline Utilities
//===----------------------------------------------------------------------===//

/// Prints a CLI "--help"-type description to stdout that describes each option
/// associated with the pipeline.
void printPipelineHelp(mlir::MLIRContext *ctx);

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_CLIENT
