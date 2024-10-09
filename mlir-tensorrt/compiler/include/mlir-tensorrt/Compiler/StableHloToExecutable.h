//===- StableHloToExecutable.h ----------------------------------*- C++ -*-===//
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
/// MLIR programs from StableHLO to an MLIR-TensorRT executable. This API
/// should only be used by clients that are building the project from source in
/// order to avoid sending C++ objects across the API boundary. For all other
/// cases, including when MLIR-TRT's C++ compilation options may differ from
/// the client, a C API is also provided (see the
/// `include/mlir-tensorrt-c/Compiler` directory).
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/TypeID.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// StableHLOProgramSignatureRefinementOptions
//===----------------------------------------------------------------------===//

struct StableHLOProgramSignatureRefinementOptions
    : public mlir::OptionsContext {
  /// Creates default compilation options.
  StableHLOProgramSignatureRefinementOptions() {
    this->addOption("func-name", funcName, llvm::cl::init("main"));
    debugOptions.addToOptions(*this);
  }

  /// Set the entrypoint function name.
  StableHLOProgramSignatureRefinementOptions &
  setFuncName(const std::string &name) {
    funcName = name;
    return *this;
  }

  std::string funcName = "main";

  DebugOptions debugOptions;
};

//===----------------------------------------------------------------------===//
// StableHLO Signature Refinement Entrypoint
//===----------------------------------------------------------------------===//

/// Attempt to refine the function signature of a StableHLO program through
/// canonicalization and constant folding. Returns the refined signature of the
/// specified function of the module.
mlirtrt::StatusOr<mlir::FunctionType> getStableHLOProgramRefinedSignature(
    CompilerClient &client, mlir::ModuleOp module,
    const StableHLOProgramSignatureRefinementOptions &options);

//===----------------------------------------------------------------------===//
// StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

class StableHloToExecutableTask;

struct StableHLOToExecutableOptions : public mlir::OptionsContext {
  /// Initializes the options. The extensions in the provided registry
  /// must be extensions for the StableHloToExecutable task.
  StableHLOToExecutableOptions(TaskExtensionRegistry extensions);

  /// Set the target device compute capability (SM version) and max shared
  /// memory per block (in kilobytes). The `maxSharedMemoryPerBlockKb` is the
  /// maximum shared memory per block allowed for kernels and is passed to the
  /// TensorRT builder.
  StableHLOToExecutableOptions &
  setDeviceOptions(int64_t computeCapability,
                   int64_t maxSharedMemoryPerBlockKb);

  /// Infer target device information from the first visible CUDA device on the
  /// host executing this code.
  Status inferDeviceOptionsFromHost();

  /// Get the mutable DebugOptions.
  DebugOptions &getDebugOptions() { return debugOptions; }

  /// Return the hash of the options. Returns `nullopt` when the TensorRT
  /// layer metadata callback is set since that can't be reliably hashed.
  std::optional<llvm::hash_code> getHash() const override;

  /// The host index bit-width.
  int64_t executorIndexBitwidth{64};

  /// Whether to pass memref's as struct/table in function calls.
  bool executorUsePackedMemRefCConv{true};

  /// Target device compute capability (SM version)
  int64_t deviceComputeCapability;

  /// Target device max shared memory per block (kilobytes)
  int64_t deviceMaxSharedMemoryPerBlockKb;

  /// Whether to ignore `deviceX` options and instead infer them from the GPUs
  /// on the host system running the compilation.
  bool shouldInferDeviceOptionsFromHost = false;

  /// Whether to disallow host tensors in TensorRT clusters.
  bool disallowHostTensorsInTensorRTClusters = false;

  /// Entrypoint function name.
  std::string entrypoint = "main";

  DebugOptions debugOptions;

  std::function<std::string(mlir::Operation *)> layerMetadataCallback{nullptr};

  /// Base class for extensions associated with StableHloToExecutableTask.
  class ExtensionBase : public TaskExtensionBase {
  public:
    ExtensionBase(mlir::TypeID typeID)
        : TaskExtensionBase(typeID,
                            mlir::TypeID::get<StableHloToExecutableTask>()) {}

    static bool classof(const TaskExtensionBase *extension) {
      return extension->getTaskID() ==
             mlir::TypeID::get<StableHloToExecutableTask>();
    }

    enum class Phase {
      PreClustering,
      PostClustering,
      PreBufferization,
      PostBufferization,
      ExecutorLowering
    };

    /// Hook invoked for populating passes associated with a particular phase.
    /// It is not guarunteed the order in which different extensions are run
    /// relative to each other (yet).
    virtual void
    populatePasses(mlir::OpPassManager &pm, Phase phase,
                   const StableHLOToExecutableOptions &options) const = 0;
  };

  /// A StableHLOToExecutableOptions::Extension is an extension that must
  /// implement the base hooks to modify how a
  template <typename DerivedTy>
  class Extension : public ExtensionBase {
  public:
    Extension() : ExtensionBase(mlir::TypeID::get<DerivedTy>()) {}
  };

  /// List of extensions (in no defined order).
  TaskExtensionRegistry extensions;
};

//===----------------------------------------------------------------------===//
// StableHloToExecutableTask
//===----------------------------------------------------------------------===//

/// A StableHloToExecutableTask is a concrete CompilationTask (PassManager) that
/// accepts StableHLO input IR and lowers it down to Executor IR which can be
/// translated into a MLIR-TensorRT executable.
class StableHloToExecutableTask
    : public CompilationTask<StableHloToExecutableTask,
                             StableHLOToExecutableOptions> {
public:
  using Base::Base;

  /// Build the clustering pipeline that occurs on Stablehlo Ops.
  static void
  buildStablehloClusteringPipeline(mlir::OpPassManager &pm,
                                   const StableHLOToExecutableOptions &options);

  /// Build the pipeline (bufferization and lowering) that runs after
  /// clustering.
  static void
  buildPostClusteringPipeline(mlir::OpPassManager &pm,
                              const StableHLOToExecutableOptions &options);

  static void populatePassManager(mlir::PassManager &pm,
                                  const StableHLOToExecutableOptions &options);

  /// Compile a StableHLO module into a MLIR-TensorRT Runtime executable.
  /// This is the "functional" entrypoint that will allocate a new PassManager
  /// for a single run.
  static mlirtrt::StatusOr<std::unique_ptr<runtime::Executable>>
  compileStableHLOToExecutable(mlir::ModuleOp module,
                               const StableHLOToExecutableOptions &options);

  /// Compile a StableHLO module into a MLIR-TensorRT Runtime executable.
  /// This is the "functional" entrypoint that will allocate a new PassManager
  /// for a single run.
  static mlirtrt::StatusOr<std::unique_ptr<runtime::Executable>>
  compileStableHLOToExecutable(CompilerClient &client, mlir::ModuleOp module,
                               const StableHLOToExecutableOptions &options);
};

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

/// Register the StableHLO clustering and compilation pipelines.
/// Note that currently it's not possible to use dynamically loaded extensions
/// when using pass pipelines directly from the command line. Instead, you need
/// to invoke the extension passes directly in the appropriate locations.
/// TODO: this limitation is caused by not having access to MLIRContext when the
/// pass pipeline is constructed. We can only use the dynamic extension
/// population mechanism when we have a context/CompilationClient, e.g. in
/// or from Python API.
/// The pipelines registered here will use "default extensions" (e.g. TensorRT).
void registerStablehloClusteringPipelines();

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlirtrt::compiler::StableHloToExecutableTask)

#endif // MLIR_TRT_ENABLE_HLO
#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE
