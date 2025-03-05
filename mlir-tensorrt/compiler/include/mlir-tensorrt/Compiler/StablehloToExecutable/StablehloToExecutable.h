//===- StablehloToExecutable.h ----------------------------------*- C++ -*-===//
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
#include "mlir-tensorrt-dialect/Utils/OptionsBundle.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/TypeID.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

class StablehloToExecutableTask;

struct StablehloToExecutableOptions
    : public mlir::OptionsBundle<DebugOptions, ExecutorOptions, DeviceOptions,
                                 PlanAllocOptions> {
  /// Initializes the options. The extensions in the provided registry
  /// must be extensions for the StableHloToExecutable task.
  StablehloToExecutableOptions(TaskExtensionRegistry extensions);

  /// Initializes the options using a default extension set (TensorRT
  /// extension).
  StablehloToExecutableOptions();

  /// Whether to disallow host tensors in TensorRT clusters.
  Option<bool> disallowHostTensorsInTensorRTClusters{
      this, "plan-clustering-disallow-host-tensors-in-tensorrt-clusters",
      llvm::cl::init(false),
      llvm::cl::desc("Don't allow TensorRt clusters to contain host tensor "
                     "calculations (but they can still be inputs)")};

  Option<std::string> hostTarget{
      this, "host-target", llvm::cl::init("executor"),
      llvm::cl::desc("Specifies host target, which can be either "
                     "\"executor\" or \"llvm\" or \"emitc\"")};

  Option<std::string> artifactDirectory{
      this, "artifacts-dir", llvm::cl::init(""),
      llvm::cl::desc(
          "specifies a directory where to save large artifacts as external "
          "files that may be referenced symbolically by filename in the IR")};

  Option<std::string> entrypoint{this, "entrypoint", llvm::cl::init("main"),
                                 llvm::cl::desc("entrypoint function name")};

  /// Base class for extensions associated with StableHloToExecutableTask.
  class ExtensionBase : public TaskExtensionBase {
  public:
    ExtensionBase(mlir::TypeID typeID)
        : TaskExtensionBase(typeID,
                            mlir::TypeID::get<StablehloToExecutableTask>()) {}

    static bool classof(const TaskExtensionBase *extension) {
      return extension->getTaskID() ==
             mlir::TypeID::get<StablehloToExecutableTask>();
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
                   const StablehloToExecutableOptions &options) const = 0;
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
class StablehloToExecutableTask
    : public CompilationTask<StablehloToExecutableTask,
                             StablehloToExecutableOptions> {
public:
  StablehloToExecutableTask(mlir::MLIRContext *ctx,
                            const StablehloToExecutableOptions &options);

  /// Build the clustering pipeline that occurs on Stablehlo Ops.
  static void
  buildStablehloClusteringPipeline(mlir::OpPassManager &pm,
                                   const StablehloToExecutableOptions &options);

  /// Build the pipeline (bufferization and lowering) that runs after
  /// clustering.
  static void
  buildPostClusteringPipeline(mlir::OpPassManager &pm,
                              const StablehloToExecutableOptions &options);

  static void populatePassManager(mlir::OpPassManager &pm,
                                  const StablehloToExecutableOptions &options);

  /// Compile a StableHLO module into a MLIR-TensorRT Runtime executable.
  /// This is the "functional" entrypoint that will allocate a new PassManager
  /// for a single run.
  static mlirtrt::StatusOr<std::unique_ptr<runtime::Executable>>
  compileStableHLOToExecutable(CompilerClient &client, mlir::ModuleOp module,
                               const StablehloToExecutableOptions &options);
};

/// Register the task/options with the client's registry.
void registerStableHloToExecutableTask();

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlirtrt::compiler::StablehloToExecutableTask)

#endif // MLIR_TRT_ENABLE_HLO
#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE
