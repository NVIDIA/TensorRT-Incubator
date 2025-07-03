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
    : public CompilationTaskOptions<ExecutorOptions, DeviceOptions,
                                    PlanAllocOptions> {
  /// Initializes the options. The extensions in the provided registry
  /// must be extensions for the StableHloToExecutable task.
  StablehloToExecutableOptions(TaskExtensionRegistry extensions,
                               bool enableDebugOptions);

  /// Initializes the options using a default extension set (TensorRT
  /// extension).
  StablehloToExecutableOptions(bool enableDebugOptions = false);

  //===----------------------------------------------------------------------===//
  // Options
  //===----------------------------------------------------------------------===//

  /// TODO: Somehow move this to the TensorRT extension class? This is used when
  /// populating the default backend metadata. We should instead enable
  /// specification of backend default as a raw string which is parsed inside
  /// the `PopulateDefaultBackendMetadata` class.
  Option<bool> disallowHostTensorsInTensorRTClusters{
      *this, "plan-clustering-disallow-host-tensors-in-tensorrt-clusters",
      llvm::cl::init(false),
      llvm::cl::desc("Don't allow TensorRt clusters to contain host tensor "
                     "calculations (but they can still be inputs)")};

  /// This is exposed to enable experimentating with disabling certain
  /// optimizations applied during pre-processing which may not always be
  /// beneficial.
  ListOption<std::string> stablehloTargetSpecificPatternSets{
      *this, "stablehlo-input-optimization-pattern-sets",
      llvm::cl::list_init<std::string>({"all"}),
      llvm::cl::desc(
          "Optional target-specific optimization pattern sets to enable for "
          "the StableHLO "
          "preprocessing pipeline. Available pattern sets: dot-general, "
          "gather, scatter, convolution, gather-to-slice, all. Default is "
          "'all'.")};

  /// This is exposed to enable controlling the aggressiveness of rewrite-based
  /// constant folding. Setting this to large can result in slow compilation
  /// times and higher compilation-time memory usage (due to use of
  /// DenseElementsAttr).
  Option<int64_t> stablehloInputRewriteConstantFoldVolumeLimit{
      *this, "stablehlo-input-rewrite-constant-fold-volume-limit",
      llvm::cl::init(65536),
      llvm::cl::desc("Specifies the maximum tensor volume for the "
                     "rewrite-based Stablehlo constant folding patterns.")};

  //===----------------------------------------------------------------------===//
  // Extension Utilities
  //===----------------------------------------------------------------------===//

  /// Base class for extensions associated with StableHloToExecutableTask.
  class ExtensionBase : public TaskExtensionBase {
  public:
    ExtensionBase(mlir::TypeID typeID, CompilationTaskOptionsBase &ctx)
        : TaskExtensionBase(
              typeID, mlir::TypeID::get<StablehloToExecutableTask>(), ctx) {}

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
    Extension(CompilationTaskOptionsBase &ctx)
        : ExtensionBase(mlir::TypeID::get<DerivedTy>(), ctx) {}
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
  StablehloToExecutableTask(
      mlir::MLIRContext *ctx,
      std::unique_ptr<StablehloToExecutableOptions> options);

  /// Build the clustering pipeline that occurs on Stablehlo Ops.
  static void
  buildClusteringPipeline(mlir::OpPassManager &pm,
                          const StablehloToExecutableOptions &options);

  /// Build the pipeline (bufferization and lowering) that runs after
  /// clustering.
  static void
  buildPostClusteringPipeline(mlir::OpPassManager &pm,
                              const StablehloToExecutableOptions &options);

  static void populatePassManager(mlir::OpPassManager &pm,
                                  const StablehloToExecutableOptions &options);
};

/// Register the task/options with the client's registry.
void registerStableHloToExecutableTask();

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlirtrt::compiler::StablehloToExecutableTask)

#endif // MLIR_TRT_ENABLE_HLO
#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE
