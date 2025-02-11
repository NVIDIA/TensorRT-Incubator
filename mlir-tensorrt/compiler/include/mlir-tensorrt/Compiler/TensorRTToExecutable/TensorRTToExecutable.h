//===- TensorRTToExecutable.h -----------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
#ifndef MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE
#define MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE

// TODO (pranavm): MLIR_TRT_TARGET_TENSORRT is only needed because we pull in
// the TranslateToTensorRT.h header. If we move the translation options, we
// won't need it.
#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"

#include "mlir-tensorrt-dialect/Utils/OptionsBundle.h"
#include "mlir-tensorrt/Compiler/Client.h"

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRTToExecutableOptions
//===----------------------------------------------------------------------===//

class TensorRTToExecutableTask;

// TODO (pranavm): Figure out a better way to reuse TRT translation options -
// maybe move to options providers?
struct TensorRTOptions : public OptionsProvider<TensorRTOptions> {
public:
  using OptionsProvider::OptionsProvider;
  mlir::tensorrt::TensorRTTranslationOptions options;

  TensorRTOptions(mlir::OptionsContext &ctx) : OptionsProvider(ctx) {
    options.addToOptions(ctx);
  }
};

struct TensorRTToExecutableOptions
    : public mlir::OptionsBundle<DeviceOptions, DebugOptions, ExecutorOptions,
                                 TensorRTOptions> {
  // Default initialization does not require any extensions.
  TensorRTToExecutableOptions() = default;

  TensorRTToExecutableOptions(TaskExtensionRegistry extensions);

  Option<std::string> entrypoint{this, "entrypoint", llvm::cl::init("main"),
                                 llvm::cl::desc("entrypoint function name")};

  /// Forces entrypoint functions to return allocations corresponding to the
  /// original tensor results. Otherwise, entrypoints will be lowered to use
  /// destination passing style whenever possible, but some results may still
  /// lower to returned allocations (because the output shape may not be
  /// computable from the inputs). In either case, the user should verify the
  /// final calling convention of the compiled function(s) by inspecting the
  /// compiled function signature metadata.
  Option<bool> forceEntrypointsReturnAllocs{
      this, "force-entrypoints-return-allocs", llvm::cl::init(false),
      llvm::cl::desc(
          "Require entrypoint functions to return allocations corresponding to"
          " the original tensor results, otherwise they are transformed"
          " into destination arguments whenever possible.")};
};

//===----------------------------------------------------------------------===//
// TensorRTToExecutableTask
//===----------------------------------------------------------------------===//

class TensorRTToExecutableTask
    : public CompilationTask<TensorRTToExecutableTask,
                             TensorRTToExecutableOptions> {
public:
  TensorRTToExecutableTask(mlir::MLIRContext *ctx,
                           const TensorRTToExecutableOptions &options);

  /// Build the clustering pipeline that occurs on TensorRT Ops.
  static void
  buildTensorRTClusteringPipeline(mlir::OpPassManager &pm,
                                  const TensorRTToExecutableOptions &options);

  /// Build the compilation pipeline that runs after clustering.
  static void
  buildPostClusteringPipeline(mlir::OpPassManager &pm,
                              const TensorRTToExecutableOptions &options);

  static void populatePassManager(mlir::PassManager &pm,
                                  const TensorRTToExecutableOptions &options);
};

/// Register the task/options with the client's registry.
void registerTensorRTToExecutableTask();

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlirtrt::compiler::TensorRTToExecutableTask)

#endif // MLIR_TRT_TARGET_TENSORRT
#endif // MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE
