//===- KernelGenExtension.h -------------------------------------*- C++ -*-===//
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
/// Codegen extension for the StableHloToExecutable compiler task API.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_KERNELGENEXTENSION
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_KERNELGENEXTENSION

#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"

namespace mtrt::compiler {

//===----------------------------------------------------------------------===//
// KernelGenExtension
//===----------------------------------------------------------------------===//

class StablehloToExecutableKernelGenExtension
    : public Extension<StablehloToExecutableKernelGenExtension,
                       StablehloToExecutableTask> {
public:
  static llvm::StringRef getName() { return "kernel-gen-extension"; }

  using Extension::Extension;

  /// Override the hook invoked when the options have been parsed/finalized.
  /// We use this to update the default backends to include the KernelBackend.
  void onOptionsParsed() final;

  /// Hook invoked for populating passes associated with a particular phase.
  void populatePasses(mlir::OpPassManager &pm, Phase phase) const final;

  /// Override the hook invoked immediately prior to running the MLIR
  /// compilation pipeline. We use this to emit a warning if the Kernel backend
  /// is not enabled but the module specifies use of the '#plan.kernel_backend'
  /// attribute.
  mlir::LogicalResult onBeforePipelineRun(mlir::ModuleOp module) const override;

  /// Directory where PTX data will be saved for debugging.
  Option<std::string> dumpPtxDir{
      this->ctx, "dump-ptx-dir", llvm::cl::init(""),
      llvm::cl::desc("path to directory where PTX files will be dumped")};

  ListOption<std::string> generatorBenefit{
      this->ctx, "generator-benefit",
      llvm::cl::desc("A list of 'name:benefit' pairs to adjust generator "
                     "benefits for kernel generation.")};

  Option<bool> disabled{this->ctx, "disable-kernel-gen-extension",
                        llvm::cl::init(false)};

  Option<bool> enableV2constantFolding{this->ctx, "enable-v2-constant-folding",
                                       llvm::cl::init(true)};
};

/// Uses a dialect extension ensure that StableHLOToExecutableKernelGenExtension
/// gets inserted in to the PlanDialect extension registry when PlanDialect is
/// loaded.
void registerStablehloToExecutableKernelGenExtension(
    mlir::DialectRegistry &registry);

/// Register StablehloToExecutableTask Pass Pipelines (so they can be invoked
/// from the CLI for convenience). These pipelines use the default extension
/// set plus the KernelGen extension.
void registerExtendedStablehloToExecutablePipelines();

/// Build a pipeline that converts StableHLO operations to Linalg kernels.
void buildStableHloToLinalgKernelsPipeline(mlir::OpPassManager &pm);

/// Register Pass Pipelines associated with the StablehloToExecutableKernelGen
/// extension.
void registerStablehloToExecutableKernelGenExtensionPipelines();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_KERNELGENEXTENSION
