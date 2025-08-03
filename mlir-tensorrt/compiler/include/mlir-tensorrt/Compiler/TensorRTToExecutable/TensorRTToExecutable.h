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

#include "mlir-tensorrt-common/Support/CommandLineExtras.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/Client.h"

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRTToExecutableOptions
//===----------------------------------------------------------------------===//

class TensorRTToExecutableTask;

struct TensorRTOptions : public OptionsProvider {
public:
  using OptionsProvider::OptionsProvider;

  Option<std::string> timingCachePath{this->ctx, "tensorrt-timing-cache-path",
                                      llvm::cl::init("")};
  Option<int> tensorrtBuilderOptLevel{this->ctx, "tensorrt-builder-opt-level",
                                      llvm::cl::init(0)};
  Option<bool> enableStronglyTyped{this->ctx, "tensorrt-strongly-typed",
                                   llvm::cl::init(false)};
  Option<std::string> saveTensorRTEnginesToDirectory{
      this->ctx, "tensorrt-engines-dir", llvm::cl::init("")};
  Option<std::string> saveTensorRTLayerInfoDirectory{
      this->ctx, "tensorrt-layer-info-dir", llvm::cl::init("")};
  Option<std::optional<uint64_t>, mlir::ByteSizeParser>
      workspaceMemoryPoolLimit{this->ctx,
                               "tensorrt-workspace-memory-pool-limit",
                               llvm::cl::init(std::nullopt)};

  Option<TensorRTTargetFormat> format{
      this->ctx, "tensorrt-target",
      llvm::cl::desc("specifies the target compilation format for "
                     "functions offloaded to TensorRT"),
      llvm::cl::init(TensorRTTargetFormat::Engine),
      llvm::cl::values(clEnumValN(TensorRTTargetFormat::Engine, "engine",
                                  "lower to compiled TensorRT engines"),
                       clEnumValN(TensorRTTargetFormat::CPP, "cpp",
                                  "lower to C++ TensorRT nvinfer API "))};

  Option<bool> forceDefaultSliceInBounds{
      this->ctx, "tensorrt-force-default-slice-in-bounds",
      llvm::cl::init(false),
      llvm::cl::desc("Specifies whether we should constrain dynamic offset and "
                     "sizes operands of 'default' (no OOB access allowed) "
                     "slice ops so that all accesses will be in bounds")};

  mlir::tensorrt::TensorRTTranslationOptions options() const {
    mlir::tensorrt::TensorRTTranslationOptions translationOpts{};
    translationOpts.saveTensorRTEnginesToDirectory =
        saveTensorRTEnginesToDirectory;
    translationOpts.saveTensorRTLayerInfoDirectory =
        saveTensorRTLayerInfoDirectory;
    translationOpts.tensorrtBuilderOptLevel = tensorrtBuilderOptLevel;
    translationOpts.enableStronglyTyped = enableStronglyTyped;
    translationOpts.timingCachePath = timingCachePath;
    translationOpts.workspaceMemoryPoolLimit = workspaceMemoryPoolLimit;
    return translationOpts;
  }
};

struct TensorRTToExecutableOptions
    : public CompilationTaskOptions<DeviceOptions, ExecutorOptions,
                                    BufferizationOptions, TensorRTOptions> {
  using CompilationTaskOptions::CompilationTaskOptions;
};

//===----------------------------------------------------------------------===//
// TensorRTToExecutableTask
//===----------------------------------------------------------------------===//

class TensorRTToExecutableTask
    : public CompilationTask<TensorRTToExecutableTask,
                             TensorRTToExecutableOptions> {
public:
  TensorRTToExecutableTask(
      mlir::MLIRContext *ctx,
      std::unique_ptr<TensorRTToExecutableOptions> options);

  static llvm::StringRef getName() { return "tensorrt-to-executable"; }

  void populatePassManager() final;
};

/// Register the task/options with the client's registry.
void registerTensorRTToExecutableTask();

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE
