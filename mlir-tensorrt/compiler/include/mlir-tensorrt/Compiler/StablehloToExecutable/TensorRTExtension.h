//===- TensorRTExtension.h --------------------------------------*- C++ -*-===//
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
/// Declarations for TensorRT-specific compilation options and pipeline hooks.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_TENSORRTEXTENSION
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_TENSORRTEXTENSION

#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRT-specific compilation data
//===----------------------------------------------------------------------===//

class StablehloToExecutableTensorRTExtension
    : public StablehloToExecutableOptions::Extension<
          StablehloToExecutableTensorRTExtension> {
public:
  using StablehloToExecutableOptions::Extension<
      StablehloToExecutableTensorRTExtension>::Extension;

  llvm::StringRef getName() const final { return "tensorrt-extension"; }

  /// Hook invoked for populating passes associated with a particular phase.
  /// It is not guarunteed the order in which different extensions are run
  /// relative to each other (yet).
  void populatePasses(mlir::OpPassManager &pm, Phase phase,
                      const StablehloToExecutableOptions &options) const final;

  /// Override the current options.
  void setOptions(mlir::tensorrt::TensorRTTranslationOptions options) {
    this->timingCachePath = options.timingCachePath;
    this->enableStronglyTyped = options.enableStronglyTyped;
    this->tensorrtBuilderOptLevel = options.tensorrtBuilderOptLevel;
    this->saveTensorRTEnginesToDirectory =
        options.saveTensorRTLayerInfoDirectory;
    this->saveTensorRTLayerInfoDirectory =
        options.saveTensorRTLayerInfoDirectory;
    this->workspaceMemoryPoolLimit = options.workspaceMemoryPoolLimit;
  }

  Option<bool> disabled{this->ctx, "disable-tensorrt-extension",
                        llvm::cl::init(false)};

  Option<TensorRTTargetFormat> format{
      this->ctx, "tensorrt-target",
      llvm::cl::desc("specifies the target compilation format for "
                     "functions offloaded to TensorRT"),
      llvm::cl::init(TensorRTTargetFormat::Engine),
      llvm::cl::values(clEnumValN(TensorRTTargetFormat::Engine, "engine",
                                  "lower to compiled TensorRT engines"),
                       clEnumValN(TensorRTTargetFormat::CPP, "cpp",
                                  "lower to C++ TensorRT nvinfer API "))};

  /// Whether to use global CL config for options.
  Option<bool> useGlobalCLFlags{this->ctx,
                                "use-global-tensorrt-translation-flags",
                                llvm::cl::init(false)};

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
  Option<std::optional<uint64_t>, mlir::tensorrt::ByteSizeParser>
      workspaceMemoryPoolLimit{this->ctx,
                               "tensorrt-workspace-memory-pool-limit",
                               llvm::cl::init(std::nullopt)};

  Option<bool> forceDefaultSliceInBounds{
      this->ctx, "tensorrt-force-default-slice-in-bounds",
      llvm::cl::init(false),
      llvm::cl::desc("Specifies whether we should constrain dynamic offset and "
                     "sizes operands of 'default' (no OOB access allowed) "
                     "slice ops so that all accesses will be in bounds")};
};

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(
    mlirtrt::compiler::StablehloToExecutableTensorRTExtension)

#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_TENSORRTEXTENSION
