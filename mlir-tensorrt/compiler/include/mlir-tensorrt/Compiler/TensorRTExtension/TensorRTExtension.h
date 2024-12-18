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
#ifndef MLIR_TENSORRT_COMPILER_TENSORRTEXTENSION_TENSORRTEXTENSION
#define MLIR_TENSORRT_COMPILER_TENSORRTEXTENSION_TENSORRTEXTENSION

#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/StableHloToExecutable.h"

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRT-specific compilation data
//===----------------------------------------------------------------------===//

class StableHLOToExecutableTensorRTExtension
    : public StablehloToExecutableOptions::Extension<
          StableHLOToExecutableTensorRTExtension> {
public:
  StableHLOToExecutableTensorRTExtension();

  llvm::StringRef getName() const final { return "tensorrt-extension"; }

  /// Hook invoked for populating passes associated with a particular phase.
  /// It is not guarunteed the order in which different extensions are run
  /// relative to each other (yet).
  void populatePasses(mlir::OpPassManager &pm, Phase phase,
                      const StablehloToExecutableOptions &options) const final;

  /// Allows the extension to hook into the option parsing infrastructure.
  void addToOptions(mlir::OptionsContext &context) final {
    context.addOption("disable-tensorrt-extension", disabled,
                      llvm::cl::init(false));
    translationOptions.addToOptions(context);
  }

  /// Override the current options.
  void setOptions(mlir::tensorrt::TensorRTTranslationOptions options) {
    this->translationOptions = std::move(options);
  }

private:
  /// Options for MLIR-to-TensorRT translation.
  mlir::tensorrt::TensorRTTranslationOptions translationOptions;

  /// Path where we should persist the timing cache to storage.
  std::string timingCachePath;
};

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(
    mlirtrt::compiler::StableHLOToExecutableTensorRTExtension)

#endif // MLIR_TENSORRT_COMPILER_TENSORRTEXTENSION_TENSORRTEXTENSION
