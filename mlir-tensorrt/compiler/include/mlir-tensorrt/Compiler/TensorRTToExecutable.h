//===- TensorRTToExecutable.h -----------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE
#define MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE

// TODO (pranavm): MLIR_TRT_TARGET_TENSORRT is only needed because we pull in
// the TranslateToTensorRT.h header. If we move the translation options, we
// won't need it.
#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir-tensorrt-dialect/Utils/OptionsBundle.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir/Support/TypeID.h"

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRTToExecutableOptions
//===----------------------------------------------------------------------===//

// TODO (pranavm): Figure out a better way to reuse TRT translation options -
// maybe move to options providers?
struct TensorRTOptions
    : public mlirtrt::compiler::OptionsProvider<TensorRTOptions> {
  mlir::tensorrt::TensorRTTranslationOptions options;

  void addToOptions(mlir::OptionsContext &context) {
    options.addToOptions(context);
  }
};

struct TensorRTToExecutableOptions
    : public mlir::OptionsBundle<DeviceOptions, DebugOptions, ExecutorOptions,
                                 CommonCompilationOptions, TensorRTOptions> {

  TensorRTToExecutableOptions(TaskExtensionRegistry extensions);
};

//===----------------------------------------------------------------------===//
// TensorRTToExecutableTask
//===----------------------------------------------------------------------===//

class TensorRTToExecutableTask
    : public CompilationTask<TensorRTToExecutableTask,
                             TensorRTToExecutableOptions> {
public:
  using Base::Base;

  static void populatePassManager(mlir::PassManager &pm,
                                  const TensorRTToExecutableOptions &options);
};

/// Register the task/options with the client's registry.
void registerTensorRTToExecutableTask();

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

// TODO (pranavm): How to do pipeline registration?
// void registerTensorRTPipelines();

} // namespace mlirtrt::compiler

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlirtrt::compiler::TensorRTToExecutableTask)

#endif
#endif // MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE
