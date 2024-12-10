//===- TensorRTToExecutable.cpp ---------------------------------*- C++ -*-===//
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
#ifdef MLIR_TRT_TARGET_TENSORRT

#include "mlir-tensorrt/Compiler/TensorRTToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/OptionsRegistry.h"
#include "mlir-tensorrt/Compiler/PassManagerUtils.h"

using namespace mlirtrt::compiler;

TensorRTToExecutableOptions::TensorRTToExecutableOptions(
    TaskExtensionRegistry extensions) {
  // TODO (pranavm): Do we need to support extensions?
}

void TensorRTToExecutableTask::populatePassManager(
    mlir::PassManager &pm, const TensorRTToExecutableOptions &options) {
  if (failed(setupPassManager(pm, options.get<DebugOptions>()))) {
    /// TODO: Ignored. This can fail if pass manager static CL options were not
    /// registered/initialized. This happens through invocation of e.g. this
    /// function in e.g. Python bindings or standalone calls to C++ or C API
    /// without doing all the typical static CL setup. We should instead be
    /// accepting a PassManager here that has already been setup to the caller's
    /// specifications.
  }

  // TODO (pranavm): Which passes go here?

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  stdToExecOpts.usePackedMemRefCConv = true;
  mlir::executor::buildExecutorLoweringPipeline(pm, stdToExecOpts);
}

void mlirtrt::compiler::registerTensorRTToExecutableTask() {
  registerOption("tensorrt-to-executable",
                 optionsCreateFromArgs<TensorRTToExecutableOptions,
                                       TensorRTToExecutableTask>);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlirtrt::compiler::TensorRTToExecutableTask)

#endif
