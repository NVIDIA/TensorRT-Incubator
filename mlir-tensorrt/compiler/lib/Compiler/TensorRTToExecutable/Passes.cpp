//===- Passes.cpp --------------------------------------------------------===//
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
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

#ifdef MLIR_TRT_ENABLE_HLO

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

namespace {
class TensorRTToExecutablePassPipelineOptions
    : public PassPipelineOptionsAdaptor<
          TensorRTToExecutablePassPipelineOptions,
          TensorRTToExecutableOptions> {};
} // namespace

void mlirtrt::compiler::registerTensorRTToExecutablePipelines() {
  PassPipelineRegistration<TensorRTToExecutablePassPipelineOptions>(
      "tensorrt-clustering-pipeline",
      "apply clustering to tensorrt IR",
      [](OpPassManager &pm,
         const TensorRTToExecutablePassPipelineOptions &opts) {
        TensorRTToExecutableTask::buildTensorRTClusteringPipeline(pm, opts);
      });

  PassPipelineRegistration<TensorRTToExecutablePassPipelineOptions>(
      "tensorrt-compilation-pipeline", "apply compilation post-clustering",
      [](OpPassManager &pm,
         const TensorRTToExecutablePassPipelineOptions &opts) {
        TensorRTToExecutableTask::buildPostClusteringPipeline(pm, opts);
      });
}

#endif // MLIR_TRT_ENABLE_HLO