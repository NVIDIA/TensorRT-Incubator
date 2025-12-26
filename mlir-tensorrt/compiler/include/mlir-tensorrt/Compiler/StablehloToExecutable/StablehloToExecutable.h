//===- StablehloToExecutable.h ----------------------------------*- C++ -*-===//
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

#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir/Pass/PassManager.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mtrt::compiler {

//===----------------------------------------------------------------------===//
// StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

class StablehloToExecutableTask;

struct StablehloToExecutableOptions
    : public PipelineOptions<ExecutorOptions, DeviceOptions,
                             BufferizationOptions> {
  using PipelineOptions::PipelineOptions;

  //===----------------------------------------------------------------------===//
  // Options
  //===----------------------------------------------------------------------===//

  /// This is exposed to enable experimenting with disabling certain
  /// optimizations applied during pre-processing which may not always be
  /// beneficial.
  ListOption<std::string> stablehloDisableOptimizationPatternSet{
      *this, "stablehlo-disable-optimization-pattern-sets",
      llvm::cl::list_init<std::string>({}),
      llvm::cl::desc("List specific optimization pattern sets to disable. "
                     "Available pattern sets: dot-general, "
                     "gather, scatter, convolution, gather-to-slice")};

  /// This is exposed to enable controlling the aggressiveness of rewrite-based
  /// constant folding. Setting this to large can result in slow compilation
  /// times and higher compilation-time memory usage (due to use of
  /// DenseElementsAttr).
  Option<int64_t> stablehloInputRewriteConstantFoldVolumeLimit{
      *this, "stablehlo-input-rewrite-constant-fold-volume-limit",
      llvm::cl::init(65536),
      llvm::cl::desc("Specifies the maximum tensor volume for the "
                     "rewrite-based Stablehlo constant folding patterns.")};

  Option<int64_t> unrollThreshold{
      *this, "unroll-threshold", llvm::cl::init(100),
      llvm::cl::desc("The cost threshold for unrolling for loops. Loops with a "
                     "cost <= the threshold will be unrolled. The cost is "
                     "estimated by counting the number of operations in the "
                     "loop body and multiplying it by the trip count.")};

  Option<bool> hoistAllocsToGlobals{
      *this, "hoist-allocs-to-globals", llvm::cl::init(true),
      llvm::cl::desc(
          "Hoist large local allocations to static global allocations if "
          "possible. May also apply some memory reuse optimizations.")};

  ListOption<std::string> defaultBackends{
      *this, "backends",
      // clang-format off
      llvm::cl::list_init<std::string>({
        "#plan.tensorrt_backend<disallow_shape_tensor_calculations=false, benefit=2>",
        "#plan.host_backend<benefit=1>"
      }),
      // clang-format on
      llvm::cl::desc(
          "The default list of backends to use for the compilation if no "
          "explicit 'plan.backends' attribute is provided on the top-level "
          "module.")};
};

//===----------------------------------------------------------------------===//
// StableHloToExecutableTask
//===----------------------------------------------------------------------===//

/// A StableHloToExecutableTask is a concrete Pipeline (PassManager) that
/// accepts StableHLO input IR and lowers it down to Executor IR which can be
/// translated into a MLIR-TensorRT executable.
class StablehloToExecutableTask
    : public Pipeline<StablehloToExecutableTask, StablehloToExecutableOptions> {
public:
  static llvm::StringRef getName() { return "stablehlo-to-executable"; }

  StablehloToExecutableTask(
      mlir::MLIRContext *ctx,
      std::unique_ptr<StablehloToExecutableOptions> options);

  void populatePassManager() final;
};

/// Register the task/options with the client's registry.
void registerStableHloToExecutableTask();

} // namespace mtrt::compiler

#endif // MLIR_TRT_ENABLE_HLO
#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE
