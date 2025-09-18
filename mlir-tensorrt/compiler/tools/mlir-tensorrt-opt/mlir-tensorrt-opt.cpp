//===- mlir-tensorrt-opt.cpp ---------------------------------------------===//
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
/// Entry point for the `mlir-tensorrt-opt` tool
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/InitAllDialects.h"
#include "mlir-tensorrt/Compiler/InitAllPasses.h"
#include "mlir-tensorrt/Features.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


using namespace mlir;
using namespace llvm;

#ifdef MLIR_TRT_ENABLE_TESTING
namespace mlir {
void registerTestTensorKindAnalysisPass();
void registerTestTensorRTShapeInferencePass();

#ifdef MLIR_TRT_ENABLE_HLO
void registerTestBoundsAnalysisPass();
#endif // MLIR_TRT_ENABLE_HLO
} // namespace mlir

static void registerTestPasses() {
  mlir::registerTestTensorKindAnalysisPass();
  mlir::registerTestTensorRTShapeInferencePass();
  IF_MLIR_TRT_ENABLE_HLO({ mlir::registerTestBoundsAnalysisPass(); });
}
#endif // MLIR_TRT_ENABLE_TESTING

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mtrt::compiler::registerAllDialects(registry);
  mtrt::compiler::registerAllExtensions(registry);
  mtrt::compiler::registerAllPasses();

  IF_MLIR_TRT_ENABLE_TESTING({ registerTestPasses(); });

  return mlir::asMainReturnCode(
      MlirOptMain(argc, argv, "MLIR-TensorRT Optimizer", registry));
}
