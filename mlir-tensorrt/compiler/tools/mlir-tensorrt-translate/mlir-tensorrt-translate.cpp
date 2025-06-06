//===- MlirTensorRtTranslateMain.cpp-----------------------------*- C++ -*-===//
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
//
// This file is the entry point for the `mlir-tensorrt-translate` tool.
//
//===----------------------------------------------------------------------===//
#include "mlir-executor/Target/Lua/TranslateToLua.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt/Features.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#endif // MLIR_TRT_TARGET_TENSORRT

int main(int argc, char **argv) {
  mlir::registerToCppTranslation();
  mlir::registerToLuaTranslation();
  mlir::registerToRuntimeExecutableTranslation();

  IF_MLIR_TRT_TARGET_TENSORRT({
    mlir::tensorrt::registerTensorRTTranslationCLOpts();
    mlir::registerToTensorRTTranslation();
  });

  return failed(mlir::mlirTranslateMain(argc, argv,
                                        "MLIR-TensorRT Translation Tool"))
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
