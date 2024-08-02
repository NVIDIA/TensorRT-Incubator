//===- RegisterMlirTensorRtTranslations.h -----------------------*- C++ -*-===//
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
// Register translations.
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTTRANSLATIONS
#define MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTTRANSLATIONS

#include "mlir/Tools/mlir-translate/Translation.h"

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#endif

#ifdef MLIR_TRT_TARGET_LUA
#include "mlir-executor/Target/Lua/TranslateToLua.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#endif // MLIR_TRT_TARGET_LUA

#ifdef MLIR_TRT_TARGET_CPP
namespace mlir {
/// MLIR doesn't provide this declaration except in `InitAllTranslations.h`, so
/// just declare it ourselves here.
void registerToCppTranslation();
} // namespace mlir
#endif

namespace mlir {
inline void registerAllMlirTensorRtTranslations() {
#ifdef MLIR_TRT_TARGET_TENSORRT
  // Register the TensorRT translation target.
  registerToTensorRTTranslation();
#endif // MLIR_TRT_TARGET_TENSORRT

#ifdef MLIR_TRT_TARGET_CPP
  registerToCppTranslation();
#endif // MLIR_TRT_TARGET_CPP

#ifdef MLIR_TRT_TARGET_LUA
  registerToLuaTranslation();
  registerToRuntimeExecutableTranslation();
#endif // MLIR_TRT_TARGET_LUA
}

} // namespace mlir

#endif // MLIR_TENSORRT_REGISTRATION_REGISTERMLIRTENSORRTTRANSLATIONS
