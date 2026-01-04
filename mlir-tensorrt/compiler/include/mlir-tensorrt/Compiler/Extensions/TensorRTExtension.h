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
#ifndef MLIR_TENSORRT_COMPILER_TOEXECUTABLE_TENSORRTEXTENSION
#define MLIR_TENSORRT_COMPILER_TOEXECUTABLE_TENSORRTEXTENSION

#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"

namespace mtrt::compiler {

//===----------------------------------------------------------------------===//
// TensorRT-specific compilation data
//===----------------------------------------------------------------------===//

class TensorRTExtension : public Extension<TensorRTExtension, MainOptions> {
public:
  static llvm::StringRef getName() { return "tensorrt-extension"; }

  using Extension::Extension;

  /// Hook invoked for populating passes associated with a particular phase.
  /// It is not guaranteed what order different extensions are run relative to
  /// each other (yet).
  void populatePasses(mlir::OpPassManager &pm, Phase phase) const final;
};

/// Register the TensorRT extension.
void registerTensorRTExtension();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_TOEXECUTABLE_TENSORRTEXTENSION
