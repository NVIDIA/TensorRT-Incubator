//===- KernelGenExtension.h -------------------------------------*- C++ -*-===//
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
/// Declarations for the compiler pipeline kernel generation extension.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_EXTENSIONS_KERNELGENEXTENSION
#define MLIR_TENSORRT_COMPILER_EXTENSIONS_KERNELGENEXTENSION

#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"

namespace mtrt::compiler {

//===----------------------------------------------------------------------===//
// KernelGenExtension
//===----------------------------------------------------------------------===//

class KernelGenExtension : public Extension<KernelGenExtension, MainOptions> {
public:
  static llvm::StringRef getName() { return "kernel-gen-extension"; }

  using Extension::Extension;

  /// Hook invoked for populating passes associated with a particular extension
  /// point.
  void populatePasses(mlir::OpPassManager &pm,
                      ExtensionPoint point) const final;
};

/// Register StableHLO input pipelines (so they can be invoked from the CLI for
/// convenience). These pipelines use the default extension set plus the
/// KernelGen extension.
void registerStablehloToLinalgPipeline();

/// Build a pipeline that (a) convers all remaining input IR operations to
/// linalg if not already, (b) re-clusters and outlines linalg operations into
/// discrete kernels based on simple heuristics, (c) de-duplicates the kernels
/// after outlining.
void buildKernelGenReclusteringPipeline(mlir::OpPassManager &pm,
                                        mlir::plan::InputKind inputKind);

/// Register additional pass pipelines associated with the KernelGen extension.
void registerKernelGenExtensionPipelines();

/// Register the unified KernelGen extension (StableHLO path hooks).
void registerKernelGenExtension();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_EXTENSIONS_KERNELGENEXTENSION
