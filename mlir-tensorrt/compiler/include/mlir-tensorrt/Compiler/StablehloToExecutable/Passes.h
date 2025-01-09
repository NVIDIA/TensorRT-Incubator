//===- Passes.h ----------------------------------------------===//
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
/// Declarations for opt tool pipeline command-line registration for pipelines
/// related to "stablehlo-to-executable".
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlirtrt::compiler {

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

/// Register the StableHLO clustering and compilation pipelines.
/// Note that currently it's not possible to use dynamically loaded extensions
/// when using pass pipelines directly from the command line. Instead, you need
/// to invoke the extension passes directly in the appropriate locations.
/// TODO: this limitation is caused by not having access to MLIRContext when the
/// pass pipeline is constructed. We can only use the dynamic extension
/// population mechanism when we have a context/CompilationClient, e.g. in
/// or from Python API.
/// The pipelines registered here will use "default extensions"
/// (TensorRTExtension).
void registerStablehloToExecutablePipelines();

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_PASSES
