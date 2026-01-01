//===- InitAllDialects.h ----------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Registration methods for MLIR dialects.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_INITALLDIALECTS
#define MLIR_TENSORRT_COMPILER_INITALLDIALECTS

namespace mlir {
class DialectRegistry;
}

namespace mtrt::compiler {

/// Register all the dialects used by MLIR-TensorRT to the registry.
void registerAllDialects(mlir::DialectRegistry &registry);

/// Register all the Dialect extensions used by MLIR-TensorRT to the registry.
void registerAllExtensions(mlir::DialectRegistry &registry);

/// Register all global compiler pipeline extensions. This should be invoked
/// before global LLVM command line options are parsed if a MainOptions object
/// is also registered against the global CL scope.
void registerAllCompilerTaskExtensions();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_INITALLDIALECTS
