//===- HostToEmitC.h ------------------------------------------------------===//
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
/// Declarations for the `convert-host-to-emitc` pass.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_HOSTTOEMITC
#define MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_HOSTTOEMITC

#include "llvm/ADT/StringRef.h"

namespace mlir {
class OpPassManager;
}

namespace mtrt::compiler {
struct EmitCOptions;

/// Populate the pass manager with the EmitC lowering pipeline.
///
/// The pipeline is parameterized by the EmitC options, and additionally takes
/// the tool output path and entrypoint to enable optional emission of C++
/// support files (StandaloneCPP runtime sources/headers, example CMake, and/or
/// a test driver) as artifacts.
void applyEmitCLoweringPipeline(mlir::OpPassManager &pm,
                                const EmitCOptions &opts,
                                llvm::StringRef outputPath,
                                llvm::StringRef entrypoint);
} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_HOSTTOEMITC
