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
/// related to "tensorrt-to-executable".
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE_PASSES
#define MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE_PASSES

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlirtrt::compiler {

// TODO: Does this also need Tablegen'd pass?

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

/// Register the TensorRT clustering and compilation pipelines.
// TODO (pranavm): How to do pipeline registration?
void registerTensorRTToExecutablePipelines();

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_TENSORRTTOEXECUTABLE_PASSES
