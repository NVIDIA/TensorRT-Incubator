//===- Passes.td -------------------------------------------*- Tablegen -*-===//
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
#ifndef MLIR_TENSORRT_TARGET_TENSORRT_PASSES
#define MLIR_TENSORRT_TARGET_TENSORRT_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TranslateToTensorRTEnginePass
//===----------------------------------------------------------------------===//

def TranslateToTensorRTEnginePass : Pass<"translate-tensorrt-to-engine"> {
  let summary = "Translate MLIR to TensorRT within a module";
  let description = [{
    This pass processes all function operations in the module and translates
    each one into a TensorRT engine. The pass supports two modes of operation:

    1. **Engine Building Mode**: Builds new TensorRT engines from the MLIR
       representation. The pass creates a TensorRT network from the function,
       configures builder options (precision flags, optimization level, timing
       cache, etc.), and serializes the resulting engine.

    2. **Engine Loading Mode**: Loads pre-built engines from a directory
       (specified via `--load-tensorrt-engines`). This is primarily used for
       debugging and testing.

    In both modes, the serialized engine is attached to each function as a
    `tensorrt.engine` attribute (a dense elements attribute of i8 values).

    The pass supports various TensorRT builder options including:
    - Precision modes: FP16, INT8, FP8, BF16 (depending on TensorRT version)
    - Strongly typed networks (TensorRT 10.0+)
    - Builder optimization levels (0-5)
    - Timing cache for faster subsequent builds
    - Workspace memory pool limits
    - Plugin serialization

    Optional debugging features:
    - Save engines to disk (`--save-tensorrt-engines`)
    - Save layer information JSON (`--save-tensorrt-layer-info`)

    The pass requires CUDA devices to be available and will fail if no devices
    are found. It also manages timing cache serialization/deserialization to
    speed up repeated builds.
  }];
  let dependentDialects = ["::mlir::tensorrt::TensorRTDialect"];
}

#endif // MLIR_TENSORRT_TARGET_TENSORRT_PASSES
