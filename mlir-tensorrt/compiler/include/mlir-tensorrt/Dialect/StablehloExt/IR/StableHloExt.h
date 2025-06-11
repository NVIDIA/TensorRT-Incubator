//===- StablehloExt.h -------------------------------------------*- C++ -*-===//
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
/// Declarations for Stable HLO dialect-related items that cannot be upstreamed.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_IR_STABLEHLOEXT_H
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_IR_STABLEHLOEXT_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::stablehlo {

/// Register StableHlo op implementations for TensorKindOpInterface.
void registerTensorKindOpInterfaceExternalModels(DialectRegistry &registry);

/// Register StableHlo op implementations for ReifyRankedShapedTypeOpInterface.
void registerTypeInferenceExternalModels(DialectRegistry &registry);

/// Register StableHlo op implementations for InferTensorValueRangeInterface.
void registerInferTensorValueRangeInterfaceExternalModels(
    DialectRegistry &registry);

} // namespace mlir::stablehlo

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_IR_STABLEHLOEXT_H
