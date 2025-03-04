//===- TensorRTBase.h -------------------------------------------*- C++ -*-===//
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
/// Contains some core definitions in a library separate from the main TensorRT
/// dialect library.
///
/// Functions declared here may be used by both the core dialect library as
/// well as the NvInfer plugin support library. To avoid circular dependencies,
/// these functions must comprise their own standalone library.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTBASE
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTBASE

#include "mlir/IR/Types.h"

namespace mlir::tensorrt {

/// Returns true if the element type is either a signless `i8` type or a
/// `!quant.uniform` of type i8 approximaing f32 or f16 with zero shift value.
bool isTensorRTInt8Type(Type elType);

} // namespace mlir::tensorrt

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTBASE
