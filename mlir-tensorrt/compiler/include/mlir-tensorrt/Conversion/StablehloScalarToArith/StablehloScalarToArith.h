//===- StablehloScalarToArith.h ----------------------------------*- C++-*-===//
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
/// Declarations for the `convert-stablehlo-to-scalar-arith` pass.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_STABLEHLOSCALARTOARITH_STABLEHLOSCALARTOARITH_H
#define MLIR_TENSORRT_CONVERSION_STABLEHLOSCALARTOARITH_STABLEHLOSCALARTOARITH_H

#include "mlir/Transforms/OneToNTypeConversion.h"

namespace mlir::stablehlo_ext {

/// Return true if the given type is scalarizable. The current conditions
/// required are that the type `t` is a TensorType, has rank of 0 or 1 and <= 4
/// elements.
bool isScalarizableType(Type t);

/// Return the 1-to-N type converter for scalarization.
TypeConverter getScalarizationTypeConverter();

} // namespace mlir::stablehlo_ext

#endif // MLIR_TENSORRT_CONVERSION_STABLEHLOSCALARTOARITH_STABLEHLOSCALARTOARITH_H
