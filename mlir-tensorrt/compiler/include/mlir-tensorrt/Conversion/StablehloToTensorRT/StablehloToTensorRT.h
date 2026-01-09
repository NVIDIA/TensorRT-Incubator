//===- HloToTensorRT.h ------------------------------------------*- C++ -*-===//
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
/// Declarations for (chlo|stablehlo)-to-tensorrt conversion patterns.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_HLOTOTENSORRT_HLOTOTENSORRT_H
#define MLIR_TENSORRT_CONVERSION_HLOTOTENSORRT_HLOTOTENSORRT_H

#include "mlir-tensorrt-common/Utils/ShapeInfo.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"

namespace mlir {
class ConversionTarget;

/// Populate patterns for converting Stablehlo reduction and contraction ops to
/// TensorRT.
void populateStablehloReductionAndContractionToTensorRtConversionPattern(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit = 1, PatternBenefit dotToEinsumBenefit = 0);

/// Collection of rewrite patterns for lowering of Stable HLO to TensorRT
/// dialect.
/// The `preferEinsum` parameter controls whether `tensorrt.einsum` is used
/// as the primary method for converting `stablehlo.dot_general` or only for
/// fallback when conversion to `tensorrt.matrix_multiply` is not possible.
void populateStablehloToTensorRtConversionPattern(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns,
    ShapeInfoCallbacks shapeInfoCallbacks = {}, bool preferEinsum = false);

/// Populate patterns for convert Chlo ops to TensorRT ops.
void populateChloToTensorRtLegalityAndPatterns(
    TensorRTTypeConverter &typeConverter, ConversionTarget &target,
    RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_HLOTOTENSORRT_HLOTOTENSORRT_H
