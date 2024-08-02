//===- ControlFlowOps.h -----------------------------------------*- C++ -*-===//
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
/// Control flow operation conversion utilities.
///
//===----------------------------------------------------------------------===//
#ifndef CONVERSION_HLOTOTENSORRT_CONTROLFLOWOPS_H
#define CONVERSION_HLOTOTENSORRT_CONTROLFLOWOPS_H

#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

/// Populate patterns for converting control flow ops (e.g. `stablehlo.while`)
/// to `tensorrt` dialect equivalents.
void populateStablehloControlFlowToTensorRtPatterns(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns,
    bool convertLoops, bool convertConditionals);

} // namespace mlir

#endif // CONVERSION_HLOTOTENSORRT_CONTROLFLOWOPS_H
