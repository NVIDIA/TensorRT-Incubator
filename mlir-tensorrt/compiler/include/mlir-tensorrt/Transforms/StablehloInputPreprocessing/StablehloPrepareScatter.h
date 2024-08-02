//===- StablehloPrepareScatter.h --------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_TRANSFORMS_STABLEHLOINPUTPREPROCESSING_STABLEHLOPREPARESCATTER_H
#define MLIR_TENSORRT_TRANSFORMS_STABLEHLOINPUTPREPROCESSING_STABLEHLOPREPARESCATTER_H

#include "mlir/IR/PatternMatch.h"
namespace mlir {
namespace stablehlo {
class ScatterOp;
}
namespace tensorrt {
/// Returns true if the `scatterOp` has a configuration that corresponds to the
/// ONNX ScatterNd operation semantic.
bool isCanonicalScatterNd(stablehlo::ScatterOp scatterOp);

/// Populate the pattern set with patterns to canonicalize `stablehlo.scatter`
/// operations to correspond to `tensorrt.scatter`/`onnx.ScatterNd`.
void populateCanonicalizeStablehloScatterForTensorRTPatterns(
    RewritePatternSet &patterns);
} // namespace tensorrt
} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_STABLEHLOINPUTPREPROCESSING_STABLEHLOPREPARESCATTER_H