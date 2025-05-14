//===- Patterns.h -----------------------------------------------*- C++ -*-===//
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
/// Declarations for pattern sets related to StableHlo.
///
//===----------------------------------------------------------------------===//
namespace mlir {
class RewritePatternSet;
namespace stablehlo_ext {

/// Populate patterns that let `stablehlo` operations absorb generalizing
/// `tensor.cast` producers.
void populateStableHloAbsorbTensorCastPatterns(RewritePatternSet &patterns);

/// Populate patterns to canonicalize `stablehlo.convolution`.
void populateCanonicalizeStablehloConvolutionPatterns(
    RewritePatternSet &patterns);

/// Populate the pattern set with patterns to canonicalize `stablehlo.scatter`
/// operations to correspond to `tensorrt.scatter`/`onnx.ScatterNd`.
void populateCanonicalizeStablehloScatterPatterns(RewritePatternSet &patterns);

} // namespace stablehlo_ext
} // namespace mlir
