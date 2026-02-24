//===- MaterializeShapeCalculations.h ---------------------------*- C++ -*-===//
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
/// Plan dialect materialize shape calculations pass declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_MATERIALIZE_SHAPE_CALCULATIONS_H
#define MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_MATERIALIZE_SHAPE_CALCULATIONS_H

namespace mlir {
class RewritePatternSet;
class SymbolTableCollection;

namespace plan {
/// Populate patterns that allow Stablehlo operations to be simplified with
/// tensor.extract and tensor.dim operations.
void populateMaterializeShapeCalculationsStablehloPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTable);
} // namespace plan
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_MATERIALIZE_SHAPE_CALCULATIONS_H
