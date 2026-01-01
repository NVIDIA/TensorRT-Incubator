//===- Passes.h -------------------------------------------------*- C++ -*-===//
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
/// Declarations for StableHloExt passes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir::stablehlo_ext {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"

/// Options for enabling/disabling sets of opionated/target-specific Stablehlo
/// simplification patterns.
struct TargetSpecificCanonicalizationOptions {

  /// Enable canonicalization of `stablehlo.dot_general` operations.
  bool enableDotGeneralCanonicalization = true;

  /// Enable canonicalization of `stablehlo.gather` operations.
  bool enableGatherCanonicalization = true;

  /// Enable canonicalization of `stablehlo.scatter` operations.
  bool enableScatterCanonicalization = true;

  /// Enable canonicalization of `stablehlo.convolution` operations.
  bool enableConvolutionCanonicalization = true;

  /// Enable canonicalization of `stablehlo.gather` operations to
  /// `stablehlo.slice` where possible.
  bool enableGatherToSlice = true;

  /// Parse the given arguments into a `TargetSpecificCanonicalizationOptions`
  static FailureOr<TargetSpecificCanonicalizationOptions>
  parse(llvm::ArrayRef<std::string> disabled = {});

  /// Disable a specific pattern set.
  void disable(llvm::StringRef patternSetMnemonic);
};

/// Create a pass that applies the given target-specific patterns and has the
/// specified constant folding size limit.
std::unique_ptr<mlir::Pass> createTargetSpecificOptimizationsPass(
    TargetSpecificCanonicalizationOptions options,
    int64_t constantFoldSizeLimit);

} // namespace mlir::stablehlo_ext

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_TRANSFORMS_PASSES
