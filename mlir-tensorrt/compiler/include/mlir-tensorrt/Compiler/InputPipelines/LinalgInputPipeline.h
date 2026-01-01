//===- LinalgInputPipeline.h --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Declarations for Linalg input pipelines.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_INPUTPIPELINES_LINALGINPUTPIPELINE
#define MLIR_TENSORRT_COMPILER_INPUTPIPELINES_LINALGINPUTPIPELINE

#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir/Pass/PassManager.h"

namespace mtrt::compiler {

struct LinalgInputOptions : public mlir::OptionsGroup {
  using mlir::OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<bool> enableLinalgElementwiseFusion{
      this->ctx, "linalg-input-elementwise-fusion", llvm::cl::init(true),
      llvm::cl::desc("Enable linalg elementwise fusion."),
      llvm::cl::cat(category)};
};

/// Build a pipeline for preprocessing Linalg IR to convert it into the
/// canonical form. Some passes in this pipeline transforms ops to simplify
/// TensorRT conversion.
void buildLinalgInputPipeline(mlir::OpPassManager &pm,
                              const LinalgInputOptions &opts);
} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_INPUTPIPELINES_LINALGINPUTPIPELINE
