//===- LinalgInputPipeline.cpp --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Definitions for Linalg input pipelines.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/InputPipelines/LinalgInputPipeline.h"
#include "mlir-tensorrt-common/Utils/PassManagerUtils.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

llvm::cl::OptionCategory LinalgInputOptions::category = {
    "MLIR-TensorRT Linalg Input Options", ""};

void mtrt::compiler::buildLinalgInputPipeline(OpPassManager &pm,
                                              const LinalgInputOptions &opts) {
  addNestedPasses<func::FuncOp>(pm, [&opts](OpPassManager &funcPM) {
    funcPM.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    if (opts.enableLinalgElementwiseFusion)
      funcPM.addPass(mtrt::createLinalgElementwiseFusionPass());
    funcPM.addPass(mtrt::createLinalgSimplifyExtractSlicePass());
    funcPM.addPass(mtrt::createTensorExtPadToInsertSlicePass());
    funcPM.addPass(mlir::createCSEPass());
    funcPM.addPass(mlir::createCanonicalizerPass());
  });
}
