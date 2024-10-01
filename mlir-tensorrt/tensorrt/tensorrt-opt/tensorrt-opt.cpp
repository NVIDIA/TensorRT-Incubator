//===- tensorrt-opt.cpp ---------------------------------------------------===//
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
#include "mlir-tensorrt-dialect/Target/Passes.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
void registerTestTensorKindAnalysisPass();
void registerTestTensorRTShapeInferencePass();
} // namespace mlir

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tensorrt::TensorRTDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect, mlir::quant::QuantizationDialect,
                  mlir::scf::SCFDialect>();
  mlir::registerTestTensorKindAnalysisPass();
  mlir::registerTestTensorRTShapeInferencePass();
  mlir::func::registerInlinerExtension(registry);
  mlir::tensorrt::registerTensorRTTranslationCLOpts();
  mlir::tensorrt::registerTensorRTPasses();
  mlir::tensorrt::registerTensorRTTranslationPasses();
  mlir::registerTransformsPasses();
  mlir::tensorrt::registerTensorKindOpInterfaceExternalModels(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
