//===- TranslateToTensorRT.cpp ----------------------------------*- C++ -*-===//
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
// Registers translation MLIR -> TensorRT engine.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

static mlir::LogicalResult translateToTensorRT(mlir::Operation *op,
                                               llvm::raw_ostream &os);

namespace mlir {

//===----------------------------------------------------------------------===//
// Register the "mlir-to-tensorrt" translation
//===----------------------------------------------------------------------===//

void registerToTensorRTTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-tensorrt", "translate from mlir to tensorrt",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToTensorRT(op, output);
      },
      [](DialectRegistry &registry) {
        tensorrt::registerTensorRTTranslationCLOpts();
        registry.insert<tensorrt::TensorRTDialect, func::FuncDialect,
                        quant::QuantizationDialect, arith::ArithDialect>();
      });
}
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

static LogicalResult translateToTensorRT(Operation *op, llvm::raw_ostream &os) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return failure();
  TensorRTSerializedTimingCache timingCache;
  FailureOr<std::shared_ptr<TensorRTBuilderContext>> trtContext =
      TensorRTBuilderContext::create();

  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    FailureOr<TensorRTEngineResult> result =
        buildFunction(funcOp, **trtContext, timingCache);
    if (failed(result))
      return failure();
    const std::unique_ptr<nvinfer1::IHostMemory> &engine =
        result->serializedEngine;
    os.write(reinterpret_cast<const char *>(engine->data()), engine->size());
    break;
  }
  return success();
}
