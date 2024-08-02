//===- executor-opt.cpp ---------------------------------------------------===//
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
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::executor::ExecutorDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, mlir::DLTIDialect>();
  mlir::func::registerInlinerExtension(registry);
  mlir::executor::registerExecutorTransformsPasses();
  mlir::registerTransformsPasses();
  mlir::executor::registerMLIRExecutorConversionPasses();
  mlir::registerSCFToControlFlow();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
