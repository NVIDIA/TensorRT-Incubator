//===- kernel-opt.cpp -----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Entrypoint for the optimizer driver.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir-kernel/InitAllDialects.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/TransformSchedules/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Transforms/Passes.h"

namespace mlir::kernel {
void registerTestOutliningPass();
void registerKernelBufferizationTestPass();
} // namespace mlir::kernel

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::kernel::registerAllRequiredDialects(registry);
  mlir::kernel::registerTestOutliningPass();
  mlir::kernel::registerKernelBufferizationTestPass();
  mlir::registerLowerAffinePass();
  mlir::registerKernelConversionPasses();
  mlir::registerConvertVectorToSCFPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::kernel::registerKernelPipelines();
  mlir::kernel::registerKernelPasses();
  mlir::kernel::registerKernelTransformSchedulesPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerTransformsPasses();
  mlir::transform::registerTransformPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerLinalgPasses();
  mlir::arith::registerArithPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR-Kernel optimizer driver\n", registry));
}
