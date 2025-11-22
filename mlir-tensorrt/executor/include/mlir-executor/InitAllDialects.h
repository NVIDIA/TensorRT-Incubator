//===- InitAllDialects.h ----------------------------------------*- C++ -*-===//
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
#ifndef MLIR_EXECUTOR_INITALLDIALECTS
#define MLIR_EXECUTOR_INITALLDIALECTS

#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/BufferizationOpInterfaceImpls.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::executor {

inline void registerAllRequiredDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
      mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect,
      mlir::DLTIDialect,
      mlir::executor::ExecutorDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::math::MathDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect
    >();
  // clang-format on

  mlir::func::registerInlinerExtension(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::executor::registerBufferizationOpInterfaceExternalModels(registry);
}

} // namespace mlir::executor

#endif // MLIR_EXECUTOR_INITALLDIALECTS
