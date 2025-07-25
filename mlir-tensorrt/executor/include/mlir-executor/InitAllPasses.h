//===- InitAllPasses.h ------------------------------------------*- C++ -*-===//
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
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::executor {

/// Register all passes defined by or required by the Executor dialect
inline void registerAllPasses() {
  mlir::arith::registerArithPasses();
  mlir::executor::registerExecutorConversionPasses();
  mlir::executor::registerExecutorPassPipelines();
  mlir::executor::registerExecutorTransformsPasses();
  mlir::func::registerDuplicateFunctionEliminationPass();
  mlir::memref::registerMemRefPasses();
  mlir::registerTransformsPasses();
  mlir::registerSCFToControlFlowPass();
}

} // namespace mlir::executor
