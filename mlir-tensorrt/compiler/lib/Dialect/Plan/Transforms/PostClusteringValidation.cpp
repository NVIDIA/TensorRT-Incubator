//===- PostClusteringValidation.cpp ---------------------------------------===//
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
/// This pass runs validation on public functions after clustering. Each op from
/// each public function is checked for being from a correct dialect and having
/// a correct type.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "post-clustering-validation"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir::plan {
#define GEN_PASS_DEF_POSTCLUSTERINGVALIDATIONPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// This function attempts to identify operations that we know we can't support
/// after the clustering phase. This includes operations that operate on tensor
/// types but are not bufferizable.
static bool isValidOp(Operation *op) {
  auto isNonTensor = [](Type t) { return !isa<ShapedType>(t); };
  if (isPure(op) && llvm::all_of(op->getOperandTypes(), isNonTensor) &&
      llvm::all_of(op->getResultTypes(), isNonTensor))
    return true;
  return isa<bufferization::BufferizableOpInterface>(op) ||
         isa<stablehlo::ConstantOp, affine::AffineApplyOp>(op) ||
         isa<tensorrt::TensorRTDialect, trtrt::TensorRTRuntimeDialect,
             executor::ExecutorDialect, func::FuncDialect, arith::ArithDialect,
             scf::SCFDialect, tensor::TensorDialect,
             bufferization::BufferizationDialect, plan::PlanDialect,
             cuda::CUDADialect>(op->getDialect());
}

/// Checks if op type is valid after clustering.
static bool isValidOpType(Operation *op) {
  return llvm::all_of(op->getResultTypes(), [](Type t) {
    if (!t.isIntOrIndexOrFloat())
      return true;
    return t.isIndex() || t.isInteger(1) || t.isInteger(8) || t.isInteger(32) ||
           t.isInteger(64) || t.isF16() || t.isF32() || t.isBF16() ||
           t.isInteger(4) || t.isFloat8E4M3FN();
  });
}

namespace {
class PostClusteringValidationPass
    : public plan::impl::PostClusteringValidationPassBase<
          PostClusteringValidationPass> {
  using Base::Base;
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isPrivate())
      return;

    func->walk([&](Operation *op) {
      if (!isValidOp(op) || !isValidOpType(op)) {
        emitError(op->getLoc())
            << "op: " << op << " from function " << func.getSymName()
            << " is invalid, post clustering.";
        return signalPassFailure();
      }
      return;
    });
  }
};
} // namespace
