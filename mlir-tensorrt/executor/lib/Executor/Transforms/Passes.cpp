//===- ExecutorPipelines.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::executor;

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void executor::buildExecutorLoweringPipeline(
    OpPassManager &pm, const ConvertStdToExecutorPassOptions &stdToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions) {
  pm.addPass(createConvertComplexToStandardPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  addCleanupPasses(pm);
  pm.addPass(affine::createAffineExpandIndexOpsAsAffinePass());
  pm.addPass(mlir::createLowerAffinePass());
  addCleanupPasses(pm);
  pm.addPass(createConvertLinalgToExecutorPass(
      ConvertLinalgToExecutorPassOptions{stdToExecutorOpts.indexBitwidth}));
  pm.addPass(createConvertMemRefToExecutorPass(
      ConvertMemRefToExecutorPassOptions{stdToExecutorOpts.indexBitwidth}));
  addCleanupPasses(pm);
  // Expand unsupported Math operations before converting to Executor dialect.
  // This rewrites operations like sinh, cosh, powf, asin, acos, atan, etc.
  // into compositions of operations that the Executor dialect supports.
  pm.addPass(createExecutorExpandMathOpsPass());
  addCleanupPasses(pm);
  pm.addPass(createConvertStdToExecutorPass(stdToExecutorOpts,
                                            populateAdditionalTypeConversions));
  pm.addPass(createExecutorEmulateUnsupportedFloatsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createConvertExecutorToExecutorPass(
      ConvertExecutorToExecutorPassOptions{stdToExecutorOpts.indexBitwidth},
      populateAdditionalTypeConversions));
  pm.addPass(createExecutorLowerGlobalsPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createExecutorDecomposeAggregateLoadsAndStoresPass());
  // Optimize allocas BEFORE expanding: SROA splits table allocas, mem2reg
  // promotes to SSA
  pm.addPass(mlir::createSROA());
  pm.addPass(mlir::createMem2Reg());
  addCleanupPasses(pm);
  pm.addPass(createExecutorExpandOpsPass());
  addCleanupPasses(pm);
  pm.addPass(createExecutorLowerToRuntimeBuiltinsPass());
  pm.addPass(createExecutorPackArgumentsPass(
      executor::ExecutorPackArgumentsPassOptions{/*maxArguments=*/50}));
  addCleanupPasses(pm);
}

namespace {

/// Base class for introducing common options accross multiple pipelines.
template <typename T>
struct ExecutorLoweringCommonOptions : public PassPipelineOptions<T> {
  using Derived = PassPipelineOptions<T>;
  typename Derived::template Option<unsigned> indexBitWidth{
      *this, "index-bitwidth", llvm::cl::init(64),
      llvm::cl::desc("all index types will be converted to signless integers "
                     "of this bitwidth")};
};

/// Options for `executor-lowering-pipeline`.
struct ExecutorLoweringPipelineOptions
    : public ExecutorLoweringCommonOptions<ExecutorLoweringPipelineOptions> {};

} // namespace

void executor::registerExecutorPassPipelines() {
  PassPipelineRegistration<ExecutorLoweringPipelineOptions> registration(
      "executor-lowering-pipeline",
      "Lower executor-compatible IR to its final form prior to translation",
      [](OpPassManager &pm, const ExecutorLoweringPipelineOptions &opts) {
        ConvertStdToExecutorPassOptions stdToExecOpts;
        stdToExecOpts.indexBitwidth = opts.indexBitWidth;
        buildExecutorLoweringPipeline(pm, stdToExecOpts);
      });
}
