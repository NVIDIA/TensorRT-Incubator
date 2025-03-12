//===- Passes.cpp --------------------------------------------------------===//
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
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mlirtrt::compiler {
#define GEN_PASS_DEF_PROCESSSTABLEHLOHOSTCLUSTERSPASS
#define GEN_PASS_DEF_CONVERTSTABLEHLOCONSTANTSTOARITHPASS
#define GEN_PASS_DEF_POPULATEDEFAULTBACKENDMETADATAPASS
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h.inc"
} // namespace mlirtrt::compiler

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// PopulateDefaultBackendMetadataPass
//===----------------------------------------------------------------------===//
// This pass executes a "convert-stablehlo-scalar-to-arith" dynamically on all
// functions with the `cluster.host` attribute.
class PopulateDefaultBackendMetadataPass
    : public compiler::impl::PopulateDefaultBackendMetadataPassBase<
          PopulateDefaultBackendMetadataPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(plan::PlanDialect::kModuleClusterKindsAttrName))
      return;
    SmallVector<Attribute> clusterKinds;
    clusterKinds.push_back(mlir::plan::TensorRTClusterKindAttr::get(
        module->getContext(), disallowHostTensorsInTensorRTClusters, 10,
        tensorrtVersionMajor));
    clusterKinds.push_back(
        mlir::plan::HostClusterKindAttr::get(module->getContext(), 9));
    module->setAttr(plan::PlanDialect::kModuleClusterKindsAttrName,
                    ArrayAttr::get(module->getContext(), clusterKinds));
  }
};

//===----------------------------------------------------------------------===//
// ProcessHostClustersPass
//===----------------------------------------------------------------------===//

// This pass executes a "convert-stablehlo-scalar-to-arith" dynamically on all
// functions with the `cluster.host` attribute.
class ProcessHostClustersPass
    : public compiler::impl::ProcessStablehloHostClustersPassBase<
          ProcessHostClustersPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.isPrivate() || !func->hasAttr("cluster.host"))
      return;

    OpPassManager dynamicPM("func.func");
    dynamicPM.addPass(createConvertStablehloScalarToArithPass());
    if (failed(runPipeline(dynamicPM, func)))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

//===----------------------------------------------------------------------===//
//  ConvertStablehloConstantToArithPass
//===----------------------------------------------------------------------===//

class ConvertStablehloConstantToArithPass
    : public compiler::impl::ConvertStablehloConstantsToArithPassBase<
          ConvertStablehloConstantToArithPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Apply other preparation and simplification patterns.
    RewritePatternSet patterns(func->getContext());
    // Convert `stablehlo.constant` to `arith.constant`.
    patterns.add<stablehlo::ConstantOp>(
        +[](stablehlo::ConstantOp constOp,
            PatternRewriter &rewriter) -> LogicalResult {
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp,
                                                         constOp.getValue());
          return success();
        });

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      emitError(func.getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

namespace {
/// Declare adaptor so we can use StablehloToExecutableOptions as a MLIR
/// pipeline options object.
class StablehloToExecutablePassPipelineOptions
    : public PassPipelineOptionsAdaptor<
          StablehloToExecutablePassPipelineOptions,
          StablehloToExecutableOptions> {};
} // namespace

void mlirtrt::compiler::registerStablehloToExecutablePipelines() {
  PassPipelineRegistration<StablehloToExecutablePassPipelineOptions>(
      "stablehlo-clustering-pipeline",
      "apply clustering and initial transformations to stablehlo IR",
      [](OpPassManager &pm,
         const StablehloToExecutablePassPipelineOptions &opts) {
        StablehloToExecutableTask::buildStablehloClusteringPipeline(pm, opts);
      });

  PassPipelineRegistration<StablehloToExecutablePassPipelineOptions>(
      "stablehlo-to-executable-pipeline",
      "apply the full stablehlo-to-executable pipeline",
      [](OpPassManager &pm,
         const StablehloToExecutablePassPipelineOptions &opts) {
        StablehloToExecutableTask::populatePassManager(pm, opts);
      });

  PassPipelineRegistration<StablehloToExecutablePassPipelineOptions>(
      "post-clustering-pipeline", "apply compilation post-clustering",
      [](OpPassManager &pm,
         const StablehloToExecutablePassPipelineOptions &opts) {
        StablehloToExecutableTask::buildPostClusteringPipeline(pm, opts);
      });
}

#endif // MLIR_TRT_ENABLE_HLO
