//===- Passes.cpp --------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt/Backends/Host/Passes.h"
#include "mlir-tensorrt-common/Conversion/Passes.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"

namespace mtrt::compiler {
#define GEN_PASS_DEF_PROCESSHOSTCLUSTERSPASS
#include "mlir-tensorrt/Backends/Host/Passes.h.inc"
} // namespace mtrt::compiler

using namespace mtrt;
using namespace mtrt::compiler;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ProcessHostClustersPass
//===----------------------------------------------------------------------===//

// This pass executes a "convert-stablehlo-scalar-to-arith" dynamically on all
// functions with the #plan.host_backend target attribute.
class ProcessHostClustersPass
    : public compiler::impl::ProcessHostClustersPassBase<
          ProcessHostClustersPass> {
public:
  ProcessHostClustersPass() {
    dynamicPM = OpPassManager("func.func");
    dynamicPM.addPass(mlir::createStablehloToLinalgPass());
    dynamicPM.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    dynamicPM.addPass(mtrt::createLinalgElementwiseFusionPass());
    dynamicPM.addPass(mlir::createLinalgDetensorizePass(
        mlir::LinalgDetensorizePassOptions{/*aggressiveMode=*/true}));
    dynamicPM.addPass(mlir::createConvertToLoops());
    dynamicPM.addPass(mlir::createCSEPass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(mtrt::createSCFDetensorizePass());
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.isPrivate())
      return;

    auto HostBackend = func->getAttrOfType<plan::HostBackendAttr>(
        plan::PlanDialect::kFuncTargetKind);
    if (!HostBackend)
      return;

    if (failed(runPipeline(dynamicPM, func)))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  OpPassManager dynamicPM;
};
} // namespace

#endif // MLIR_TRT_ENABLE_HLO
