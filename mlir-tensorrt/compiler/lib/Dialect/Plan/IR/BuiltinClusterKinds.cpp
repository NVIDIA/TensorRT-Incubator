//===- BuiltinClusterKinds.cpp --------------------------------------------===//
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
/// Definitions for TensorRT and Host Cluster Kinds
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace mlir;
using namespace mlir::plan;

#define DEBUG_TYPE "builtin-cluster-kinds"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// HostClusterKindAttr
//===----------------------------------------------------------------------===//

std::string HostClusterKindAttr::getClusterKindName() const { return "host"; }

int64_t HostClusterKindAttr::getClusterBenefit() const { return getBenefit(); }

/// ClusteringOpts that identifies groups of `stablehlo` ops that can be
/// converted to scalars and will be clustered into scalar cluster.
ClusteringOpts HostClusterKindAttr::getClusterKindOptions(
    DataFlowSolver &solver, std::optional<int64_t> trtMajorVersion) const {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [&solver](Operation *op) {
    return plan::detail::shouldRunOnHost(op, solver);
  };
  return opts;
}

std::unique_ptr<Pass> HostClusterKindAttr::getClusterKindPass() const {
  return nullptr;
}

/// Determines whether a cluster being outlined should clone a constant or
/// pass constant by value.
static bool shouldCloneProducer(Value v, Region &cluster) {
  Operation *producer = v.getDefiningOp();
  if (!producer->hasTrait<OpTrait::ConstantLike>() ||
      producer->getNumResults() != 1)
    return false;
  RankedTensorType type =
      dyn_cast<RankedTensorType>(producer->getResultTypes().front());
  if (!type || !type.hasStaticShape())
    return false;

  // A value should be cloned if all of its uses are in the cluster.
  if (llvm::all_of(v.getUsers(), [&](Operation *user) {
        return cluster.isAncestor(user->getParentRegion());
      }))
    return true;
  return type.getNumElements() *
             llvm::divideCeil(type.getElementTypeBitWidth(), 8) <
         1024 * 1024;
}

std::optional<OutlineRegionOptions>
HostClusterKindAttr::getClusterOutliningOptions(
    MLIRContext *ctx, SymbolTable &moduleSymbolTable) const {
  OpBuilder b(ctx);
  return OutlineRegionOptions{
      /*typeConverter=*/stablehlo_ext::getScalarizationTypeConverter(),
      /*shouldCloneProducer=*/shouldCloneProducer,
      /*createFunc=*/
      OutlineRegionOptions::getDefaultCreateFuncAndCallStubFunc(
          moduleSymbolTable, {b.getNamedAttr("cluster.host", b.getUnitAttr())},
          "host_cluster")};
}

std::function<bool(const Cluster &)>
HostClusterKindAttr::getClusterFilter() const {
  return [](const Cluster &cluster) { return true; };
}

//===----------------------------------------------------------------------===//
// TensorRTClusterKindAttr
//===----------------------------------------------------------------------===//

std::string TensorRTClusterKindAttr::getClusterKindName() const {
  return "tensorrt";
}

/// ClusteringOpts that identifies groups of TensorRT operations and will be
/// clustered into one TensorRT function (which is eventually translated to a
/// engine).
ClusteringOpts TensorRTClusterKindAttr::getClusterKindOptions(
    DataFlowSolver &solver, std::optional<int64_t> trtMajorVersion) const {
  // Any properties used in the returned lambdas must be copied by value,
  // otherwise it will not work correctly.
  bool disallowShapeTensorCalculations = getDisallowShapeTensorCalculations();

  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [solver = &solver, disallowShapeTensorCalculations,
                          trtMajorVersion](Operation *op) {
    if (!trtMajorVersion.has_value())
      return false;
    if (op->hasTrait<OpTrait::ConstantLike>())
      return false;
    if (llvm::isa<plan::WithShapeOp>(op))
      return true;
    if (llvm::isa<plan::WithValuesOp>(op))
      return !disallowShapeTensorCalculations;
    if (llvm::isa<tensorrt::TensorRTOpInterface>(op))
      return true;
    if (!llvm::isa<stablehlo::StablehloDialect, chlo::ChloDialect>(
            op->getDialect()))
      return false;
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    LowerToTensorRTOptions loweringOptions;
    loweringOptions.setTensorRTVersion(*trtMajorVersion);
    TensorRTTypeConverter typeConverter(ctx, loweringOptions);
    TensorRTConversionTarget target(*ctx, typeConverter);
    populateStablehloToTensorRtConversionPattern(typeConverter, patterns);
    populateChloToTensorRtLegalityAndPatterns(typeConverter, target, patterns);

    // Analyze the convertible operations.
    ConversionConfig conversionConfig;
    DenseSet<Operation *> legalizedOps;
    conversionConfig.legalizableOps = &legalizedOps;
    if (failed(applyAnalysisConversion(op, target, std::move(patterns),
                                       conversionConfig)))
      emitError(op->getLoc()) << "failed to apply TensorRT conversion analysis";

    if (!legalizedOps.contains(op))
      return false;

    if (!disallowShapeTensorCalculations)
      return true;

    return !plan::detail::shouldRunOnHost(op, *solver);
  };
  return opts;
}

std::unique_ptr<Pass> TensorRTClusterKindAttr::getClusterKindPass() const {
  return nullptr;
}

int64_t TensorRTClusterKindAttr::getClusterBenefit() const {
  return getBenefit();
}

std::function<bool(const Cluster &)>
TensorRTClusterKindAttr::getClusterFilter() const {
  // Disregard the cluster if it is all constant ops.
  return [](const Cluster &cluster) -> bool {
    return !llvm::all_of(cluster, [](Operation *op) {
      return op->hasTrait<OpTrait::ConstantLike>() ||
             llvm::isa<plan::WithShapeOp, plan::WithValuesOp>(op);
    });
  };
}

std::optional<OutlineRegionOptions>
TensorRTClusterKindAttr::getClusterOutliningOptions(
    MLIRContext *ctx, SymbolTable &moduleSymbolTable) const {
  return {};
}
