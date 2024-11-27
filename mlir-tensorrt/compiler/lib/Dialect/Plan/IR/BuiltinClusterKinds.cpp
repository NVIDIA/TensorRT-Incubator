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
#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Utils/ShapeInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

int64_t HostClusterKindAttr::getClusterBenefit() const { return getBenefit(); }

/// ClusteringOpts that identifies groups of `stablehlo` ops that can be
/// converted to scalars and will be clustered into scalar cluster.
FailureOr<ClusteringOpts>
HostClusterKindAttr::getClusterKindOptions(Operation *op,
                                           DataFlowSolver &solver) const {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [&solver](Operation *op) {
    if (llvm::isa<plan::WithValuesOp>(op))
      return true;
    return plan::detail::shouldRunOnHost(op, solver);
  };
  return opts;
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
  return [](const Cluster &cluster) {
    return !llvm::all_of(cluster, [](Operation *op) {
      return op->hasTrait<OpTrait::ConstantLike>() ||
             llvm::isa<plan::WithValuesOp>(op);
    });
  };
}

//===----------------------------------------------------------------------===//
// TensorRTClusterKindAttr
//===----------------------------------------------------------------------===//

static ShapeInfoCallbacks getShapeInfoCallbacks() {
  ShapeInfoCallbacks shapeInfoCallbacks{};
  shapeInfoCallbacks.isElementValueEqualToConstant =
      [](TensorElementValue elementValue,
         Attribute constValue) -> std::optional<bool> {
    auto withValuesOp =
        elementValue.getTensor().getDefiningOp<plan::WithValuesOp>();
    if (!withValuesOp)
      return {};
    Value element = withValuesOp.getElements()[elementValue.getLinearIndex()];

    Attribute intAttr = {};
    if (!matchPattern(element, m_Constant(&intAttr)))
      return {};
    return intAttr == constValue;
  };
  shapeInfoCallbacks.isElementValueEqualToShapeDimExtent =
      [](TensorElementValue elementValue,
         TensorShapeDimExtent dimExtent) -> std::optional<bool> {
    assert(elementValue.getTensor().getType().getElementType().isIntOrIndex() &&
           "expected int or integer tensor");
    auto withValuesOp =
        elementValue.getTensor().getDefiningOp<plan::WithValuesOp>();
    if (!withValuesOp)
      return {};

    /// Scalar value will be of type equivalent to `elementValue.tensor` element
    /// type.
    Value scalarValue =
        withValuesOp.getElements()[elementValue.getLinearIndex()];

    /// Check if it is statically known to be equal to the `dimExtent`.
    IntegerAttr constInt = {};
    if (std::optional<int64_t> staticSize = dimExtent.getConstantSize()) {
      if (matchPattern(scalarValue, m_Constant(&constInt)))
        return constInt.getValue().getSExtValue() == *staticSize;
    }

    /// Otherwise, we need to check equivalence of the dynamic values.
    /// There are two cases to consider: either both have the same type, or
    /// `plan.with_shape` may have index type scalars and `plan.with_values`
    /// will have a more specific integer type that matches the shape tensor.
    /// We can try to handle the later case where the conversion is done by
    /// `arith.index_cast`.
    /// TODO: we should change the shape materialization pass so that we infer
    /// the desired shape tensor element type and have all `plan.with_shape`
    /// materialize with that scalar type using casts.
    if (auto withShape = dimExtent.tensor.getDefiningOp<plan::WithShapeOp>()) {
      Value dimExtentValue = withShape.getShape()[dimExtent.dim];
      if (dimExtentValue == scalarValue)
        return true;
      if (auto indexCastOp =
              dyn_cast<arith::IndexCastOp>(scalarValue.getDefiningOp())) {
        if (indexCastOp.getOperand() == dimExtentValue)
          return true;
      }
    }

    return {};
  };
  return shapeInfoCallbacks;
}

/// Return true if the op is an input dialect operation.
static bool isStableHloOrChloOp(Operation *op) {
  return llvm::isa_and_present<stablehlo::StablehloDialect, chlo::ChloDialect,
                               tensorrt::TensorRTDialect>(op->getDialect());
}

/// ClusteringOpts that identifies groups of TensorRT operations and will be
/// clustered into one TensorRT function (which is eventually translated to a
/// engine).
FailureOr<ClusteringOpts>
TensorRTClusterKindAttr::getClusterKindOptions(Operation *op,
                                               DataFlowSolver &solver) const {
  // Any properties used in the returned lambdas must be copied by value,
  // otherwise it will not work correctly.
  bool disallowShapeTensorCalculations = getDisallowShapeTensorCalculations();

  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;

  std::optional<int64_t> tensorrtMajorVersion = getTensorrtMajorVersion();
  if (!tensorrtMajorVersion)
    tensorrtMajorVersion = NV_TENSORRT_MAJOR;

  MLIRContext *ctx = op->getContext();
  RewritePatternSet patterns(ctx);
  LowerToTensorRTOptions loweringOptions;
  loweringOptions.setTensorRTVersion(*tensorrtMajorVersion);
  TensorRTTypeConverter typeConverter(ctx, loweringOptions);
  TensorRTConversionTarget target(*ctx, typeConverter);
  populateStablehloToTensorRtConversionPattern(typeConverter, patterns,
                                               getShapeInfoCallbacks());
  populateChloToTensorRtLegalityAndPatterns(typeConverter, target, patterns);

  // Analyze the convertible operations.
  ConversionConfig conversionConfig;
  DenseSet<Operation *> legalizedOps;
  conversionConfig.legalizableOps = &legalizedOps;
  if (failed(applyAnalysisConversion(op, target, std::move(patterns),
                                     conversionConfig)))
    return emitError(op->getLoc())
           << "failed to apply TensorRT conversion analysis";

  opts.isClusterableOp = [solver = &solver, disallowShapeTensorCalculations,
                          legalizedOps](Operation *op) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      return false;
    if (llvm::isa<plan::WithShapeOp>(op))
      return true;
    if (llvm::isa<plan::WithValuesOp>(op))
      return !disallowShapeTensorCalculations;
    if (llvm::isa<tensorrt::TensorRTOpInterface>(op))
      return true;
    if (!isStableHloOrChloOp(op))
      return false;
    if (!legalizedOps.contains(op))
      return false;
    // Don't cluster operations inside of stablehlo ops with regions.
    // For example, if we set `disallowShapeTensorCalculations`, then
    // a parent `stablehlo.reduce` might not be clustered even though it was
    // converted. The operations inside the `stablehlo.reduce` are considered
    // legalized since the parent was legalized, but we don't want to cluster
    // them since they weren't directly replaced.
    Operation *parent = op->getParentOp();
    if (parent && isStableHloOrChloOp(parent) && legalizedOps.contains(parent))
      return false;
    if (!disallowShapeTensorCalculations)
      return true;
    return !plan::detail::shouldRunOnHost(op, *solver);
  };

  return opts;
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
