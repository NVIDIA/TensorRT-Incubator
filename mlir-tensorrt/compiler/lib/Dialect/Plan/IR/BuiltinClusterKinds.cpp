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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Transforms/Clustering/Clustering.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::plan;

#define DEBUG_TYPE "builtin-cluster-kinds"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// Operation Placement Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the given operation should run "on the host". This means
/// that the operation can be converted to Executor IR. It derives this
/// information based on the operation, the operands, and the TensorKindAnalysis
/// information.
static bool shouldRunOnHost(Operation *op, DataFlowSolver &solver) {
  // An operation can't be placed on the host if the types are too big.
  LLVM_DEBUG(DBGS() << "should run on host? " << *op << "\n");
  auto isHostType = [](Type t) {
    return t.isIntOrIndexOrFloat() || stablehlo_ext::isScalarizableType(t);
  };
  if (!llvm::all_of(op->getResultTypes(), isHostType) ||
      !llvm::all_of(op->getOperandTypes(), isHostType)) {
    LLVM_DEBUG(DBGS() << "  types not all host compatible\n");
    return false;
  }

  // Filter for StableHLO dialect ops. Don't consider stablehlo ops nested in
  // other stablehlo ops.
  if (!isa<stablehlo::StablehloDialect>(op->getDialect()) ||
      isa<stablehlo::StablehloDialect>(op->getParentOp()->getDialect())) {
    LLVM_DEBUG(DBGS() << "  not stablehlo op\n");
    return false;
  }

  // Ignore constants. We don't cluster constants. They are cloned during the
  // outlining step.
  if (op->hasTrait<OpTrait::ConstantLike>())
    return false;

  // Filter for which operations we support on the host.
  if (!op->hasTrait<OpTrait::Elementwise>() &&
      !isa<stablehlo::ConcatenateOp, stablehlo::IotaOp, stablehlo::ReshapeOp,
           stablehlo::BroadcastInDimOp, stablehlo::SliceOp,
           stablehlo::BitcastConvertOp, stablehlo::ConvertOp,
           stablehlo::SelectOp, stablehlo::ReduceOp>(op)) {
    LLVM_DEBUG(DBGS() << "  not a supported op\n");
    return false;
  }

  // If the operation doesn't have any operands, then we can run on host if
  // the result is required on host (e.g. `stablehlo.arange : tensor<4xi32>`).
  if (op->getNumOperands() == 0) {
    LLVM_DEBUG(DBGS() << "  checking result TensorKinds\n");
    return llvm::all_of(op->getResults(), [&](Value v) {
      const auto *lattice = solver.lookupState<TensorKindLattice>(v);
      LLVM_DEBUG({
        if (lattice)
          DBGS() << "  arg: ";
        lattice->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      return lattice && !lattice->getValue().isUninitialized() &&
             lattice->getValue().isHostVisible();
    });
  }

  // If all the types are small enough and they are host tensors, then we can
  // place the computation on the host. Note that the TensorKind of the
  // results doesn't matter here. If the operands and result types are small,
  // then we can run the computation on the host as long as the inputs are on
  // the host. A result TensorKind of 'device' or 'both' just means the result
  // must be transferred to the device afterwards.
  LLVM_DEBUG(DBGS() << "  checking operand TensorKinds\n");
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(operand);
    LLVM_DEBUG({
      if (lattice)
        DBGS() << "  arg: ";
      lattice->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    return lattice && !lattice->getValue().isUninitialized() &&
           lattice->getValue().isHostVisible();
  });
}

//===----------------------------------------------------------------------===//
// HostClusterKindAttr
//===----------------------------------------------------------------------===//

std::string HostClusterKindAttr::getClusterKindName() const { return "host"; }

int64_t HostClusterKindAttr::getClusterBenefit() const { return getBenefit(); }

/// ClusteringOpts that identifies groups of `stablehlo` ops that can be
/// converted to scalars and will be clustered into scalar cluster.
ClusteringOpts
HostClusterKindAttr::getClusterKindOptions(DataFlowSolver &solver) const {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [&solver](Operation *op) {
    return shouldRunOnHost(op, solver);
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

/// Create a `func.func` operation that represents `regionOp` and inserts into
/// the `module` SymbolTable. The function is given a name starting with
/// `nameBase` but may have numbers appended in order to unique the name. The
/// created function has argument/result types as indicated by the parameters.
static FailureOr<FunctionOpInterface>
createOutlinedFunc(RewriterBase &rewriter, Location loc, Operation *regionOp,
                   Operation *module, StringRef nameBase, StringRef tagName,
                   TypeRange funcArgTypes, TypeRange funcResultTypes) {
  OpBuilder::InsertionGuard g(rewriter);

  // Create the func for outlining the region body.
  FunctionType type =
      FunctionType::get(rewriter.getContext(), funcArgTypes, funcResultTypes);
  auto outlinedFunc = mlir::func::FuncOp::create(loc, nameBase, type, {});
  Block *funcBody = outlinedFunc.addEntryBlock();

  // Add an empty terminator.
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(loc);

  // Insert into the module.
  SymbolTable(module).insert(outlinedFunc,
                             module->getRegions().front().front().end());

  // Tag the function with a UnitAttr for identifying the different kinds of
  // functions based on the cluster type.
  outlinedFunc->setAttr(tagName, rewriter.getUnitAttr());
  return cast<FunctionOpInterface>(outlinedFunc.getOperation());
}

std::optional<OutlineRegionOptions>
HostClusterKindAttr::getClusterOutliningOptions() const {
  return OutlineRegionOptions{
      /*typeConverter=*/stablehlo_ext::getScalarizationTypeConverter(),
      /*shouldCloneProducer=*/shouldCloneProducer,
      /*createFunc=*/
      [](RewriterBase &rewriter, Location loc, Operation *regionOp,
         ArrayRef<Value> callOperands, ArrayRef<Type> convertedOperandTypes,
         ArrayRef<Type> results)
          -> FailureOr<std::pair<FunctionOpInterface, SmallVector<Value>>> {
        ModuleOp module = regionOp->getParentOfType<ModuleOp>();
        FailureOr<FunctionOpInterface> func =
            createOutlinedFunc(rewriter, loc, regionOp, module, "host_cluster",
                               "cluster.host", convertedOperandTypes, results);
        if (failed(func))
          return failure();
        func->setPrivate();

        rewriter.setInsertionPoint(regionOp);
        SmallVector<Value> callReplacements =
            rewriter
                .create<func::CallOp>(loc, llvm::cast<func::FuncOp>(*func),
                                      callOperands)
                .getResults();
        return std::make_pair(*func, callReplacements);
      }};
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
ClusteringOpts
TensorRTClusterKindAttr::getClusterKindOptions(DataFlowSolver &solver) const {
  // Any properties used in the returned lambdas must be copied by value,
  // otherwise it will not work correctly.
  bool disallowShapeTensorCalculations = getDisallowShapeTensorCalculations();

  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;
  opts.isClusterableOp = [solver = &solver,
                          disallowShapeTensorCalculations](Operation *op) {
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
    loweringOptions.setI64Lowering(
        LowerToTensorRTOptions::I64Lowering::FailOnI64);
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

    return !shouldRunOnHost(op, *solver);
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
TensorRTClusterKindAttr::getClusterOutliningOptions() const {
  return {};
}
