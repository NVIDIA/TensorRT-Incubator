//===- KernelSegmentation.cpp ---------------------------------------------===//
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
/// Implementation of the `plan-kernel-segmentation` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-tensorrt/Backends/Kernel/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Casting.h"

namespace mtrt::compiler {
#define GEN_PASS_DEF_KERNELSEGMENTATIONPASS
#include "mlir-tensorrt/Backends/Kernel/Passes.h.inc"
} // namespace mtrt::compiler

using namespace mlir;
using namespace mlir::plan;
using namespace mtrt::compiler;

static FailureOr<SmallVector<Cluster>>
runClusteringTransforms(func::FuncOp func) {
  ClusteringOpts opts;
  opts.clusterTarget = plan::KernelBackendAttr::get(func->getContext(), 1);

  // Traversal direction shouldn't have much of an effect since we duplicate
  // producers if profitable.
  opts.bfsRootTraversal = ClusteringRootTraversalDirection::PostOrder;

  // An op is clusterable if it can be represented as a GPU kernel.
  opts.isClusterableOp = [](Operation *op) {
    auto tileableOp = dyn_cast<TilingInterface>(op);
    if (!tileableOp)
      return false;

    if (auto scatterOp = dyn_cast<kernel::ScatterOp>(op))
      return true;

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // TODO: Remove this once we have a better way to handle negative
      // dimension reversals.
      if (linalg_ext::hasNegativeMultiplicationCoefficients(
              linalgOp.getIndexingMapsArray()))
        return false;
    }

    return true;
  };

  // Fuse elementwise ops into consumer clusters.
  opts.shouldGrowClusterFn = [](Operation *producer, ClusterRange,
                                Operation *consumer, ClusterRange) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(producer);
    if (!linalgOp)
      return false;

    // If the linalg op has multiple users, don't grow. Removing these
    // dependencies should be handled by the preconditioning.
    if (!linalgOp->hasOneUse() &&
        !llvm::all_of(producer->getUsers(),
                      [&](Operation *user) { return user == consumer; }))
      return false;

    return linalgOp.getNumReductionLoops() == 0 &&
           isa<linalg::LinalgOp>(consumer);
  };

  // Don't do horizontal fusion for now.
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return false; };

  FailureOr<SmallVector<Cluster>> clusters =
      analyzeAndClusterOperations(func, opts);
  if (failed(clusters))
    return failure();
  return clusters;
}

/// Determines whether a cluster being outlined should clone a constant or
/// pass constant by value.
static bool shouldCloneProducer(Value v, Region &cluster) {
  Operation *producer = v.getDefiningOp();
  if (!producer)
    return false;

  if (llvm::isa<tensor::EmptyOp>(producer))
    return true;

  if (!producer->hasTrait<OpTrait::ConstantLike>() ||
      producer->getNumResults() != 1)
    return false;

  Type resultType = producer->getResultTypes().front();

  // Clone any scalar or vector type into the kernel function.
  if (isa<IntegerType, FloatType, IndexType, ComplexType, VectorType>(
          resultType))
    return true;

  RankedTensorType type = dyn_cast<RankedTensorType>(resultType);
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

/// Creates an empty `plan.cluster` operation for a given type of cluster
/// target.
static Operation *createInlineGroupOp(OpBuilder &b, Location loc,
                                      TypeRange types, Attribute target) {
  auto regionOp = b.create<plan::ClusterOp>(
      loc, types, cast<CompilerBackendAttrInterface>(target));
  b.setInsertionPointToStart(&regionOp.getRegion().emplaceBlock());
  b.create<plan::YieldOp>(loc);
  return regionOp;
}

/// Return a 1-to-N type converter for scalarizing Tensor types to unpacked
/// scalar types.
static TypeConverter getSignedToSignlessConverter() {
  // Add a type converter, target and source materialization to convert
  // `tensor<1xdtype>` to `dtype` and back.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type t) -> std::optional<Type> { return t; });
  typeConverter.addConversion(
      [&](Type t,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        auto rtt = dyn_cast<RankedTensorType>(t);
        if (!rtt || (!rtt.getElementType().isSignedInteger() &&
                     !rtt.getElementType().isUnsignedInteger()))
          return std::nullopt;
        result.push_back(RankedTensorType::get(
            rtt.getShape(),
            IntegerType::get(t.getContext(),
                             rtt.getElementType().getIntOrFloatBitWidth()),
            rtt.getEncoding()));
        return success();
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
         Location loc) -> SmallVector<Value> {
        if (inputs.size() != 1)
          return {};
        Value input = inputs.front();
        auto rtt = dyn_cast<RankedTensorType>(input.getType());
        if (!rtt || (!rtt.getElementType().isSignedInteger() &&
                     !rtt.getElementType().isUnsignedInteger()))
          return {};
        SmallVector<Value> result;
        llvm::append_range(
            result,
            builder.create<UnrealizedConversionCastOp>(loc, resultTypes, input)
                .getResults());
        return result;
      });
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs, Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  return typeConverter;
}

static OutlineRegionOptions
getClusterOutliningOptions(MLIRContext *ctx, SymbolTable &symbolTable) {
  OpBuilder b(ctx);
  return OutlineRegionOptions{
      /*typeConverter=*/getSignedToSignlessConverter(),
      /*shouldCloneProducer=*/shouldCloneProducer,
      /*createFunc=*/
      OutlineRegionOptions::getDefaultCreateFuncAndCallStubFunc(
          symbolTable,
          {b.getNamedAttr(getKernelGenClusterAttrName(), b.getUnitAttr())},
          "codegen_cluster")};
}

namespace {
class KernelSegmentationPass
    : public mtrt::compiler::impl::KernelSegmentationPassBase<
          KernelSegmentationPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Remove all cf::assert ops in Linalg ops so it will not error in
    // lower-to-nvvm pass
    module->walk([](cf::AssertOp op) {
      if (op->getParentOfType<linalg::LinalgOp>())
        op.erase();
    });

    OpPassManager dynamicPM;
    dynamicPM.addPass(mlir::createInlinerPass());
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();

    SymbolTable symbolTable(module);
    IRRewriter rewriter(ctx);

    SmallVector<func::FuncOp> funcs;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      funcs.push_back(func);
    }

    for (func::FuncOp func : funcs) {
      FailureOr<SmallVector<Cluster>> clusters = runClusteringTransforms(func);
      if (failed(clusters))
        return signalPassFailure();

      OutlineRegionOptions outliningOpts =
          getClusterOutliningOptions(ctx, symbolTable);
      for (const Cluster &cluster : *clusters) {

        auto regionOp = cast<ClusterOp>(
            createRegionOpFromCluster(cluster, rewriter, createInlineGroupOp));
        if (!regionOp) {
          func->emitError("failed to create region op from cluster");
          return signalPassFailure();
        }

        if (failed(outlineRegionOp(rewriter, regionOp, outliningOpts))) {
          cluster.getRoot()->emitError()
              << "failed to outline region op to func";
          return signalPassFailure();
        }
      }
    }
  }
};
} // namespace
