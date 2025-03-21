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
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassOptions.h"

namespace mlirtrt::compiler {
#define GEN_PASS_DEF_OUTLINETENSORRTOPPASS
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h.inc"
} // namespace mlirtrt::compiler

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

/// ClusteringOpts that identifies groups of TensorRT operations and will be
/// clustered into one TensorRT function (which is eventually translated to a
/// engine).
static FailureOr<ClusteringOpts> getTensorRTClusteringOptions(Operation *op) {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = Attribute{};
  opts.isClusterableOp = [](Operation *op) {
    return llvm::isa_and_present<tensorrt::TensorRTDialect>(op->getDialect());
  };

  return opts;
}

/// Create a `func.func` operation that represents `regionOp` and inserts into
/// the `module` SymbolTable. The function is given a name starting with
/// `nameBase` but may have numbers appended in order to unique the name. The
/// created function has argument/result types as indicated by the parameters.
static FailureOr<FunctionOpInterface>
createOutlinedFunc(RewriterBase &rewriter, Location loc, Operation *module,
                   StringRef nameBase, TypeRange funcArgTypes,
                   TypeRange funcResultTypes) {
  OpBuilder::InsertionGuard g(rewriter);

  // Create the func for outlining the region body.
  FunctionType type =
      FunctionType::get(rewriter.getContext(), funcArgTypes, funcResultTypes);
  auto outlinedFunc = func::FuncOp::create(loc, nameBase, type, {});
  Block *funcBody = outlinedFunc.addEntryBlock();

  // Add an empty terminator.
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(loc);

  // Insert into the module.
  SymbolTable(module).insert(outlinedFunc,
                             module->getRegions().front().front().end());

  // Tag the function with a UnitAttr for identifying the different kinds of
  // functions based on the cluster type.
  return cast<FunctionOpInterface>(outlinedFunc.getOperation());
}

/// Given the `op`, find the closest ModuleOp and check if the module has a
/// `tensorrt.module` operation in it. If it does, then return the existing
/// `tensorrt.module` operation. Otherwise, create a new `tensorrt.module`.
static tensorrt::TensorRTModuleOp
getOrCreateTensorRTModuleOp(ModuleOp moduleOp) {
  SymbolTable symbolTable(moduleOp);
  tensorrt::TensorRTModuleOp result = nullptr;
  for (auto trtModuleOp :
       moduleOp.getBody()->getOps<tensorrt::TensorRTModuleOp>()) {
    result = trtModuleOp;
    break;
  }
  if (result)
    return result;

  // Create the function. Symbol name de-duplication occurs with insert into the
  // symbol table.
  result = tensorrt::TensorRTModuleOp::create(moduleOp.getLoc(), "trt_engines");
  symbolTable.insert(result);
  return result;
}

static FailureOr<tensorrt::CallAllocOp>
outlineOp(RewriterBase &rewriter, tensorrt::TensorRTModuleOp trtModule,
          const Cluster &cluster) {
  auto inlineGroupOp =
      cast<plan::InlineGroupOp>(mlir::createRegionOpFromCluster(
          cluster, rewriter,
          [](OpBuilder &b, Location loc, TypeRange types, Attribute target) {
            auto regionOp = b.create<plan::InlineGroupOp>(loc, types, target);
            b.setInsertionPointToStart(&regionOp.getRegion().emplaceBlock());
            b.create<plan::YieldOp>(loc);
            return regionOp;
          }));

  // Make the region isolated from above. This captures the input operands.
  SmallVector<Value> inputs =
      makeRegionIsolatedFromAbove(rewriter, inlineGroupOp.getRegion());

  // Create the outlined function
  FailureOr<FunctionOpInterface> func = createOutlinedFunc(
      rewriter, inlineGroupOp.getLoc(), trtModule, "tensorrt_cluster",
      TypeRange(inputs), inlineGroupOp->getResultTypes());
  if (failed(func))
    return failure();

  StringRef tensorrtShapeBoundsAttrName =
      mlir::tensorrt::TensorRTDialect::getShapeProfileArgAttrName();
  func::FuncOp funcContainingCluster =
      cluster.back()->getParentOfType<func::FuncOp>();
  SmallVector<Attribute> profileAttrsPerInput;
  for (Value v : inputs) {
    auto rtt = dyn_cast<RankedTensorType>(v.getType());
    if (!rtt || rtt.hasStaticShape()) {
      profileAttrsPerInput.push_back(Attribute{});
      continue;
    }

    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg ||
        blockArg.getOwner()->getParentOp() != funcContainingCluster) {
      return emitError(blockArg.getLoc())
             << "Block argument is not part of the signature of the function "
                "containing this TRT cluster";
    }

    int64_t argIndex = blockArg.getArgNumber();
    profileAttrsPerInput.push_back(
        funcContainingCluster.getArgAttrOfType<tensorrt::ShapeProfileAttr>(
            argIndex, tensorrtShapeBoundsAttrName));

    if (!profileAttrsPerInput.back()) {
      return emitError(blockArg.getLoc())
             << "Profile attribute (" << tensorrtShapeBoundsAttrName
             << ") of argument " << argIndex << " is not set";
    }
  }

  for (unsigned idx = 0; idx < func->getNumArguments(); idx++) {
    if (!profileAttrsPerInput[idx])
      continue;
    func->setArgAttr(idx, tensorrtShapeBoundsAttrName,
                     profileAttrsPerInput[idx]);
  }

  rewriter.setInsertionPoint(inlineGroupOp);
  auto callOp = rewriter.create<tensorrt::CallAllocOp>(
      inlineGroupOp.getLoc(), inlineGroupOp.getResultTypes(), inputs,
      SymbolRefAttr::get(trtModule.getNameAttr(),
                         {FlatSymbolRefAttr::get(*func)}));

  // Populate the function entry block.
  rewriter.eraseBlock(&func->getFunctionBody().front());

  // Move region op operations to the func body.
  Operation *regionYieldOp = inlineGroupOp.getYield();
  rewriter.inlineRegionBefore(inlineGroupOp.getRegion(),
                              func->getFunctionBody(),
                              func->getFunctionBody().end());
  rewriter.setInsertionPoint(regionYieldOp);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(regionYieldOp,
                                              regionYieldOp->getOperands());
  // replace the original region results.
  rewriter.replaceOp(inlineGroupOp, callOp);

  return callOp;
}

namespace {

//===----------------------------------------------------------------------===//
// OutlineTensorRTOpPass
//===----------------------------------------------------------------------===//
class OutlineTensorRTOpPass
    : public compiler::impl::OutlineTensorRTOpPassBase<OutlineTensorRTOpPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    FailureOr<ClusteringOpts> opts = getTensorRTClusteringOptions(module);
    if (failed(opts)) {
      emitError(module.getLoc()) << "failed to create clustering options";
      return signalPassFailure();
    }

    FailureOr<SmallVector<Cluster>> clusters =
        mlir::analyzeAndClusterOperations(module, *opts);
    if (failed(clusters)) {
      emitError(module.getLoc()) << "failed to cluster operations";
      return signalPassFailure();
    }

    tensorrt::TensorRTModuleOp trtModule = getOrCreateTensorRTModuleOp(module);

    for (const auto &cluster : *clusters) {
      if (failed(outlineOp(rewriter, trtModule, cluster)))
        return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

void mlirtrt::compiler::registerTensorRTToExecutablePipelines() {
  PassPipelineRegistration<TensorRTToExecutableOptions>(
      "tensorrt-clustering-pipeline", "apply clustering to tensorrt IR",
      [](OpPassManager &pm, const TensorRTToExecutableOptions &opts) {
        TensorRTToExecutableTask::buildTensorRTClusteringPipeline(pm, opts);
      });

  PassPipelineRegistration<TensorRTToExecutableOptions>(
      "tensorrt-compilation-pipeline", "apply compilation post-clustering",
      [](OpPassManager &pm, const TensorRTToExecutableOptions &opts) {
        TensorRTToExecutableTask::buildPostClusteringPipeline(pm, opts);
      });
}
