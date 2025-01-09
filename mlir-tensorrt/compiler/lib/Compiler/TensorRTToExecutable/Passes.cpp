//===- Passes.cpp --------------------------------------------------------===//
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
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassOptions.h"

#ifdef MLIR_TRT_ENABLE_HLO

namespace mlirtrt::compiler {
#define GEN_PASS_DEF_OUTLINETENSORRTOPPASS
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h.inc"
} // namespace mlirtrt::compiler

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// OutlineTensorRTOpPass
//===----------------------------------------------------------------------===//

/// ClusteringOpts that identifies groups of TensorRT operations and will be
/// clustered into one TensorRT function (which is eventually translated to a
/// engine).
static FailureOr<ClusteringOpts>
getTensorRTClusteringOptions(Operation *op) {
  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = Attribute{};
  opts.isClusterableOp = [](Operation *op) {
    if (llvm::isa<tensorrt::TensorRTDialect>(op->getDialect()))
      return true;
    return false;
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
static tensorrt::TensorRTModuleOp getOrCreateTensorRTModuleOp(Operation *op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;
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
  symbolTable.insert(result, op->getParentOp() == moduleOp ? Block::iterator(op)
                                                           : Block::iterator{});
  return result;
}

static FailureOr<tensorrt::CallAllocOp>
outlineOp(RewriterBase &rewriter, tensorrt::TensorRTModuleOp trtModule, plan::InlineGroupOp op) {

  // Make the region isolated from above. This captures the input operands.
  SmallVector<Value> inputs =
      makeRegionIsolatedFromAbove(rewriter, op.getRegion());

  // Create the outlined function
  FailureOr<FunctionOpInterface> func =
      createOutlinedFunc(rewriter, op.getLoc(), trtModule,
                         "tensorrt_cluster", TypeRange(inputs), op->getResultTypes());
  if (failed(func))
    return failure();

  rewriter.setInsertionPoint(op);
  auto callOp = rewriter.create<tensorrt::CallAllocOp>(
      op.getLoc(), op.getResultTypes(), inputs,
      SymbolRefAttr::get(trtModule.getNameAttr(),
                         {FlatSymbolRefAttr::get(*func)}));

  // Populate the function entry block.
  rewriter.eraseBlock(&func->getFunctionBody().front());

  // Move region op operations to the func body.
  Operation *regionYieldOp = op.getYield();
  rewriter.inlineRegionBefore(op.getRegion(), func->getFunctionBody(),
                              func->getFunctionBody().end());
  rewriter.setInsertionPoint(regionYieldOp);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(regionYieldOp,
                                              regionYieldOp->getOperands());
  // replace the original region results.
  rewriter.replaceOp(op, callOp);

  return callOp;
}


class OutlineTensorRTOpPass
    : public compiler::impl::OutlineTensorRTOpPassBase<
          OutlineTensorRTOpPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SymbolTableCollection symbolTable;
    IRRewriter rewriter(&getContext());

    FailureOr<ClusteringOpts> opts = getTensorRTClusteringOptions(module);
    if (failed(opts)) {
      emitError(module.getLoc()) << "failed to create clustering options";
      return signalPassFailure();
    }
    // What do they do here?
    // patterns.add(*opts, createInlineGroupOp, isOpInClusterRegion,
    //             target.getClusterFilter(),
    //             PatternBenefit(target.getClusterBenefit()));

    // FailureOr<SmallVector<Operation *>> regionOps =
    //     rewrite->findClusterAndCreateRegionOp(module, rewriter);
    // if (failed(regionOps)) {
    //   emitError(module.getLoc())
    //       << "clustering rewrite " << rewrite->getTarget() << " failed ";
    //   return signalPassFailure();
    // }

    tensorrt::TensorRTModuleOp trtModuleOp = getOrCreateTensorRTModuleOp(module);

    SmallVector<plan::InlineGroupOp> clusters;
    module.walk(
        [&](plan::InlineGroupOp cluster) { clusters.push_back(cluster); });

    for (plan::InlineGroupOp cluster : clusters) {
      if (failed(outlineOp(rewriter, trtModuleOp, cluster)))
        return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pipeline Registrations
//===----------------------------------------------------------------------===//

namespace {
class TensorRTToExecutablePassPipelineOptions
    : public PassPipelineOptionsAdaptor<
          TensorRTToExecutablePassPipelineOptions,
          TensorRTToExecutableOptions> {};
} // namespace

void mlirtrt::compiler::registerTensorRTToExecutablePipelines() {
  PassPipelineRegistration<TensorRTToExecutablePassPipelineOptions>(
      "tensorrt-clustering-pipeline",
      "apply clustering to tensorrt IR",
      [](OpPassManager &pm,
         const TensorRTToExecutablePassPipelineOptions &opts) {
        TensorRTToExecutableTask::buildTensorRTClusteringPipeline(pm, opts);
      });

  PassPipelineRegistration<TensorRTToExecutablePassPipelineOptions>(
      "tensorrt-compilation-pipeline", "apply compilation post-clustering",
      [](OpPassManager &pm,
         const TensorRTToExecutablePassPipelineOptions &opts) {
        TensorRTToExecutableTask::buildPostClusteringPipeline(pm, opts);
      });
}

#endif // MLIR_TRT_ENABLE_HLO