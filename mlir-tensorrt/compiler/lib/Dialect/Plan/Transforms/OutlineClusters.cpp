//===- OutlineClusters.cpp  -----------------------------------------------===//
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
/// Implementation of the `plan-outline-clusters` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "plan-outline-clusters"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

namespace mlir::plan {
#define GEN_PASS_DEF_OUTLINECLUSTERSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

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

/// Given the `op`, find the closest ModuleOp and check if the module has a
/// `tensorrt.module` operation in it. If it does, then return the existing
/// `tensorrt.module` operation. Otherwise, create a new `tensorrt.module`.
static tensorrt::TensorRTModuleOp getOrCreateTensorRTModuleOp(Operation *op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;
  SymbolTable symbolTable(moduleOp);
  tensorrt::TensorRTModuleOp result = nullptr;
  for (auto kernelModuleOp :
       moduleOp.getBody()->getOps<tensorrt::TensorRTModuleOp>()) {
    result = kernelModuleOp;
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

static ClusterKindAttrInterface getClusterTargetForRegionOp(Operation *op) {
  if (auto regionOp = dyn_cast<plan::InlineGroupOp>(op))
    return cast<ClusterKindAttrInterface>(regionOp.getTarget());
  if (auto regionOp = dyn_cast<plan::InlineClosedGroupOp>(op))
    return cast<ClusterKindAttrInterface>(regionOp.getTarget());
  llvm_unreachable("unknown cluster region op kind");
}

/// Returns the paramters that should be used for region outlining for a

static FailureOr<OutlineRegionOptions>
getOutliningParam(Operation *op, SymbolTable &moduleSymbolTable) {
  ClusterKindAttrInterface target = getClusterTargetForRegionOp(op);
  std::optional<OutlineRegionOptions> opts =
      target.getClusterOutliningOptions(op->getContext(), moduleSymbolTable);
  if (!opts)
    return failure();
  return *opts;
}

static FailureOr<tensorrt::ShapeProfileAttr>
getTensorRTShapeProfile(plan::BoundsAttr attr, Value v) {
  // The TensorRT group formation pass doesn't give any information about "opt"
  // profiles, so we just choose the midpoint of the upper and lower bounds.
  auto getProfileAttr = [&](ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
    SmallVector<int64_t> opt;
    opt.reserve(lb.size());
    for (auto [l, u] : llvm::zip_equal(lb, ub))
      opt.push_back((l + u) / 2);
    return tensorrt::ShapeProfileAttr::get(attr.getContext(), lb, opt, ub);
  };

  RankedTensorType rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt || attr.isNone())
    return failure();

  auto apIntConvertToI64 = [](const APInt &apInt) {
    return apInt.getSExtValue();
  };
  if (attr.isValueBound())
    return getProfileAttr(
        llvm::map_to_vector(attr.getMinValues().getValues<APInt>(),
                            apIntConvertToI64),
        llvm::map_to_vector(attr.getMaxValues().getValues<APInt>(),
                            apIntConvertToI64));

  if (rtt.hasStaticShape())
    return getProfileAttr(rtt.getShape(), rtt.getShape());

  assert(attr.isShapeBound() && "expected shape bound");
  return getProfileAttr(attr.getMinShape(), attr.getMaxShape());
}

static LogicalResult outlineTensorRTRegion(RewriterBase &rewriter,
                                           plan::InlineClosedGroupOp op) {
  tensorrt::TensorRTModuleOp trtModuleOp = getOrCreateTensorRTModuleOp(op);
  auto funcArgTypes = llvm::to_vector(TypeRange(op.getInputs()));
  FailureOr<FunctionOpInterface> func = createOutlinedFunc(
      rewriter, op.getLoc(), op, trtModuleOp, "tensorrt_cluster",
      "cluster.tensorrt", TypeRange(op.getInputs()),
      op.getYield()->getOperandTypes());
  if (failed(func))
    return failure();
  assert(func->getFunctionBody().getBlocks().size() == 1 &&
         "expected body with one block");
  func->setPublic();

  rewriter.setInsertionPoint(op);
  auto callOp = rewriter.create<tensorrt::CallOp>(
      op.getLoc(), op.getResultTypes(), op.getInputs(), op.getOuts(),
      SymbolRefAttr::get(trtModuleOp.getNameAttr(),
                         {FlatSymbolRefAttr::get(*func)}));

  // Populate the function arguments attributes.
  for (unsigned i = 0; i < (*func).getNumArguments(); i++) {
    BoundsAttr srcAttr = cast<BoundsAttr>(op.getInputAttrs()[i]);
    // We may have scalar (index|signless int)-typed values since we haven't
    // eliminated `plan.(with_shape|with_values)` ops yet.
    if (!op.argHasTensorType(i) || srcAttr.isNone())
      continue;
    FailureOr<tensorrt::ShapeProfileAttr> boundAttr =
        getTensorRTShapeProfile(srcAttr, op.getInputs()[i]);
    if (failed(boundAttr))
      return op->emitOpError("failed to create TensorRT shape profile "
                             "attribute from Plan BoundsAttr for argument #")
             << i << " (" << srcAttr << ")";
    if (srcAttr.isShapeBound()) {
      func->setArgAttr(i,
                       tensorrt::TensorRTDialect::getShapeProfileArgAttrName(),
                       *boundAttr);
      continue;
    }
    assert(srcAttr.isValueBound() && "expected value bound or shape bound");
    func->setArgAttr(
        i, tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName(),
        *boundAttr);
    func->setArgAttr(i, mlir::getHostTensorArgAttrName(),
                     rewriter.getUnitAttr());
  }
  // Populate the function result attributes.
  for (unsigned i = 0; i < (*func).getNumResults(); i++) {
    BoundsAttr srcAttr = cast<BoundsAttr>(op.getResAttrs()[i]);
    if (srcAttr.isNone())
      continue;
    FailureOr<tensorrt::ShapeProfileAttr> boundsAttr =
        getTensorRTShapeProfile(srcAttr, op.getResults()[i]);
    if (failed(boundsAttr))
      return op->emitOpError("failed to create TensorRT shape profile "
                             "attribute from Plan BoundsAttr for result #")
             << i << " (" << srcAttr << ")";
    if (srcAttr.isShapeBound()) {
      func->setResultAttr(
          i, tensorrt::TensorRTDialect::getShapeProfileArgAttrName(),
          *boundsAttr);
      continue;
    }
    assert(srcAttr.isValueBound() && "expected value bound or shape bound");
    func->setResultAttr(
        i, tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName(),
        *boundsAttr);
    func->setResultAttr(i, mlir::getHostTensorArgAttrName(),
                        rewriter.getUnitAttr());
  }

  // Populate the function entry block.
  rewriter.eraseBlock(&func->getFunctionBody().front());

  // Move private decomposition funcs associated with all `stablehlo.composite`
  // ops to the `tensorrt.module` op. This is needed since `tensorrt.module` op
  // has its own symbol table.
  SymbolTableCollection symbolTable;
  for (auto compositeOp : op.getBody().getOps<stablehlo::CompositeOp>()) {
    auto decompositionFunc = dyn_cast_if_present<func::FuncOp>(
        symbolTable.lookupSymbolIn(op->getParentOfType<ModuleOp>(),
                                   compositeOp.getDecompositionAttr()));
    if (!decompositionFunc)
      return emitError(compositeOp.getLoc())
             << "failed to lookup stablehlo.composite decomposition "
                "function: "
             << compositeOp.getDecompositionAttr();
    rewriter.moveOpAfter(decompositionFunc, func->getOperation());
  }

  // Move region op operations to the func body.
  Operation *regionYieldOp = op.getYield();
  rewriter.inlineRegionBefore(op.getRegion(), func->getFunctionBody(),
                              func->getFunctionBody().end());
  rewriter.setInsertionPoint(regionYieldOp);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(regionYieldOp,
                                              regionYieldOp->getOperands());

  // Erase the DPS arugments, which now should be unused.
  if (llvm::any_of(func->getArguments().take_back(op.getOuts().size()),
                   [](BlockArgument arg) { return !arg.use_empty(); }))
    return failure();
  func->getFunctionBody().front().eraseArguments(op.getInputs().size(),
                                                 op.getOuts().size());

  // replace the original region results.
  rewriter.replaceOp(op, callOp);
  return success();
}

/// Create outlined functions for each `scf.execute_region` operation within
/// `region`.
static FailureOr<SmallVector<FunctionOpInterface>>
createFunctionsFromRegions(RewriterBase &rewriter, Region &region,
                           SymbolTable &moduleSymbolTable) {
  SmallVector<FunctionOpInterface> outlinedFuncs;

  WalkResult result = region.walk([&](Operation *op) {
    if (!isa<plan::InlineGroupOp, plan::InlineClosedGroupOp>(op))
      return WalkResult::advance();

    if (!isa<TensorRTClusterKindAttr>(getClusterTargetForRegionOp(op))) {
      FailureOr<OutlineRegionOptions> opts =
          getOutliningParam(op, moduleSymbolTable);
      if (failed(opts))
        return WalkResult::interrupt();
      FailureOr<std::pair<FunctionOpInterface, SetVector<Value>>>
          outlineResult = outlineRegionOp(rewriter, op, *opts);
      if (failed(outlineResult)) {
        emitError(op->getLoc())
            << "failed to outline cluster region op to function";
        return WalkResult::interrupt();
      }
      auto [outlinedFunc, callOperands] = *outlineResult;
      outlinedFuncs.push_back(outlinedFunc);
      return WalkResult::advance();
    }

    if (auto dpsGroup = dyn_cast<plan::InlineClosedGroupOp>(op)) {
      if (failed(outlineTensorRTRegion(rewriter, dpsGroup)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  return outlinedFuncs;
}

namespace {
class OutlineClustersPass
    : public plan::impl::OutlineClustersPassBase<OutlineClustersPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable moduleSymbolTable(module);

    SmallVector<func::FuncOp> funcs = llvm::to_vector(llvm::make_filter_range(
        module.getOps<func::FuncOp>(), [](func::FuncOp func) {
          return !func.isDeclaration() && !func.isExternal() &&
                 !(func.isPrivate() && func->hasAttr("plan.decomposition"));
        }));

    IRRewriter rewriter(module->getContext());
    for (func::FuncOp func : funcs) {
      SmallVector<plan::InlineGroupOp> clusters;
      func->walk([&](plan::InlineGroupOp clusterOp) {
        if (!isa<ClusterKindAttrInterface>(clusterOp.getTarget()))
          return WalkResult::advance();
        clusters.push_back(clusterOp);
        return WalkResult::skip();
      });

      if (failed(createFunctionsFromRegions(rewriter, func.getFunctionBody(),
                                            moduleSymbolTable)))
        return signalPassFailure();
    }
  }
};
} // namespace
