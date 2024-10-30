//===- Bufferize.cpp ------------------------------------------------------===//
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
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace plan {
#define GEN_PASS_DEF_PLANBUFFERIZEPASS
#define GEN_PASS_DEF_PLANOWNERSHIPBASEDBUFFERDEALLOCATIONPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace plan
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::plan;

/// Return true if the given load/store operation operates on memory that is not
/// accessible to the host.
template <typename OpType>
static bool isLoadStoreOnDevMem(Operation *op) {
  auto loadOp = dyn_cast<OpType>(op);
  return loadOp &&
         loadOp.getMemRefType().getMemorySpace() ==
             plan::MemorySpaceAttr::get(op->getContext(), MemorySpace::device);
}

namespace {
/// Rewrite `memref.load` that acts on device memory to first copy the buffer to
/// the host and load from the host buffer.
struct RewriteDeviceLoads : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = loadOp->getContext();
    if (loadOp.getMemref().getType().getMemorySpace() !=
        MemorySpaceAttr::get(ctx, MemorySpace::device))
      return failure();
    MemRefType memrefType = loadOp.getMemRefType();
    MemRefType hostMemRefType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        memrefType.getLayout(),
                        MemorySpaceAttr::get(ctx, MemorySpace::host_pinned));

    // For the allocation, we must assemble the dynamic shape values, if
    // present. We just use `memref.dim` to create the values.
    SmallVector<Value> dims;
    for (auto [i, extent] : llvm::enumerate(memrefType.getShape())) {
      if (!ShapedType::isDynamic(extent))
        continue;
      Location loc = loadOp.getLoc();
      Value indexVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      dims.push_back(
          rewriter.create<memref::DimOp>(loc, loadOp.getMemRef(), indexVal));
    }

    Value hostBuffer =
        rewriter.create<memref::AllocOp>(loadOp.getLoc(), hostMemRefType, dims);
    rewriter.create<memref::CopyOp>(loadOp.getLoc(), loadOp.getMemRef(),
                                    hostBuffer);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, hostBuffer,
                                                loadOp.getIndices());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ExecutorBufferizationOptions
//===----------------------------------------------------------------------===//

ExecutorBufferizationOptions::ExecutorBufferizationOptions(
    ModuleOp targetOp, plan::MemorySpace defaultMemorySpace)
    : defaultExecutorMemorySpace(defaultMemorySpace) {
  this->defaultMemorySpaceFn =
      [ctx = targetOp->getContext(),
       defaultMemorySpace](TensorType tensorType) -> std::optional<Attribute> {
    auto rtt = dyn_cast<RankedTensorType>(tensorType);
    if (rtt && isa_and_nonnull<plan::MemorySpaceAttr>(rtt.getEncoding()))
      return rtt.getEncoding();
    return plan::MemorySpaceAttr::get(ctx, defaultMemorySpace);
  };

  this->allowUnknownOps = false;
  this->bufferizeFunctionBoundaries = true;
  this->setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  this->testAnalysisOnly = false;
  this->bufferAlignment = 16;
  this->allowReturnAllocsFromLoops = true;

  // Filter out anything not in the runtime executor code.
  bufferization::OpFilter filter;
  filter.denyOperation([=](Operation *op) {
    // Deny symbol tables that are not the target op.
    if (op->hasTrait<OpTrait::SymbolTable>())
      return op != targetOp;
    // Deny operations like func.func that are not nested in target op.
    Operation *parentTable = op->getParentWithTrait<OpTrait::SymbolTable>();
    return parentTable != targetOp;
  });
  this->opFilter = std::move(filter);
}

//===----------------------------------------------------------------------===//
// ExecutorBufferize Transformation
//===----------------------------------------------------------------------===//

LogicalResult plan::executorOneShotModuleBufferize(
    ModuleOp targetOp, const ExecutorBufferizationOptions &options) {

  if (failed(bufferization::runOneShotModuleBufferize(targetOp, options)))
    return targetOp.emitError()
           << "failed to run one-shot-module-bufferize on module";

  MLIRContext *ctx = targetOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<RewriteDeviceLoads>(ctx);
  if (failed(applyPatternsAndFoldGreedily(targetOp, std::move(patterns))))
    return targetOp.emitError()
           << "failed to correct device loads after one-shot-module-bufferize";

  // Validate all accesses to ensure they are legal.
  SmallVector<Operation *> illegalOps;
  targetOp->walk([&](Operation *op) {
    if (isLoadStoreOnDevMem<memref::LoadOp>(op) ||
        isLoadStoreOnDevMem<memref::StoreOp>(op))
      illegalOps.push_back(op);
    return WalkResult::advance();
  });
  if (!illegalOps.empty()) {
    for (Operation *illegalOp : illegalOps)
      illegalOp->emitOpError() << "operation accesses device memory from the "
                                  "host after bufferization";
    return targetOp.emitError()
           << "failed to resolve device memory accesses after bufferization; "
              "consider using unified memory as a workaround";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PlanBufferizePass
//===----------------------------------------------------------------------===//

namespace {
class PlanBufferizePass
    : public plan::impl::PlanBufferizePassBase<PlanBufferizePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp op = getOperation();

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op))) {
      op.emitError() << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }

    if (failed(executorOneShotModuleBufferize(
            op, ExecutorBufferizationOptions(op))))
      return signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PlanOwnershipBasedBufferDeallocationPass
//===----------------------------------------------------------------------===//

namespace {

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional clones if
/// necessary. It uses the algorithm described at the top of the file.
struct PlanOwnershipBasedBufferDeallocationPass
    : public plan::impl::PlanOwnershipBasedBufferDeallocationPassBase<
          PlanOwnershipBasedBufferDeallocationPass> {
  using Base::Base;

  void runOnOperation() override {
    bufferization::DeallocationOptions options;
    options.privateFuncDynamicOwnership = privateFuncDynamicOwnership;
    SmallVector<func::FuncOp> hostFuncs =
        llvm::to_vector(getOperation().getOps<func::FuncOp>());

    for (auto func : hostFuncs) {
      if (func.isExternal())
        continue;

      if (failed(bufferization::deallocateBuffersOwnershipBased(func, options)))
        return signalPassFailure();
    }
  }
};
} // namespace
