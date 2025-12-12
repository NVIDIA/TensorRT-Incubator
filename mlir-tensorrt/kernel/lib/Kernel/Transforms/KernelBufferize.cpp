//===- KernelBufferize.cpp ------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of OneShotBufferization-related functionality for the
/// KernelDialect. This includes options for performing bufferization on
/// `gpu.module` as well as post-bufferization actions.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_KERNELBUFFERIZEPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;
using bufferization::BufferizationStatistics;
using bufferization::LayoutMapOption;
using bufferization::OneShotAnalysisState;
using bufferization::OneShotBufferizationOptions;

/// Analyze the specified functions in gpu::GPUModuleOp. These functions are
/// guaranteed not to call each other.
static LogicalResult analyzeGpuModuleOp(gpu::GPUModuleOp moduleOp,
                                        ArrayRef<func::FuncOp> funcs,
                                        OneShotAnalysisState &state,
                                        BufferizationStatistics *statistics) {
  for (func::FuncOp funcOp : funcs) {
    if (failed(analyzeOp(funcOp, state, statistics)))
      return failure();
  }
  return success();
}

/// Bufferize the specified functions in gpu::GPUModuleOp. These functions are
/// guaranteed not to call each other.
static LogicalResult
bufferizeGpuModuleOp(gpu::GPUModuleOp moduleOp, ArrayRef<func::FuncOp> funcs,
                     const OneShotBufferizationOptions &options,
                     BufferizationStatistics *statistics) {
  assert(options.bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  IRRewriter rewriter(moduleOp.getContext());
  for (func::FuncOp funcOp : funcs) {
    if (failed(bufferizeOp(funcOp, options, statistics)))
      return failure();
  }
  return success();
}

/// Analyze and bufferize gpu::GPUModuleOp. We only analyze and bufferize
/// functions that do not have any callers.
static LogicalResult
runOneShotKernelModuleBufferize(gpu::GPUModuleOp moduleOp,
                                const SymbolUserMap &useMap,
                                const OneShotBufferizationOptions &options,
                                BufferizationStatistics *statistics) {

  SmallVector<func::FuncOp> funcs;
  for (func::FuncOp func : moduleOp.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    if (llvm::any_of(useMap.getUsers(func), llvm::IsaPred<CallOpInterface>))
      continue;
    funcs.push_back(func);
  }

  OneShotAnalysisState state(moduleOp, options);
  if (failed(analyzeGpuModuleOp(moduleOp, funcs, state, statistics)))
    return failure();
  if (failed(bufferization::insertTensorCopies(moduleOp.getOperation(), state)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  if (failed(bufferizeGpuModuleOp(moduleOp, funcs, options, statistics)))
    return failure();
  return success();
}

/// materializeKernelArgAlignmentHints:
/// check the ArgAttributes of the input FunctionOp
/// materialize all "kernel.alignment" attributes
/// by adding memref.assume_alignment
/// and remove the kernel.alignment attrivutes
static LogicalResult
materializeKernelArgAlignmentHints(FunctionOpInterface &funcOp,
                                   mlir::OpBuilder &builder) {
  if (funcOp.isDeclaration())
    return success();
  builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  for (size_t arg_id = 0; arg_id < funcOp.getArguments().size(); ++arg_id) {
    mlir::IntegerAttr kernelAlignmentAttr =
        funcOp.getArgAttrOfType<IntegerAttr>(
            arg_id, mlir::kernel::KernelDialect::kKernelAlignmentArgAttrName);
    if (kernelAlignmentAttr) {
      builder.create<mlir::memref::AssumeAlignmentOp>(
          funcOp->getLoc(), funcOp.getArgument(arg_id),
          builder.getI32IntegerAttr(kernelAlignmentAttr.getInt()));
      funcOp.removeArgAttr(
          arg_id, mlir::kernel::KernelDialect::kKernelAlignmentArgAttrName);
    }
  }
  return success();
}

/// Drop function buffer results that are equivalent to block arguments.
/// TODO: this function logic is borrowed from upstream because the upstream
/// does not support invoking `mlir::dropEquivalentBufferResults` on ops other
/// than `builtin.module`.
static LogicalResult
dropEquivalentFuncBufferResults(RewriterBase &rewriter, func::FuncOp funcOp,
                                const SymbolUserMap &symbolUseMap) {
  if (funcOp.isDeclaration() || !funcOp.getBody().hasOneBlock())
    return success();

  func::ReturnOp returnOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

  // Compute erased results.
  SmallVector<Value> newReturnValues;
  BitVector erasedResultIndices(funcOp.getFunctionType().getNumResults());
  DenseMap<int64_t, int64_t> resultToArgs;
  for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
    bool erased = false;
    for (BlockArgument bbArg : funcOp.getArguments()) {
      Value val = it.value();
      while (auto castOp = val.getDefiningOp<memref::CastOp>())
        val = castOp.getSource();

      if (val == bbArg) {
        resultToArgs[it.index()] = bbArg.getArgNumber();
        erased = true;
        break;
      }
    }

    if (erased) {
      erasedResultIndices.set(it.index());
    } else {
      newReturnValues.push_back(it.value());
    }
  }

  // Update function.
  funcOp.eraseResults(erasedResultIndices);
  returnOp.getOperandsMutable().assign(newReturnValues);

  // Update function calls.
  for (Operation *symbolUser : symbolUseMap.getUsers(funcOp)) {
    func::CallOp callOp = dyn_cast<func::CallOp>(symbolUser);
    if (!callOp)
      continue;

    rewriter.setInsertionPoint(callOp);
    auto newCallOp = rewriter.create<func::CallOp>(callOp.getLoc(), funcOp,
                                                   callOp.getOperands());
    SmallVector<Value> newResults;
    int64_t nextResult = 0;
    for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
      if (!resultToArgs.count(i)) {
        // This result was not erased.
        newResults.push_back(newCallOp.getResult(nextResult++));
        continue;
      }

      // This result was erased.
      Value replacement = callOp.getOperand(resultToArgs[i]);
      Type expectedType = callOp.getResult(i).getType();
      if (replacement.getType() != expectedType) {
        // A cast must be inserted at the call site.
        replacement = rewriter.create<memref::CastOp>(
            callOp.getLoc(), expectedType, replacement);
      }
      newResults.push_back(replacement);
    }
    rewriter.replaceOp(callOp, newResults);
  }

  return success();
}

/// Drop function buffer results that are equivalent to block arguments.
static LogicalResult dropEquivalentFuncBufferResultsInKernelModule(
    RewriterBase &rewriter, gpu::GPUModuleOp module,
    const SymbolUserMap &symbolUseMap) {
  for (auto funcOp : module.getOps<func::FuncOp>())
    if (failed(dropEquivalentFuncBufferResults(rewriter, funcOp, symbolUseMap)))
      return failure();

  return success();
}

bufferization::OneShotBufferizationOptions
kernel::getKernelModuleBufferizationOptions() {
  bufferization::OneShotBufferizationOptions opts;
  opts.allowUnknownOps = false;
  opts.bufferAlignment = 16;
  opts.bufferizeFunctionBoundaries = true;
  opts.copyBeforeWrite = false;
  opts.allowReturnAllocsFromLoops = true;
  opts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::FullyDynamicLayoutMap);

  opts.memCpyFn = [](OpBuilder &b, Location loc, Value from, Value to) {
    b.create<memref::CopyOp>(loc, from, to);
    return success();
  };
  opts.allocationFn = [](OpBuilder &b, Location loc, MemRefType type,
                         ValueRange dynShape,
                         unsigned bufferAlignment) -> FailureOr<Value> {
    // We allow allocations without address spaces. We assume these are private.
    // After bufferization, we try to promote these to stack. Failure to promote
    // causes an failure.
    if (!type.getMemorySpace())
      // Use 'memref.alloc' to allocate the private memory.
      return b
          .create<memref::AllocOp>(loc, type, dynShape,
                                   bufferAlignment == 0
                                       ? IntegerAttr{}
                                       : b.getI64IntegerAttr(bufferAlignment))
          .getResult();

    // Otherwise, we check for GPU address space. Allowed options are shared and
    // private.
    auto space =
        dyn_cast_if_present<gpu::AddressSpaceAttr>(type.getMemorySpace());
    if (!space || (space.getValue() != gpu::AddressSpace::Workgroup &&
                   space.getValue() != gpu::AddressSpace::Private))
      return emitError(loc) << "bufferization attempted to allocate "
                               "memory inside of device function with an "
                               "unsupported address space attribute: "
                            << type.getMemorySpace();

    return b
        .create<memref::AllocOp>(loc, type, dynShape,
                                 bufferAlignment == 0
                                     ? IntegerAttr{}
                                     : b.getI64IntegerAttr(bufferAlignment))
        .getResult();
  };
  return opts;
}

LogicalResult
kernel::runKernelModulePostBufferizationActions(gpu::GPUModuleOp op,
                                                SymbolUserMap &symbolUserMap) {
  IRRewriter builder(op->getContext());
  for (mlir::Region &region : op->getRegions()) {
    for (auto funcOp : region.getOps<mlir::FunctionOpInterface>()) {
      if (failed(materializeKernelArgAlignmentHints(funcOp, builder)))
        return failure();
    }
  }

  RewritePatternSet patterns(op->getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, op->getContext());
  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    return emitError(op->getLoc())
           << "failed to apply loop canonicalization patterns during "
              "post-bufferization actions";

  return dropEquivalentFuncBufferResultsInKernelModule(builder, op,
                                                       symbolUserMap);
}

LogicalResult
kernel::bufferizeKernelModule(gpu::GPUModuleOp op,
                              LayoutMapOption functionBoundaryTypeConversion) {
  using bufferization::OneShotBufferizationOptions;

  OneShotBufferizationOptions options = getKernelModuleBufferizationOptions();

  SymbolTableCollection symbolTables;
  SymbolUserMap useMap(symbolTables, op);

  // Override the function boundary bufferization type.
  options.setFunctionBoundaryTypeConversion(functionBoundaryTypeConversion);
  if (failed(runOneShotKernelModuleBufferize(op, useMap, options, nullptr)))
    return failure();

  if (failed(runKernelModulePostBufferizationActions(op, useMap)))
    return failure();

  return success();
}
