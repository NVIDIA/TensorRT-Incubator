//===- RefineArgumentLayouts.cpp ------------------------------------------===//
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
/// Refine the layout attribute of MemRef kernel function arguments is the
/// actual types passed by callers are only identity layouts.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_KERNELREFINEARGUMENTLAYOUTSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

/// Return a new memref type with the default identity layout representation
/// and whose other properties are the same as the original type.
static MemRefType getIdentityMemRefType(MemRefType type) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         MemRefLayoutAttrInterface{}, type.getMemorySpace());
}

/// Returns true if the argument at `argIdx` of all callers can be refined to
/// an identity layout.
static bool canRefineToIdentityLayout(unsigned argIdx,
                                      ArrayRef<CallOpInterface> callers) {
  for (CallOpInterface caller : callers) {
    Value arg = caller.getArgOperands()[argIdx];
    auto callOperandType = dyn_cast<MemRefType>(arg.getType());
    if (!callOperandType)
      return false;
    if (callOperandType.getLayout().isIdentity())
      continue;
    auto castOp = arg.getDefiningOp<memref::CastOp>();
    if (!castOp)
      return false;
    auto castSrcType = dyn_cast<MemRefType>(castOp.getSource().getType());
    if (!castSrcType.getLayout().isIdentity())
      return false;
  }
  return true;
}

/// Returns a bitmask indicating which arguments can be refined to an identity
/// layout.
static llvm::SmallBitVector
getRefinedArgumentTypes(ArrayRef<CallOpInterface> callers,
                        FunctionOpInterface callee) {
  unsigned numArgs = callee.getNumArguments();
  llvm::SmallBitVector result(numArgs, false);
  for (unsigned i = 0; i < numArgs; i++) {
    MemRefType calleeArgType =
        dyn_cast<MemRefType>(callee.getArgument(i).getType());
    if (!calleeArgType || calleeArgType.getLayout().isIdentity())
      continue;

    if (canRefineToIdentityLayout(i, callers))
      result.set(i);
  }
  return result;
}

namespace {
class KernelRefineArgumentLayoutsPassPass
    : public kernel::impl::KernelRefineArgumentLayoutsPassBase<
          KernelRefineArgumentLayoutsPassPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    llvm::DenseMap<FunctionOpInterface, SmallVector<CallOpInterface>>
        funcsToCallers;

    ModuleOp op = getOperation();
    SymbolTableCollection symbolTable;
    SymbolUserMap userMap(symbolTable, op);
    IRRewriter rewriter(ctx);

    SmallVector<func::FuncOp> kernelFuncs;
    for (auto kernelModule : op.getOps<gpu::GPUModuleOp>()) {
      for (auto kernelFunc : kernelModule.getOps<func::FuncOp>()) {
        if (!llvm::all_of(
                userMap.getUsers(kernelFunc), [&](Operation *symbolUser) {
                  if (!isa<CallOpInterface>(symbolUser)) {
                    symbolUser->emitWarning(
                        "cannot refine the argument types of kernel function ")
                        << kernelFunc.getName()
                        << " due to unknown symbol user";
                    return false;
                  }
                  return true;
                }))
          continue;
        funcsToCallers[kernelFunc] = llvm::map_to_vector(
            userMap.getUsers(kernelFunc),
            [](Operation *user) { return cast<CallOpInterface>(user); });
      }
    }

    for (auto [kernel, callers] : funcsToCallers) {
      if (!kernel.getResultTypes().empty())
        continue;

      llvm::SmallBitVector refineableArgs =
          getRefinedArgumentTypes(callers, kernel);

      OpBuilder::InsertionGuard g(rewriter);

      unsigned numArgs = kernel.getNumArguments();
      SmallVector<Type> argTypes(kernel.getArgumentTypes());
      SmallVector<Type> resultTypes(kernel.getResultTypes());
      for (unsigned i = 0; i < numArgs; i++) {
        auto argType =
            llvm::dyn_cast<MemRefType>(kernel.getArgument(i).getType());
        if (!argType)
          continue;
        if (argType.getLayout().isIdentity() || !refineableArgs[i])
          continue;

        MemRefType newKernelArgType = getIdentityMemRefType(argType);
        argTypes[i] = newKernelArgType;
        rewriter.modifyOpInPlace(
            kernel, [&kernel = kernel, &i, &newKernelArgType]() {
              kernel.getArgument(i).setType(newKernelArgType);
            });

        // Update callee operands by casting to the refined type. The new cast
        // will cancel out with the existing cast. Because there are several
        // equivalent ways of describing "identity layout", we can't rely on
        // just replacing each operand with the source of the current cast.
        for (CallOpInterface caller : callers) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPoint(caller);
          OpOperand &arg =
              cast<CallOpInterface>(*caller).getArgOperandsMutable()[i];
          auto newCastOp = rewriter.create<memref::CastOp>(
              caller.getLoc(),
              getIdentityMemRefType(cast<MemRefType>(arg.get().getType())),
              arg.get());
          rewriter.modifyOpInPlace(caller, [&]() { arg.set(newCastOp); });
        }

        // Some users of the block argument, e.g.
        // `memref.load|store|assume_alignment`, can accommodate a direct update
        // without creating a cast. Others cannot (e.g. `memref.subview`).
        // To be on the safe side, we cast the block argument type back to the
        // original (more general) layout using `memref.cast` and use that for
        // replacement outside of a few pre-specified user types.
        BlockArgument arg = kernel.getArgument(i);
        rewriter.setInsertionPointToStart(&kernel.getFunctionBody().front());
        auto castOp =
            rewriter.create<memref::CastOp>(arg.getLoc(), argType, arg);
        auto shouldReplaceWithCast = [&](OpOperand &use) {
          // Can't replace cast's use with itself.
          if (use.getOwner() == castOp)
            return false;
          // Allow updating in-place for a set of operations where we know that
          // is always valid.
          return !isa<memref::LoadOp, memref::StoreOp, memref::CastOp>(
              use.getOwner());
        };
        rewriter.replaceUsesWithIf(arg, castOp, shouldReplaceWithCast);
      }
      kernel.setType(FunctionType::get(ctx, argTypes, resultTypes));
    }
  }
};
} // namespace
