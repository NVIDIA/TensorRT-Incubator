//===- KernelToLLVM.cpp ---------------------------------------------------===//
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
/// Implementation of the `convert-kernel-to-emitc` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTKERNELTOLLVM
#include "mlir-kernel/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Convert `kernel.call` into an LLVM IR call. We utilize the CUDA Runtime
/// pre-C11-compatible function `cudaLaunchKernelExtC` to perform the launch.
struct KernelCallConverter : public ConvertOpToLLVMPattern<kernel::CallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(kernel::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Pack the result types into a struct.
    // CallOp should be bufferized.
    if (callOp.getNumResults() != 0)
      return failure();

    MLIRContext *ctx = rewriter.getContext();

    Type i32Type = rewriter.getIntegerType(32);
    Type i64Type = rewriter.getIntegerType(64);

    /// We must use LLVMarray type in the LLVM struct type for
    /// `cudaLaunchConfig_t` to get the correct size and alignment. For
    /// populating the actual fields of grid and block shape in the struct, we
    /// use GEP to access scalar positions and only load/store scalars.
    auto dim3Type = LLVM::LLVMArrayType::get(i32Type, 3);

    Location loc = callOp.getLoc();

    auto zeroI32 = createIndexAttrConstant(rewriter, loc, i32Type, 0);
    auto zeroI64 = createIndexAttrConstant(rewriter, loc, i64Type, 0);
    auto zero = createIndexAttrConstant(rewriter, loc, getIndexType(), 0);
    auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto nullStream = rewriter.create<LLVM::ZeroOp>(loc, llvmPtrType);
    Value one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                  rewriter.getIndexAttr(1));

    // First we use the LLVMTypeConverter to expand the operand list into
    // scalars; this flattens and expands the MemRef struct types into scalars.
    // We only need to perform this scalarization for the input/output operands
    // of the call.
    SmallVector<Value> originalCallOperands(callOp.getInputs());
    llvm::append_range(originalCallOperands, callOp.getOuts());
    SmallVector<Value> adaptorCallOperands(adaptor.getInputs());
    llvm::append_range(adaptorCallOperands, adaptor.getOuts());

    SmallVector<Value> storagePtrs;
    SmallVector<Value, 4> scalarizedCallOperands =
        getTypeConverter()->promoteOperands(
            loc, originalCallOperands, adaptorCallOperands, rewriter, false);

    // Passing parameters for the CUDA kernel via `cudaLaunchKernelExtC`
    // requires that we pass an array-of-pointers that each point to the storage
    // for the corresponding parameter. Therefore, we need to promote all
    // SSA values in `scalarizedCallOperands` to stack storage since LLVM SSA
    // values don't have address, e.g. they are r-values in C terminology). This
    // is the inverse transform of what people call `mem2reg`.

    // Promote to stack storage.
    for (Value v : scalarizedCallOperands) {
      assert((v.getType().isIntOrIndexOrFloat() ||
              isa<LLVM::LLVMPointerType>(v.getType())) &&
             "expected call operands to be scalars");
      // Allocate a scalar on the stack.
      Value valuePtr =
          rewriter.create<LLVM::AllocaOp>(loc, llvmPtrType, v.getType(), one);
      // Store the scalar.
      rewriter.create<LLVM::StoreOp>(loc, v, valuePtr);
      storagePtrs.push_back(valuePtr);
    }

    // Create and populate the array-of-pointers that is required by the launch
    // config.
    auto operandPtrStorageType =
        LLVM::LLVMArrayType::get(llvmPtrType, storagePtrs.size());
    auto argPtrsPtr = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, operandPtrStorageType, one);
    for (auto [idx, value] : llvm::enumerate(storagePtrs)) {
      auto gepOp = rewriter.create<LLVM::GEPOp>(
          loc, llvmPtrType, operandPtrStorageType, argPtrsPtr,
          ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(zero), LLVM::GEPArg(idx)});
      rewriter.create<LLVM::StoreOp>(loc, value, gepOp);
    }

    // Create and populate the launch config.
    auto launchConfigType = LLVM::LLVMStructType::getLiteral(
        ctx, {dim3Type, dim3Type, i64Type, llvmPtrType, llvmPtrType, i32Type});
    auto launchConfigPtr = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, launchConfigType, one);

    // Fill in the block/grid shape fields.
    // As noted above, we must perform stores at the scalar level.
    std::array<SmallVector<Value, 3>, 2> gridBlockRanges = {
        adaptor.getGridSize(), adaptor.getBlockSize()};
    for (auto [idx, scalarsToAssign] : llvm::enumerate(gridBlockRanges)) {
      if (scalarsToAssign.size() > 3)
        return failure();
      while (scalarsToAssign.size() < 3) {
        scalarsToAssign.push_back(rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));
      }
      for (auto [i, scalar] : llvm::enumerate(scalarsToAssign)) {
        assert(scalar.getType() == i32Type ||
               scalar.getType().isSignlessInteger(64) &&
                   "expected i32 or i64 scalar");
        Value memberPtr = rewriter.create<LLVM::GEPOp>(
            loc, llvmPtrType, launchConfigType, launchConfigPtr,
            ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(zero), LLVM::GEPArg(idx),
                                   LLVM::GEPArg(i)});
        if (scalar.getType() != i32Type)
          scalar = rewriter.create<LLVM::TruncOp>(loc, i32Type, scalar);
        rewriter.create<LLVM::StoreOp>(loc, scalar, memberPtr);
      }
    }

    // Fill in the rest of the fields for the launch config.
    SmallVector<Value, 4> assignmentValues = {zeroI64, nullStream, nullStream,
                                              zeroI32};
    for (auto [idx, valueToAssign] : llvm::enumerate(assignmentValues)) {
      Value memberPtr = rewriter.create<LLVM::GEPOp>(
          loc, llvmPtrType, launchConfigType, launchConfigPtr,
          ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(zero), LLVM::GEPArg(idx + 2)});
      rewriter.create<LLVM::StoreOp>(loc, valueToAssign, memberPtr);
    }

    // Get the address of the kernel to be launched. Note that this isn't
    // actually how CUDA works, since this address should point to a CUDA
    // function stub, but this will translate to EmitC nicely.
    LLVM::LLVMFuncOp llvmCallee =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
            callOp, callOp.getKernelSymAttr());
    if (!llvmCallee)
      return failure();
    auto calleeAddr = rewriter.create<LLVM::AddressOfOp>(
        loc, llvmPtrType, FlatSymbolRefAttr::get(llvmCallee));

    SmallVector<Value> launchCallOperands = {launchConfigPtr, calleeAddr,
                                             argPtrsPtr};

    FailureOr<LLVM::LLVMFuncOp> launchFunc = LLVM::lookupOrCreateFn(
        rewriter, callOp->getParentOfType<ModuleOp>(), "cudaLaunchKernelExC",
        llvm::to_vector(TypeRange(launchCallOperands)), i32Type);
    if (failed(launchFunc))
      return failure();

    rewriter.create<LLVM::CallOp>(callOp.getLoc(), *launchFunc,
                                  launchCallOperands);

    rewriter.eraseOp(callOp);
    return success();
  }
};

class ConvertKernelToLLVM
    : public mlir::impl::ConvertKernelToLLVMBase<ConvertKernelToLLVM> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // If KernelModule exists, inline the kernel now.
    IRRewriter rewriter(moduleOp);
    SmallVector<gpu::GPUModuleOp> kernelModules =
        llvm::to_vector(moduleOp.getOps<gpu::GPUModuleOp>());
    SymbolTableCollection symbolTableCollection;
    SymbolUserMap symbolUserMap(symbolTableCollection, moduleOp);
    for (auto kernelModule : kernelModules) {
      for (Operation &op : kernelModule.getOps()) {
        if (!isa<LLVM::LLVMDialect>(op.getDialect())) {
          emitError(op.getLoc())
              << getArgument()
              << " expects all operations in gpu.module operations to be "
                 "lowered to LLVM as a precondition";
          return signalPassFailure();
        }
      }

      // We are going to inline all contents of the kernel module into the outer
      // module. Before we do that, for each function in the kernel module, get
      // symbol users outside the `gpu.module` and remove the root symbol
      // refering to the `gpu.module` symbol name.
      for (auto funcOp : kernelModule.getOps<FunctionOpInterface>()) {
        if (funcOp.isDeclaration())
          continue;

        // We can't use `SymbolTable::replaceAllSymbolUses` since that function
        // is just for renaming a symbol, retaining original scope. We want to
        // "move" the symbol into parent table.
        SymbolRefAttr oldAttr = SymbolRefAttr::get(
            rewriter.getContext(), kernelModule.getSymName(),
            ArrayRef<FlatSymbolRefAttr>{FlatSymbolRefAttr::get(funcOp)});
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](SymbolRefAttr attr) -> std::pair<Attribute, WalkResult> {
              if (attr == oldAttr)
                return {FlatSymbolRefAttr::get(attr.getLeafReference()),
                        WalkResult::skip()};
              return {attr, WalkResult::skip()};
            });
        for (Operation *user : symbolUserMap.getUsers(funcOp))
          replacer.replaceElementsIn(user);
      }

      rewriter.inlineBlockBefore(kernelModule.getBody(0), kernelModule);
      rewriter.eraseOp(kernelModule);
    }

    LLVMTypeConverter typeConverter(&getContext());

    LLVMConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<kernel::KernelDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelCallConverter>(typeConverter);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
