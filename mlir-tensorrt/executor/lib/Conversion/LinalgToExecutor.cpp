//===- LinalgToExecutor.cpp -----------------------------------------------===//
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
/// Implementation of linalg-to-executor lowerings.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::executor {
#define GEN_PASS_DEF_CONVERTLINALGTOEXECUTORPASS
#include "mlir-executor/Conversion/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

using namespace mlir;

using executor::ConvertOpToExecutorPattern;
using executor::MemoryType;
using executor::MemRefDescriptor;

namespace {
template <typename OpType>
struct ConvertLinalgOpToExecutorPattern
    : public ConvertOpToExecutorPattern<OpType> {

  using ConvertOpToExecutorPattern<OpType>::ConvertOpToExecutorPattern;

  MLIRContext *ctx = this->getContext();
  executor::MemoryTypeAttr deviceMemorySpace =
      executor::MemoryTypeAttr::get(ctx, MemoryType::device);
  Type i32Type = IntegerType::get(ctx, 32);
  Type i16Type = IntegerType::get(ctx, 16);
  Type i8Type = IntegerType::get(ctx, 8);
  Type devicePointerType = executor::PointerType::get(ctx, MemoryType::device);
  Type hostPointerType = executor::PointerType::get(ctx, MemoryType::host);
  Type indexType = this->getTypeConverter()->getIndexType();
  executor::ExecutorCallBuilder deviceFillI32 = {
      this->getContext(),
      "__cuda_memset_32",
      {},
      {devicePointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i32Type}};
  executor::ExecutorCallBuilder deviceFillI16 = {
      this->getContext(),
      "__cuda_memset_16",
      {},
      {devicePointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i16Type}};
  executor::ExecutorCallBuilder deviceFillI8 = {
      this->getContext(),
      "__cuda_memset_8",
      {},
      {devicePointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i8Type}};
  executor::ExecutorCallBuilder hostFillI32 = {
      this->getContext(),
      "__memset_32",
      {},
      {hostPointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i32Type}};
  executor::ExecutorCallBuilder hostFillI16 = {
      this->getContext(),
      "__memset_16",
      {},
      {hostPointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i16Type}};
  executor::ExecutorCallBuilder hostFillI8 = {
      this->getContext(),
      "__memset_8",
      {},
      {hostPointerType, /*offset*/ indexType, /*size*/ indexType,
       /*value*/ i8Type}};
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Op Lowerings
//===----------------------------------------------------------------------===//

namespace {
/// Convert `linalg.fill` on contiguous buffers to cuda/host memset builtins.
struct LinalgFillToExecutorPattern
    : public ConvertLinalgOpToExecutorPattern<linalg::FillOp> {
  using ConvertLinalgOpToExecutorPattern::ConvertLinalgOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();
    MemRefType memrefType = cast<MemRefType>(op.getOutputs().front().getType());
    if (!isContiguous(memrefType))
      return rewriter.notifyMatchFailure(op, "memref type not contiguous");

    // Note: the `getTypeSizeInBits` will return 1 for i1. We need to round to
    // byte size.
    Type elementType = memrefType.getElementType();
    uint64_t typeSizeInBytes = getDataLayout().getTypeSize(elementType);
    unsigned fillIntegerTypeBitWidth =
        llvm::PowerOf2Ceil(typeSizeInBytes) * CHAR_BIT;
    if (fillIntegerTypeBitWidth > 32)
      return rewriter.notifyMatchFailure(op, "type size > 4 bytes");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefDescriptor dest(adaptor.getOutputs().front(), memrefType);
    auto alignedPtr =
        cast<TypedValue<executor::PointerType>>(dest.alignedPtr(b));
    executor::MemoryType addrSpace = alignedPtr.getType().getAddressSpace();
    if (addrSpace != MemoryType::device && addrSpace != MemoryType::host)
      return rewriter.notifyMatchFailure(
          op, "only device and host memory spaces are supported");

    Value offsetBytes =
        convertOffsetInElementsToBytes(b, dest.offset(b), memrefType);
    Value sizeBytes = convertOffsetInElementsToBytes(
        b, dest.shapeVolumeInElements(b), memrefType);

    IntegerType requiredIntegerFillType =
        rewriter.getIntegerType(fillIntegerTypeBitWidth);

    Value fillValue = adaptor.getInputs().front();
    assert(fillValue.getType() == memrefType.getElementType() &&
           "expected fill value type to match output element type");

    if (fillValue.getType() != requiredIntegerFillType) {
      // If the fill value does not match the required integer type, then we
      // require to do a cast. If we are an integer of smaller width than
      // required, zero-extend to the required width. Otherwise, if we only
      // differ by type (e.g. f32 vs i32), then just do a bit-wise cast.
      if (fillValue.getType().isSignlessInteger() &&
          fillValue.getType().getIntOrFloatBitWidth() < fillIntegerTypeBitWidth)
        fillValue =
            b.create<executor::ZExtOp>(requiredIntegerFillType, fillValue);
      else if (fillValue.getType().getIntOrFloatBitWidth() ==
               fillIntegerTypeBitWidth)
        fillValue =
            b.create<executor::BitcastOp>(requiredIntegerFillType, fillValue);
      else
        return rewriter.notifyMatchFailure(
            op, "no path to convert fill value to i32, i16, or i8");
    }

    if (addrSpace == MemoryType::device) {
      switch (fillIntegerTypeBitWidth) {
      case 32:
        deviceFillI32.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                             {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      case 16:
        deviceFillI16.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                             {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      case 8:
        deviceFillI8.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                            {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      default:
        return failure();
      }
    } else {
      // This condition is enforced above.
      assert(addrSpace == MemoryType::host && "expected 'host' space");
      switch (fillIntegerTypeBitWidth) {
      case 32:
        hostFillI32.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                           {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      case 16:
        hostFillI16.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                           {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      case 8:
        hostFillI8.create(b, op.getLoc(), op->getParentOfType<ModuleOp>(),
                          {alignedPtr, offsetBytes, sizeBytes, fillValue});
        break;
      default:
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct GenericToFillPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    // Must have no inputs, one output.
    if (generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1)
      return failure();

    // Check body.
    auto &block = generic.getRegion().front();
    if (!llvm::hasSingleElement(block))
      return failure();

    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();

    // The yielded value must not be defined inside the block (must be
    // loop-invariant).
    Value fillVal = yield.getValues().front();
    if (fillVal.getParentBlock() == &block)
      return failure();

    // Replace with linalg.fill.
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        generic, fillVal, generic.getDpsInitOperand(0)->get());
    return success();
  }
};

} // namespace

void executor::populateLinalgToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter) {
  patterns.add<LinalgFillToExecutorPattern>(typeConverter,
                                            patterns.getContext());
}

namespace {
class ConvertLinalgToExecutorPass
    : public executor::impl::ConvertLinalgToExecutorPassBase<
          ConvertLinalgToExecutorPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // Step 1: Apply linalg.generic -> linalg.fill simplification as
    // preprocessing
    {
      RewritePatternSet patterns(ctx);
      patterns.add<GenericToFillPattern>(ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to simplify linalg.generic to linalg.fill";
        return signalPassFailure();
      }
    }

    // Step 2: Convert linalg operations to executor
    executor::ExecutorConversionTarget target(*ctx);
    LowerToExecutorOptions opts;
    // We allow index type during memref lowering prior to lowering of certain
    // executor ops to func-calls.
    opts.indexType = IntegerType::get(ctx, indexBitwidth);

    FailureOr<DataLayout> dataLayout =
        executor::setDataLayoutSpec(op, indexBitwidth, 64);
    if (failed(dataLayout)) {
      emitError(op->getLoc())
          << "failed to set DataLayout; op has DLTI spec that is "
             "inconsistent with provided options";
      return signalPassFailure();
    }
    ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));

    // We create executor constants in this pass. Mark them as legal.
    target.addIllegalDialect<linalg::LinalgDialect>();

    RewritePatternSet patterns(ctx);
    executor::populateLinalgToExecutorPatterns(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to perform conversion in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
