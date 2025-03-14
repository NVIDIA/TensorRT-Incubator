//===- ExpandOps.cpp ------------------------------------------------------===//
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
/// Implementation of the `executor-expand-ops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTOREXPANDOPSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

/// Return the value for `ofr` or create a constant value if required. If `ofr`
/// is a value, its type is checked against `intType`. If it is an attribute,
/// then it is checked that it can be losslessly converted to `intType`.
static FailureOr<Value> getOrCreateAndCheckIndexValue(RewriterBase &rewriter,
                                                      GetOffsetOp op,
                                                      IntegerType intType,
                                                      OpFoldResult ofr) {
  Location loc = op.getLoc();
  if (auto val = dyn_cast<Value>(ofr)) {
    if (val.getType() != intType)
      return failure();
    return val;
  }

  IntegerAttr srcAttr = cast<IntegerAttr>(cast<Attribute>(ofr));
  APInt srcInt = srcAttr.getValue();
  if (srcInt.getBitWidth() == intType.getWidth())
    return rewriter
        .create<ConstantOp>(loc, rewriter.getIntegerAttr(intType, srcInt))
        .getResult();

  if (srcInt.getBitWidth() < intType.getWidth())
    return rewriter
        .create<ConstantOp>(loc, rewriter.getIntegerAttr(
                                     intType, srcInt.zext(intType.getWidth())))
        .getResult();

  if (!srcInt.isIntN(intType.getWidth()))
    return failure();

  return rewriter
      .create<ConstantOp>(loc, rewriter.getIntegerAttr(
                                   intType, srcInt.trunc(intType.getWidth())))
      .getResult();
}

/// Lower the `executor.getoffset` operation into more primitive ops.
static FailureOr<Value> lowerGetOffset(RewriterBase &rewriter,
                                       const DataLayout &layout,
                                       GetOffsetOp op) {
  SmallVector<OpFoldResult> indices = op.getIndices();
  Location loc = op.getLoc();

  // The type we should use for index calculations.
  IntegerType computeType = rewriter.getIntegerType(
      layout.getTypeSizeInBits(rewriter.getIndexType()));

  if (computeType != op.getType())
    return op.emitOpError() << "result type (" << op.getType()
                            << ") does not match the width of the IndexType ("
                            << computeType << ") specified by the DataLayout";

  auto getIndexConst = [&](int64_t value) -> Value {
    return rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(computeType, value));
  };

  FailureOr<Value> indexValue =
      getOrCreateAndCheckIndexValue(rewriter, op, computeType, indices[0]);
  if (failed(indexValue))
    return op.emitOpError()
           << llvm::formatv("index #0 ({0}) cannot be converted losslessly to "
                            "the width of the "
                            "IndexType ({1}) specified by the data layout",
                            indices[0], computeType);

  Value offset = rewriter.create<MulIOp>(
      loc, *indexValue, getIndexConst(layout.getTypeSize(op.getElemType())));

  Type currentType = op.getElemType();
  for (OpFoldResult index : llvm::drop_begin(indices)) {
    if (auto structType = dyn_cast<TableType>(currentType)) {
      ArrayRef<Type> body = structType.getBody();
      // This is a plain cast since the verifier checks that indices into
      // aggregates are constants.
      IntegerAttr indexStatic = llvm::cast<IntegerAttr>(cast<Attribute>(index));
      assert(static_cast<unsigned>(indexStatic.getInt()) < body.size() &&
             "getoffset index is out-of-bounds for indexed aggregate");
      for (int64_t i = 0, e = indexStatic.getInt(); i < e; i++) {
        llvm::TypeSize typeSize = layout.getTypeSize(body[i]);
        IntegerAttr alignment =
            rewriter.getUI32IntegerAttr(layout.getTypeABIAlignment(body[i]));
        offset = rewriter.create<AlignToOp>(loc, offset, alignment);
        offset = rewriter.create<AddIOp>(loc, offset, getIndexConst(typeSize));
      }
      IntegerAttr alignment = rewriter.getUI32IntegerAttr(
          layout.getTypeABIAlignment(body[indexStatic.getInt()]));
      offset = rewriter.create<AlignToOp>(loc, offset, alignment);
      currentType = body[indexStatic.getInt()];
      continue;
    }

    // This could also be an assertion. If this occurs then the the op should be
    // invalid.
    return op->emitOpError("failed to lower invalid executor.getoffset op");
  }
  return offset;
}

namespace {

/// Lowers `executor.getoffset` by creating more primitive arithmetic
/// operations. May also produce `executor.alignto`.
struct LowerGetOffsetPattern : public OpRewritePattern<GetOffsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerGetOffsetPattern(const DataLayout &dataLayout, MLIRContext *ctx,
                        PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(GetOffsetOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> offset = lowerGetOffset(rewriter, dataLayout, op);
    if (failed(offset)) {
      return failure();
    }

    rewriter.replaceOp(op, *offset);
    return success();
  }

  const DataLayout &dataLayout;
};

/// Lowers `executor.alloca` by replacing with a normal allocation and adding a
/// dealloc at the end of the block.
struct LowerAllocaPattern : public OpRewritePattern<AllocaOp> {

  LowerAllocaPattern(const DataLayout &dataLayout, MLIRContext *ctx,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(AllocaOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto indexType = rewriter.getIntegerType(
        dataLayout.getTypeSizeInBits(rewriter.getIndexType()));
    Value numBytes = rewriter.create<GetOffsetOp>(
        loc, indexType, op.getElementType(),
        ArrayRef<OpFoldResult>{op.getNumElements()});
    Value alignment = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, dataLayout.getTypeABIAlignment(
                                                    op.getElementType())));
    Value alloc = rewriter.create<AllocateOp>(
        loc, PointerType::get(rewriter.getContext(), MemoryType::host),
        numBytes, alignment);
    rewriter.setInsertionPoint(op->getBlock()->getTerminator());
    rewriter.create<executor::DeallocateOp>(loc, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }

  const DataLayout &dataLayout;
};

class ExecutorExpandOpsPass
    : public executor::impl::ExecutorExpandOpsPassBase<ExecutorExpandOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout =
        dataLayoutAnalysis.getAtOrAbove(getOperation());

    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    target.addLegalDialect<executor::ExecutorDialect>();
    if (lowerGetOffset) {
      target.addIllegalOp<executor::GetOffsetOp>();
      patterns.add<LowerGetOffsetPattern>(dataLayout, ctx);
    }
    if (lowerAlloca) {
      target.addIllegalOp<executor::AllocaOp>();
      patterns.add<LowerAllocaPattern>(dataLayout, ctx);
    }

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
