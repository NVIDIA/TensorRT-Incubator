//===- PrepareLinalgForCodegen.cpp ----------------------------------------===//
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
///===- PrepareLinalgForCodegen.cpp ---------------------------------------===//
//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Definition of pass that performs rewrites on linalg operations to prepare
/// for codegen schedule generation.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_PREPARELINALGFORCODEGENPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

namespace {

/// Rewrite a `linalg.map` operation that yields a value defined above as a
/// `linalg.fill`.
struct RewriteMapAsGeneric : public OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MapOp op,
                                PatternRewriter &rewriter) const override {
    Operation *term = op.getBody()->getTerminator();
    if (!op.getInputs().empty() || op->getNumResults() != 1 ||
        !op.getBody()->getOps<linalg::IndexOp>().empty())
      return failure();

    // Inline the body above and replace with a fill.
    rewriter.inlineBlockBefore(op.getBody(), op);
    auto yieldValue = term->getOperand(0);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, yieldValue, op.getInit());
    rewriter.eraseOp(term);
    return success();
  }
};

/// Rewrite `tensor.pad` to a linalg generic op.
struct RewritePadToGeneric : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PadOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Operation *> result =
        mlir::linalg::rewriteInDestinationPassingStyle(rewriter, op);
    if (failed(result))
      return failure();
    return success();
  }
};
} // namespace

static FailureOr<Value> getOrCreateDestination(RewriterBase &rewriter,
                                               Location loc, Value v,
                                               ArrayRef<int64_t> permutation) {
  TensorType tensorType = cast<TensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes;
  if (!tensorType.hasStaticShape() && isa<OpResult>(v)) {
    ReifiedRankedShapedTypeDims reifiedShapes;
    if (failed(reifyResultShapes(rewriter, v.getDefiningOp(), reifiedShapes)))
      return failure();
    mixedSizes = reifiedShapes[cast<OpResult>(v).getResultNumber()];
  } else if (tensorType.hasStaticShape()) {
    for (int64_t sz : tensorType.getShape())
      mixedSizes.push_back(rewriter.getIndexAttr(sz));
  } else {
    return failure();
  }

  applyPermutationToVector(mixedSizes, permutation);
  return rewriter
      .create<tensor::EmptyOp>(loc, mixedSizes, tensorType.getElementType())
      .getResult();
}

/// For a given operand at position `idx` of a matmul-like `op`, transpose the
/// operand so that the reduction dimension is last (contiguous dimension) and
/// return the transposed value and the new indexing map that should be used. If
/// the reduction dimension is already last, return the exiting operand and
/// indexing map.
static FailureOr<std::pair<Value, AffineMap>>
transposeOperandToPutReductionDimLast(RewriterBase &rewriter,
                                      linalg::GenericOp op, unsigned idx,
                                      unsigned reductionDim) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  OpOperand *operand = op.getDpsInputOperand(idx);
  AffineMap operandMap = op.getMatchingIndexingMap(operand);
  if (!operandMap.isProjectedPermutation(false))
    return failure();

  SmallVector<int64_t> affineMapPerm = operandMap.compose(
      llvm::to_vector(llvm::seq<int64_t>(0, op.getNumLoops())));
  auto redDimIt = llvm::find(affineMapPerm, reductionDim);
  if (redDimIt == affineMapPerm.end())
    return failure();
  unsigned reductionDimPos = std::distance(affineMapPerm.begin(), redDimIt);
  // Nothing to do. Return the original operand and indexing map.
  if (reductionDimPos == affineMapPerm.size() - 1)
    return std::make_pair(operand->get(), operandMap);
  // Otherwise, transpose the operand so that the reduction dimension comes
  // last.
  std::swap(affineMapPerm[reductionDimPos], affineMapPerm.back());
  SmallVector<int64_t> transposePerm = to_vector(llvm::seq<int64_t>(
      0, cast<TensorType>(operand->get().getType()).getRank()));
  std::swap(transposePerm[reductionDimPos], transposePerm.back());

  // Create the output operand.
  FailureOr<Value> out =
      getOrCreateDestination(rewriter, loc, operand->get(), transposePerm);
  if (failed(out))
    return failure();
  Value transposedOperand =
      rewriter
          .create<linalg::TransposeOp>(op.getLoc(), operand->get(), *out,
                                       transposePerm)
          ->getResult(0);

  // Create the new indexing map. We should just need to switch the dimension
  // positions.
  SmallVector<AffineExpr> newMapExprs;
  for (auto dim : affineMapPerm)
    newMapExprs.push_back(rewriter.getAffineDimExpr(dim));
  auto newMap = AffineMap::get(operandMap.getNumInputs(), 0, newMapExprs, ctx);
  return std::make_pair(transposedOperand, newMap);
}

namespace {

/// Given a linalg operation, ensure reduction dimensions are the last
/// dimensions with respect to the loop ordering.
struct InterchangeReductionDimsToEnd
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumReductionLoops() == 0 || op.getNumParallelLoops() == 0)
      return failure();
    SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();
    SmallVector<unsigned> permutation;
    SmallVector<unsigned> redDims;
    op.getReductionDims(redDims);
    for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
      if (iterType == utils::IteratorType::parallel)
        permutation.push_back(idx);
    }
    permutation.append(redDims);
    if (permutation ==
        llvm::to_vector(llvm::seq<unsigned>(0, op.getNumLoops())))
      return failure();

    FailureOr<linalg::GenericOp> result =
        linalg::interchangeGenericOp(rewriter, op, permutation);
    if (failed(result))
      return failure();
    return success();
  }
};

/// Given a matrix-multiplication like operation, transpose operands to ensure
/// that the reduction dimension comes last in each operand's indexing map. This
/// helps make the transposition required for efficient use of TensorCores
/// explicit, and the transposition should be pulled into the consumer during
/// tile-and-fuse.
struct RelayoutMatMulOperands : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto matcher =
        LinalgOpMatcher()
            .setNumReductionLoops(1)
            .setNumDpsInits(1)
            .setNumDpsInputs(2)
            .setNumLoopsBounds(3, std::numeric_limits<unsigned>::max())
            .iteratorIsReduction(-1)
            .addRegionMatchRootedAtOutput(
                0, LinalgOpMatcher::predicates::IsMatMul());
    if (!matcher.match(op.getOperation()))
      return failure();

    // Transpose operands appropriately.
    int64_t reductionDim = op.getNumLoops() - 1;
    SmallVector<Value> operands;
    SmallVector<AffineMap> newIndexingMaps;
    for (unsigned i = 0; i < 2; i++) {
      FailureOr<std::pair<Value, AffineMap>> v =
          transposeOperandToPutReductionDimLast(rewriter, op, i, reductionDim);
      if (failed(v))
        return failure();
      auto [value, map] = *v;
      operands.push_back(value);
      newIndexingMaps.push_back(map);
    }
    llvm::append_range(newIndexingMaps,
                       ArrayRef(op.getIndexingMapsArray()).drop_front(2));
    if (llvm::equal(operands, op.getOperands().take_front(2)))
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperand(0, operands[0]);
      op->setOperand(1, operands[1]);
      op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newIndexingMaps));
    });

    return success();
  }
};

/// Given a Conv2D operation in NHWC HWCF format, convert to two linalg
/// operations (im2col + reduction).
struct ConvertConv2DToMatMul
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::pair<Operation *, Operation *>> result =
        linalg_ext::rewriteInIm2Col(rewriter, op);
    if (failed(result))
      return failure();
    return success();
  }
};

/// Rewrite conv2d operations so that the filters are in the form `fhwc`.
struct ConvertConv2DFilterHwcfToFhwc
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op,
                                PatternRewriter &rewriter) const override {
    return linalg_ext::rewriteConv2dFilterToHwcf(rewriter, op);
  }
};

/// Simplify extract of fill.
struct ReplaceExtractOfFill : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    linalg::LinalgOp producer =
        op.getTensor().getDefiningOp<linalg::LinalgOp>();
    if (!producer || !LinalgOpMatcher::getFillMatcher().match(producer))
      return failure();
    rewriter.replaceOp(op, producer.getDpsInputOperand(0)->get());
    return success();
  }
};

/// Simplify extract of map-like linalg operation (one input that produces the
/// output by point-wise set of unary ops or binary ops that utilize constants).
struct ReplaceExtractOfLinalgMap : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    linalg::LinalgOp producer =
        op.getTensor().getDefiningOp<linalg::LinalgOp>();
    if (!producer || !LinalgOpMatcher::getUnaryLikeMatcher().match(producer))
      return failure();

    SmallVector<Value> indices(op.getIndices());
    AffineMap inputMap =
        producer.getMatchingIndexingMap(producer.getDpsInputOperand(0));
    assert(inputMap.isPermutation() &&
           inputMap.getNumResults() ==
               cast<RankedTensorType>(producer->getResultTypes().front())
                   .getRank() &&
           "expected permutation map");

    auto extractOp = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), producer.getDpsInputOperand(0)->get(),
        applyPermutationMap<Value>(inputMap, indices));

    // Inline the linalg op body into the current position.
    IRMapping mapping;
    Block *body = producer.getBlock();
    auto bodyInputArgs = producer.getRegionInputArgs();
    assert(bodyInputArgs.size() == 1 && "expected single input arg");
    mapping.map(bodyInputArgs[0], extractOp.getResult());
    for (auto &nestedOp : body->without_terminator()) {
      Operation *clone = rewriter.clone(nestedOp, mapping);
      mapping.map(nestedOp.getResults(), clone->getResults());
    }
    Operation *term = body->getTerminator();
    assert(term->getNumOperands() == 1 && "expected single operand terminator");
    Value yieldedValue = mapping.lookup(term->getOperand(0));
    rewriter.replaceOp(op, yieldedValue);
    return success();
  }
};

} // namespace

void kernel::populatePrepareLinalgForCodegenPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ReplaceExtractOfFill, ReplaceExtractOfLinalgMap>(
      ctx, PatternBenefit(200));
  patterns.add<ConvertConv2DFilterHwcfToFhwc>(ctx, PatternBenefit(200));
  patterns.add<ConvertConv2DToMatMul>(ctx, PatternBenefit(100));
  patterns.add<RewriteMapAsGeneric, RewritePadToGeneric, RelayoutMatMulOperands,
               InterchangeReductionDimsToEnd>(ctx);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
}

namespace {

class PrepareLinalgForCodegenPass
    : public kernel::impl::PrepareLinalgForCodegenPassBase<
          PrepareLinalgForCodegenPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();
    RewritePatternSet patterns(ctx);
    populatePrepareLinalgForCodegenPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply " << getArgument() << " rewrite patterns";
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }
};
} // namespace
