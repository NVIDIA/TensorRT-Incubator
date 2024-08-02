//===- ApplyBugWorkarounds.cpp --------------------------------------------===//
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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_APPLYBUGWORKAROUNDSPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir
using namespace mlir;
using namespace mlir::tensorrt;

namespace {
/// Expand `tensorrt.abs` for i8 so that dimensions are at least 2D.
struct Unary8IWorkaround : public OpRewritePattern<tensorrt::UnaryOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().getElementType().isInteger(8) ||
        op.getType().getRank() >= 2)
      return failure();

    RankedTensorType inputType = op.getType().cast<RankedTensorType>();
    int64_t rank = inputType.getRank();
    auto expandedShape = RankedTensorType::Builder(inputType);
    for (int64_t i = rank; i < 2; i++)
      expandedShape.insertDim(1, 0);
    Value expanded = rewriter.create<tensorrt::ExpandRankOp>(
        op.getLoc(), Type(expandedShape), op.getInput());
    Value expandedUnary =
        rewriter.create<UnaryOp>(op.getLoc(), expanded, op.getUnaryOperation());
    rewriter.replaceOpWithNewOp<CollapseRankOp>(op, op.getType(),
                                                expandedUnary);
    return success();
  }
};

/// If TensorRT strongly typed is enabled, rewrite `tensorrt.matrix_multiply` as
/// vector multiply if the following conditions hold true,
/// 1. LHS is `1xK` and RHS `1xK` (this will mean that LHS matrix operation is
/// `KNONE` and RHS matrix operation is `kTRANSPOSE`).
/// 2. `tensorrt.collapse_rank` op follows `tensorrt.matrix_multiply` and it
/// collapses a tensor to a scalar (that means all tensor dims are 1).
struct MatToVecMatmul : public OpRewritePattern<tensorrt::CollapseRankOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CollapseRankOp op,
                                PatternRewriter &rewriter) const override {
    // CollapseRank op remove all ones from all unit dims tensor resulting in
    // scalar output. We are being conservative by forcing op input rank to 2
    // since this is WAR for a specific case.
    if (op.getType().getRank() != 0 || op.getInput().getType().getRank() != 2)
      return failure();

    auto matmulOp = op.getInput().getDefiningOp<tensorrt::MatrixMultiplyOp>();
    if (!matmulOp)
      return failure();
    RankedTensorType lhs = matmulOp.getInput0().getType();
    RankedTensorType rhs = matmulOp.getInput1().getType();
    int64_t lhsVectorLength =
        matmulOp.getOp0() == tensorrt::MatrixOperation::kNONE
            ? lhs.getDimSize(lhs.getRank() - 1)
            : lhs.getDimSize(lhs.getRank() - 2);
    int64_t rhsVectorLength =
        matmulOp.getOp1() == tensorrt::MatrixOperation::kNONE
            ? rhs.getDimSize(rhs.getRank() - 2)
            : rhs.getDimSize(rhs.getRank() - 1);
    RankedTensorType lhsVectorType =
        RankedTensorType::get({lhsVectorLength}, lhs.getElementType());
    RankedTensorType rhsVectorType =
        RankedTensorType::get({rhsVectorLength}, rhs.getElementType());
    Value lhsCollapsed = rewriter.create<tensorrt::CollapseRankOp>(
        op->getLoc(), lhsVectorType, matmulOp.getInput0());
    Value rhsCollapsed = rewriter.create<tensorrt::CollapseRankOp>(
        op->getLoc(), rhsVectorType, matmulOp.getInput1());
    rewriter.replaceOpWithNewOp<tensorrt::MatrixMultiplyOp>(
        op,
        /*input0=*/lhsCollapsed,
        /*input1=*/rhsCollapsed,
        /*op0=*/tensorrt::MatrixOperation::kVECTOR,
        /*op1=*/tensorrt::MatrixOperation::kVECTOR);
    return success();
  }
};
} // namespace

/// Given a string `[MAJOR].[MINNOR]`, return a tuple of integers for the major
/// and minor numbers.
static FailureOr<std::pair<int, int>> parseVersion(StringRef tensorrtVersion) {
  bool error = false;
  auto nums = llvm::to_vector<2>(
      llvm::map_range(llvm::split(tensorrtVersion, "."), [&](StringRef sub) {
        int num;
        error &= llvm::to_integer<int>(sub, num, 10);
        return num;
      }));
  if (error || nums.size() != 2)
    return failure();
  return std::make_pair(nums[0], nums[1]);
}
namespace {

/// Applies the workarounds above according to the specified TensorRT version.
class ApplyBugWorkaroundsPass
    : public tensorrt::impl::ApplyBugWorkaroundsPassBase<
          ApplyBugWorkaroundsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    MLIRContext *ctx = &getContext();
    FailureOr<std::pair<int, int>> version = parseVersion(tensorrtVersion);
    if (failed(version)) {
      emitError(UnknownLoc::get(ctx))
          << "failed to parse version string \"" << tensorrtVersion << "\"";
      return signalPassFailure();
    }
    auto [major, minor] = *version;

    // WAR for TRT 8 for minor versions 8.6 and below.
    if (major == 8 && minor <= 6)
      patterns.add<Unary8IWorkaround>(ctx);
    if (tensorrtStronglyTyped)
      patterns.add<MatToVecMatmul>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
