#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

struct DimOfReifyRankedShapedTypeOpInterface
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() {
    OpRewritePattern<tensor::DimOp>::setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dyn_cast<OpResult>(dimOp.getSource());
    if (!dimValue)
      return failure();
    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    ReifiedRankedShapedTypeDims reifiedResultShapes;
    if (failed(reifyResultShapes(rewriter, dimValue.getOwner(),
                                 reifiedResultShapes)))
      return failure();
    unsigned resultNumber = dimValue.getResultNumber();
    // Do not apply pattern if the IR is invalid (dim out of bounds).
    if ((size_t)(*dimIndex) >= reifiedResultShapes[resultNumber].size())
      return rewriter.notifyMatchFailure(dimOp, "dimension is out of bounds");
    Value replacement = getValueOrCreateConstantIndexOp(
        rewriter, dimOp.getLoc(), reifiedResultShapes[resultNumber][*dimIndex]);
    rewriter.replaceOp(dimOp, replacement);
    return success();
  }
};

namespace {
class TestTypeInferencePass
    : public mlir::PassWrapper<TestTypeInferencePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeInferencePass)

  TestTypeInferencePass() = default;

  llvm::StringRef getArgument() const override {
    return "test-tensorrt-shape-inference";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DimOfReifyRankedShapedTypeOpInterface>(ctx);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to converge in " << getArgument();
      return signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }
};
} // namespace

namespace mlir {
void registerTestTensorRTShapeInferencePass() {
  PassRegistration<TestTypeInferencePass>();
}
} // namespace mlir
