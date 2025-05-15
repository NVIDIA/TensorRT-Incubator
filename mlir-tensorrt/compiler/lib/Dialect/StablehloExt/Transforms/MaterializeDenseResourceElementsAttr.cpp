//===- MaterializeDenseResourceElementsAttr.cpp ---------------------------===//
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
/// Materialize `DenseElementsAttr` to `DenseResourceElementsAttr`.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_MATERIALIZEDENSERESOURCEELEMENTSATTRPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;

static std::string getUniqueName(Type type) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "k_";
  cast<ShapedType>(type).getElementType().print(os);
  return name;
}

template <typename T>
static DenseResourceElementsAttr
createDenseResourceElementsAttrImpl(DenseElementsAttr value) {
  AsmResourceBlob blob = HeapAsmResourceBlob::allocateAndCopyWithAlign(
      value.getRawData(), alignof(T), false);
  assert(((blob.getData().size() % sizeof(T)) == 0) &&
         "size mismatch between expected element width and blob size");
  return DenseResourceElementsAttr::get(
      value.getType(), getUniqueName(value.getType()), std::move(blob));
}

static DenseResourceElementsAttr
createDenseResourceElementsAttr(DenseElementsAttr value) {
  Type elementType = value.getElementType();
  if (elementType.isInteger(1))
    // Alignment of `bool` is 1. 8 bool values can be stored in 1 byte.
    return createDenseResourceElementsAttrImpl<bool>(value);
  if (elementType.isInteger(4) || isa<Float4E2M1FNType>(elementType) ||
      elementType.isInteger(8) || isa<Float8E4M3FNType>(elementType) ||
      elementType.isUnsignedInteger(8))
    // MLIR represents `i4` and `fp4E2M1FN` as lower 4 bits of `i8`.
    // Thus, blob can hold `i4`/`fp4E2M1FN` with same alignment as `i8`.
    // `i8`, `ui8` and `f8E4M3FN` has obvious 1 byte alignment.
    return createDenseResourceElementsAttrImpl<int8_t>(value);
  if (elementType.isInteger(16) || elementType.isUnsignedInteger(16))
    return createDenseResourceElementsAttrImpl<int16_t>(value);
  if (elementType.isInteger(32) || elementType.isUnsignedInteger(32))
    return createDenseResourceElementsAttrImpl<int32_t>(value);
  if (elementType.isInteger(64) || elementType.isUnsignedInteger(64))
    return createDenseResourceElementsAttrImpl<int64_t>(value);
  if (elementType.isF16() || elementType.isBF16())
    // Alignment of `fp16` and `bf16` is 2 i.e. same as `int16_t`.
    return createDenseResourceElementsAttrImpl<int16_t>(value);
  if (elementType.isF32())
    return createDenseResourceElementsAttrImpl<float>(value);
  if (elementType.isF64())
    return createDenseResourceElementsAttrImpl<double>(value);

  emitError(UnknownLoc::get(value.getContext()))
      << "Unsupported element type for DenseElementsAttr -> "
         "DenseResourceElementsAttr conversion!";
  return nullptr;
}

namespace {
/// Materialize `DenseElementsAttr` into `DenseResourceElementsAttr` in
/// `stablehlo.constant` operations. Conversion only happens if number of
/// elements in `DenseElementsAttr` exceed `elementCountThreshold`.
class MaterializeConstantOp : public OpRewritePattern<stablehlo::ConstantOp> {
public:
  MaterializeConstantOp(MLIRContext *context, int64_t elementCountThreshold)
      : OpRewritePattern<stablehlo::ConstantOp>(context),
        elementCountThreshold(elementCountThreshold) {}

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr value = op.getValue();
    if (auto denseElementsAttr = dyn_cast<DenseElementsAttr>(value)) {
      // We keep splat constant as is.
      if (denseElementsAttr.isSplat()) {
        return failure();
      }
      // Check if the number of elements exceeds the threshold
      if (denseElementsAttr.getNumElements() < elementCountThreshold) {
        return failure();
      }
      auto denseResourceElementsAttr =
          createDenseResourceElementsAttr(denseElementsAttr);
      if (!denseResourceElementsAttr) {
        return failure();
      }
      op.setValueAttr(denseResourceElementsAttr);
      return success();
    }
    return failure();
  }

private:
  int64_t elementCountThreshold;
};

class MaterializeDenseResourceElementsAttrPass
    : public mlir::stablehlo_ext::impl::
          MaterializeDenseResourceElementsAttrPassBase<
              MaterializeDenseResourceElementsAttrPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<MaterializeConstantOp>(ctx, elementCountThreshold);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc()) << "failed to run patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace