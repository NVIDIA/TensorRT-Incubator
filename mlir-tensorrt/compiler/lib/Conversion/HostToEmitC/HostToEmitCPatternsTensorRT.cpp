//===- HostToEmitCPatternsTensorRT.cpp ------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// TensorRT runtime op lowering patterns for `convert-host-to-emitc`.
///
/// These patterns are intentionally "thin": they mostly marshal arguments into
/// the StandaloneCPP ABI types and then call the runtime wrappers:
///   - `mtrt::tensorrt_enqueue(...)`
///   - `mtrt::tensorrt_enqueue_alloc(...)`
///
/// The generated C++ is expected to look like:
///   mtrt::UnrankedMemRef inputs[N];
///   mtrt::UnrankedMemRef outputs[M];
///   int32_t st = mtrt::tensorrt_enqueue(ctx, stream, N, inputs, M, outputs);
///   mtrt::abort_on_error(st);
//===----------------------------------------------------------------------===//

#include "HostToEmitCDetail.h"
#include "HostToEmitCDetailCommon.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"

using namespace mlir;
using namespace mlir::host_to_emitc;

namespace {

template <typename PatternT>
static FailureOr<Value>
buildUnrankedMemRefArray(const PatternT &p, ConversionPatternRewriter &rewriter,
                         Location loc, Operation *failureOp,
                         ValueRange shapedValues, TypeRange originalTypes,
                         StringRef failureMessage) {
  // Intended C++ (schematic), for each operand:
  //   mtrt::UnrankedMemRef arr[n];
  //   mtrt::PtrAndShape<rank> tmp = mtrt::make_ptr_shape_descriptor<rank>(...);
  //   arr[i] = mtrt::make_unranked_descriptor(rank, tmp);
  //
  // IMPORTANT: `mtrt::UnrankedMemRef` stores a pointer to the ranked
  // descriptor, so `tmp` must have a stable address (i.e. be a named local),
  // not a temporary expression.
  const int64_t n = static_cast<int64_t>(shapedValues.size());
  if (n == 0) {
    Type ptrTy = emitc::PointerType::get(p.unrankedDescriptorType);
    return p.getNullptr(rewriter, loc, ptrTy);
  }

  Value array = rewriter.create<emitc::VariableOp>(
      loc, p.getArrayType({n}, p.unrankedDescriptorType), p.getOpaqueAttr(""));

  for (auto [idx, v, originalType] :
       llvm::enumerate(shapedValues, originalTypes)) {
    auto memRefType = dyn_cast<MemRefType>(originalType);
    if (!memRefType || !memRefType.hasRank())
      return rewriter.notifyMatchFailure(failureOp, failureMessage);

    const int64_t rank = memRefType.getRank();
    Value rankVal = p.getI32Val(rewriter, loc, rank);

    Type ptrShapeTy = getPointerShapeDescriptorType(p.ctx, rank);
    Value ptrShapeVar = rewriter.create<emitc::VariableOp>(
        loc, p.getLValueType(ptrShapeTy), p.getOpaqueAttr(""));

    Value ptrShapeVal =
        getMemRefPtrShape(rewriter, loc, p.dataLayout, memRefType, v);
    rewriter.create<emitc::AssignOp>(loc, ptrShapeVar, ptrShapeVal);

    Value ptrShapeCopy =
        rewriter.create<emitc::LoadOp>(loc, ptrShapeTy, ptrShapeVar);

    Value ur = createCallOpaque(rewriter, loc, p.unrankedDescriptorType,
                                "mtrt::make_unranked_descriptor",
                                {rankVal, ptrShapeCopy})
                   .getResult(0);

    Value arrayElement = rewriter.create<emitc::SubscriptOp>(
        loc, p.getLValueType(p.unrankedDescriptorType), array,
        p.getI32Val(rewriter, loc, idx));
    rewriter.create<emitc::AssignOp>(loc, arrayElement, ur);
  }

  return array;
}

template <typename PatternT>
struct TRTOutputs {
  Value outputPtrs;
  SmallVector<Value> rankedVars;
};

template <typename PatternT>
static FailureOr<TRTOutputs<PatternT>> buildUnrankedMemRefMutOutputs(
    const PatternT &p, ConversionPatternRewriter &rewriter, Location loc,
    Operation *failureOp, TypeRange resultTypes, StringRef failureMessage) {
  // Intended C++ (schematic):
  //   mtrt::RankedMemRef<rank0> out0;
  //   ...
  //   mtrt::UnrankedMemRefMut outs[M];
  //   outs[i] = mtrt::make_unranked_descriptor_mut_ptr(rank_i, &out_i);
  //
  // `tensorrt_enqueue_alloc` will populate `out_i` with allocated pointers,
  // shapes, and strides during execution.
  emitc::OpaqueType unrankedMutTy =
      emitc::OpaqueType::get(p.ctx, "mtrt::UnrankedMemRefMut");
  const int64_t numOutputs = static_cast<int64_t>(resultTypes.size());

  Value outputPtrs{};
  if (numOutputs == 0) {
    outputPtrs =
        p.getNullptr(rewriter, loc, emitc::PointerType::get(unrankedMutTy));
  } else {
    outputPtrs = rewriter.create<emitc::VariableOp>(
        loc, p.getArrayType({numOutputs}, unrankedMutTy), p.getOpaqueAttr(""));
  }

  SmallVector<Value> rankedVars;
  rankedVars.reserve(numOutputs);

  for (auto [idx, resultTy] : llvm::enumerate(resultTypes)) {
    auto memRefType = dyn_cast<MemRefType>(resultTy);
    if (!memRefType || !memRefType.hasRank())
      return rewriter.notifyMatchFailure(failureOp, failureMessage);

    const int64_t rank = memRefType.getRank();
    Type rankedTy = getMemRefDescriptorType(p.ctx, rank);
    Value rankedVar = rewriter.create<emitc::VariableOp>(
        loc, p.getLValueType(rankedTy), p.getOpaqueAttr(""));
    rankedVars.push_back(rankedVar);

    Value rankVal64 = rewriter.create<emitc::ConstantOp>(
        loc, p.i64Type, rewriter.getI64IntegerAttr(rank));
    Value addr = p.getAddr(rewriter, loc, rankedVar);
    Value addrVoid = p.createCast(rewriter, p.voidPtrType, addr);
    Value unrankedMut =
        createCallOpaque(rewriter, loc, unrankedMutTy,
                         "mtrt::make_unranked_descriptor_mut_ptr",
                         {rankVal64, addrVoid})
            .getResult(0);

    if (numOutputs != 0) {
      Value arrayElement = rewriter.create<emitc::SubscriptOp>(
          loc, p.getLValueType(unrankedMutTy), outputPtrs,
          p.getI32Val(rewriter, loc, idx));
      rewriter.create<emitc::AssignOp>(loc, arrayElement, unrankedMut);
    }
  }

  return TRTOutputs<PatternT>{outputPtrs, std::move(rankedVars)};
}

struct TRTEnqueueConverter : EmitCConversionPattern<trtrt::EnqueueOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   mtrt::UnrankedMemRef inputs[N];
    //   mtrt::UnrankedMemRef outputs[M];
    //   int32_t st = mtrt::tensorrt_enqueue(ctx, stream, N, inputs, M,
    //   outputs); mtrt::abort_on_error(st);
    Location loc = op.getLoc();

    FailureOr<Value> inputPtrsOrFailure = buildUnrankedMemRefArray(
        *this, rewriter, loc, op, adaptor.getInputs(),
        TypeRange(op.getInputs()),
        "trtrt.enqueue lowering expects ranked memref operands");
    if (failed(inputPtrsOrFailure))
      return failure();
    Value inputPtrs = *inputPtrsOrFailure;

    FailureOr<Value> outputPtrsOrFailure = buildUnrankedMemRefArray(
        *this, rewriter, loc, op, adaptor.getOuts(), TypeRange(op.getOuts()),
        "trtrt.enqueue lowering expects ranked memref operands");
    if (failed(outputPtrsOrFailure))
      return failure();
    Value outputPtrs = *outputPtrsOrFailure;

    Value numInputs = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(adaptor.getInputs().size()));
    Value numOutputs = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(adaptor.getOuts().size()));

    Value st = builders.trtEnqueue.create(rewriter, loc,
                                          {adaptor.getExecutionContext(),
                                           adaptor.getStream(), numInputs,
                                           inputPtrs, numOutputs, outputPtrs});
    emitStatusCheckOrAbort(rewriter, loc, st);

    rewriter.eraseOp(op);
    return success();
  }
};

struct TRTEnqueueAllocConverter
    : EmitCConversionPattern<trtrt::EnqueueAllocOp> {
  using EmitCConversionPattern::EmitCConversionPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Intended C++ (schematic):
    //   mtrt::UnrankedMemRef inputs[N];
    //   mtrt::RankedMemRef<r0> out0;
    //   ...
    //   mtrt::UnrankedMemRefMut outputs[M];
    //   outputs[i] = mtrt::make_unranked_descriptor_mut_ptr(ri, &out_i);
    //   int32_t st = mtrt::tensorrt_enqueue_alloc(ctx, stream, N, inputs, M,
    //   outputs); mtrt::abort_on_error(st); return out0; // etc
    Location loc = op.getLoc();

    const int64_t numInputs = static_cast<int64_t>(adaptor.getInputs().size());
    FailureOr<Value> inputPtrsOrFailure = buildUnrankedMemRefArray(
        *this, rewriter, loc, op, adaptor.getInputs(),
        TypeRange(op.getInputs()),
        "trtrt.enqueue_alloc lowering expects ranked memref inputs");
    if (failed(inputPtrsOrFailure))
      return failure();
    Value inputPtrs = *inputPtrsOrFailure;

    FailureOr<TRTOutputs<TRTEnqueueAllocConverter>> outputsOrFailure =
        buildUnrankedMemRefMutOutputs(
            *this, rewriter, loc, op, TypeRange(op.getResultTypes()),
            "trtrt.enqueue_alloc lowering expects ranked memref results");
    if (failed(outputsOrFailure))
      return failure();

    const int64_t numOutputs =
        static_cast<int64_t>((*outputsOrFailure).rankedVars.size());
    Value outputPtrs = (*outputsOrFailure).outputPtrs;
    SmallVector<Value> rankedVars = std::move((*outputsOrFailure).rankedVars);

    Value numInputsVal = rewriter.create<emitc::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(numInputs));
    Value numOutputsVal = rewriter.create<emitc::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(numOutputs));

    Value st = builders.trtEnqueueAlloc.create(
        rewriter, loc,
        {adaptor.getExecutionContext(), adaptor.getStream(), numInputsVal,
         inputPtrs, numOutputsVal, outputPtrs});
    emitStatusCheckOrAbort(rewriter, loc, st);

    SmallVector<Value> results;
    results.reserve(numOutputs);
    for (auto [v, resultTy] :
         llvm::zip_equal(rankedVars, op.getResultTypes())) {
      Type rankedTy = cast<emitc::LValueType>(v.getType()).getValueType();
      results.push_back(rewriter.create<emitc::LoadOp>(loc, rankedTy, v));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

namespace mlir::host_to_emitc {
void populateHostToEmitCTensorRTPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         const DataLayout &dataLayout) {
  patterns.add<TRTEnqueueConverter, TRTEnqueueAllocConverter>(
      typeConverter, dataLayout, patterns.getContext());
}
} // namespace mlir::host_to_emitc
