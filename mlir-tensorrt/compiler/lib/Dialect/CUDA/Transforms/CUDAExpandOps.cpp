//===- CUDAExpandOps.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// Implementation of `cuda-expand-ops`.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::cuda {

#define GEN_PASS_DEF_CUDAEXPANDOPSPASS
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"

} // namespace mlir::cuda

using namespace mlir;
using namespace mlir::cuda;

/// Returns the device associated with the given stream.
static FailureOr<Value> getStreamDevice(Value stream) {
  if (auto createOp = stream.getDefiningOp<cuda::StreamCreateOp>())
    return createOp.getDevice();
  if (auto getGlobalStreamOp = stream.getDefiningOp<cuda::GetGlobalStreamOp>())
    return getGlobalStreamOp.getDevice();
  return failure();
}

namespace {

class CUDAExpandOpsPass
    : public cuda::impl::CUDAExpandOpsPassBase<CUDAExpandOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    IRRewriter rewriter(func.getContext());
    SmallVector<cuda::EventCreateOnStreamOp> ops;
    func.walk([&](cuda::EventCreateOnStreamOp op) { ops.push_back(op); });

    for (cuda::EventCreateOnStreamOp op : ops) {
      Value stream = op.getStream();
      FailureOr<Value> device = getStreamDevice(stream);
      if (failed(device)) {
        op.emitOpError(
            "failed to determine the device associated with the stream");
        return signalPassFailure();
      }

      rewriter.setInsertionPoint(op);
      Value event = cuda::EventCreateOp::create(rewriter, op.getLoc(), *device);
      cuda::StreamRecordEventOp::create(rewriter, op.getLoc(), stream, event);
      op.replaceAllUsesWith(event);
      rewriter.eraseOp(op);
    }
  }
};

} // namespace
