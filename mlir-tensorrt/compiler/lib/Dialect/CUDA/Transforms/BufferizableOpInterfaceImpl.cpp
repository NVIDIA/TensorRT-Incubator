//===- BufferizableOpInterfaceImpl.cpp ------------------------------------===//
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
#include "mlir-tensorrt/Dialect/CUDA/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::cuda;

using bufferization::AnalysisState;
using bufferization::BufferizationOptions;
using bufferization::replaceOpWithBufferizedValues;

namespace {

/// Get size as `SmallVector<int64_t>`. Returns failure if any dimension is
/// dynamic.
static FailureOr<SmallVector<int64_t>> getMemrefShape(MemRefType t) {
  if (t == nullptr || t.getRank() == 0 || !t.hasStaticShape())
    return failure();
  return llvm::to_vector(t.getShape());
}

struct BlasRunGemmOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          BlasRunGemmOpInterface, BlasRunGemmOp> {
  /// All operands can be read, including destination.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  /// Only dps inits are written.
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    BlasRunGemmOp callOp = cast<BlasRunGemmOp>(op);
    return callOp.isDpsInit(&opOperand);
  }

  /// Bufferize the `cuda.blas.run_gemm` operation.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    BlasRunGemmOp callOp = cast<BlasRunGemmOp>(op);
    rewriter.setInsertionPoint(callOp);
    bool hasAlpha = false;
    FailureOr<Value> bufferAlpha;
    if (callOp.getAlpha()) {
      hasAlpha = true;
      bufferAlpha = getBuffer(rewriter, callOp.getAlpha(), options);
      if (failed(bufferAlpha))
        return failure();
    }
    FailureOr<Value> bufferA = getBuffer(rewriter, callOp.getMatA(), options);
    if (failed(bufferA))
      return failure();
    FailureOr<Value> bufferB = getBuffer(rewriter, callOp.getMatB(), options);
    if (failed(bufferB))
      return failure();
    bool hasBeta = false;
    FailureOr<Value> bufferBeta;
    if (callOp.getBeta()) {
      hasBeta = true;
      bufferBeta = getBuffer(rewriter, callOp.getBeta(), options);
      if (failed(bufferBeta))
        return failure();
    }

    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, callOp.getMatC(), options);
    if (failed(resultBuffer))
      return failure();
    Value algo;
    // Check if `algo` is passed, otherwise, create algorithm selection op.
    if (!callOp.getAlgo()) {
      bool transposeA = callOp->hasAttr("transpose_a") ? true : false;
      bool transposeB = callOp->hasAttr("transpose_b") ? true : false;

      MemRefType bufferAType = (*bufferA).getType().dyn_cast<MemRefType>();
      Type dataType = bufferAType.getElementType();
      FailureOr<SmallVector<int64_t>> sizeA = getMemrefShape(bufferAType);
      if (failed(sizeA))
        return failure();
      SmallVector<long, 6> strideA =
          mlir::getStridesAndOffset(bufferAType).first;
      MemRefType bufferBType = (*bufferB).getType().dyn_cast<MemRefType>();
      FailureOr<SmallVector<int64_t>> sizeB = getMemrefShape(bufferBType);
      if (failed(sizeB))
        return failure();
      SmallVector<long, 6> strideB =
          mlir::getStridesAndOffset(bufferBType).first;
      MemRefType resultBufferType =
          (*resultBuffer).getType().dyn_cast<MemRefType>();
      FailureOr<SmallVector<int64_t>> sizeC = getMemrefShape(resultBufferType);
      if (failed(sizeC))
        return failure();
      SmallVector<long, 6> strideC =
          mlir::getStridesAndOffset(resultBufferType).first;
      SmallVector<int64_t> tileSizes{0, 0};
      algo = rewriter.create<BlasHeuristicAlgoSelectionOp>(
          op->getLoc(), callOp.getHandle(), dataType, *sizeA, strideA,
          transposeA, *sizeB, strideB, transposeB, *sizeC, strideC, tileSizes);
    } else {
      algo = callOp.getAlgo();
    }

    // Both `alpha` and `beta` are provided (GEMM is implemented)
    if (hasAlpha && hasBeta)
      rewriter.create<BlasRunGemmOp>(
          op->getLoc(), callOp.getTransposeAAttr(), callOp.getTransposeBAttr(),
          callOp.getHandle(), callOp.getStream(), algo, *bufferAlpha, *bufferA,
          *bufferB, *bufferBeta, *resultBuffer);
    // No `alpha` and `beta` is provided (C += A@B is implemented)
    else
      rewriter.create<BlasRunGemmOp>(
          op->getLoc(), callOp.getTransposeAAttr(), callOp.getTransposeBAttr(),
          callOp.getHandle(), callOp.getStream(), algo, Value(), *bufferA,
          *bufferB, Value(), *resultBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *resultBuffer);
    return success();
  }
};

} // namespace

void cuda::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cuda::CUDADialect *dialect) {
    cuda::BlasRunGemmOp::attachInterface<BlasRunGemmOpInterface>(*ctx);
  });
}