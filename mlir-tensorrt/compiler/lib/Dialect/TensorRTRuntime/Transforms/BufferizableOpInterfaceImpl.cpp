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
#include "mlir-tensorrt/Dialect/TensorRTRuntime/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::trtrt;

using bufferization::BufferizationOptions;
using bufferization::replaceOpWithBufferizedValues;

/// Return true if the type is in the specified space. By default, we assume
/// memrefs are device if no space annotation is attached.
static bool isInMemorySpace(Type memrefType, plan::MemorySpace memType) {
  assert(isa<MemRefType>(memrefType) && "expected MemRefType");
  auto type = cast<MemRefType>(memrefType);
  Attribute space = type.getMemorySpace();
  if (!space)
    return memType == plan::MemorySpace::device;
  return space == plan::MemorySpaceAttr::get(type.getContext(), memType);
}

namespace {
struct EnqueueOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          EnqueueOpInterface, EnqueueOp> {
  /// Only our dps input operands are read. Dps init are guaranteed to be just
  /// outputs in our use-case.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    EnqueueOp enqueueOp = cast<EnqueueOp>(op);
    return enqueueOp.isDpsInput(&opOperand);
  }

  /// Only dps inits are written.
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    EnqueueOp enqueueOp = cast<EnqueueOp>(op);
    return enqueueOp.isDpsInit(&opOperand);
  }

  // TensorRT will guarantee that the input will be read before the result
  // buffer is written.
  bool bufferizesToElementwiseAccess(Operation *op,
                                     const bufferization::AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    return true;
  }

  /// Bufferize the `trtrt.enqueue` operation.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    EnqueueOp enqueueOp = cast<EnqueueOp>(op);
    MLIRContext *ctx = op->getContext();
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(enqueueOp);

    // For the inputs, check the memory space and insert a copy if it is not in
    // the correct space.
    SmallVector<Value> newInputBuffers;
    newInputBuffers.reserve(enqueueOp.getNumDpsInputs());
    for (auto [idx, opOperand] :
         llvm::enumerate(enqueueOp.getDpsInputOperands())) {

      // The context and steam operands are considered "DPS inputs" and
      // therefore they'll be skipped here.
      if (enqueueOp.isScalar(opOperand)) {
        newInputBuffers.push_back(opOperand->get());
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
      if (failed(buffer))
        return failure();

      MemRefType memRefType = cast<MemRefType>(buffer->getType());

      // Check if this input is a host tensor. Insert a copy if required. Note
      // that we subtract two from the index to account for context/stream
      // arguments.
      if (enqueueOp.isOperandOnHost(idx) &&
          !isInMemorySpace(memRefType, plan::MemorySpace::host_pinned)) {
        FailureOr<Value> pinnedAlloc = options.createAlloc(
            rewriter, op->getLoc(),
            MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                            memRefType.getLayout(),
                            plan::MemorySpaceAttr::get(
                                ctx, plan::MemorySpace::host_pinned)),
            ValueRange{});
        if (failed(pinnedAlloc))
          return failure();
        if (failed(options.createMemCpy(rewriter, loc, *buffer, *pinnedAlloc)))
          return failure();
        newInputBuffers.push_back(*pinnedAlloc);
        continue;
      }

      // If we are in host space, then copy to the device.
      if (!enqueueOp.isOperandOnHost(idx) &&
          !isInMemorySpace(memRefType, plan::MemorySpace::device)) {
        FailureOr<Value> devAlloc = options.createAlloc(
            rewriter, op->getLoc(),
            MemRefType::get(
                memRefType.getShape(), memRefType.getElementType(),
                memRefType.getLayout(),
                plan::MemorySpaceAttr::get(ctx, plan::MemorySpace::device)),
            ValueRange{});
        if (failed(devAlloc))
          return failure();
        if (failed(options.createMemCpy(rewriter, loc, *buffer, *devAlloc)))
          return failure();
        newInputBuffers.push_back(*devAlloc);
        continue;
      }

      // We are in device space, nothing to do.
      newInputBuffers.push_back(*buffer);
    }

    SmallVector<Value> newOutputBuffers;
    newOutputBuffers.reserve(enqueueOp.getNumDpsInits());
    for (OpResult opResult : op->getOpResults()) {
      OpOperand *opOperand =
          enqueueOp.getDpsInitOperand(opResult.getResultNumber());
      FailureOr<Value> resultBuffer =
          getBuffer(rewriter, opOperand->get(), options);
      if (failed(resultBuffer))
        return failure();
      newOutputBuffers.push_back(*resultBuffer);
    }

    rewriter.create<EnqueueOp>(
        op->getLoc(), newInputBuffers[0], newInputBuffers[1],
        ValueRange(newInputBuffers).drop_front(2), newOutputBuffers,
        enqueueOp.getHostTensorArgsAttr());
    replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);
    return success();
  }
};

} // namespace

void trtrt::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, trtrt::TensorRTRuntimeDialect *dialect) {
        trtrt::EnqueueOp::attachInterface<EnqueueOpInterface>(*ctx);
      });
}
