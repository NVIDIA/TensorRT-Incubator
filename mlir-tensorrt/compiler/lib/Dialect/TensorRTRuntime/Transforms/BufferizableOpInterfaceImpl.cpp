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
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

static bool hasCanonicalStrides(MemRefType memRefType) {
  if (memRefType.getLayout().isIdentity())
    return true;

  if (memRefType.hasStaticShape()) {
    auto shape = memRefType.getShape();
    auto [strides, offset] = memRefType.getStridesAndOffset();
    if (llvm::equal(strides, mlir::computeSuffixProduct(shape)))
      return true;
  }
  return false;
}

namespace {

static FailureOr<Value> getBufferCopy(Operation *op, RewriterBase &rewriter,
                                      MLIRContext *ctx, Location loc,
                                      MemRefType memRefType, Value buffer,
                                      const BufferizationOptions &options,
                                      plan::MemorySpace memSpace) {
  FailureOr<Value> alloc = options.createAlloc(
      rewriter, op->getLoc(),
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      MemRefLayoutAttrInterface(),
                      plan::MemorySpaceAttr::get(ctx, memSpace)),
      ValueRange{});
  if (failed(alloc))
    return failure();
  if (failed(options.createMemCpy(rewriter, loc, buffer, *alloc)))
    return failure();
  return alloc;
}

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
    return false;
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
    for (OpOperand *opOperand : enqueueOp.getDpsInputOperands()) {

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

      // Check if this input is a host tensor. Insert a copy if required.
      if (enqueueOp.isOperandOnHost(opOperand) &&
          (!isInMemorySpace(memRefType, plan::MemorySpace::host_pinned) ||
           !hasCanonicalStrides(memRefType))) {
        FailureOr<Value> hostBuffer =
            getBufferCopy(enqueueOp, rewriter, ctx, loc, memRefType, *buffer,
                          options, plan::MemorySpace::host_pinned);
        if (failed(hostBuffer))
          return failure();
        newInputBuffers.push_back(*hostBuffer);
        continue;
      }

      // If we are in host space, then copy to the device.
      if (!enqueueOp.isOperandOnHost(opOperand) &&
          (!isInMemorySpace(memRefType, plan::MemorySpace::device) ||
           !hasCanonicalStrides(memRefType))) {
        FailureOr<Value> deviceBuffer =
            getBufferCopy(enqueueOp, rewriter, ctx, loc, memRefType, *buffer,
                          options, plan::MemorySpace::device);
        if (failed(deviceBuffer))
          return failure();
        newInputBuffers.push_back(*deviceBuffer);
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

struct EnqueueAllocOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          EnqueueAllocOpInterface, EnqueueAllocOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    auto enqueueAllocOp = cast<EnqueueAllocOp>(op);
    OperandRange inputs = enqueueAllocOp.getInputs();
    return std::find(inputs.begin(), inputs.end(), opOperand.get()) !=
           inputs.end();
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    return false; // This op doesn't write to its inputs
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const bufferization::AnalysisState &state) const {
    // EnqueueAllocOp creates new outputs, doesn't modify inputs in-place
    return false;
  }

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const {
    return {}; // This op doesn't alias its inputs to its outputs
  }

  bool bufferizesToElementwiseAccess(Operation *op,
                                     const bufferization::AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto enqueueAllocOp = cast<EnqueueAllocOp>(op);
    MLIRContext *ctx = op->getContext();
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(enqueueAllocOp);

    // Handle inputs
    SmallVector<Value> newInputBuffers;
    newInputBuffers.reserve(enqueueAllocOp.getInputs().size());
    for (OpOperand &opOperand : enqueueAllocOp->getOpOperands()) {
      if (!isa<RankedTensorType>(opOperand.get().getType()))
        continue;
      FailureOr<Value> buffer = getBuffer(rewriter, opOperand.get(), options);
      if (failed(buffer))
        return failure();

      MemRefType memRefType = cast<MemRefType>(buffer->getType());

      // Check if this input is a host tensor. Insert a copy if required.
      if (enqueueAllocOp.isOperandOnHost(&opOperand) &&
          !isInMemorySpace(memRefType, plan::MemorySpace::host_pinned)) {
        FailureOr<Value> hostBuffer =
            getBufferCopy(enqueueAllocOp, rewriter, ctx, loc, memRefType,
                          *buffer, options, plan::MemorySpace::host_pinned);
        if (failed(hostBuffer))
          return failure();
        newInputBuffers.push_back(*hostBuffer);
        continue;
      }

      // If we are in host space, then copy to the device.
      if (!enqueueAllocOp.isOperandOnHost(&opOperand) &&
          !isInMemorySpace(memRefType, plan::MemorySpace::device)) {
        FailureOr<Value> deviceBuffer =
            getBufferCopy(enqueueAllocOp, rewriter, ctx, loc, memRefType,
                          *buffer, options, plan::MemorySpace::device);
        if (failed(deviceBuffer))
          return failure();
        newInputBuffers.push_back(*deviceBuffer);
        continue;
      }

      newInputBuffers.push_back(*buffer);
    }

    // Handle results
    SmallVector<Type> outputBufferTypes;
    outputBufferTypes.reserve(enqueueAllocOp.getNumResults());
    for (unsigned i = 0; i < enqueueAllocOp.getNumResults(); ++i) {
      Type resultType = enqueueAllocOp->getResultTypes()[i];
      assert(isa<RankedTensorType>(resultType) &&
             "result must be a ranked tensor type");
      auto tensorType = dyn_cast<RankedTensorType>(resultType);
      auto memRefType = MemRefType::get(
          tensorType.getShape(), tensorType.getElementType(),
          MemRefLayoutAttrInterface(),
          plan::MemorySpaceAttr::get(ctx, plan::MemorySpace::device));
      outputBufferTypes.push_back(memRefType);
    }

    // Create the new operation
    auto bufferizedOp = rewriter.create<EnqueueAllocOp>(
        loc, TypeRange(outputBufferTypes), enqueueAllocOp.getExecutionContext(),
        enqueueAllocOp.getStream(), newInputBuffers,
        enqueueAllocOp.getHostTensorArgsAttr());
    replaceOpWithBufferizedValues(rewriter, op, bufferizedOp.getResults());
    return success();
  }
};

} // namespace

void trtrt::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, trtrt::TensorRTRuntimeDialect *dialect) {
        trtrt::EnqueueOp::attachInterface<EnqueueOpInterface>(*ctx);
        trtrt::EnqueueAllocOp::attachInterface<EnqueueAllocOpInterface>(*ctx);
      });
}
