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
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/DialectRegistry.h"

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
    : public bufferization::BufferizableOpInterface::ExternalModel<
          EnqueueOpInterface, EnqueueOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    // All operands are read
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const bufferization::AnalysisState &state) const {
    // The op doesn't write to its operands
    return false;
  }

  bufferization::AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const bufferization::AnalysisState &state) const {
    return {};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const bufferization::AnalysisState &state) const {
    // EnqueueOp creates new outputs, doesn't modify inputs in-place
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
    // the correct space.    SmallVector<Value> newInputBuffers;
    SmallVector<Value> newInputBuffers;
    newInputBuffers.reserve(enqueueOp->getNumOperands());
    for (auto [idx, opOperand] :
         llvm::enumerate(enqueueOp->getOperands())) {
      if (!isa<TensorType, MemRefType>(opOperand.getType())) {
        // This is a scalar or non-tensor/memref type
        newInputBuffers.push_back(opOperand);
        continue;
      }

      FailureOr<Value> buffer = getBuffer(rewriter, opOperand, options);
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
            rewriter, loc,
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

    // Handle the output
    SmallVector<Type> resultTypes;
    for (Type resultType : enqueueOp->getResultTypes()) {
      if (auto tensorType = resultType.dyn_cast<TensorType>()) {
        // Convert TensorType to MemrefType
        resultTypes.push_back(MemRefType::get(
            tensorType.getShape(), tensorType.getElementType(),
            MemRefLayoutAttrInterface(),
            plan::MemorySpaceAttr::get(ctx, plan::MemorySpace::device)));
      } else {
        resultTypes.push_back(resultType);
      }
    }

    // Create the bufferized EnqueueOp
    auto newOp = rewriter.create<EnqueueOp>(loc, resultTypes, newInputBuffers[0], newInputBuffers[1], ValueRange(newInputBuffers).drop_front(2), enqueueOp.getHostTensorArgsAttr());

    // Replace the old op with the new one
    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());

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
