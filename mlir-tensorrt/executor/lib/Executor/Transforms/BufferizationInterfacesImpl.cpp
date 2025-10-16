//===- BufferizationInterfacesImpl.cpp -----------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of bufferization-related operation interfaces for Executor
/// dialect operations.
///
/// Currently the only operations that are bufferizable in the Executor
/// dialect are ABISendOp and ABIRecvOp since these are used at the
/// very start of the compilation pipeline to establish the final function
/// signature.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/BufferizationOpInterfaceImpls.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct BufferDeallocationABISend
    : public BufferDeallocationOpInterface::ExternalModel<
          BufferDeallocationABISend, executor::ABISendOp> {

  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    auto abiSendOp = cast<executor::ABISendOp>(op);
    auto funcOp = cast<FunctionOpInterface>(abiSendOp->getParentOp());
    auto ptr = cast<BlockArgument>(abiSendOp.getPtr());
    Value value = abiSendOp.getValue();
    if (!isa<MemRefType>(value.getType()))
      return op;

    // Get the ABI attribute on the argument.
    executor::ArgumentABIAttr abiAttr =
        executor::abi::getArgumentABIAttr(funcOp, ptr);
    assert(abiAttr && "expected ABI attribute");
    const bool isUndef = abiAttr.getUndef();
    IRRewriter rewriter(op);

    // We need a unique ownership indicator for the value. This will insert a
    // clone if the ownership is not yet unique, otherwise it does nothing.
    auto [newMemRef, ownership] = state.getMemrefWithUniqueOwnership(
        rewriter, value, abiSendOp->getBlock());
    abiSendOp.getValueMutable().assign(newMemRef);
    abiSendOp.getOwnershipMutable().assign(ownership);

    // If the descriptor is marked 'undef', then we will pass ownership to
    // the caller. Therefore, don't deallocate the value. If the descriptor is
    // not marked 'undef', then we will still allocate any owned memref values.
    if (isUndef) {
      state.dropMemrefToDeallocate(newMemRef, abiSendOp->getBlock());
    }
    return op;
  }
};

struct BufferizableOpInterfaceABIRecv
    : public BufferizableOpInterface::ExternalModel<
          BufferizableOpInterfaceABIRecv, executor::ABIRecvOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const {
    // ABIRecvOp bufferizes to allocation if the argument is an output argument
    // and the descriptor is marked 'undef'.
    auto recvOp = cast<executor::ABIRecvOp>(op);
    auto func = cast<FunctionOpInterface>(recvOp->getParentOp());
    std::optional<unsigned> outputIdx = mlir::executor::abi::isOutputArgument(
        func, cast<BlockArgument>(recvOp.getPtr()));
    if (!outputIdx.has_value())
      return false;
    executor::ArgumentABIAttr abiAttr = mlir::executor::abi::getArgumentABIAttr(
        func, cast<BlockArgument>(recvOp.getPtr()));
    assert(abiAttr && "expected ABI attribute");
    return abiAttr.getUndef();
  }

  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    auto recvOp = cast<executor::ABIRecvOp>(op);
    assert(recvOp.getResult() == opResult && "expected result");
    BlockArgument arg = cast<BlockArgument>(recvOp.getPtr());
    auto func = recvOp->getParentOfType<FunctionOpInterface>();
    assert(func && "expected function");

    // Output operands are just storage unless they alias an input.
    // Retrieving the buffer with ABIRecvOp does not correspond to a def-event.
    // Its contents are undefined.
    if (executor::abi::isOutputArgument(func, arg))
      return false;

    // Input buffers have defined contents, so this corresponds to a def-event.
    return true;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto recvOp = cast<executor::ABIRecvOp>(op);
    BlockArgument arg = cast<BlockArgument>(recvOp.getPtr());
    auto func = recvOp->getParentOfType<FunctionOpInterface>();
    assert(func && "expected function");

    // Output buffers are always writable, even if the descriptor is
    // marked 'undef'. The reason is that if marked 'undef', then this op is
    // lowered to allocation.
    if (executor::abi::isOutputArgument(func, arg))
      return true;

    // TODO: Input operands are writable only if they alias a result.
    return false;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto recvOp = cast<executor::ABIRecvOp>(op);
    auto value = recvOp.getResult();

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return success();
    auto memorySpace = recvOp.getMemorySpaceAttr();
    if (!memorySpace) {
      if (std::optional<Attribute> defaultSpace =
              options.defaultMemorySpaceFn(tensorType))
        memorySpace = *defaultSpace;
    }

    BaseMemRefType memrefType =
        bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                             memorySpace);

    auto replacementOp = rewriter.create<executor::ABIRecvOp>(
        recvOp.getLoc(), memrefType, recvOp.getPtr());

    // Update the executor.abi attribute on the function argument
    if (auto blockArg = dyn_cast<BlockArgument>(recvOp.getPtr())) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(
              blockArg.getOwner()->getParentOp())) {
        unsigned argIdx = blockArg.getArgNumber();
        if (auto abiAttr = funcOp.getArgAttrOfType<executor::ArgumentABIAttr>(
                argIdx, executor::ExecutorDialect::kArgABIAttrName)) {
          // Create a new ABI attribute with the memref type
          auto newAbiAttr = abiAttr.cloneWithValueType(memrefType);
          funcOp.setArgAttr(argIdx, executor::ExecutorDialect::kArgABIAttrName,
                            newAbiAttr);
        }
      }
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 replacementOp.getResult());

    return success();
  }
};

struct BufferizableOpInterfaceABISend
    : public BufferizableOpInterface::ExternalModel<
          BufferizableOpInterfaceABISend, executor::ABISendOp> {
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }
  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto sendOp = cast<executor::ABISendOp>(op);
    auto value = sendOp.getValue();

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return success();

    FailureOr<Value> buffer =
        bufferization::getBuffer(rewriter, value, options);
    if (failed(buffer))
      return failure();

    // Get the memref type from the buffer
    auto memrefType = cast<BaseMemRefType>(buffer->getType());

    // Update the executor.abi attribute on the function argument
    if (auto blockArg = dyn_cast<BlockArgument>(sendOp.getPtr())) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(
              blockArg.getOwner()->getParentOp())) {
        unsigned argIdx = blockArg.getArgNumber();
        if (auto abiAttr = funcOp.getArgAttrOfType<executor::ArgumentABIAttr>(
                argIdx, executor::ExecutorDialect::kArgABIAttrName)) {
          // Create a new ABI attribute with the memref type
          auto newAbiAttr = abiAttr.cloneWithValueType(memrefType);
          funcOp.setArgAttr(argIdx, executor::ExecutorDialect::kArgABIAttrName,
                            newAbiAttr);
        }
      }
    }

    rewriter.modifyOpInPlace(
        sendOp, [&]() { sendOp.getValueMutable().assign(*buffer); });

    return success();
  }
};

} // namespace

namespace mlir::executor {
void registerBufferizationOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            executor::ExecutorDialect *dialect) {
    executor::ABISendOp::attachInterface<BufferDeallocationABISend>(*ctx);
    executor::ABIRecvOp::attachInterface<BufferizableOpInterfaceABIRecv>(*ctx);
    executor::ABISendOp::attachInterface<BufferizableOpInterfaceABISend>(*ctx);
  });
}
} // namespace mlir::executor
