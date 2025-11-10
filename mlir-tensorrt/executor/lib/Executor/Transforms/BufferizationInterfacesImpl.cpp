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
/// Currently the operations that are bufferizable in the Executor dialect are:
/// - ABISendOp and ABIRecvOp (used at the start of the compilation pipeline
///   to establish the final function signature)
/// - CallPluginOp (for FFI plugin calls)
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/BufferizationOpInterfaceImpls.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SetVector.h"

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
    return true;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto recvOp = cast<executor::ABIRecvOp>(op);
    if (auto memrefType = dyn_cast<BaseMemRefType>(value.getType()))
      return memrefType;
    auto tensorType = cast<RankedTensorType>(recvOp.getResult().getType());
    auto memorySpace = recvOp.getMemorySpaceAttr();
    if (!memorySpace) {
      if (std::optional<Attribute> defaultSpace =
              options.defaultMemorySpaceFn(tensorType))
        memorySpace = *defaultSpace;
    }
    return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                memorySpace);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto recvOp = cast<executor::ABIRecvOp>(op);
    auto value = recvOp.getResult();
    auto func = recvOp->getParentOfType<FunctionOpInterface>();
    assert(func && "expected function");
    BlockArgument ptrArg = cast<BlockArgument>(recvOp.getPtr());

    SmallVector<Value> invocationStack;
    auto memrefType = this->getBufferType(op, value, options, invocationStack);
    if (failed(memrefType))
      return failure();

    auto replacementOp = rewriter.create<executor::ABIRecvOp>(
        recvOp.getLoc(), *memrefType, recvOp.getPtr());

    if (std::optional<unsigned> inputIdx =
            executor::abi::isInputArgument(func, ptrArg.getArgNumber())) {
      executor::abi::updateABIInputArgumentValueType(func, *inputIdx,
                                                     *memrefType);
    } else if (std::optional<unsigned> outputIdx =
                   executor::abi::isOutputArgument(func,
                                                   ptrArg.getArgNumber())) {
      executor::abi::updateABIOutputArgumentValueType(func, *outputIdx,
                                                      *memrefType);
    } else {
      llvm_unreachable("expected input or output argument");
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
    return true;
  }
  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {{op->getResult(0), bufferization::BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto sendOp = cast<executor::ABISendOp>(op);
    if (auto memrefType = dyn_cast<BaseMemRefType>(value.getType()))
      return memrefType;
    auto tensorType = cast<RankedTensorType>(sendOp.getValue().getType());
    Attribute memorySpace;
    if (std::optional<Attribute> defaultSpace =
            options.defaultMemorySpaceFn(tensorType))
      memorySpace = *defaultSpace;
    return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                memorySpace);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto sendOp = cast<executor::ABISendOp>(op);
    auto value = sendOp.getValue();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    FailureOr<Value> buffer =
        bufferization::getBuffer(rewriter, value, options);
    if (failed(buffer))
      return failure();

    // Get the expected memref type from the buffer
    SmallVector<Value> invocationStack;
    auto memrefType = this->getBufferType(op, value, options, invocationStack);
    if (failed(memrefType))
      return failure();

    // Update the executor.abi attribute on the function argument
    assert(funcOp && "expected function parent");
    auto blockArg = cast<BlockArgument>(sendOp.getPtr());
    unsigned outputIdx =
        executor::abi::getOutputArgumentIndex(funcOp, blockArg);
    executor::abi::updateABIOutputArgumentValueType(funcOp, outputIdx,
                                                    *memrefType);
    executor::ArgumentABIAttr abiAttr =
        executor::abi::getArgumentABIAttr(funcOp, blockArg);

    const bool isUndef = abiAttr.getUndef();
    if (!isUndef) {
      auto recvOp = executor::abi::getOrCreateABIRecv(rewriter, funcOp,
                                                      blockArg, *memrefType);
      if (failed(
              options.createMemCpy(rewriter, sendOp.getLoc(), *buffer, recvOp)))
        return failure();

      // Output buffers that are not 'undef' are not live-out since they are
      // views into storage that is owned by the caller. Do not return any
      // allocation. To retain the same type structure of the function, we
      // insert a poison op.
      auto poison =
          rewriter.create<mlir::ub::PoisonOp>(sendOp.getLoc(), *memrefType);
      bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                   {poison.getResult()});
      return success();
    } else if (*memrefType != buffer->getType()) {
      // Make sure the buffer has the required layout specified by the ABI.
      FailureOr<Value> castOrCopy = bufferization::castOrReallocMemRefValue(
          rewriter, *buffer, cast<MemRefType>(*memrefType), options);
      if (failed(castOrCopy))
        return failure();
      buffer = *castOrCopy;
    }

    // Output arguments that are 'undef' are live-out since they are
    // callee-allocated storage. Therefore, return the buffer. Later we will
    // lower the return into storing the descriptor to the output argument.
    bufferization::replaceOpWithBufferizedValues(rewriter, op, {*buffer});
    return success();
  }
};
} // namespace

/// Creates a copy of the given buffer with the specified memory space.
/// Allocates a new buffer and copies the contents from the source buffer.
static FailureOr<Value> getBufferCopy(Operation *op, RewriterBase &rewriter,
                                      MLIRContext *ctx, Location loc,
                                      MemRefType memRefType, Value buffer,
                                      const BufferizationOptions &options,
                                      Attribute memSpace) {
  FailureOr<Value> alloc = options.createAlloc(
      rewriter, op->getLoc(),
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      MemRefLayoutAttrInterface(), memSpace),
      ValueRange{});
  if (failed(alloc))
    return failure();
  if (failed(options.createMemCpy(rewriter, loc, buffer, *alloc)))
    return failure();
  return alloc;
}

/// Returns `true` if the memref has canonical (row-major) strides.
/// A memref has canonical strides if it has an identity layout or if its
/// strides match the suffix product of its shape (row-major ordering).
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
struct BufferizeCallPluginInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizeCallPluginInterface, executor::CallPluginOp> {

  /// The `executor.call_plugin` operation is always in destination style.
  /// Currently we require outputs to be statically shaped.
  bool bufferizesToAllocation(Operation *op, Value value) const {
    return false;
  }
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto callOp = cast<executor::CallPluginOp>(op);
    return llvm::is_contained(callOp.getOutputs(), opOperand.get());
  }
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto callOp = cast<executor::CallPluginOp>(op);
    return llvm::is_contained(callOp.getArgs(), opOperand.get());
  }

  /// Returning `true` here tells the bufferization analysis that tensor both
  /// read and written by this operation can bufferize to the same buffer.
  /// Since we explicitly repeat operands in the `args` and `outputs` list when
  /// I/O aliasing is involved, return true for that case.
  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    return true;
  }

  /// Returns `false` to allow bufferization to create copies when needed.
  /// I/O aliasing is handled explicitly through the operands list.
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &operand,
                    const bufferization::AnalysisState &state) const {
    auto callOp = cast<executor::CallPluginOp>(op);
    auto it = llvm::find(callOp.getOutputs(), operand.get());
    if (it == callOp.getOutputs().end())
      return {};
    auto outIdx = std::distance(callOp.getOutputs().begin(), it);
    return {{callOp->getResult(outIdx),
             bufferization::BufferRelation::Equivalent, /*isDefinite=*/true}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return failure();
    std::optional<Attribute> memorySpace =
        options.defaultMemorySpaceFn(tensorType);
    if (!memorySpace)
      return failure();
    return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                *memorySpace);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto callOp = cast<executor::CallPluginOp>(op);
    MLIRContext *ctx = op->getContext();

    // There may be duplicate arguments in inputs/outputs. Since we may insert
    // alloc+copy to enforce layout or memory space requirements, make sure we
    // only do that once per value, otherwise the result is not correct.
    llvm::SmallDenseMap<Value, Value, 8> argsTobufferize;
    for (Value v : op->getOperands()) {
      if (!isa<TensorType>(v.getType()))
        continue;
      if (argsTobufferize.contains(v))
        continue;
      FailureOr<Value> buffer = bufferization::getBuffer(rewriter, v, options);
      if (failed(buffer))
        return failure();

      // Check type.
      if (auto bufferType = cast<MemRefType>(buffer->getType());
          !hasCanonicalStrides(bufferType)) {
        FailureOr<Value> bufferCopy =
            getBufferCopy(callOp, rewriter, ctx, callOp.getLoc(),
                          cast<MemRefType>(buffer->getType()), *buffer, options,
                          bufferType.getMemorySpace());
        if (failed(bufferCopy))
          return failure();
        buffer = bufferCopy;
      }
      argsTobufferize.insert(std::make_pair(v, *buffer));
    }

    SmallVector<Value> bufferizedArgs;
    for (Value arg : callOp.getArgs()) {
      bufferizedArgs.push_back(argsTobufferize.lookup(arg));
      assert(bufferizedArgs.back() && "expected bufferized argument");
    }

    for (Value arg : callOp.getOutputs()) {
      bufferizedArgs.push_back(argsTobufferize.lookup(arg));
      assert(bufferizedArgs.back() && "expected bufferized output");
    }

    rewriter.create<executor::CallPluginOp>(
        callOp.getLoc(), callOp.getCallee(), callOp.getStream(),
        ArrayRef(bufferizedArgs).take_front(callOp.getArgs().size()),
        ArrayRef(bufferizedArgs).drop_front(callOp.getArgs().size()),
        callOp.getImmediateArgsAttr(), callOp.getArgSpecAttr(),
        callOp.getIoAliasingAttr(), callOp.getArgAttrsAttr(),
        callOp.getResAttrsAttr());

    replaceOpWithBufferizedValues(
        rewriter, op,
        ArrayRef(bufferizedArgs).drop_front(callOp.getArgs().size()));

    return success();
  }
};

} // namespace

namespace mlir::executor {

LogicalResult
bufferizeABIWrapperFunctionType(FunctionOpInterface abiFuncOp,
                                const BufferizationOptions &options) {
  FailureOr<FunctionType> abiFuncType =
      executor::abi::getABIFunctionType(abiFuncOp);
  if (failed(abiFuncType))
    return failure();

  for (auto [i, argType] : enumerate(abiFuncType->getInputs())) {
    if (auto tensorType = dyn_cast<RankedTensorType>(argType)) {
      auto newType = bufferization::getMemRefTypeWithStaticIdentityLayout(
          tensorType, *options.defaultMemorySpaceFn(tensorType));
      abi::updateABIInputArgumentValueType(abiFuncOp, i, newType);
    }
  }
  for (auto [i, resultType] : enumerate(abiFuncType->getResults())) {
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
      auto newType = bufferization::getMemRefTypeWithStaticIdentityLayout(
          tensorType, *options.defaultMemorySpaceFn(tensorType));
      executor::abi::updateABIOutputArgumentValueType(abiFuncOp, i, newType);
    }
  }
  return success();
}

void registerBufferizationOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            executor::ExecutorDialect *dialect) {
    executor::ABISendOp::attachInterface<BufferDeallocationABISend>(*ctx);
    executor::ABIRecvOp::attachInterface<BufferizableOpInterfaceABIRecv>(*ctx);
    executor::ABISendOp::attachInterface<BufferizableOpInterfaceABISend>(*ctx);
    executor::CallPluginOp::attachInterface<BufferizeCallPluginInterface>(*ctx);
  });
}
} // namespace mlir::executor
