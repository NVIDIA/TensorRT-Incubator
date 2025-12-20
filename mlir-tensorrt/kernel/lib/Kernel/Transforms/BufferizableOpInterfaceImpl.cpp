//===- BufferizableOpInterfaceImpl.cpp ------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-kernel/Kernel/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "kernel-bufferize"
#define DBGV(fmt, ...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[kernel-bufferize] ";                            \
             llvm::dbgs() << llvm::formatv(fmt "\n", __VA_ARGS__));

using namespace mlir;
using namespace mlir::kernel;

using bufferization::AliasingValueList;
using bufferization::AnalysisState;
using bufferization::BufferizationOptions;
using bufferization::BufferRelation;
using bufferization::OneShotAnalysisState;
using bufferization::replaceOpWithBufferizedValues;
using namespace bufferization::func_ext;

/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const AnalysisState &state) {
  assert(isa<OneShotAnalysisState>(state) && "expected OneShotAnalysisState");
  auto *result = static_cast<const OneShotAnalysisState &>(state)
                     .getExtension<FuncAnalysisState>();
  assert(result && "FuncAnalysisState does not exist");
  return *result;
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

static MemRefType getMemRefTypeWithFullyDynamicLayout(MemRefType type) {
  int64_t dynamicOffset = ShapedType::kDynamic;
  SmallVector<int64_t> dynamicStrides(type.getRank(), ShapedType::kDynamic);
  auto stridedLayout =
      StridedLayoutAttr::get(type.getContext(), dynamicOffset, dynamicStrides);
  return MemRefType::get(type.getShape(), type.getElementType(), stridedLayout,
                         type.getMemorySpace());
}

static bool areTypesCongrent(MemRefType a, RankedTensorType b) {
  return a.getRank() == b.getRank() && a.getElementType() == b.getElementType();
}

static bool areTypesCongrent(Type funcArg, Type callArg) {
  auto aMemRef = dyn_cast<MemRefType>(funcArg);
  auto bMemRef = dyn_cast<RankedTensorType>(callArg);
  if (aMemRef && bMemRef)
    return areTypesCongrent(aMemRef, bMemRef);
  return funcArg == callArg;
}

static bool isFuncOpCongruentWithCallOp(FuncOp funcOp, ValueRange callArgs) {
  if (funcOp.getNumArguments() != callArgs.size())
    return false;
  for (auto [funcArgType, callArgType] :
       llvm::zip_equal(funcOp.getArgumentTypes(), TypeRange(callArgs))) {
    if (isa<TensorType>(funcArgType))
      return false;
    if (!areTypesCongrent(funcArgType, callArgType))
      return false;
  }
  return true;
}

namespace {
struct KernelCallOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          KernelCallOpInterface, CallOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    CallOp callOp = cast<CallOp>(op);

    if (!callOp.isForwardedOperand(&opOperand))
      return false;
    func::FuncOp callee = getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, callee) != FuncOpAnalysisState::Analyzed)
      return llvm::is_contained(callOp.getInputs(), opOperand.get()) ||
             llvm::is_contained(callOp.getOuts(), opOperand.get());

    // Adjust the operand index taking into account that `kernel.call` has more
    // leading arguments than what is passed to the callee.
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    bool result = funcState.readBbArgs.lookup(callee).contains(
        callOp.getForwardedArgumentIndex(&opOperand));
    DBGV("arg {0} of {1} read = {2}",
         callOp.getForwardedArgumentIndex(&opOperand), callee.getName(),
         result);
    return result;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    CallOp callOp = cast<CallOp>(op);
    if (!callOp.isForwardedOperand(&opOperand))
      return false;

    func::FuncOp callee = getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, callee) != FuncOpAnalysisState::Analyzed)
      return llvm::is_contained(callOp.getOuts(), opOperand.get());

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    bool result = funcState.writtenBbArgs.lookup(callee).contains(
        callOp.getForwardedArgumentIndex(&opOperand));
    DBGV("arg {0} of {1} written = {2}",
         callOp.getForwardedArgumentIndex(&opOperand), callee.getName(),
         result);
    return result;
  }

  static Value getFullyDynamicBuffer(RewriterBase &rewriter, Value buffer,
                                     const BufferizationOptions &options) {
    MemRefType memRefType = cast<MemRefType>(buffer.getType());
    MemRefType dynMemRefType = getMemRefTypeWithFullyDynamicLayout(memRefType);
    if (memRefType == dynMemRefType)
      return buffer;

    Value castedBuffer =
        rewriter.create<memref::CastOp>(buffer.getLoc(), dynMemRefType, buffer);
    return castedBuffer;
  }

  LogicalResult getBufferizedCallOperand(
      RewriterBase &rewriter, Value arg, Type funcArgType,
      const BufferizationOptions &options, SmallVectorImpl<Value> &result,
      bufferization::BufferizationState &state,
      SmallVectorImpl<Value> *replacements = nullptr) const {
    auto funcMemRefType = dyn_cast<MemRefType>(funcArgType);
    if (!funcMemRefType) {
      result.push_back(arg);
      return success();
    }

    FailureOr<Value> buffer = getBuffer(rewriter, arg, options, state);
    if (failed(buffer))
      return failure();

    auto targetType = MemRefType::get(
        funcMemRefType.getShape(), funcMemRefType.getElementType(),
        funcMemRefType.getLayout(),
        cast<MemRefType>(buffer->getType()).getMemorySpace());
    if (targetType == buffer->getType()) {
      result.push_back(*buffer);
      if (replacements)
        replacements->push_back(*buffer);
      return success();
    }

    FailureOr<Value> replacement = bufferization::castOrReallocMemRefValue(
        rewriter, *buffer, targetType, options);
    if (failed(replacement))
      return failure();
    result.push_back(*replacement);
    bool wasCopied = *buffer != *replacement &&
                     !replacement->getDefiningOp<memref::CastOp>();
    if (replacements)
      replacements->push_back(wasCopied ? *replacement : *buffer);
    return success();
  }

  LogicalResult getBufferizedCallOperand(
      RewriterBase &rewriter, Value arg,
      kernel::KernelArgLayoutMapOption defaultLayout,
      const BufferizationOptions &options, SmallVectorImpl<Value> &result,
      bufferization::BufferizationState &state,
      SmallVectorImpl<Value> *replacements = nullptr) const {
    FailureOr<Value> buffer = getBuffer(rewriter, arg, options, state);
    if (failed(buffer))
      return failure();
    auto memRefType = cast<MemRefType>(buffer->getType());
    MemRefLayoutAttrInterface layout =
        defaultLayout == KernelArgLayoutMapOption::Identity
            ? MemRefLayoutAttrInterface{}
            : StridedLayoutAttr::get(
                  memRefType.getContext(), ShapedType::kDynamic,
                  SmallVector<int64_t>(memRefType.getRank(),
                                       ShapedType::kDynamic));

    auto targetType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        layout, memRefType.getMemorySpace());
    if (targetType == buffer->getType()) {
      result.push_back(*buffer);
      if (replacements)
        replacements->push_back(*buffer);
      return success();
    }

    FailureOr<Value> casted = bufferization::castOrReallocMemRefValue(
        rewriter, *buffer, targetType, options);
    if (failed(casted))
      return failure();
    bool wasCopied =
        *buffer != *casted && !casted->getDefiningOp<memref::CastOp>();
    if (replacements)
      replacements->push_back(wasCopied ? *casted : *buffer);
    result.push_back(*casted);
    return success();
  }

  // Some callee's may have bufferized early depending on what kind of GPU
  // module it is embedded in. If there is no FunOp or if the format of the
  // function is not congruent with the callOp, then use the module kind
  // attribute's default layout map information to derive the target memref
  // kind.
  LogicalResult getBufferizedCallForwardedOperands(
      RewriterBase &rewriter, CallOp callOp, ValueRange args,
      func::FuncOp callee, const BufferizationOptions &options,
      SmallVectorImpl<Value> &inputs, SmallVectorImpl<Value> &outs,
      SmallVectorImpl<Value> &replacements,
      bufferization::BufferizationState &state) const {

    // Case 1: callee is available and bufferized. Get MemRef target types from
    // the callee.
    if (isFuncOpCongruentWithCallOp(callee, args)) {
      unsigned numOuts = callOp.getNumDpsInits();
      for (auto [funcArgType, arg] :
           llvm::zip(callee.getArgumentTypes().drop_back(numOuts),
                     args.drop_back(numOuts))) {
        if (failed(getBufferizedCallOperand(rewriter, arg, funcArgType, options,
                                            inputs, state)))
          return failure();
      }
      for (auto [funcArgType, arg] :
           llvm::zip(callee.getArgumentTypes().take_back(numOuts),
                     args.take_back(numOuts))) {
        if (failed(getBufferizedCallOperand(rewriter, arg, funcArgType, options,
                                            outs, state, &replacements)))
          return failure();
      }
      return success();
    }

    // Case 2: callee is not available or not in different format.
    auto gpuModule = callee->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
      return failure();

    auto moduleLayout =
        gpuModule->getAttrOfType<kernel::GPUModuleLoweringAttrInterface>(
            kernel::KernelDialect::getGpuModuleKindAttrName());
    if (!moduleLayout)
      return failure();

    KernelArgLayoutMapOption targetLayout =
        moduleLayout.getMemRefArgumentDefaultLayoutMap(gpuModule);
    unsigned numOuts = callOp.getNumDpsInits();
    for (Value arg : args.drop_back(numOuts)) {
      if (failed(getBufferizedCallOperand(rewriter, arg, targetLayout, options,
                                          inputs, state)))
        return failure();
    }
    for (Value arg : args.take_back(numOuts)) {
      if (failed(getBufferizedCallOperand(rewriter, arg, targetLayout, options,
                                          outs, state, &replacements)))
        return failure();
    }
    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    CallOp callOp = cast<CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);

    rewriter.setInsertionPoint(callOp);

    SmallVector<Value> inputs, outs, replacements;
    if (failed(getBufferizedCallForwardedOperands(
            rewriter, callOp,
            callOp.getOperands().drop_front(
                callOp.getNumNonForwardedArguments()),
            funcOp, options, inputs, outs, replacements, state)))
      return failure();

    rewriter.create<CallOp>(op->getLoc(), TypeRange{}, callOp.getGridSize(),
                            callOp.getBlockSize(), inputs, outs,
                            callOp.getKernelSymAttr(), callOp.getArgAttrsAttr(),
                            callOp.getResAttrsAttr());
    replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

struct KernelExtCallOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          KernelExtCallOpInterface, kernel::ExtCallOp> {

  // Any I/O aliasing here is intentional.
  bool bufferizesToElementwiseAccess(Operation *op,
                                     const bufferization::AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    return true;
  }

  bool bufferizesToAllocation(Operation *op, Value value) const {
    // kernel.ext_call never allocates - it works with existing buffers
    // Outs alias inputs, so no new allocations needed
    return false;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    kernel::ExtCallOp callOp = cast<kernel::ExtCallOp>(op);
    unsigned argIdx = callOp.getForwardedArgumentIndex(&opOperand);
    return callOp.argHasReadEffect(argIdx);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    kernel::ExtCallOp callOp = cast<kernel::ExtCallOp>(op);
    unsigned argIdx = callOp.getForwardedArgumentIndex(&opOperand);
    return callOp.argHasWriteEffect(argIdx);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    kernel::ExtCallOp callOp = cast<kernel::ExtCallOp>(op);
    AliasingValueList result;

    unsigned argIdx = callOp.getForwardedArgumentIndex(&opOperand);

    // Check if this argument aliases any result
    auto aliasingConfig = callOp.getResultAliases();
    auto it = llvm::find(aliasingConfig, argIdx);
    if (it == aliasingConfig.end())
      return result;
    unsigned resultIdx = std::distance(aliasingConfig.begin(), it);
    result.addAlias({callOp->getResult(resultIdx), BufferRelation::Equivalent,
                     /*isDefinite=*/true});
    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    kernel::ExtCallOp extCallOp = cast<kernel::ExtCallOp>(op);
    Location loc = extCallOp.getLoc();

    // Track original buffers (before casts) and casted buffers separately
    // Use original buffers for results to preserve aliasing identity
    llvm::SmallDenseMap<Value, Value, 8> valueToBuffer;

    // Collect bufferized args
    SmallVector<Value> bufferizedArgs;
    for (Value arg : extCallOp.getArgs()) {
      // Only bufferize tensor types
      auto tensorType = dyn_cast<TensorType>(arg.getType());
      if (!tensorType) {
        bufferizedArgs.push_back(arg);
        continue;
      }
      // Check if already bufferized (in case of duplicates)
      if (Value castedBuffer = valueToBuffer.lookup(arg)) {
        bufferizedArgs.push_back(castedBuffer);
        continue;
      }

      FailureOr<Value> bufferResult =
          bufferization::getBuffer(rewriter, arg, options, state);
      if (failed(bufferResult))
        return failure();

      SmallVector<Value> invocationStack;
      FailureOr<BaseMemRefType> memrefType =
          getBufferType(op, arg, options, state, invocationStack);
      if (failed(memrefType))
        return failure();

      if (bufferResult->getType() == *memrefType) {
        bufferizedArgs.push_back(*bufferResult);
        valueToBuffer.insert({arg, *bufferResult});
        continue;
      }

      // Cast to a dynamic layout and the appropriate memory space. This may
      // incur an allocation + copy.
      auto maybeCast = bufferization::castOrReallocMemRefValue(
          rewriter, *bufferResult, cast<MemRefType>(*memrefType), options);
      if (failed(maybeCast))
        return failure();

      valueToBuffer.insert({arg, *maybeCast});
      bufferizedArgs.push_back(*maybeCast);
    }

    // Find which arguments should be used to replace the tensor results. Cast
    // these back to the original layout.
    SmallVector<Value> results;
    for (auto [resultIdx, argIdx] :
         llvm::enumerate(extCallOp.getResultAliases())) {
      Value replacement = valueToBuffer.lookup(extCallOp.getArgs()[argIdx]);
      assert(replacement && isa<MemRefType>(replacement.getType()) &&
             "expected memref type for result");
      if (auto castOp = replacement.getDefiningOp<memref::CastOp>())
        results.push_back(castOp.getSource());
      else
        results.push_back(replacement);
    }

    rewriter.create<kernel::ExtCallOp>(
        loc, /*resultTypes=*/TypeRange{}, extCallOp.getGridSize(),
        extCallOp.getBlockSize(), bufferizedArgs, extCallOp.getKernelSym(),
        extCallOp.getArgAttrs().value_or(ArrayAttr{}),
        extCallOp.getResAttrs().value_or(ArrayAttr{}),
        extCallOp.getResultAliasesAttr(), extCallOp.getEffectsAttr());

    bufferization::replaceOpWithBufferizedValues(rewriter, extCallOp, results);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return failure();
    std::optional<Attribute> memorySpace =
        options.defaultMemorySpaceFn(tensorType);
    if (!memorySpace)
      return failure();
    return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                              *memorySpace);
  }
};

} // namespace

void kernel::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, kernel::KernelDialect *dialect) {
    kernel::CallOp::attachInterface<KernelCallOpInterface>(*ctx);
    kernel::ExtCallOp::attachInterface<KernelExtCallOpInterface>(*ctx);
  });
}
