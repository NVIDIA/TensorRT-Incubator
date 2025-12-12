//===- Ops.cpp ------------------------------------------------------------===//
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
///
/// Definition of Kernel dialect operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Utils/ScatterUtils.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "kernel-dialect"
#define DBGS() (llvm::dbgs() << __FILE__ << ":" << __LINE__ << ": ")

using namespace mlir;
using namespace mlir::kernel;

//===----------------------------------------------------------------------===//
// Parse/Print Directives
//===----------------------------------------------------------------------===//

/// Prints the functional type for kernel operations.
///
/// This prints the type in the format: `(input_types..., out_types...) ->
/// result_types...` where input_types and out_types are concatenated to form
/// the full argument list.
static void printKernelFunctionalType(AsmPrinter &printer, Operation *,
                                      ValueRange, ValueRange,
                                      TypeRange inputTypes, TypeRange outsTypes,
                                      TypeRange results) {
  printer.printFunctionalType(llvm::concat<const Type>(inputTypes, outsTypes),
                              results);
}

/// Parses the functional type for kernel operations.
///
/// Expected format: `(type1, type2, ..., typeN) -> result_type1, ...,
/// result_typeM` The parsed types are split between inputs and outs based on
/// the number of operands.
///
/// @param parser The AsmParser instance
/// @param inputs Input operands (used for counting)
/// @param outs Output operands (used for counting)
/// @param inputTypes Output parameter for input types
/// @param outsTypes Output parameter for output types
/// @param results Output parameter for result types
/// @return Success if parsing succeeds, failure otherwise
static ParseResult parseKernelFunctionalType(
    AsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> inputs,
    ArrayRef<OpAsmParser::UnresolvedOperand> outs,
    SmallVectorImpl<Type> &inputTypes, SmallVectorImpl<Type> &outsTypes,
    SmallVectorImpl<Type> &results) {
  SmallVector<Type> operandTypes;

  // Parse the operand types: (type1, type2, ...)
  if (parser.parseLParen() || parser.parseTypeList(operandTypes) ||
      parser.parseRParen())
    return failure();

  // Validate the number of parsed types matches expected operands
  if (operandTypes.size() != inputs.size() + outs.size())
    return parser.emitError(parser.getNameLoc(), "expected ")
           << inputs.size() + outs.size() << " operand types, got "
           << operandTypes.size();

  // Split types between inputs and outs
  llvm::append_range(inputTypes,
                     ArrayRef(operandTypes).take_front(inputs.size()));
  llvm::append_range(outsTypes,
                     ArrayRef(operandTypes).drop_front(inputs.size()));

  // Parse the result types: -> type1, type2, ...
  if (parser.parseArrowTypeList(results))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CombinerOp
//===----------------------------------------------------------------------===//

LogicalResult
CombinerOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                             ValueRange operands, DictionaryAttr,
                             OpaqueProperties, RegionRange,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() % 2 != 0) {
    return emitOptionalError(location, "expected an even number of arguments");
  }

  unsigned numPairs = operands.size() / 2;

  inferredReturnTypes.reserve(numPairs);
  for (auto [idx, t] :
       llvm::enumerate(TypeRange(operands).drop_back(numPairs))) {
    if (t != operands[idx + numPairs].getType())
      return emitOptionalError(
          location, "expected the types of the last ", numPairs,
          " arguments to be the same as the first ", numPairs, " arguments");
    inferredReturnTypes.push_back(t);
  }

  return success();
}

void CombinerOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the InlineClosedGroupOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

OperandRange CombinerOp::getEntrySuccessorOperands(RegionBranchPoint) {
  return getInputs();
}

void CombinerOp::getAsmBlockArgumentNames(Region &,
                                          OpAsmSetValueNameFn setNameFn) {
  unsigned numInputs = getInputs().size() / 2;
  for (BlockArgument arg : getBody().getArguments()) {
    StringRef name = arg.getArgNumber() < numInputs ? "in" : "out";
    setNameFn(arg, name);
  }
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::build(OpBuilder &builder, OperationState &odsState) {
  build(builder, odsState, {});
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  CallOp::Adaptor adaptor(operands, attributes, properties, regions);

  // If the `outs` operands are tensor types, then we should return those as
  // results. Otherwise, for memref outs, we do not return results.
  for (Type t : TypeRange(adaptor.getOuts())) {
    auto tensorType = dyn_cast<TensorType>(t);
    if (!tensorType)
      continue;
    inferredReturnTypes.push_back(tensorType);
  }
  return success();
}

LogicalResult CallOp::verify() {
  if (auto module = (*this)->getParentOfType<gpu::GPUModuleOp>())
    return emitOpError() << "cannot be nested within a "
                         << gpu::GPUModuleOp::getOperationName();
  auto atLeastOneAtMostThree = [this](StringRef name,
                                      ValueRange vals) -> LogicalResult {
    if (vals.empty() || vals.size() > 3)
      return emitOpError()
             << name << " should have between one and three values, but it has "
             << vals.size() << " values";
    return success();
  };

  if (failed(atLeastOneAtMostThree("grid size", getGridSize())))
    return failure();
  if (failed(atLeastOneAtMostThree("block size", getBlockSize())))
    return failure();

  return success();
}

/// Checks if the types are equivalent with respect to shape and element type.
/// Since we allow the GPU module to be bufferized separately from the outer
/// host module, we allow for mixed memref/tensor types. When both types are
/// 'memref', then we also check the layout (but not the address space, since
/// address space scheme used inside gpu.module may differ from outside).
///
/// Type equivalence rules:
/// - Non-shaped types must be exactly equal
/// - Tensor types: shapes and element types must match (encodings ignored)
/// - MemRef types: shapes, element types, and layouts must match (address
/// spaces ignored)
/// - Mixed tensor/memref: shapes and element types must match
///
/// @param lhs First type to compare
/// @param rhs Second type to compare
/// @return True if types are congruent according to the rules above
static bool areKernelArgTypesCongruent(Type lhs, Type rhs) {
  // Non-shaped types must match exactly
  if (!isa<ShapedType>(lhs) || !isa<ShapedType>(rhs))
    return lhs == rhs;

  // Case 1: Both types are tensors
  {
    auto lTensor = dyn_cast<RankedTensorType>(lhs);
    auto rTensor = dyn_cast<RankedTensorType>(rhs);

    // If both types are tensor, then we disregard the encoding.
    if (lTensor && rTensor)
      return lTensor.getShape() == rTensor.getShape() &&
             lTensor.getElementType() == rTensor.getElementType();
  }

  // Case 2: Both types are memrefs
  {
    auto lMemref = dyn_cast<MemRefType>(lhs);
    auto rMemref = dyn_cast<MemRefType>(rhs);

    if (lMemref && rMemref)
      return lMemref.getShape() == rMemref.getShape() &&
             lMemref.getElementType() == rMemref.getElementType() &&
             lMemref.getLayout() == rMemref.getLayout();
  }

  // Case 3: One type is memref and one is tensor
  auto lShaped = cast<ShapedType>(lhs);
  auto rShaped = cast<ShapedType>(rhs);
  return lShaped.getShape() == rShaped.getShape() &&
         lShaped.getElementType() == rShaped.getElementType();
}

/// Checks if the signatures are equivalent with respect to shape and element
/// type.
///
/// Verifies that function signatures match for kernel calls, allowing for
/// differences in memory spaces and tensor/memref conversions.
///
/// @param callOp The call operation for error reporting
/// @param lhs Expected function type (from call site)
/// @param rhs Actual function type (from callee)
/// @return Success if signatures match, failure with diagnostic otherwise
static LogicalResult signaturesEquivalentUpToMemorySpaces(Operation *callOp,
                                                          FunctionType lhs,
                                                          FunctionType rhs) {
  // Check input types
  for (auto [idx, it] :
       llvm::enumerate(llvm::zip(lhs.getInputs(), rhs.getInputs()))) {
    if (!areKernelArgTypesCongruent(std::get<0>(it), std::get<1>(it)))
      return callOp->emitOpError("callee argument #")
             << idx << " of type " << std::get<1>(it)
             << " is not compatible with call operand type " << std::get<0>(it);
  }

  // Check result types
  for (auto [idx, it] :
       llvm::enumerate(llvm::zip(lhs.getResults(), rhs.getResults()))) {
    if (!areKernelArgTypesCongruent(std::get<0>(it), std::get<1>(it)))
      return callOp->emitOpError("results at index ")
             << idx << " are not congruent";
  }
  return success();
}

FunctionOpInterface
CallOp::getKernelCallee(SymbolTableCollection &symbolTable) {
  Operation *module = (*this)->getParentWithTrait<OpTrait::SymbolTable>();
  assert(module && "expected call to be nested within symbol table");
  return dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(module, getKernelSym()));
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  FunctionOpInterface kernel = getKernelCallee(symbolTable);
  if (!kernel)
    return emitOpError() << "no valid kernel found with symbol name "
                         << getKernelSym();

  if (auto funcOp = dyn_cast<func::FuncOp>(kernel.getOperation())) {
    SmallVector<Type> expectedInputTypes((TypeRange(getInputs())));
    llvm::append_range(expectedInputTypes, TypeRange(getOuts()));
    auto expectedFuncType = FunctionType::get(
        getContext(), TypeRange(expectedInputTypes), getResultTypes());

    FunctionType kernelFuncType = funcOp.getFunctionType();

    if (kernelFuncType.getNumInputs() != getInputs().size() + getOuts().size())
      emitOpError() << "kernel signature " << kernelFuncType << " has "
                    << kernelFuncType.getNumInputs()
                    << " arguments, but the call operation expects a total of "
                    << getInputs().size() + getOuts().size() << " arguments";

    return signaturesEquivalentUpToMemorySpaces(*this, expectedFuncType,
                                                kernelFuncType);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtCallOp
//===----------------------------------------------------------------------===//

LogicalResult ExtCallOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ExtCallOp::Adaptor adaptor(operands, attributes, properties, regions);

  // Get aliasing pairs to determine which args produce results
  ArrayRef<int32_t> aliasingAttr = adaptor.getResultAliases();
  ArrayAttr effects = adaptor.getEffects();
  ValueRange args = adaptor.getArgs();

  if (effects.size() != args.size())
    return emitOptionalError(location, "effects array size (", effects.size(),
                             ") must match number of arguments (", args.size(),
                             ")");

  llvm::SmallDenseSet<int32_t, 4> seenArgs;
  for (int32_t argIdx : aliasingAttr) {
    if (seenArgs.contains(argIdx))
      return emitOptionalError(
          location,
          "aliasing configuration contains repeat argument index: ", argIdx);
    seenArgs.insert(argIdx);
  }

  for (auto [resultIdx, argIdx] : llvm::enumerate(aliasingAttr)) {
    if (static_cast<size_t>(argIdx) >= adaptor.getArgs().size())
      return emitOptionalError(location, "aliasing_args[", resultIdx,
                               "] =", argIdx, "is out of bounds for",
                               adaptor.getArgs().size(), "arguments");

    Type argType = args[argIdx].getType();
    if (!isa<RankedTensorType, MemRefType>(argType))
      return emitOptionalError(location, "aliasing_args[", resultIdx,
                               "] =", argIdx, ", but argument at index ",
                               argIdx, " has type ", argType,
                               " which is not a ranked tensor or memref type");
    // Only tensor arguments produce results
    if (isa<RankedTensorType>(argType))
      inferredReturnTypes.push_back(argType);

    auto effect = dyn_cast<StringAttr>(effects[argIdx]);
    if (!effect)
      return emitOptionalError(location, "effects[", argIdx,
                               "] is not a string attribute");
    if (effect.getValue() != "w" && effect.getValue() != "rw")
      return emitOptionalError(location, "aliasing_args[", resultIdx,
                               "] = ", argIdx, ", but effect at index ", argIdx,
                               " is \"", effect.getValue(),
                               "\", which is not \"w\" or \"rw\"");
  }
  return success();
}

FunctionOpInterface
ExtCallOp::getKernelCallee(SymbolTableCollection &symbolTable) {
  Operation *module = (*this)->getParentWithTrait<OpTrait::SymbolTable>();
  assert(module && "expected call to be nested within symbol table");
  return dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(module, getKernelSym()));
}

LogicalResult ExtCallOp::verify() {
  if (auto module = (*this)->getParentOfType<gpu::GPUModuleOp>())
    return emitOpError() << "cannot be nested within a "
                         << gpu::GPUModuleOp::getOperationName();
  auto atLeastOneAtMostThree = [this](StringRef name,
                                      ValueRange vals) -> LogicalResult {
    if (vals.empty() || vals.size() > 3)
      return emitOpError()
             << name << " should have between one and three values, but it has "
             << vals.size() << " values";
    return success();
  };

  if (failed(atLeastOneAtMostThree("grid size", getGridSize())))
    return failure();
  if (failed(atLeastOneAtMostThree("block size", getBlockSize())))
    return failure();

  return success();
}

/// Check that the memref shapes, strides/offsets, and element types are
/// compatible. They are compatible if these components of the target type are
/// either more general or the same as the source. We do not have to worry about
/// memory space differences.
static bool areMemRefTypesCompatible(MemRefType source, MemRefType target) {
  int64_t sourceOffset, targetOffset;
  SmallVector<int64_t, 4> sourceStrides, targetStrides;
  if (failed(source.getStridesAndOffset(sourceStrides, sourceOffset)) ||
      failed(target.getStridesAndOffset(targetStrides, targetOffset)))
    return false;
  auto staticToDynamicOrMatching = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(b) || (a == b);
  };
  if (!staticToDynamicOrMatching(sourceOffset, targetOffset))
    return false;
  for (auto [srcStride, targetStride, srcDim, targetDim] :
       zip_equal(sourceStrides, targetStrides, source.getShape(),
                 target.getShape())) {
    if (!staticToDynamicOrMatching(srcStride, targetStride))
      return false;
    if (!staticToDynamicOrMatching(srcDim, targetDim))
      return false;
  }
  return source.getElementType() == target.getElementType();
}

LogicalResult ExtCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  FunctionOpInterface kernel = getKernelCallee(symbolTable);
  if (!kernel)
    return emitOpError() << "no valid kernel found with symbol name "
                         << getKernelSym();

  auto calleeType = dyn_cast<FunctionType>(kernel.getFunctionType());
  if (!calleeType)
    return success();

  // `kernel.ext_call` callee must be bufferized.
  TypeRange calleeArgTypes = calleeType.getInputs();

  if (calleeType.getNumResults() != 0 ||
      llvm::any_of(calleeArgTypes,
                   [](Type type) { return isa<TensorType>(type); }))
    return emitOpError()
           << "kernel.ext_call callee must be bufferized but has signature "
           << calleeType;

  auto checkArgType = [&](Type argType, Type calleeArgType) {
    if (!isa<MemRefType, RankedTensorType>(argType)) {
      if (argType != calleeArgType)
        return failure();
      return success();
    }

    if (auto argShapedType = dyn_cast<ShapedType>(argType)) {
      auto calleeShapedType = dyn_cast<MemRefType>(calleeArgType);
      if (!calleeShapedType)
        return failure();
      if (failed(mlir::verifyCompatibleShape(argShapedType.getShape(),
                                             calleeShapedType.getShape())))
        return failure();
      if (calleeShapedType.getElementType() != argShapedType.getElementType())
        return failure();

      if (auto argMemRefType = dyn_cast<MemRefType>(argType)) {
        if (!areMemRefTypesCompatible(argMemRefType, calleeShapedType))
          return failure();
      }
      return success();
    }

    return failure();
  };

  if (calleeType.getNumInputs() != getArgs().size())
    return emitOpError() << "number of arguments " << getArgs().size()
                         << " does not match callee function type "
                         << calleeType;

  for (auto [argType, calleeArgType] :
       llvm::zip_equal(getArgs().getTypes(), calleeArgTypes)) {
    if (failed(checkArgType(argType, calleeArgType)))
      return emitOpError() << "argument type " << argType
                           << " is not compatible with callee argument type "
                           << calleeArgType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExtCallOp - CallOpInterface methods
//===----------------------------------------------------------------------===//

ExtCallOp::operand_range ExtCallOp::getArgOperands() { return getArgs(); }

CallInterfaceCallable ExtCallOp::getCallableForCallee() {
  return getKernelSym();
}

void ExtCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  this->setKernelSymAttr(cast<SymbolRefAttr>(callee));
}

unsigned ExtCallOp::getNumNonForwardedArguments() {
  return getGridSize().size() + getBlockSize().size();
}

MutableOperandRange ExtCallOp::getArgOperandsMutable() {
  return MutableOperandRange(getOperation(), getNumNonForwardedArguments(),
                             getArgs().size());
}

bool ExtCallOp::isForwardedOperand(OpOperand *operand) {
  return operand->getOperandNumber() >= getNumNonForwardedArguments();
}

unsigned ExtCallOp::getForwardedArgumentIndex(OpOperand *operand) {
  assert(isForwardedOperand(operand) && "expected operand is forwarded");
  return operand->getOperandNumber() - getNumNonForwardedArguments();
}

//===----------------------------------------------------------------------===//
// ExtCallOp - Aliasing and effects methods
//===----------------------------------------------------------------------===//

bool ExtCallOp::argHasWriteEffect(unsigned argIdx) {
  auto effects = getEffects();
  assert(argIdx < effects.size() && "argument index out of bounds");
  StringRef effect = cast<StringAttr>(effects[argIdx]).getValue();
  return effect == "rw" || effect == "w";
}

bool ExtCallOp::argHasReadEffect(unsigned argIdx) {
  auto effects = getEffects();
  assert(argIdx < effects.size() && "argument index out of bounds");
  StringRef effect = cast<StringAttr>(effects[argIdx]).getValue();
  return effect == "r" || effect == "rw";
}

void ExtCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto [operand, effect] : llvm::zip_equal(
           getArgsMutable(), getEffects().getAsValueRange<StringAttr>())) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    if (effect.find('r') != llvm::StringRef::npos)
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
    if (effect.find('w') != llvm::StringRef::npos)
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  return mlir::kernel::verifyStablehloLikeScatterOp(
      getLoc(), getInits(), getIndices(), getUpdates(), getUpdateWindowDims(),
      getInsertedWindowDims(), getInputBatchingDims(),
      getScatterIndicesBatchingDims(), getScatterDimsToOperandDims(),
      getIndexVectorDim(), getUpdateComputation());
}

MutableOperandRange ScatterOp::getDpsInitsMutable() {
  return getInitsMutable();
}

LogicalResult
ScatterOp::reifyResultShapes(OpBuilder &builder,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> shapes;
  Location loc = getOperation()->getLoc();
  IRRewriter rewriter(builder);
  ShapedType inputShapedType =
      llvm::cast<ShapedType>(getInits().front().getType());
  for (int64_t dim : llvm::seq<int64_t>(0, inputShapedType.getRank())) {
    if (!inputShapedType.isDynamicDim(dim)) {
      // Static dim: Return IntegerAttr.
      shapes.push_back(builder.getIndexAttr(inputShapedType.getDimSize(dim)));
    } else {
      // Dynamic dim: Return Value.
      OpFoldResult ofr =
          linalg::createOrFoldDimOp(builder, loc, getInits().front(), dim);
      shapes.push_back(getValueOrCreateConstantIndexOp(builder, loc, ofr));
    }
  }
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  Operation *op = getOperation();

  // Must have 1 input (keys-only) or 2 inputs (key-value)
  const size_t numInputs = getInputs().size();
  if (numInputs != 1 && numInputs != 2)
    return op->emitOpError("expected one or two input operands");

  // Must have same number of outputs as inputs
  if (getResults().size() != numInputs)
    return op->emitOpError(
        "expected number of results to match number of inputs");

  if (getInputs().getTypes() != getResults().getTypes())
    return op->emitOpError("expected result types to match input types");

  if (getBlockThreads() <= 0)
    return op->emitOpError("block_threads must be positive");

  if (getItemsPerThread() <= 0)
    return op->emitOpError("items_per_thread must be positive");

  return success();
}

LogicalResult
SortOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.assign(operands.getTypes().begin(),
                             operands.getTypes().end());
  return success();
}

LogicalResult
SortOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Location loc = getOperation()->getLoc();

  // For each output, reify its shape
  for (Value output : getInputs()) {
    SmallVector<OpFoldResult> shapes;
    ShapedType outputType = cast<RankedTensorType>(output.getType());

    for (int64_t dim : llvm::seq<int64_t>(0, outputType.getRank())) {
      if (!outputType.isDynamicDim(dim)) {
        // Static dim: Return IntegerAttr
        shapes.push_back(builder.getIndexAttr(outputType.getDimSize(dim)));
      } else {
        // Dynamic dim: Return Value
        OpFoldResult ofr = linalg::createOrFoldDimOp(builder, loc, output, dim);
        shapes.push_back(getValueOrCreateConstantIndexOp(builder, loc, ofr));
      }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KernelDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct KernelDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Kernel call-like should not allow inlining.
  bool isLegalToInline(Operation *, Operation *, bool) const final {
    return false;
  }

  /// All operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All functions can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void KernelDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "mlir-kernel/Kernel/IR/Ops.cpp.inc"
      >();
}

void KernelDialect::initialize() {
  registerOps();
  registerAttributes();
  addInterface<KernelDialectInlinerInterface>();
}

LogicalResult
KernelDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                        unsigned argIndex,
                                        NamedAttribute attribute) {
  if (attribute.getName() ==
      kernel::KernelDialect::kKernelAlignmentArgAttrName) {
    auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op);
    if (!funcOp)
      return op->emitError()
             << "kernel.alignment must decorate a function argument";
    auto kernelAlignmentAttr = dyn_cast<IntegerAttr>(attribute.getValue());

    if (!kernelAlignmentAttr)
      return funcOp->emitError()
             << "kernel.alignment's value should be Integer type.";
    // the kernel_alignment must decorate a tensor type
    if (argIndex >= funcOp.getArgumentTypes().size()) {
      return funcOp->emitError()
             << argIndex << " is out of the bound of input arguments "
             << funcOp.getArgumentTypes().size();
    }
    if (!mlir::isa<mlir::TensorType>(funcOp.getArgumentTypes()[argIndex])) {
      return funcOp->emitError()
             << "kernel.alignment must decorate a tensor type, but got "
             << funcOp.getArgumentTypes()[argIndex];
    }
    // the value type of kernel.alignment should be int64
    if (!kernelAlignmentAttr.getType().isInteger(64)) {
      return funcOp->emitError()
             << "kernel.alignment's value should have i64 type, but got "
             << kernelAlignmentAttr.getType();
    }
    // the value of kernel.alignment should be power of 2
    if (!llvm::isPowerOf2_64(kernelAlignmentAttr.getInt())) {
      return funcOp->emitError()
             << "kernel.alignment's value should be power of two, but got "
             << kernelAlignmentAttr.getInt();
    }
  }

  return success();
}

LogicalResult
KernelDialect::verifyOperationAttribute(Operation *op,
                                        NamedAttribute attribute) {
  if (attribute.getName() ==
      kernel::KernelDialect::getKernelFunctionNumThreadsAttrName()) {
    auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op);
    if (!funcOp)
      return op->emitError() << "'kernel.num_threads' must decorate a function";
    if (!funcOp->getParentOfType<gpu::GPUModuleOp>())
      return op->emitError() << "'kernel.num_threads' must decorate a "
                                "function nested in a gpu.module";
    auto numThreadsAttr = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!numThreadsAttr)
      return op->emitError() << "'kernel.num_threads' value should be an i64 "
                                "integer attribute.";
    if (numThreadsAttr.getInt() <= 0)
      return op->emitError()
             << "'kernel.num_threads' value should be positive integer";
    return success();
  }

  if (attribute.getName() ==
      kernel::KernelDialect::getGpuModuleWarpSizeAttrName()) {
    auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op);
    if (!gpuModule)
      return op->emitError() << "'kernel.warp_size' must decorate a gpu.module";
    return success();
  }

  return success();
}

std::optional<int64_t>
kernel::KernelDialect::getNumThreadsRequired(FunctionOpInterface func) {
  if (auto numThreads = func->getAttrOfType<IntegerAttr>(
          kernel::KernelDialect::getKernelFunctionNumThreadsAttrName()))
    return numThreads.getInt();
  return std::nullopt;
}

int64_t
kernel::KernelDialect::getGpuModuleWarpSize(gpu::GPUModuleOp gpuModule) {
  if (auto warpSize = gpuModule->getAttrOfType<IntegerAttr>(
          kernel::KernelDialect::getGpuModuleWarpSizeAttrName()))
    return warpSize.getInt();
  return 32;
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition.
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-kernel/Kernel/IR/Ops.cpp.inc"
