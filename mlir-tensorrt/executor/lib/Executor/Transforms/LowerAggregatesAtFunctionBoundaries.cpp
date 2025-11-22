//===- LowerAggregatesAtFunctionBoundaries.cpp ----------------------------===//
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
/// Implements the `executor-lower-aggregates-at-function-boundaries` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORLOWERAGGREGATESATFUNCTIONBOUNDARIESPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

namespace {
enum class StructLoweringMode { Direct, Indirect, Unpacked };
} // namespace

/// Parse the string representation of a struct lowering mode.
/// Valid modes are "direct", "indirect", and "unpacked".
/// @param mode The string to parse
/// @return The corresponding StructLoweringMode enum value
/// @throws Calls report_fatal_error if mode is invalid
static StructLoweringMode parseStructLoweringMode(StringRef mode) {
  if (mode == "direct")
    return StructLoweringMode::Direct;
  if (mode == "indirect")
    return StructLoweringMode::Indirect;
  if (mode == "unpacked")
    return StructLoweringMode::Unpacked;
  llvm::report_fatal_error("invalid struct lowering mode");
}

/// Recursively flatten a TableType value into a list of non-aggregate values.
/// Nested tables are recursively flattened until only scalar values remain.
/// @param builder The OpBuilder to create extraction operations
/// @param loc The location for created operations
/// @param val The table value to flatten (must be of TableType)
/// @return A vector of scalar values extracted from the table
static SmallVector<Value> flattenStructToValues(OpBuilder &builder,
                                                Location loc, Value val) {
  auto structType = cast<executor::TableType>(val.getType());
  SmallVector<Value> results;
  for (auto [idx, t] : llvm::enumerate(structType.getBody())) {
    Value extracted =
        builder.create<executor::ExtractTableValueOp>(loc, t, val, idx);
    if (!isa<executor::TableType>(extracted.getType())) {
      results.push_back(extracted);
      continue;
    }
    llvm::append_range(results, flattenStructToValues(builder, loc, extracted));
  }
  return results;
}

/// Create a TableType from a flat list of values. This correctly handles nested
/// table types.
static TypedValue<TableType> createTableTypeFromValues(OpBuilder &builder,
                                                       Location loc,
                                                       TableType structType,
                                                       ValueRange values) {

  /// Instead of recursion, use a stack to parse nested table types.
  /// Each stack frame has the available range of values, the remaining types of
  /// the struct body to parse, and the operands of the `CreateTableOp`
  /// accumulated so far. Except for `operands`, each of these is just a view of
  /// the original input arguments.
  /// When the `types` is empty, we know that we can pop the frame and create
  /// the `CreateTableOp` with the accumulated operands, then forward that to
  /// the parent frame.
  struct FrameInfo {
    TableType structType;
    ValueRange values;
    ArrayRef<Type> types;
    SmallVector<Value> operands;
  };

  // Initialize the stack with the root frame.
  std::vector<FrameInfo> frames = {
      FrameInfo{structType, values, structType.getBody(), {}}};

  while (!frames.empty()) {
    // Try to parse individual elements from the value list.
    while (!frames.back().types.empty()) {
      FrameInfo &frame = frames.back();
      Type t = frame.types.front();
      assert(!frame.values.empty() && "expected values");
      assert(!frame.types.empty() && "expected types");
      // If the current value can be used as is, add it to the operands
      // and drop value/type.
      if (frame.values.front().getType() == t) {
        frame.operands.push_back(frame.values.front());
        frame.values = frame.values.drop_front();
        frame.types = frame.types.drop_front();
        continue;
      }
      // If this a TableType, then push a new frame.
      assert(isa<executor::TableType>(t) && "expected table type");
      auto subTableType = cast<executor::TableType>(t);
      frames.push_back(
          FrameInfo{subTableType, frame.values, subTableType.getBody(), {}});
      break;
    }
    // If the current frame is done, pop it and create the `CreateTableOp`.
    if (frames.back().types.empty()) {
      FrameInfo frame = frames.back();
      frames.pop_back();
      assert(frame.operands.size() == frame.structType.getBody().size() &&
             "expected same number of operands and types");
      auto tableOp = builder.create<executor::CreateTableOp>(
          loc, frame.structType, frame.operands);
      if (!frames.empty()) {
        // Forward the result to the parent frame.
        FrameInfo &parentFrame = frames.back();
        parentFrame.operands.push_back(tableOp.getResult());
        parentFrame.values = frame.values;
        parentFrame.types = parentFrame.types.drop_front();
      } else {
        // This is the root frame, return the result.
        return cast<TypedValue<TableType>>(tableOp.getResult());
      }
    }
  }
  llvm_unreachable("expected to return a table type");
}

namespace {

class AggregateTypeConverter : public TypeConverter {
public:
  AggregateTypeConverter(MLIRContext *ctx, const DataLayout &dataLayout,
                         StructLoweringMode mode);

  /// Create a zero constant of the appropriate index type.
  /// Used as offset for memory operations.
  /// @param builder The OpBuilder to create the constant
  /// @param loc The location for the created constant
  /// @return A typed zero constant of index type
  TypedValue<IntegerType> buildZeroOffset(OpBuilder &builder,
                                          Location loc) const;

  /// Allocate stack memory (alloca) sufficient to store an aggregate value.
  /// @param builder The OpBuilder to create the alloca operation
  /// @param loc The location for the created operation
  /// @param aggregateType The type of aggregate to allocate space for
  /// @return A pointer to the allocated stack memory
  TypedValue<executor::PointerType>
  allocateStackMemoryForAggregate(OpBuilder &builder, Location loc,
                                  TableType aggregateType) const;

  /// Promote an aggregate value to stack memory by allocating space and storing
  /// it. This is used for indirect mode where aggregates are passed by
  /// reference.
  /// @param builder The OpBuilder to create operations
  /// @param loc The location for created operations
  /// @param aggregate The aggregate value to promote
  /// @return A pointer to the stack location containing the aggregate
  TypedValue<executor::PointerType>
  promoteAggregateToStack(OpBuilder &builder, Location loc,
                          TypedValue<TableType> aggregate) const;

  /// Load an aggregate value from memory.
  /// @param builder The OpBuilder to create the load operation
  /// @param loc The location for the created operation
  /// @param memory The pointer to load from
  /// @param aggregateType The expected type of the loaded aggregate
  /// @return The loaded aggregate value
  TypedValue<TableType>
  loadAggregateFromMemory(OpBuilder &builder, Location loc,
                          TypedValue<executor::PointerType> memory,
                          TableType aggregateType) const;

  /// Store an aggregate value to memory.
  /// @param builder The OpBuilder to create the store operation
  /// @param loc The location for the created operation
  /// @param aggregate The aggregate value to store
  /// @param memory The pointer to store to
  void storeAggregateToMemory(OpBuilder &builder, Location loc, Value aggregate,
                              Value memory) const;

  /// SignatureRemapping is just a SignatureConversion plus a DictionaryAttr
  /// array for converting argument/result attributes.
  struct SignatureRemapping : public SignatureConversion {
    using SignatureConversion::SignatureConversion;
    SmallVector<DictionaryAttr> attributes;
  };

  /// FuncTypeRemapInfo contains the remapping information for a function type
  /// and argument/result attribute dictionaries. Arguments are remapped
  /// according to "inputRemapping" and results are remapped according to
  /// "resultRemapping".
  struct FuncTypeRemapInfo {
    FunctionType functionType;
    SignatureRemapping inputRemapping;
    SignatureRemapping resultRemapping;

    FuncTypeRemapInfo(FunctionType functionType)
        : functionType(functionType),
          inputRemapping(functionType.getNumInputs()),
          resultRemapping(functionType.getNumResults()) {}

    /// Return the argument index in the remapped signature that marks the start
    /// of the arguments corresponding to pointers for promoted results. Note
    /// that the promoted list may be empty so this may point to the end of the
    /// remapped arg list.
    unsigned getStartOfPromotedResultArgIndex() const {
      unsigned numInputsBeforeTransform = functionType.getNumInputs();
      if (numInputsBeforeTransform > 0) {
        std::optional<SignatureConversion::InputMapping> mapping =
            inputRemapping.getInputMapping(numInputsBeforeTransform - 1);
        assert(mapping && "expected mapping");
        return mapping->inputNo + mapping->size;
      }
      return 0;
    }

    /// For debugging purposes, dump the remapping information to
    /// `llvm::dbgs()`.
    void dump() const;
  };

  /// Compute the type remapping for a function signature based on the lowering
  /// mode. This determines how aggregate arguments and results are transformed.
  /// In indirect mode, aggregate results become additional pointer arguments.
  /// In unpacked mode, aggregates are flattened into multiple scalar values.
  /// @param funcType The original function type
  /// @param argAttrDicts Attributes for function arguments (may be null)
  /// @param resultAttrDicts Attributes for function results (may be null)
  /// @param funcTypeRemapInfo Output parameter containing the remapping
  /// information
  /// @return Success if remapping succeeded, failure otherwise
  LogicalResult
  getFuncTypeRemapping(FunctionType funcType, ArrayAttr argAttrDicts,
                       ArrayAttr resultAttrDicts,
                       FuncTypeRemapInfo &funcTypeRemapInfo) const;

  MLIRContext *ctx;
  const DataLayout &dataLayout;
  const StructLoweringMode structLoweringMode;
  Type indexType;
  executor::PointerType hostPointerType;
};
} // namespace

void AggregateTypeConverter::FuncTypeRemapInfo::dump() const {
#ifndef NDEBUG
  llvm::dbgs() << "FuncTypeRemapInfo:\n";
  FunctionType convertedFunctionType = FunctionType::get(
      functionType.getContext(), inputRemapping.getConvertedTypes(),
      resultRemapping.getConvertedTypes());
  llvm::dbgs() << llvm::formatv(
      "original function type: {0}\nconverted function type: "
      "{1}\nstartOfPromotedResultArgIndex: {2}\n",
      functionType, convertedFunctionType, getStartOfPromotedResultArgIndex());
#endif // NDEBUG
}

AggregateTypeConverter::AggregateTypeConverter(MLIRContext *ctx,
                                               const DataLayout &dataLayout,
                                               StructLoweringMode mode)
    : ctx(ctx), dataLayout(dataLayout), structLoweringMode(mode) {

  this->indexType = IntegerType::get(
      this->ctx, this->dataLayout.getTypeSizeInBits(IndexType::get(this->ctx)));
  this->hostPointerType =
      executor::PointerType::get(this->ctx, executor::MemoryType::host);

  this->addConversion([](Type t) { return t; });
  this->addConversion(
      [&](executor::TableType t,
          SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
        if (structLoweringMode == StructLoweringMode::Direct) {
          results.push_back(t);
          return success();
        }

        if (structLoweringMode == StructLoweringMode::Indirect) {
          results.push_back(hostPointerType);
          return success();
        }

        assert(structLoweringMode == StructLoweringMode::Unpacked &&
               "unexpected struct lowering mode");
        for (Type t : t.getBody()) {
          SmallVector<Type> subTypes;
          if (failed(convertType(t, subTypes)))
            return failure();
          llvm::append_range(results, subTypes);
        }
        return success();
      });

  addSourceMaterialization([&](OpBuilder &builder, TableType resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (structLoweringMode == StructLoweringMode::Unpacked)
      return createTableTypeFromValues(builder, loc, resultType, inputs);
    if (structLoweringMode == StructLoweringMode::Indirect &&
        inputs.size() == 1 && inputs.front().getType() == hostPointerType) {
      return loadAggregateFromMemory(
          builder, loc, cast<TypedValue<executor::PointerType>>(inputs.front()),
          resultType);
    }
    return {};
  });
  addTargetMaterialization([&](OpBuilder &builder, TypeRange resultTypes,
                               ValueRange inputs,
                               Location loc) -> SmallVector<Value> {
    if (inputs.size() != 1 || !isa<TableType>(inputs.front().getType()))
      return {};
    if (structLoweringMode == StructLoweringMode::Indirect) {
      assert(resultTypes.size() == 1 && resultTypes.front() == hostPointerType);
      TypedValue<executor::PointerType> memory = promoteAggregateToStack(
          builder, loc, cast<TypedValue<TableType>>(inputs.front()));
      return {memory};
    }
    if (structLoweringMode == StructLoweringMode::Unpacked) {
      SmallVector<Value> elements =
          flattenStructToValues(builder, loc, inputs.front());
      assert(elements.size() == resultTypes.size() &&
             "expected same number of elements and result types");
      assert(TypeRange(elements) == resultTypes && "expected same types");
      return elements;
    }
    return {};
  });
}

TypedValue<IntegerType>
AggregateTypeConverter::buildZeroOffset(OpBuilder &builder,
                                        Location loc) const {
  return cast<TypedValue<IntegerType>>(
      builder.create<executor::ConstantOp>(loc, builder.getZeroAttr(indexType))
          .getResult());
}

void AggregateTypeConverter::storeAggregateToMemory(OpBuilder &builder,
                                                    Location loc,
                                                    Value aggregate,
                                                    Value memory) const {
  builder.create<executor::StoreOp>(loc, memory, buildZeroOffset(builder, loc),
                                    aggregate);
}

TypedValue<TableType> AggregateTypeConverter::loadAggregateFromMemory(
    OpBuilder &builder, Location loc, TypedValue<executor::PointerType> memory,
    TableType aggregateType) const {
  return cast<TypedValue<TableType>>(
      builder
          .create<executor::LoadOp>(loc, aggregateType, memory,
                                    buildZeroOffset(builder, loc))
          .getResult());
}

TypedValue<executor::PointerType>
AggregateTypeConverter::allocateStackMemoryForAggregate(
    OpBuilder &builder, Location loc, TableType aggregateType) const {
  Value one =
      builder.create<executor::ConstantOp>(loc, builder.getOneAttr(indexType));
  return cast<TypedValue<executor::PointerType>>(
      builder
          .create<executor::AllocaOp>(loc, hostPointerType,
                                      /*num_elements=*/one,
                                      /*alignment=*/IntegerAttr{},
                                      /*element_type=*/aggregateType)
          .getResult());
}

TypedValue<executor::PointerType>
AggregateTypeConverter::promoteAggregateToStack(
    OpBuilder &builder, Location loc, TypedValue<TableType> aggregate) const {
  auto allocaOp =
      allocateStackMemoryForAggregate(builder, loc, aggregate.getType());
  storeAggregateToMemory(builder, loc, aggregate, allocaOp);
  return allocaOp;
}

/// Returns a new dictionary attribute with the given updates applied
/// only if the keys are not present in the original dictionary.
static DictionaryAttr
updateIfMissing(DictionaryAttr attr, ArrayRef<NamedAttribute> updateIfMissing) {
  llvm::SmallVector<NamedAttribute> kv = llvm::to_vector(attr);
  for (NamedAttribute update : updateIfMissing) {
    // Insert but don't overwrite.
    if (attr.getNamed(update.getName()) == std::nullopt)
      kv.push_back(update);
  }
  return DictionaryAttr::get(attr.getContext(), kv);
}

LogicalResult AggregateTypeConverter::getFuncTypeRemapping(
    FunctionType funcType, ArrayAttr argAttrDicts, ArrayAttr resultAttrDicts,
    FuncTypeRemapInfo &funcTypeRemapInfo) const {
  const auto emptyDictAttr = DictionaryAttr::get(ctx, {});
  const IntegerType indexType = IntegerType::get(ctx, 32);
  auto updateAttrDicts = [&](ArrayAttr attrDicts, unsigned originalIndex,
                             unsigned convertedTypesSize,
                             SmallVectorImpl<DictionaryAttr> &attributes,
                             std::optional<int64_t> resultIndex) {
    llvm::SmallVector<NamedAttribute, 1> updates;
    if (resultIndex)
      updates.push_back(
          NamedAttribute(ExecutorDialect::kResultArgAttrName,
                         IntegerAttr::get(indexType, *resultIndex)));
    if (!attrDicts) {
      auto firstArgDict = updateIfMissing(emptyDictAttr, updates);
      attributes.push_back(firstArgDict);
      if (convertedTypesSize > 1)
        attributes.append(convertedTypesSize - 1, emptyDictAttr);
      return;
    }
    assert(originalIndex < attrDicts.size() &&
           "original index is out of bounds");
    attributes.push_back(updateIfMissing(
        cast<DictionaryAttr>(attrDicts[originalIndex]), updates));
    if (convertedTypesSize > 1)
      attributes.append(convertedTypesSize - 1, emptyDictAttr);
  };

  // Convert args first since some results might be promoted to arguments;
  // promoted results should go at the end of the new argument list.
  for (auto [idx, argType] : llvm::enumerate(funcType.getInputs())) {
    SmallVector<Type> convertedTypes;
    if (failed(convertType(argType, convertedTypes)))
      return failure();
    funcTypeRemapInfo.inputRemapping.addInputs(idx, convertedTypes);
    updateAttrDicts(argAttrDicts, idx, convertedTypes.size(),
                    funcTypeRemapInfo.inputRemapping.attributes, std::nullopt);
  }

  assert(funcTypeRemapInfo.inputRemapping.attributes.size() ==
             funcTypeRemapInfo.inputRemapping.getConvertedTypes().size() &&
         "expected same number of attributes and converted types");

  for (auto [idx, resultType] : llvm::enumerate(funcType.getResults())) {
    SmallVector<Type> convertedTypes;
    if (failed(convertType(resultType, convertedTypes)))
      return failure();
    if (isa<executor::TableType>(resultType)) {
      if (structLoweringMode == StructLoweringMode::Indirect) {
        assert(convertedTypes.size() == 1 &&
               convertedTypes.front() == hostPointerType);
        funcTypeRemapInfo.inputRemapping.addInputs(convertedTypes);
        // Make sure that we update the *input* attributes from the *result*
        // attributes.
        updateAttrDicts(resultAttrDicts, idx, convertedTypes.size(),
                        funcTypeRemapInfo.inputRemapping.attributes, idx);
        continue;
      }
    }
    // Append result types and propagate result attributes.
    funcTypeRemapInfo.resultRemapping.addInputs(idx, convertedTypes);
    updateAttrDicts(resultAttrDicts, idx, convertedTypes.size(),
                    funcTypeRemapInfo.resultRemapping.attributes, idx);
  }

  assert(funcTypeRemapInfo.inputRemapping.attributes.size() ==
             funcTypeRemapInfo.inputRemapping.getConvertedTypes().size() &&
         "expected same number of attributes and converted types");

  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionOpInterface op. This only supports ops which use FunctionType to
/// represent their type.
namespace {
struct FunctionOpInterfaceSignatureConversion
    : public OpInterfaceConversionPattern<FunctionOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(FunctionOpInterface funcOp, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
    if (!type)
      return failure();

    const auto &typeConverter = *getTypeConverter<AggregateTypeConverter>();

    // Convert the original function types.
    AggregateTypeConverter::FuncTypeRemapInfo funcTypeRemapInfo(type);
    if (failed(typeConverter.getFuncTypeRemapping(type, funcOp.getAllArgAttrs(),
                                                  funcOp.getAllResultAttrs(),
                                                  funcTypeRemapInfo)))
      return failure();

    // Convert the function body.
    Block *newEntry{nullptr};
    if (!funcOp.isDeclaration()) {
      newEntry = rewriter.applySignatureConversion(
          &funcOp.getFunctionBody().front(), funcTypeRemapInfo.inputRemapping,
          &typeConverter);
      if (!newEntry)
        return failure();
    }

    // Update the function signature in-place.
    auto newType = FunctionType::get(
        rewriter.getContext(),
        funcTypeRemapInfo.inputRemapping.getConvertedTypes(),
        funcTypeRemapInfo.resultRemapping.getConvertedTypes());

    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setType(newType);
      funcOp.setAllArgAttrs(funcTypeRemapInfo.inputRemapping.attributes);
      funcOp.setAllResultAttrs(funcTypeRemapInfo.resultRemapping.attributes);
    });

    // Now also convert the return ops. We do this here since we need to know
    // that the function mapping has already been handled to access promoted
    // result arguments.
    funcOp->walk([&](func::ReturnOp op) {
      rewriter.setInsertionPoint(op);
      SmallVector<Value> convertedOperands;
      unsigned promotedArgIdx =
          funcTypeRemapInfo.getStartOfPromotedResultArgIndex();
      for (auto [idx, v] : llvm::enumerate(op.getOperands())) {
        if (isa<executor::TableType>(v.getType()) &&
            typeConverter.structLoweringMode == StructLoweringMode::Indirect) {
          assert(promotedArgIdx < newEntry->getNumArguments() &&
                 "invalid promoted argument index");
          BlockArgument arg = newEntry->getArgument(promotedArgIdx++);
          assert(arg.getType() == typeConverter.hostPointerType &&
                 "expected host pointer type");
          typeConverter.storeAggregateToMemory(rewriter, op.getLoc(),
                                               op.getOperands()[idx], arg);
          continue;
        }
        if (typeConverter.isLegal(v.getType())) {
          convertedOperands.push_back(v);
          continue;
        }
        SmallVector<Type> convertedTypes;
        if (failed(typeConverter.convertType(v.getType(), convertedTypes)))
          return WalkResult::interrupt();
        SmallVector<Value> materialized =
            typeConverter.materializeTargetConversion(
                rewriter, op.getLoc(), convertedTypes, op.getOperands()[idx]);
        llvm::append_range(convertedOperands, materialized);
      }
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, convertedOperands);
      return WalkResult::advance();
    });

    return success();
  }
};

struct CallConversionPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const AggregateTypeConverter *typeConverter =
        this->getTypeConverter<AggregateTypeConverter>();
    FunctionType functionType = FunctionType::get(
        op.getContext(), op.getOperandTypes(), op.getResultTypes());
    AggregateTypeConverter::FuncTypeRemapInfo funcTypeRemapInfo(functionType);
    if (failed(typeConverter->getFuncTypeRemapping(
            functionType, ArrayAttr{}, ArrayAttr{}, funcTypeRemapInfo)))
      return failure();

    SmallVector<Value> convertedOperands;
    for (ValueRange v : adaptor.getOperands())
      llvm::append_range(convertedOperands, v);

    SmallVector<Value> promotedResultOutputBuffers;
    for (auto [idx, resultType] : llvm::enumerate(functionType.getResults())) {
      if (isa<executor::TableType>(resultType) &&
          typeConverter->structLoweringMode == StructLoweringMode::Indirect) {
        TypedValue<executor::PointerType> outputMemory =
            typeConverter->allocateStackMemoryForAggregate(
                rewriter, op.getLoc(), cast<TableType>(resultType));
        convertedOperands.push_back(outputMemory);
      }
    }

    auto newCallOp = rewriter.create<func::CallOp>(
        op.getLoc(), op.getCallee(),
        funcTypeRemapInfo.resultRemapping.getConvertedTypes(),
        convertedOperands);

    unsigned dpsArgIdx = funcTypeRemapInfo.getStartOfPromotedResultArgIndex();
    SmallVector<Value> replacements;
    for (auto [idx, resultType] : llvm::enumerate(functionType.getResults())) {
      // If this is a aggregate result and we are lowering indirectly, then
      // load the result from the memory buffer.
      if (isa<executor::TableType>(resultType) &&
          typeConverter->structLoweringMode == StructLoweringMode::Indirect) {
        assert(dpsArgIdx < convertedOperands.size() &&
               "unexpected number of converted operands");
        auto memory = cast<TypedValue<executor::PointerType>>(
            convertedOperands[dpsArgIdx++]);
        TypedValue<TableType> result = typeConverter->loadAggregateFromMemory(
            rewriter, op.getLoc(), memory, cast<TableType>(resultType));
        replacements.push_back(result);
        continue;
      }

      // Otherwise, we need to handle non-aggregate and aggregate unpacked
      // results.
      SmallVector<Value> components;
      std::optional<AggregateTypeConverter::SignatureConversion::InputMapping>
          mapping = funcTypeRemapInfo.resultRemapping.getInputMapping(idx);
      assert(mapping && "expected mapping");
      llvm::append_range(components, newCallOp->getResults().slice(
                                         mapping->inputNo, mapping->size));
      // If no conversion is required, just
      // append the value.
      if (components.size() == 1 &&
          components.front().getType() == resultType) {
        replacements.push_back(components.front());
        continue;
      }
      // Otherwise, create the source materialization.
      assert(isa<executor::TableType>(resultType) && "expected table type");
      auto replacement = createTableTypeFromValues(
          rewriter, op.getLoc(), cast<TableType>(resultType), components);
      replacements.push_back(replacement);
      continue;
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Handle return-like branching operation conversion.
class BranchOpInterfaceTypeConversion
    : public OpInterfaceConversionPattern<BranchOpInterface> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(BranchOpInterface op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Note that we can't handle non-return-like branching ops
    // or ops with attr-sized operand segments with
    // this pattern.
    if (!op->hasTrait<OpTrait::ReturnLike>() ||
        op->hasTrait<OpTrait::AttrSizedOperandSegments>())
      return failure();

    // For a branch operation, only some operands go to the target blocks, so
    // only rewrite those.
    SmallVector<SmallVector<Value>> newOperands = llvm::map_to_vector(
        op->getOperands(), [&](Value v) { return SmallVector<Value>{v}; });
    for (int succIdx = 0, succEnd = op->getBlock()->getNumSuccessors();
         succIdx < succEnd; ++succIdx) {
      OperandRange forwardedOperands =
          op.getSuccessorOperands(succIdx).getForwardedOperands();
      if (forwardedOperands.empty())
        continue;
      for (unsigned idx = forwardedOperands.getBeginOperandIndex(),
                    eidx = idx + forwardedOperands.size();
           idx < eidx; ++idx) {
        newOperands[idx] = llvm::to_vector(adaptor[idx]);
      }
    }
    SmallVector<Value> flattened;
    for (ArrayRef<Value> v : newOperands)
      llvm::append_range(flattened, v);

    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(flattened); });
    return success();
  }
};

class LowerAggregatesAtFunctionBoundariesPass
    : public executor::impl::
          ExecutorLowerAggregatesAtFunctionBoundariesPassBase<
              LowerAggregatesAtFunctionBoundariesPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *module = getOperation();

    const mlir::DataLayoutAnalysis &dataLayoutAnalysis =
        getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(module);
    markAnalysesPreserved<DataLayoutAnalysis>();

    StructLoweringMode loweringMode = parseStructLoweringMode(mode);

    AggregateTypeConverter typeConverter(&getContext(), dataLayout,
                                         loweringMode);

    RewritePatternSet patterns(&getContext());
    patterns.add<FunctionOpInterfaceSignatureConversion, CallConversionPattern,
                 BranchOpInterfaceTypeConversion>(typeConverter, &getContext());

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&typeConverter](func::CallOp op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (!op->hasTrait<OpTrait::ReturnLike>())
        return true;
      return isLegalForReturnOpTypeConversionPattern(
          op, typeConverter,
          /*returnOpAlwaysLegal*/ false);
    });

    ConversionConfig config;
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
