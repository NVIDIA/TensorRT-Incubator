//===- Executor.cpp -------------------------------------------------------===//
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
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::executor;

//===----------------------------------------------------------------------===//
// FunctionMetadataAttr
//===----------------------------------------------------------------------===//

/// Check type of `min` matches type of `max` and that all elements of `min` are
/// <= elements of `max`
LogicalResult executor::DimensionBoundsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr min,
    DenseI64ArrayAttr max) {

  if (min.getElementType() != max.getElementType())
    return emitError() << "DimensionBoundsAttr 'min' and 'max' must have "
                          "matching element types; found min type: "
                       << min.getElementType()
                       << ", max type: " << max.getElementType();

  if (min.getSize() != max.getSize())
    return emitError() << "DimensionBoundsAttr 'min' and 'max' must have the "
                          "same size; found min size: "
                       << min.getSize() << ", max size: " << max.getSize();
  for (int i = 0; i < min.getSize(); ++i) {
    if (min[i] < 0)
      return emitError() << "DimensionBoundsAttr min[" << i << "] : " << min[i]
                         << " must be greater than or equal to 0";
    if (min[i] > max[i])
      return emitError() << "DimensionBoundsAttr min[" << i << "] : " << min[i]
                         << " must be less than equal to "
                         << "max[" << i << "] : " << max[i];
  }
  return success();
}

/// Check type of `min` matches type of `max` and that all elements of `min` are
/// <= elements of `max`.
LogicalResult executor::ValueBoundsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, DenseElementsAttr min,
    DenseElementsAttr max) {
  if (min.getType() != max.getType())
    return emitError() << "ValueBoundsAttr 'min' and 'max' must have "
                          "matching types; found min type: "
                       << min.getType() << ", max type: " << max.getType();

  if (!min.getType().getElementType().isIntOrIndex())
    return emitError()
           << "ValueBoundsAttr 'min' and 'max' value bounds type must "
              "be either i64 or "
              "an index";

  // Compare underlying values.
  auto minV = min.getValues<int64_t>();
  auto maxV = max.getValues<int64_t>();
  for (unsigned i = 0; i < minV.size(); ++i) {
    if (minV[i] < 0)
      return emitError() << "ValueBoundsAttr min[" << i << "] : " << minV[i]
                         << " must be greater than or equal to 0";
    if (minV[i] > maxV[i])
      return emitError() << "ValueBoundsAttr min[" << i << "] : " << minV[i]
                         << " must be less than equal to "
                         << "max[" << i << "] : " << maxV[i];
  }
  return success();
}

static LogicalResult
checkDimensionBoundsAttr(Type type, DimensionBoundsAttr attr,
                         function_ref<InFlightDiagnostic()> emitError) {
  if (!isa<MemRefType>(type))
    return emitError() << "DimensionBoundsAttr is only for a memref type";
  return success();
}

static LogicalResult
checkValueBoundsAttr(Type type, ValueBoundsAttr attr,
                     function_ref<InFlightDiagnostic()> emitError) {

  if (!isa<MemRefType>(type) && !type.isIntOrIndexOrFloat())
    return emitError()
           << "ValueBoundsAttr is only for memref, index, int, or float type";

  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (!memref.hasStaticShape())
      return emitError()
             << "ValueBoundsAttr must not be present for dynamic memref";

    if (attr.getMin().getElementType() != memref.getElementType())
      return emitError() << "ValueBoundsAttr 'min/max' and corresponding "
                            "memref type must have matching element types; "
                            "found min/max type: "
                         << attr.getMin().getElementType()
                         << ", memref type: " << memref.getElementType();

    if (attr.getMin().getType().getShape() != memref.getShape())
      return emitError()
             << "ValueBoundsAttr 'min/max' and corresponding "
                "memref type must have matching shapes; found min/max shape: "
             << attr.getMin().getType().getShape()
             << ", memref shape: " << memref.getShape();
  }

  return success();
}

static LogicalResult
checkAttributeType(ArrayRef<Type> types, ArrayRef<Attribute> bounds,
                   function_ref<InFlightDiagnostic()> emitError) {
  for (unsigned i = 0; i < types.size(); ++i) {
    if (auto dimensionBounds = dyn_cast<DimensionBoundsAttr>(bounds[i])) {
      if (failed(
              checkDimensionBoundsAttr(types[i], dimensionBounds, emitError)))
        return emitError() << "Unsupported DimensionBoundsAttr";
    }
    if (auto valueBounds = dyn_cast<ValueBoundsAttr>(bounds[i])) {
      if (failed(checkValueBoundsAttr(types[i], valueBounds, emitError)))
        return emitError() << "Unsupported ValueBoundsAttr";
    }
    if (!isa<UnitAttr>(bounds[i]) && !isa<DimensionBoundsAttr>(bounds[i]) &&
        !isa<ValueBoundsAttr>(bounds[i]))
      return emitError() << "Unsupported attribute type";
  }
  return success();
}

LogicalResult executor::FunctionMetadataAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Type> args,
    ArrayRef<Type> results, int64_t nbOutArgs, ArrayRef<Attribute> argBounds,
    ArrayRef<Attribute> resBounds, FlatSymbolRefAttr shapeFuncName,
    CallingConvention callingConvention) {
  if (nbOutArgs < 0)
    return emitError() << "Number of output arguments must be non-negative";

  // Check sizes of arguments and bounds.
  if (args.size() != argBounds.size())
    return emitError() << "Size of args and arg bounds must be same";
  if (results.size() != resBounds.size())
    return emitError() << "Size of results and result bounds must be same";

  // Check for appropriate attribute types for arguments and results.
  if (failed(checkAttributeType(args, argBounds, emitError)))
    return failure();

  return checkAttributeType(results, resBounds, emitError);
}

void printTypesWithBoundsAttrs(AsmPrinter &printer, ArrayRef<Type> types,
                               ArrayRef<Attribute> attrs) {
  assert(types.size() == attrs.size() &&
         "Mismatched sizes of types and bounds attributes");
  printer.getStream() << "[";
  for (int i = 0, e = types.size(); i != e; ++i) {
    if (i != 0)
      printer.getStream() << ", ";
    types[i].print(printer.getStream());
    // Skip UnitAttr which depict a NoneType.
    if (!isa<UnitAttr>(attrs[i])) {
      printer.getStream() << " {";
      attrs[i].print(printer.getStream());
      printer.getStream() << "}";
    }
  }
  printer.getStream() << "]";
}

ParseResult parseTypesWithBoundsAttrs(AsmParser &parser,
                                      SmallVectorImpl<Type> &types,
                                      SmallVectorImpl<Attribute> &attrs) {
  // Parse the opening '['
  if (parser.parseLSquare())
    return failure();

  // Loop until ']' is encountered. `true` mean missing `]`.
  while (parser.parseOptionalRSquare()) {
    Type type;
    Attribute attr;

    // Parse optional type. `true` mean mssing `Type`.
    auto res = parser.parseOptionalType(type);
    if (!res.has_value())
      continue; // No type is present.

    if (res.has_value() && res.value())
      return failure(); // Type is present, but failed to parse.

    // Parsed type successfully.
    types.push_back(type);

    // Check for optional attribute enclosed in '{' '}'
    if (parser.parseOptionalLBrace()) {
      // If no attribute is present, push a null attribute
      attrs.push_back(parser.getBuilder().getUnitAttr());
      // Optional comma between elements. If no comma is found, move on.
      (void)parser.parseOptionalComma();
      continue; // No attribute, move on.
    }

    if (parser.parseAttribute(attr))
      return failure(); // Attribute must exist following a `{`

    if (parser.parseRBrace())
      return failure(); // Attribute must be followed by a closing `}`

    attrs.push_back(attr);

    // Optional comma between elements. If no comma is found, move on.
    (void)parser.parseOptionalComma();
  }

  return success();
}

static Attribute getFuncBounds(
    func::FuncOp func, int64_t idx,
    std::function<Attribute(func::FuncOp, unsigned, StringRef)> getAttrFunc) {
  if (Attribute shapeBounds =
          getAttrFunc(func, idx, ExecutorDialect::getShapeBoundsAttrName())) {
    assert(isa<executor::DimensionBoundsAttr>(shapeBounds) &&
           "expected ExecutorDimensionBoundsAttr");
    return shapeBounds;
  }
  if (Attribute valueBounds =
          getAttrFunc(func, idx, ExecutorDialect::getValueBoundsAttrName())) {
    assert(isa<executor::ValueBoundsAttr>(valueBounds) &&
           "expected executor::ValueBoundsAttr");
    return valueBounds;
  }

  return UnitAttr::get(func.getContext());
}

// Usage for argument attributes
Attribute executor::getFuncArgsBounds(func::FuncOp func, int64_t argIdx) {
  return getFuncBounds(func, argIdx,
                       [](func::FuncOp op, unsigned index, StringRef name) {
                         return op.getArgAttr(index, name);
                       });
}

// Usage for result attributes
Attribute executor::getFuncResultBounds(func::FuncOp func, int64_t argIdx) {
  return getFuncBounds(
      func, argIdx,
      [](func::FuncOp op, unsigned index, StringRef name) -> Attribute {
        return op.getResultAttr(index, name);
      });
}

//===----------------------------------------------------------------------===//
// RuntimeBuiltinInterface
//===----------------------------------------------------------------------===//

LogicalResult
executor::detail::verifyRuntimeBuiltinInterface(Operation *op,
                                                const DataLayout &dataLayout) {
  auto interfaceOp = cast<RuntimeBuiltinInterface>(op);
  if (!llvm::all_of(interfaceOp.getTypesForNameSuffix(), [&](Type t) {
        if (auto ptrType = dyn_cast<PointerType>(t))
          return true;
        return t.isSignlessIntOrIndexOrFloat() || isa<TableType>(t);
      }))
    return op->emitOpError("types for external function name suffix encoding "
                           "should be scalars or tables with elements with "
                           "uniform bitwidth of 32 or 64");
  return success();
}

FailureOr<std::string> executor::detail::getRuntimeBuiltinFunctionNameImpl(
    Operation *op, ArrayRef<Type> suffixTypes, const DataLayout &dataLayout) {
  std::string funcName;
  llvm::raw_string_ostream ss(funcName);
  ss << "_" << llvm::join(llvm::split(op->getName().stripDialect(), "."), "_");

  if (!suffixTypes.empty()) {
    ss << "_";
    bool error{false};
    llvm::interleave(
        suffixTypes, ss,
        [&](Type t) {
          if (t.isIntOrIndexOrFloat()) {
            ss << t;
            return;
          }
          if (auto ptr = dyn_cast<PointerType>(t)) {
            ss << "ptr_"
               << executor::stringifyMemoryType(ptr.getAddressSpace());
            return;
          }

          error = true;
        },
        "_");
    if (error)
      return failure();
  }

  ss.flush();
  return funcName;
}

FailureOr<CallOpInterface> executor::detail::lowerToCallDefaultImpl(
    Operation *op, ArrayRef<Value> operands, ModuleOp moduleOp,
    RewriterBase &rewriter, const TypeConverter &typeConverter,
    const DataLayout &dataLayout) {
  SmallVector<Type> convertedTypes;
  if (failed(typeConverter.convertTypes(op->getResultTypes(), convertedTypes)))
    return failure();
  FailureOr<std::string> funcName =
      cast<RuntimeBuiltinInterface>(op).getRuntimeBuiltinFunctionName(
          dataLayout);
  if (failed(funcName))
    return failure();
  mlir::SymbolRefAttr symbolRef = [&] {
    auto *context = moduleOp.getContext();
    if (moduleOp.lookupSymbol<executor::FuncOp>(*funcName))
      return SymbolRefAttr::get(context, *funcName);

    // Insert the private function declaration into the body of the parent
    // module.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto funcOp = rewriter.create<FuncOp>(
        op->getLoc(), *funcName,
        cast<RuntimeBuiltinInterface>(op).getRuntimeBuiltinFunctionType(
            convertedTypes, TypeRange(operands)));
    funcOp.setSymVisibility("private");
    return SymbolRefAttr::get(context, *funcName);
  }();

  return cast<CallOpInterface>(
      rewriter
          .create<executor::CallOp>(op->getLoc(), convertedTypes,
                                    symbolRef.getLeafReference(), operands)
          .getOperation());
}

//===----------------------------------------------------------------------===//
// Executor Utilities
//===----------------------------------------------------------------------===//

SymbolRefAttr executor::getOrInsertFuncDeclaration(OpBuilder &rewriter,
                                                   Location loc,
                                                   ModuleOp module,
                                                   StringRef name,
                                                   ExecutorFunctionType sig) {
  auto *context = module.getContext();
  if (module.lookupSymbol<FuncOp>(name))
    return SymbolRefAttr::get(context, name);

  // Insert the private function declaration into the body of the parent
  // module.
  auto funcOp = FuncOp::create(loc, name, sig);
  funcOp.setSymVisibility("private");
  SymbolTable(module).insert(funcOp,
                             Block::iterator(module.getBody()->front()));
  return SymbolRefAttr::get(context, name);
}

//===----------------------------------------------------------------------===//
// ExecutorFunctionType
//===----------------------------------------------------------------------===//

ExecutorFunctionType ExecutorFunctionType::clone(TypeRange inputs,
                                                 TypeRange results) {
  return ExecutorFunctionType::get(getContext(), llvm::to_vector(inputs),
                                   llvm::to_vector(results),
                                   getTrailingVarArg());
}

static ParseResult parseVarArgEllipses(AsmParser &p, SmallVector<Type> &params,
                                       UnitAttr &trailingVarArg) {
  Type type;
  trailingVarArg = nullptr;
  do {
    if (succeeded(p.parseOptionalEllipsis())) {
      trailingVarArg = UnitAttr::get(p.getContext());
      return ParseResult::success();
    }
    auto parseTypeResult = p.parseOptionalType(type);
    if (!parseTypeResult.has_value())
      return ParseResult::success();
    if (parseTypeResult.has_value() && failed(parseTypeResult.value()))
      return failure();
    params.push_back(type);
  } while (succeeded(p.parseOptionalComma()));
  return ParseResult::success();
}

static void printVarArgEllipses(AsmPrinter &p, ArrayRef<Type> params,
                                UnitAttr trailingVarArg) {
  llvm::interleaveComma(params, p, [&](Type type) { p.printType(type); });
  if (trailingVarArg) {
    if (!params.empty())
      p << ", ";
    p << "...";
  }
}

static ParseResult parseFuncResults(AsmParser &p, SmallVector<Type> &results) {
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                   [&]() -> ParseResult {
                                     Type t;
                                     if (p.parseType(t))
                                       return failure();
                                     results.push_back(t);
                                     return ParseResult::success();
                                   });
}

static void printFuncResults(AsmPrinter &p, ArrayRef<Type> params) {
  p << "(" << params << ")";
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

static constexpr uint64_t kDefaultPointerSizeBits = 64;
static constexpr uint64_t kDefaultPointerAlignment = 8;
static constexpr uint64_t kBitsInByte = 8;

llvm::TypeSize
PointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                               DataLayoutEntryListRef params) const {
  if (params.empty())
    return llvm::TypeSize::getFixed(kDefaultPointerSizeBits);
  for (auto entry : params) {
    if (!entry.isTypeEntry())
      continue;
    auto ptrType =
        llvm::dyn_cast<executor::PointerType>(entry.getKey().get<Type>());
    if (!ptrType || ptrType.getAddressSpace() != getAddressSpace())
      continue;
    return llvm::TypeSize::getFixed(
        llvm::cast<IntegerAttr>(entry.getValue()).getInt());
  }
  return llvm::TypeSize::getFixed(kDefaultPointerSizeBits);
}

uint64_t PointerType::getABIAlignment(const DataLayout &dataLayhout,
                                      DataLayoutEntryListRef params) const {
  return kDefaultPointerAlignment;
}

uint64_t
PointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  return getABIAlignment(dataLayout, params);
}

//===----------------------------------------------------------------------===//
// ExecutorTableType
//===----------------------------------------------------------------------===//

llvm::TypeSize
TableType::getTypeSizeInBits(const DataLayout &dataLayout,
                             DataLayoutEntryListRef params) const {
  // We adopt a serialization spec identical to that of LLVM structs.
  // The following code was adapted from
  // `third_party/llvm-project/mlir/lib/Dialect/LLVMIR/IR/LLVMTypes.cpp`.
  auto structSize = llvm::TypeSize::getFixed(0);
  uint64_t structAlignment = 1;
  for (Type element : getBody()) {
    // Increment `structSize` by adding padding for the alignemnt requirements
    // followed by the element byte size. The trailing padding is derived
    // from the struct alignment requirement, which is the max of element
    // alignment requirements (e.g. for creating arrays of structs).
    uint64_t elementAlignment = dataLayout.getTypeABIAlignment(element);
    structSize = llvm::alignTo(structSize, elementAlignment);
    structSize += dataLayout.getTypeSize(element);
    structAlignment = std::max(elementAlignment, structAlignment);
  }
  structSize = llvm::alignTo(structSize, structAlignment);
  return structSize * kBitsInByte;
}

uint64_t TableType::getABIAlignment(const DataLayout &dataLayout,
                                    DataLayoutEntryListRef params) const {
  // The alignment requirement of a struct is equal to the strictest alignment
  // requirement of its elements.
  uint64_t structAlignment = 1;
  for (Type iter : getBody())
    structAlignment = std::max<uint64_t>(dataLayout.getTypeABIAlignment(iter),
                                         structAlignment);
  return structAlignment;
}

uint64_t TableType::getPreferredAlignment(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  return getABIAlignment(dataLayout, params);
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  // If the assertion is statically known to pass, then it can be erased.
  if (matchPattern(op.getArg(), m_One())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult executor::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

LogicalResult executor::ConstantOp::verify() {
  auto type = getType();
  // Integer values must be signless.
  if (llvm::isa<IntegerType>(type) &&
      !llvm::cast<IntegerType>(type).isSignless())
    return emitOpError("integer return type must be signless");
  // Only int/float or ElementsAttr are allowed.
  if (!isa<IntegerAttr, FloatAttr, ElementsAttr>(getValue()))
    return emitOpError(
        "value must be an integer, float, or elements attribute");
  return success();
}

void executor::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  llvm::SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  if (auto intCst = llvm::dyn_cast<IntegerAttr>(getValue())) {
    auto intType = llvm::dyn_cast<IntegerType>(type);
    specialName << 'c' << intCst.getValue();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
    return;
  }
  specialName << "cst_" << type;
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  if (hasInitRegion()) {
    if (!getBodyRegion().hasOneBlock()) {
      return emitOpError()
             << "initialization region expected to have one block";
    }
    auto term = dyn_cast<ReturnOp>(getInitBody()->getTerminator());
    if (!term)
      return emitOpError()
             << "expected region to be terminated by executor.return";

    if (term->getNumOperands() != 1 ||
        term->getOperand(0).getType() != getType())
      return emitOpError()
             << "expected initialization region to return one value of type "
             << getType();
  }

  if (hasInitRegion() && getInitialValueAttr())
    return emitOpError() << "expected either initialization region or "
                            "initial_value but not both";

  // Initial value must match type unless this returns a pointer. Then we
  // allow the initializer to be dense elements.
  if (getInitialValueAttr() && getInitialValue()->getType() != getType() &&
      !llvm::isa<PointerType>(getType()))
    return emitOpError() << "expected initial_value to have type " << getType();

  return success();
}

void GlobalOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     DenseI8ResourceElementsAttr attr, bool constant) {
  return build(builder, state, name, attr.getShapedType(),
               cast<TypedAttr>(attr), constant);
}

void GlobalOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     Type type,
                     std::function<void(OpBuilder &, Location)> initBuilder,
                     bool constant) {
  state.addAttribute(GlobalOp::getTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.addAttribute(GlobalOp::getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  if (constant)
    state.addAttribute(GlobalOp::getConstantAttrName(state.name),
                       builder.getUnitAttr());
  Region *region = state.addRegion();
  Block &body = region->emplaceBlock();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&body);
  initBuilder(builder, state.location);
}

//===----------------------------------------------------------------------===//
// ConstantResourceOp
//===----------------------------------------------------------------------===//

ConstantResourceOp ConstantResourceOp::create(Location loc, StringRef name,
                                              ElementsAttr value) {
  OpBuilder b(loc.getContext());
  return b.create<ConstantResourceOp>(loc, name, value);
}

//===----------------------------------------------------------------------===//
// ConstantResourceLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantResourceLoadOp::verifySymbolUses(
    ::mlir::SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto globalOp = dyn_cast_or_null<ConstantResourceOp>(
      symbolTable.lookupSymbolIn(module, getNameAttr()));
  if (!globalOp)
    return emitOpError() << "constant resource op with name " << getNameAttr()
                         << " not found";

  return success();
}

//===----------------------------------------------------------------------===//
// AlignToOp
//===----------------------------------------------------------------------===//

template <typename IntType>
static IntType alignToImpl(IntType arg, uint32_t alignment) {
  typename std::make_unsigned<IntType>::type bump =
      static_cast<typename std::make_unsigned<IntType>::type>(arg) + alignment -
      1;
  return static_cast<IntType>(bump - bump % alignment);
}

OpFoldResult AlignToOp::fold(FoldAdaptor adaptor) {
  // Align to 1 byte is no-op:
  if (getAlignment() == 1)
    return getArg();

  if (IntegerAttr arg =
          llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getArg()))
    return IntegerAttr::get(arg.getType(),
                            alignToImpl<int64_t>(arg.getInt(), getAlignment()));

  // Replace `y` by `x` in the case of redundant alignment operations.
  // ---
  // x = alignTo(src, c*N)
  // y = alignTo(x, N)
  // ---
  if (auto producer = getArg().getDefiningOp<AlignToOp>()) {
    if (producer.getAlignment() >= getAlignment() &&
        producer.getAlignment() % getAlignment() == 0)
      return producer.getResult();
  }

  // Replace `y` by `x` in the casse where `x` is provably a multiple
  // of `alignment`.
  // ---
  // x = muli(src, c*N)
  // y = alignTo(x, N)
  // ---
  if (auto producer = getArg().getDefiningOp<MulIOp>()) {
    IntegerAttr intAttr{};
    if (!matchPattern(producer.getRhs(), m_Constant(&intAttr)) ||
        !intAttr.getValue().isAligned(llvm::Align(getAlignment())))
      return {};
    return producer.getResult();
  }

  // We often see patterns like:
  // ---
  // x = alignTo(src, N)
  // y = add(x, const)
  // z = alignTo(y, O)
  // ---
  // In such cases, we can return `x` if `N` is a multiple of `O` and
  // `const` is a multiple of `O`.
  // TODO: remove this when we have better alignment analysis.
  if (auto addOp = getArg().getDefiningOp<AddIOp>()) {
    APInt offsetConstant;
    if (!matchPattern(addOp.getRhs(), m_ConstantInt(&offsetConstant)))
      return {};
    if (!offsetConstant.isAligned(llvm::Align(getAlignment())))
      return {};
    if (auto alignProducerOp = addOp.getLhs().getDefiningOp<AlignToOp>()) {
      if (!alignProducerOp ||
          !alignProducerOp.getAlignmentAttr().getValue().isAligned(
              llvm::Align(getAlignment())))
        return {};
      return addOp.getResult();
    }
    if (auto mulProducerOp = addOp.getLhs().getDefiningOp<MulIOp>()) {
      APInt mulConstant;
      if (!mulProducerOp ||
          !matchPattern(mulProducerOp.getRhs(), m_ConstantInt(&mulConstant)) ||
          !mulConstant.isAligned(llvm::Align(getAlignment())))
        return {};
      return addOp.getResult();
    }
  }

  return {};
}

LogicalResult AlignToOp::verify() {
  if (!llvm::isPowerOf2_64(getAlignment()))
    return emitOpError() << "alignment must be a power of two, but got "
                         << getAlignment();
  return success();
}

FailureOr<CallOpInterface>
AlignToOp::lowerToCall(ArrayRef<Value> operands, RewriterBase &rewriter,
                       ModuleOp module, const TypeConverter &typeConverter,
                       const DataLayout &dataLayout) {
  Value alignment = rewriter.create<executor::ConstantOp>(
      getLoc(), rewriter.getI32IntegerAttr(getAlignment()));
  std::optional<Type> convertedType = typeConverter.convertType(getType());
  if (!convertedType)
    return failure();

  SmallVector<Value> callArgs(operands);
  callArgs.push_back(alignment);
  std::string funcName = llvm::formatv("_alignto_{0}", *convertedType);

  ExecutorFunctionType funcType = ExecutorFunctionType::get(
      rewriter.getContext(), llvm::to_vector(TypeRange(callArgs)),
      *convertedType, UnitAttr{});

  mlir::SymbolRefAttr symbolRef = getOrInsertFuncDeclaration(
      rewriter, getLoc(), module, funcName, funcType);
  return cast<CallOpInterface>(
      rewriter
          .create<executor::CallOp>(getLoc(), *convertedType,
                                    symbolRef.getLeafReference(), callArgs)
          .getOperation());
}

//===----------------------------------------------------------------------===//
// GetOffsetOp
//===----------------------------------------------------------------------===//

static ParseResult parseExecutorMixedIndices(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
    DenseI64ArrayAttr &rawConstantIndices) {
  SmallVector<int64_t> constantIndices;

  auto idxParser = [&]() -> ParseResult {
    int64_t constantIndex;
    OptionalParseResult parsedInteger =
        parser.parseOptionalInteger(constantIndex);
    if (parsedInteger.has_value()) {
      if (failed(parsedInteger.value()))
        return failure();
      constantIndices.push_back(constantIndex);
      return success();
    }

    constantIndices.push_back(ShapedType::kDynamic);
    return parser.parseOperand(indices.emplace_back());
  };
  if (parser.parseCommaSeparatedList(idxParser))
    return failure();

  rawConstantIndices =
      DenseI64ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

static void printExecutorMixedIndices(OpAsmPrinter &printer,
                                      executor::GetOffsetOp GetOffsetOp,
                                      OperandRange indices,
                                      DenseI64ArrayAttr staticIndices) {
  assert(static_cast<unsigned>(
             llvm::count(staticIndices.asArrayRef(), ShapedType::kDynamic)) ==
         indices.size());
  unsigned dynamicIdx = 0;
  llvm::interleaveComma(staticIndices.asArrayRef(), printer, [&](int64_t cst) {
    if (!ShapedType::isDynamic(cst))
      printer << cst;
    else
      printer.printOperand(indices[dynamicIdx++]);
  });
}

SmallVector<OpFoldResult> GetOffsetOp::getIndices() {
  OpBuilder builder(getContext());
  return mlir::getMixedValues(getStaticIndices(), getDynamicIndices(), builder);
}

/// For the given `indices`, check if they comply with `baseGEPType`,
/// especially check against LLVMStructTypes nested within.
static LogicalResult
verifyStructIndices(Type baseGEPType, unsigned indexPos,
                    ArrayRef<OpFoldResult> indices,
                    function_ref<InFlightDiagnostic()> emitOpError) {
  if (indexPos >= indices.size())
    // Stop searching
    return success();

  return TypeSwitch<Type, LogicalResult>(baseGEPType)
      .Case<TableType>([&](TableType structType) -> LogicalResult {
        auto indexAttr = dyn_cast<Attribute>(indices[indexPos]);
        if (!indexAttr)
          return emitOpError() << "expected index " << indexPos
                               << " indexing a struct to be constant";

        int64_t gepIndex = cast<IntegerAttr>(indexAttr).getInt();
        ArrayRef<Type> elementTypes = structType.getBody();
        if (gepIndex < 0 ||
            static_cast<size_t>(gepIndex) >= elementTypes.size())
          return emitOpError() << "index " << indexPos
                               << " indexing a struct is out of bounds";

        // Instead of recursively going into every children types, we only
        // dive into the one indexed by gepIndex.
        return verifyStructIndices(elementTypes[gepIndex], indexPos + 1,
                                   indices, emitOpError);
      })
      .Case<VectorType>([&](auto containerType) -> LogicalResult {
        return verifyStructIndices(containerType.getElementType(), indexPos + 1,
                                   indices, emitOpError);
      })
      .Default([&](auto otherType) -> LogicalResult {
        return emitOpError()
               << "type " << otherType << " cannot be indexed (index #"
               << indexPos << ")";
      });
}

LogicalResult GetOffsetOp::verify() {
  assert(static_cast<unsigned>(
             llvm::count(getStaticIndices(), ShapedType::kDynamic)) ==
         getDynamicIndices().size());
  return verifyStructIndices(getElemType(), 1, getIndices(),
                             [&]() { return emitOpError(); });
}

OpFoldResult GetOffsetOp::fold(FoldAdaptor adaptor) {
  SmallVector<OpFoldResult> indices = getIndices();

  // executor.gep [0] -> 0
  if (indices.size() == 1)
    if (auto integer = llvm::dyn_cast_or_null<IntegerAttr>(
            dyn_cast<Attribute>(indices[0])))
      if (integer.getValue().isZero())
        return IntegerAttr::get(getType(), 0);

  // Canonicalize any dynamic indices of constant value to constant indices.
  bool changed = false;
  SmallVector<int64_t> newStaticIndices;
  SmallVector<Value> newDynamicIndices;
  for (auto [idx, val] : llvm::enumerate(indices)) {
    auto constVal =
        dyn_cast_or_null<IntegerAttr>(llvm::dyn_cast<Attribute>(val));
    auto dynVal = llvm::dyn_cast<Value>(val);
    // Constant indices can only be int32_t, so if integer does not fit we
    // are forced to keep it dynamic, despite being a constant.

    IntegerAttr folded{};
    if (dynVal && matchPattern(dynVal, m_Constant(&folded)) &&
        folded.getValue().isSignedIntN(64)) {
      newStaticIndices.emplace_back(folded.getInt());
      changed = true;
      continue;
    }
    if (dynVal) {
      newDynamicIndices.push_back(dynVal);
      newStaticIndices.push_back(ShapedType::kDynamic);
      continue;
    }
    assert(constVal && "expected valid static index");
    newStaticIndices.emplace_back(constVal.getInt());
  }

  if (!changed)
    return {};
  setStaticIndicesAttr(DenseI64ArrayAttr::get(getContext(), newStaticIndices));
  getDynamicIndicesMutable().assign(newDynamicIndices);
  return getResult();
}

void GetOffsetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        Type resultType, Type elementType,
                        ArrayRef<OpFoldResult> indices) {
  SmallVector<Value> dynamicIndices;
  SmallVector<int64_t> staticIndices;
  staticIndices.reserve(indices.size());
  for (OpFoldResult ofr : indices) {
    if (IntegerAttr attr =
            dyn_cast_or_null<IntegerAttr>(llvm::dyn_cast<Attribute>(ofr))) {
      staticIndices.push_back(attr.getInt());
      continue;
    }
    Value val = ofr.get<Value>();
    staticIndices.push_back(ShapedType::kDynamic);
    dynamicIndices.push_back(val);
  }

  build(odsBuilder, odsState, resultType, dynamicIndices, staticIndices,
        elementType);
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

LogicalResult AllocaOp::verify() { return success(); }

SmallVector<MemorySlot> AllocaOp::getPromotableSlots() {
  return {MemorySlot{getResult(), getElementType()}};
}

Value AllocaOp::getDefaultValue(const MemorySlot &slot, OpBuilder &rewriter) {
  return rewriter.create<UndefinedOp>(getLoc(), slot.elemType);
}

void AllocaOp::handleBlockArgument(const MemorySlot &slot, BlockArgument arg,
                                   OpBuilder &rewriter) {}

std::optional<PromotableAllocationOpInterface>
AllocaOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                  OpBuilder &rewriter) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

FailureOr<CallOpInterface>
LoadOp::lowerToCall(ArrayRef<Value> operands, RewriterBase &rewriter,
                    ModuleOp moduleOp, const TypeConverter &typeConverter,
                    const DataLayout &dataLayout) {
  if (isa<TableType>(getType()))
    return failure();
  return detail::lowerToCallDefaultImpl(getOperation(), operands, moduleOp,
                                        rewriter, typeConverter, dataLayout);
}

bool LoadOp::storesTo(const MemorySlot &slot) { return false; }
bool LoadOp::loadsFrom(const MemorySlot &slot) { return slot.ptr == getPtr(); }
Value LoadOp::getStored(const MemorySlot &slot, OpBuilder &, Value reachingDef,
                        const DataLayout &dataLayout) {
  llvm_unreachable("LoadOp::getStored should never be called");
}
bool LoadOp::canUsesBeRemoved(const MemorySlot &slot,
                              const SmallPtrSetImpl<OpOperand *> &blockingUses,
                              SmallVectorImpl<OpOperand *> &newBlockingUses,
                              const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value use = (*blockingUses.begin())->get();
  return use == slot.ptr && getPtr() == slot.ptr &&
         matchPattern(getOffset(), m_Zero()) && getType() == slot.elemType;
}

DeletionKind
LoadOp::removeBlockingUses(const MemorySlot &slot,
                           const SmallPtrSetImpl<OpOperand *> &blockingUses,
                           OpBuilder &rewriter, Value reachingDefinition,
                           const DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

bool StoreOp::storesTo(const MemorySlot &slot) { return slot.ptr == getPtr(); }
bool StoreOp::loadsFrom(const MemorySlot &slot) { return false; }
Value StoreOp::getStored(const MemorySlot &slot, OpBuilder &, Value reachingDef,
                         const DataLayout &dataLayout) {
  return getValue();
}
bool StoreOp::canUsesBeRemoved(const MemorySlot &slot,
                               const SmallPtrSetImpl<OpOperand *> &blockingUses,
                               SmallVectorImpl<OpOperand *> &newBlockingUses,
                               const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value use = (*blockingUses.begin())->get();
  return use == slot.ptr && getPtr() == slot.ptr &&
         matchPattern(getOffset(), m_Zero()) &&
         getValue().getType() == slot.elemType;
}
DeletionKind
StoreOp::removeBlockingUses(const MemorySlot &slot,
                            const SmallPtrSetImpl<OpOperand *> &blockingUses,
                            OpBuilder &rewriter, Value reachingDefinition,
                            const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

GlobalOp GetGlobalOp::getGlobal(SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto globalOp = dyn_cast_or_null<GlobalOp>(
      symbolTable.lookupSymbolIn(module, getNameAttr()));
  return globalOp;
}

LogicalResult
GetGlobalOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  GlobalOp globalOp = getGlobal(symbolTable);
  if (!globalOp)
    return emitOpError() << "global op with name " << getNameAttr()
                         << " not found";

  if (getType() != globalOp.getType())
    return emitOpError() << "global has type " << globalOp.getType()
                         << " vs type " << getType();
  return success();
}

//===----------------------------------------------------------------------===//
// SetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
SetGlobalOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto globalOp = dyn_cast_or_null<GlobalOp>(
      symbolTable.lookupSymbolIn(module, getNameAttr()));
  if (!globalOp)
    return emitOpError() << "global op with name " << getNameAttr()
                         << " not found";

  if (globalOp.getConstant()) {
    auto moduleParent = (*this)->getParentOfType<ModuleOp>();
    auto parentFunc = (*this)->getParentOfType<func::FuncOp>();
    auto initAttr = moduleParent->getAttrOfType<FlatSymbolRefAttr>(
        getExecutorGlobalInitializerFuncNameAttr());
    if (!initAttr || initAttr != FlatSymbolRefAttr::get(parentFunc))
      return emitOpError() << "trying to set a global marked as constant";
  }

  if (getValue().getType() != globalOp.getType())
    return emitOpError() << "global has type " << globalOp.getType()
                         << " vs value of type " << getValue().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag varFlag,
                          std::string &) {
    return ExecutorFunctionType::get(
        builder.getContext(), argTypes, results,
        varFlag.isVariadic() ? UnitAttr::get(builder.getContext()) : nullptr);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/true,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  ExecutorFunctionType funcType = getFunctionType();
  UnitAttr varArg = funcType.getTrailingVarArg();
  function_interface_impl::printFunctionOp(
      p, *this,
      /*isVariadic=*/varArg ? true : false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult FuncOp::verify() { return success(); }

LogicalResult FuncOp::verifyRegions() { return success(); }

FuncOp FuncOp::create(Location location, StringRef name,
                      ExecutorFunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   ExecutorFunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs);
  state.addRegion();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  if (std::optional<ArrayAttr> argsAttr = getImmediateArgs()) {
    for (Attribute arg : *argsAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(arg);
      if (!intAttr)
        return emitOpError(
            "immediate constant attribute arg must be an IntegerType");
    }
  }
  return success();
}

LogicalResult
CallOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  auto callee = dyn_cast_or_null<executor::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));
  if (!callee)
    return emitOpError() << "could not find executor.func symbol with name "
                         << getCallee();

  ExecutorFunctionType funcType = callee.getFunctionType();

  if ((!funcType.getTrailingVarArg() &&
       getArgs().size() != funcType.getArgs().size()) ||
      (funcType.getTrailingVarArg() &&
       getArgs().size() < funcType.getArgs().size()) ||
      TypeRange(getArgs()).take_front(funcType.getArgs().size()) !=
          funcType.getArgs() ||
      getResults().size() != funcType.getResults().size() ||
      getResultTypes() != funcType.getResults())
    return emitOpError()
           << "call signature is not compatible with the callee signature "
           << funcType;

  return success();
}

//===----------------------------------------------------------------------===//
// Binary Integer Arithmetic Folders
//===----------------------------------------------------------------------===//

OpFoldResult AddIOp::fold(FoldAdaptor adaptor) {
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  if (matchPattern(getLhs(), m_Zero()))
    return getRhs();
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs + rhs; });
}
OpFoldResult SubIOp::fold(FoldAdaptor adaptor) {
  // x - 0 = x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs - rhs; });
}
OpFoldResult SDivIOp::fold(FoldAdaptor adaptor) {
  // x / 0 -> dont fold
  if (matchPattern(getRhs(), m_Zero()))
    return nullptr;
  if (matchPattern(getRhs(), m_One()))
    return getLhs();
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs.sdiv(rhs); });
}
OpFoldResult SRemIOp::fold(FoldAdaptor adaptor) {
  auto zero = Builder(getContext()).getZeroAttr(getType());
  // x % 0 -> dont fold
  if (matchPattern(getRhs(), m_Zero()))
    return nullptr;
  // x % 1  = 0
  if (matchPattern(getRhs(), m_One()))
    return zero;
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs.srem(rhs); });
}
OpFoldResult SFloorDivIOp::fold(FoldAdaptor adaptor) {
  // x / 0 -> dont fold
  if (matchPattern(getRhs(), m_Zero()))
    return nullptr;
  if (matchPattern(getRhs(), m_One()))
    return getLhs();
  // TODO: const folder
  return nullptr;
}
OpFoldResult MulIOp::fold(FoldAdaptor adaptor) {
  // x * 1 = x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();
  // 1 * x = x
  if (matchPattern(getLhs(), m_One()))
    return getRhs();
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs * rhs; });
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

auto isBitcastSupported = [](Type inputType, Type resultType) -> bool {
  if (inputType.isInteger(16) || inputType.isF16())
    return resultType.isF16() || resultType.isInteger(16);
  if (inputType.isInteger(32) || inputType.isF32())
    return resultType.isF32() || resultType.isInteger(32);
  if (inputType.isInteger(64) || inputType.isF64())
    return resultType.isF64() || resultType.isInteger(64);
  return false;
};

LogicalResult BitcastOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  // Supported casts
  // i16 | F16 <-> i16 | F16
  // i32 | F32 <-> i32 | F32
  // i64 | F64 <-> i64 | F64
  if (!isBitcastSupported(inputType, resultType))
    return emitOpError() << "Bitcast between input type " << inputType
                         << "and result type " << resultType
                         << "is not supported";
  return success();
}

//===----------------------------------------------------------------------===//
// ZExtOp
//===----------------------------------------------------------------------===//

LogicalResult ZExtOp::verify() {
  if (getType().getIntOrFloatBitWidth() <=
      getOperand().getType().getIntOrFloatBitWidth())
    return emitOpError(
        "result type should be have a larger bitwidth than input type");
  return success();
}

//===----------------------------------------------------------------------===//
// SIExtOp
//===----------------------------------------------------------------------===//

LogicalResult SIExtOp::verify() {
  if (getType().getIntOrFloatBitWidth() <=
      getOperand().getType().getIntOrFloatBitWidth())
    return emitOpError(
        "result type should be have a larger bitwidth than input type");
  return success();
}

//===----------------------------------------------------------------------===//
// CreateTableOp
//===----------------------------------------------------------------------===//

void CreateTableOp::build(OpBuilder &b, OperationState &state, Type result,
                          Value allocatedPtr, Value alignedPtr, Value offset,
                          ValueRange sizes, ValueRange strides) {
  state.addTypes(result);
  state.addOperands({allocatedPtr, alignedPtr, offset});
  state.addOperands(sizes);
  state.addOperands(strides);
}

void CreateTableOp::build(OpBuilder &b, OperationState &state, Type result,
                          Value allocatedPtr, Value alignedPtr,
                          OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
                          ArrayRef<OpFoldResult> strides) {
  auto getVal = [&](OpFoldResult ofr) {
    if (ofr.is<Value>())
      return ofr.get<Value>();
    Attribute attr = ofr.get<Attribute>();
    assert(llvm::isa<IntegerAttr>(attr));
    return b.create<executor::ConstantOp>(state.location, cast<TypedAttr>(attr))
        .getResult();
  };

  SmallVector<Value> sizeVals = llvm::to_vector(llvm::map_range(sizes, getVal));
  SmallVector<Value> strideVals =
      llvm::to_vector(llvm::map_range(strides, getVal));

  state.addTypes(result);
  state.addOperands({allocatedPtr, alignedPtr, getVal(offset)});
  state.addOperands(sizeVals);
  state.addOperands(strideVals);
}

LogicalResult CreateTableOp::verify() {
  auto resultType = llvm::cast<TableType>(getType());
  if (!getInit().empty()) {
    auto init = getInit();
    if (init.size() > resultType.getBody().size())
      return emitOpError(
          "number of initial values exceeds body size of result type");
    for (auto [idx, val] : llvm::enumerate(init)) {
      if (val.getType() != resultType.getBody()[idx])
        return emitOpError() << "initial value at position " << idx
                             << " has type " << val.getType()
                             << " but expected " << resultType.getBody()[idx];
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractTableValueOp
//===----------------------------------------------------------------------===//

OpFoldResult ExtractTableValueOp::fold(FoldAdaptor adaptor) {
  unsigned pos = getIndex();
  if (auto createOp = getTable().getDefiningOp<CreateTableOp>()) {
    if (pos >= createOp.getInit().size())
      return nullptr;
    return createOp.getInit()[pos];
  }

  OpFoldResult result{};
  auto insertOp = getTable().getDefiningOp<InsertTableValueOp>();
  while (insertOp) {
    if (pos == insertOp.getIndex())
      return insertOp.getValue();
    getTableMutable().assign(insertOp.getTable());
    result = getResult();
    insertOp = insertOp.getTable().getDefiningOp<InsertTableValueOp>();
  }
  return result;
}

LogicalResult ExtractTableValueOp::verify() {
  auto t = llvm::cast<TableType>(getTable().getType());
  unsigned index = getIndex();
  if (index >= t.getBody().size())
    return emitOpError() << getIndexAttrName() << " " << index
                         << " exceeds body size (" << t.getBody().size() << ")";
  return success();
}

void ExtractTableValueOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add(+[](ExtractTableValueOp op,
                  PatternRewriter &rewriter) -> LogicalResult {
    // Try to materialize constant locally when creating constant global
    // tables. This is useful when e.g. global defines a memref and we are
    // retrieving the a static shape parameter.
    unsigned pos = op.getIndex();
    auto getGlobalOp = op.getTable().getDefiningOp<GetGlobalOp>();
    if (!getGlobalOp)
      return rewriter.notifyMatchFailure(op, "not defined by a global");
    SymbolTableCollection collection;
    GlobalOp global = getGlobalOp.getGlobal(collection);
    if (!global.getConstant() || !global.hasInitRegion())
      return rewriter.notifyMatchFailure(
          op, "no init region or global is not constant");
    Value returnedTable = global.getInitBody()->getTerminator()->getOperand(0);
    auto createOp = returnedTable.getDefiningOp<CreateTableOp>();
    if (!createOp)
      return failure();
    assert(pos < createOp.getInit().size() && "expected valid extract index");
    Value inserted = createOp.getInit()[pos];
    Attribute constVal;
    if (!matchPattern(inserted, m_Constant(&constVal)))
      return rewriter.notifyMatchFailure(op,
                                         "inserted value is not a constant");
    rewriter.replaceOpWithNewOp<executor::ConstantOp>(
        op, cast<TypedAttr>(constVal));
    return success();
  });
}

LogicalResult ExtractTableValueOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ExtractTableValueOp::Adaptor adaptor(operands, attributes, properties,
                                       regions);
  auto t = llvm::cast<TableType>(adaptor.getTable().getType());
  unsigned index = adaptor.getIndex();
  if (index >= t.getBody().size())
    return emitOptionalError(location, "index is out of bounds");
  inferredReturnTypes.push_back(t.getBody()[index]);
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicExtractTableValueOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicExtractTableValueOp::verify() {
  auto t = llvm::cast<TableType>(getTable().getType());
  for (unsigned i = 0; i < t.getBody().size(); ++i) {
    if (t.getBody()[i] != getResult().getType())
      return emitOpError() << "inconsistent type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InsertTableValueOp
//===----------------------------------------------------------------------===//

LogicalResult InsertTableValueOp::verify() {
  Type insertType = getValue().getType();
  TableType tableType = getTable().getType();
  unsigned index = getIndex();
  if (index >= tableType.getBody().size())
    return emitOpError() << getIndexAttrName() << " " << index
                         << " exceeds body size (" << tableType.getBody().size()
                         << ")";
  if (tableType.getBody()[index] != insertType)
    return emitOpError() << "insert item type " << insertType
                         << " does not match expected type "
                         << tableType.getBody()[index];
  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult AllocateOp::verify() {
  APInt alignConst;
  if (matchPattern(getAlignment(), m_ConstantInt(&alignConst))) {
    if (!llvm::isPowerOf2_64(alignConst.getSExtValue()))
      return emitOpError() << "alignment must be a power of 2, but got "
                           << alignConst.getSExtValue();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

StringRef executor::getExecutorGlobalInitializerFuncNameAttr() {
  return "executor.global_init_func";
}

FailureOr<ArrayRef<int64_t>> executor::getModuleProcessGridShape(ModuleOp op) {
  DenseI64ArrayAttr attr = op->getAttrOfType<DenseI64ArrayAttr>(
      ExecutorDialect::kProcessGridShapeAttrName);
  if (!attr)
    return failure();
  return attr.asArrayRef();
}

LogicalResult executor::setModuleProcessGridShape(ModuleOp op,
                                                  ArrayRef<int64_t> shape) {
  FailureOr<ArrayRef<int64_t>> existingShape = getModuleProcessGridShape(op);
  if (failed(existingShape)) {
    op->setAttr(ExecutorDialect::kProcessGridShapeAttrName,
                DenseI64ArrayAttr::get(op->getContext(), shape));
    return success();
  }
  return success(shape == *existingShape);
}

//===----------------------------------------------------------------------===//
// ExecutorDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with func
/// operations.
struct ExecutorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// `tensorrt.enqueue` cannot be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
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

void ExecutorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-executor/Executor/IR/ExecutorOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-executor/Executor/IR/ExecutorOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-executor/Executor/IR/ExecutorAttributes.cpp.inc"
      >();

  addInterfaces<ExecutorInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Dialect hooks
//===----------------------------------------------------------------------===//
Operation *ExecutorDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto typedAttr = dyn_cast<TypedAttr>(value);
  if (!typedAttr)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, typedAttr);
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd attributes definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definition.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum definition.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-executor/Executor/IR/ExecutorOps.cpp.inc"
