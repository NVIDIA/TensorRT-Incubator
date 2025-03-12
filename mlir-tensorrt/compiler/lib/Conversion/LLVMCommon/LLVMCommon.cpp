//===- LLVMCommon.cpp -----------------------------------------------------===//
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
//===----------------------------------------------------------------------===//
///
/// Implementation of utilities for conversion LLVM MLIR dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/LLVMCommon/LLVMCommon.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <string>

using namespace mlir;

LLVM::CallOp LLVMOpaqueCallBuilder::create(Location loc, OpBuilder &builder,
                                           ArrayRef<Value> arguments,
                                           SymbolTable *symbolTable) const {
  auto function = [&] {
    OpBuilder::InsertionGuard g(builder);
    LLVM::LLVMFuncOp lookupResult{};

    if (symbolTable)
      lookupResult = symbolTable->lookup<LLVM::LLVMFuncOp>(functionName);
    else
      lookupResult = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
          builder.getBlock()->getParentOp(),
          builder.getStringAttr(functionName));
    if (lookupResult)
      return lookupResult;
    Operation *module =
        SymbolTable::getNearestSymbolTable(builder.getBlock()->getParentOp());
    builder.setInsertionPointToEnd(&module->getRegion(0).front());
    auto func =
        builder.create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
    // Update the symbol table if provided.
    if (symbolTable) {
      symbolTable->insert(func);
      return func;
    }
    return func;
  }();
  return builder.create<LLVM::CallOp>(loc, function, arguments);
}

static std::string ensureSymbolNameIsUnique(Operation *symbolTableOp,
                                            SymbolTable *cachedTable,
                                            StringRef symbolName) {
  static int counter = 0;
  std::string uniqueName = std::string(symbolName);
  auto uniqueChecker = [&]() {
    if (cachedTable)
      return cachedTable->lookup(uniqueName);
    else
      return SymbolTable::lookupSymbolIn(symbolTableOp, uniqueName);
  };
  while (uniqueChecker())
    uniqueName = std::string(symbolName) + "_" + std::to_string(counter++);
  return uniqueName;
}

/// Performs a symbol table lookup for the given name then invokes different
/// callbacks based on whether a match was found. This helper assists with using
/// and updating `cachedTable` if it is provided, otherwise a SymbolTable is
/// constructed on the fly.
///
/// The `matchVerifier` (optional) allows caller to specifiy finer-grained
/// criteria for a match besides just symbol name.
template <typename T, typename FoundCallable, typename NotFoundCallable>
auto lookupOrUpdateSymbolTable(
    OpBuilder &rewriter, SymbolTable *cachedTable, StringRef name,
    FoundCallable foundHandler, NotFoundCallable notFoundHandler,
    llvm::function_ref<bool(T)> matchVerifier = nullptr) {
  Operation *symbolTableOp = cachedTable
                                 ? cachedTable->getOp()
                                 : SymbolTable::getNearestSymbolTable(
                                       rewriter.getBlock()->getParentOp());

  if (cachedTable) {
    T result = cachedTable->lookup<T>(name);
    if (result)
      return foundHandler(result);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());
    std::string uniqueName =
        ensureSymbolNameIsUnique(symbolTableOp, cachedTable, name);
    LLVM::GlobalOp newGlobalOp = notFoundHandler(rewriter, uniqueName);
    cachedTable->insert(newGlobalOp);
    return newGlobalOp;
  }

  if (auto result = llvm::dyn_cast_if_present<T>(
          SymbolTable::lookupSymbolIn(symbolTableOp, name)))
    return foundHandler(result);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());
  std::string uniqueName =
      ensureSymbolNameIsUnique(symbolTableOp, cachedTable, name);
  return notFoundHandler(rewriter, uniqueName);
}

LLVM::GlobalOp mlir::lookupOrInsertGlobal(
    OpBuilder &rewriter, Location loc, StringRef symbolName, bool constant,
    Type type, LLVM::Linkage linkage, Attribute initialValue,
    SymbolTable *symbolTable,
    llvm::function_ref<Value(OpBuilder &rewriter, Location loc)> initBuilder) {
  auto isMatch = [&](LLVM::GlobalOp candidate) {
    return candidate.getGlobalType() == type &&
           candidate.getConstant() == constant &&
           candidate.getLinkage() == linkage &&
           candidate.getValue() == initialValue;
  };
  return lookupOrUpdateSymbolTable<LLVM::GlobalOp>(
      rewriter, symbolTable, symbolName, [&](LLVM::GlobalOp op) { return op; },
      [&](OpBuilder &rewriter, StringRef uniquedName) {
        LLVM::GlobalOp op = rewriter.create<LLVM::GlobalOp>(
            loc, type, constant, linkage, uniquedName, initialValue);
        if (initBuilder) {
          Block *initBlock = &op.getInitializerRegion().emplaceBlock();
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(initBlock);
          Value result = initBuilder(rewriter, loc);
          rewriter.create<LLVM::ReturnOp>(loc, result);
        }
        return op;
      },
      isMatch);
}

LLVM::GlobalOp mlir::insertLLVMGlobal(
    OpBuilder &rewriter, Location loc, StringRef symbolName, bool constant,
    Type type, LLVM::Linkage linkage, Attribute initialValue,
    SymbolTable *symbolTable,
    llvm::function_ref<Value(OpBuilder &rewriter, Location loc)> initBuilder) {
  Operation *symbolTableOp = symbolTable
                                 ? symbolTable->getOp()
                                 : SymbolTable::getNearestSymbolTable(
                                       rewriter.getBlock()->getParentOp());
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());

  // Ensure the symbol name is unique. We do this prior to symbol table
  // insertion because in the middle of a conversion pass the "all symbols are
  // unique" invariant may be temporarily violated until the conversion
  // completes and old symbols are erased. In theory this can be costly and
  // incur repeated scans until upstream provides better infrastructure to
  // handle symbol replacements. In practice, however, this isn't really
  // noticable since the number of top-level ops is relatively small.
  std::string name =
      ensureSymbolNameIsUnique(symbolTableOp, symbolTable, symbolName);
  LLVM::GlobalOp op = rewriter.create<LLVM::GlobalOp>(
      loc, type, constant, linkage, name, initialValue);
  if (initBuilder) {
    Block *initBlock = &op.getInitializerRegion().emplaceBlock();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(initBlock);
    Value result = initBuilder(rewriter, loc);
    rewriter.create<LLVM::ReturnOp>(loc, result);
  }

  // Update the symbol table if provided.
  if (symbolTable)
    symbolTable->insert(op);
  return op;
}

Value mlir::insertLLVMStringLiteral(OpBuilder &rewriter, Location loc,
                                    StringRef string, StringRef symbolName,
                                    SymbolTable *symbolTable) {
  // Create a zero-terminated byte representation and allocate global symbol.
  llvm::SmallString<32> elementVals;
  elementVals.append(string.begin(), string.end());
  elementVals.push_back('\0');
  Type i8Type = rewriter.getI8Type();
  auto dataAttr = StringAttr::get(rewriter.getContext(), elementVals);
  auto arrayTy = LLVM::LLVMArrayType::get(i8Type, elementVals.size());
  LLVM::GlobalOp globalOp =
      insertLLVMGlobal(rewriter, loc, symbolName, true, arrayTy,
                       LLVM::Linkage::Private, dataAttr, symbolTable);
  auto msgAddr = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
  return msgAddr;
}

// First make sure we have an unranked memref descriptor representation.
UnrankedMemRefDescriptor
mlir::getUnrankedLLVMMemRefDescriptor(OpBuilder &rewriter, Location loc,
                                      const LLVMTypeConverter &typeConverter,
                                      Value ranked, MemRefType type) {
  Value rank = rewriter.create<LLVM::ConstantOp>(
      loc, typeConverter.getIndexType(), type.getRank());
  Value ptr = typeConverter.promoteOneMemRefDescriptor(loc, ranked, rewriter);
  auto unrankedType =
      UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
  return UnrankedMemRefDescriptor(UnrankedMemRefDescriptor::pack(
      rewriter, loc, typeConverter, unrankedType, ValueRange{rank, ptr}));
}

SmallVector<Value> mlir::promoteLLVMMemRefDescriptorsToUnranked(
    OpBuilder &rewriter, Location loc, const LLVMTypeConverter &typeConverter,
    ValueRange originalOperands, ValueRange convertedOperands) {
  SmallVector<Value> newOperands;
  newOperands.reserve(convertedOperands.size());
  for (auto [original, converted] :
       llvm::zip(originalOperands, convertedOperands)) {
    if (auto memrefType = dyn_cast<MemRefType>(original.getType())) {
      assert(isa<LLVM::LLVMStructType>(converted.getType()) &&
             "expected converted value to type LLVMStructType");
      Value unrankedDescriptor = getUnrankedLLVMMemRefDescriptor(
          rewriter, loc, typeConverter, converted, memrefType);
      newOperands.push_back(unrankedDescriptor);
      continue;
    }
    newOperands.push_back(converted);
  }
  return newOperands;
}

LLVM::LLVMFuncOp mlir::insertLLVMCtorFunction(
    OpBuilder &rewriter, Location loc, SymbolTable &symbolTable, StringRef name,
    int32_t priority,
    const std::function<void(OpBuilder &, Location)> &bodyBuilder) {
  OpBuilder::InsertionGuard g(rewriter);
  Operation *module = symbolTable.getOp();
  rewriter.setInsertionPointToEnd(&module->getRegion(0).front());

  MLIRContext *ctx = rewriter.getContext();
  LLVM::LLVMFunctionType functionType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});

  LLVM::LLVMFuncOp func =
      rewriter.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *ctorBlock = func.addEntryBlock(rewriter);
  rewriter.setInsertionPointToEnd(ctorBlock);
  bodyBuilder(rewriter, loc);
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

  symbolTable.insert(func);

  // Register the constructor function
  rewriter.setInsertionPointAfter(func);
  rewriter.create<LLVM::GlobalCtorsOp>(
      loc, rewriter.getArrayAttr({FlatSymbolRefAttr::get(ctx, name)}),
      rewriter.getI32ArrayAttr({priority}));
  return func;
}

LLVM::LLVMFuncOp mlir::insertLLVMDtorFunction(
    OpBuilder &rewriter, Location loc, SymbolTable &symbolTable, StringRef name,
    int32_t priority,
    const std::function<void(OpBuilder &, Location)> &bodyBuilder) {
  OpBuilder::InsertionGuard g(rewriter);
  Operation *module =
      SymbolTable::getNearestSymbolTable(rewriter.getBlock()->getParentOp());
  assert(module && "expected module");
  rewriter.setInsertionPointToEnd(&module->getRegion(0).front());

  MLIRContext *ctx = rewriter.getContext();
  LLVM::LLVMFunctionType functionType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});

  LLVM::LLVMFuncOp func =
      rewriter.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *ctorBlock = func.addEntryBlock(rewriter);
  rewriter.setInsertionPointToEnd(ctorBlock);
  bodyBuilder(rewriter, loc);
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

  symbolTable.insert(func);

  // Register the constructor function
  rewriter.setInsertionPointAfter(func);
  rewriter.create<LLVM::GlobalDtorsOp>(
      loc, rewriter.getArrayAttr({FlatSymbolRefAttr::get(ctx, name)}),
      rewriter.getI32ArrayAttr({priority}));
  return func;
}

static FailureOr<std::unique_ptr<llvm::ToolOutputFile>>
serializeDenseResourceElementsAttrToFile(
    Location loc, DenseResourceElementsAttr denseResourceAttr,
    StringRef outputPath) {
  assert(denseResourceAttr && "expected non-null attribute");
  ShapedType type = denseResourceAttr.getType();
  assert(type.getNumElements() > 0 && "Expected non-empty elements attribute");

  AsmResourceBlob *blob = denseResourceAttr.getRawHandle().getBlob();
  if (!blob) {
    if (denseResourceAttr.getRawHandle().getKey() == "__elided__") {
      std::string err;
      std::unique_ptr<llvm::ToolOutputFile> of =
          mlir::openOutputFile(outputPath, &err);
      if (!of)
        return emitError(UnknownLoc::get(loc->getContext()))
               << "failed to open output file: " << err;
      return of;
    }
    return emitError(loc, "resource does not exist");
  }

  ArrayRef<char> rawData = blob->getData();

  int64_t numElements = denseResourceAttr.getType().getNumElements();
  int64_t elementByteSize = rawData.size() / numElements;
  if (8 * elementByteSize != type.getElementTypeBitWidth())
    return emitError(loc, "raw data size does not match element type size");

  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(outputPath, &err);
  if (!of)
    return emitError(UnknownLoc::get(loc->getContext()))
           << "failed to open output file: " << err;
  of->os().write(rawData.data(), rawData.size());
  return of;
}

static FailureOr<std::unique_ptr<llvm::ToolOutputFile>>
serializeDenseElementsAttrToFile(Location loc,
                                 DenseElementsAttr denseResourceAttr,
                                 StringRef outputPath) {
  assert(denseResourceAttr && "expected non-null attribute");
  ShapedType type = denseResourceAttr.getType();
  assert(type.getNumElements() > 0 && "Expected non-empty elements attribute");

  if (denseResourceAttr.isSplat())
    return emitError(loc, "could not serialize elements of type ") << type;

  ArrayRef<char> rawData = denseResourceAttr.getRawData();
  int64_t numElements = denseResourceAttr.getType().getNumElements();
  int64_t elementByteSize = rawData.size() / numElements;
  if (8 * elementByteSize != type.getElementTypeBitWidth())
    return emitError(loc, "raw data size does not match element type size");

  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(outputPath, &err);
  if (!of)
    return emitError(UnknownLoc::get(loc->getContext()))
           << "failed to open output file: " << err;
  of->os().write(rawData.data(), rawData.size());
  return of;
}

static FailureOr<std::unique_ptr<llvm::ToolOutputFile>>
serializeDenseSplatElementsAttrToFile(Location loc, DenseElementsAttr values,
                                      StringRef outputPath) {
  assert(values && "expected non-null attribute");
  ShapedType type = values.getType();
  assert(type.getNumElements() > 0 && "Expected non-empty elements attribute");
  assert(values.isSplat() && "expected splat elements");
  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(outputPath, &err);
  if (!of)
    return emitError(UnknownLoc::get(loc->getContext()))
           << "failed to open output file: " << err;
  auto rtt = cast<ShapedType>(values.getType());
  auto fill = [&](auto element, int64_t divisor = 1) {
    for (int64_t i = 0; i < rtt.getNumElements() / divisor; i++)
      of->os().write(reinterpret_cast<const char *>(&element), sizeof(element));
  };

  if (rtt.getElementType().isInteger(32)) {
    fill(values.getSplatValue<int32_t>());
    return of;
  }
  if (rtt.getElementType().isInteger(64)) {
    fill(values.getSplatValue<int64_t>());
    return of;
  }
  if (rtt.getElementType().isInteger(8)) {
    fill(values.getSplatValue<int8_t>());
    return of;
  }
  if (rtt.getElementType().isF32()) {
    fill(values.getSplatValue<float>());
    return of;
  }
  if (rtt.getElementType().isF16() || rtt.getElementType().isBF16()) {
    APInt tmp = values.getSplatValue<APFloat>().bitcastToAPInt();
    assert(tmp.getBitWidth() == 16 && "unexpected bitwidth");
    uint16_t fillValue = *reinterpret_cast<const uint16_t *>(tmp.getRawData());
    fill(fillValue);
    return of;
  }
  if (isa<Float8E4M3FNType>(rtt.getElementType())) {
    APInt tmp = values.getSplatValue<APFloat>().bitcastToAPInt();
    assert(tmp.getBitWidth() == 8 && "unexpected bitwidth");
    uint8_t fillValue = *reinterpret_cast<const uint8_t *>(tmp.getRawData());
    fill(fillValue);
    return of;
  }
  if (rtt.getElementType().isInteger(4)) {
    APInt tmp = values.getSplatValue<APInt>();
    assert(tmp.getBitWidth() == 4 && "expected 4 bit integer");
    uint8_t packed = 0;
    uint8_t value = *reinterpret_cast<const uint8_t *>(tmp.getRawData());
    // Pack `value` in the upper and the lower nibble
    packed |= (value & 0x0F);
    packed |= ((value & 0x0F) << 4);
    // Fill `data` vector with `packed`
    fill(packed, 2);
    return of;
  }
  return emitError(loc) << "cannot serialize splat elements of type"
                        << values.getType();
}

FailureOr<std::unique_ptr<llvm::ToolOutputFile>>
mlir::serializeElementsAttrToFile(Location loc, ElementsAttr attr,
                                  StringRef outputDir, StringRef filename) {
  if (!attr.getShapedType().getElementType().isSignlessIntOrFloat())
    return emitError(loc, "could not serialize elements of type ")
           << attr.getShapedType();
  llvm::SmallString<64> outputPath(outputDir);
  llvm::sys::path::append(outputPath, filename);

  if (auto dense = dyn_cast<DenseResourceElementsAttr>(attr))
    return serializeDenseResourceElementsAttrToFile(loc, dense, outputPath);
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (dense.isSplat())
      return serializeDenseSplatElementsAttrToFile(loc, dense, outputPath);
    else
      return serializeDenseElementsAttrToFile(loc, dense, outputPath);
  }
  return emitError(loc, "could not serialize elements of type ")
         << attr.getType();
}
