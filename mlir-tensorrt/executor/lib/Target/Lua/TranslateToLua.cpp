//===- TranslateToLua.cpp  ------------------------------------------------===//
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
///
/// Implementation of Lua translation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Target/Lua/TranslateToLua.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <stack>

using namespace mlir;

namespace {
/// Helper class for emitting Lua code. It is inspired by the C++ emitter in
/// upstream MLIR. Potentially upstream could be refactored to provide the
/// common components without the code duplication here.
class LuaEmitter {
public:
  explicit LuaEmitter(MLIRContext *ctx, raw_ostream &os,
                      const DataLayout &dataLayout);

  /// Emit Lua ofr a "module-like" operation. This creates a new scope for all
  /// resources. It is expected that this is only used as the top-level
  /// entrypoint and that the module-like operation does not contain nested
  /// modules.
  LogicalResult emitModule(Operation &op);

  /// Emit Lua for an operation.
  LogicalResult emitOperation(Operation &op);

  /// Emit Lua for a Block.
  LogicalResult emitBlock(Block &block, bool isEntryBlock);

  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// RAII helper function to manage entering/exiting scopes.
  struct RegionScope {
    explicit RegionScope(LuaEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
      emitter.globalsInScopeCount.push(emitter.globalsInScopeCount.top());
    }
    ~RegionScope() {
      emitter.labelInScopeCount.pop();
      emitter.globalsInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    LuaEmitter &emitter;
  };

  struct LocalVariableScope {
    explicit LocalVariableScope(LuaEmitter &emitter, unsigned additionalLocals)
        : localMapperScope(emitter.localMapper), emitter(emitter) {
      emitter.localsInScopeCount.push(emitter.localsInScopeCount.top() +
                                      additionalLocals);
    }
    ~LocalVariableScope() { emitter.localsInScopeCount.pop(); }

  private:
    llvm::ScopedHashTableScope<Value, std::string> localMapperScope;
    LuaEmitter &emitter;
  };

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block) const;

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateLabel(Block &block);
  StringRef getLabel(Block &block) {
    assert(hasBlockLabel(block));
    return *blockMapper.begin(&block);
  }

  /// Emit "(var_name `,`)* `var_name ="
  LogicalResult emitAssignPrefix(Operation *op);

  /// Generate a new variable name consisting of `v+[number]`. The number will
  /// always be monotonic within a single scope, regardless of which prefix is
  /// used.
  StringRef createGlobalVariableName(Value val, StringRef prefix = "v");
  StringRef getOrCreateGlobalVariableName(Value val, StringRef prefix = "v");
  StringRef createLocalVariableName(Value val, StringRef prefix = "l");

  StringRef getVariableName(Value val);

  /// Returns true if the type is supported for operands/results.
  bool isSupportedFloatingPointType(Type type) const;

  [[nodiscard]] bool isValueInScope(Value val) const;

  raw_indented_ostream &getStream() { return os; }

  template <typename T>
  LuaEmitter &operator<<(const T &data) {
    os << data;
    return *this;
  }

  void setVariableName(Value val, StringRef name) {
    valueMapper.insert(val, name.str());
  }

protected:
  /// Map from value to name of lua variable that contain the name.
  ValueMapper valueMapper;
  ValueMapper localMapper;

  /// Map from block to name of lua label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> localsInScopeCount;
  std::stack<int64_t> globalsInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  MLIRContext *ctx;
  raw_indented_ostream os;

  /// The data layout of the module.
  mlir::DataLayout moduleDataLayout;
};
} // namespace

bool LuaEmitter::isSupportedFloatingPointType(Type type) const {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
         isa<Float8E4M3FNType>(type);
}

//===----------------------------------------------------------------------===//
// Helpers for emitting attributes
//===----------------------------------------------------------------------===//

static LogicalResult emitAttribute(raw_ostream &os, Location loc,
                                   IntegerAttr attr) {
  // For boolean, directly convert to true/false.
  APInt val = attr.getValue();
  if (val.getBitWidth() == 1) {
    if (val.getBoolValue())
      os << "1";
    else
      os << "0";
    return success();
  }

  // Otherwise, convert to a Lua integer. We convert i4, i8, 16, i32 and i64 to
  // Lua integer (`number` strictly). In any case, we never use unsigned
  // representation.
  if (val.getBitWidth() == 32 || val.getBitWidth() == 64 ||
      val.getBitWidth() == 16 || val.getBitWidth() == 8 ||
      val.getBitWidth() == 4) {
    SmallString<128> strValue;
    val.toString(strValue, 10, /*Signed=*/true, /*formatAsCLiteral=*/false);
    os << strValue;
    return success();
  }

  return emitError(loc)
         << "can only translate integers with bitwidth of 1, 32, or 64";
}

static LogicalResult emitAttribute(raw_ostream &os, Location loc,
                                   FloatAttr attr) {
  APFloat val = attr.getValue();
  if (val.isFinite()) {
    SmallString<128> strValue;
    // Use default values of toString except don't truncate zeros.
    val.toString(strValue, 0, 0, false);
    os << strValue;
    return success();
  }
  if (val.isNaN()) {
    // "-(0/0)" converts to "+nan", "(0/0)" is "-nan".
    if (!val.isNegative())
      os << "-";
    os << "(0/0)";
    return success();
  }
  if (val.isInfinity()) {
    // Use 1/0 to represent "inf".
    if (val.isNegative())
      os << "-";
    os << "1/0";
    return success();
  }
  return emitError(loc) << "unhandled float attr case: " << attr;
}

static LogicalResult emitAttribute(raw_ostream &os, Location loc,
                                   Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return emitAttribute(os, loc, intAttr);
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return emitAttribute(os, loc, floatAttr);
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    /// Using the stream operator will also print the required quotes.
    os << attr;
    return success();
  }
  return emitError(loc) << "unhandled attribute type: " << attr;
}

//===----------------------------------------------------------------------===//
// `cf` dialect ops
//===----------------------------------------------------------------------===//

static LogicalResult printControlFlowOp(LuaEmitter &emitter, cf::BranchOp op) {
  Block *destBlock = op.getDest();
  // Declare non-local args to hold block arguments.
  for (auto [operand, blockArg] :
       llvm::zip(op.getDestOperands(), destBlock->getArguments())) {
    // If we are branching from a entry block, we can can use a local.
    emitter << emitter.getVariableName(blockArg);
    emitter << " = " << emitter.getVariableName(operand) << ";\n";
  }
  emitter << "goto " << emitter.getOrCreateLabel(*op.getDest()) << ";\n";
  return success();
}

static LogicalResult printControlFlowOp(LuaEmitter &emitter,
                                        cf::CondBranchOp op) {

  SmallVector<Value> trueOperands, falseOperands;

  // Assign variables for the destination Block's BlockArguments.
  auto emitBlockArgs = [&](Block *destBlock, ValueRange operands) {
    // Declare non-local args to hold block arguments.
    for (auto [operand, blockArg] :
         llvm::zip(operands, destBlock->getArguments())) {
      emitter << emitter.getVariableName(blockArg);
      emitter << " = " << emitter.getVariableName(operand) << ";\n";
    }
  };

  auto condName = emitter.getVariableName(op.getCondition());

  emitter << "if (" << condName << " == 1) or (" << condName << " == true)"
          << " then\n";

  auto emitBranch = [&](Block *destBlock, ValueRange operands) {
    emitter.getStream().indent();
    emitBlockArgs(destBlock, operands);
    emitter << "goto " << emitter.getOrCreateLabel(*destBlock) << ";\n";
    emitter.getStream().unindent();
  };

  emitBranch(op.getTrueDest(), op.getTrueDestOperands());
  emitter << "else\n";
  emitBranch(op.getFalseDest(), op.getFalseDestOperands());
  emitter << "end\n";
  return success();
}

//===----------------------------------------------------------------------===//
// `func` dialect ops
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(LuaEmitter &emitter, func::ReturnOp op) {
  emitter << "return";
  if (op->getNumOperands() > 0)
    emitter << " ";
  llvm::interleaveComma(op->getOperands(), emitter.getStream(), [&](Value v) {
    emitter << emitter.getVariableName(v);
  });
  emitter << ";\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter, func::FuncOp op) {
  // We don't actually need to translate pure declarations for external C
  // functions.
  if (op.isDeclaration()) {
    emitter << "__check_for_function(\"" << op.getName() << "\");\n";
    return success();
  }

  emitter << "function " << op.getName() << " ";

  // We augment the local count with what is required for call ops.
  LuaEmitter::LocalVariableScope localScope(emitter, /*additionalLocals=*/0);

  emitter << "(";
  llvm::interleaveComma(
      op.getArguments(), emitter.getStream(), [&](BlockArgument &arg) {
        emitter << emitter.createLocalVariableName(arg, "arg");
      });
  emitter << ")\n";
  emitter.getStream().indent();

  for (auto [idx, block] : llvm::enumerate(op.getBody())) {
    if (failed(emitter.emitBlock(block, /*isEntryBlock=*/idx == 0)))
      return failure();
  }
  emitter.getStream().unindent();
  emitter << "end\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter, func::CallOp op) {
  if (op->getNumResults() > 0) {
    if (failed(emitter.emitAssignPrefix(op.getOperation())))
      return failure();
  }
  emitter << op.getCallee() << "(";
  llvm::interleaveComma(op->getOperands(), emitter.getStream(), [&](Value v) {
    emitter << emitter.getVariableName(v);
  });
  emitter << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    func::CallIndirectOp op) {
  if (op->getNumResults() > 0) {
    if (failed(emitter.emitAssignPrefix(op.getOperation())))
      return failure();
  }
  emitter << emitter.getVariableName(op.getCallee()) << "(";
  llvm::interleaveComma(
      op.getCalleeOperands(), emitter.getStream(),
      [&](Value v) { emitter << emitter.getVariableName(v); });
  emitter << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Executor op printers
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::AssertOp op) {
  StringRef varName = emitter.getVariableName(op.getArg());
  emitter << "assert("
          << "(" << varName << " == 1) or (" << varName << " == true), \""
          << op.getMsg() << "\");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::ConstantOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();

  auto wrapInFuncCall = [&](auto func, StringRef typeName) {
    emitter << "executor_constant_" << typeName << "(";
    if (failed(func()))
      return failure();
    emitter << ");\n";
    return success();
  };

  auto emitAttr = [&]() {
    return emitAttribute(emitter.getStream(), op.getLoc(), op.getValue());
  };

  if (op.getType().isF16())
    return wrapInFuncCall(emitAttr, "f16");

  if (isa<Float8E4M3FNType>(op.getType()))
    return wrapInFuncCall(emitAttr, "f8E4M3FN");

  if (op.getType().isBF16())
    return wrapInFuncCall(emitAttr, "bf16");

  if (op.getType().isInteger(4))
    return wrapInFuncCall(emitAttr, "i4");

  if (failed(emitAttr()))
    return failure();
  emitter << ";\n";
  return success();
}

static LogicalResult printPtrToIntOp(LuaEmitter &emitter,
                                     executor::PtrToIntOp op,
                                     const DataLayout &dataLayout) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();

  uint64_t ptrWidth = dataLayout.getTypeSizeInBits(op.getArg().getType());
  emitter << "_ptrtoint_i" << ptrWidth << "_" << op.getType() << "("
          << emitter.getVariableName(op.getArg()) << ");\n";
  return success();
}

static LogicalResult printIntToPtrOp(LuaEmitter &emitter,
                                     executor::IntToPtrOp op,
                                     const DataLayout &dataLayout) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  uint64_t ptrWidth = dataLayout.getTypeSizeInBits(op.getType());
  emitter << "_inttoptr_i" << ptrWidth << "_" << op.getOperand().getType()
          << "(" << emitter.getVariableName(op.getArg()) << ");\n";
  return success();
}

static LogicalResult printExecutorBinaryInfixOperation(LuaEmitter &emitter,
                                                       Operation *op) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << emitter.getVariableName(lhs) << " ";
  if (isa<executor::SubIOp>(op) || isa<executor::SubFOp>(op)) {
    emitter << "-";
  } else if (isa<executor::AddIOp>(op) || isa<executor::AddFOp>(op)) {
    emitter << "+";
  } else if (isa<executor::MulIOp>(op) || isa<executor::MulFOp>(op)) {
    emitter << "*";
  } else if (isa<executor::SRemIOp>(op)) {
    emitter << "%";
  } else if (isa<executor::SFloorDivIOp>(op)) {
    emitter << "//";
  } else {
    return op->emitOpError() << "unsupported binary executor arithmetic op";
  }
  emitter << " " << emitter.getVariableName(rhs) << ";\n";
  return success();
}

static LogicalResult printRuntimeBuiltinUnaryOp(LuaEmitter &emitter,
                                                Operation *op,
                                                StringRef opName) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "_" << opName << "_" << op->getResultTypes().front() << "("
          << emitter.getVariableName(op->getOperands().front()) << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::CreateTableOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "{";
  llvm::interleaveComma(op.getInit(), emitter, [&](Value v) {
    emitter << emitter.getVariableName(v);
  });
  emitter << "};\n";
  return success();
}

static LogicalResult printRemF(LuaEmitter &emitter, executor::RemFOp op) {
  if (!emitter.isSupportedFloatingPointType(op.getType()))
    return op->emitOpError()
           << "unsupported floating-point type: " << op.getType();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "_remf_" << op.getType() << "("
          << emitter.getVariableName(op.getLhs()) << ", "
          << emitter.getVariableName(op.getRhs()) << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::ExtractTableValueOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << emitter.getVariableName(op.getTable()) << "[" << op.getIndex() + 1
          << "];\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::DynamicExtractTableValueOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << emitter.getVariableName(op.getTable()) << "["
          << emitter.getVariableName(op.getIndex()) << " + 1];\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::InsertTableValueOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  // We have to do a copy of the source table since otherwise lua's semantic is
  // to do a reference modification. Our canonicalization patterns should limit
  // the number of copies. Note: this actually may have bugs for nested tables.
  emitter << "{};\n";
  emitter << "for j,x in ipairs(" << emitter.getVariableName(op.getTable())
          << ") do " << emitter.getVariableName(op.getResult())
          << "[j] = x end;\n";
  emitter << emitter.getVariableName(op.getResult()) << "[" << op.getIndex() + 1
          << "] = " << emitter.getVariableName(op.getValue()) << ";\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter, executor::PrintOp op) {
  if (std::optional<StringRef> format = op.getFormat()) {
    emitter << "print(string.format(\"" << *format << "\"";
    if (op->getNumOperands() > 0)
      emitter << ", ";
    llvm::interleaveComma(op->getOperands(), emitter, [&](Value v) {
      emitter << emitter.getVariableName(v);
    });
    emitter << "));\n";
    return success();
  }
  emitter << "print(";
  llvm::interleaveComma(op->getOperands(), emitter, [&](Value v) {
    emitter << emitter.getVariableName(v);
  });
  emitter << ");\n";
  return success();
}

static LogicalResult printMemCpyOp(LuaEmitter &emitter, executor::MemcpyOp op) {
  emitter << "executor_memcpy(" << emitter.getVariableName(op.getSrc()) << ", "
          << emitter.getVariableName(op.getSrcOffset()) << ", "
          << emitter.getVariableName(op.getDest()) << ", "
          << emitter.getVariableName(op.getDestOffset()) << ", "
          << emitter.getVariableName(op.getNumBytes()) << ");\n";
  return success();
}

/// Emit binary arithmetic op that calls out to a special function.
/// Since we don't want this function name to be calculated until all types are
/// resolved, we avoid handling this in "executor expand ops" pass. Also, this
/// is a specific peculiarity of Lua.
static LogicalResult
printExecutorBinarySpecialFunctionOperation(LuaEmitter &emitter,
                                            Operation *op) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  SmallVector<llvm::StringRef> frags;
  llvm::SplitString(op->getName().getStringRef(), frags, ".");
  emitter << "_" << frags[1] << "_" << lhs.getType() << "("
          << emitter.getVariableName(lhs) << ", "
          << emitter.getVariableName(rhs) << ");\n";
  return success();
}

static LogicalResult printExecutorICmpOp(LuaEmitter &emitter,
                                         executor::ICmpOp op) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  SmallVector<llvm::StringRef> frags;
  llvm::SplitString(op->getName().getStringRef(), frags, ".");
  emitter << "_" << frags[1] << "_"
          << executor::stringifyICmpType(op.getPredicate()) << "_"
          << lhs.getType() << "(" << emitter.getVariableName(lhs) << ", "
          << emitter.getVariableName(rhs) << ");\n";
  return success();
}

static LogicalResult printExecutorFCmpOp(LuaEmitter &emitter,
                                         executor::FCmpOp op) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  SmallVector<llvm::StringRef> frags;
  llvm::SplitString(op->getName().getStringRef(), frags, ".");
  emitter << "_" << frags[1] << "_"
          << executor::stringifyFCmpType(op.getPredicate()) << "_"
          << lhs.getType() << "(" << emitter.getVariableName(lhs) << ", "
          << emitter.getVariableName(rhs) << ");\n";
  return success();
}

static LogicalResult printSelectOp(LuaEmitter &emitter, executor::SelectOp op) {
  Value condition = op->getOperand(0);
  Value trueValue = op.getOperand(1);
  Value falseValue = op.getOperand(2);
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "_select(" << emitter.getVariableName(condition) << ","
          << emitter.getVariableName(trueValue) << ","
          << emitter.getVariableName(falseValue) << ");\n";
  return success();
}

static LogicalResult printBitcastOp(LuaEmitter &emitter,
                                    executor::BitcastOp op) {
  Value input = op.getInput();
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "_bitcast_" << input.getType() << "_" << op.getType() << "("
          << emitter.getVariableName(input) << ");\n";
  return success();
}

/// Translate string literal.
static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::StrLiteralOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "\"" << op.getValue() << "\""
          << ";\n";
  return success();
}

/// Translate `executor.get_global`.
static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::GetGlobalOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << op.getName() << ";\n";
  return success();
}

/// Translate `executor.set_global`.
static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::SetGlobalOp op) {
  emitter << op.getName() << " = " << emitter.getVariableName(op.getValue())
          << ";\n";
  return success();
}

/// Translate `executor.load_data_segment`.
static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::ConstantResourceLoadOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << op.getName() << ";\n";
  return success();
}

/// Translate `executor.call`.
static LogicalResult printOperation(LuaEmitter &emitter, executor::CallOp op) {

  if (op->getNumResults() > 0) {
    if (failed(emitter.emitAssignPrefix(op.getOperation())))
      return failure();
  }
  emitter << op.getCallee() << "(";

  llvm::interleaveComma(op->getOperands(), emitter.getStream(), [&](Value v) {
    emitter << emitter.getVariableName(v);
  });

  emitter << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::CoroAwaitOp op) {
  if (op->getNumResults() > 0) {
    if (failed(emitter.emitAssignPrefix(op.getOperation())))
      return failure();
  }

  emitter << "coroutine.resume(" << emitter.getVariableName(op.getCallee());
  if (!op.getCalleeOperands().empty()) {
    emitter << ", ";
    llvm::interleaveComma(
        op.getCalleeOperands(), emitter.getStream(),
        [&](Value v) { emitter << emitter.getVariableName(v); });
  }
  emitter << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::CoroCreateOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << "coroutine.create(";
  emitter << op.getFunc();
  emitter << ");\n";
  return success();
}

static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::CoroYieldOp op) {
  emitter << "coroutine.yield(";
  llvm::interleaveComma(op.getYielded(), emitter.getStream(), [&](Value v) {
    emitter << emitter.getVariableName(v);
  });
  emitter << ");\n";
  return success();
}

/// Translate `executor.func`. Currently we only allow these for variadic
/// function declarations.
static LogicalResult printOperation(LuaEmitter &emitter, executor::FuncOp op) {
  if (!op.isDeclaration())
    return op.emitOpError("expected all executor.func to be declarations");
  emitter << "__check_for_function(\"" << op.getName() << "\");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// LuaEmitter implementation
//===----------------------------------------------------------------------===//

LuaEmitter::LuaEmitter(MLIRContext *ctx, raw_ostream &os,
                       const DataLayout &dataLayout)
    : ctx(ctx), os(os), moduleDataLayout(dataLayout) {
  localsInScopeCount.push(0);
  labelInScopeCount.push(0);
  globalsInScopeCount.push(0);
}

bool LuaEmitter::hasBlockLabel(Block &block) const {
  return blockMapper.count(&block) > 0;
}

LogicalResult LuaEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getLabel(block) << ":\n";
  return success();
}

/// Return the existing or a new label for a Block.
StringRef LuaEmitter::getOrCreateLabel(Block &block) {
  if (!hasBlockLabel(block))
    blockMapper.insert(&block,
                       llvm::formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool LuaEmitter::isValueInScope(Value val) const {
  return valueMapper.count(val) != 0 || localMapper.count(val) != 0;
}

StringRef LuaEmitter::createGlobalVariableName(Value val, StringRef prefix) {
  assert(!isValueInScope(val) && "expected val not to be in scope");
  std::string valueName =
      llvm::formatv("{0}{1}", prefix, ++globalsInScopeCount.top()).str();
  valueMapper.insert(val, valueName);
  return *valueMapper.begin(val);
}

StringRef LuaEmitter::getOrCreateGlobalVariableName(Value val,
                                                    StringRef prefix) {
  if (localMapper.count(val) != 0)
    return *localMapper.begin(val);
  if (valueMapper.count(val) == 0)
    return createGlobalVariableName(val, prefix);
  return *valueMapper.begin(val);
}

StringRef LuaEmitter::createLocalVariableName(Value val, StringRef prefix) {
  assert(!isValueInScope(val) && "expected val not to be in scope");
  std::string valueName =
      llvm::formatv("{0}{1}", prefix, localsInScopeCount.top()++).str();
  localMapper.insert(val, valueName);
  return *localMapper.begin(val);
}

StringRef LuaEmitter::getVariableName(Value val) {
  auto it = valueMapper.begin(val);
  if (it != valueMapper.end())
    return *it;
  auto itLocal = localMapper.begin(val);
  if (itLocal != localMapper.end())
    return *itLocal;
  llvm::report_fatal_error("value is not in scope");
}

LogicalResult LuaEmitter::emitAssignPrefix(Operation *op) {
  // In Lua, a variable can be declared local as long is it is not used in
  // another block. Lua locals go out of scope when the block terminates. Since
  // we translate Blocks 1-1 with Lua blocks, just check if it is used in
  // another block (that is not a child). The only exception is the entry-block
  // since MLIR guarantees that we can't jump into the scope of these variables.
  constexpr unsigned kMaxNumLocals = 200;
  bool isEntryBlock = op->getBlock()->isEntryBlock();
  // Helper to check if a value can be emitted as a local variable.
  auto isLocalVar = [&](Value v) -> bool {
    return (isEntryBlock || !v.isUsedOutsideOfBlock(op->getBlock())) &&
           localsInScopeCount.top() + op->getNumResults() < kMaxNumLocals;
  };
  SmallVector<Value> localVars;
  for (Value v : op->getResults()) {
    if (isLocalVar(v))
      localVars.push_back(v);
  }

  // Inline variable declarations when all results are locals.
  if (localVars.size() == op->getNumResults()) {
    os << "local ";
    llvm::interleaveComma(op->getResults(), os,
                          [&](Value v) { os << createLocalVariableName(v); });
    // Starting in Lua 5.4, it supports "<const>" attributes for local vars,
    // which can save a lot of register movement, especially when passed to
    // calls.
    os << " <const> = ";
    return success();
  }

  // Otherwise, declare the locals separately first before assignment.
  if (!localVars.empty()) {
    os << "local ";
    llvm::interleaveComma(localVars, os,
                          [&](Value v) { os << createLocalVariableName(v); });
    os << " <const>;\n";
  }
  llvm::interleaveComma(op->getResults(), os, [&](Value v) {
    // TODO: if a value is not in scope at this point, we emit it as a global
    // variable. This can lead to unintended global sharing between Lua threads.
    // Use a local table to store the values instead.
    if (isValueInScope(v))
      os << getVariableName(v);
    else
      os << createGlobalVariableName(v);
  });
  os << " = ";

  return success();
}

// In MLIR, SSA values scoping depends on dominance. For example, all
// SSA values defined in a function entry block are visible to all other
// blocks in the function body region. Typically, in a multi-block region,
// blocks pass values through branching terminators and block arguments.
// However, it's also possible to define SSA values inside a non-entry block
// that are visible to blocks below in the region as long as the definition
// of the SSA value dominates the other block, the SSA value is in scope.
//
// Lua variables can either be 'locals' or 'globals'. We prefer to use
// locals as much as possible since it is performant and offers less surprises.
// At first glance, you might expect all SSA values to be "local" variables.
// However, due to the above MLIR subtleties, it's impossible to exactly match
// the MLIR rules using Lua blocks and only 'local' variables. In Lua, you can
// declare a "block" and give it a label as follows:
//
// ```lua
// ::bb0:: do
//   <nested statements...>
// end
// ```
//
// Lua 'local' variables are block-scoped, which means any variable
// declared as 'local' can't be accessed after the 'end'. Furthermore,
// similar to the rules about dominance in MLIR, in Lua you can't jump (goto)
// a label if the required definition of a local variable would be bypassed. We
// can ensure this always is true if we follow some simple rules:
//
// 1. MLIR blocks should be translated 1-1 with Lua blocks using the form
//   described above.
// 2. An SSA variable can only be declared 'local' if a) it is in a function
//   entry block or b) it is only used within its Block.
//
// We allow SSA variables in 2.a to always be 'local' since the entry block
// always dominates all other blocks in the function region and MLIR disallows
// branching to entry blocks (entry block must have no predecessors).
//
// In addition to the above rules, there is a hard limit on the number of
// local variables that are allowed within a single block (200). This is
// due to Lua bytecode limitations. Importantly, function arguments are also
// considered locals, so functions with huge argument lists can exhaust the
// available local slots. In such cases, we might have to resort to using
// globals variables, even for SSA values defined function entry blocks.
LogicalResult LuaEmitter::emitBlock(Block &block, bool isEntryBlock) {
  // If this is a non-entry block, then push a new scope for the locals.
  // We don't need additional locals since no new arguments are created.
  std::unique_ptr<LocalVariableScope> scope =
      isEntryBlock
          ? std::unique_ptr<LocalVariableScope>(nullptr)
          : std::make_unique<LocalVariableScope>(*this, /*additionalLocals=*/0);

  // Only declare new Lua block scope for non-entry blocks.
  if (!isEntryBlock) {
    os << "::" << getOrCreateLabel(block) << ":: do\n";
    os.indent();
  } else {
    // In the entry block, declare all of the block arguments needed throughout
    // the region as local variables. Initialize them all to nil. This avoids
    // having to use ad-hoc globals at the branch points.
    Region *region = block.getParent();
    for (auto [idx, otherBlock] : llvm::enumerate(region->getBlocks())) {
      // We don't need to declare block arguments for the entry block; those are
      // e.g. function arguments and are handled by the parent op.
      if (idx == 0)
        continue;
      for (BlockArgument arg : otherBlock.getArguments())
        getStream() << "local " << createLocalVariableName(arg, "barg")
                    << " = nil;\n";

      // Declare all results of operations in the block as locals in the entry
      // block if they are used outside of the block.
      for (Operation &op : otherBlock) {
        for (Value result : op.getResults()) {
          if (result.isUsedOutsideOfBlock(&otherBlock)) {
            getStream() << "local " << createLocalVariableName(result)
                        << " = nil;\n";
          }
        }
      }
    }
  }

  for (Operation &op : block.getOperations()) {
    if (failed(emitOperation(op)))
      return failure();
  }
  // Terminate the block if required.
  if (!isEntryBlock) {
    os.unindent();
    os << "end\n";
  }
  return success();
}

static bool isModuleLike(Operation &op) {
  return op.hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         op.hasTrait<OpTrait::SymbolTable>() && op.getNumRegions() == 1 &&
         op.getRegion(0).hasOneBlock();
}

LogicalResult LuaEmitter::emitModule(Operation &op) {
  assert(isModuleLike(op) && "expected module-like operation");
  LuaEmitter::RegionScope scope(*this);
  LuaEmitter::LocalVariableScope localScope(*this, /*additionalLocals=*/0);
  return emitBlock(op.getRegion(0).front(), /*isEntryBlock=*/true);
}

LogicalResult LuaEmitter::emitOperation(Operation &op) {
  // Global/const resource declarations don't need to get emitted.
  if (isa<executor::DataSegmentOp, executor::GlobalOp>(op))
    return success();

  if (isa<executor::ExecutorDialect>(op.getDialect())) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
        .Case<executor::FuncOp, executor::CallOp, executor::ConstantOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::AbsFOp>([&](auto op) {
          return printRuntimeBuiltinUnaryOp(*this, op.getOperation(), "absf");
        })
        .Case<executor::CopysignOp>([&](auto op) {
          return printRuntimeBuiltinUnaryOp(*this, op.getOperation(),
                                            "copysign");
        })
        .Case<executor::MulIOp, executor::MulFOp, executor::SFloorDivIOp,
              executor::AddIOp, executor::AddFOp, executor::SubIOp,
              executor::SubFOp, executor::SRemIOp>([&](auto op) {
          return printExecutorBinaryInfixOperation(*this, op);
        })
        .Case<executor::RemFOp>([&](auto op) { return printRemF(*this, op); })
        .Case<executor::BitwiseAndIOp, executor::BitwiseOrIOp,
              executor::BitwiseXOrIOp, executor::ShiftLeftIOp,
              executor::ShiftRightArithmeticIOp, executor::ShiftRightLogicalIOp,
              executor::SDivIOp, executor::DivFOp>([&](auto op) {
          return printExecutorBinarySpecialFunctionOperation(*this, op);
        })
        .Case<executor::ICmpOp>(
            [&](auto op) { return printExecutorICmpOp(*this, op); })
        .Case<executor::FCmpOp>(
            [&](executor::FCmpOp op) { return printExecutorFCmpOp(*this, op); })
        .Case<executor::BitcastOp>(
            [&](auto op) { return printBitcastOp(*this, op); })
        .Case<executor::SelectOp>(
            [&](auto op) { return printSelectOp(*this, op); })
        .Case<executor::CreateTableOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::ExtractTableValueOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::DynamicExtractTableValueOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::InsertTableValueOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::PrintOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::MemcpyOp>(
            [&](auto op) { return printMemCpyOp(*this, op); })
        .Case<executor::StrLiteralOp, executor::ConstantResourceLoadOp,
              executor::GetGlobalOp, executor::SetGlobalOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::AssertOp>(
            [&](executor::AssertOp op) { return printOperation(*this, op); })
        .Case<executor::CoroAwaitOp, executor::CoroYieldOp,
              executor::CoroCreateOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::PtrToIntOp>([&](auto op) {
          return printPtrToIntOp(*this, op, moduleDataLayout);
        })
        .Case<executor::IntToPtrOp>([&](auto op) {
          return printIntToPtrOp(*this, op, moduleDataLayout);
        })
        .Default([&](Operation *) {
          return op.emitOpError("unable to find printer for op");
        });
  }
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      // Builtin ops.
      .Case<ModuleOp>([&](ModuleOp op) { return emitModule(*op); })
      // Func ops.
      .Case<func::CallOp, func::FuncOp, func::ReturnOp, func::CallIndirectOp>(
          [&](auto op) { return printOperation(*this, op); })
      // CF ops
      .Case<cf::BranchOp, cf::CondBranchOp>(
          [&](auto op) { return printControlFlowOp(*this, op); })
      .Default([&](Operation *) {
        return op.emitOpError("unable to find printer for op");
      });
}

LogicalResult mlir::translateToLua(Operation *op, raw_ostream &os) {
  LuaEmitter luaEmitter(op->getContext(), os, DataLayout::closest(op));
  if (isa<FunctionOpInterface>(op))
    return luaEmitter.emitOperation(*op);
  if (isModuleLike(*op))
    return luaEmitter.emitModule(*op);

  return emitError(op->getLoc())
         << "expected FunctionOpInterface or Module-like operation";
}

void mlir::registerToLuaTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-lua", "translate from MLIR to Lua",
      [](Operation *op, llvm::raw_ostream &output) {
        return mlir::translateToLua(op, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<func::FuncDialect,
                        cf::ControlFlowDialect,
                        executor::ExecutorDialect,
                        DLTIDialect>();
        // clang-format on
      });
}
