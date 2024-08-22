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
  explicit LuaEmitter(MLIRContext *ctx, raw_ostream &os);

  LogicalResult emitOperation(Operation &op);

  LogicalResult emitBlock(Block &block);

  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// RAII helper function to manage entering/exiting scopes.
  struct Scope {
    explicit Scope(LuaEmitter &emitter, unsigned additionalLocals = 0)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
      emitter.localsInScopeCount.push(emitter.localsInScopeCount.top() +
                                      additionalLocals);
    }
    ~Scope() {
      emitter.labelInScopeCount.pop();
      emitter.localsInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
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
  StringRef getOrCreateVariableName(Value val, StringRef prefix = "v",
                                    bool isLocal = false);

  StringRef getVariableName(Value val);

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

  /// Map from block to name of lua label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  unsigned globalsInScope = 0;
  std::stack<int64_t> localsInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  MLIRContext *ctx;
  raw_indented_ostream os;
};
} // namespace

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
    emitter << emitter.getOrCreateVariableName(blockArg, "blockArg") << " = "
            << emitter.getVariableName(operand) << ";\n";
  }
  emitter << "goto " << emitter.getOrCreateLabel(*op.getDest()) << ";\n";
  return success();
}

static LogicalResult printControlFlowOp(LuaEmitter &emitter,
                                        cf::CondBranchOp op) {
  auto condName = emitter.getVariableName(op.getCondition());
  emitter << "if (" << condName << " == 1) or (" << condName << " == true)"
          << " then\n";

  auto emitBranch = [&](Block *destBlock, ValueRange operands) {
    emitter.getStream().indent();
    // Declare non-local args to hold block arguments.
    for (auto [operand, blockArg] :
         llvm::zip(operands, destBlock->getArguments())) {
      emitter << emitter.getOrCreateVariableName(blockArg, "blockArg") << " = "
              << emitter.getVariableName(operand) << ";\n";
    }
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

/// For a given function, the number of additional locals required is the
/// function arguments plus the number of arguments given to function calls.
static unsigned getLocalRegisterRequirements(func::FuncOp op) {
  unsigned numLocals = op.getNumArguments();
  op.walk([&](func::CallOp op) { numLocals += op->getNumOperands(); });
  op.walk([&](executor::CallOp op) { numLocals += op->getNumOperands(); });
  return numLocals;
}

static LogicalResult printOperation(LuaEmitter &emitter, func::FuncOp op) {
  // We don't actually need to translate pure declarations for external C
  // functions.
  if (op.isDeclaration()) {
    emitter << "__check_for_function(\"" << op.getName() << "\");\n";
    return success();
  }

  emitter << "function " << op.getName() << " ";
  LuaEmitter::Scope scope(
      emitter, /*additionalLocals=*/getLocalRegisterRequirements(op));
  emitter << "(";
  llvm::interleaveComma(op.getArguments(), emitter.getStream(),
                        [&](BlockArgument &arg) {
                          emitter << emitter.getOrCreateVariableName(arg);
                        });
  emitter << ")\n";
  emitter.getStream().indent();

  for (Block &block : op.getBody()) {
    if (failed(emitter.emitBlock(block)))
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

  if (op.getType().isFloat8E4M3FN())
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
  emitter << " " << emitter.getOrCreateVariableName(rhs) << ";\n";
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
      emitter << emitter.getOrCreateVariableName(v);
    });
    emitter << "));\n";
    return success();
  }
  emitter << "print(";
  llvm::interleaveComma(op->getOperands(), emitter, [&](Value v) {
    emitter << emitter.getOrCreateVariableName(v);
  });
  emitter << ");\n";
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

/// Translate `executor.load_constant_resource`.
static LogicalResult printOperation(LuaEmitter &emitter,
                                    executor::ConstantResourceLoadOp op) {
  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << op.getName() << ";\n";
  return success();
}

/// Translate `executor.call`.
static LogicalResult printOperation(LuaEmitter &emitter, executor::CallOp op) {
  std::optional<ArrayAttr> immArgs = op.getImmediateArgs();

  if (op->getNumResults() > 0) {
    if (failed(emitter.emitAssignPrefix(op.getOperation())))
      return failure();
  }
  emitter << op.getCallee() << "(";

  if (immArgs) {
    bool error = false;
    llvm::interleaveComma(*immArgs, emitter.getStream(), [&](Attribute attr) {
      if (failed(emitAttribute(emitter.getStream(), op.getLoc(), attr)))
        error = true;
    });
    if (error)
      return failure();

    if (op.getNumOperands() > 0 && immArgs->size() > 0)
      emitter << ", ";
  }

  llvm::interleaveComma(op->getOperands(), emitter.getStream(), [&](Value v) {
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

LuaEmitter::LuaEmitter(MLIRContext *ctx, raw_ostream &os) : ctx(ctx), os(os) {
  localsInScopeCount.push(0);
  labelInScopeCount.push(0);
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
  return valueMapper.count(val) != 0;
}

StringRef LuaEmitter::getOrCreateVariableName(Value val, StringRef prefix,
                                              bool isLocal) {
  if (!isValueInScope(val)) {
    std::string valueName =
        isLocal ? llvm::formatv("localV{0}", ++localsInScopeCount.top()).str()
                : llvm::formatv("v{0}", ++globalsInScope).str();
    valueMapper.insert(val, valueName);
  }
  return *valueMapper.begin(val);
}

StringRef LuaEmitter::getVariableName(Value val) {
  assert(isValueInScope(val) && "expected value to be in scope");
  return *valueMapper.begin(val);
}

LogicalResult LuaEmitter::emitAssignPrefix(Operation *op) {
  // In Lua, a variable can be declared local as long is it is not used in
  // another block. Lua locals go out of scope when the block terminates. Since
  // we translate Blocks 1-1 with Lua blocks, just check if it is used in
  // another block (that is not a child).
  bool localVar = !op->isUsedOutsideOfBlock(op->getBlock());
  if (localVar && localsInScopeCount.top() + op->getNumResults() < 200) {
    os << "local ";
    localsInScopeCount.top() += op->getNumResults();
  }
  llvm::interleaveComma(op->getResults(), os,
                        [this](Value v) { os << getOrCreateVariableName(v); });
  os << " = ";
  return success();
}

LogicalResult LuaEmitter::emitBlock(Block &block) {
  // In Lua, you cannot "jump/goto" a label that is in the scope of a local
  // variable. Instead, we make all blocks have their own do...end scope and
  // only declare "local" the variables that are used internal to only one
  // block. There are other potential strategies that could be explored, such as
  // having `do...end` scope enclose a region instead of a block.
  os << "::" << getOrCreateLabel(block) << ":: do\n";
  os.indent();
  for (Operation &op : block.getOperations()) {
    if (failed(emitOperation(op)))
      return failure();
  }
  os.unindent();
  os << "end\n";
  return success();
}

/// Emit a module. This is the root of the translation.
static LogicalResult printOperation(LuaEmitter &emitter, ModuleOp op) {
  LuaEmitter::Scope scope(emitter);
  for (Operation &op : op.getBody()->getOperations()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

LogicalResult LuaEmitter::emitOperation(Operation &op) {
  // Global/const resource declarations don't need to get emitted.
  if (isa<executor::ConstantResourceOp, executor::GlobalOp>(op))
    return success();

  if (isa<executor::ExecutorOpInterface>(op)) {
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
        .Case<executor::StrLiteralOp, executor::ConstantResourceLoadOp,
              executor::GetGlobalOp, executor::SetGlobalOp>(
            [&](auto op) { return printOperation(*this, op); })
        .Case<executor::AssertOp>(
            [&](executor::AssertOp op) { return printOperation(*this, op); })
        .Default([&](Operation *) {
          return op.emitOpError("unable to find printer for op");
        });
  }
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      // Builtin ops.
      .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
      // Func ops.
      .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
          [&](auto op) { return printOperation(*this, op); })
      // CF ops
      .Case<cf::BranchOp, cf::CondBranchOp>(
          [&](auto op) { return printControlFlowOp(*this, op); })
      .Default([&](Operation *) {
        return op.emitOpError("unable to find printer for op");
      });
}

LogicalResult mlir::translateToLua(Operation *op, raw_ostream &os) {
  LuaEmitter luaEmitter(op->getContext(), os);
  return luaEmitter.emitOperation(*op);
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
