//===- ExecuteFunctionWithLuaBackendTests.cpp -----------------------------===//
//
// Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Unit tests for testing the parsing of sol::protected_function_results
///
//===----------------------------------------------------------------------===//

#include "mlir-executor/InitAllDialects.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "gtest/gtest.h"

using namespace mtrt;
using namespace mtrt;

namespace {

class TestRuntime : public ::testing::Test {
protected:
  void SetUp() override {
    mlir::executor::registerAllRequiredDialects(registry);
    mtrt::registerLuaRuntimeExtensions();
    context = std::make_unique<mlir::MLIRContext>(registry);
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseModule(const std::string &moduleStr) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        moduleStr, mlir::ParserConfig(context.get()));
  }

  StatusOr<Ref<RuntimeClient>> createRuntimeClient() {
    return RuntimeClient::create();
  }

  StatusOr<std::unique_ptr<LuaRuntimeSession>>
  createLuaRuntimeSession(Ref<RuntimeClient> client,
                          const std::unique_ptr<mtrt::Executable> &executable) {
    RuntimeSessionOptions options;
    options.enableFeatures({"core"});
    return LuaRuntimeSession::create(client, options, executable->getView(),
                                     {});
  }

  void assertScalarValuesEqual(const ScalarValue *result,
                               const ScalarValue *reference) {
    ASSERT_TRUE(result && "Result value is not a ScalarValue");
    ASSERT_TRUE(reference && "Reference value is not a ScalarValue");
    ASSERT_EQ(result->getType(), reference->getType())
        << "ScalarValue types don't match";

    switch (result->getType()) {
    case ScalarTypeCode::i32:
      ASSERT_EQ(result->get<int32_t>(), reference->get<int32_t>())
          << "Int32 values don't match";
      break;
    default:
      FAIL() << "Incorrect ScalarType";
    }
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(TestRuntime, TestRuntimeExecution) {
  const char *moduleStr = R"(
    module @executor {
      func.func @main(%arg0: i32, %arg1: i32) -> (i32, i32, i32, i32) attributes { executor.function_metadata = #executor.func_meta<[i32, i32], [i32, i32, i32, i32], num_output_args = 0>} {
        %c1 = executor.constant 1 : i32
        %c2 = executor.constant 2 : i32
        %0 = executor.addi %arg0, %c1 : i32
        %1 = executor.addi %arg1, %c2 : i32
        return %0, %1, %c1, %c2: i32, i32, i32, i32
      }
      func.func private @executor_init_globals() {
        return
      }
    }
  )";

  auto module = parseModule(moduleStr);
  ASSERT_TRUE(module);

  auto exeStorage = mlir::translateToRuntimeExecutable(*module);
  ASSERT_TRUE(mlir::succeeded(exeStorage));

  auto executable = std::make_unique<mtrt::Executable>(std::move(*exeStorage));

  auto client = createRuntimeClient();
  ASSERT_TRUE(client.isOk()) << client.getString();

  auto session = createLuaRuntimeSession(*client, executable);
  ASSERT_TRUE(session.isOk()) << session.getString();

  std::vector<ScalarValue> scalarValues;
  scalarValues.reserve(4);
  scalarValues.emplace_back(1, ScalarTypeCode::i32);
  scalarValues.emplace_back(2, ScalarTypeCode::i32);
  scalarValues.emplace_back(3, ScalarTypeCode::i32);
  scalarValues.emplace_back(4, ScalarTypeCode::i32);

  llvm::SmallVector<RuntimeValue *> reference = {
      &scalarValues[2], &scalarValues[3], &scalarValues[0], &scalarValues[1]};
  llvm::SmallVector<RuntimeValue *> inputArgs = {&scalarValues[1],
                                                 &scalarValues[1]};

  auto results = (*session)->executeFunction("main", inputArgs, {},
                                             /*stream=*/nullptr);
  ASSERT_TRUE(results.isOk()) << results.getStatus().getString();

  ASSERT_EQ(results->size(), reference.size()) << "Vector sizes don't match";

  for (size_t i = 0; i < results->size(); ++i) {
    assertScalarValuesEqual(llvm::dyn_cast<ScalarValue>((*results)[i].get()),
                            llvm::dyn_cast<ScalarValue>(reference[i]));
  }
}

} // namespace
