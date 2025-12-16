//===- Compiler.cpp -------------------------------------------------------===//
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
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt-common/Utils/OwningModuleRef.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-pjrt/Client.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.h"
#include "mlir-tensorrt/Compiler/InitAllDialects.h"
#include "mlir-tensorrt/Compiler/InitAllPasses.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include <string_view>

#define DEBUG_TYPE "pjrt-compiler"

using namespace mtrt;
using namespace mtrt::pjrt;

namespace cl = llvm::cl;

namespace {
struct CompilerFlagOptions {
  cl::opt<bool> preferCodegen{"pjrt-experimental-prefer-codegen",
                              llvm::cl::init(false),
                              llvm::cl::desc("prefer codegen over TensorRT")};

  cl::opt<bool> disallowHostTensorsInTensorRTClusters{
      "pjrt-disallow-host-tensors-in-tensorrt-clusters", llvm::cl::init(true),
      llvm::cl::desc("Disallow host tensor calculations in TensorRT clusters")};

  cl::list<std::string> pipelineOptions{
      "mlir-compile-opts",
      llvm::cl::desc("the options to use for the MLIR compilation pipeline"),
      llvm::cl::CommaSeparated};

  cl::opt<unsigned> pjrtOptLevel{
      "mtrt-pjrt-opt-level", llvm::cl::init(0),
      llvm::cl::desc("The optimization level to use for the MLIR-TensorRT "
                     "compilation pipeline. Takes values in range [0, 5].")};
};
} // namespace

static llvm::ManagedStatic<CompilerFlagOptions> clOptionsConfig;

/// Register options that can be parsed from MLIR_TRT_FLAGS.
void mtrt::pjrt::registerPJRTCompilerCLOptions() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tensorrt::registerTensorRTTranslationCLOpts();

  mtrt::registerLuaRuntimeExtensions();
  mtrt::compiler::registerAllPasses();
  // Ensure options get registered.
  *clOptionsConfig;
}

StatusOr<std::unique_ptr<Compiler>>
Compiler::create(llvm::ThreadPoolInterface &threadPool) {
  auto context = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);
  context->setThreadPool(threadPool);
  mlir::DialectRegistry registry;
  mtrt::compiler::registerAllDialects(registry);
  mtrt::compiler::registerAllExtensions(registry);
  context->appendDialectRegistry(registry);

  StatusOr<std::unique_ptr<compiler::CompilerClient>> compilerClient =
      compiler::CompilerClient::create(context.get());
  if (!compilerClient.isOk())
    return compilerClient.getStatus();

  return std::unique_ptr<Compiler>(
      new Compiler(std::move(context), std::move(*compilerClient)));
}

/// Parse the given string as an MLIR source file containing a top-level
/// ModuleOp.
static StatusOr<mlir::OwningModuleRef> parseSource(mlir::MLIRContext *context,
                                                   std::string_view code) {
  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();
  // Set scope exit callback to re-enable threading if applicable.
  auto scopeExit = llvm::make_scope_exit(
      [&]() { context->enableMultithreading(wasThreadingEnabled); });
  mlir::ParserConfig config(context, /*verifyAfterParse=*/false);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()), config);
  if (!module)
    return getStatusWithMsg(StatusCode::InvalidArgument,
                            "failed to deserialize StableHLO bytecode");

  // Deserialize VHLO
  mlir::PassManager pm((*module)->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(*module)))
    return mtrt::getStatusWithMsg(
        StatusCode::InvalidArgument,
        "failed to run StableHLO deserialization pipeline");

  return mlir::OwningModuleRef(module.release());
}

/// Returns the set of options which should be passed to the compilation task
/// construction API.
static llvm::SmallVector<llvm::StringRef> getCompilationTaskOptions() {
  llvm::SmallVector<llvm::StringRef> options;
  for (const auto &opt : clOptionsConfig->pipelineOptions)
    options.push_back(llvm::StringRef(opt));
  options.push_back("--device-infer-from-host");
  options.push_back("--tensorrt-force-default-slice-in-bounds");
  options.push_back("--use-global-tensorrt-translation-flags");
  options.push_back("--abi-version=1");
  // We use a large unroll factor for any optimization level > 0.
  if (clOptionsConfig->pjrtOptLevel > 0)
    options.push_back("--unroll-threshold=9223372036854775807");
  return options;
}

/// Return true if any of the `options` contains the given substring. Useful for
/// detecting the precense of a flag regardless of whether it was prefixed with
/// `--` or not.
static bool
compilationTasksOptionsContains(llvm::ArrayRef<llvm::StringRef> options,
                                llvm::StringRef partialOptionText) {
  return llvm::any_of(options, [&](llvm::StringRef option) {
    return option.contains_insensitive(partialOptionText);
  });
}

StatusOr<std::unique_ptr<PJRTExecutable>>
Compiler::compileMlirModule(std::string_view mlirIR,
                            const xla::CompileOptionsProto &compileOptions) {
  // Currently we use a single MLIRContext, so only one program maybe compiled
  // at a time.
  std::lock_guard<std::mutex> lock(compileMutex);

  StatusOr<mlir::OwningModuleRef> module = parseSource(getContext(), mlirIR);
  if (!module.isOk())
    return module.getStatus();

  static std::atomic<unsigned> moduleCount = 0;
  std::optional<llvm::StringRef> moduleName = (**module).getSymName();
  std::string newName = llvm::formatv("{0}_{1}", moduleCount++,
                                      moduleName ? *moduleName : "no-name");
  (**module).setSymName(newName);

  // Set the clustering options on the module.
  mlir::MLIRContext *ctx = (*module)->getContext();
  int64_t codegenBenefit = clOptionsConfig->preferCodegen ? 99 : 2;
  llvm::SmallVector<mlir::Attribute> clusterKinds;
  llvm::SmallVector<llvm::StringRef> options = getCompilationTaskOptions();

  // We must load the Plan dialect in order to create backend attributes.
  ctx->loadDialect<mlir::plan::PlanDialect>();

  if (!compilationTasksOptionsContains(options, "disable-tensorrt-extension"))
    clusterKinds.push_back(mlir::plan::TensorRTBackendAttr::get(
        ctx, clOptionsConfig->disallowHostTensorsInTensorRTClusters, 3,
        /*tensorrt_major_version=*/NV_TENSORRT_MAJOR,
        /*prefer_destination_style_calling_convention=*/true));

  clusterKinds.push_back(
      mlir::plan::KernelBackendAttr::get(ctx, codegenBenefit));
  clusterKinds.push_back(mlir::plan::HostBackendAttr::get(ctx, 1));

  (**module).getOperation()->setAttr(mlir::plan::PlanDialect::kBackendsAttrName,
                                     mlir::ArrayAttr::get(ctx, clusterKinds));

  // Create the CompilationTask or get the cached one.
  mtrt::StatusOr<mtrt::compiler::CompilationTaskBase *> pm =
      client->getCompilationTask(
          mtrt::compiler::StablehloToExecutableTask::getName(), options);
  if (!pm.isOk())
    return pm.getStatus();

  // Run the pass manager.
  if (failed((*pm)->run(**module)))
    return mtrt::getInternalErrorStatus(
        "failed to run compilation on module with symbol name: {0}",
        (**module).getName() ? (**module).getName() : "no-symbol-name");

  // Translate to Runtime Executable
  mlir::FailureOr<std::unique_ptr<mtrt::ExecutableStorage>> exeStorage =
      mlir::translateToRuntimeExecutable(**module);
  if (failed(exeStorage))
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to translate compiled MLIR module to a "
                            "MLIR-TensorRT runtime Executable");
  auto exe = std::make_unique<mtrt::Executable>(std::move(*exeStorage));

  // Check for the main function.
  llvm::SmallVector<mtrt::FunctionView> functions = exe->getFunctions();
  if (llvm::find_if(functions, [](const mtrt::FunctionView &func) {
        return func.getName() == "main";
      }) == functions.end())
    return getStatusWithMsg(StatusCode::InternalError,
                            "executable does not have a 'main' function");

  if (exe->getProcessorGridShape().size() != 2)
    return mtrt::getInternalErrorStatus(
        "expected compiled program to have a process grid shape "
        "attribute of rank 2, but got process_grid_shape={0:$[, ]}",
        exe->getProcessorGridShape());

  return std::make_unique<PJRTExecutable>(std::move(*exe));
}
