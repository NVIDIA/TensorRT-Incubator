//===- mlir-tensorrt-compiler.cpp -----------------------------------------===//
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
/// Entrypoint for the 'mlir-tensorrt-compiler' tool.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/InitAllDialects.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir/Debug/Counter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace llvm;
using namespace mtrt;
using namespace mtrt::compiler;

static cl::OptionCategory OptCat{"mlir-tensorrt-compiler Tool Options"};

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input MLIR file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
    outputPath("o", cl::desc("<output directory or file name>"), cl::init("."),
               cl::value_desc("directory"), cl::cat(OptCat),
               cl::sub(cl::SubCommand::getAll()));

static cl::opt<bool>
    outputMLIR("mlir",
               cl::desc("output the MLIR instead of performing translation"),
               cl::cat(OptCat), cl::init(false));

static cl::opt<std::string>
    printPipeline("print-pass-pipeline",
                  cl::desc("print the pass pipeline and exit"),
                  cl::cat(OptCat));

static cl::opt<std::string>
    dumpDirectory("dump-dir",
                  cl::desc("equivalent to --mlir-print-ir-after-all "
                           "--mlir-print-ir-tree-dir=<dir>"),
                  cl::cat(OptCat), cl::init(""));

static cl::opt<std::string> crashReproducerPath(
    "crash-repro",
    cl::desc("equivalent to --mlir-pass-pipeline-crash-reproducer=<file path> "
             "--mlir-pass-pipeline-local-reproducer -mlir-disable-threading"),
    cl::cat(OptCat), cl::init(""));

/// Parse the input MLIR file using the provided MLIRContext. The result is
/// returned in the `module`. Failure to parse results in a fatal error.
static void parseInputMLIR(llvm::SourceMgr &sourceMgr,
                           mlir::MLIRContext &context,
                           mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    exit(1);
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(2);
  }
}

static LogicalResult
runCompilation(CompilerClient &client, mlir::ModuleOp module,
               llvm::IntrusiveRefCntPtr<MainOptions> options) {
  std::optional<llvm::StringRef> outputExtensionOverride =
      outputMLIR ? std::optional<llvm::StringRef>(".mlir") : std::nullopt;

  auto pipeline = std::make_unique<Pipeline>(client.getContext(), options);
  if (printPipeline.getNumOccurrences() > 0) {
    llvm::errs() << "Loaded Extensions:\n";
    for (const auto &extension : pipeline->getOptions().getExtensions()) {
      llvm::errs() << "  " << extension.getKey() << "\n";
    }

    llvm::errs() << "Pass Pipeline:\n";
    // Prefer stable textual pipeline printing over PassManager::dump(), which
    // is intended for debugging and can change format between MLIR versions.
    pipeline->getPassManager().printAsTextualPipeline(
        llvm::errs(),
        /*pretty=*/printPipeline == "pretty");
    llvm::errs() << "\n";
    return success();
  }

  if (failed(pipeline->run(module)))
    return failure();

  std::string errorMessage;
  auto output = pipeline->openOutputFile(outputPath, errorMessage,
                                         outputExtensionOverride);
  if (!output) {
    llvm::errs() << "failed to open output file: " << errorMessage << "\n";
    return failure();
  }

  if (outputMLIR) {
    module->print(output->os());
  } else {
    if (failed(pipeline->translateToTargetFormat(module, output->os())))
      return failure();
  }
  output->keep();
  return success();
}

/// LLVM declares a large number of global CL options under the "General"
/// category. This function hides most of them from '--help' output while
/// keeping MLIR-TensorRT and MLIR core CL options visible.
static void hideUnrelatedOptions() {
  auto &top = llvm::cl::SubCommand::getTopLevel();
  for (auto &I : top.OptionsMap) {
    if ((I.second->Categories.empty() ||
         llvm::is_contained(I.second->Categories,
                            &llvm::cl::getGeneralCategory())) &&
        I.second->hasArgStr() && !I.second->ArgStr.starts_with("mlir")) {
      I.second->setHiddenFlag(llvm::cl::ReallyHidden);
    }
  }
}

/// Create the final list of command-line arguments. The LLVM CL infrastructure
/// does not supporting adding aliases that reference multiple options, so we
/// need to handle our `-dump-dir` alias option manually by modifying the
/// argument list before parsing.
static LogicalResult parseCommandlineArguments(int argc, char **argv) {
  llvm::SmallVector<const char *, 8> args;
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver saver(allocator);
  if (argc > 0)
    args.assign(argv, argv + argc);

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "MLIR-TensorRT Compiler")) {
    llvm::errs() << "failed to parse command line options";
    return failure();
  }

  if (!dumpDirectory.empty()) {
    // Create the directory. Upstream MLIR has an issue where
    // `--mlir-print-ir-tree-dir` will create the directory only if the parent
    // exists.
    if (std::error_code EC = llvm::sys::fs::create_directories(dumpDirectory)) {
      llvm::errs() << "failed to create dump directory: " << dumpDirectory
                   << ": " << EC.message() << "\n";
      return failure();
    }

    llvm::StringRef extraArg = saver.save(
        llvm::formatv("--mlir-print-ir-tree-dir={0}", dumpDirectory).str());
    args.push_back("--mlir-print-ir-after-all");
    // All StringSaver::save() returned strings are null-terminated.
    args.push_back(extraArg.data());
  }

  if (!crashReproducerPath.empty()) {
    if (llvm::sys::path::has_parent_path(crashReproducerPath)) {
      if (std::error_code EC = llvm::sys::fs::create_directories(
              llvm::sys::path::parent_path(crashReproducerPath))) {
        llvm::errs() << "failed to create crash-reproducer directory: "
                     << dumpDirectory << ": " << EC.message() << "\n";
        return failure();
      }
    }
    SmallVector<std::string> crashReproducerArgs = {
        "--mlir-pass-pipeline-crash-reproducer=" + crashReproducerPath,
        "--mlir-pass-pipeline-local-reproducer", "-mlir-disable-threading"};
    for (const auto &newArgs : crashReproducerArgs)
      args.push_back(saver.save(newArgs).data());
  }

  // Re-parse now that the options have been modified.
  llvm::cl::ResetAllOptionOccurrences();
  if (!llvm::cl::ParseCommandLineOptions(args.size(), args.data(),
                                         "MLIR-TensorRT Compiler")) {
    llvm::errs() << "failed to parse command line options";
    return failure();
  }
  return success();
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugCounter::registerCLOptions();
  mtrt::compiler::registerAllCompilerTaskExtensions();
  mlir::tensorrt::registerTensorRTTranslationCLOpts();

  ExtensionList extensions = compiler::getAllExtensions();
  auto pipelineOptions = llvm::makeIntrusiveRefCnt<MainOptions>(
      mlir::CLOptionScope::GlobalScope{}, std::move(extensions));

  hideUnrelatedOptions();

  if (failed(parseCommandlineArguments(argc, argv)))
    return EXIT_FAILURE;

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  mtrt::compiler::registerAllDialects(registry);
  mtrt::compiler::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  // Open and parse input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  parseInputMLIR(sourceMgr, context, module);

  StatusOr<std::unique_ptr<mtrt::compiler::CompilerClient>> client =
      mtrt::compiler::CompilerClient::create(&context);
  if (!client.isOk()) {
    llvm::errs() << "[error] failed to create compiler client: "
                 << client.getStatus() << "\n";
    return 1;
  }

  return succeeded(runCompilation(**client, *module, pipelineOptions))
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
