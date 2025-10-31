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
#include "mlir/Debug/Counter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
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

static cl::OptionCategory OptCat{"MLIR-TensorRT Options"};

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input MLIR file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
    outputPath("o", cl::desc("<output directory or file name>"), cl::init("."),
               cl::value_desc("directory"), cl::cat(OptCat),
               cl::sub(cl::SubCommand::getAll()));

static cl::opt<std::string>
    pipelineOptions("opts",
                    llvm::cl::desc("options for the compilation pipeline"),
                    cl::cat(OptCat));

static cl::opt<std::string>
    inputKind("input", llvm::cl::desc("the kind of input to compile"),
              cl::cat(OptCat), cl::init("stablehlo"));

static cl::opt<bool> pipelineHelp(
    "pipeline-help",
    llvm::cl::desc("print the compilation pipeline options and exit"),
    cl::cat(OptCat));

static cl::opt<bool>
    outputMLIR("mlir",
               cl::desc("output the MLIR instead of performing translation"),
               cl::cat(OptCat), cl::init(false));

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

/// Returns true if the output path likely refers to a file or stdout.
static bool outputIsFile(StringRef path) {
  return !llvm::sys::fs::is_directory(path) &&
         (path == "-" || llvm::sys::fs::is_symlink_file(path) ||
          llvm::sys::fs::is_regular_file(path) ||
          !llvm::sys::path::extension(path).empty());
}

static LogicalResult runCompilation(CompilerClient &client, StringRef taskName,
                                    mlir::ModuleOp module,
                                    llvm::StringRef pipelineOptions) {
  llvm::StringRef artifactsDirectoryOverride =
      !outputIsFile(outputPath) ? llvm::StringRef(outputPath)
                                : llvm::sys::path::parent_path(outputPath);
  std::optional<llvm::StringRef> outputExtensionOverride =
      outputMLIR ? std::optional<llvm::StringRef>(".mlir") : std::nullopt;

  StatusOr<CompilationTaskBase *> task = client.getCompilationTask(
      taskName, {pipelineOptions}, artifactsDirectoryOverride);
  if (!task.isOk()) {
    llvm::errs() << task.getStatus() << "\n";
    return failure();
  }

  if (failed((*task)->run(module)))
    return failure();

  std::string errorMessage;
  auto output = (*task)->openOutputFile(outputPath, errorMessage,
                                        outputExtensionOverride);
  if (!output) {
    llvm::errs() << "failed to open output file: " << errorMessage << "\n";
    return failure();
  }

  if (outputMLIR) {
    module->print(output->os());
  } else {
    if (failed((*task)->translateToTargetFormat(module, output->os())))
      return failure();
  }
  output->keep();
  return success();
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader =
      "MLIR-TensorRT Compiler\nAvailable compilation tasks: ";
  {
    llvm::raw_string_ostream os(helpHeader);
    llvm::interleaveComma(
        llvm::ArrayRef<llvm::StringRef>{"stablehlo-to-executable",
                                        "tensorrt-to-executable"},
        os);
  }

  cl::ParseCommandLineOptions(argc, argv, helpHeader);

  std::string taskName = llvm::StringSwitch<std::string>(inputKind)
                             .CaseLower("tensorrt", "tensorrt-to-executable")
                             .CaseLower("stablehlo", "stablehlo-to-executable")
                             .Default("stablehlo-to-executable");

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  mtrt::compiler::registerAllDialects(registry);
  mtrt::compiler::registerAllExtensions(registry);

  context.appendDialectRegistry(registry);

  if (pipelineHelp) {
    printCompilationTaskHelpInfo(&context, taskName);
    exit(0);
  }

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

  return succeeded(runCompilation(**client, taskName, *module, pipelineOptions))
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
