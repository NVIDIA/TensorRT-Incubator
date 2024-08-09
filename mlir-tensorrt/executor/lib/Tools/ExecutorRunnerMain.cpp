//===- ExecutorRunner.cpp -------------------------------------------------===//
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
#include "mlir-executor/Tools/ExecutorRunnerMain.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Support/MPI.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::executor;
using namespace mlirtrt::runtime;
using namespace mlirtrt;

namespace cl = llvm::cl;

#ifdef MLIR_EXECUTOR_ENABLE_NCCL
static llvm::ManagedStatic<std::unique_ptr<MPIManager>> mpiManager;
#endif // MLIR_EXECUTOR_ENABLE_NCCL

static Status maybeInitializeMpi() {
#ifdef MLIR_EXECUTOR_ENABLE_NCCL

  StatusOr<std::unique_ptr<MPIManager>> mgr = MPIManager::create();
  if (!mgr.isOk())
    return mgr.getStatus();

  *mpiManager = std::move(*mgr);

  return Status::getOk();
#else
  return mlirtrt::Status::getOk();
#endif // MLIR_EXECUTOR_ENABLE_NCCL
}

/// Try to infer the input type from the filename, otherwise return failure.
static FailureOr<InputType> inferInputType(StringRef filename) {
  if (filename.ends_with(".lua"))
    return Lua;
  if (filename.ends_with(".edb"))
    return ExecutorRuntimeExecutable;
  return failure();
}

namespace {
struct Options {
  cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                     cl::init("-")};
  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-")};

  cl::opt<std::string> splitInputFile{
      "split-input-file", cl::ValueOptional,
      cl::desc("Split the input file into pieces and "
               "process each chunk independently"),
      cl::callback([&](const std::string &str) {
        if (str.empty())
          splitInputFile.setValue(kDefaultSplitMarker);
      }),
      cl::init("")};

  cl::opt<bool> dumpFunctionSignature{"dump-function-signature",
                                      cl::desc("Dump function signature"),
                                      cl::init(false)};

  cl::opt<enum InputType> inputType{
      "input-type", cl::init(Unspecified),
      cl::desc("override how to interpret the input file"),
      cl::values(clEnumValN(Lua, "lua", "interpret the input as Lua code")),
      cl::values(clEnumValN(ExecutorRuntimeExecutable, "rtexe",
                            "load the input file as an Executor executable"))};
};
} // namespace

LogicalResult
executor::ExecutorRunnerMain(int argc, char **argv,
                             std::function<void()> postInitCallback) {
  llvm::InitLLVM initLLVM(argc, argv);

  Status mpiStatus = maybeInitializeMpi();

  if (!mpiStatus.isOk()) {
    llvm::errs() << "failed to initialize MPI: " << mpiStatus.getString()
                 << "\n";
    return failure();
  }

  // Register and parse CLI args.
  Options options;
  cl::ParseCommandLineOptions(argc, argv, "MLIR-TensorRT Runtime Interpreter");

  if (postInitCallback)
    postInitCallback();

  // We'll use MLIR just for emitting diagnostics.
  MLIRContext context;
  Location loc = UnknownLoc::get(&context);

  // Try to deduce the input type from the filename.
  if (options.inputType == Unspecified) {
    FailureOr<InputType> deducedInputType =
        inferInputType(options.inputFilename);
    if (failed(deducedInputType))
      return emitError(UnknownLoc::get(&context))
             << "failed to deduce input type from the filename; use "
                "--input-type "
                "to specify how to interpret the input file";
    options.inputType = *deducedInputType;
  }

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input =
      openInputFile(options.inputFilename, &errorMessage);
  if (!input)
    return emitError(loc) << "failed to open input buffer: " << errorMessage;

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(options.outputFilename, &errorMessage);
  if (!output)
    return emitError(loc) << "failed to open output buffer: " << errorMessage;

  // Initialize the CUDA driver.
  int device = 0;
  // Context must be created for the correct device we will be using in this
  // process. Currently, assume direct mapping from local rank -> device.
  // TODO: Context creation for multi-gpu should be improved somehow.
  if (const char *rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    device = std::stoi(rank_str);
  cudaError_t result = cudaSetDevice(device);
  if (result != cudaSuccess)
    return emitError(loc) << "failed to initialize CUDA device: "
                          << cudaGetErrorString(result);

  // Make a single no-op CUDA call in order to force CUDART to setup the context
  // for the current thread. It's possible that the first CUDA call when loading
  // globals will be a CUDA driver call, which doesn't do any automatic context
  // initialization.
  result = cudaFree(0);
  if (result != cudaSuccess)
    return emitError(loc) << "cudaFree failed: " << cudaGetErrorString(result);

  // Read the buffer as a Lua script and execute.

  if (options.inputType == Lua) {
    assert(!options.dumpFunctionSignature &&
           "Can not dump function signature for Lua input type.");
    mlirtrt::StatusOr<int64_t> result =
        mlirtrt::runtime::runExecutorLuaScript(input->getBuffer());
    if (!result.isOk())
      return emitError(UnknownLoc::get(&context)) << result.getString();
    return success(*result == 0);
  }

  assert(options.inputType == ExecutorRuntimeExecutable &&
         "expected executor executable input type");

  mlirtrt::StatusOr<std::unique_ptr<mlirtrt::runtime::Executable>> executable =
      mlirtrt::runtime::Executable::loadFromBuffer(std::move(input));
  if (!executable.isOk())
    return emitError(UnknownLoc::get(&context))
           << "failed to load executable from buffer: "
           << executable.getString();

  if (options.dumpFunctionSignature) {
    for (unsigned i = 0; i < executable->get()->getNumFunctions(); ++i) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      mlirtrt::runtime::print(ss,
                              executable->get()->getFunction(i).getSignature());
      llvm::outs() << ss.str() << "\n";
    }
    return success();
  }

  mlirtrt::StatusOr<int64_t> executionResult =
      mlirtrt::runtime::runExecutorExecutable(std::move(*executable));
  if (!executionResult.isOk())
    return emitError(UnknownLoc::get(&context))
           << "failed to load and run executable: "
           << executionResult.getString();

  return success(*executionResult == 0);
}