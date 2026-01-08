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

#ifdef MLIR_TRT_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif // MLIR_TRT_ENABLE_CUDA

using namespace mlir;
using namespace mlir::executor;
using namespace mtrt;
using namespace mtrt;

namespace cl = llvm::cl;

#ifdef MLIR_TRT_ENABLE_NCCL
static llvm::ManagedStatic<std::unique_ptr<MPIManager>> mpiManager;
#endif // MLIR_TRT_ENABLE_NCCL

static Status maybeInitializeMpi() {
#ifdef MLIR_TRT_ENABLE_NCCL

  StatusOr<std::unique_ptr<MPIManager>> mgr = MPIManager::create();
  if (!mgr.isOk())
    return mgr.getStatus();

  *mpiManager = std::move(*mgr);

  return Status::getOk();
#else
  return mtrt::Status::getOk();
#endif // MLIR_TRT_ENABLE_NCCL
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

  cl::opt<bool> dumpDataSegments{
      "dump-data-segments",
      cl::desc("Dump global data segments with constant initializers"),
      cl::init(false)};

  cl::opt<std::string> outputSplitMarker{
      "output-split-marker",
      llvm::cl::desc("Split marker to use for merging the ouput"),
      llvm::cl::init("")};

  cl::opt<enum InputType> inputType{
      "input-type", cl::init(Unspecified),
      cl::desc("override how to interpret the input file"),
      cl::values(clEnumValN(Lua, "lua", "interpret the input as Lua code")),
      cl::values(clEnumValN(ExecutorRuntimeExecutable, "rtexe",
                            "load the input file as an Executor executable"))};

  cl::list<std::string> features{
      "features", llvm::cl::list_init<std::string>({"core"}),
      cl::CommaSeparated, cl::desc("runtime features/modules to enable")};
};
} // namespace

static LogicalResult initializeCudaRuntime() {
#ifdef MLIR_TRT_ENABLE_CUDA
  int device = 0;
  // Context must be created for the correct device we will be using in this
  // process. Currently, assume direct mapping from local rank -> device.
  // TODO: Context creation for multi-gpu should be improved somehow.
  if (const char *rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    device = std::stoi(rank_str);

  cudaError_t result = cudaSetDevice(device);
  if (result != cudaSuccess) {
    llvm::errs() << "failed to initialize CUDA device: "
                 << cudaGetErrorString(result);
    return failure();
  }

  // Make a single no-op CUDA call in order to force CUDART to setup the context
  // for the current thread. It's possible that the first CUDA call when loading
  // globals will be a CUDA driver call, which doesn't do any automatic context
  // initialization.
  result = cudaFree(0);
  if (result != cudaSuccess) {
    llvm::errs() << "cudaFree failed: " << cudaGetErrorString(result);
    return failure();
  }
  return success();
#else
  llvm::errs() << "runtime was not built with CUDA support\n";
  return failure();
#endif
}

static StatusOr<RuntimeSessionOptions>
getRuntimeSessionOptions(const Options &options,
                         ArrayRef<std::string> features) {
#ifdef MLIR_TRT_ENABLE_NCCL
  if (llvm::is_contained(features, "nccl")) {
    StatusOr<RuntimeSessionOptions> opts =
        RuntimeSessionOptions::createUsingSingleHostMpi();
    if (!opts.isOk())
      return opts.getStatus();
    opts->enableFeatures(features);
    return opts;
  }
#endif

  auto opts = RuntimeSessionOptions::getSPMDOptions();
  opts.enableFeatures(features);
  return opts;
}

LogicalResult executor::ExecutorRunnerMain(
    int argc, char **argv, std::function<void()> postInitCallback,
    mtrt::LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs) {
  llvm::InitLLVM initLLVM(argc, argv);

  // Register and parse CLI args.
  Options options;
  cl::ParseCommandLineOptions(argc, argv, "MLIR-TensorRT Runtime Interpreter");

  if (!options.dumpFunctionSignature) {
    if (llvm::is_contained(options.features, "nccl")) {
      Status mpiStatus = maybeInitializeMpi();
      if (!mpiStatus.isOk()) {
        llvm::errs() << "failed to initialize MPI: " << mpiStatus.getMessage()
                     << "\n";
        return failure();
      }
    }

    if (llvm::is_contained(options.features, "cuda") &&
        failed(initializeCudaRuntime()))
      return failure();
  }

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

  // Since an empty input could parse as a valid serialized flatbuffer, we
  // explicitly guard against this case. Otherwise, could get false positives
  // in faulty shell pipelines used in LIT tests.
  if (input->getBufferSize() == 0)
    return emitError(loc) << "input buffer is empty";

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(options.outputFilename, &errorMessage);
  if (!output)
    return emitError(loc) << "failed to open output buffer: " << errorMessage;

  if (options.dumpFunctionSignature) {
    if (options.inputType != InputType::ExecutorRuntimeExecutable)
      return emitError(loc) << "function signature can only be dumped with "
                               "Runtime Executable inputs";
  }

  StatusOr<RuntimeSessionOptions> sessionOptions =
      getRuntimeSessionOptions(options, options.features);
  if (!sessionOptions.isOk())
    return emitError(UnknownLoc::get(&context))
           << "failed to get runtime session options: "
           << sessionOptions.getStatus().getMessage();

  // Read the buffer as a Lua script and execute.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> input,
                           llvm::raw_ostream &os) -> LogicalResult {
    if (options.inputType == Lua) {
      assert(!options.dumpFunctionSignature &&
             "Can not dump function signature for Lua input type.");
      mtrt::StatusOr<int64_t> result = mtrt::runExecutorLuaScript(
          *sessionOptions, input->getBuffer(), registerExtraLuaFuncs);
      if (!result.isOk())
        return emitError(UnknownLoc::get(&context))
               << result.getStatus().getMessage();
      return success(*result == 0);
    }

    assert(options.inputType == ExecutorRuntimeExecutable &&
           "expected executor executable input type");

    mtrt::StatusOr<std::unique_ptr<mtrt::Executable>> executable =
        mtrt::Executable::loadFromBuffer(std::move(input));
    if (!executable.isOk())
      return emitError(UnknownLoc::get(&context))
             << "failed to load executable from buffer: "
             << executable.getStatus().getMessage();

    if (options.dumpFunctionSignature) {
      for (unsigned i = 0; i < executable->get()->getNumFunctions(); ++i) {
        std::string str;
        llvm::raw_string_ostream ss(str);
        mtrt::print(ss, executable->get()->getFunction(i));
        llvm::outs() << ss.str() << "\n";
      }
      return success();
    }

    if (options.dumpDataSegments) {
      for (const DataSegmentInfo &dataSegment :
           executable->get()->getDataSegments()) {
        mtrt::print(llvm::outs(), dataSegment);
        llvm::outs() << "\n";
        if (!dataSegment.isUninitialized()) {
          assert(dataSegment.data() && "data segment should be initialized");
          llvm::outs() << llvm::formatv(
              "Data: [{0:$[, ]@(x)}]\n",
              llvm::iterator_range(llvm::ArrayRef<uint8_t>(
                  reinterpret_cast<const uint8_t *>(dataSegment.data()),
                  dataSegment.size())));
        }
      }
      return success();
    }

    mtrt::StatusOr<int64_t> executionResult =
        mtrt::runExecutorExecutable(*sessionOptions, std::move(*executable),
                                    std::move(registerExtraLuaFuncs));
    if (!executionResult.isOk())
      return emitError(UnknownLoc::get(&context))
             << "failed to load and run executable: "
             << executionResult.getStatus().getMessage();

    return success();
  };

  if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                         output->os(), options.splitInputFile,
                                         options.outputSplitMarker)))
    return failure();
  output->keep();
  return success();
}
