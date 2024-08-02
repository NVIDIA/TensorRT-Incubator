//===- TranslateToTensorRT.cpp --------------------------------------------===//
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
/// Implementation of target translation MLIR -> TensorRT engine.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/Target/Passes.h"
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/NetworkEncoder.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Duration.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "translate-to-tensorrt"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_TRANSLATETOTENSORRTENGINEPASS
#include "mlir-tensorrt-dialect/Target/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

namespace {
struct TensorRTTranslationCLFlags {
  llvm::cl::OptionCategory optCategory{"MLIR-to-TensorRT translation options"};

  //===----------------------------------------------------------------------===//
  // Builder optimization flags
  //===----------------------------------------------------------------------===//
  llvm::cl::opt<uint32_t> tensorrtBuilderOptLevel{
      "tensorrt-builder-opt-level",
      llvm::cl::desc(
          "sets the optimization level (0-5) for the TensorRT engine builder"),
      llvm::cl::init(0), llvm::cl::cat(optCategory)};
  llvm::cl::opt<bool> tensorrtEnableTimingCache{
      "tensorrt-enable-timing-cache",
      llvm::cl::desc(
          "enables sharing timing cache between "
          "TensorRT engines during the build process. May speed up the build."),
      llvm::cl::init(true), llvm::cl::cat(optCategory)};
  llvm::cl::opt<std::string> timingCachePath{
      "tensorrt-timing-cache-path",
      llvm::cl::desc("filesystem path to serialized timing cache. will try to "
                     "load and save the timing cache to this path"),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  llvm::cl::opt<bool> enableTensorRTFp16{
      "tensorrt-fp16",
      llvm::cl::desc(
          "allows TensorRT builder to try fp16 kernels regardless of "
          "the original model's precision."),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};
  llvm::cl::opt<bool> tensorrtObeyPrecisionConstraints{
      "tensorrt-obey-precision-constraints",
      llvm::cl::desc("forces TensorRT builder to use the precision of the "
                     "original model."),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};
  llvm::cl::opt<bool> tensorrtStronglyTyped{
      "tensorrt-strongly-typed",
      llvm::cl::desc("forces TensorRT builder to build a strongly typed "
                     "network."),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // TensorRT Builder Logging
  //===----------------------------------------------------------------------===//
  llvm::cl::opt<bool> enableTensorRTVerboseLogging{
      "tensorrt-verbose",
      llvm::cl::desc("enable verbose logging from tensorrt"),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // Plugin Handling
  //===----------------------------------------------------------------------===//
  llvm::cl::list<std::string> pluginPathsToSerialize{
      "serialize-plugin-with-engine",
      llvm::cl::desc(
          "serializes specified plugin library into TensorRT engine."),
      llvm::cl::value_desc("pluginPathToSerialize"),
      llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // Engine Inspector and Debugging
  //===----------------------------------------------------------------------===//
  llvm::cl::opt<std::string> saveTensorRTEngines{
      "save-tensorrt-engines",
      llvm::cl::desc("Directory where to save TensorRT engines for debugging. "
                     "Path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  llvm::cl::opt<std::string> loadTensorRTEngines{
      "load-tensorrt-engines",
      llvm::cl::desc("Directory where to load TensorRT engines. This path is "
                     "primarily used for debugging and the path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  llvm::cl::opt<std::string> saveTensorRTLayerInfo{
      "save-tensorrt-layer-info",
      llvm::cl::desc(
          "Directory where to save TensorRT LayerInformation JSON for "
          "debugging. Path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
};
} // namespace

/// Global for the translation options. To enable CLI options, call the
/// `registerTensorRTTranslationCLOpts` function before parsing LLVM CL opts.
static llvm::ManagedStatic<TensorRTTranslationCLFlags>
    clTensorRTTranslationOptions;

void mlir::tensorrt::registerTensorRTTranslationCLOpts() {
  (void)*clTensorRTTranslationOptions;
}

//===----------------------------------------------------------------------===//
// TensorRTTranslationOptions
//===----------------------------------------------------------------------===//

TensorRTTranslationOptions TensorRTTranslationOptions::fromCLFlags() {
  assert(clTensorRTTranslationOptions.isConstructed() &&
         "TensorRT translation CL options are not constructed; did you forget "
         "to call `registerTensorRTTranslationCLOpts`?");
  TensorRTTranslationOptions options;
  options.tensorrtBuilderOptLevel =
      clTensorRTTranslationOptions->tensorrtBuilderOptLevel;
  options.enableTimingCache =
      clTensorRTTranslationOptions->tensorrtEnableTimingCache;
  options.timingCachePath = clTensorRTTranslationOptions->timingCachePath;
  options.forceEnableFP16 = clTensorRTTranslationOptions->enableTensorRTFp16;
  options.obeyPrecisionConstraints =
      clTensorRTTranslationOptions->tensorrtObeyPrecisionConstraints;
  options.enableStronglyTyped =
      clTensorRTTranslationOptions->tensorrtStronglyTyped;
  options.enableVerboseLogs =
      clTensorRTTranslationOptions->enableTensorRTVerboseLogging;
  options.pluginPathsToSerialize =
      llvm::to_vector(clTensorRTTranslationOptions->pluginPathsToSerialize);
  options.saveTensorRTLayerInfoDirectory =
      clTensorRTTranslationOptions->saveTensorRTLayerInfo;
  options.saveTensorRTEnginesToDirectory =
      clTensorRTTranslationOptions->saveTensorRTEngines;
  options.loadTensorRTEnginesFromDirectory =
      clTensorRTTranslationOptions->loadTensorRTEngines;

  return options;
}

namespace {

template <typename T>
using TRTUniquePtr = nvinfer1::adaptor::UniquePtr<T>;
} // namespace

//===----------------------------------------------------------------------===//
// Logger
//===----------------------------------------------------------------------===//

void tensorrt::Logger::log(Severity severity, const char *msg) noexcept {
  if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
    llvm::errs() << msg << "\n";
    return;
  }
  if (severity == Severity::kWARNING) {
    llvm::errs() << msg << "\n";
    return;
  }
  if (verbose)
    llvm::errs() << msg << "\n";
}

//===----------------------------------------------------------------------===//
// TensorRTBuilderContext
//===----------------------------------------------------------------------===//

/// Return version information for the runtime loaded TensorRT library. Emit a
/// warning if this differs from the TensorRT version present at compile time.
static TensorRTVersion getTensorRTLoadedLibraryVersion() {
  TensorRTVersion result = TensorRTVersion::getLoadedVersion();

  static llvm::once_flag flag;
  llvm::call_once(
      flag,
      [](const TensorRTVersion &version) {
        llvm::errs() << "Loaded TensorRT version " << version.getAsString();
        if (version != TensorRTVersion::getCompileTimeVersion())
          llvm::errs() << " but compiled for TensorRT " << NV_TENSORRT_MAJOR
                       << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
                       << "." << NV_TENSORRT_BUILD
                       << ". This can result in crashes or unintended behavior";
        llvm::errs() << ".\n";
      },
      result);
  return result;
}

FailureOr<std::shared_ptr<TensorRTBuilderContext>>
TensorRTBuilderContext::create(bool verbose, int32_t cudaDevice) {
  auto version = getTensorRTLoadedLibraryVersion();
  cudaError_t status = cudaSetDevice(cudaDevice);
  if (status != cudaSuccess)
    return failure();

  auto logger = std::make_unique<Logger>(verbose);
  if (!logger)
    return failure();

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*logger));
  if (!builder)
    return failure();

  return std::shared_ptr<TensorRTBuilderContext>(new TensorRTBuilderContext(
      version, cudaDevice, std::move(logger), std::move(builder)));
}

//===----------------------------------------------------------------------===//
// TensorRTSerializedTimingCache
//===----------------------------------------------------------------------===//

std::unique_ptr<nvinfer1::ITimingCache>
TensorRTSerializedTimingCache::createCache(nvinfer1::IBuilderConfig &config) {
  std::scoped_lock<std::mutex> g(lock);

  LLVM_DEBUG(DBGS() << "deserializing TensorRT builder timing cache ("
                    << data.size() << " bytes)\n");

  return std::unique_ptr<nvinfer1::ITimingCache>(
      config.createTimingCache(data.data(), data.size()));
}

void TensorRTSerializedTimingCache::replaceWith(ArrayRef<char> newData) {
  std::scoped_lock<std::mutex> g(lock);

  LLVM_DEBUG(DBGS() << "replacing cache with updated data (" << data.size()
                    << " -> " << newData.size() << " bytes)\n");

  data.resize(newData.size());
  std::copy(newData.begin(), newData.end(), data.begin());
}

void TensorRTSerializedTimingCache::write(llvm::raw_ostream &os) {
  std::scoped_lock<std::mutex> g(lock);

  LLVM_DEBUG(DBGS() << "serializing TensorRT builder timing cache ("
                    << data.size() << " bytes)\n");

  os.write(data.data(), data.size());
}

//===----------------------------------------------------------------------===//
// Core TensorRT Translation Entrypoint
//===----------------------------------------------------------------------===//

/// Set the 'builder optimization level' on the TensorRT builder. This is
/// primarily used to tradeoff compilation time and performance. Since we
/// support building the project with TensorRT 8.5 and loading different
/// versions at runtime, the invocation is guarded staticly and dynamically.
static void
setBuilderOptimizationLevel(nvinfer1::IBuilderConfig *config, uint32_t optLevel,
                            const TensorRTVersion &loadedTensorRTVersion) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(8, 6, 0)
  assert(optLevel >= 0 && optLevel <= 5 &&
         "expected TensorRT builder optimization level >= 0 and <= 5");
  if (loadedTensorRTVersion >= TensorRTVersion(8, 6, 0)) {
    LLVM_DEBUG(DBGS() << "Setting builder optimization level to " << optLevel
                      << "\n");
    config->setBuilderOptimizationLevel(optLevel);
  }
#endif
}

/// If the project was built with a TensorRT version > 9.2, then allow setting
/// the `strongly typed` option if TensorRT 10 or higher was loaded.
static LogicalResult maybeSetStronglyTypedOption(
    Location loc, const TensorRTBuilderContext &builderContext,
    const TensorRTTranslationOptions &opts, uint32_t &networkCreationFlags) {
  if (!opts.enableStronglyTyped)
    return success();

  LLVM_DEBUG(
      DBGS() << "enabling 'strongly-typed' mode in TensorRT translation\n");

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  if (builderContext.getTensorRTVersion() >= TensorRTVersion(10, 0, 0)) {
    networkCreationFlags |=
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    return success();
  }
  return failure();
#else
  return failure();
#endif
}

FailureOr<TensorRTEngineResult>
tensorrt::buildFunction(mlir::FunctionOpInterface op,
                        TensorRTBuilderContext &builderContext,
                        TensorRTSerializedTimingCache &serializedTimingCache,
                        const TensorRTTranslationOptions &opts) {
  assert(builderContext.getBuilder() && "expected valid builder context");
  std::unique_ptr<nvinfer1::IBuilder> &builder = builderContext.getBuilder();

  uint32_t networkCreationFlags = 0;
  if (failed(maybeSetStronglyTypedOption(UnknownLoc::get(op.getContext()),
                                         builderContext, opts,
                                         networkCreationFlags)))
    return emitError(UnknownLoc::get(op->getContext()))
           << "TensorRT strongly-typed mode is currently restricted to "
              "TensorRT versions >= 10.0.0; although it is available in "
              "TensorRT >= 9.1, certain bugs in TensorRT 9.x make the feature "
              "inadvisable for use prior to TensorRT 10.0";

  TRTUniquePtr<nvinfer1::INetworkDefinition> network =
      nvinfer1::adaptor::createNetworkV2(builder, networkCreationFlags);
  if (network == nullptr)
    return failure();
  nvinfer1::IOptimizationProfile *optimProfile =
      builder->createOptimizationProfile();

  NvInferNetworkEncoder encoder(network.get(), optimProfile,
                                builderContext.getTensorRTVersion(),
                                opts.enableStronglyTyped);

  // Currently we only support single-block functions with unique return
  // terminator ops.
  assert(op.getFunctionBody().hasOneBlock() &&
         "only single-block function-like region supported");

  // builder calls.
  if (failed(encoder.encodeFunc(op)))
    return failure();

  // Build the network.
  auto config =
      TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->addOptimizationProfile(optimProfile);

  TensorRTVersion loadedTRTVersion = TensorRTVersion::getLoadedVersion();

  if (!encoder.isStronglyTyped()) {
    if (encoder.hasInt8Usage())
      config->setFlag(nvinfer1::BuilderFlag::kINT8);

    if (encoder.hasFp16Usage() || opts.forceEnableFP16)
      config->setFlag(nvinfer1::BuilderFlag::kFP16);

    if (encoder.hasFp8Usage() && loadedTRTVersion.isGreaterThanOrEqualTRT10()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP8);
    } else if (encoder.hasFp8Usage()) {
      return emitError(UnknownLoc::get(op->getContext()))
             << " FP8 uses detected in the network but loaded TensorRT version "
                "is "
             << loadedTRTVersion.getAsString()
             << " . FP8 type is only supported from TRT 10";
    }

    if (encoder.hasBf16Usage() &&
        loadedTRTVersion.isGreaterThanOrEqualTRT10()) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
      config->setFlag(nvinfer1::BuilderFlag::kBF16);
#endif
    } else if (encoder.hasBf16Usage()) {
      return emitError(UnknownLoc::get(op->getContext()))
             << " BF16 uses detected in the network but loaded TensorRT "
                "version "
                "is "
             << loadedTRTVersion.getAsString()
             << " . BF16 type is only supported from TRT 10";
    }

    if (encoder.hasInt4Usage() && !loadedTRTVersion.isGreaterThanOrEqualTRT10())
      return emitError(UnknownLoc::get(op->getContext()))
             << " INT4 uses detected in the network but loaded TensorRT "
                "version "
                "is "
             << loadedTRTVersion.getAsString()
             << " . INT4 type is only supported from TRT 10";

    if (opts.obeyPrecisionConstraints)
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
  } else if (opts.forceEnableFP16 || opts.obeyPrecisionConstraints) {
    return emitError(UnknownLoc::get(op->getContext()))
           << "Invalid precision flags when strongly typed mode is enabled.";
  }

  std::unique_ptr<nvinfer1::ITimingCache> timingCache{nullptr};
  // Attach timing cache
  if (opts.enableTimingCache) {
    timingCache = serializedTimingCache.createCache(*config);
    if (timingCache == nullptr)
      return failure();
    if (!config->setTimingCache(*timingCache, false))
      return emitError(UnknownLoc::get(op->getContext()))
             << "failed to set timing cache";
  }

  // If created, engines and their layer information are
  // with detailed description.
  if (!opts.saveTensorRTEnginesToDirectory.empty() ||
      !opts.saveTensorRTLayerInfoDirectory.empty())
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

  setBuilderOptimizationLevel(config.get(), opts.tensorrtBuilderOptLevel,
                              builderContext.getTensorRTVersion());

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(8, 6, 0)
  if (!opts.pluginPathsToSerialize.empty() &&
      builderContext.getTensorRTVersion() >= TensorRTVersion(8, 6, 0)) {
    std::vector<char const *> pluginPaths;
    for (auto &path : opts.pluginPathsToSerialize)
      pluginPaths.push_back(path.c_str());
    config->setPluginsToSerialize(pluginPaths.data(), pluginPaths.size());
  }
#endif

  FailureOr<std::vector<std::string>> names =
      getResultTensorNames(op.getNumResults(), encoder);
  if (failed(names))
    return failure();

  nvinfer1::IHostMemory *hostMem =
      builder->buildSerializedNetwork(*network, *config);
  if (!hostMem)
    return failure();

  // Write timing cache
  if (opts.enableTimingCache) {
    TRTUniquePtr<nvinfer1::IHostMemory> updatedTimingCache(
        config->getTimingCache()->serialize());
    serializedTimingCache.replaceWith(mlir::ArrayRef(
        reinterpret_cast<const char *>(updatedTimingCache->data()),
        updatedTimingCache->size()));
  }
  return TensorRTEngineResult{std::unique_ptr<nvinfer1::IHostMemory>(hostMem),
                              *names};
}

/// Return the symbol names of parent symbol tables and ending with the symbol
/// name of `op`.
static SmallVector<StringRef> getSymbolNames(Operation *op) {
  SmallVector<StringRef> pathElements;
  while (op) {
    if (!op->hasTrait<OpTrait::SymbolTable>() &&
        !op->hasAttr(SymbolTable::getSymbolAttrName())) {
      op = op->getParentOp();
      continue;
    }

    StringAttr symbolName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

    pathElements.push_back(symbolName ? symbolName.strref() : "no-symbol-name");
    op = op->getParentOp();
  }
  // Return in the order of top level (module) down to `op`.
  std::reverse(pathElements.begin(), pathElements.end());
  return pathElements;
}

/// Add details of the op to the given hash. This will only include the
/// operation name and the operand and result types.
static llvm::hash_code combineOperationHash(Operation *op,
                                            const llvm::hash_code &rhs) {
  llvm::hash_code code = llvm::hash_combine(rhs, op->getName().getStringRef());
  if (op->getNumResults() == 0 || op->getNumOperands() == 0)
    return code;
  llvm::SmallString<128> dataToHash;
  {
    llvm::raw_svector_ostream os(dataToHash);
    os << op->getNumOperands();
    for (Type t : op->getOperandTypes())
      os << t;
    os << op->getNumResults();
    for (Type t : op->getResultTypes())
      os << t;
  }
  return llvm::hash_combine(code, dataToHash);
}

/// Hash some details of the function. This isn't good enough to protect against
/// all differences, but it should prevent one from reusing a saved engine if it
/// has a different name, type, or contains a different number of type of
/// operations in the body (but doesn't hash use-def). A better equivalence
/// would require saving the MLIR, loading it, and comparing the loaded MLIR to
/// the current func.
static llvm::hash_code hashFunc(func::FuncOp func) {
  llvm::hash_code hash = {};
  llvm::SmallString<128> dataToHash;
  {
    llvm::raw_svector_ostream os(dataToHash);
    os << func.getName() << func.getFunctionType();
  }
  llvm::hash_combine(hash, dataToHash);
  func->walk([&](Operation *op) { hash = combineOperationHash(op, hash); });
  return hash;
}

/// Create all directories in `path`, ignoring those that already exist. Failure
/// to create a directory results in emitting a diagnostic and returning
/// failure.
static LogicalResult createDirectories(Location loc, StringRef path) {
  if (std::error_code EC =
          llvm::sys::fs::create_directories(path, /*IgnoreExisting=*/true)) {
    emitError(loc) << "[translate-to-tensorrt] could not create directory '"
                   << path << "': " << EC.message();
    return failure();
  }
  return success();
}

/// Returns the engine file name as a sequence of concatenated symbol names
/// starting from the highest level SymbolTable parent and ending with the
/// function name.
static std::string getFuncSignatureString(func::FuncOp func) {
  SmallVector<StringRef> symbolNames = getSymbolNames(func);
  llvm::hash_code hash = hashFunc(func);
  return llvm::formatv("{0:$[_]}_{1}",
                       llvm::make_range(symbolNames.begin(), symbolNames.end()),
                       hash);
}

static llvm::SmallString<128> getEngineFileName(func::FuncOp func,
                                                StringRef basePath) {
  llvm::SmallString<128> path = basePath;
  llvm::sys::path::append(
      path, llvm::formatv("{0}.engine", getFuncSignatureString(func)));
  return path;
}

/// Write the `serializedEngine` to a file at the given directory. The filename
/// is calculated from the function name and the hash of the function.
static LogicalResult saveTensorRTEngineToFile(
    func::FuncOp func, StringRef directoryPath,
    const std::unique_ptr<nvinfer1::IHostMemory> &serializedEngine) {
  if (failed(createDirectories(func.getLoc(), directoryPath)))
    return failure();
  llvm::SmallString<128> fileName = getEngineFileName(func, directoryPath);
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(fileName, &error);
  if (!of) {
    emitError(func.getLoc()) << "failed to open " << fileName << ": " << error;
    return failure();
  }
  of->os().write(static_cast<char *>(serializedEngine->data()),
                 serializedEngine->size());
  if (of->os().has_error()) {
    emitError(func.getLoc()) << "failed to write TensorRT engine to "
                             << fileName << ": " << of->os().error().message();
    return failure();
  }
  of->keep();
  return success();
}

/// Load a serialized timing cache from the given path.
static std::shared_ptr<mlir::tensorrt::TensorRTSerializedTimingCache>
loadSerializedTimingCache(StringRef timingCachePath) {
  if (timingCachePath.empty()) {
    LLVM_DEBUG(DBGS() << "timing cache path was not specified, creating a "
                         "fresh timing cache\n");
    return std::make_shared<mlir::tensorrt::TensorRTSerializedTimingCache>();
  }

  std::string error;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      mlir::openInputFile(timingCachePath, &error);
  if (!inputFile) {
    llvm::errs() << "failed to load TensorRT builder timing cache from "
                    "provided file path ("
                 << timingCachePath << "), error on opening file: " << error
                 << ", the compiler will start with a fresh empty cache\n";
    return std::make_shared<mlir::tensorrt::TensorRTSerializedTimingCache>();
  }

  LLVM_DEBUG(DBGS() << "loading timing cache of size "
                    << inputFile->getBufferSize() << " from " << timingCachePath
                    << "\n");
  return std::make_shared<mlir::tensorrt::TensorRTSerializedTimingCache>(
      inputFile->getBufferStart(), inputFile->getBufferSize());
}

/// Try to lock the output file and write the cache if it is filled with data.
static void maybeWriteTimingCache(TensorRTSerializedTimingCache &timingCache,
                                  StringRef timingCachePath) {
  if (timingCachePath.empty() || timingCache.size() == 0)
    return;

  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(timingCachePath, &error);
  if (!of) {
    llvm::errs() << "failed to open output file for timing cache: " << error
                 << "\n";
    return;
  }

  constexpr std::chrono::milliseconds kTryLockTimeout(10);
  llvm::Expected<llvm::sys::fs::FileLocker> lock =
      of->os().tryLockFor(llvm::Duration(kTryLockTimeout));
  if (lock) {
    timingCache.write(of->os());
    if (of->os().has_error()) {
      llvm::errs() << "failed to write TensorRT builder timing cache to file ("
                   << timingCachePath << "), received file stream error: "
                   << of->os().error().message() << "\n";
      return;
    }
    of->keep();
    LLVM_DEBUG(DBGS() << "wrote TensorRT builder timing cache to "
                      << timingCachePath << " (" << timingCache.size()
                      << " bytes)\n");

    return;
  }

  llvm::Error err = lock.takeError();
  llvm::errs() << "failed to lock TensorRT timing cache file ("
               << timingCachePath
               << ") for writing. Received the following error when "
                  "trying to lock the file: "
               << llvm::to_string(err) << "\n";
}

namespace {
class TranslateToTensorRTEnginePass
    : public tensorrt::impl::TranslateToTensorRTEnginePassBase<
          TranslateToTensorRTEnginePass> {
public:
  TranslateToTensorRTEnginePass()
      : builderContext(nullptr),
        translationOptions(TensorRTTranslationOptions::fromCLFlags()) {}

  explicit TranslateToTensorRTEnginePass(
      std::shared_ptr<TensorRTBuilderContext> builderContext,
      TensorRTTranslationOptions options)
      : builderContext(builderContext), translationOptions(std::move(options)) {
  }

  LogicalResult initialize(MLIRContext *context) final {
    if (!this->builderContext) {
      FailureOr<std::shared_ptr<TensorRTBuilderContext>> builderResult =
          TensorRTBuilderContext::create(
              clTensorRTTranslationOptions->enableTensorRTVerboseLogging);
      if (failed(builderResult))
        return emitError(UnknownLoc::get(context))
               << "Failed to create TensorRT builder context";
      this->builderContext = std::move(*builderResult);
      LLVM_DEBUG(
          DBGS()
          << "TranslateToTensorRTEnginePass is generating a new TensorRT "
             "builder\n");
    }

    if (!this->timingCache)
      this->timingCache =
          loadSerializedTimingCache(translationOptions.timingCachePath);

    return success();
  }

  void runOnOperation() override {
    assert(this->builderContext && "expected valid TensorRT builder context");
    assert(this->timingCache && "expected valid timing cache");

    Operation *rootOp = getOperation();
    if (rootOp->getNumRegions() == 0)
      return;
    SmallVector<func::FuncOp> funcs =
        llvm::to_vector(rootOp->getRegion(0).getOps<func::FuncOp>());
    if (funcs.empty())
      return;

    // Set the current device to the one the builder is associated with.
    if (cudaSetDevice(builderContext->getCudaDeviceNumber()) != cudaSuccess) {
      emitError(rootOp->getLoc()) << "failed to set the current CUDA device";
      return signalPassFailure();
    }

    for (auto func : funcs) {
      if (!translationOptions.loadTensorRTEnginesFromDirectory.empty()) {
        llvm::SmallString<128> fileName = getEngineFileName(
            func, translationOptions.loadTensorRTEnginesFromDirectory);
        std::string error;
        std::unique_ptr<llvm::MemoryBuffer> buffer =
            mlir::openInputFile(fileName, &error);
        if (!buffer) {
          emitError(rootOp->getLoc())
              << "failed to open input file " << fileName << ": " << error;
          return signalPassFailure();
        }

        // Attach the engine as an attribute on the function.
        auto engineAttr = DenseElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(buffer->getBufferSize())},
                IntegerType::get(&getContext(), 8)),
            llvm::ArrayRef<int8_t>(
                reinterpret_cast<const int8_t *>(buffer->getBufferStart()),
                buffer->getBufferSize()));
        func->setAttr("tensorrt.engine", engineAttr);
        continue;
      }

      FailureOr<TensorRTEngineResult> engineResult = buildFunction(
          func, *builderContext, *timingCache, translationOptions);
      if (failed(engineResult) || !engineResult->serializedEngine) {
        func.emitError() << "failed to translate function '" << func.getName()
                         << "' to a TensorRT engine";
        return signalPassFailure();
      }
      const std::unique_ptr<nvinfer1::IHostMemory> &serializedEngine =
          engineResult->serializedEngine;

      if (!translationOptions.saveTensorRTEnginesToDirectory.empty() &&
          failed(saveTensorRTEngineToFile(
              func, translationOptions.saveTensorRTEnginesToDirectory,
              serializedEngine)))
        return signalPassFailure();

      if (!translationOptions.saveTensorRTLayerInfoDirectory.empty()) {
        std::unique_ptr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(*builderContext->getLogger())};
        std::unique_ptr<nvinfer1::ICudaEngine> cudaEngine{
            runtime->deserializeCudaEngine(serializedEngine->data(),
                                           serializedEngine->size())};
        auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(
            cudaEngine->createEngineInspector());
        llvm::SmallString<128> fileName =
            StringRef(translationOptions.saveTensorRTLayerInfoDirectory);
        llvm::sys::path::append(
            fileName, llvm::formatv("{0}.json", getFuncSignatureString(func)));
        std::string error;
        std::unique_ptr<llvm::ToolOutputFile> of =
            mlir::openOutputFile(fileName, &error);
        if (!of) {
          emitError(UnknownLoc::get(&getContext()))
              << "failed to open " << fileName << ": " << error;
          return signalPassFailure();
        }
        of->os() << inspector->getEngineInformation(
            nvinfer1::LayerInformationFormat::kJSON);
        if (of->os().has_error()) {
          emitError(UnknownLoc::get(&getContext()))
              << "failed to write TensorRT LayerInformation JSON to "
              << fileName << ": " << of->os().error().message();
          return signalPassFailure();
        }
        of->keep();
      }

      // Attach the engine as an attribute on the function.
      auto engineAttr = DenseElementsAttr::get(
          RankedTensorType::get(
              {static_cast<int64_t>(serializedEngine->size())},
              IntegerType::get(&getContext(), 8)),
          llvm::ArrayRef<int8_t>(
              reinterpret_cast<const int8_t *>(serializedEngine->data()),
              serializedEngine->size()));
      func->setAttr("tensorrt.engine", engineAttr);
    }

    // update the timing cache if required.
    maybeWriteTimingCache(*timingCache, translationOptions.timingCachePath);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorrt::TensorRTDialect>();
  }

private:
  /// The builder context is default null and can be provided at pass
  /// construction time or it will lazily constructed during the pass
  /// execution.
  std::shared_ptr<TensorRTBuilderContext> builderContext{nullptr};

  /// Holds a serialized timing cache (default empty) that will be populated
  /// and reused over translation calls.
  std::shared_ptr<TensorRTSerializedTimingCache> timingCache{nullptr};

  /// Options affecting TensorRT translation.
  TensorRTTranslationOptions translationOptions;
};
} // namespace

std::unique_ptr<mlir::Pass> tensorrt::createTranslateTensorRTPass(
    std::shared_ptr<tensorrt::TensorRTBuilderContext> context,
    TensorRTTranslationOptions options) {
  return std::make_unique<TranslateToTensorRTEnginePass>(context, options);
}
