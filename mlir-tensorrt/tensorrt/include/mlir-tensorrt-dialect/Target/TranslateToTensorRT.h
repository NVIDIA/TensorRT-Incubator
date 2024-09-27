//===- TranslateToTensorRT.h ------------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_TARGET_TENSORRT_TRANSLATETOTENSORRT_H
#define MLIR_TENSORRT_TARGET_TENSORRT_TRANSLATETOTENSORRT_H

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/NetworkEncoder.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInfer.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace mlir {
class FunctionOpInterface;
class Operation;
class Pass;
namespace tensorrt {

/// A simple logger that implements TensorRT's logging interface. Errors and
/// warnings are reported stderr. If the 'verbose' flags is active, then all
/// messages are printed to stderr.
class Logger : public nvinfer1::ILogger {
public:
  explicit Logger(bool verbose = false) : verbose(verbose) {}

protected:
  void log(Severity severity, const char *msg) noexcept override;

  /// Print only 'error' and 'warning' messages if false, otehrwise print all
  /// messages.
  bool verbose;
};

/// A llvm::cl::opt parser for turning strings like "1024gb" into a number of
/// bytes. Allowed suffixes are strings like 'gb', 'GiB', 'kb', 'mb', 'b' (case
/// insensitive, we interpret both 'b|B' as meaning "byte"). This example comes
/// straight from the LLVM documentation
/// (https://llvm.org/docs/CommandLine.html#writing-a-custom-parser).
struct ByteSizeParser : public llvm::cl::parser<std::optional<uint64_t>> {
  using llvm::cl::parser<std::optional<uint64_t>>::parser;
  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef ArgValue,
             std::optional<uint64_t> &Val);
};

/// TensorRTTranslationOptions wraps all available options for constructing
/// TensorRT engine from MLIR. This should expose all TensorRT
/// `nvinfer1::IBuilder` options that we currently support.
struct TensorRTTranslationOptions {
  /// Creates default set of options that are initialized from `llvm::cl` flags.
  static TensorRTTranslationOptions fromCLFlags();

  /// The `nvinfer1::IBuilder` optimization level.
  uint32_t tensorrtBuilderOptLevel = 0;

  /// Whether to enable or disable use of timing cache.
  bool enableTimingCache = true;

  /// Where to persist the timing cache to storage.
  std::string timingCachePath = "";

  /// Force use of FP16 mode even if there are not FP16 tensors in the program.
  bool forceEnableFP16 = false;

  /// Set the nvinfer::IBuilder flag to obey precision constraints.
  bool obeyPrecisionConstraints = false;

  /// Set the nvinfer::NetworkDefinitionCreationFlag to use strongly typed mode.
  bool enableStronglyTyped = false;

  /// Maximum workspace/scratchspace (in bytes) allowed. The abcense of a value
  /// indicates that the limit is the maximum device memory.
  std::optional<uint64_t> workspaceMemoryPoolLimit = std::nullopt;

  /// Enable TensorRT verbose logs
  bool enableVerboseLogs = false;

  /// Specifies a list of plugin `.so` library paths to serialize with the
  /// engine.
  SmallVector<std::string> pluginPathsToSerialize;

  //===----------------------------------------------------------------------===//
  // Debugging options
  //===----------------------------------------------------------------------===//

  /// Dump the layer info JSON file to the specified directory.
  std::string saveTensorRTLayerInfoDirectory;

  /// Save TensorRT engines, which are given a unique filename consisting of the
  /// name of the function symbol plus the hash of the function, to the
  /// specified directory.
  std::string saveTensorRTEnginesToDirectory;

  /// Load TensorRT engines from the specified directory. Note that this will
  /// look for the name consisting of the function symbol name and the function
  /// hash.
  std::string loadTensorRTEnginesFromDirectory;

  /// Add command line options to mlir::OptionsContext and configure struct to
  /// serve as backend storage for the options.
  void addToOptions(mlir::OptionsContext &context) {
    context.addOption("tensorrt-timing-cache-path", timingCachePath,
                      llvm::cl::init(""));
    context.addOption("tensorrt-builder-opt-level", tensorrtBuilderOptLevel,
                      llvm::cl::init(0));
    context.addOption("tensorrt-strongly-typed", enableStronglyTyped,
                      llvm::cl::init(false));
    context.addOption("tensorrt-engines-dir", saveTensorRTEnginesToDirectory,
                      llvm::cl::init(""));
    context.addOption("tensorrt-layer-info-dir", saveTensorRTLayerInfoDirectory,
                      llvm::cl::init(""));
    context.addOptionWithParser<ByteSizeParser>(
        "tensorrt-workspace-memory-pool-limit", workspaceMemoryPoolLimit,
        llvm::cl::init(std::nullopt));
  }
};

/// TensorRTBuilderContext encapsulates the TensorRT logger and builder objects.
/// A single context can be shared over multiple runs of e.g. a translation run
/// or pass that invokes the translation entrypoint `buildFunction` below.
class TensorRTBuilderContext {
private:
  TensorRTBuilderContext(TensorRTVersion version, int32_t cudaDevice,
                         std::unique_ptr<Logger> logger,
                         std::unique_ptr<nvinfer1::IBuilder> builder)
      : version(version), cudaDevice(cudaDevice), logger(std::move(logger)),
        builder(std::move(builder)) {}

public:
  /// Create a TensorRTBuilderContext from a log configuration and CUDA device
  /// number. It will also load the TensorRT shlib dynamically and check the
  /// version against the version in the header at compile time. The CUDA device
  /// will be set so that the builder is created on that device by calling
  /// `setCudaDevice`, so this call will change current device if it is not the
  /// same.
  static FailureOr<std::shared_ptr<TensorRTBuilderContext>>
  create(bool verbose = false, int32_t cudaDevice = 0);

  /// Return a handle to the TensorRT builder.
  const std::unique_ptr<nvinfer1::IBuilder> &getBuilder() const {
    return builder;
  }

  /// Return a handle to the TensorRT builder.
  std::unique_ptr<nvinfer1::IBuilder> &getBuilder() { return builder; }

  /// Return the loaded TensorRT version information.
  const TensorRTVersion &getTensorRTVersion() const { return version; }

  /// Return which CUDA device the builder is associated with.
  int32_t getCudaDeviceNumber() const { return cudaDevice; }

  /// Return a handle to the logger.
  const std::unique_ptr<Logger> &getLogger() const { return logger; }

private:
  TensorRTVersion version;
  /// The CUDA device that this builder is associated with.
  int32_t cudaDevice;
  std::unique_ptr<Logger> logger;
  std::unique_ptr<nvinfer1::IBuilder> builder;
};

/// Encapsulates the results of building a TensorRT engine from MLIR IR
/// representing a function.
struct TensorRTEngineResult {
  /// The serialized TensorRT engine.
  std::unique_ptr<nvinfer1::IHostMemory> serializedEngine;
  /// Identifies the names of the ITensor objects that represent the function
  /// results, in the order that they are returned in the MLIR IR.
  std::vector<std::string> resultNames;
};

/// Encapsulates a serialized timing cache.
class TensorRTSerializedTimingCache {
public:
  TensorRTSerializedTimingCache() = default;
  TensorRTSerializedTimingCache(const char *buffer, size_t size)
      : data(buffer, buffer + size) {}

  /// Create a new nvinfer1::ITimingCache using the specified config. Does not
  /// set the config's cache.
  std::unique_ptr<nvinfer1::ITimingCache>
  createCache(nvinfer1::IBuilderConfig &config);

  /// Replace the serialized cache with the given data;
  void replaceWith(ArrayRef<char> data);

  /// Write the cache to given stream.
  void write(llvm::raw_ostream &os);

  /// Return the size of the cache in bytes.
  size_t size() const { return data.size(); }

private:
  std::vector<char> data{};
  std::mutex lock;
};

/// Given the function-like `op`, try to translate it into a TensorRT engine and
/// return the serialized engine data. If `verbose` is true, it prints the
/// TensorRT builder logs to stderr. This function expects that the
/// `tensorrt.shape_profile` arguments have been populated for each argument
/// that has unknown dimensions.
/// TODO(cbate): add additional options here for builder configuration.
FailureOr<TensorRTEngineResult>
buildFunction(mlir::FunctionOpInterface op,
              TensorRTBuilderContext &builderContext,
              TensorRTSerializedTimingCache &serializedTimingCache,
              const TensorRTTranslationOptions &options =
                  TensorRTTranslationOptions::fromCLFlags());

/// Create an instance of a translate-to-tensorrt pass using an existing
/// TensorRTBuilderContext.
std::unique_ptr<mlir::Pass> createTranslateTensorRTPass(
    std::shared_ptr<tensorrt::TensorRTBuilderContext> context,
    TensorRTTranslationOptions options =
        TensorRTTranslationOptions::fromCLFlags());

/// Register llvm::cl opts related to TensorRT translation. This should be
/// called before having LLVM parse CL options.
void registerTensorRTTranslationCLOpts();

} // namespace tensorrt

/// Register the "to-tensorrt" translation with the MLIR target registry.
void registerToTensorRTTranslation();

} // namespace mlir

#endif // MLIR_TRT_TARGET_TENSORRT
#endif // MLIR_TENSORRT_TARGET_TENSORRT_TRANSLATETOTENSORRT_H
