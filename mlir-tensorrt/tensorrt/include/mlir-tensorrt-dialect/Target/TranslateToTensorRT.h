//===- TranslateToTensorRT.h ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
#ifndef MLIR_TENSORRT_DIALECT_TARGET_TRANSLATETOTENSORRT
#define MLIR_TENSORRT_DIALECT_TARGET_TRANSLATETOTENSORRT

#include "mlir-tensorrt-common/Support/CommandLineExtras.h"
#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir-tensorrt-common/Utils/TensorRTVersion.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

namespace nvinfer1 {
class IBuilder;
class IHostMemory;
class IBuilderConfig;
class ITimingCache;
} // namespace nvinfer1

namespace mlir {
class FunctionOpInterface;
class Operation;
class Pass;
namespace tensorrt {

/// TensorRTTranslationOptions wraps all available options for constructing
/// TensorRT engine from MLIR. This should expose all TensorRT
/// `nvinfer1::IBuilder` options that we currently support.
struct TensorRTTranslationOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory optCategory;
  static bool getStronglyTypedDefault();

  /// Creates default set of options that are initialized from `llvm::cl` flags.
  static const TensorRTTranslationOptions &fromCLFlags();
  //===----------------------------------------------------------------------===//
  // Builder optimization flags
  //===----------------------------------------------------------------------===//
  Option<uint32_t> builderOptLevel{
      this->ctx, "tensorrt-builder-opt-level",
      llvm::cl::desc(
          "sets the optimization level (0-5) for the TensorRT engine builder"),
      llvm::cl::init(0), llvm::cl::cat(optCategory)};
  Option<bool> enableTimingCache{
      this->ctx, "tensorrt-enable-timing-cache",
      llvm::cl::desc(
          "enables sharing timing cache between "
          "TensorRT engines during the build process. May speed up the build."),
      llvm::cl::init(true), llvm::cl::cat(optCategory)};
  Option<std::string> timingCachePath{
      this->ctx, "tensorrt-timing-cache-path",
      llvm::cl::desc("filesystem path to serialized timing cache. will try to "
                     "load and save the timing cache to this path"),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  Option<bool> forceEnableFP16{
      this->ctx, "tensorrt-fp16",
      llvm::cl::desc(
          "allows TensorRT builder to try fp16 kernels regardless of "
          "the original model's precision."),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};
  Option<bool> obeyPrecisionConstraints{
      this->ctx, "tensorrt-obey-precision-constraints",
      llvm::cl::desc("forces TensorRT builder to use the precision of the "
                     "original model."),
      llvm::cl::init(false), llvm::cl::cat(optCategory)};
  Option<bool> stronglyTyped{
      this->ctx, "tensorrt-strongly-typed",
      llvm::cl::desc("forces TensorRT builder to build a strongly typed "
                     "network."),
      llvm::cl::init(getStronglyTypedDefault()), llvm::cl::cat(optCategory)};
  Option<std::optional<uint64_t>, mlir::ByteSizeParser>
      workspaceMemoryPoolLimit{
          this->ctx, "tensorrt-workspace-memory-pool-limit",
          llvm::cl::desc(
              "TensorRT workspace memory pool limit "
              "in bytes with optional size suffix like 'GiB' or 'KiB'"),
          llvm::cl::init(std::nullopt), llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // TensorRT Builder Logging
  //===----------------------------------------------------------------------===//
  Option<bool> verbose{this->ctx, "tensorrt-verbose",
                       llvm::cl::desc("enable verbose logging from tensorrt"),
                       llvm::cl::init(false), llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // Plugin Handling
  //===----------------------------------------------------------------------===//
  ListOption<std::string> pluginPathsToSerialize{
      this->ctx, "serialize-plugin-with-engine",
      llvm::cl::desc(
          "serializes specified plugin library into TensorRT engine."),
      llvm::cl::value_desc("pluginPathToSerialize"),
      llvm::cl::cat(optCategory)};

  //===----------------------------------------------------------------------===//
  // Engine Inspector and Debugging
  //===----------------------------------------------------------------------===//
  Option<std::string> saveTensorRTEngines{
      this->ctx, "tensorrt-save-engines-dir",
      llvm::cl::desc("Directory where to save TensorRT engines for debugging. "
                     "Path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  Option<std::string> loadTensorRTEngines{
      this->ctx, "tensorrt-load-engines-dir",
      llvm::cl::desc("Directory where to load TensorRT engines. This path is "
                     "primarily used for debugging and the path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
  Option<std::string> saveTensorRTLayerInfo{
      this->ctx, "tensorrt-layer-info-dir",
      llvm::cl::desc(
          "Directory where to save TensorRT LayerInformation JSON for "
          "debugging. Path must exist."),
      llvm::cl::init(""), llvm::cl::cat(optCategory)};
};

/// TensorRTBuilderContext encapsulates the TensorRT logger and builder objects.
/// A single context can be shared over multiple runs of e.g. a translation run
/// or pass that invokes the translation entrypoint `buildFunction` below.
class TensorRTBuilderContext {
private:
  TensorRTBuilderContext(TensorRTVersion version, int32_t cudaDevice,
                         std::unique_ptr<nvinfer1::IBuilder> builder);

public:
  ~TensorRTBuilderContext();

  /// Create a TensorRTBuilderContext from a log configuration and CUDA device
  /// number. It will also load the TensorRT shlib dynamically and check the
  /// version against the version in the header at compile time. The CUDA device
  /// will be set so that the builder is created on that device by calling
  /// `setCudaDevice`, so this call will change current device if it is not the
  /// same.
  static FailureOr<std::shared_ptr<TensorRTBuilderContext>>
  create(bool verbose = false, int32_t cudaDevice = 0);

  /// Return a handle to the TensorRT builder.
  const std::unique_ptr<nvinfer1::IBuilder> &getBuilder() const;

  /// Return a handle to the TensorRT builder.
  std::unique_ptr<nvinfer1::IBuilder> &getBuilder();

  /// Return the loaded TensorRT version information.
  const TensorRTVersion &getTensorRTVersion() const;

  /// Return which CUDA device the builder is associated with.
  int32_t getCudaDeviceNumber() const { return cudaDevice; }

private:
  TensorRTVersion version;
  /// The CUDA device that this builder is associated with.
  int32_t cudaDevice;
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
    const TensorRTTranslationOptions &options =
        TensorRTTranslationOptions::fromCLFlags());

/// Register llvm::cl opts related to TensorRT translation. This should be
/// called before having LLVM parse CL options.
void registerTensorRTTranslationCLOpts();

/// Return the CL flag category for global TensorRT translation flags.
const llvm::cl::OptionCategory &getTensorRTCLOptionCategory();

} // namespace tensorrt

/// Register the "to-tensorrt" translation with the MLIR target registry.
void registerToTensorRTTranslation();

} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_TARGET_TRANSLATETOTENSORRT
