//===- OptionsProviders.h ---------------------------------------*- C++ -*-===//
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
///
/// Data structures and functions for manipulating compiler options.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_OPTIONSPROVIDERS
#define MLIR_TENSORRT_COMPILER_OPTIONSPROVIDERS

#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

namespace mlirtrt::compiler {

/// An OptionsProvider is a utility to attach options onto a different
/// PassOptions/CompilationTaskOptions struct. The parent to attach to is given
/// in the constructor, and therefore all member `Option|ListOption' can be
/// attached to the parent by giving `this->ctx` as the first argument in their
/// initializer lists.
struct OptionsProvider {
  explicit OptionsProvider(mlir::detail::PassOptions *ctx) : ctx(*ctx) {}
  virtual ~OptionsProvider() = default;

  /// We don't allow move construction since the actual ptrs/locations of
  /// individual member elements of an OptionsProvider are captured into the
  /// parent option. If the parent is populated upon construction,
  /// moving can change the memory location of the owned values, which will
  /// cause a crash later on. This is in particular can happen if you are
  /// constructing a tuple of `OptionsProviders`. Since we are deleting the move
  /// constructor, one must instead use a tuple of
  /// `unique_ptr<OptionsProviders...>`.
  OptionsProvider(OptionsProvider &&) = delete;

  mlir::detail::PassOptions &ctx;

  template <typename T, typename... Mods>
  using Option = mlir::detail::PassOptions::Option<T, Mods...>;
  template <typename T, typename... Mods>
  using ListOption = mlir::detail::PassOptions::ListOption<T, Mods...>;
};

/// DebugOptions are options that are common to different compiler API
/// interfaces.
struct DebugOptions : public OptionsProvider {
public:
  using OptionsProvider::OptionsProvider;

  //===--------------------------------------------------------------------===//
  // Crash Reproducer Generator
  //===--------------------------------------------------------------------===//
  Option<std::string> reproducerFile{
      this->ctx, "mlir-pass-pipeline-crash-reproducer",
      llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                     " if the pass manager crashes or fails")};
  Option<bool> localReproducer{
      this->ctx, "mlir-pass-pipeline-local-reproducer",
      llvm::cl::desc("When generating a crash reproducer, attempt to generated "
                     "a reproducer with the smallest pipeline."),
      llvm::cl::init(false)};

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//

  Option<bool> printBeforeAll{this->ctx, "mlir-print-ir-before-all",
                              llvm::cl::desc("Print IR before each pass"),
                              llvm::cl::init(false)};
  Option<bool> printAfterAll{this->ctx, "mlir-print-ir-after-all",
                             llvm::cl::desc("Print IR after each pass"),
                             llvm::cl::init(false)};
  Option<bool> printAfterChange{
      this->ctx, "mlir-print-ir-after-change",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      llvm::cl::init(false)};
  Option<bool> printAfterFailure{
      this->ctx, "mlir-print-ir-after-failure",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the pass failed"),
      llvm::cl::init(false)};
  Option<bool> printModuleScope{
      this->ctx, "mlir-print-ir-module-scope",
      llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                     "always print the top-level operation"),
      llvm::cl::init(false)};
  Option<std::string> printTreeDir{
      this->ctx, "mlir-print-ir-tree-dir",
      llvm::cl::desc("When printing the IR before/after a pass, print file "
                     "tree rooted at this directory. Use in conjunction with "
                     "mlir-print-ir-* flags")};

  //===----------------------------------------------------------------------===//
  // Printing Flags
  //===----------------------------------------------------------------------===//

  Option<unsigned> elideElementsAttrIfLarger{
      this->ctx, "mlir-elide-elementsattrs-if-larger",
      llvm::cl::desc("Elide ElementsAttrs with \"...\" that have "
                     "more elements than the given upper limit")};

  Option<unsigned> elideResourceStringsIfLarger{
      this->ctx, "mlir-elide-resource-strings-if-larger",
      llvm::cl::desc(
          "Elide printing value of resources if string is too long in chars.")};

  //===--------------------------------------------------------------------===//
  // Pass Statistics
  //===--------------------------------------------------------------------===//
  Option<bool> passStatistics{
      this->ctx, "mlir-pass-statistics",
      llvm::cl::desc("Display the statistics of each pass"),
      llvm::cl::init(false)};

  //===--------------------------------------------------------------------===//
  // Pass Timing
  //===--------------------------------------------------------------------===//
  Option<bool> enableTiming{
      this->ctx, "mlir-timing",
      llvm::cl::desc(
          "Time each pass and print to stderr after the pipeline completes"),
      llvm::cl::init(false)};

  //===----------------------------------------------------------------------===//
  // Debug Printing
  //===----------------------------------------------------------------------===//

  /// Whether the LLVM 'debug' flag that enables execution of code guarded by
  /// the `LLVM_DEBUG` macro should be set to 'on'. This results in very verbose
  /// output from the compiler dumped to stderr.
  Option<bool> enableLLVMDebugFlag{this->ctx, "debug", llvm::cl::init(false)};

  /// A set of names to be given to the LLVM 'debug types' option, akin to
  /// setting
  /// `-debug-types=...` from the command line.
  ListOption<std::string> llvmDebugTypes{this->ctx, "debug-only",
                                         llvm::cl::ZeroOrMore};

  /// If set to `true`, we populate the pass manager instrumentation using
  /// global MLIR CL options rather than the local options contained here.
  Option<bool> useGlobalCLPrintingOptions{this->ctx, "use-global-cl-options",
                                          llvm::cl::init(false)};

  /// Apply these options to the current pass manager.
  void applyToPassManager(mlir::PassManager &pm) const;
};

struct ExecutorOptions : public OptionsProvider {
  using OptionsProvider::OptionsProvider;

  Option<int64_t> indexBitwidth{this->ctx, "executor-index-bitwidth",
                                llvm::cl::init(64),
                                llvm::cl::desc("executor index bitwidth")};
};

struct DeviceOptions : public OptionsProvider {
public:
  DeviceOptions(mlir::detail::PassOptions *ctx);

  /// Get results as DeviceInfo struct.
  DeviceInfo info() {
    DeviceInfo info;
    info.computeCapability = computeCapability;
    info.maxRegistersPerBlock = maxRegistersPerBlock;
    info.maxSharedMemoryPerBlockKb = maxSharedMemoryPerBlockKb;
    return info;
  }

  Option<bool> shouldInferFromHost{
      this->ctx, "device-infer-from-host", llvm::cl::init(true),
      llvm::cl::desc("whether to ignore `deviceX` options and instead infer "
                     "them from the host GPU")};

  Option<int64_t> computeCapability{
      this->ctx, "device-compute-capability", llvm::cl::init(60),
      llvm::cl::desc("Sets the device compute capability. Only relevant "
                     "if '--device-infer-from-host=false'")};
  Option<int64_t> maxSharedMemoryPerBlockKb{
      this->ctx, "device-max-shared-memory-per-block-kb", llvm::cl::init(48)};
  Option<uint64_t> maxRegistersPerBlock{
      this->ctx, "device-max-registers-per-block", llvm::cl::init(65536)};

private:
  /// Stores host device info. This is populated by the callback of
  /// `shouldInferFromHost`. If present, then it will also override the other
  /// options in their callbacks.
  std::optional<DeviceInfo> hostDeviceInfo{};
};

/// Encapsulates options related to the bufferization pipeline.
struct BufferizationOptions : public OptionsProvider {
public:
  using OptionsProvider::OptionsProvider;

  Option<bool> forceEntrypointsReturnAllocs{
      this->ctx, "force-entrypoints-return-allocs", llvm::cl::init(false),
      llvm::cl::desc(
          "Require entrypoint functions to return allocations corresponding to "
          "the original tensor results, otherwise they are transformed into "
          "destination arguments whenever possible.")};

  Option<bool> deallocationPrivateFuncDynamicOwnership{
      this->ctx, "deallocation-private-func-dynamic-ownership",
      llvm::cl::init(false),
      llvm::cl::desc(
          "Overrides the default private function ABI in the buffer "
          "deallocation pipeline to allow for dynamic ownership of memref "
          "arguments and returned memrefs.")};

  Option<bool> enablePinnedMemoryPromotion{
      this->ctx, "enable-pinned-memory-promotion", llvm::cl::init(true),
      llvm::cl::desc("Enable promotion of host buffers to pinned memory using "
                     "heuristics.")};

  Option<bool> enableBufferLoopHoisting{
      this->ctx, "enable-buffer-loop-hoisting", llvm::cl::init(true),
      llvm::cl::desc("Enable buffer hoisting out of loops.")};

  Option<bool> enableBufferHoisting{this->ctx, "enable-buffer-hoisting",
                                    llvm::cl::init(true),
                                    llvm::cl::desc("Enable buffer hoisting.")};
};

//===----------------------------------------------------------------------===//
// Common Enum Definitions
//===----------------------------------------------------------------------===//

/// An enum encapsulating the compilation target for the host code.
enum class HostTarget { Executor, LLVM, EmitC };

/// Specifies the compilation target for tensorrt code.
enum class TensorRTTargetFormat {
  /// Indicates that the compiler should produce compiled TensorRT engines for
  /// computations that are offloaded to TensorRT. This uses the GPU(s)
  /// attached
  /// to the host system (the system on which the compiler is running).
  Engine,
  /// Indicates that the compiler should produce C++ code which invokes the
  /// TensorRT C++ "nvinfer" API. This option is only valid if the host code
  /// is
  /// also being lowered to TensorRT.
  CPP
};

//===----------------------------------------------------------------------===//
// CompilationTaskOptions
//===----------------------------------------------------------------------===//

/// CompilationTaskOptionsBase provides the base class and common options for
/// all task options. Options containers associated with a CompilationTask must
/// inherit from this class.
class CompilationTaskOptionsBase
    : public mlir::PassPipelineOptions<CompilationTaskOptionsBase> {
public:
  /// Construct the options, and if enableDebugOptions is true, then
  /// DebugOptions are attached to the instance and the values will be parsed
  /// along with the other options attached. Otherwise, it is expected that
  /// caller will populate pass manager instrumentation via some other
  /// mechanism (e.g. global CL options).
  CompilationTaskOptionsBase(bool enableDebugOptions = false) {
    if (enableDebugOptions)
      debugOptions = std::make_unique<DebugOptions>(this);
  }

  virtual ~CompilationTaskOptionsBase() = default;

  std::optional<llvm::hash_code> getHash() const;

  /// Populate the values of all associated options by parsing the given
  /// arguments.
  mlir::LogicalResult parse(llvm::ArrayRef<llvm::StringRef> args,
                            std::string &err);

  /// Returns true if this owns a valid DebugOptions struct. If it returns
  /// false, then default MLIR global CL options should be used to populate pass
  /// instrumentation.
  bool hasDebugOptions() const { return debugOptions != nullptr; }

  /// Get the DebugOptions pointer, which may be null.
  const DebugOptions *getDebugOptions() const { return debugOptions.get(); }

  //===----------------------------------------------------------------------===//
  // Options common to all tasks
  //===----------------------------------------------------------------------===//

  Option<HostTarget> hostTarget{
      *this, "host-target", llvm::cl::init(HostTarget::Executor),
      llvm::cl::desc(
          "specifies the target compilation format for host functions"),
      llvm::cl::values(
          clEnumValN(HostTarget::Executor, "executor",
                     "compile host code to MLIR-TRT interpretable executable"),
          clEnumValN(HostTarget::LLVM, "llvm", "compile host code to LLVM IR"),
          clEnumValN(HostTarget::EmitC, "emitc", "compile host code to C++"))};

  Option<std::string> artifactsDirectory{
      *this, "artifacts-dir", llvm::cl::init(""),
      llvm::cl::desc("Specifies where large artifacts can be offloaded as "
                     "external files referenced by filename in the IR")};

  Option<std::string> entrypoint{*this, "entrypoint", llvm::cl::init("main"),
                                 llvm::cl::desc("entrypoint function name")};

  Option<bool> disableAllExtensions{*this, "disable-all-extensions",
                                    llvm::cl::init(false),
                                    llvm::cl::desc("disable all extensions")};

protected:
  std::unique_ptr<DebugOptions> debugOptions{nullptr};
};

/// CompilationTaskOptions is just like CompilationTaskOptionsBase, but it
/// allows specifying any of the handy "OptionsProviders" defined above
/// inorder to automatically incorporate the options associated with each
/// provider.
template <typename... Providers>
class CompilationTaskOptions : public CompilationTaskOptionsBase {
public:
  CompilationTaskOptions(bool enableDebugOptions = false)
      : CompilationTaskOptionsBase(enableDebugOptions),
        optionProviders(std::make_unique<Providers>(this)...) {}

  /// Access provider value.
  template <typename OptionsProviderT>
  const OptionsProviderT &get() const {
    if constexpr (std::is_same_v<OptionsProviderT, DebugOptions>) {
      if (hasDebugOptions())
        return *this->debugOptions;
      llvm::report_fatal_error(
          "debug options are not enabled on the task options instance");
    } else {
      return *std::get<std::unique_ptr<OptionsProviderT>>(optionProviders);
    }
  }

  /// Access provider value.
  template <typename OptionsProviderT>
  OptionsProviderT &get() {
    if constexpr (std::is_same_v<OptionsProviderT, DebugOptions>) {
      if (hasDebugOptions())
        return *debugOptions;
      llvm::report_fatal_error(
          "debug options are not enabled on the task options instance");
    } else {
      return *std::get<std::unique_ptr<OptionsProviderT>>(optionProviders);
    }
  }

private:
  std::tuple<std::unique_ptr<Providers>...> optionProviders;
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_OPTIONSPROVIDERS
