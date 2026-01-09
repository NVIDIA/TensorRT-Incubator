//===- Options.h ---------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#ifndef MLIR_TENSORRT_COMPILER_OPTIONS
#define MLIR_TENSORRT_COMPILER_OPTIONS

#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/InputPipelines/LinalgInputPipeline.h"
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <string>

namespace mtrt::compiler {

/// Phase describes the overall phases of the compilation pipeline. These
/// phases can be used to control which parts of the pipeline are populated
/// via the `--phase-start` and `--phase-end` options.
enum class Phase {
  Setup,         // populateSetupPipeline
  Input,         // populateInputPipeline
  Clustering,    // Input-kind specific clustering/segmentation
  Bufferization, // Plan bufferization pipeline
  Lowering       // Host target lowering (Executor/LLVM/EmitC)
};

namespace detail {
/// Returns the llvm::cl::values for the Phase enum.
inline auto createPhaseClOptions() {
  return llvm::cl::values(
      clEnumValN(Phase::Setup, "setup", "setup phase"),
      clEnumValN(Phase::Input, "input", "input phase"),
      clEnumValN(Phase::Clustering, "clustering",
                 "clustering/segmentation phase"),
      clEnumValN(Phase::Bufferization, "bufferization", "bufferization phase"),
      clEnumValN(Phase::Lowering, "lowering", "host target lowering phase"));
}
} // namespace detail

/// DebugOptions are options that are common to different compiler API
/// interfaces.
struct DebugOptions : public mlir::OptionsGroup {
  static llvm::cl::OptionCategory category;

  /// DebugOptions has an explicit constructor since it asserts in the
  /// constructor that the OptionsSet has a local scope. These options mirror
  /// the MLIR global CL options of the same spelling, so attempting to register
  /// it against the global scope will cause a duplicate registration error.
  DebugOptions(mlir::CLOptionScope &ctx);

  //===--------------------------------------------------------------------===//
  // Crash Reproducer Generator
  //===--------------------------------------------------------------------===//
  Option<std::string> reproducerFile{
      this->ctx, "mlir-pass-pipeline-crash-reproducer",
      llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                     " if the pass manager crashes or fails"),
      llvm::cl::cat(category)};
  Option<bool> localReproducer{
      this->ctx, "mlir-pass-pipeline-local-reproducer",
      llvm::cl::desc("When generating a crash reproducer, attempt to generated "
                     "a reproducer with the smallest pipeline."),
      llvm::cl::init(false), llvm::cl::cat(category)};

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//

  Option<bool> printBeforeAll{this->ctx, "mlir-print-ir-before-all",
                              llvm::cl::desc("Print IR before each pass"),
                              llvm::cl::init(false), llvm::cl::cat(category)};
  Option<bool> printAfterAll{this->ctx, "mlir-print-ir-after-all",
                             llvm::cl::desc("Print IR after each pass"),
                             llvm::cl::init(false), llvm::cl::cat(category)};
  Option<bool> printAfterChange{
      this->ctx, "mlir-print-ir-after-change",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      llvm::cl::init(false), llvm::cl::cat(category)};
  Option<bool> printAfterFailure{
      this->ctx, "mlir-print-ir-after-failure",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the pass failed"),
      llvm::cl::init(false), llvm::cl::cat(category)};
  Option<bool> printModuleScope{
      this->ctx, "mlir-print-ir-module-scope",
      llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                     "always print the top-level operation"),
      llvm::cl::init(false), llvm::cl::cat(category)};
  Option<std::string> printTreeDir{
      this->ctx, "mlir-print-ir-tree-dir",
      llvm::cl::desc("When printing the IR before/after a pass, print file "
                     "tree rooted at this directory. Use in conjunction with "
                     "mlir-print-ir-* flags"),
      llvm::cl::cat(category)};

  //===----------------------------------------------------------------------===//
  // Printing Flags
  //===----------------------------------------------------------------------===//

  Option<unsigned> elideElementsAttrIfLarger{
      this->ctx, "mlir-elide-elementsattrs-if-larger",
      llvm::cl::desc("Elide ElementsAttrs with \"...\" that have "
                     "more elements than the given upper limit"),
      llvm::cl::cat(category)};

  Option<unsigned> elideResourceStringsIfLarger{
      this->ctx, "mlir-elide-resource-strings-if-larger",
      llvm::cl::desc(
          "Elide printing value of resources if string is too long in chars."),
      llvm::cl::cat(category)};

  //===--------------------------------------------------------------------===//
  // Pass Statistics
  //===--------------------------------------------------------------------===//
  Option<bool> passStatistics{
      this->ctx, "mlir-pass-statistics",
      llvm::cl::desc("Display the statistics of each pass"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  //===--------------------------------------------------------------------===//
  // Pass Timing
  //===--------------------------------------------------------------------===//
  Option<bool> enableTiming{
      this->ctx, "mlir-timing",
      llvm::cl::desc(
          "Time each pass and print to stderr after the pipeline completes"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  //===----------------------------------------------------------------------===//
  // Debug Printing
  //===----------------------------------------------------------------------===//

  /// Whether the LLVM 'debug' flag that enables execution of code guarded by
  /// the `LLVM_DEBUG` macro should be set to 'on'. This results in very verbose
  /// output from the compiler dumped to stderr.
  Option<bool> enableLLVMDebugFlag{this->ctx, "debug", llvm::cl::init(false),
                                   llvm::cl::cat(category)};

  /// A set of names to be given to the LLVM 'debug types' option, akin to
  /// setting
  /// `-debug-types=...` from the command line.
  ListOption<std::string> llvmDebugTypes{
      this->ctx, "debug-only", llvm::cl::ZeroOrMore, llvm::cl::cat(category)};

  /// Apply these options to the current pass manager.
  void applyToPassManager(mlir::PassManager &pm) const;
};

//===----------------------------------------------------------------------===//
// ExecutorOptions
//===----------------------------------------------------------------------===//

struct ExecutorOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<int64_t> indexBitwidth{
      this->ctx, "executor-index-bitwidth", llvm::cl::init(64),
      llvm::cl::desc("executor index bitwidth"), llvm::cl::cat(category)};
};

//===----------------------------------------------------------------------===//
// EmitCOptions
//===----------------------------------------------------------------------===//

struct EmitCOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<bool> emitSupportFiles{
      this->ctx, "emitc-emit-support-files", llvm::cl::init(false),
      llvm::cl::desc(
          "Emit EmitC support files (runtime sources/headers, example CMake, "
          "and a test driver) into the artifacts directory."),
      llvm::cl::cat(category)};

  Option<bool> emitRuntimeFiles{
      this->ctx, "emitc-emit-runtime-files", llvm::cl::init(false),
      llvm::cl::desc("Emit the required subset of StandaloneCPP runtime source "
                     "and header files needed by the generated C++ code."),
      llvm::cl::cat(category)};

  Option<bool> emitCMakeFile{
      this->ctx, "emitc-emit-cmake-file", llvm::cl::init(false),
      llvm::cl::desc("Emit an example CMake file for compiling the generated "
                     "C++ code (and emitted runtime files, if requested)."),
      llvm::cl::cat(category)};

  Option<bool> emitTestDriver{
      this->ctx, "emitc-emit-test-driver", llvm::cl::init(false),
      llvm::cl::desc("Emit a C++ test driver source file for building an "
                     "executable that includes and runs the generated C++ "
                     "code."),
      llvm::cl::cat(category)};

  Option<bool> wrapModuleInEmitCClass{
      this->ctx, "emitc-wrap-in-class", llvm::cl::init(false),
      llvm::cl::desc("Wrap the module in an EmitC class"),
      llvm::cl::cat(category)};
};

//===----------------------------------------------------------------------===//
// DeviceOptions
//===----------------------------------------------------------------------===//

struct DeviceOptions : public mlir::OptionsGroup {
  DeviceOptions(mlir::CLOptionScope &ctx);

  static llvm::cl::OptionCategory category;

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
                     "them from the host GPU"),
      llvm::cl::cat(category)};

  Option<int64_t> computeCapability{
      this->ctx, "device-compute-capability", llvm::cl::init(60),
      llvm::cl::desc("Sets the device compute capability. Only relevant "
                     "if '--device-infer-from-host=false'"),
      llvm::cl::cat(category)};
  Option<int64_t> maxSharedMemoryPerBlockKb{
      this->ctx, "device-max-shared-memory-per-block-kb", llvm::cl::init(48),
      llvm::cl::cat(category)};
  Option<uint64_t> maxRegistersPerBlock{
      this->ctx, "device-max-registers-per-block", llvm::cl::init(65536),
      llvm::cl::cat(category)};

private:
  /// Stores host device info. This is populated by the callback of
  /// `shouldInferFromHost`. If present, then it will also override the other
  /// options in their callbacks.
  std::optional<DeviceInfo> hostDeviceInfo{};
};

//===----------------------------------------------------------------------===//
// BufferizationOptions
//===----------------------------------------------------------------------===//

/// Encapsulates options related to the bufferization pipeline.
struct BufferizationOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<bool> forceEntrypointsReturnAllocs{
      this->ctx, "force-entrypoints-return-allocs", llvm::cl::init(false),
      llvm::cl::desc(
          "Require entrypoint functions to return allocations corresponding to "
          "the original tensor results, otherwise they are transformed into "
          "destination arguments whenever possible."),
      llvm::cl::cat(category)};

  Option<bool> deallocationPrivateFuncDynamicOwnership{
      this->ctx, "deallocation-private-func-dynamic-ownership",
      llvm::cl::init(false),
      llvm::cl::desc(
          "Overrides the default private function ABI in the buffer "
          "deallocation pipeline to allow for dynamic ownership of memref "
          "arguments and returned memrefs."),
      llvm::cl::cat(category)};

  Option<bool> enablePinnedMemoryPromotion{
      this->ctx, "enable-pinned-memory-promotion", llvm::cl::init(true),
      llvm::cl::desc("Enable promotion of host buffers to pinned memory using "
                     "heuristics."),
      llvm::cl::cat(category)};

  Option<bool> enableBufferLoopHoisting{
      this->ctx, "enable-buffer-loop-hoisting", llvm::cl::init(true),
      llvm::cl::desc("Enable buffer hoisting out of loops."),
      llvm::cl::cat(category)};

  Option<bool> enableBufferHoisting{
      this->ctx, "enable-buffer-hoisting", llvm::cl::init(true),
      llvm::cl::desc("Enable buffer hoisting."), llvm::cl::cat(category)};
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
// TensorRTOptions
//===----------------------------------------------------------------------===//

/// TensorRT-related compilation options shared across all input kinds.
struct TensorRTOptions : public mlir::OptionsGroup {
  TensorRTOptions(mlir::CLOptionScope &ctx) : OptionsGroup(ctx) {
    /// Translation options are only registered in local scopes.
    if (!this->ctx.isGlobalScope())
      translationOptions =
          std::make_unique<mlir::tensorrt::TensorRTTranslationOptions>(
              this->ctx);
  }

  static llvm::cl::OptionCategory category;

  Option<bool> forceDefaultSliceInBounds{
      this->ctx, "tensorrt-force-default-slice-in-bounds",
      llvm::cl::init(false),
      llvm::cl::desc("Constrain dynamic offset/sizes for default slice ops so "
                     "that accesses will be in bounds"),
      llvm::cl::cat(category)};

  Option<bool> tensorrtPreferEinsum{
      this->ctx, "tensorrt-prefer-einsum", llvm::cl::init(true),
      llvm::cl::desc(
          "Prefer 'tensorrt.einsum' over 'tensorrt.matrix_multiply'"),
      llvm::cl::cat(category)};

  /// Return the translation options.
  const mlir::tensorrt::TensorRTTranslationOptions &
  getTranslationOptions() const {
    if (translationOptions)
      return *translationOptions;
    return mlir::tensorrt::TensorRTTranslationOptions::fromCLFlags();
  }

private:
  std::unique_ptr<mlir::tensorrt::TensorRTTranslationOptions>
      translationOptions{nullptr};
};

//===----------------------------------------------------------------------===//
// KernelGenOptions
//===----------------------------------------------------------------------===//

/// Kernel generation and device-code compilation options shared across input
/// kinds.
struct KernelGenOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  /// Directory where PTX data will be saved for debugging.
  Option<std::string> dumpPtxDir{
      this->ctx, "dump-ptx-dir", llvm::cl::init(""),
      llvm::cl::desc("path to directory where PTX files will be dumped"),
      llvm::cl::cat(category)};

  ListOption<std::string> generatorBenefit{
      this->ctx, "generator-benefit",
      llvm::cl::desc("A list of 'name:benefit' pairs to adjust generator "
                     "benefits for kernel generation."),
      llvm::cl::cat(category)};
};

//===----------------------------------------------------------------------===//
// OptimizationOptions
//===----------------------------------------------------------------------===//

struct OptimizationOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<int64_t> unrollThreshold{
      this->ctx, "scf-unroll-threshold", llvm::cl::init(100),
      llvm::cl::desc("Cost threshold for loop unrolling."),
      llvm::cl::cat(category)};

  Option<bool> hoistAllocsToGlobals{
      this->ctx, "hoist-allocs-to-globals", llvm::cl::init(true),
      llvm::cl::desc("Hoist large local allocations to static globals when "
                     "possible."),
      llvm::cl::cat(category)};
};

//===----------------------------------------------------------------------===//
// MainOptions (pipeline options)
//===----------------------------------------------------------------------===//

/// MainOptions is the option container for the unified compilation pipeline.
/// It aggregates the common pipeline options plus additional option providers
/// (e.g. `TensorRTOptions`, `KernelGenOptions`) and loads all registered
/// compiler extensions into the options context.
class MainOptions : public mlir::CLOptionScope,
                    public llvm::ThreadSafeRefCountedBase<MainOptions> {
protected:
  ExtensionList extensions;

  template <typename... Ts>
  using SubGroupsTuple = std::tuple<std::unique_ptr<Ts>...>;

  // clang-format off
  /// Attach option subgroups to this scope.
  using SubGroups = mlir::options_group_tuple<
    BufferizationOptions,
    DeviceOptions,
    EmitCOptions,
    ExecutorOptions,
    KernelGenOptions,
    LinalgInputOptions,
    OptimizationOptions,
    mlir::plan::PlanClusteringOptions,
    StablehloInputOptions,
    TensorRTOptions
  >;
  // clang-format on
  SubGroups groups;

  /// Treat DebugOptions separately since it is only constructed for
  /// locally-scoped OptionSets.
  std::unique_ptr<DebugOptions> debugOptions{nullptr};

  static llvm::cl::OptionCategory category;

  /// Whether the options have been finalized.
  bool finalized{false};

public:
  //===----------------------------------------------------------------------===//
  // Options common to all tasks
  //===----------------------------------------------------------------------===//

  /// Output directory or file name for the translated host-target output.
  ///
  /// This option is only registered in the global CLI scope. It is
  /// intentionally not registered for locally-scoped option parsing (e.g.
  /// `MainOptions::fromString`) to avoid duplicate option registrations and to
  /// keep local parsing focused on pipeline behavior rather than tool output.
  std::optional<Option<std::string>> outputPath;

  /// Returns the output path value. If the option is not registered (local
  /// scope), returns ".".
  llvm::StringRef getOutputPath() const {
    if (outputPath)
      return outputPath->getValue();
    return ".";
  }

  Option<HostTarget> hostTarget{
      *this,
      "host-target",
      llvm::cl::init(HostTarget::Executor),
      llvm::cl::desc(
          "specifies the target compilation format for host functions"),
      llvm::cl::values(
          clEnumValN(HostTarget::Executor, "executor",
                     "compile host code to MLIR-TRT interpretable executable"),
          clEnumValN(HostTarget::LLVM, "llvm", "compile host code to LLVM IR"),
          clEnumValN(HostTarget::EmitC, "emitc", "compile host code to C++")),
      llvm::cl::cat(category)};

  Option<uint32_t> runtimeABIVersion{
      *this, "abi-version", llvm::cl::init(1),
      llvm::cl::desc("specifies the Executor ABI version"),
      llvm::cl::cat(category)};

  Option<std::string> artifactsDirectory{
      *this, "artifacts-dir", llvm::cl::init(""),
      llvm::cl::desc("Specifies where large artifacts can be offloaded as "
                     "external files referenced by filename in the IR"),
      llvm::cl::cat(category)};

  Option<std::string> entrypoint{*this, "entrypoint", llvm::cl::init("main"),
                                 llvm::cl::desc("entrypoint function name"),
                                 llvm::cl::cat(category)};

  ListOption<std::string> defaultBackends{
      *this, "backends",
      // clang-format off
      llvm::cl::list_init<std::string>({
      "#plan.tensorrt_backend<disallow_shape_tensor_calculations=false, benefit=3>",
      "#plan.kernel_backend<benefit=2>",
      "#plan.host_backend<benefit=1>"
      }),
      // clang-format on
      llvm::cl::desc(
          "Default list of plan backends if none specified on the module."),
      llvm::cl::cat(category), llvm::cl::CommaSeparated};

  Option<std::string> defaultMemorySpace{
      *this, "default-memory-space",
      llvm::cl::init("#plan.memory_space<device>"),
      llvm::cl::desc("Default bufferization memory space for the module"),
      llvm::cl::cat(category)};

  /// Filter the default backends to remove any backends that contain the given
  /// substring.
  void filterBackends(llvm::StringRef substring) {
    auto filteredBackends = llvm::filter_to_vector(
        defaultBackends, [substring](llvm::StringRef backend) {
          return !backend.contains(substring);
        });
    defaultBackends.assign(filteredBackends);
  }

  Option<bool> disableAllExtensions{
      *this,
      "disable-all-extensions",
      llvm::cl::init(false),
      llvm::cl::desc("disable all extensions"),
      llvm::cl::cat(category),
      llvm::cl::callback([this](const bool &value) {
        if (value) {
          this->disableTensorRTExtension.setValue(true);
          this->disableKernelGenExtension.setValue(true);
          this->defaultBackends.assign({"#plan.host_backend<benefit=1>"});
          this->defaultMemorySpace.setValue("#plan.memory_space<host>");
        }
      })};

  Option<mlir::plan::InputKind> inputKind{
      *this,
      "input",
      llvm::cl::desc("the kind of input IR to compile"),
      llvm::cl::init(mlir::plan::InputKind::Stablehlo),
      llvm::cl::values(mlir::plan::detail::createInputKindClOptions()),
      llvm::cl::cat(category)};

  Option<bool> enableV2constantFolding{
      *this, "enable-v2-constant-folding", llvm::cl::init(true),
      llvm::cl::desc(
          "Enable v2 constant folding (requires KernelGen extension)"),
      llvm::cl::cat(category)};

  //===--------------------------------------------------------------------===//
  // Phase control
  //===--------------------------------------------------------------------===//

  /// Starting phase for pipeline population. All phases before this are
  /// skipped.
  Option<Phase> phaseStart{
      *this,
      "phase-start",
      llvm::cl::init(Phase::Setup),
      llvm::cl::desc("Start populating passes from this phase"),
      detail::createPhaseClOptions(),
      llvm::cl::cat(category)};

  /// Ending phase for pipeline population. All phases after this are skipped.
  Option<Phase> phaseEnd{
      *this,
      "phase-end",
      llvm::cl::init(Phase::Lowering),
      llvm::cl::desc("Stop populating passes after this phase"),
      detail::createPhaseClOptions(),
      llvm::cl::cat(category)};

  //===--------------------------------------------------------------------===//
  // Extensions toggles
  //===--------------------------------------------------------------------===//

  Option<bool> disableTensorRTExtension{
      *this,
      "disable-tensorrt-extension",
      llvm::cl::init(false),
      llvm::cl::desc("Disable TensorRT integration extension passes (StableHLO "
                     "path)."),
      llvm::cl::cat(category),
      llvm::cl::callback([this](const bool &value) {
        if (value)
          this->filterBackends("tensorrt_backend");
      })};

  Option<bool> disableKernelGenExtension{
      *this,
      "disable-kernel-gen-extension",
      llvm::cl::init(false),
      llvm::cl::desc("Disable KernelGen extension passes"),
      llvm::cl::callback([this](const bool &value) {
        // Force V2 constant folding to be disabled if the KernelGen extension
        // is disabled.
        if (value) {
          this->enableV2constantFolding.setValue(false);
          this->filterBackends("kernel_backend");
        }
      }),
      llvm::cl::cat(category)};

  ///===--------------------------------------------------------------------===//
  // Methods
  ///===--------------------------------------------------------------------===//

  /// Parse the options from a string within a "local scope" and finalize them.
  static StatusOr<llvm::IntrusiveRefCntPtr<MainOptions>>
  fromString(llvm::StringRef optionString, ExtensionList extensions);

  /// Default-construct the options, attaching them to the global CL scope.
  /// DebugOptions are explicitly not constructed.
  MainOptions(mlir::CLOptionScope::GlobalScope, ExtensionList extensions)
      : mlir::CLOptionScope(GlobalScope{}), extensions(std::move(extensions)),
        groups(mlir::make_options_group_tuple<SubGroups>(*this)) {
    // Register tool output path only in the global scope.
    if (this->isGlobalScope()) {
      outputPath.emplace(
          *this, "o", llvm::cl::desc("<output directory or file name>"),
          llvm::cl::init("."), llvm::cl::value_desc("directory or file"),
          llvm::cl::cat(category));
    }
    this->extensions.loadExtensions(*this);
  }

  /// Default-construct the options, attaching them to a local CL scope.
  /// DebugOptions are constructed and attached to the instance.
  MainOptions(mlir::CLOptionScope::LocalScope, ExtensionList extensions)
      : mlir::CLOptionScope(LocalScope{}), extensions(std::move(extensions)),
        groups(mlir::make_options_group_tuple<SubGroups>(*this)),
        debugOptions(std::make_unique<DebugOptions>(*this)) {
    this->extensions.loadExtensions(*this);
  }

  /// Validate the options. Does not mutate the values in any way.
  Status validate() const;

  /// Finalize the MainOptions. This should be called after invoking the parse
  /// method or manually populating the option values. It may update some values
  /// in-place to coerce them to valid states if defaults are incompatible with
  /// user-specified options. Anytime this occurs, a remark will be emitted so
  /// that it is clearly visible to the user. Finally, it will call validate()
  /// again to ensure options are well-formed.
  Status finalize();

  /// Returns true if the options have been finalized.
  bool isFinalized() const { return finalized; }

  virtual ~MainOptions() = default;

  std::optional<llvm::hash_code> getHash() const;

  /// Get the Extensions associated with this pipeline.
  const ExtensionList &getExtensions() const { return extensions; }

  /// Access provider value.
  template <typename OptionsGroupT>
  const OptionsGroupT &get() const {
    return *std::get<std::unique_ptr<OptionsGroupT>>(groups);
  }

  /// Access provider value.
  template <typename OptionsGroupT>
  OptionsGroupT &get() {
    return *std::get<std::unique_ptr<OptionsGroupT>>(groups);
  }

  /// Get the debug options.
  const DebugOptions &getDebugOptions() const {
    assert(debugOptions &&
           "debug options are not enabled on the pipeline options instance");
    return *debugOptions;
  }

  /// Returns true if debug options are enabled.
  bool hasDebugOptions() const { return debugOptions != nullptr; }
};

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_OPTIONS
