//===- OptionsProviders.h ---------------------------------------*- C++ -*-===//
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
/// Data structures and functions for manipulating compiler options.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_OPTIONS
#define MLIR_TENSORRT_COMPILER_OPTIONS

#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlirtrt::compiler {

// Use SFINAE to check whether the `finalizeImpl()` method is defined on a type.
// If it is, the specialization (where the value is true) will be the better
// match. Otherwise, we'll get the default value of false.
template <typename, typename = void>
constexpr bool has_finalize_impl_v = false;

template <typename T>
constexpr bool has_finalize_impl_v<
    T, std::void_t<decltype(std::declval<T>().finalizeImpl())>> = true;

// We use CRTP here so we can call `finalizeImpl()` if it's defined or provide
// a default implementation otherwise.
template <typename Derived>
struct OptionsProvider {
  using OmitFromCLI = mlir::OptionsContext::OmitFromCLI;

  OptionsProvider(mlir::OptionsContext &ctx) : ctx(ctx) {}

  // We don't allow move construction since the actual ptrs/locations of
  // individual member elements of an OptionsProvider are captured into the
  // OptionsContext. If the OptionsContext is populated upon construction,
  // moving can change the memory location of the owned values, which will cause
  // a crash later on. This is in particular can happen if you are constructing
  // a tuple of `OptionsProviders`. Since we are deleting the move constructor,
  // one must instead use a tuple of `unique_ptr<OptionsProviders...>`.
  OptionsProvider(OptionsProvider &&) = delete;

  mlir::OptionsContext &ctx;

  template <typename T, typename... Mods>
  using Option = mlir::OptionsContext::Option<T, Mods...>;
  template <typename T, typename... Mods>
  using ListOption = mlir::OptionsContext::ListOption<T, Mods...>;

  /// Modifies options after parsing. This is required since we may need
  /// to make changes to options based on the values of other options.
  /// Do *not* override this method; instead, implement `finalizeImpl()`.
  llvm::Error finalize() {
    if constexpr (has_finalize_impl_v<Derived>)
      return static_cast<Derived *>(this)->finalizeImpl();
    else
      return llvm::Error::success();
  }
};

/// DebugOptions are options that are common to different compiler API
/// interfaces.
struct DebugOptions : public OptionsProvider<DebugOptions> {
public:
  using OptionsProvider::OptionsProvider;
  //===--------------------------------------------------------------------===//
  // Crash Reproducer Generator
  //===--------------------------------------------------------------------===//
  Option<std::string> reproducerFile{
      &this->ctx, "mlir-pass-pipeline-crash-reproducer",
      llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                     " if the pass manager crashes or fails"),
      OmitFromCLI{}};
  Option<bool> localReproducer{
      &this->ctx, "mlir-pass-pipeline-local-reproducer",
      llvm::cl::desc("When generating a crash reproducer, attempt to generated "
                     "a reproducer with the smallest pipeline."),
      llvm::cl::init(false), OmitFromCLI{}};

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//

  Option<bool> printBeforeAll{&this->ctx, "mlir-print-ir-before-all",
                              llvm::cl::desc("Print IR before each pass"),
                              llvm::cl::init(false), OmitFromCLI{}};
  Option<bool> printAfterAll{&this->ctx, "mlir-print-ir-after-all",
                             llvm::cl::desc("Print IR after each pass"),
                             llvm::cl::init(false), OmitFromCLI{}};
  Option<bool> printAfterChange{
      &this->ctx, "mlir-print-ir-after-change",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      llvm::cl::init(false), OmitFromCLI{}};
  Option<bool> printAfterFailure{
      &this->ctx, "mlir-print-ir-after-failure",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the pass failed"),
      llvm::cl::init(false), OmitFromCLI{}};
  Option<bool> printModuleScope{
      &this->ctx, "mlir-print-ir-module-scope",
      llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                     "always print the top-level operation"),
      llvm::cl::init(false), OmitFromCLI{}};
  Option<std::string> printTreeDir{
      &this->ctx, "mlir-print-ir-tree-dir",
      llvm::cl::desc("When printing the IR before/after a pass, print file "
                     "tree rooted at this directory. Use in conjunction with "
                     "mlir-print-ir-* flags"),
      OmitFromCLI{}};

  //===--------------------------------------------------------------------===//
  // Pass Statistics
  //===--------------------------------------------------------------------===//
  Option<bool> passStatistics{
      &this->ctx, "mlir-pass-statistics",
      llvm::cl::desc("Display the statistics of each pass"),
      llvm::cl::init(false), OmitFromCLI{}};

  //===--------------------------------------------------------------------===//
  // Pass Timing
  //===--------------------------------------------------------------------===//
  Option<bool> enableTiming{
      &this->ctx, "mlir-timing",
      llvm::cl::desc(
          "Time each pass and print to stderr after the pipeline completes"),
      llvm::cl::init(false), OmitFromCLI{}};

  //===----------------------------------------------------------------------===//
  // Debug Printing
  //===----------------------------------------------------------------------===//

  /// Whether the LLVM 'debug' flag that enables execution of code guarded by
  /// the `LLVM_DEBUG` macro should be set to 'on'. This results in very verbose
  /// output from the compiler dumped to stderr.
  Option<bool> enableLLVMDebugFlag{&this->ctx, "debug", llvm::cl::init(false),
                                   OmitFromCLI{}};

  /// A set of names to be given to the LLVM 'debug types' option, akin to
  /// setting
  /// `-debug-types=...` from the command line.
  ListOption<std::string> llvmDebugTypes{
      &this->ctx, "debug-only", llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
      OmitFromCLI{}};

  /// If set to `true`, we populate the pass manager instrumentation using
  /// global MLIR CL options rather than the local options contained here.
  Option<bool> useGlobalCLPrintingOptions{&this->ctx, "use-global-cl-options",
                                          llvm::cl::init(false), OmitFromCLI{}};

  /// Apply these options to the current pass manager.
  void applyToPassManager(mlir::PassManager &pm) const;
};

struct ExecutorOptions : public OptionsProvider<ExecutorOptions> {
public:
  using OptionsProvider::OptionsProvider;

  Option<int64_t> indexBitwidth{&this->ctx, "executor-index-bitwidth",
                                llvm::cl::init(64),
                                llvm::cl::desc("executor index bitwidth")};

  Option<bool> usePackedMemRefCConv{
      &this->ctx, "executor-use-packed-memref-cconv", llvm::cl::init(true),
      llvm::cl::desc(
          "whether to use packed or unpacked memref calling convention")};
};

struct DeviceOptions : public OptionsProvider<DeviceOptions> {
public:
  using OptionsProvider::OptionsProvider;

  /// Device information. Members are manually bound to options in the
  /// constructor.
  DeviceInfo info;

  Option<bool> shouldInferFromHost{
      &this->ctx, "device-infer-from-host", llvm::cl::init(true),
      llvm::cl::desc("whether to ignore `deviceX` options and instead infer "
                     "them from the host GPU")};

public:
  DeviceOptions(mlir::OptionsContext &ctx) : OptionsProvider(ctx) {
    ctx.addOption(
        "device-compute-capability", info.computeCapability, llvm::cl::init(60),
        llvm::cl::desc("Sets the device compute capability. Only relevant "
                       "if '--device-infer-from-host=false'"));
    ctx.addOption("device-max-shared-memory-per-block-kb",
                  info.maxSharedMemoryPerBlockKb, llvm::cl::init(48));
    ctx.addOption("device-max-registers-per-block", info.maxRegistersPerBlock,
                  llvm::cl::init(65536));
  }

  llvm::Error finalizeImpl();
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_OPTIONS
