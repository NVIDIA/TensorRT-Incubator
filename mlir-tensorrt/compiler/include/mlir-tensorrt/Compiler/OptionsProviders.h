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
  /// A directory path where the IR will be dumped during compilation
  /// using the `mlir-print-ir-tree-dir` mechanism.
  std::string dumpIRPath = "";

  /// Whether the LLVM 'debug' flag that enables execution of code guarded by
  /// the `LLVM_DEBUG` macro should be set to 'on'. This results in very verbose
  /// output from the compiler dumped to stderr.
  bool enableLLVMDebugFlag = false;

  /// A set of names to be given to the LLVM 'debug types' option, akin to
  /// setting
  /// `-debug-types=...` from the command line.
  mlir::SmallVector<std::string> llvmDebugTypes = {};

public:
  void addToOptions(mlir::OptionsContext &context) {
    context.addOption("mlir-print-ir-tree-dir", dumpIRPath, llvm::cl::init(""));
    context.addOption("debug", enableLLVMDebugFlag);
    context.addList<std::string>("debug-only", llvmDebugTypes,
                                 llvm::cl::ZeroOrMore,
                                 llvm::cl::CommaSeparated);
  }
};

struct ExecutorOptions : public OptionsProvider<ExecutorOptions> {
public:
  /// The host index bit-width.
  int64_t indexBitwidth{64};

  /// Whether to pass memref's as struct/table in function calls.
  bool usePackedMemRefCConv{true};

public:
  void addToOptions(mlir::OptionsContext &context) {
    context.addOption("executor-index-bitwidth", indexBitwidth,
                      llvm::cl::init(64));
  }
};

struct DeviceOptions : public OptionsProvider<DeviceOptions> {
public:
  DeviceInfo info;

  /// Whether to ignore `deviceX` options and instead infer them from the GPUs
  /// on the host system running the compilation.
  bool shouldInferFromHost = false;
  Status inferFromHost();

public:
  void addToOptions(mlir::OptionsContext &context) {
    context.addOption(
        "device-compute-capability", info.computeCapability, llvm::cl::init(60),
        llvm::cl::desc("Sets the device compute capbility. Only relevant "
                       "if '--device-infer-from-host=false'"));
    context.addOption("device-max-shared-memory-per-block-kb",
                      info.maxSharedMemoryPerBlockKb, llvm::cl::init(48));
    context.addOption("device-max-registers-per-block",
                      info.maxRegistersPerBlock, llvm::cl::init(65536));
    context.addOption("device-infer-from-host", shouldInferFromHost,
                      llvm::cl::init(true),
                      llvm::cl::desc("Infers device information from host"));
  }

  llvm::Error finalizeImpl();
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_OPTIONS
