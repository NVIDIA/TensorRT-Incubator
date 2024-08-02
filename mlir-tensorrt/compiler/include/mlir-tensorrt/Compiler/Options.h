//===- Options.h ------------------------------------------------*- C++ -*-===//
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

#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace mlirtrt::compiler {

/// DebugOptions are options that are common to different compiler API
/// interfaces.
struct DebugOptions {
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

  void addToOptions(mlir::OptionsContext &context) {
    context.addOption("mlir-print-ir-tree-dir", dumpIRPath, llvm::cl::init(""));
    context.addOption("debug", enableLLVMDebugFlag);
    context.addList<std::string>("debug-only", llvmDebugTypes,
                                 llvm::cl::ZeroOrMore,
                                 llvm::cl::CommaSeparated);
  }
};

} // namespace mlirtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_OPTIONS
