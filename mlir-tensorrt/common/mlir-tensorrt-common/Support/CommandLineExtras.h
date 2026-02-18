//===- CommandLineExtras.h --------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for classes related to `llvm::cl::` options types.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_SUPPORT_COMMANDLINEEXTRAS
#define MLIR_TENSORRT_COMMON_SUPPORT_COMMANDLINEEXTRAS

#include "llvm/Support/CommandLine.h"

namespace mlir {
/// A llvm::cl::opt parser for turning strings like "1024gb" into a number of
/// bytes. Allowed suffixes are strings like 'gb', 'GiB', 'kb', 'mb', 'b' (case
/// insensitive, we interpret both 'b|B' as meaning "byte"). This example comes
/// straight from the LLVM documentation
/// (https://llvm.org/docs/CommandLine.html#writing-a-custom-parser).
struct ByteSizeParser : public llvm::cl::parser<std::optional<uint64_t>> {
  using llvm::cl::parser<std::optional<uint64_t>>::parser;
  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, std::optional<uint64_t> &Val);
};
} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_SUPPORT_COMMANDLINEEXTRAS