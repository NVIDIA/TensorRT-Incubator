//===- CommandLineExtras.cpp
//------------------------------------------------===//
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
/// Implementation of utilities related to `llvm::cl::` options.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Support/CommandLineExtras.h"

//===----------------------------------------------------------------------===//
// ByteSizeParser
//===----------------------------------------------------------------------===//

bool mlir::ByteSizeParser::parse(llvm::cl::Option &option,
                                 llvm::StringRef argName, llvm::StringRef arg,
                                 std::optional<uint64_t> &val) {
  val = std::nullopt;
  if (arg.empty() || arg.lower() == "none")
    return false;

  char *End;

  // Parse integer part, leaving 'End' pointing to the first non-integer char
  val = std::strtoull(arg.data(), &End, 0);

  while (1) {
    if (std::distance(arg.data(), static_cast<const char *>(End)) >=
        static_cast<int64_t>(arg.size()))
      return false;
    switch (*End++) {
    case 0:
      return false; // No error
    case 'i':       // Ignore the 'i' in KiB if people use that
    case 'b':
    case 'B': // Ignore B suffix
      break;
    case 'g':
    case 'G':
      *val *= 1024ull * 1024ull * 1024ull;
      break;
    case 'm':
    case 'M':
      *val *= 1024ull * 1024ull;
      break;
    case 'k':
    case 'K':
      *val *= 1024ull;
      break;
    default:
      // Print an error message if unrecognized character.
      return option.error("'" + arg +
                          "' value invalid for byte size argument!");
    }
  }
  return false;
}
