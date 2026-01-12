//===- Support.h ------------------------------------------------*- C++ -*-===//
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
/// Declarations for utilities that are shared betweeen API and
/// backend libraries.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_COMMON_DEBUG
#define MLIR_TENSORRT_RUNTIME_COMMON_DEBUG

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace mtrt {

//===----------------------------------------------------------------------===//
// Debugging and logging tools
//===----------------------------------------------------------------------===//

/// Prints the given printf-style formatted data to stderr if the 'runtime'
/// debug module is enabled. Has no effect in non-assert builds.
/// Note that we prepend a space to assist with readability when the logs are
/// prefixed by other text when wrapped by another runtime system (e.g.
/// 'mpirun').
#define MTRT_DBGF(fmt, ...)                                                    \
  DEBUG_WITH_TYPE("runtime", fprintf(stderr, "%s:%d [runtime][DBG] " fmt "\n", \
                                     __FILE__, __LINE__, __VA_ARGS__))

template <typename... Args>
void _MTRT_DBGV(const char *fmt, const char *file, int64_t line,
                Args &&...args) {
  DEBUG_WITH_TYPE(
      "runtime", llvm::dbgs() << file << ":" << line << " [runtime][DBG] "
                              << llvm::formatv(fmt, std::forward<Args>(args)...)
                              << "\n");
}

#define MTRT_ERRF(fmt, ...)                                                    \
  fprintf(stderr, "%s:%d " fmt "\n", __FILE__, __LINE__, __VA_ARGS__)

/// Prints a warning message where "format" and "...args" are pased to
/// llvm::formatv. The message is prefixed by `<file>:<line> [WARN] `.
template <typename... Args>
void _MTRT_WARNV(const char *format, const char *file, int64_t line,
                 Args &&...args) {
  llvm::dbgs() << file << ":" << line << "[runtime][WARN] "
               << llvm::formatv(format, std::forward<Args>(args)...).str();
}

/// Prints an error message where "format" and "...args" are pased to
/// llvm::formatv. The message is prefixed by `<file>:<line> [ERR] `.
template <typename... Args>
void _MTRT_ERRV(const char *format, const char *file, int64_t line,
                Args &&...args) {
  llvm::errs() << file << ":" << line << "[runtime][ERR] "
               << llvm::formatv(format, std::forward<Args>(args)...).str();
}

/// Prints a warning message where "format" and "...args" are passed to
/// llvm::formatv.
#define MTRT_WARNV(format, ...)                                                \
  do {                                                                         \
    ::mtrt::_MTRT_WARNV(format, __FILE__, __LINE__, __VA_ARGS__);              \
  } while (false)

/// Prints an error message where "format" and "...args" are passed to
/// llvm::formatv. The message is prefixed by `<file>:<line> [ERR] `.
#define MTRT_ERRV(format, ...)                                                 \
  do {                                                                         \
    ::mtrt::_MTRT_ERRV(format, __FILE__, __LINE__, __VA_ARGS__);               \
  } while (false)

/// Prints an error message where "format" and "...args" are passed to
/// llvm::formatv. The message is prefixed by `<file>:<line> [DBG] `.
#define MTRT_DBG(format, ...)                                                  \
  do {                                                                         \
    ::mtrt::_MTRT_DBGV(format, __FILE__, __LINE__, __VA_ARGS__);               \
  } while (false)

} // namespace mtrt

#endif // MLIR_TENSORRT_RUNTIME_COMMON_DEBUG
