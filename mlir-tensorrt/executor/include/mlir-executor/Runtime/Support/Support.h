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
#include <iostream>

namespace mlirtrt::runtime {

//===----------------------------------------------------------------------===//
// Debugging and logging tools
//===----------------------------------------------------------------------===//

#define MTRT_DBGF(fmt, ...)                                                    \
  DEBUG_WITH_TYPE("runtime", fprintf(stderr, "%s:%d " fmt "\n", __FILE__,      \
                                     __LINE__, __VA_ARGS__))

template <typename... Args>
void MTRT_DBG(const char *fmt, Args... args) {
  DEBUG_WITH_TYPE(
      "runtime",
      fprintf(stderr, "[runtime] %s\n",
              llvm::formatv(fmt, std::forward<Args>(args)...).str().c_str()));
}

#define MTRT_ERRF(fmt, ...)                                                    \
  fprintf(stderr, "%s:%d " fmt "\n", __FILE__, __LINE__, __VA_ARGS__)

/// Prints a warning message where "format" and "...args" are pased to
/// llvm::formatv. The message is prefixed by `<file>:<line> [WARN] `.
template <typename... Args>
void _MTRT_WARNV(const char *format, const char *file, int64_t line,
                 Args &&...args) {
  llvm::dbgs() << file << ":" << line << " [WARN] "
               << llvm::formatv(format, std::forward<Args>(args)...).str();
}

/// Prints a warning message where "format" and "...args" are passed to
/// llvm::formatv.
#define MTRT_WARNV(format, ...)                                                \
  do {                                                                         \
    _MTRT_WARNV(format, __FILE__, __LINE__, __VA_ARGS__);                      \
  } while (false)

#ifndef MTRT_RETURN_IF_ERROR
#define MTRT_RETURN_IF_ERROR(x, y)                                             \
  do {                                                                         \
    if (!x.ok()) {                                                             \
      std::cerr << x.getStatus() << std::endl;                                 \
      return y.getStatus();                                                    \
    }                                                                          \
  } while (false)
#endif

//===----------------------------------------------------------------------===//
// Utilities for "unreachable"
//===----------------------------------------------------------------------===//

/// This function is called when a fatal error is reached, and the program's
/// only options is to abort. It will print the message and file/line# to
/// stderr before aborting.
[[noreturn]] inline void unreachable_internal(const char *msg, const char *file,
                                              unsigned line) {
  std::cerr << (msg != nullptr ? msg : "") << "\n"
            << "UNREACHABLE executed";
  if (file != nullptr)
    std::cerr << " at " << file << ":" << line;
  std::cerr << "!\n";
  abort();
}

/// An implementation of `std::unreachable` (C++ 23), according to
/// https://en.cppreference.com/w/cpp/utility/unreachable. This function should
/// be used on release build code paths instead of `unreachable_internal`
[[noreturn]] inline void unreachable_internal_opt(const char *msg = nullptr,
                                                  const char *file = nullptr,
                                                  unsigned line = 0) {
  unreachable_internal(msg, file, line);
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#else
#if defined(_MSC_VER) // MSVC
  __assume(false);
#endif
#endif
}

} // namespace mlirtrt::runtime

/// Marks that the current location is not supposed to be reachable. If the code
/// is reached in a non-release build, prints a message and aborts.
#ifndef NDEBUG
#define mlir_trt_unreachable(msg)                                              \
  ::mlirtrt::runtime::unreachable_internal(msg, __FILE__, __LINE__)
#else
#define mlir_trt_unreachable(msg)                                              \
  ::mlirtrt::runtime::unreachable_internal_opt(msg, __FILE__, __LINE__)
#endif

#endif // MLIR_TENSORRT_RUNTIME_COMMON_DEBUG
