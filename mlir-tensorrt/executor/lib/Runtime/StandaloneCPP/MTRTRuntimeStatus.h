//===- MTRTRuntimeStatus.h --------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Lightweight, LLVM-free error handling utilities for the StandaloneCPP
/// EmitC support runtime.
///
/// Design:
/// - Functions return `mtrt::Status` (0 == success).
/// - On failure, the runtime stores a thread-local error message retrievable
/// via
///   `mtrt::get_last_error_message()`.
/// - No heap allocations are required in the common case.
//===----------------------------------------------------------------------===//

#ifndef MTRT_RUNTIME_STATUS_H
#define MTRT_RUNTIME_STATUS_H

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace mtrt {

/// Error codes returned by StandaloneCPP runtime functions.
enum class ErrorCode : int32_t {
  Success = 0,
  InvalidArgument = 1,
  NotFound = 2,
  IOError = 3,
  CUDADriverError = 4,
  CUDARuntimeError = 5,
  TensorRTError = 6,
  InternalError = 7,
  Unimplemented = 8,
};

/// Status is an int32_t compatible with ErrorCode. 0 means success.
using Status = int32_t;

inline constexpr Status ok() { return 0; }

/// Returns a pointer to a thread-local, null-terminated message for the most
/// recent error reported by this runtime thread.
const char *get_last_error_message();

/// Clear the current thread-local error message.
void clear_last_error_message();

namespace detail {

/// Sets the thread-local error message using a printf-style format string.
void set_last_error_message_v(const char *fmt, va_list ap);

inline void set_last_error_message(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  set_last_error_message_v(fmt, ap);
  va_end(ap);
}

} // namespace detail

inline Status make_status(ErrorCode code) { return static_cast<Status>(code); }

/// Convenience for call sites that cannot reasonably propagate Status via a
/// return value (e.g. functions returning non-status payloads).
/// Aborts the process after printing the last error message to stderr.
inline void abort_on_error(Status st) {
  if (st == ok())
    return;
  const char *msg = get_last_error_message();
  if (msg && msg[0] != '\0')
    std::fprintf(stderr, "%s\n", msg);
  std::abort();
}

} // namespace mtrt

//===----------------------------------------------------------------------===//
// Convenience macros for runtime code.
//===----------------------------------------------------------------------===//

#define MTRT_RETURN_ERROR(codeEnum, fmt, ...)                                  \
  do {                                                                         \
    ::mtrt::detail::set_last_error_message("%s:%d:%s(): " fmt, __FILE__,       \
                                           __LINE__, __func__, ##__VA_ARGS__); \
    return ::mtrt::make_status(codeEnum);                                      \
  } while (false)

#define MTRT_RETURN_IF_FALSE(cond, codeEnum, fmt, ...)                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      MTRT_RETURN_ERROR(codeEnum, fmt, ##__VA_ARGS__);                         \
    }                                                                          \
  } while (false)

#endif // MTRT_RUNTIME_STATUS_H
