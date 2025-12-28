//===- MTRTRuntimeStatus.cpp ------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MTRTRuntimeStatus.h"

#include <cstddef>
#include <cstring>

namespace mtrt {
namespace {

// Thread-local last error buffer. Keep it reasonably small and stack-free.
thread_local char lastError[4096];

} // namespace

const char *get_last_error_message() { return lastError; }

void clear_last_error_message() { lastError[0] = '\0'; }

namespace detail {

void set_last_error_message_v(const char *fmt, va_list ap) {
  if (!fmt) {
    lastError[0] = '\0';
    return;
  }
  // Ensure we always null-terminate.
  int n = std::vsnprintf(lastError, sizeof(lastError), fmt, ap);
  (void)n;
  lastError[sizeof(lastError) - 1] = '\0';
}

} // namespace detail
} // namespace mtrt
