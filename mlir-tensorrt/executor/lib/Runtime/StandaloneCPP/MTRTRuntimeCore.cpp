//===- MTRTRuntimeCore.cpp ------------------------------------------------===//
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
/// This file contains an example implementation of C++ functions required
/// to interact with generated C++ host code.
///
//===----------------------------------------------------------------------===//
#include "MTRTRuntimeCore.h"
#include "MTRTRuntimeStatus.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace mtrt;

static std::vector<std::string>
getSearchCandidates(const std::string &filename);

#define MTRT_DBGF(fmt, ...)                                                    \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      std::fprintf(stderr, "%s:%d:%s(): " fmt "\n", "MTRTRuntimeCore.cpp",     \
                   __LINE__, __func__, __VA_ARGS__);                           \
  } while (0)

static const char *kDebugEnvironmentVariable = "MTRT_DEBUG";
static const char *kArtifactsDirEnvironmentVariable = "MTRT_ARTIFACTS_DIR";

/// Helper method that checks environment value for debugging.
[[maybe_unused]] static bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized) {
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
    isInitialized = true;
  }
  return isEnabled;
}

namespace {

[[maybe_unused]] constexpr int32_t kPlanMemorySpaceUnknown = 0;
[[maybe_unused]] constexpr int32_t kPlanMemorySpaceHost = 1;
[[maybe_unused]] constexpr int32_t kPlanMemorySpaceHostPinned = 2;
[[maybe_unused]] constexpr int32_t kPlanMemorySpaceDevice = 3;
[[maybe_unused]] constexpr int32_t kPlanMemorySpaceUnified = 4;

static inline bool isPowerOfTwo(uint32_t x) {
  return x && ((x & (x - 1)) == 0);
}

static inline size_t roundUpTo(size_t value, size_t multiple) {
  if (multiple == 0)
    return value;
  size_t rem = value % multiple;
  if (rem == 0)
    return value;
  return value + (multiple - rem);
}

static Status reportFileOpenError(const std::string &filename) {
  std::string msg;
  msg += "Error opening file '";
  msg += filename;
  msg += "'.\nTried:\n";
  for (const std::string &candidate : getSearchCandidates(filename)) {
    msg += "  - ";
    msg += candidate;
    msg += "\n";
  }
  msg += "Hint: set ";
  msg += kArtifactsDirEnvironmentVariable;
  msg += " to the directory containing manifest.json\n";
  mtrt::detail::set_last_error_message("%s", msg.c_str());
  return mtrt::make_status(mtrt::ErrorCode::NotFound);
}

/// Open the first readable candidate for `filename` and query its size.
/// This provides an explicit "exists + sane size" check before any reads.
static Status openExistingFileForRead(const std::string &filename,
                                      std::ifstream &file, size_t &outSize) {
  outSize = 0;
  for (const std::string &candidate : getSearchCandidates(filename)) {
    file.open(candidate, std::ios::binary | std::ios::ate);
    if (!file) {
      file.clear();
      continue;
    }

    std::streampos pos = file.tellg();
    if (pos < 0) {
      file.close();
      MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError,
                        "Error determining file size for '%s'",
                        candidate.c_str());
    }

    size_t size = static_cast<size_t>(pos);
    if (size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      file.close();
      MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError, "File '%s' is too large",
                        candidate.c_str());
    }

    // Reset to the start so callers can read immediately.
    file.seekg(0, std::ios::beg);
    if (!file) {
      file.close();
      MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError,
                        "Error seeking to start of file '%s'",
                        candidate.c_str());
    }

    outSize = size;
    return mtrt::ok();
  }
  return reportFileOpenError(filename);
}

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
namespace mtrt::detail {
Status readInputFile(const std::string &filename, std::vector<char> &buffer);
Status readInputFile(const std::string &filename, char *buffer,
                     size_t expectedSize);
Status getFileSize(const std::string &filename, size_t *outSize);
} // namespace mtrt::detail

static bool isAbsolutePath(const std::string &path) {
  return !path.empty() && path.front() == '/';
}

static std::string joinPath(const std::string &dir, const std::string &rel) {
  if (dir.empty())
    return rel;
  if (rel.empty())
    return dir;
  if (dir.back() == '/')
    return dir + rel;
  return dir + "/" + rel;
}

static std::vector<std::string>
getSearchCandidates(const std::string &filename) {
  std::vector<std::string> candidates;
  candidates.reserve(3);

  if (filename.empty())
    return candidates;

  if (isAbsolutePath(filename)) {
    candidates.push_back(filename);
    return candidates;
  }

  if (const char *env = std::getenv(kArtifactsDirEnvironmentVariable)) {
    std::string base(env);
    if (!base.empty())
      candidates.push_back(joinPath(base, filename));
  }

  // Legacy behavior: treat filename as relative to CWD.
  candidates.push_back(filename);
  return candidates;
}

Status mtrt::detail::getFileSize(const std::string &filename, size_t *outSize) {
  if (!outSize)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outSize must not be null");
  std::ifstream file;
  size_t size = 0;
  Status st = openExistingFileForRead(filename, file, size);
  if (st != mtrt::ok())
    return st;
  *outSize = size;
  return mtrt::ok();
}

Status mtrt::detail::readInputFile(const std::string &filename,
                                   std::vector<char> &buffer) {
  std::ifstream file;
  size_t size = 0;
  Status st = openExistingFileForRead(filename, file, size);
  if (st != mtrt::ok())
    return st;

  // Create a vector to hold the file contents.
  buffer.resize(size);
  if (size == 0)
    return mtrt::ok();

  // Read the entire file into the vector.
  if (!file.read(buffer.data(), static_cast<std::streamsize>(size)))
    MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError, "Error reading file '%s'",
                      filename.c_str());
  return mtrt::ok();
}

Status mtrt::detail::readInputFile(const std::string &filename, char *buffer,
                                   size_t expectedSize) {
  if (!buffer && expectedSize != 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "buffer must not be null when expectedSize != 0");
  std::ifstream file;
  size_t size = 0;
  Status st = openExistingFileForRead(filename, file, size);
  if (st != mtrt::ok())
    return st;

  if (size != expectedSize) {
    MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError,
                      "file size mismatch for '%s': %zu != %zu",
                      filename.c_str(), size, expectedSize);
  }

  // Read the entire file into the vector
  if (size == 0)
    return mtrt::ok();
  if (!file.read(buffer, static_cast<std::streamsize>(size)))
    MTRT_RETURN_ERROR(mtrt::ErrorCode::IOError, "Error reading file '%s'",
                      filename.c_str());
  return mtrt::ok();
}

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

Status mtrt::host_aligned_alloc(int64_t sizeBytes, int32_t alignment,
                                void **outPtr) {
  if (!outPtr)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outPtr must not be null");
  *outPtr = nullptr;
  if (sizeBytes < 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "sizeBytes must be >= 0");
  if (alignment <= 0 || !isPowerOfTwo(static_cast<uint32_t>(alignment)))
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "alignment must be a power-of-two > 0 (got %d)",
                      alignment);
  if (sizeBytes == 0)
    return mtrt::ok();

  size_t roundedSize =
      roundUpTo(static_cast<size_t>(sizeBytes), static_cast<size_t>(alignment));
  void *ptr = std::aligned_alloc(static_cast<size_t>(alignment), roundedSize);
  if (!ptr)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InternalError,
                      "aligned_alloc failed (size=%zu align=%d)", roundedSize,
                      alignment);
  *outPtr = ptr;
  return mtrt::ok();
}

void mtrt::host_free(void *ptr) { ::free(ptr); }

namespace mtrt::detail {
// These hooks are provided by MTRTRuntimeCuda.cpp when linked.
Status cuda_alloc_and_copy_constant(int32_t space, const void *src,
                                    size_t bytes, void **outPtr)
    __attribute__((weak));
Status cuda_free_constant(int32_t space, void *ptr) __attribute__((weak));
} // namespace mtrt::detail

Status mtrt::constant_load_from_file(const char *filename, int32_t align,
                                     int32_t space, void **outPtr) {
  if (!outPtr)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outPtr must not be null");
  *outPtr = nullptr;
  if (!filename || filename[0] == '\0')
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "filename must not be empty");

  size_t fileSize = 0;
  Status st = detail::getFileSize(filename, &fileSize);
  if (st != mtrt::ok())
    return st;

  // Host-visible paths: read directly into the destination buffer.
  if (space == kPlanMemorySpaceUnknown || space == kPlanMemorySpaceHost) {
    void *buf = nullptr;
    st = host_aligned_alloc(static_cast<int64_t>(fileSize),
                            std::max<int32_t>(align, 1), &buf);
    if (st != mtrt::ok())
      return st;
    st = detail::readInputFile(filename, reinterpret_cast<char *>(buf),
                               fileSize);
    if (st != mtrt::ok()) {
      host_free(buf);
      return st;
    }
    *outPtr = buf;
    return mtrt::ok();
  }

  // Non-host spaces: require CUDA runtime support to allocate/copy.
  if (!detail::cuda_alloc_and_copy_constant) {
    MTRT_RETURN_ERROR(
        mtrt::ErrorCode::Unimplemented,
        "constant_load_from_file requested memory space %d but CUDA runtime "
        "support is not linked; include/compile MTRTRuntimeCuda.cpp",
        space);
  }

  std::vector<char> tmp;
  st = detail::readInputFile(filename, tmp);
  if (st != mtrt::ok())
    return st;

  return detail::cuda_alloc_and_copy_constant(space, tmp.data(), tmp.size(),
                                              outPtr);
}

void mtrt::constant_destroy(void *data, int32_t space) {
  if (!data)
    return;
  if (space == kPlanMemorySpaceUnknown || space == kPlanMemorySpaceHost) {
    host_free(data);
    return;
  }
  if (detail::cuda_free_constant) {
    (void)detail::cuda_free_constant(space, data);
    return;
  }
  // Best-effort fallback: avoid crashing if the caller uses a non-host space
  // without linking CUDA support.
  mtrt::detail::set_last_error_message(
      "constant_destroy: cannot free pointer for memory space %d without CUDA "
      "runtime support; include/compile MTRTRuntimeCuda.cpp",
      space);
}
