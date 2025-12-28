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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mtrt;

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
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
namespace mtrt::detail {
int readInputFile(const std::string &filename, std::vector<char> &buffer);
int readInputFile(const std::string &filename, char *buffer,
                  size_t expectedSize);
size_t getFileSize(const std::string &filename);
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

size_t mtrt::detail::getFileSize(const std::string &filename) {
  for (const std::string &candidate : getSearchCandidates(filename)) {
    std::ifstream file(candidate, std::ios::binary | std::ios::ate);
    if (file)
      return file.tellg();
  }
  std::cerr << "Error opening file '" << filename << "'.\n";
  std::cerr << "Tried:\n";
  for (const std::string &candidate : getSearchCandidates(filename))
    std::cerr << "  - " << candidate << "\n";
  std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
            << " to the directory containing manifest.json\n";
  return 0;
}

int mtrt::detail::readInputFile(const std::string &filename,
                                std::vector<char> &buffer) {
  std::ifstream file;
  std::string openedPath;
  for (const std::string &candidate : getSearchCandidates(filename)) {
    file.open(candidate, std::ios::binary | std::ios::ate);
    if (file) {
      openedPath = candidate;
      break;
    }
    file.clear();
  }
  if (!file) {
    std::cerr << "Error opening file '" << filename << "'.\n";
    std::cerr << "Tried:\n";
    for (const std::string &candidate : getSearchCandidates(filename))
      std::cerr << "  - " << candidate << "\n";
    std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
              << " to the directory containing manifest.json\n";
    return 1;
  }

  // Get the size of the file
  std::streamsize size = file.tellg();

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Create a vector to hold the file contents
  buffer.resize(size);

  // Read the entire file into the vector
  if (file.read(buffer.data(), size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

int mtrt::detail::readInputFile(const std::string &filename, char *buffer,
                                size_t expectedSize) {
  std::ifstream file;
  std::string openedPath;
  for (const std::string &candidate : getSearchCandidates(filename)) {
    file.open(candidate, std::ios::binary | std::ios::ate);
    if (file) {
      openedPath = candidate;
      break;
    }
    file.clear();
  }
  if (!file) {
    std::cerr << "Error opening file '" << filename << "'.\n";
    std::cerr << "Tried:\n";
    for (const std::string &candidate : getSearchCandidates(filename))
      std::cerr << "  - " << candidate << "\n";
    std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
              << " to the directory containing manifest.json\n";
    return 1;
  }

  // Get the size of the file
  size_t size = file.tellg();

  if (size != expectedSize) {
    std::cerr << "Error: file size mismatch: " << size << " != " << expectedSize
              << std::endl;
    return 1;
  }

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Read the entire file into the vector
  if (file.read(buffer, size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

void *mtrt::host_alloc(int64_t size, int32_t alignment) {
  if (size % alignment != 0)
    size = ((size + alignment - 1) / alignment) * alignment;
  return std::aligned_alloc(size, alignment);
}

void mtrt::host_free(void *ptr) { ::free(ptr); }
