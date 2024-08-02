//===- TensorRTVersion.h ----------------------------------------*- C++ -*-===//
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
/// Utilities for handling TensorRT versions.
///
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_TENSORRTVERSION
#define INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_TENSORRTVERSION
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "NvInferVersion.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstdint>
#include <sstream>

extern "C" int32_t getInferLibVersion() noexcept;
extern "C" int32_t getInferLibBuildVersion() noexcept;

/// Evaluates to true if the version of TensorRT that we are compiling against
/// is greater than or equal to the given version number.
#define MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(major, minor, patch)        \
  (NV_TENSORRT_MAJOR > major ||                                                \
   (NV_TENSORRT_MAJOR == major &&                                              \
    (NV_TENSORRT_MINOR > minor ||                                              \
     (NV_TENSORRT_MINOR == minor && NV_TENSORRT_PATCH >= patch))))

/// Evaluates to true if the version of TensorRT that we are compiling against
/// is less than the given version number.
#define MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(major, minor, patch)         \
  (NV_TENSORRT_MAJOR < major ||                                                \
   (NV_TENSORRT_MAJOR == major &&                                              \
    (NV_TENSORRT_MINOR < minor ||                                              \
     (NV_TENSORRT_MINOR == minor && NV_TENSORRT_PATCH < patch))))

namespace mlir::tensorrt {

/// Encapsulates the version information for the TensorRT library loaded at
/// runtime.
struct TensorRTVersion {
  int64_t major, minor, patch, build;

  // `major` and `minor` are macros exported by glibc as of 2.24
  // (https://bugzilla.redhat.com/show_bug.cgi?id=130601). Using brace
  // initialization to work around that.
  constexpr TensorRTVersion(int64_t major, int64_t minor, int64_t patch = 0,
                            int64_t build = 0)
      : major{major}, minor{minor}, patch{patch}, build{build} {}

  TensorRTVersion() { *this = TensorRTVersion::getCompileTimeVersion(); }

  static TensorRTVersion getCompileTimeVersion() {
    return TensorRTVersion(NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                           NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);
  }

  static TensorRTVersion getLoadedVersion() {
    // The format is as for TENSORRT_VERSION: (TENSORRT_MAJOR *
    // MAJOR_VERSION_DIVISOR) + (TENSORRT_MINOR * 100) + TENSOR_PATCH.
    int32_t version = getInferLibVersion();
    // Build number is known only from TRT >= 10.x
    int32_t build = -1;
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    build = getInferLibBuildVersion();
#endif
    // TensorRT 10.0 uses `TENSORRT_MAJOR * 10000` instead of `1000`.
    const int32_t MAJOR_VERSION_DIVISOR = version >= 10000 ? 10000 : 1000;
    return TensorRTVersion(version / MAJOR_VERSION_DIVISOR,
                           (version % MAJOR_VERSION_DIVISOR) / 100,
                           version % 100, build);
  }

  std::string getAsString() const {
    std::ostringstream ss;
    ss << major << "." << minor << "." << patch << ".";
    if (build != -1)
      ss << build;
    return ss.str();
  }

  constexpr bool isGreaterThanOrEqualTRT10() {
    return *this >= TensorRTVersion(10, 0, 0, 0);
  }

  constexpr bool operator==(const TensorRTVersion &other) const {
    return other.major == major && other.minor == minor &&
           other.patch == patch && other.build == build;
  }

  constexpr bool operator!=(const TensorRTVersion &other) const {
    return !(*this == other);
  }

  constexpr bool operator>=(const TensorRTVersion &rhs) const {
    return major > rhs.major ||
           (major == rhs.major &&
            (minor > rhs.minor ||
             (minor == rhs.minor &&
              (patch > rhs.patch ||
               (patch == rhs.patch && build >= rhs.build)))));
  }
};
} // namespace mlir::tensorrt

#endif // INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_TENSORRTVERSION
