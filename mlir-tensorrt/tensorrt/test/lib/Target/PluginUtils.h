//===- PluginUtils.h -----------------------------------------*- C++ -*-===//
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
/// Declares utilities for plugin tests.
///
//===----------------------------------------------------------------------===//
#ifndef LIB_TARGET_PLUGINUTILS
#define LIB_TARGET_PLUGINUTILS

#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"
#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include "tensorrttestplugins_export.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace nvinfer1;

// We use a special field to indicate to the plugin creator that it should fail
// and return a nullptr in createPlugin.
constexpr const char *kPLUGIN_FAILURE_TRIGGER_FIELD_NAME = "trigger_failure";

inline static bool
creatorShouldFail(const nvinfer1::PluginFieldCollection *fc) {
  return std::any_of(
      fc->fields, fc->fields + fc->nbFields, [](const PluginField &pf) {
        return std::string(pf.name) == kPLUGIN_FAILURE_TRIGGER_FIELD_NAME &&
               *static_cast<const int32_t *>(pf.data) == 1;
      });
}

inline static std::optional<unsigned> getWidth(const PluginField &field) {
#define HANDLE_CASE(x, w)                                                      \
  case PluginFieldType::x:                                                     \
    return w;
  switch (field.type) {
    HANDLE_CASE(kFLOAT16, 16)
    HANDLE_CASE(kFLOAT32, 32)
    HANDLE_CASE(kFLOAT64, 64)
    HANDLE_CASE(kINT8, 8)
    HANDLE_CASE(kINT16, 16)
    HANDLE_CASE(kINT32, 32)
    HANDLE_CASE(kCHAR, 8)
    HANDLE_CASE(kDIMS, {})
    HANDLE_CASE(kUNKNOWN, {})
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    HANDLE_CASE(kBF16, 16)
    HANDLE_CASE(kINT64, 64)
    HANDLE_CASE(kFP8, 8)
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 2, 0)
    HANDLE_CASE(kINT4, 4);
#endif
  }
#undef HANDLE_CASE
  llvm_unreachable("unhandled PluginFieldType enumeration value");
}

/// Read a scalar field and extend to 64 bits.
inline static std::optional<uint64_t> readScalarField(const PluginField &field,
                                                      unsigned idx = 0) {
#define HANDLE_CASE(x, w)                                                      \
  case PluginFieldType::x:                                                     \
    return *(reinterpret_cast<const w *>(field.data) + idx);
  switch (field.type) {
    HANDLE_CASE(kFLOAT16, uint16_t)
    HANDLE_CASE(kFLOAT32, uint32_t)
    HANDLE_CASE(kFLOAT64, uint64_t)
    HANDLE_CASE(kINT8, uint8_t)
    HANDLE_CASE(kINT16, uint16_t)
    HANDLE_CASE(kINT32, uint32_t)
    HANDLE_CASE(kCHAR, uint8_t)
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    HANDLE_CASE(kBF16, uint16_t)
    HANDLE_CASE(kINT64, uint64_t)
    HANDLE_CASE(kFP8, uint8_t)
#endif
  default:
    return {};
  }
#undef HANDLE_CASE
  llvm_unreachable("unhandled PluginFieldType enumeration value");
}

// Print either a float or integer scalar. Uses LLVM APInt/APFloat to perform
// correct formatting basd on type.
template <typename T>
static void printScalar(std::ostream &os, unsigned width, uint64_t value,
                        PluginFieldType type) {
  llvm::raw_os_ostream adaptor(os);
  llvm::APInt apInt(width, value);
  if constexpr (std::is_same_v<T, llvm::APInt>) {
    apInt.print(adaptor, false);
    return;
  }
  if constexpr (std::is_same_v<T, llvm::APFloat>) {
    llvm::APFloat apFloat(llvm::APFloat::IEEEsingle());
    switch (type) {
    case PluginFieldType::kFLOAT16:
      apFloat = llvm::APFloat(llvm::APFloat::IEEEhalf(), apInt.trunc(16));
      break;
    case PluginFieldType::kFLOAT32:
      apFloat = llvm::APFloat(llvm::APFloat::IEEEsingle(), apInt.trunc(32));
      break;
    case PluginFieldType::kFLOAT64:
      apFloat = llvm::APFloat(llvm::APFloat::IEEEdouble(), apInt.trunc(64));
      break;
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    case PluginFieldType::kBF16:
      apFloat = llvm::APFloat(llvm::APFloat::BFloat(), apInt.trunc(16));
      break;
    case PluginFieldType::kFP8:
      apFloat = llvm::APFloat(llvm::APFloat::Float8E5M2FNUZ(), apInt.trunc(8));
      break;
#endif
    default:
      llvm_unreachable("unrecognized PluginField floating-point type");
    }

    llvm::SmallString<128> strVal;
    apFloat.toString(strVal);
    adaptor << strVal;
  }
  return;
}

template <typename T>
void printScalar(std::ostream &os, const PluginField &field, unsigned idx = 0) {
  std::optional<unsigned> width = getWidth(field);
  if (!width)
    return;
  std::optional<uint64_t> value = readScalarField(field, idx);
  if (!value)
    return;
  llvm::raw_os_ostream adaptor(os);
  llvm::APInt apInt(*width, *value);
  return printScalar<T>(os, *width, *value, field.type);
}

/// Print a Dims object.
inline static void printDims(std::ostream &os, const Dims &dims) {
  llvm::raw_os_ostream adaptor(os);
  adaptor << "Dims<";
  llvm::interleave(llvm::make_range(dims.d, dims.d + dims.nbDims), adaptor,
                   "x");
  adaptor << ">";
}

/// Prints the plugin field name, type, and value to the stream.
inline static void printField(std::ostream &os, const PluginField &field) {
  os << "name=" << field.name << ", ";
  os << "type=";
#define HANDLE_CASE(x)                                                         \
  case PluginFieldType::x:                                                     \
    os << #x;                                                                  \
    break;
  switch (field.type) {
    HANDLE_CASE(kFLOAT16)
    HANDLE_CASE(kFLOAT32)
    HANDLE_CASE(kFLOAT64)
    HANDLE_CASE(kINT8)
    HANDLE_CASE(kINT16)
    HANDLE_CASE(kINT32)
    HANDLE_CASE(kCHAR)
    HANDLE_CASE(kDIMS)
    HANDLE_CASE(kUNKNOWN)
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    HANDLE_CASE(kBF16)
    HANDLE_CASE(kINT64)
    HANDLE_CASE(kFP8)
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 2, 0)
    HANDLE_CASE(kINT4)
#endif
  }
  os << ", length=" << field.length << ", ";

#undef HANDLE_CASE

  os << "data=[";
  switch (field.type) {
  case PluginFieldType::kFLOAT16:
  case PluginFieldType::kFLOAT32:
  case PluginFieldType::kFLOAT64:
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  case PluginFieldType::kBF16:
  case PluginFieldType::kFP8:
#endif
  {
    for (int32_t i = 0; i < field.length; i++) {
      printScalar<llvm::APFloat>(os, field, i);
      if (i < field.length - 1)
        os << ", ";
    }
    break;
  }
  case PluginFieldType::kINT8:
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  case PluginFieldType::kINT64:
#endif
  case PluginFieldType::kINT32:
  case PluginFieldType::kINT16: {
    for (int32_t i = 0; i < field.length; i++) {
      printScalar<llvm::APInt>(os, field, i);
      if (i < field.length - 1)
        os << ", ";
    }
    break;
  }
  case nvinfer1::PluginFieldType::kCHAR:
    os << std::string_view(reinterpret_cast<const char *>(field.data),
                           field.length);
    break;
  case PluginFieldType::kDIMS: {
    const Dims *dimsPtr = reinterpret_cast<const Dims *>(field.data);
    for (const nvinfer1::Dims &dims :
         llvm::make_range(dimsPtr, dimsPtr + field.length)) {
      os << " ";
      printDims(os, dims);
      os << " ";
    }
    break;
  }
  default:
    os << "unk";
    break;
  }
  os << "]\n";
}

#endif // LIB_TARGET_PLUGINUTILS
