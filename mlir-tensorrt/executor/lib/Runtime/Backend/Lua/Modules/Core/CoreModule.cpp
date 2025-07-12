//===- CoreModule.cpp -----------------------------------------------------===//
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
/// Executor Core module Lua runtime implementation.
///
//===----------------------------------------------------------------------===//
#include "../../../C/CoreModule.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Common/Int4.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Support/Allocators.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <climits>
#include <type_traits>

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

typedef __nv_fp8_e4m3 fp8_e4m3fn;

//===----------------------------------------------------------------------===//
// Helpers for f16 type (half) other math ops
//===----------------------------------------------------------------------===//

static std::ostream &operator<<(std::ostream &os, const half &scalar) {
  return os << __half2float(scalar);
}

static std::ostream &operator<<(std::ostream &os, const fp8_e4m3fn &scalar) {
  return os << float(scalar);
}

static std::ostream &operator<<(std::ostream &os, const nv_bfloat16 &scalar) {
  return os << __bfloat162float(scalar);
}

//===----------------------------------------------------------------------===//
// Templated helpers
//===----------------------------------------------------------------------===//

/// Wrapper for `std::isnan` that correctly handles float16 as well as standard
/// types.
template <typename T>
bool isNan(T val) {
  if constexpr (std::is_same_v<T, half>)
    return std::isnan(__half2float(val));
  else if constexpr (std::is_same_v<T, fp8_e4m3fn>)
    return std::isnan(float(val));
  else if constexpr (std::is_same_v<T, nv_bfloat16>)
    return std::isnan(__bfloat162float(val));
  else
    return std::isnan(val);
}

/// Negation operator for in unary op definitions below.
template <typename T>
T negate(T x) {
  return static_cast<T>(-1.) * x;
}

/// Truncation methods.
template <typename InputType, typename ResultType, unsigned ResultBits>
ResultType integerTruncate(InputType input) {
  constexpr unsigned InputBits = sizeof(InputType) * CHAR_BIT;
  static_assert(ResultBits < InputBits,
                "result bitwidth must be smaller than input bitwidth");
  static_assert(ResultBits <= sizeof(ResultType) * CHAR_BIT,
                "result bitwidth must be smaller than input bitwidth");
  if constexpr (sizeof(ResultType) * CHAR_BIT == ResultBits) {
    return static_cast<ResultType>(input);
  }
  return static_cast<ResultType>(input & ((1U << ResultBits) - 1));
}

template <typename IntType>
static IntType alignToImpl(IntType arg, uint32_t alignment) {
  typename std::make_unsigned<IntType>::type bump =
      static_cast<typename std::make_unsigned<IntType>::type>(arg) + alignment -
      1;
  return static_cast<IntType>(bump - bump % alignment);
}

template <typename ElementType>
static bool checkAccessBounds(lua_State *state, const AllocTracker &tracker,
                              uintptr_t basePtr, uint64_t offset) {
#ifndef NDEBUG
  if (tracker.contains(basePtr)) {
    const PointerInfo &srcInfo = tracker.get(basePtr);
    if (!srcInfo.isHostVisible()) {
      MTRT_ERRF(
          "attempting to access memory  0x%lx + %lu which is not host-visible",
          basePtr, offset);
      return false;
    }

    if (offset + sizeof(ElementType) > srcInfo.size) {
      auto errMsg = llvm::formatv(
          "attempting to access memory in range [{0} + {1}, {0} + {2}), "
          "which will access memory out of bounds (size of allocation is {3})",
          (void *)basePtr, offset, offset + sizeof(ElementType), srcInfo.size);
      luaL_error(state, errMsg.str().c_str());
      return false;
    }
  }
#endif
  return true;
}

/// Implementation of 'executor.remf', which is equivalent to `std::fmod`. For
/// small floating point types we upcast to f32 then truncate.
template <typename T>
T remf(T lhs, T rhs) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fmod<T>(lhs, rhs);
  } else if constexpr (std::is_same_v<T, __half>) {
    return std::fmod(__half2float(lhs), __half2float(rhs));
  } else if constexpr (std::is_same_v<T, fp8_e4m3fn>) {
    return fp8_e4m3fn(std::fmod(float(lhs), float(rhs)));
  } else {
    static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported llvm_remf type");
    return nv_bfloat16(std::fmod(__bfloat162float(lhs), __bfloat162float(rhs)));
  }
}

template <typename InpType, typename ResType>
ResType sitofp(InpType input) {
  return static_cast<ResType>(input);
}

template <typename InpType, typename ResType>
ResType uitofp(InpType input) {
  return static_cast<ResType>(
      *reinterpret_cast<const std::make_unsigned_t<InpType> *>(&input));
}

//===----------------------------------------------------------------------===//
// Executor - Core operations
//===----------------------------------------------------------------------===//
static void registerExecutorCoreModuleLuaRuntimeMethods(
    lua_State *luaState, PinnedMemoryAllocator *pinnedMemoryAllocator,
    AllocTracker *allocTracker) {
  sol::state_view lua(luaState);

  lua["__check_for_function"] = [](sol::this_state state,
                                   const std::string &name) {
    sol::state_view lua(state);
    sol::protected_function func = lua[name];
    if (!func.valid()) {
      std::string err = llvm::formatv("expected runtime to provide function "
                                      "\"{0}\", but it was not found",
                                      name);
      luaL_error(state, err.c_str());
      return;
    }
  };

  //===----------------------------------------------------------------------===//
  // Type-erased select operator
  //===----------------------------------------------------------------------===//

  lua["_select"] = [](int8_t condition, sol::object trueValue,
                      sol::object falseValue) {
    return condition != 0 ? trueValue : falseValue;
  };

  //===----------------------------------------------------------------------===//
  // f16 "half" type registration and ancillary operations.
  //===----------------------------------------------------------------------===//
  lua.new_usertype<__half>(
      "half", sol::constructors<__half(), __half(float), __half(__half)>(),
      sol::meta_function::addition,
      [](const half &lhs, const half &rhs) -> half { return lhs + rhs; },
      sol::meta_function::multiplication,
      [](const half &lhs, const half &rhs) -> half { return lhs * rhs; },
      sol::meta_function::subtraction,
      [](const half &lhs, const half &rhs) -> half { return lhs - rhs; },
      sol::meta_function::division,
      [](const half &lhs, const half &rhs) -> half { return lhs / rhs; });

  lua["executor_constant_f16"] = [](float val) { return __half(val); };

  //===----------------------------------------------------------------------===//
  // fp8_e4m3fn "f8E4M3FN" type registration and ancillary operations.
  //===----------------------------------------------------------------------===//
  lua.new_usertype<fp8_e4m3fn>(
      "fp8_e4m3fn", sol::constructors<fp8_e4m3fn(__half), fp8_e4m3fn(float)>(),
      sol::meta_function::addition,
      [](const fp8_e4m3fn &lhs, const fp8_e4m3fn &rhs) -> fp8_e4m3fn {
        return fp8_e4m3fn(float(lhs) + float(rhs));
      },
      sol::meta_function::multiplication,
      [](const fp8_e4m3fn &lhs, const fp8_e4m3fn &rhs) -> fp8_e4m3fn {
        return fp8_e4m3fn(float(lhs) * float(rhs));
      },
      sol::meta_function::subtraction,
      [](const fp8_e4m3fn &lhs, const fp8_e4m3fn &rhs) -> fp8_e4m3fn {
        return fp8_e4m3fn(float(lhs) - float(rhs));
      },
      sol::meta_function::division,
      [](const fp8_e4m3fn &lhs, const fp8_e4m3fn &rhs) -> fp8_e4m3fn {
        return fp8_e4m3fn(float(lhs) / float(rhs));
      });

  lua["executor_constant_f8E4M3FN"] = [](float val) { return fp8_e4m3fn(val); };

  //===----------------------------------------------------------------------===//
  // bf16 "nv_bfloat16" type registration and ancillary operations.
  //===----------------------------------------------------------------------===//
  lua.new_usertype<nv_bfloat16>(
      "bf16", sol::constructors<nv_bfloat16(double), nv_bfloat16(float)>(),
      sol::meta_function::addition,
      [](const nv_bfloat16 &lhs, const nv_bfloat16 &rhs) -> nv_bfloat16 {
        return lhs + rhs;
      },
      sol::meta_function::multiplication,
      [](const nv_bfloat16 &lhs, const nv_bfloat16 &rhs) -> nv_bfloat16 {
        return lhs * rhs;
      },
      sol::meta_function::subtraction,
      [](const nv_bfloat16 &lhs, const nv_bfloat16 &rhs) -> nv_bfloat16 {
        return lhs - rhs;
      },
      sol::meta_function::division,
      [](const nv_bfloat16 &lhs, const nv_bfloat16 &rhs) -> nv_bfloat16 {
        return lhs / rhs;
      });

  lua["executor_constant_bf16"] = [](float val) { return nv_bfloat16(val); };

  //===----------------------------------------------------------------------===//
  // i4 "nv_int4" type registration and ancillary operations.
  //===----------------------------------------------------------------------===//

  lua.new_usertype<nv_int4>(
      "i4", sol::constructors<nv_int4(float), nv_int4(double)>(),
      sol::meta_function::addition,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs + rhs;
      },
      sol::meta_function::subtraction,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs - rhs;
      },
      sol::meta_function::multiplication,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs * rhs;
      },
      sol::meta_function::division,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs / rhs;
      },
      sol::meta_function::modulus,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs % rhs;
      },
      sol::meta_function::floor_division,
      [](const nv_int4 &lhs, const nv_int4 &rhs) -> nv_int4 {
        return lhs / rhs;
      });

  lua["executor_constant_i4"] = [](float val) { return nv_int4(val); };

  //===----------------------------------------------------------------------===//
  // arithmetic extension ops
  //===----------------------------------------------------------------------===//

#define DEFINE_UICMP_METHOD(pred, suffix, type, cast_type, opSymbol)           \
  lua["_icmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_icmp_" #type);                                 \
    return (*reinterpret_cast<cast_type *>(&lhs))opSymbol(                     \
        *reinterpret_cast<cast_type *>(&rhs));                                 \
  }
#define DEFINE_SICMP_METHOD(pred, suffix, type, opSymbol)                      \
  lua["_icmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_icmp_" #type);                                 \
    return lhs opSymbol rhs;                                                   \
  }
#define DEFINE_SICMP_METHODS(pred, opSymbol)                                   \
  DEFINE_SICMP_METHOD(pred, i64, int64_t, opSymbol);                           \
  DEFINE_SICMP_METHOD(pred, i32, int32_t, opSymbol);                           \
  DEFINE_SICMP_METHOD(pred, i16, int16_t, opSymbol);                           \
  DEFINE_SICMP_METHOD(pred, i8, int8_t, opSymbol);                             \
  DEFINE_SICMP_METHOD(pred, i4, nv_int4, opSymbol)
#define DEFINE_ICMP_METHODS(pred, opSymbol)                                    \
  DEFINE_SICMP_METHODS(s##pred, opSymbol);                                     \
  DEFINE_UICMP_METHOD(u##pred, i64, int64_t, uint64_t, opSymbol);              \
  DEFINE_UICMP_METHOD(u##pred, i32, int32_t, uint32_t, opSymbol);              \
  DEFINE_UICMP_METHOD(u##pred, i16, int16_t, uint16_t, opSymbol);              \
  DEFINE_UICMP_METHOD(u##pred, i8, int8_t, uint8_t, opSymbol);                 \
  DEFINE_UICMP_METHOD(u##pred, i4, nv_int4, nv_uint4, opSymbol)

  DEFINE_SICMP_METHODS(eq, ==);
  DEFINE_SICMP_METHODS(ne, !=);
  DEFINE_ICMP_METHODS(gt, >);
  DEFINE_ICMP_METHODS(lt, <);
  DEFINE_ICMP_METHODS(ge, >=);
  DEFINE_ICMP_METHODS(le, <=);

  DEFINE_SICMP_METHOD(eq, i1, int8_t, ==);
  DEFINE_SICMP_METHOD(ne, i1, int8_t, !=);

#undef DEFINE_SICMP_METHOD
#undef DEFINE_UICMP_METHOD
#undef DEFINE_SICMP_METHODS
#undef DEFINE_ICMP_METHODS

#define DEFINE_SDIVI_METHOD(suffix, type)                                      \
  lua["_sdivi_" #suffix] = [](type lhs, type rhs) -> type {                    \
    ADD_CORE_MODULE_RANGE("core_sdivi_" #suffix);                              \
    return lhs / rhs;                                                          \
  }

  DEFINE_SDIVI_METHOD(i64, int64_t);
  DEFINE_SDIVI_METHOD(i32, int32_t);
  DEFINE_SDIVI_METHOD(i16, int16_t);
  DEFINE_SDIVI_METHOD(i8, int8_t);
  DEFINE_SDIVI_METHOD(i4, nv_int4);
#undef DEFINE_SDIVI_METHOD

#define DEFINE_DIVF_METHOD(suffix, type)                                       \
  lua["_divf_" #suffix] = [](type lhs, type rhs) -> type {                     \
    ADD_CORE_MODULE_RANGE("core_divf_" #suffix);                               \
    return lhs / rhs;                                                          \
  }

  DEFINE_DIVF_METHOD(f32, float);
  DEFINE_DIVF_METHOD(f64, double);
  DEFINE_DIVF_METHOD(f16, __half);
  DEFINE_DIVF_METHOD(bf16, nv_bfloat16);
#undef DEFINE_DIVF_METHOD

  lua["_divf_f8E4M3FN"] = [](fp8_e4m3fn lhs, fp8_e4m3fn rhs) -> fp8_e4m3fn {
    ADD_CORE_MODULE_RANGE("core_divf_f8E4M3FN");
    return fp8_e4m3fn(float(lhs) / float(rhs));
  };

#define DEFINE_BITWISE_METHOD(suffix, type, opSymbol, opName)                  \
  lua["_bitwise_" #opName "_" #suffix] = [](type lhs, type rhs) -> type {      \
    ADD_CORE_MODULE_RANGE("core_bitwise_" #opName "_" #suffix);                \
    return lhs opSymbol rhs;                                                   \
  }

  DEFINE_BITWISE_METHOD(i64, int64_t, |, ori);
  DEFINE_BITWISE_METHOD(i32, int32_t, |, ori);
  DEFINE_BITWISE_METHOD(i16, int16_t, |, ori);
  DEFINE_BITWISE_METHOD(i8, int8_t, |, ori);
  DEFINE_BITWISE_METHOD(i1, int8_t, |, ori);
  DEFINE_BITWISE_METHOD(i4, nv_int4, |, ori);

  DEFINE_BITWISE_METHOD(i64, int64_t, &, andi);
  DEFINE_BITWISE_METHOD(i32, int32_t, &, andi);
  DEFINE_BITWISE_METHOD(i16, int16_t, &, andi);
  DEFINE_BITWISE_METHOD(i8, int8_t, &, andi);
  DEFINE_BITWISE_METHOD(i1, int8_t, &, andi);
  DEFINE_BITWISE_METHOD(i4, nv_int4, &, andi);

  DEFINE_BITWISE_METHOD(i64, int64_t, ^, xori);
  DEFINE_BITWISE_METHOD(i32, int32_t, ^, xori);
  DEFINE_BITWISE_METHOD(i16, int16_t, ^, xori);
  DEFINE_BITWISE_METHOD(i8, int8_t, ^, xori);
  DEFINE_BITWISE_METHOD(i1, int8_t, ^, xori);
  DEFINE_BITWISE_METHOD(i4, nv_int4, ^, xori);

  //===----------------------------------------------------------------------===//
  // executor.bitcast
  //===----------------------------------------------------------------------===//

#undef DEFINE_BITWISE_METHOD

#define DEFINE_BITCAST_METHOD(inpSuffix, resSuffix, inpType, resType)          \
  lua["_bitcast_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {  \
    ADD_CORE_MODULE_RANGE("_bitcast_" #inpSuffix "_" #resSuffix);              \
    return *reinterpret_cast<resType *>(&input);                               \
  }

  DEFINE_BITCAST_METHOD(i64, f64, int64_t, double);
  DEFINE_BITCAST_METHOD(i32, f32, int32_t, float);
  DEFINE_BITCAST_METHOD(i16, f16, int16_t, __half);
  DEFINE_BITCAST_METHOD(f64, i64, double, int64_t);
  DEFINE_BITCAST_METHOD(f32, i32, float, int32_t);
  DEFINE_BITCAST_METHOD(f16, i16, __half, int16_t);
#undef DEFINE_BITCAST_METHOD

  //===----------------------------------------------------------------------===//
  // executor.fmax
  //===----------------------------------------------------------------------===//

#define DEFINE_FMAX_METHOD(suffix, type)                                       \
  lua["_fmax_" #suffix] = [](type lhs, type rhs) -> type {                     \
    ADD_CORE_MODULE_RANGE("core_fmax");                                        \
    return std::max(lhs, rhs);                                                 \
  };                                                                           \
  lua["_fmin_" #suffix] = [](type lhs, type rhs) -> type {                     \
    ADD_CORE_MODULE_RANGE("core_fmin");                                        \
    return std::min(lhs, rhs);                                                 \
  }

  DEFINE_FMAX_METHOD(f64, double);
  DEFINE_FMAX_METHOD(f32, float);
  DEFINE_FMAX_METHOD(f16, __half);
  DEFINE_FMAX_METHOD(bf16, nv_bfloat16);
#undef DEFINE_FMAX_METHOD

  lua["_fmax_f8E4M3FN"] = [](fp8_e4m3fn lhs, fp8_e4m3fn rhs) -> fp8_e4m3fn {
    ADD_CORE_MODULE_RANGE("core_fmax");
    return fp8_e4m3fn(std::max(float(lhs), float(rhs)));
  };

  lua["_fmin_f8E4M3FN"] = [](fp8_e4m3fn lhs, fp8_e4m3fn rhs) -> fp8_e4m3fn {
    ADD_CORE_MODULE_RANGE("core_fmin");
    return fp8_e4m3fn(std::min(float(lhs), float(rhs)));
  };

  //===----------------------------------------------------------------------===//
  // executor.shift_left, executor.shift_right_arithmetic,
  // executor.shift_right_logical
  //===----------------------------------------------------------------------===//

#define DEFINE_SHIFT_LEFT_METHOD(suffix, type)                                 \
  lua["_shift_lefti_" #suffix] = [](type lhs, type rhs) -> type {              \
    ADD_CORE_MODULE_RANGE("core_shift_left");                                  \
    return lhs << rhs;                                                         \
  }

  DEFINE_SHIFT_LEFT_METHOD(i64, int64_t);
  DEFINE_SHIFT_LEFT_METHOD(i32, int32_t);
  DEFINE_SHIFT_LEFT_METHOD(i16, int16_t);
  DEFINE_SHIFT_LEFT_METHOD(i8, int8_t);
  DEFINE_SHIFT_LEFT_METHOD(i4, nv_int4);
#undef DEFINE_SHIFT_LEFT_METHOD

#define DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(suffix, type)                     \
  lua["_shift_right_arithmetici_" #suffix] = [](type lhs, type rhs) -> type {  \
    ADD_CORE_MODULE_RANGE("core_shift right arithmetic");                      \
    return lhs >> rhs;                                                         \
  }

  DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(i64, int64_t);
  DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(i32, int32_t);
  DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(i16, int16_t);
  DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(i8, int8_t);
  DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD(i4, nv_int4);
#undef DEFINE_SHIFT_RIGHT_ARITHMETIC_METHOD

#define DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(suffix, dtype)                       \
  lua["_shift_right_logicali_" #suffix] = [](dtype lhs, dtype rhs) -> dtype {  \
    ADD_CORE_MODULE_RANGE("core_shift right logical");                         \
    return static_cast<dtype>(                                                 \
        static_cast<typename std::make_unsigned<dtype>::type>(lhs) >> rhs);    \
  }

  DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(i64, int64_t);
  DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(i32, int32_t);
  DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(i16, int16_t);
  DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(i8, int8_t);
  DEFINE_SHIFT_RIGHT_LOGICAL_METHOD(i4, nv_int4);
#undef DEFINE_SHIFT_RIGHT_LOGICAL_METHOD

  //===----------------------------------------------------------------------===//
  // executor.sitofp
  //===----------------------------------------------------------------------===//
  // executor.sitofp, executor.uitofp
  //===----------------------------------------------------------------------===//

#define REGISTER_TOFP_FUNCS(inpSuffix, resSuffix, inpType, resType)            \
  lua["_sitofp_" #inpSuffix "_" #resSuffix] = sitofp<inpType, resType>;        \
  lua["_uitofp_" #inpSuffix "_" #resSuffix] = uitofp<inpType, resType>;

  REGISTER_TOFP_FUNCS(i8, f8E4M3FN, int8_t, fp8_e4m3fn);
  REGISTER_TOFP_FUNCS(i16, f8E4M3FN, int16_t, fp8_e4m3fn);
  REGISTER_TOFP_FUNCS(i32, f8E4M3FN, int32_t, fp8_e4m3fn);
  REGISTER_TOFP_FUNCS(i64, f8E4M3FN, int64_t, fp8_e4m3fn);
  REGISTER_TOFP_FUNCS(i8, bf16, int8_t, nv_bfloat16);
  REGISTER_TOFP_FUNCS(i16, bf16, int16_t, nv_bfloat16);
  REGISTER_TOFP_FUNCS(i32, bf16, int32_t, nv_bfloat16);
  REGISTER_TOFP_FUNCS(i64, bf16, int64_t, nv_bfloat16);
  REGISTER_TOFP_FUNCS(i8, f16, int8_t, __half);
  REGISTER_TOFP_FUNCS(i16, f16, int16_t, __half);
  REGISTER_TOFP_FUNCS(i32, f16, int32_t, __half);
  REGISTER_TOFP_FUNCS(i64, f16, int64_t, __half);
  REGISTER_TOFP_FUNCS(i8, f32, int8_t, float);
  REGISTER_TOFP_FUNCS(i16, f32, int16_t, float);
  REGISTER_TOFP_FUNCS(i32, f32, int32_t, float);
  REGISTER_TOFP_FUNCS(i64, f32, int64_t, float);
  REGISTER_TOFP_FUNCS(i8, f64, int8_t, double);
  REGISTER_TOFP_FUNCS(i16, f64, int16_t, double);
  REGISTER_TOFP_FUNCS(i32, f64, int32_t, double);
  REGISTER_TOFP_FUNCS(i64, f64, int64_t, double);
  REGISTER_TOFP_FUNCS(i4, f32, nv_int4, float);
  REGISTER_TOFP_FUNCS(i4, f64, nv_int4, double);
  REGISTER_TOFP_FUNCS(i4, f16, nv_int4, __half);
  REGISTER_TOFP_FUNCS(i4, f8E4M3FN, nv_int4, fp8_e4m3fn);
  REGISTER_TOFP_FUNCS(i4, bf16, nv_int4, nv_bfloat16);

#undef REGISTER_TOFP_FUNCS

  //===----------------------------------------------------------------------===//
  // executor.fptosi
  //===----------------------------------------------------------------------===//

#define DEFINE_FPTOSI_METHOD(inpSuffix, resSuffix, inpType, resType)           \
  lua["_fptosi_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {   \
    ADD_CORE_MODULE_RANGE("core_fptosi");                                      \
    return static_cast<resType>(input);                                        \
  }

  DEFINE_FPTOSI_METHOD(f8E4M3FN, i8, fp8_e4m3fn, int8_t);
  DEFINE_FPTOSI_METHOD(f8E4M3FN, i16, fp8_e4m3fn, int16_t);
  DEFINE_FPTOSI_METHOD(f8E4M3FN, i32, fp8_e4m3fn, int32_t);
  DEFINE_FPTOSI_METHOD(f8E4M3FN, i64, fp8_e4m3fn, int64_t);
  DEFINE_FPTOSI_METHOD(bf16, i8, nv_bfloat16, int8_t);
  DEFINE_FPTOSI_METHOD(bf16, i16, nv_bfloat16, int16_t);
  DEFINE_FPTOSI_METHOD(bf16, i32, nv_bfloat16, int32_t);
  DEFINE_FPTOSI_METHOD(bf16, i64, nv_bfloat16, int64_t);
  DEFINE_FPTOSI_METHOD(f16, i8, __half, int8_t);
  DEFINE_FPTOSI_METHOD(f16, i16, __half, int16_t);
  DEFINE_FPTOSI_METHOD(f16, i32, __half, int32_t);
  DEFINE_FPTOSI_METHOD(f16, i64, __half, int64_t);
  DEFINE_FPTOSI_METHOD(f32, i8, float, int8_t);
  DEFINE_FPTOSI_METHOD(f32, i16, float, int16_t);
  DEFINE_FPTOSI_METHOD(f32, i32, float, int32_t);
  DEFINE_FPTOSI_METHOD(f32, i64, float, int64_t);
  DEFINE_FPTOSI_METHOD(f64, i8, double, int8_t);
  DEFINE_FPTOSI_METHOD(f64, i16, double, int16_t);
  DEFINE_FPTOSI_METHOD(f64, i32, double, int32_t);
  DEFINE_FPTOSI_METHOD(f64, i64, double, int64_t);
  DEFINE_FPTOSI_METHOD(f64, i4, double, nv_int4);
  DEFINE_FPTOSI_METHOD(f32, i4, float, nv_int4);
  DEFINE_FPTOSI_METHOD(f16, i4, __half, nv_int4);
  DEFINE_FPTOSI_METHOD(bf16, i4, nv_bfloat16, nv_int4);
  DEFINE_FPTOSI_METHOD(f8E4M3FN, i4, fp8_e4m3fn, nv_int4);
#undef DEFINE_FPTOSI_METHOD

  //===----------------------------------------------------------------------===//
  // executor.zext | executor.siext
  //===----------------------------------------------------------------------===//

#define DEFINE_IEXT_METHOD(inpSuffix, resSuffix, inpType, resType)             \
  lua["_zext_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {     \
    ADD_CORE_MODULE_RANGE("core_zext");                                        \
    auto tmp =                                                                 \
        static_cast<u##resType>(*reinterpret_cast<u##inpType *>(&input));      \
    return *reinterpret_cast<const resType *>(&tmp);                           \
  };                                                                           \
  lua["_siext_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {    \
    ADD_CORE_MODULE_RANGE("core_siext");                                       \
    auto tmp = static_cast<resType>(input);                                    \
    return *reinterpret_cast<const resType *>(&tmp);                           \
  }
  DEFINE_IEXT_METHOD(i32, i64, int32_t, int64_t);
  DEFINE_IEXT_METHOD(i8, i32, int8_t, int64_t);
  DEFINE_IEXT_METHOD(i8, i64, int8_t, int64_t);
  DEFINE_IEXT_METHOD(i1, i8, int8_t, int8_t);
  DEFINE_IEXT_METHOD(i1, i32, int8_t, int32_t);
  DEFINE_IEXT_METHOD(i1, i64, int8_t, int32_t);

#define DEFINE_IEXT_METHOD_I4(inpSuffix, resSuffix, inpType, resType)          \
  lua["_zext_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {     \
    ADD_CORE_MODULE_RANGE("core_zext");                                        \
    auto tmp = static_cast<u##resType>(*reinterpret_cast<nv_uint4 *>(&input)); \
    return *reinterpret_cast<const resType *>(&tmp);                           \
  };                                                                           \
  lua["_siext_" #inpSuffix "_" #resSuffix] = [](inpType input) -> resType {    \
    ADD_CORE_MODULE_RANGE("core_siext");                                       \
    auto tmp = static_cast<resType>(input);                                    \
    return *reinterpret_cast<const resType *>(&tmp);                           \
  }

  DEFINE_IEXT_METHOD_I4(i4, i8, nv_int4, int8_t);
  DEFINE_IEXT_METHOD_I4(i4, i16, nv_int4, int16_t);
  DEFINE_IEXT_METHOD_I4(i4, i32, nv_int4, int32_t);
  DEFINE_IEXT_METHOD_I4(i4, i64, nv_int4, int32_t);
#undef DEFINE_IEXT_METHOD
#undef DEFINE_IEXT_METHOD_I4

  //===----------------------------------------------------------------------===//
  // executor.trunc
  //===----------------------------------------------------------------------===//

#define DEFINE_TRUNC_METHOD(resSuffix, inpSuffix, resType, inpType)            \
  lua["_trunc_i" #inpSuffix "_i" #resSuffix] = [](inpType input) -> resType {  \
    ADD_CORE_MODULE_RANGE("core_trunc");                                       \
    return integerTruncate<inpType, resType, resSuffix>(input);                \
  }

  DEFINE_TRUNC_METHOD(32, 64, int32_t, int64_t);
  DEFINE_TRUNC_METHOD(16, 64, int16_t, int64_t);
  DEFINE_TRUNC_METHOD(8, 64, int8_t, int64_t);
  DEFINE_TRUNC_METHOD(4, 64, nv_int4, int64_t);
  DEFINE_TRUNC_METHOD(1, 64, int8_t, int64_t);

  DEFINE_TRUNC_METHOD(16, 32, int16_t, int32_t);
  DEFINE_TRUNC_METHOD(8, 32, int8_t, int32_t);
  DEFINE_TRUNC_METHOD(4, 32, nv_int4, int32_t);
  DEFINE_TRUNC_METHOD(1, 32, int8_t, int32_t);

  DEFINE_TRUNC_METHOD(4, 8, nv_int4, int8_t);
  DEFINE_TRUNC_METHOD(1, 8, int8_t, int8_t);

#undef DEFINE_TRUNC_METHOD

  //===----------------------------------------------------------------------===//
  // executor.smin/executor.smax
  //===----------------------------------------------------------------------===//

#define DEFINE_MIN_MAX(suffix, inpType)                                        \
  lua["_smin_" #suffix] = [](inpType lhs, inpType rhs) -> inpType {            \
    ADD_CORE_MODULE_RANGE("core_imin");                                        \
    return std::min(lhs, rhs);                                                 \
  };                                                                           \
  lua["_smax_" #suffix] = [](inpType lhs, inpType rhs) -> inpType {            \
    ADD_CORE_MODULE_RANGE("core_imax");                                        \
    return std::max(lhs, rhs);                                                 \
  }

  DEFINE_MIN_MAX(i8, int8_t);
  DEFINE_MIN_MAX(i16, int16_t);
  DEFINE_MIN_MAX(i32, int32_t);
  DEFINE_MIN_MAX(i64, int64_t);
  DEFINE_MIN_MAX(i4, nv_int4);

#undef DEFINE_MIN_MAX

  //===----------------------------------------------------------------------===//
  // executor.extf/executor.truncf
  //===----------------------------------------------------------------------===//

#define DEFINE_FLOAT_CAST_METHODS(shortSuffix, longSuffix, shortType,          \
                                  longType)                                    \
  lua["_extf_" #shortSuffix "_" #longSuffix] =                                 \
      [](shortType input) -> longType {                                        \
    ADD_CORE_MODULE_RANGE("core_extf");                                        \
    return longType(input);                                                    \
  };                                                                           \
  lua["_truncf_" #longSuffix "_" #shortSuffix] =                               \
      [](longType input) -> shortType {                                        \
    ADD_CORE_MODULE_RANGE("core_truncf");                                      \
    return shortType(input);                                                   \
  }

  DEFINE_FLOAT_CAST_METHODS(f16, f32, __half, float);
  DEFINE_FLOAT_CAST_METHODS(f8E4M3FN, f16, fp8_e4m3fn, __half);
  DEFINE_FLOAT_CAST_METHODS(f8E4M3FN, f32, fp8_e4m3fn, float);
  DEFINE_FLOAT_CAST_METHODS(bf16, f32, nv_bfloat16, float);

#undef DEFINE_FLOAT_CAST_METHODS

  //===----------------------------------------------------------------------===//
  // executor.fcmp
  //===----------------------------------------------------------------------===//

#define DEFINE_OFCMP_METHOD(pred, suffix, type, cast_to, op)                   \
  lua["_fcmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_fcmp_ordered");                                \
    return !(isNan(lhs) || isNan(rhs)) &&                                      \
           (static_cast<cast_to>(lhs) op static_cast<cast_to>(rhs));           \
  }
#define DEFINE_UFCMP_METHOD(pred, suffix, type, cast_to, op)                   \
  lua["_fcmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_fcmp_unordered");                              \
    return (isNan(lhs) || isNan(rhs)) ||                                       \
           (static_cast<cast_to>(lhs) op static_cast<cast_to>(rhs));           \
  }
#define DEFINE_OTHERFCMP_METHOD(pred, suffix, type, cast_to, impl)             \
  lua["_fcmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_fcmp_other");                                  \
    return impl;                                                               \
  }
#define DEFINE_FCMP_ORD_METHOD(pred, suffix, type)                             \
  lua["_fcmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_fcmp_other");                                  \
    return (isNan(lhs) || isNan(rhs)) ? false : true;                          \
  }
#define DEFINE_FCMP_UORD_METHOD(pred, suffix, type)                            \
  lua["_fcmp_" #pred "_" #suffix] = [](type lhs, type rhs) -> int8_t {         \
    ADD_CORE_MODULE_RANGE("core_fcmp_other");                                  \
    return (isNan(lhs) || isNan(rhs)) ? true : false;                          \
  }
#define DEFINE_FCMP_METHODS(suffix, type, cast_to)                             \
  DEFINE_OTHERFCMP_METHOD(_false, suffix, type, cast_to, false);               \
  DEFINE_OTHERFCMP_METHOD(_true, suffix, type, cast_to, true);                 \
  DEFINE_OFCMP_METHOD(oeq, suffix, type, cast_to, ==);                         \
  DEFINE_OFCMP_METHOD(ogt, suffix, type, cast_to, >);                          \
  DEFINE_OFCMP_METHOD(oge, suffix, type, cast_to, >=);                         \
  DEFINE_OFCMP_METHOD(olt, suffix, type, cast_to, <);                          \
  DEFINE_OFCMP_METHOD(ole, suffix, type, cast_to, <=);                         \
  DEFINE_OFCMP_METHOD(one, suffix, type, cast_to, !=);                         \
  DEFINE_FCMP_ORD_METHOD(ord, suffix, type);                                   \
  DEFINE_UFCMP_METHOD(ueq, suffix, type, cast_to, ==);                         \
  DEFINE_UFCMP_METHOD(ugt, suffix, type, cast_to, >);                          \
  DEFINE_UFCMP_METHOD(uge, suffix, type, cast_to, >=);                         \
  DEFINE_UFCMP_METHOD(ult, suffix, type, cast_to, <);                          \
  DEFINE_UFCMP_METHOD(ule, suffix, type, cast_to, <=);                         \
  DEFINE_UFCMP_METHOD(une, suffix, type, cast_to, !=);                         \
  DEFINE_FCMP_UORD_METHOD(uno, suffix, type)

  DEFINE_FCMP_METHODS(f32, float, float);
  DEFINE_FCMP_METHODS(f64, double, double);
  DEFINE_FCMP_METHODS(f16, __half, __half);
  DEFINE_FCMP_METHODS(f8E4M3FN, fp8_e4m3fn, float);
  DEFINE_FCMP_METHODS(bf16, nv_bfloat16, float);

#undef DEFINE_OFCMP_METHOD
#undef DEFINE_UFCMP_METHOD
#undef DEFINE_OTHERFCMP_METHOD

  //===----------------------------------------------------------------------===//
  // arith.remf
  //===----------------------------------------------------------------------===//

  lua["_remf_f64"] = remf<double>;
  lua["_remf_f32"] = remf<float>;
  lua["_remf_f16"] = remf<__half>;
  lua["_remf_f8E4M3FN"] = remf<fp8_e4m3fn>;
  lua["_remf_bf16"] = remf<nv_bfloat16>;

  //===----------------------------------------------------------------------===//
  // aggregate ops
  //===----------------------------------------------------------------------===//

  lua["executor_struct_extract_value"] = [](sol::table table, int64_t pos) {
    ADD_CORE_MODULE_RANGE("core_struct_extract_value");
    auto value = table[pos];
    return value;
  };
  lua["executor_struct_set_value"] = [](sol::table table, int64_t pos,
                                        sol::object value) {
    ADD_CORE_MODULE_RANGE("core_struct_set_value");
    table[pos] = value;
    return table;
  };

  //===----------------------------------------------------------------------===//
  // Pointer cast operations
  //===----------------------------------------------------------------------===//

  // The pointer operation name is `_[opname]_[ptr width]_[int width]`.

  lua["_ptrtoint_i64_i32"] = [](uintptr_t pointer) {
    return static_cast<uint32_t>(pointer);
  };
  lua["_ptrtoint_i64_i64"] = [](uintptr_t pointer) {
    return static_cast<uint64_t>(pointer);
  };
  lua["_inttoptr_i64_i32"] = [](int32_t pointer) {
    return static_cast<uint32_t>(pointer);
  };
  lua["_inttoptr_i64_i64"] = [](int64_t pointer) {
    return static_cast<uint64_t>(pointer);
  };

  //===----------------------------------------------------------------------===//
  // Alignment operations
  //===----------------------------------------------------------------------===//

  lua["_alignto_i64"] = alignToImpl<int64_t>;
  lua["_alignto_i32"] = alignToImpl<int32_t>;

  //===----------------------------------------------------------------------===//
  // host memory ops
  //===----------------------------------------------------------------------===//

  lua["_dealloc"] = [allocTracker](sol::this_state state, uintptr_t ptr) {
    ADD_CORE_MODULE_RANGE("core_dealloc");
    MTRT_DBGF("dealloc ptr @ 0x%lx", ptr);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(safeDeallocate(*allocTracker, ptr),
                                      state, );
  };

  lua["executor_alloc"] = [allocTracker](sol::this_state state, size_t bytes,
                                         unsigned alignment) -> uintptr_t {
    ADD_CORE_MODULE_RANGE("core_alloc");
    MTRT_DBGF("executor_alloc: %lu bytes align(%u)", bytes, alignment);
    StatusOr<PointerInfo> buffer =
        allocate(*allocTracker, PointerType::host, bytes, alignment, {});
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(buffer, state, 0);
    return buffer->ptr;
  };

  lua["executor_memcpy"] = [allocTracker](sol::this_state state, uintptr_t src,
                                          size_t srcOffset, uintptr_t dst,
                                          size_t destOffset, size_t numBytes) {
    ADD_CORE_MODULE_RANGE("core_memcpy");
    void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
    void *dstPtr = reinterpret_cast<void *>(dst + destOffset);

    AllocTracker &tracker = *allocTracker;
    if (tracker.contains(src)) {
      const PointerInfo &srcInfo = tracker.get(src);
      assert(srcInfo.ptr && srcInfo.isHostVisible() &&
             "expected host visible src pointer");
    }
    if (tracker.contains(dst)) {
      const PointerInfo &dstInfo = tracker.get(dst);
      assert(dstInfo.ptr && dstInfo.isHostVisible() &&
             "expected host visible dst pointer");
    }
    MTRT_DBGF("executor_memcpy host-host %lu bytes src %lx + %lu dst %lx + %lu",
              numBytes, src, srcOffset, dst, destOffset);
    std::memcpy(dstPtr, srcPtr, numBytes);
  };

// Create a method `executor_load_[suffix]` that loads a value of type
// `resultType` from a buffer of types `loadType`. The `loadType` is
// static-casted to the `resultType`.
#define DEFINE_LOAD_METHOD(suffix, loadType, resultType)                       \
  lua["_load_" #suffix] = [](uintptr_t pointer, size_t offset) -> resultType { \
    ADD_CORE_MODULE_RANGE("core_load_" #suffix);                               \
    MTRT_DBGF("executor_load_" #suffix " %lx + %lu", pointer, offset);         \
    return static_cast<resultType>(                                            \
        *reinterpret_cast<loadType *>(pointer + offset));                      \
  }

  DEFINE_LOAD_METHOD(ptr_host, uintptr_t, uintptr_t);
  DEFINE_LOAD_METHOD(ptr_host_pinned, uintptr_t, uintptr_t);
  DEFINE_LOAD_METHOD(ptr_device, uintptr_t, uintptr_t);
  DEFINE_LOAD_METHOD(f64, double, double);
  DEFINE_LOAD_METHOD(f32, float, float);
  DEFINE_LOAD_METHOD(i64, int64_t, int64_t);
  DEFINE_LOAD_METHOD(i32, int32_t, int32_t);
  DEFINE_LOAD_METHOD(i16, int16_t, int16_t);
  DEFINE_LOAD_METHOD(i8, int8_t, int8_t);
  DEFINE_LOAD_METHOD(f16, __half, __half);
  DEFINE_LOAD_METHOD(f8E4M3FN, fp8_e4m3fn, fp8_e4m3fn);
  DEFINE_LOAD_METHOD(bf16, nv_bfloat16, nv_bfloat16);
  DEFINE_LOAD_METHOD(i4, nv_int4, nv_int4);

  // Define i1 load specially to enforce truncation. Otherwise, the Lua
  // comparisons might not work correctly.
  lua["_load_i1"] = [](uintptr_t pointer, size_t offset) -> int8_t {
    ADD_CORE_MODULE_RANGE("core_load_i1");
    return 0x1 & *reinterpret_cast<const int8_t *>(pointer + offset);
  };
#undef DEFINE_LOAD_METHOD

// Create a method `executor_store_[suffix]` that stores a value of type
// `inputType` to a buffer of types `storeType`. The `inputType` is
// static-casted to the `storeType`.
#define DEFINE_STORE_METHOD(suffix, storeType, inputType)                      \
  lua["_store_" #suffix] = [allocTracker](sol::this_state state,               \
                                          uintptr_t pointer, size_t offset,    \
                                          inputType value) {                   \
    ADD_CORE_MODULE_RANGE("core_store_" #suffix);                              \
    MTRT_DBGF("executor_store_" #suffix " %lx + %lu", pointer, offset);        \
    if (!checkAccessBounds<storeType>(state, *allocTracker, pointer,           \
                                      offset)) {                               \
      return;                                                                  \
    }                                                                          \
    *reinterpret_cast<storeType *>(pointer + offset) =                         \
        static_cast<storeType>(value);                                         \
  }

  DEFINE_STORE_METHOD(f64, double, double);
  DEFINE_STORE_METHOD(f32, float, float);
  DEFINE_STORE_METHOD(i64, int64_t, int64_t);
  DEFINE_STORE_METHOD(i32, int32_t, int32_t);
  DEFINE_STORE_METHOD(i16, int16_t, int16_t);
  DEFINE_STORE_METHOD(ptr_device, uintptr_t, uintptr_t);
  DEFINE_STORE_METHOD(ptr_host, uintptr_t, uintptr_t);
  DEFINE_STORE_METHOD(ptr_host_pinned, uintptr_t, uintptr_t);
  DEFINE_STORE_METHOD(i8, int8_t, int8_t);
  DEFINE_STORE_METHOD(i1, int8_t, int8_t);
  DEFINE_STORE_METHOD(f16, __half, __half);
  DEFINE_STORE_METHOD(f8E4M3FN, fp8_e4m3fn, fp8_e4m3fn);
  DEFINE_STORE_METHOD(bf16, nv_bfloat16, nv_bfloat16);
  DEFINE_STORE_METHOD(i4, nv_int4, nv_int4);

#undef DEFINE_STORE_METHOD
  //===----------------------------------------------------------------------===//
  // MemSet Ops
  //===----------------------------------------------------------------------===//
  lua["__memset_32"] = [](sol::this_state state, uintptr_t pointer,
                          size_t offset, size_t numBytes, uint32_t fillInt) {
    MTRT_DBGF("memset32 @ 0x%lx, %lu bytes fill value = %u", pointer, numBytes,
              fillInt);
    __memset_32(pointer, offset, numBytes, fillInt);
  };

  lua["__memset_16"] = [](sol::this_state state, uintptr_t pointer,
                          size_t offset, size_t numBytes, uint16_t fillInt) {
    MTRT_DBGF("memset16 @ 0x%lx, %lu bytes fill value = %u", pointer, numBytes,
              fillInt);
    __memset_16(pointer, offset, numBytes, fillInt);
  };

  lua["__memset_8"] = [](sol::this_state state, uintptr_t pointer,
                         size_t offset, size_t numBytes, uint8_t fillInt) {
    MTRT_DBGF("memset8 @ 0x%lx, %lu bytes fill value = %u", pointer, numBytes,
              fillInt);
    __memset_8(pointer, offset, numBytes, fillInt);
  };

  //===----------------------------------------------------------------------===//
  // Stridded Copy Methods
  //===----------------------------------------------------------------------===//

  /// Generic strided copy operation. The signature is variadic in order to
  /// support multiple ranks. The variadic arguments should contain:
  /// clang-format off
  /// (srcPtr, srcPtrAligned, srcOfft, ...[srcShape], ...[srcStrides],
  ///  dstPtr, dstPtrAligned, dstOfft, ...[dstShape], ...[dstStrides])
  /// clang-format on
  lua["_strided_memref_copy"] = [](sol::this_state state, int32_t rank,
                                   int32_t elemSize,
                                   sol::variadic_args varArgs) {
    ADD_CORE_MODULE_RANGE("core_strided_memref_copy");
    if (varArgs.size() != 2 * getNumArgsPerMemRef(rank)) {
      luaL_error(
          state,
          "unexpected variadic argument pack size in for strided memref copy");
      return;
    }

    int64_t srcOffset = 0;
    std::vector<int64_t> srcShape, srcStrides;
    uintptr_t srcData =
        getMemRefInfo(state, varArgs, rank, 0, srcOffset, srcShape, srcStrides);
    if (!srcData)
      return;

    int64_t dstOffset = 0;
    std::vector<int64_t> dstShape, dstStrides;
    uintptr_t dstData =
        getMemRefInfo(state, varArgs, rank, 1, dstOffset, dstShape, dstStrides);
    if (!dstData)
      return;

    executeStridedCopy(
        elemSize, srcData, srcOffset, srcShape, srcStrides, dstData, dstOffset,
        dstShape, dstStrides,
        [](void *dst, void *src, size_t size) { std::memcpy(dst, src, size); });
  };

  //===----------------------------------------------------------------------===//
  // Builtin Math Ops
  //===----------------------------------------------------------------------===//

#define DEFINE_UNARY_OP_(name, suffix, type, op)                               \
  lua["_" #name "_" #suffix] = [](type inp) -> type {                          \
    ADD_CORE_MODULE_RANGE("core_builtin_" #name);                              \
    return op(inp);                                                            \
  }

#define DEFINE_UNARY_SPECIAL_TYPES_(name, suffix, type, op)                    \
  lua["_" #name "_" #suffix] = [](type inp) -> type {                          \
    ADD_CORE_MODULE_RANGE("core_builtin_" #name);                              \
    return static_cast<type>(op(static_cast<float>(inp)));                     \
  }

#define DEFINE_BINARY_OP_(name, suffix, type, op)                              \
  lua["_" #name "_" #suffix] = [=](type lhs, type rhs) -> type {               \
    ADD_CORE_MODULE_RANGE("core_builtin_" #name);                              \
    return op(lhs, rhs);                                                       \
  }

#define DEFINE_BINARY_SPECIAL_TYPES_(name, suffix, type, op)                   \
  lua["_" #name "_" #suffix] = [=](type lhs, type rhs) -> type {               \
    ADD_CORE_MODULE_RANGE("core_builtin_" #name);                              \
    return static_cast<type>(                                                  \
        op(static_cast<float>(lhs), static_cast<float>(rhs)));                 \
  }

// Integer unary ops
#define DEFINE_INT_UNARY_OP(name, op)                                          \
  DEFINE_UNARY_OP_(name, i32, int32_t, op);                                    \
  DEFINE_UNARY_OP_(name, i64, int64_t, op);                                    \
  DEFINE_UNARY_OP_(name, i16, int64_t, op);                                    \
  DEFINE_UNARY_OP_(name, i8, int8_t, op);                                      \
  DEFINE_UNARY_SPECIAL_TYPES_(name, i4, nv_int4, op)

  // 'Math' unary ops
  DEFINE_INT_UNARY_OP(absi, std::abs);

#undef DEFINE_INT_UNARY_OP

// Unary float ops
#define DEFINE_FLOAT_UNARY_OP(name, op)                                        \
  DEFINE_UNARY_OP_(name, f32, float, op);                                      \
  DEFINE_UNARY_OP_(name, f64, double, op);                                     \
  DEFINE_UNARY_SPECIAL_TYPES_(name, f16, half, op);                            \
  DEFINE_UNARY_SPECIAL_TYPES_(name, f8E4M3FN, fp8_e4m3fn, op);                 \
  DEFINE_UNARY_SPECIAL_TYPES_(name, bf16, nv_bfloat16, op)

  // Unary float ops
  DEFINE_FLOAT_UNARY_OP(absf, std::abs);
  DEFINE_FLOAT_UNARY_OP(cbrt, std::cbrt);
  DEFINE_FLOAT_UNARY_OP(ceil, std::ceil);
  DEFINE_FLOAT_UNARY_OP(cos, std::cos);
  DEFINE_FLOAT_UNARY_OP(sin, std::sin);
  DEFINE_FLOAT_UNARY_OP(erf, std::erf);
  DEFINE_FLOAT_UNARY_OP(exp, std::exp);
  DEFINE_FLOAT_UNARY_OP(exp2, std::exp2);
  DEFINE_FLOAT_UNARY_OP(expm1, std::expm1);
  DEFINE_FLOAT_UNARY_OP(floor, std::floor);
  DEFINE_FLOAT_UNARY_OP(log, std::log);
  DEFINE_FLOAT_UNARY_OP(log10, std::log10);
  DEFINE_FLOAT_UNARY_OP(log1p, std::log1p);
  DEFINE_FLOAT_UNARY_OP(log2, std::log2);
  DEFINE_FLOAT_UNARY_OP(negf, negate);
  DEFINE_FLOAT_UNARY_OP(sqrt, std::sqrt);
  DEFINE_FLOAT_UNARY_OP(tan, std::tan);
  DEFINE_FLOAT_UNARY_OP(tanh, std::tanh);
  DEFINE_FLOAT_UNARY_OP(round, std::round);

#undef DEFINE_FLOAT_UNARY_OP

// Binary float ops
#define DEFINE_FLOAT_BINARY_OP(name, op)                                       \
  DEFINE_BINARY_OP_(name, f32, float, op);                                     \
  DEFINE_BINARY_OP_(name, f64, double, op);                                    \
  DEFINE_BINARY_SPECIAL_TYPES_(name, f16, half, op);                           \
  DEFINE_BINARY_SPECIAL_TYPES_(name, f8E4M3FN, fp8_e4m3fn, op);                \
  DEFINE_BINARY_SPECIAL_TYPES_(name, bf16, nv_bfloat16, op)

  DEFINE_FLOAT_BINARY_OP(atan2, std::atan2);
  DEFINE_FLOAT_BINARY_OP(copysign, std::copysign);
  DEFINE_FLOAT_BINARY_OP(powf, std::pow);
}

#undef DEFINE_BINARY_OP_
#undef DEFINE_UNARY_OP_

namespace mlirtrt::runtime {
void registerLuaCoreRuntimeExtension() {
  registerLuaRuntimeExtension(
      "core",
      LuaRuntimeExtension{
          [](const RuntimeSessionOptions &options, lua_State *state,
             PinnedMemoryAllocator *pinnedMemoryAllocator,
             AllocTracker *allocTracker, ResourceTracker *resourceTracker) {
            registerExecutorCoreModuleLuaRuntimeMethods(
                state, pinnedMemoryAllocator, allocTracker);
          }});
}
} // namespace mlirtrt::runtime
