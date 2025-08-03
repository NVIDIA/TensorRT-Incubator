

//===- DataTypes.cpp ------------------------------------------------------===//
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
//
// Implementation of Float8, Float16 and BFloat16 conversion functions.
//
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include <cstring>

using namespace mlirtrt::runtime;

namespace {
// Union to enable type punning between float and uint32_t
union FloatBits {
  float f;
  uint32_t u;
};
} // namespace

//===----------------------------------------------------------------------===//
// Float8 E4M3FN Implementation
//===----------------------------------------------------------------------===//

F8E4M3FN::F8E4M3FN(float value) noexcept {
  // Handle special cases first
  if (std::isnan(value)) {
    bits_ = 0x7F; // NaN representation
    return;
  }

  FloatBits fb;
  fb.f = value;
  uint32_t f32_bits = fb.u;

  // Extract components
  uint32_t sign = (f32_bits >> 31) & 0x1;
  uint32_t exponent = (f32_bits >> 23) & 0xFF;
  uint32_t mantissa = f32_bits & 0x7FFFFF;

  // Handle zero
  if (exponent == 0 && mantissa == 0) {
    bits_ = sign << 7;
    return;
  }

  // Convert exponent from float32 bias (127) to float8 E4M3FN bias (7)
  int32_t unbiased_exp = static_cast<int32_t>(exponent) - 127;
  int32_t f8_exp = unbiased_exp + 7;

  // Check for overflow - E4M3FN cannot represent infinity
  if (f8_exp >= 15) {
    // Clamp to max value (0x7E for positive, 0xFE for negative)
    bits_ = (sign << 7) | 0x7E;
    return;
  }

  if (f8_exp <= 0) {
    // Underflow - handle denormals or return zero
    if (f8_exp < -3) {
      // Too small even for denormal
      bits_ = sign << 7;
      return;
    }

    // Denormal number
    mantissa |= 0x800000; // Add implicit leading 1
    uint32_t shift = 1 - f8_exp;
    mantissa >>= shift;

    // Round to nearest even
    uint32_t guard = mantissa & 0xFFFFF;
    mantissa >>= 20;

    if (guard > 0x80000 || (guard == 0x80000 && (mantissa & 1))) {
      mantissa++;
    }

    bits_ = (sign << 7) | (mantissa & 0x7);
    return;
  }

  // Normal number
  // Round mantissa from 23 bits to 3 bits
  uint32_t guard = mantissa & 0xFFFFF;
  mantissa >>= 20;

  // Round to nearest even
  if (guard > 0x80000 || (guard == 0x80000 && (mantissa & 1))) {
    mantissa++;
    if (mantissa > 0x7) {
      // Mantissa overflow
      mantissa = 0;
      f8_exp++;
      if (f8_exp >= 15) {
        // Exponent overflow - clamp to max
        bits_ = (sign << 7) | 0x7E;
        return;
      }
    }
  }

  bits_ = (sign << 7) | (f8_exp << 3) | (mantissa & 0x7);
}

F8E4M3FN::operator float() const noexcept {
  // Extract components
  uint32_t sign = (bits_ >> 7) & 0x1;
  uint32_t exponent = (bits_ >> 3) & 0xF;
  uint32_t mantissa = bits_ & 0x7;

  // Check for NaN
  if ((bits_ & 0x7F) == 0x7F) {
    FloatBits fb;
    fb.u = 0x7FC00000; // Quiet NaN
    return fb.f;
  }

  // Check for zero
  if ((bits_ & 0x7F) == 0) {
    FloatBits fb;
    fb.u = sign << 31;
    return fb.f;
  }

  if (exponent == 0) {
    // Denormal number
    if (mantissa == 0) {
      // Zero
      FloatBits fb;
      fb.u = sign << 31;
      return fb.f;
    }

    // Find leading one
    uint32_t leading_zeros = __builtin_clz(mantissa) - 29;
    mantissa <<= (leading_zeros + 1);
    exponent = 127 - 7 - leading_zeros;
    mantissa = (mantissa & 0x7) << 20;

    FloatBits fb;
    fb.u = (sign << 31) | (exponent << 23) | mantissa;
    return fb.f;
  }

  // Normal number
  // Convert exponent from float8 bias (7) to float32 bias (127)
  exponent = exponent - 7 + 127;

  // Expand mantissa from 3 bits to 23 bits
  mantissa <<= 20;

  FloatBits fb;
  fb.u = (sign << 31) | (exponent << 23) | mantissa;
  return fb.f;
}

int8_t F8E4M3FN::toInt8Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 127.0f)
    return 127;
  if (val <= -128.0f)
    return -128;
  return static_cast<int8_t>(val);
}

int16_t F8E4M3FN::toInt16Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 32767.0f)
    return 32767;
  if (val <= -32768.0f)
    return -32768;
  return static_cast<int16_t>(val);
}

int32_t F8E4M3FN::toInt32Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 2147483647.0f)
    return 2147483647;
  if (val <= -2147483648.0f)
    return -2147483648;
  return static_cast<int32_t>(val);
}

int64_t F8E4M3FN::toInt64Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  // Note: float cannot represent all int64_t values precisely
  // 9223372036854775807LL is INT64_MAX
  if (val >= 9.223372e18f)
    return 9223372036854775807LL;
  if (val <= -9.223372e18f)
    return (-9223372036854775807LL - 1);
  return static_cast<int64_t>(val);
}

//===----------------------------------------------------------------------===//
// Float16 Implementation
//===----------------------------------------------------------------------===//

Float16::Float16(float value) noexcept {
  // Convert float32 to float16 using round-to-nearest-even
  FloatBits fb;
  fb.f = value;
  uint32_t f32_bits = fb.u;

  // Extract components
  uint32_t sign = (f32_bits >> 31) & 0x1;
  uint32_t exponent = (f32_bits >> 23) & 0xFF;
  uint32_t mantissa = f32_bits & 0x7FFFFF;

  // Handle special cases
  if (exponent == 0xFF) {
    // NaN or Infinity
    if (mantissa != 0) {
      // NaN - preserve quiet bit
      bits_ = (sign << 15) | 0x7E00;
    } else {
      // Infinity
      bits_ = (sign << 15) | 0x7C00;
    }
    return;
  }

  // Convert exponent from float32 bias (127) to float16 bias (15)
  int32_t unbiased_exp = static_cast<int32_t>(exponent) - 127;
  int32_t f16_exp = unbiased_exp + 15;

  if (f16_exp >= 31) {
    // Overflow - return infinity
    bits_ = (sign << 15) | 0x7C00;
    return;
  }

  if (f16_exp <= 0) {
    // Underflow - handle denormals or return zero
    if (f16_exp < -10) {
      // Too small even for denormal
      bits_ = sign << 15;
      return;
    }

    // Denormal number
    mantissa |= 0x800000; // Add implicit leading 1
    uint32_t shift = 1 - f16_exp;
    mantissa >>= shift;

    // Round to nearest even
    uint32_t guard = mantissa & 0x1FFF;
    mantissa >>= 13;

    if (guard > 0x1000 || (guard == 0x1000 && (mantissa & 1))) {
      mantissa++;
    }

    bits_ = (sign << 15) | (mantissa & 0x3FF);
    return;
  }

  // Normal number
  // Round mantissa from 23 bits to 10 bits
  uint32_t guard = mantissa & 0x1FFF;
  mantissa >>= 13;

  // Round to nearest even
  if (guard > 0x1000 || (guard == 0x1000 && (mantissa & 1))) {
    mantissa++;
    if (mantissa > 0x3FF) {
      // Mantissa overflow
      mantissa = 0;
      f16_exp++;
      if (f16_exp >= 31) {
        // Exponent overflow
        bits_ = (sign << 15) | 0x7C00;
        return;
      }
    }
  }

  bits_ = (sign << 15) | (f16_exp << 10) | (mantissa & 0x3FF);
}

Float16::operator float() const noexcept {
  // Extract components
  uint32_t sign = (bits_ >> 15) & 0x1;
  uint32_t exponent = (bits_ >> 10) & 0x1F;
  uint32_t mantissa = bits_ & 0x3FF;

  // Handle special cases
  if (exponent == 0x1F) {
    // NaN or Infinity
    if (mantissa != 0) {
      // NaN
      FloatBits fb;
      fb.u = (sign << 31) | 0x7FC00000;
      return fb.f;
    } else {
      // Infinity
      FloatBits fb;
      fb.u = (sign << 31) | 0x7F800000;
      return fb.f;
    }
  }

  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      FloatBits fb;
      fb.u = sign << 31;
      return fb.f;
    } else {
      // Denormal - convert to normalized float32
      uint32_t leading_zeros = __builtin_clz(mantissa) - 22;
      mantissa <<= (leading_zeros + 1);
      exponent = 127 - 15 - leading_zeros;
      mantissa = (mantissa & 0x3FF) << 13;

      FloatBits fb;
      fb.u = (sign << 31) | (exponent << 23) | mantissa;
      return fb.f;
    }
  }

  // Normal number
  // Convert exponent from float16 bias (15) to float32 bias (127)
  exponent = exponent - 15 + 127;

  // Expand mantissa from 10 bits to 23 bits
  mantissa <<= 13;

  FloatBits fb;
  fb.u = (sign << 31) | (exponent << 23) | mantissa;
  return fb.f;
}

int8_t Float16::toInt8Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 127.0f)
    return 127;
  if (val <= -128.0f)
    return -128;
  return static_cast<int8_t>(val);
}

int16_t Float16::toInt16Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 32767.0f)
    return 32767;
  if (val <= -32768.0f)
    return -32768;
  return static_cast<int16_t>(val);
}

int32_t Float16::toInt32Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 2147483647.0f)
    return 2147483647;
  if (val <= -2147483648.0f)
    return -2147483648;
  return static_cast<int32_t>(val);
}

int64_t Float16::toInt64Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  // Note: float cannot represent all int64_t values precisely
  // 9223372036854775807LL is INT64_MAX
  if (val >= 9.223372e18f)
    return 9223372036854775807LL;
  if (val <= -9.223372e18f)
    return (-9223372036854775807LL - 1);
  return static_cast<int64_t>(val);
}

//===----------------------------------------------------------------------===//
// BFloat16 Implementation
//===----------------------------------------------------------------------===//

BFloat16::BFloat16(float value) noexcept {
  // BFloat16 is simply the top 16 bits of float32 with rounding
  FloatBits fb;
  fb.f = value;
  uint32_t f32_bits = fb.u;

  // Extract top 16 bits
  uint32_t bf16_bits = f32_bits >> 16;

  // Round to nearest even
  uint32_t rounding_bit = (f32_bits >> 15) & 0x1;
  uint32_t sticky_bits = f32_bits & 0x7FFF;

  if (rounding_bit && (sticky_bits != 0 || (bf16_bits & 0x1))) {
    bf16_bits++;
  }

  // Handle NaN - ensure we don't create infinity from NaN
  if ((bf16_bits & 0x7F80) == 0x7F80 && (f32_bits & 0x7FFFFF) != 0) {
    // Force to quiet NaN
    bf16_bits |= 0x0040;
  }

  bits_ = static_cast<int16_t>(bf16_bits);
}

BFloat16::operator float() const noexcept {
  // BFloat16 to float32 is simple - just shift left by 16
  FloatBits fb;
  fb.u = static_cast<uint32_t>(static_cast<uint16_t>(bits_)) << 16;
  return fb.f;
}

int8_t BFloat16::toInt8Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 127.0f)
    return 127;
  if (val <= -128.0f)
    return -128;
  return static_cast<int8_t>(val);
}

int16_t BFloat16::toInt16Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 32767.0f)
    return 32767;
  if (val <= -32768.0f)
    return -32768;
  return static_cast<int16_t>(val);
}

int32_t BFloat16::toInt32Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  if (val >= 2147483647.0f)
    return 2147483647;
  if (val <= -2147483648.0f)
    return -2147483648;
  return static_cast<int32_t>(val);
}

int64_t BFloat16::toInt64Sat() const noexcept {
  float val = float(*this);
  if (std::isnan(val))
    return 0;
  // Note: float cannot represent all int64_t values precisely
  // 9223372036854775807LL is INT64_MAX
  if (val >= 9.223372e18f)
    return 9223372036854775807LL;
  if (val <= -9.223372e18f)
    return (-9223372036854775807LL - 1);
  return static_cast<int64_t>(val);
}

//===----------------------------------------------------------------------===//
// Int4 Implementation
//===----------------------------------------------------------------------===//

Int4::Int4(int value) noexcept {
  // Saturate to 4-bit range [-8, 7]
  if (value > 7) {
    bits_ = 0x7;
  } else if (value < -8) {
    bits_ = 0x8;
  } else {
    // Store the lower 4 bits
    bits_ = static_cast<int8_t>(value) & 0xF;
  }
}

Int4::Int4(UInt4 val) noexcept {
  // Wrap around using two's complement - just keep lower 4 bits
  bits_ = val.getRawBits() & 0xF;
}

Int4::operator int() const noexcept {
  // Sign extend from 4 bits to full int
  int8_t value = bits_ & 0xF;
  // If the sign bit (bit 3) is set, extend with 1s
  if (value & 0x8) {
    value |= 0xF0; // Set upper 4 bits to 1
  }
  return static_cast<int>(value);
}

//===----------------------------------------------------------------------===//
// UInt4 Implementation
//===----------------------------------------------------------------------===//

UInt4::UInt4(int value) noexcept {
  // Saturate to 4-bit unsigned range [0, 15]
  if (value > 15) {
    bits_ = 0xF;
  } else if (value < 0) {
    bits_ = 0x0;
  } else {
    // Store the lower 4 bits
    bits_ = static_cast<int8_t>(value) & 0xF;
  }
}

UInt4::UInt4(Int4 val) noexcept {
  // Wrap around using two's complement - just keep lower 4 bits
  bits_ = val.getRawBits() & 0xF;
}

UInt4::operator int() const noexcept {
  // No sign extension needed for unsigned
  return static_cast<int>(bits_ & 0xF);
}
