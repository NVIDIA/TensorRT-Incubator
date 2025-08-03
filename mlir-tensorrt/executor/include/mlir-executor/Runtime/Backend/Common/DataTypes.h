//===- DataTypes.h - --------------------------------------------*- C++ -*-===//
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
// This file defines software implementations of low-precision floating point
// types: Float16 (IEEE 754 half-precision), BFloat16 (Brain Floating Point),
// and Float8 (E4M3FN format).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_DATATYPES_H
#define MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_DATATYPES_H

#include <cmath>
#include <cstdint>
#include <iostream>

namespace mlirtrt::runtime {

/// Float8 - E4M3FN format (4 exponent bits, 3 mantissa bits, no infinity)
/// Layout: [sign:1][exponent:4][mantissa:3]
/// Special values: NaN = 0x7F, Max normal = 0x7E
/// Range: ±0.001953125 to ±448
class F8E4M3FN {
public:
  /// Default constructor - initializes to zero
  constexpr F8E4M3FN() noexcept : bits_(0) {}

  /// Constructor from float value
  explicit F8E4M3FN(float value) noexcept;

  /// Constructor from raw bits
  static constexpr F8E4M3FN fromRawBits(int8_t bits) noexcept {
    F8E4M3FN result;
    result.bits_ = bits;
    return result;
  }

  /// Copy constructor
  constexpr F8E4M3FN(const F8E4M3FN &) noexcept = default;

  /// Copy assignment
  F8E4M3FN &operator=(const F8E4M3FN &) noexcept = default;

  /// Convert to float
  operator float() const noexcept;

  /// Get raw bit representation
  constexpr int8_t getRawBits() const noexcept { return bits_; }

  /// Arithmetic operators
  F8E4M3FN operator+(const F8E4M3FN &rhs) const noexcept {
    return F8E4M3FN(float(*this) + float(rhs));
  }

  F8E4M3FN operator-(const F8E4M3FN &rhs) const noexcept {
    return F8E4M3FN(float(*this) - float(rhs));
  }

  F8E4M3FN operator*(const F8E4M3FN &rhs) const noexcept {
    return F8E4M3FN(float(*this) * float(rhs));
  }

  F8E4M3FN operator/(const F8E4M3FN &rhs) const noexcept {
    return F8E4M3FN(float(*this) / float(rhs));
  }

  F8E4M3FN operator-() const noexcept {
    return F8E4M3FN::fromRawBits(bits_ ^ 0x80);
  }

  F8E4M3FN &operator+=(const F8E4M3FN &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }

  F8E4M3FN &operator-=(const F8E4M3FN &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }

  F8E4M3FN &operator*=(const F8E4M3FN &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }

  F8E4M3FN &operator/=(const F8E4M3FN &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  /// Comparison operators
  bool operator==(const F8E4M3FN &rhs) const noexcept {
    // Handle NaN comparisons
    if (isNaN() || rhs.isNaN())
      return false;
    // Handle positive/negative zero
    if (isZero() && rhs.isZero())
      return true;
    return bits_ == rhs.bits_;
  }

  bool operator!=(const F8E4M3FN &rhs) const noexcept {
    return !(*this == rhs);
  }

  bool operator<(const F8E4M3FN &rhs) const noexcept {
    return float(*this) < float(rhs);
  }

  bool operator<=(const F8E4M3FN &rhs) const noexcept {
    return float(*this) <= float(rhs);
  }

  bool operator>(const F8E4M3FN &rhs) const noexcept {
    return float(*this) > float(rhs);
  }

  bool operator>=(const F8E4M3FN &rhs) const noexcept {
    return float(*this) >= float(rhs);
  }

  /// Check if value is NaN
  bool isNaN() const noexcept { return (bits_ & 0x7F) == 0x7F; }

  /// Check if value is zero
  bool isZero() const noexcept { return (bits_ & 0x7F) == 0; }

  /// Check if value is negative
  bool isNegative() const noexcept { return (bits_ & 0x80) != 0; }

  /// Saturating conversions to signed integers
  int8_t toInt8Sat() const noexcept;
  int16_t toInt16Sat() const noexcept;
  int32_t toInt32Sat() const noexcept;
  int64_t toInt64Sat() const noexcept;

  /// Get special values
  static constexpr F8E4M3FN quietNaN() noexcept {
    return F8E4M3FN::fromRawBits(0x7F);
  }

  static constexpr F8E4M3FN epsilon() noexcept {
    return F8E4M3FN::fromRawBits(0x20); // 2^-3 = 0.125
  }

  static constexpr F8E4M3FN min() noexcept {
    return F8E4M3FN::fromRawBits(0x08); // Smallest normal: 2^-6
  }

  static constexpr F8E4M3FN max() noexcept {
    return F8E4M3FN::fromRawBits(0x7E); // (2-2^-3) * 2^8 = 448
  }

private:
  int8_t bits_;
};

/// Float16 - IEEE 754 half-precision floating point format
/// Layout: [sign:1][exponent:5][mantissa:10]
/// Range: ±6.10e-5 to ±65504
class Float16 {
public:
  /// Default constructor - initializes to zero
  constexpr Float16() noexcept : bits_(0) {}

  /// Constructor from float value
  explicit Float16(float value) noexcept;

  /// Constructor from raw bits
  static constexpr Float16 fromRawBits(int16_t bits) noexcept {
    Float16 result;
    result.bits_ = bits;
    return result;
  }

  /// Copy constructor
  constexpr Float16(const Float16 &) noexcept = default;

  /// Copy assignment
  Float16 &operator=(const Float16 &) noexcept = default;

  /// Convert to float
  operator float() const noexcept;

  /// Get raw bit representation
  constexpr int16_t getRawBits() const noexcept { return bits_; }

  /// Arithmetic operators
  Float16 operator+(const Float16 &rhs) const noexcept {
    return Float16(float(*this) + float(rhs));
  }

  Float16 operator-(const Float16 &rhs) const noexcept {
    return Float16(float(*this) - float(rhs));
  }

  Float16 operator*(const Float16 &rhs) const noexcept {
    return Float16(float(*this) * float(rhs));
  }

  Float16 operator/(const Float16 &rhs) const noexcept {
    return Float16(float(*this) / float(rhs));
  }

  Float16 operator-() const noexcept {
    return Float16::fromRawBits(bits_ ^ 0x8000);
  }

  Float16 &operator+=(const Float16 &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }

  Float16 &operator-=(const Float16 &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }

  Float16 &operator*=(const Float16 &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }

  Float16 &operator/=(const Float16 &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  /// Comparison operators
  bool operator==(const Float16 &rhs) const noexcept {
    // Handle NaN comparisons
    if (isNaN() || rhs.isNaN())
      return false;
    // Handle positive/negative zero
    if (isZero() && rhs.isZero())
      return true;
    return bits_ == rhs.bits_;
  }

  bool operator!=(const Float16 &rhs) const noexcept { return !(*this == rhs); }

  bool operator<(const Float16 &rhs) const noexcept {
    return float(*this) < float(rhs);
  }

  bool operator<=(const Float16 &rhs) const noexcept {
    return float(*this) <= float(rhs);
  }

  bool operator>(const Float16 &rhs) const noexcept {
    return float(*this) > float(rhs);
  }

  bool operator>=(const Float16 &rhs) const noexcept {
    return float(*this) >= float(rhs);
  }

  /// Check if value is NaN
  bool isNaN() const noexcept {
    return (bits_ & 0x7C00) == 0x7C00 && (bits_ & 0x03FF) != 0;
  }

  /// Check if value is infinity
  bool isInf() const noexcept { return (bits_ & 0x7FFF) == 0x7C00; }

  /// Check if value is zero
  bool isZero() const noexcept { return (bits_ & 0x7FFF) == 0; }

  /// Check if value is negative
  bool isNegative() const noexcept { return (bits_ & 0x8000) != 0; }

  /// Saturating conversions to signed integers
  int8_t toInt8Sat() const noexcept;
  int16_t toInt16Sat() const noexcept;
  int32_t toInt32Sat() const noexcept;
  int64_t toInt64Sat() const noexcept;

  /// Get special values
  static constexpr Float16 infinity() noexcept {
    return Float16::fromRawBits(0x7C00);
  }

  static constexpr Float16 negInfinity() noexcept {
    return Float16::fromRawBits(0xFC00);
  }

  static constexpr Float16 quietNaN() noexcept {
    return Float16::fromRawBits(0x7E00);
  }

  static constexpr Float16 epsilon() noexcept {
    return Float16::fromRawBits(0x1400); // 2^-10
  }

  static constexpr Float16 min() noexcept {
    return Float16::fromRawBits(0x0400); // 2^-14
  }

  static constexpr Float16 max() noexcept {
    return Float16::fromRawBits(0x7BFF); // (2-2^-10) * 2^15
  }

private:
  int16_t bits_;
};

/// BFloat16 - Brain Floating Point format
/// Layout: [sign:1][exponent:8][mantissa:7]
/// Range: Same as float32 but with reduced precision
class BFloat16 {
public:
  /// Default constructor - initializes to zero
  constexpr BFloat16() noexcept : bits_(0) {}

  /// Constructor from float value
  explicit BFloat16(float value) noexcept;

  /// Constructor from raw bits
  static constexpr BFloat16 fromRawBits(int16_t bits) noexcept {
    BFloat16 result;
    result.bits_ = bits;
    return result;
  }

  /// Copy constructor
  constexpr BFloat16(const BFloat16 &) noexcept = default;

  /// Copy assignment
  BFloat16 &operator=(const BFloat16 &) noexcept = default;

  /// Convert to float
  operator float() const noexcept;

  /// Get raw bit representation
  constexpr int16_t getRawBits() const noexcept { return bits_; }

  /// Arithmetic operators
  BFloat16 operator+(const BFloat16 &rhs) const noexcept {
    return BFloat16(float(*this) + float(rhs));
  }

  BFloat16 operator-(const BFloat16 &rhs) const noexcept {
    return BFloat16(float(*this) - float(rhs));
  }

  BFloat16 operator*(const BFloat16 &rhs) const noexcept {
    return BFloat16(float(*this) * float(rhs));
  }

  BFloat16 operator/(const BFloat16 &rhs) const noexcept {
    return BFloat16(float(*this) / float(rhs));
  }

  BFloat16 operator-() const noexcept {
    return BFloat16::fromRawBits(bits_ ^ 0x8000);
  }

  BFloat16 &operator+=(const BFloat16 &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }

  BFloat16 &operator-=(const BFloat16 &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }

  BFloat16 &operator*=(const BFloat16 &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }

  BFloat16 &operator/=(const BFloat16 &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  /// Comparison operators
  bool operator==(const BFloat16 &rhs) const noexcept {
    // Handle NaN comparisons
    if (isNaN() || rhs.isNaN())
      return false;
    // Handle positive/negative zero
    if (isZero() && rhs.isZero())
      return true;
    return bits_ == rhs.bits_;
  }

  bool operator!=(const BFloat16 &rhs) const noexcept {
    return !(*this == rhs);
  }

  bool operator<(const BFloat16 &rhs) const noexcept {
    return float(*this) < float(rhs);
  }

  bool operator<=(const BFloat16 &rhs) const noexcept {
    return float(*this) <= float(rhs);
  }

  bool operator>(const BFloat16 &rhs) const noexcept {
    return float(*this) > float(rhs);
  }

  bool operator>=(const BFloat16 &rhs) const noexcept {
    return float(*this) >= float(rhs);
  }

  /// Check if value is NaN
  bool isNaN() const noexcept {
    return (bits_ & 0x7F80) == 0x7F80 && (bits_ & 0x007F) != 0;
  }

  /// Check if value is infinity
  bool isInf() const noexcept { return (bits_ & 0x7FFF) == 0x7F80; }

  /// Check if value is zero
  bool isZero() const noexcept { return (bits_ & 0x7FFF) == 0; }

  /// Check if value is negative
  bool isNegative() const noexcept { return (bits_ & 0x8000) != 0; }

  /// Saturating conversions to signed integers
  int8_t toInt8Sat() const noexcept;
  int16_t toInt16Sat() const noexcept;
  int32_t toInt32Sat() const noexcept;
  int64_t toInt64Sat() const noexcept;

  /// Get special values
  static constexpr BFloat16 infinity() noexcept {
    return BFloat16::fromRawBits(0x7F80);
  }

  static constexpr BFloat16 negInfinity() noexcept {
    return BFloat16::fromRawBits(0xFF80);
  }

  static constexpr BFloat16 quietNaN() noexcept {
    return BFloat16::fromRawBits(0x7FC0);
  }

  static constexpr BFloat16 epsilon() noexcept {
    return BFloat16::fromRawBits(0x3C00); // 2^-7
  }

  static constexpr BFloat16 min() noexcept {
    return BFloat16::fromRawBits(0x0080); // 2^-126
  }

  static constexpr BFloat16 max() noexcept {
    return BFloat16::fromRawBits(0x7F7F); // (2-2^-7) * 2^127
  }

private:
  int16_t bits_;
};

class UInt4;

/// Int4 - 4-bit signed integer type
/// Range: -8to 7
/// Stored in the lower 4 bits of an int8_t
/// Constructor saturates, arithmetic operations use wrap-around
class Int4 {
public:
  friend class UInt4;

  /// Default constructor - initializes to zero
  constexpr Int4() noexcept : bits_(0) {}

  /// Constructor from integer value with saturation
  explicit Int4(int value) noexcept;

  /// Bitcast constructor from UInt4.
  explicit Int4(UInt4 val) noexcept;

  /// Constructor from raw bits
  static constexpr Int4 fromRawBits(int8_t bits) noexcept {
    Int4 result;
    result.bits_ = bits & 0xF; // Ensure only lower 4 bits are used
    return result;
  }

  /// Copy constructor
  constexpr Int4(const Int4 &) noexcept = default;

  /// Copy assignment
  Int4 &operator=(const Int4 &) noexcept = default;

  /// Convert to int
  operator int() const noexcept;

  /// Get raw bit representation (lower 4 bits)
  constexpr int8_t getRawBits() const noexcept { return bits_ & 0xF; }

  /// Arithmetic operators with wrap-around
  Int4 operator+(const Int4 &rhs) const noexcept {
    // Use fromRawBits to get wrap-around behavior
    int result = int(*this) + int(rhs);
    return Int4::fromRawBits(result & 0xF);
  }

  Int4 operator-(const Int4 &rhs) const noexcept {
    // Use fromRawBits to get wrap-around behavior
    int result = int(*this) - int(rhs);
    return Int4::fromRawBits(result & 0xF);
  }

  Int4 operator*(const Int4 &rhs) const noexcept {
    // Use fromRawBits to get wrap-around behavior
    int result = int(*this) * int(rhs);
    return Int4::fromRawBits(result & 0xF);
  }

  Int4 operator/(const Int4 &rhs) const noexcept {
    if (int(rhs) == 0)
      return Int4(0); // Avoid division by zero
    // Division doesn't need special handling for wrap-around
    return Int4(int(*this) / int(rhs));
  }

  Int4 operator%(const Int4 &rhs) const noexcept {
    if (int(rhs) == 0)
      return Int4(0); // Avoid division by zero
    // Modulo doesn't need special handling for wrap-around
    return Int4(int(*this) % int(rhs));
  }

  Int4 operator-() const noexcept {
    // Use fromRawBits to get wrap-around behavior
    int result = -int(*this);
    return Int4::fromRawBits(result & 0xF);
  }

  Int4 &operator+=(const Int4 &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }

  Int4 &operator-=(const Int4 &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }

  Int4 &operator*=(const Int4 &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }

  Int4 &operator/=(const Int4 &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  Int4 &operator%=(const Int4 &rhs) noexcept {
    *this = *this % rhs;
    return *this;
  }

  /// Bitwise operators
  Int4 operator&(const Int4 &rhs) const noexcept {
    return Int4::fromRawBits(bits_ & rhs.bits_);
  }

  Int4 operator|(const Int4 &rhs) const noexcept {
    return Int4::fromRawBits(bits_ | rhs.bits_);
  }

  Int4 operator^(const Int4 &rhs) const noexcept {
    return Int4::fromRawBits(bits_ ^ rhs.bits_);
  }

  Int4 operator~() const noexcept { return Int4::fromRawBits(~bits_); }

  Int4 operator<<(int shift) const noexcept {
    if (shift >= 4)
      return Int4(0);
    // Use fromRawBits to get wrap-around behavior
    int result = int(*this) << shift;
    return Int4::fromRawBits(result & 0xF);
  }

  Int4 operator>>(int shift) const noexcept {
    if (shift >= 4)
      return Int4(int(*this) < 0 ? -1 : 0);
    // Right shift preserves sign, no special handling needed
    return Int4(int(*this) >> shift);
  }

  /// Comparison operators
  bool operator==(const Int4 &rhs) const noexcept {
    return int(*this) == int(rhs);
  }

  bool operator!=(const Int4 &rhs) const noexcept { return !(*this == rhs); }

  bool operator<(const Int4 &rhs) const noexcept {
    return int(*this) < int(rhs);
  }

  bool operator<=(const Int4 &rhs) const noexcept {
    return int(*this) <= int(rhs);
  }

  bool operator>(const Int4 &rhs) const noexcept {
    return int(*this) > int(rhs);
  }

  bool operator>=(const Int4 &rhs) const noexcept {
    return int(*this) >= int(rhs);
  }

  /// Check if value is negative
  bool isNegative() const noexcept { return (bits_ & 0x8) != 0; }

  /// Get special values
  static constexpr Int4 min() noexcept {
    return Int4::fromRawBits(0x8); // -8
  }

  static constexpr Int4 max() noexcept {
    return Int4::fromRawBits(0x7); // 7
  }

  static constexpr Int4 zero() noexcept { return Int4::fromRawBits(0x0); }

  /// Bitcast to UInt4 - reinterprets the bits
  inline class UInt4 toUInt4() const noexcept;

private:
  int8_t bits_; // Only lower 4 bits are used
};

/// Uint4 - 4-bit unsigned integer type
/// Range: 0 to 15
/// Stored in the lower 4 bits of an int8_t
class UInt4 {
public:
  /// Default constructor - initializes to zero
  constexpr UInt4() noexcept : bits_(0) {}

  /// Constructor from integer value with saturation
  explicit UInt4(int value) noexcept;

  /// Bitcast constructor from Int4.
  explicit UInt4(Int4 val) noexcept;

  /// Constructor from raw bits
  static constexpr UInt4 fromRawBits(int8_t bits) noexcept {
    UInt4 result;
    result.bits_ = bits & 0xF; // Ensure only lower 4 bits are used
    return result;
  }

  /// Copy constructor
  constexpr UInt4(const UInt4 &) noexcept = default;

  /// Copy assignment
  UInt4 &operator=(const UInt4 &) noexcept = default;

  /// Convert to int
  operator int() const noexcept;

  /// Get raw bit representation (lower 4 bits)
  constexpr int8_t getRawBits() const noexcept { return bits_ & 0xF; }

  /// Arithmetic operators with saturation
  UInt4 operator+(const UInt4 &rhs) const noexcept {
    return UInt4(int(*this) + int(rhs));
  }

  UInt4 operator-(const UInt4 &rhs) const noexcept {
    return UInt4(int(*this) - int(rhs));
  }

  UInt4 operator*(const UInt4 &rhs) const noexcept {
    return UInt4(int(*this) * int(rhs));
  }

  UInt4 operator/(const UInt4 &rhs) const noexcept {
    if (int(rhs) == 0)
      return UInt4(0); // Avoid division by zero
    return UInt4(int(*this) / int(rhs));
  }

  UInt4 operator%(const UInt4 &rhs) const noexcept {
    if (int(rhs) == 0)
      return UInt4(0); // Avoid division by zero
    return UInt4(int(*this) % int(rhs));
  }

  UInt4 &operator+=(const UInt4 &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }

  UInt4 &operator-=(const UInt4 &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }

  UInt4 &operator*=(const UInt4 &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }

  UInt4 &operator/=(const UInt4 &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  UInt4 &operator%=(const UInt4 &rhs) noexcept {
    *this = *this % rhs;
    return *this;
  }

  /// Bitwise operators
  UInt4 operator&(const UInt4 &rhs) const noexcept {
    return UInt4::fromRawBits(bits_ & rhs.bits_);
  }

  UInt4 operator|(const UInt4 &rhs) const noexcept {
    return UInt4::fromRawBits(bits_ | rhs.bits_);
  }

  UInt4 operator^(const UInt4 &rhs) const noexcept {
    return UInt4::fromRawBits(bits_ ^ rhs.bits_);
  }

  UInt4 operator~() const noexcept { return UInt4::fromRawBits(~bits_); }

  UInt4 operator<<(int shift) const noexcept {
    if (shift >= 4)
      return UInt4(0);
    return UInt4(int(*this) << shift);
  }

  UInt4 operator>>(int shift) const noexcept {
    if (shift >= 4)
      return UInt4(0);
    return UInt4::fromRawBits(bits_ >> shift); // Logical shift for unsigned
  }

  /// Comparison operators
  bool operator==(const UInt4 &rhs) const noexcept {
    return getRawBits() == rhs.getRawBits();
  }

  bool operator!=(const UInt4 &rhs) const noexcept { return !(*this == rhs); }

  bool operator<(const UInt4 &rhs) const noexcept {
    return int(*this) < int(rhs);
  }

  bool operator<=(const UInt4 &rhs) const noexcept {
    return int(*this) <= int(rhs);
  }

  bool operator>(const UInt4 &rhs) const noexcept {
    return int(*this) > int(rhs);
  }

  bool operator>=(const UInt4 &rhs) const noexcept {
    return int(*this) >= int(rhs);
  }

  /// Get special values
  static constexpr UInt4 min() noexcept {
    return UInt4::fromRawBits(0x0); // 0
  }

  static constexpr UInt4 max() noexcept {
    return UInt4::fromRawBits(0xF); // 15
  }

  static constexpr UInt4 zero() noexcept { return UInt4::fromRawBits(0x0); }

  /// Bitcast to Int4 - reinterprets the bits
  inline Int4 toInt4() const noexcept;

private:
  int8_t bits_; // Only lower 4 bits are used

  friend class Int4;
};

// Implementation of bitcast conversions
inline UInt4 Int4::toUInt4() const noexcept {
  return UInt4::fromRawBits(bits_);
}

inline Int4 UInt4::toInt4() const noexcept { return Int4::fromRawBits(bits_); }

/// Stream operators
inline std::ostream &operator<<(std::ostream &os, const F8E4M3FN &value) {
  return os << float(value);
}

inline std::ostream &operator<<(std::ostream &os, const Float16 &value) {
  return os << float(value);
}

inline std::ostream &operator<<(std::ostream &os, const BFloat16 &value) {
  return os << float(value);
}

inline std::ostream &operator<<(std::ostream &os, const Int4 &value) {
  return os << int(value);
}

inline std::ostream &operator<<(std::ostream &os, const UInt4 &value) {
  return os << int(value);
}

} // namespace mlirtrt::runtime

#endif // MLIR_EXECUTOR_RUNTIME_BACKEND_COMMON_DATATYPES_H