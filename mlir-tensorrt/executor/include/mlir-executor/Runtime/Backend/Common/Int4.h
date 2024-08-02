//===- Int4.h ---------------------------------------------------*- C++ -*-===//
//
// The Int4 datatype logic is adapted from the `ml_dtypes` project
// `https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/int4.h`.
// This project has the Apache License v2.0 license. Check
// https://github.com/jax-ml/ml_dtypes/blob/main/LICENSE for the license
// information.
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// INT4 runtime type declaration.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_TARGET_LUA_INT4_H
#define MLIR_TENSORRT_TARGET_LUA_INT4_H

#include <cstdint>
#include <ostream>
#include <type_traits>

namespace mlirtrt::runtime {

//===----------------------------------------------------------------------===//
// INT4
//===----------------------------------------------------------------------===//

// INT4 base template type that represents both `int4` and `uint4` based on
// `UnderlyingType`.
// Doesn't represent INF or NaN.
// Four data bits are stored in the lower nibble (rhs) of a container of type
// `UnderlyingType` and upper nibble is ignored.
// For example, int4:
// `UnderlyingType` == 'int8_t'
// 1. Integer 2
// int8_t = 0000 0010
// int4   = 0000 0010
// 2. Integer -8
// int8_t = 1111 1000 (2's complement)
// int4   = 0000 1000
// uint4: `UnderlyingType` == 'uint8_t'
// 1. Integer 2
// uint8_t = 0000 0010
// uint4   = 0000 0010
// Constructor assumes that lower nibble of `v` holds the data.
template <typename UnderlyingType>
struct Int4 {
public:
  // Constructors
  explicit constexpr Int4() : i4Value(0) {}
  explicit constexpr Int4(UnderlyingType v) : i4Value(maskUpperNibble(v)) {}
  template <typename T>
  explicit constexpr Int4(T v) : i4Value(static_cast<UnderlyingType>(v)) {}
  constexpr Int4(const Int4 &other) noexcept = default;
  constexpr Int4(Int4 &&other) noexcept = default;
  constexpr Int4 &operator=(const Int4 &other) = default;
  constexpr Int4 &operator=(Int4 &&) = default;

  // Min and Max
  static constexpr Int4 getMinValue() {
    return std::is_signed_v<UnderlyingType> ? Int4(-8) : Int4(0);
  }
  static constexpr Int4 getMaxValue() {
    return std::is_signed_v<UnderlyingType> ? Int4(7) : Int4(15);
  }

  // Typecast overloading
  template <typename T>
  explicit constexpr inline operator T() const {
    return static_cast<T>(toInt());
  }

  // Arithmetic operators overloading
  constexpr Int4 operator-() const { return Int4(-1 * i4Value); }

  constexpr Int4 operator-(const Int4 &other) const {
    return Int4(i4Value - other.i4Value);
  }
  constexpr Int4 &operator--() {
    i4Value = maskUpperNibble(i4Value - 1);
    return *this;
  }
  constexpr Int4 operator--(int) {
    Int4 original = *this;
    this->operator--();
    return original;
  }
  constexpr Int4 &operator-=(const Int4 &other) {
    *this = *this - other;
    return *this;
  }

  constexpr Int4 operator+(const Int4 &other) const {
    return Int4(i4Value + other.i4Value);
  }
  constexpr Int4 &operator++() {
    i4Value = maskUpperNibble(i4Value + 1);
    return *this;
  }
  constexpr Int4 operator++(int) {
    Int4 original = *this;
    this->operator++();
    return original;
  }
  constexpr Int4 &operator+=(const Int4 &other) {
    *this = *this + other;
    return *this;
  }

  constexpr Int4 operator*(const Int4 &other) const {
    return Int4(i4Value * other.i4Value);
  }
  constexpr Int4 &operator*=(const Int4 &other) {
    *this = *this * other;
    return *this;
  }

  constexpr Int4 operator/(const Int4 &other) const {
    return Int4(toInt() / other.toInt());
  }
  constexpr Int4 &operator/=(const Int4 &other) {
    *this = *this / other;
    return *this;
  }

  constexpr Int4 operator%(const Int4 &other) const {
    return Int4(toInt() % other.toInt());
  }

  // Comparison operators overloading
  constexpr bool operator==(const Int4 &other) const {
    return maskUpperNibble(i4Value) == maskUpperNibble(other.i4Value);
  }
  constexpr bool operator!=(const Int4 &other) const {
    return maskUpperNibble(i4Value) != maskUpperNibble(other.i4Value);
  }
  constexpr bool operator<(const Int4 &other) const {
    return toInt() < other.toInt();
  }
  constexpr bool operator>(const Int4 &other) const {
    return toInt() > other.toInt();
  }
  constexpr bool operator<=(const Int4 &other) const {
    return toInt() <= other.toInt();
  }
  constexpr bool operator>=(const Int4 &other) const {
    return toInt() >= other.toInt();
  }

  // Bitwise operators
  constexpr Int4 operator&(const Int4 &other) const {
    return Int4(i4Value & other.i4Value);
  }
  constexpr Int4 operator|(const Int4 &other) const {
    return Int4(i4Value | other.i4Value);
  }
  constexpr Int4 operator^(const Int4 &other) const {
    return Int4(i4Value ^ other.i4Value);
  }
  constexpr Int4 operator~() const { return Int4(~i4Value); }
  template <typename T>
  constexpr Int4 operator>>(const T bits) const {
    int amount = static_cast<int>(bits);
    return Int4(toInt() >> amount);
  }
  constexpr Int4 &operator>>=(const int bits) {
    *this = *this >> bits;
    return *this;
  }
  template <typename T>
  constexpr Int4 operator<<(const T bits) const {
    int amount = static_cast<int>(bits);
    return Int4(toInt() << amount);
  }
  constexpr Int4 &operator<<=(const int bits) {
    *this = *this << bits;
    return *this;
  }

  // ostream overloading
  friend constexpr std::ostream &operator<<(std::ostream &os, const Int4 &v) {
    return os << static_cast<int32_t>(v);
  }

private:
  // Masks upper nibble of `v`.
  static inline constexpr UnderlyingType maskUpperNibble(UnderlyingType v) {
    return v & 0x0F;
  }
  // Returns `int8_t` or `uint8_t` representation of the underlying `Int4`
  // type. Conversion of positive `Int4` is straight forward. However, negative
  // `Int4` is the lower nibble of the original 2's complement `int8_t`. If MSB
  // (Most Significant Bit) of this nibble is set, that means number is
  // negative. For example, -4 is represented as 1111 1100 in int8_t and 1100 in
  // Int4.
  constexpr inline UnderlyingType toInt() const {
    if constexpr (std::is_signed_v<UnderlyingType>) {
      constexpr auto UnderlyingTypeWidth = sizeof(UnderlyingType) * 8;
      return UnderlyingType(i4Value << (UnderlyingTypeWidth - 4)) >>
             (UnderlyingTypeWidth - 4);
    }
    return maskUpperNibble(i4Value);
  }
  // Container
  UnderlyingType i4Value;
  static_assert(
      std::is_same_v<UnderlyingType, uint8_t> ||
          std::is_same_v<UnderlyingType, int8_t>,
      "The `UnderlyingType` must be a signed or unsigned 8-bit integer.");
};

// Signed int4 type with range [-8, 7]
using nv_int4 = Int4<int8_t>;
// Unsigned int4 type with range [0, 15]
using nv_uint4 = Int4<uint8_t>;

//===----------------------------------------------------------------------===//
// INT4Vec2
//===----------------------------------------------------------------------===//

// INT4Vec2 is a packed INT4 base template type that represents both `int4p` and
// `uint4p` based on `UnderlyingType`.
// Packed type holds two `Int4<UnderlyingType>` as the upper and the lower
// nibble of a container of type `UnderlyingType`, under the hood.
// For example,
// 1. Integer 2
// int8_t = 0000 0010
// 2. Integer -4
// int8_t = 1111 1100 (2's complement)
// are packed as two `Int4<int8_t>`s into an underlying container as 0010 1100.
//
// Four bits of the data is assumed to be present into the lower nibble of the
// values passed to the constructor. Upper nibble of packed INT4 represents
// first value passed to the constructor and lower nibble represents the second
// value.
template <typename UnderlyingType>
struct Int4Vec2 {
public:
  // Constructors
  explicit constexpr Int4Vec2() { pack(0, 0); }
  explicit constexpr Int4Vec2(UnderlyingType v1, UnderlyingType v2) {
    pack(v1, v2);
  }
  template <typename T>
  explicit constexpr Int4Vec2(T v1, T v2)
      : Int4Vec2(static_cast<UnderlyingType>(v1),
                 static_cast<UnderlyingType>(v2)) {}
  // Two four bit integers are already packed into `v`.
  explicit constexpr Int4Vec2(UnderlyingType v) : i4pValue(v) {}
  constexpr Int4Vec2(const Int4Vec2 &other) noexcept = default;
  constexpr Int4Vec2(Int4Vec2 &&other) noexcept = default;
  constexpr Int4Vec2 &operator=(const Int4Vec2 &other) = default;
  constexpr Int4Vec2 &operator=(Int4Vec2 &&) = default;

  // Min and Max
  static constexpr Int4Vec2 getMinValue() {
    return std::is_signed_v<UnderlyingType> ? Int4Vec2(-8, -8) : Int4Vec2(0, 0);
  }
  static constexpr Int4Vec2 getMaxValue() {
    return std::is_signed_v<UnderlyingType> ? Int4Vec2(7, 7) : Int4Vec2(15, 15);
  }

  // Arithmetic operators overloading
  constexpr Int4Vec2 operator-() const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    return Int4Vec2(-unpackedThis.first, -unpackedThis.second);
  }
  constexpr Int4Vec2 operator-(const Int4Vec2 &other) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first - unpackedOther.first,
                    unpackedThis.second - unpackedOther.second);
  }
  constexpr Int4Vec2 &operator--() {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    pack(--unpackedThis.first, --unpackedThis.second);
    return *this;
  }
  constexpr Int4Vec2 operator--(int) {
    Int4Vec2 original = *this;
    this->operator--();
    return original;
  }
  constexpr Int4Vec2 &operator-=(const Int4Vec2 &other) {
    *this = *this - other;
    return *this;
  }

  constexpr Int4Vec2 operator+(const Int4Vec2 &other) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first + unpackedOther.first,
                    unpackedThis.second + unpackedOther.second);
  }
  constexpr Int4Vec2 &operator++() {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    pack(++unpackedThis.first, ++unpackedThis.second);
    return *this;
  }
  constexpr Int4Vec2 operator++(int) {
    Int4Vec2 original = *this;
    this->operator++();
    return original;
  }
  constexpr Int4Vec2 &operator+=(const Int4Vec2 &other) {
    *this = *this + other;
    return *this;
  }

  constexpr Int4Vec2 operator*(const Int4Vec2 &other) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first * unpackedOther.first,
                    unpackedThis.second * unpackedOther.second);
  }
  constexpr Int4Vec2 &operator*=(const Int4Vec2 &other) {
    *this = *this * other;
    return *this;
  }

  constexpr Int4Vec2 operator/(const Int4Vec2 &other) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first / unpackedOther.first,
                    unpackedThis.second / unpackedOther.second);
  }
  constexpr Int4Vec2 &operator/=(const Int4Vec2 &other) {
    *this = *this / other;
    return *this;
  }

  constexpr Int4Vec2 operator%(const Int4Vec2 &other) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first % unpackedOther.first,
                    unpackedThis.second % unpackedOther.second);
  }

  // Comparison operators overloading
  constexpr bool operator==(const Int4Vec2 &other) const {
    return i4pValue == other.i4pValue;
  }
  constexpr bool operator!=(const Int4Vec2 &other) const {
    return i4pValue != other.i4pValue;
  }

  // Bitwise operators
  constexpr Int4Vec2 operator&(const Int4Vec2 &other) const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first & unpackedOther.first,
                    unpackedThis.second & unpackedOther.second);
  }
  constexpr Int4Vec2 operator|(const Int4Vec2 &other) const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first | unpackedOther.first,
                    unpackedThis.second | unpackedOther.second);
  }
  constexpr Int4Vec2 operator^(const Int4Vec2 &other) const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedOther =
        unpackI4pToI4(other.i4pValue);
    return Int4Vec2(unpackedThis.first ^ unpackedOther.first,
                    unpackedThis.second ^ unpackedOther.second);
  }
  constexpr Int4Vec2 operator~() const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    return Int4Vec2(~unpackedThis.first, ~unpackedThis.second);
  }
  constexpr Int4Vec2 operator>>(const int bits) const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    return Int4Vec2(unpackedThis.first >> bits, unpackedThis.second >> bits);
  }
  constexpr Int4Vec2 &operator>>=(const int bits) {
    *this = *this >> bits;
    return *this;
  }
  constexpr Int4Vec2 operator<<(const int bits) const {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> unpackedThis =
        unpackI4pToI4(i4pValue);
    return Int4Vec2(unpackedThis.first << bits, unpackedThis.second << bits);
  }
  constexpr Int4Vec2 &operator<<=(const int bits) {
    *this = *this << bits;
    return *this;
  }

  // ostream overloading
  friend std::ostream &operator<<(std::ostream &os, const Int4Vec2 &v) {
    std::pair<UnderlyingType, UnderlyingType> r = toInt(v);
    return os << static_cast<int32_t>(r.first) << ","
              << static_cast<int32_t>(r.second);
  }

private:
  // Packs two `Int4<UnderlyingType>`'s into a single container of type
  // `UnderlyingType`.
  inline constexpr void pack(Int4<UnderlyingType> v1, Int4<UnderlyingType> v2) {
    pack(static_cast<UnderlyingType>(v1), static_cast<UnderlyingType>(v2));
  };
  // Packs two four bit integers represented as lower nibble of `v1` and `v2`
  // into a single container of type `UnderlyingType`.
  inline constexpr void pack(UnderlyingType v1, UnderlyingType v2) {
    // Push upper nibble
    i4pValue = leftShiftByFour(v1) | maskUpperNibble(i4pValue);
    // Push lower nibble
    i4pValue = maskUpperNibble(v2) | maskLowerNibble(i4pValue);
  };
  // Unpacks two packed `Int4<UnderlyingType>`'s into two `UnderlyingType`
  // integers.
  static inline constexpr std::pair<UnderlyingType, UnderlyingType>
  toInt(const Int4Vec2 &v) {
    std::pair<UnderlyingType, UnderlyingType> r;
    UnderlyingType i4pUNMasked = maskUpperNibble(v.i4pValue);
    UnderlyingType i4pLNMaskedAndShifted =
        rightShiftByFour(maskLowerNibble(v.i4pValue));
    if constexpr (std::is_signed_v<UnderlyingType>) {
      constexpr auto UnderlyingTypeWidth = sizeof(UnderlyingType) * 8;
      r.second = UnderlyingType(i4pUNMasked << (UnderlyingTypeWidth - 4)) >>
                 (UnderlyingTypeWidth - 4);
      r.first =
          UnderlyingType(i4pLNMaskedAndShifted << (UnderlyingTypeWidth - 4)) >>
          (UnderlyingTypeWidth - 4);
      return r;
    }
    r.second = i4pUNMasked;
    r.first = i4pLNMaskedAndShifted;
    return r;
  };
  // Unpacks two packed `Int4<UnderlyingType>`'s into two separate
  // `Int4<UnderlyingType>`'s.
  static inline constexpr std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>>
  unpackI4pToI4(UnderlyingType i4pValue) {
    std::pair<Int4<UnderlyingType>, Int4<UnderlyingType>> r;
    r.first = Int4<UnderlyingType>(rightShiftByFour(maskLowerNibble(i4pValue)));
    r.second = Int4<UnderlyingType>(maskUpperNibble(i4pValue));
    return r;
  };
  static inline constexpr UnderlyingType maskLowerNibble(UnderlyingType v) {
    return v & 0xF0;
  }
  static inline constexpr UnderlyingType maskUpperNibble(UnderlyingType v) {
    return v & 0x0F;
  }
  static inline constexpr UnderlyingType rightShiftByFour(UnderlyingType v) {
    return v >> 4;
  }
  static inline constexpr UnderlyingType leftShiftByFour(UnderlyingType v) {
    return v << 4;
  }
  // Container
  UnderlyingType i4pValue;
  static_assert(
      std::is_same_v<UnderlyingType, uint8_t> ||
          std::is_same_v<UnderlyingType, int8_t>,
      "The `UnderlyingType` must be a signed or unsigned 8-bit integer.");
};

using nv_int4p = Int4Vec2<int8_t>;
using nv_uint4p = Int4Vec2<uint8_t>;

} // namespace mlirtrt::runtime

// Specialize some useful methods from std namespace
namespace std {
#define MAKE_SIGNED(UT, ST)                                                    \
  template <>                                                                  \
  struct make_signed<UT> {                                                     \
  public:                                                                      \
    typedef ST type;                                                           \
  }

#define MAKE_UNSIGNED(ST, UT)                                                  \
  template <>                                                                  \
  struct make_unsigned<ST> {                                                   \
  public:                                                                      \
    typedef UT type;                                                           \
  }

MAKE_SIGNED(mlirtrt::runtime::nv_uint4, mlirtrt::runtime::nv_int4);
MAKE_SIGNED(mlirtrt::runtime::nv_uint4p, mlirtrt::runtime::nv_int4p);
MAKE_UNSIGNED(mlirtrt::runtime::nv_int4, mlirtrt::runtime::nv_uint4);
MAKE_UNSIGNED(mlirtrt::runtime::nv_int4p, mlirtrt::runtime::nv_uint4p);
#undef MAKE_SIGNED
#undef MAKE_UNSIGNED
} // namespace std

#endif // MLIR_TENSORRT_TARGET_LUA_INT4_H
