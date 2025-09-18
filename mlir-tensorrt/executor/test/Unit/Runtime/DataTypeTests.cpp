
//===- DataTypeTests.cpp - Unit tests for low-precision float types ------===//
//
// Unit tests for Float8, Float16 and BFloat16 implementations.
//
//===----------------------------------------------------------------------===//

#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>

using namespace mtrt;

namespace {

// Helper function to check if two floats are approximately equal
bool approxEqual(float a, float b, float tolerance = 1e-5f) {
  if (std::isnan(a) && std::isnan(b))
    return true;
  if (std::isinf(a) && std::isinf(b))
    return (a > 0) == (b > 0);
  return std::abs(a - b) < tolerance;
}

//===----------------------------------------------------------------------===//
// Float8 E4M3FN Tests
//===----------------------------------------------------------------------===//

TEST(Float8Test, DefaultConstruction) {
  F8E4M3FN f;
  EXPECT_EQ(f.getRawBits(), 0);
  EXPECT_EQ(float(f), 0.0f);
}

TEST(Float8Test, FloatConversion) {
  // Test basic values
  EXPECT_EQ(float(F8E4M3FN(0.0f)), 0.0f);
  EXPECT_EQ(float(F8E4M3FN(1.0f)), 1.0f);
  EXPECT_EQ(float(F8E4M3FN(-1.0f)), -1.0f);

  // Test powers of 2
  EXPECT_EQ(float(F8E4M3FN(2.0f)), 2.0f);
  EXPECT_EQ(float(F8E4M3FN(0.5f)), 0.5f);
  EXPECT_EQ(float(F8E4M3FN(0.25f)), 0.25f);

  // Test values that require rounding (less precision than Float16)
  EXPECT_TRUE(approxEqual(float(F8E4M3FN(3.14159f)), 3.14159f, 0.2f));
  EXPECT_TRUE(approxEqual(float(F8E4M3FN(1.23456f)), 1.23456f, 0.1f));
}

TEST(Float8Test, SpecialValues) {
  // Test NaN - E4M3FN uses 0x7F for NaN
  F8E4M3FN nan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(nan.isNaN());
  EXPECT_TRUE(std::isnan(float(nan)));
  EXPECT_EQ(nan.getRawBits(), 0x7F);

  // Test that infinity gets clamped to max value (E4M3FN cannot represent
  // infinity)
  F8E4M3FN pos_inf(std::numeric_limits<float>::infinity());
  EXPECT_FALSE(pos_inf.isNaN());
  EXPECT_FALSE(std::isinf(float(pos_inf)));
  EXPECT_EQ(pos_inf.getRawBits(), 0x7E); // Max positive value

  F8E4M3FN neg_inf(-std::numeric_limits<float>::infinity());
  EXPECT_FALSE(neg_inf.isNaN());
  EXPECT_FALSE(std::isinf(float(neg_inf)));
  EXPECT_EQ(neg_inf.getRawBits(),
            static_cast<int8_t>(0xFE)); // Max negative value

  // Test zero
  F8E4M3FN zero(0.0f);
  EXPECT_TRUE(zero.isZero());
  EXPECT_FALSE(zero.isNegative());

  F8E4M3FN neg_zero(-0.0f);
  EXPECT_TRUE(neg_zero.isZero());
  EXPECT_TRUE(neg_zero.isNegative());
}

TEST(Float8Test, RangeAndPrecision) {
  // Test max value (0x7E = 448)
  F8E4M3FN max_val = F8E4M3FN::max();
  EXPECT_EQ(max_val.getRawBits(), 0x7E);
  EXPECT_TRUE(approxEqual(float(max_val), 448.0f, 1.0f));

  // Test min normalized value (2^-6 = 0.015625)
  F8E4M3FN min_val = F8E4M3FN::min();
  EXPECT_EQ(min_val.getRawBits(), 0x08);
  EXPECT_EQ(float(min_val), std::exp2(-6));

  // Test epsilon (2^-3 = 0.125)
  F8E4M3FN eps = F8E4M3FN::epsilon();
  EXPECT_EQ(eps.getRawBits(), 0x20);
  EXPECT_EQ(float(eps), std::exp2(-3));

  // Test denormal values
  F8E4M3FN denorm(0.001f);
  EXPECT_TRUE(float(denorm) > 0);
  EXPECT_TRUE(float(denorm) < float(F8E4M3FN::min()));
}

TEST(Float8Test, ArithmeticOperations) {
  F8E4M3FN a(2.0f);
  F8E4M3FN b(1.5f);

  // Addition
  F8E4M3FN sum = a + b;
  EXPECT_TRUE(approxEqual(float(sum), 3.5f, 0.5f));

  // Subtraction
  F8E4M3FN diff = a - b;
  EXPECT_TRUE(approxEqual(float(diff), 0.5f, 0.5f));

  // Multiplication
  F8E4M3FN prod = a * b;
  EXPECT_TRUE(approxEqual(float(prod), 3.0f, 0.5f));

  // Division
  F8E4M3FN quot = a / b;
  EXPECT_TRUE(approxEqual(float(quot), 1.33333f, 0.5f));

  // Negation
  F8E4M3FN neg = -a;
  EXPECT_TRUE(approxEqual(float(neg), -2.0f, 0.1f));

  // Compound assignment
  F8E4M3FN c(1.0f);
  c += a;
  EXPECT_TRUE(approxEqual(float(c), 3.0f, 0.5f));

  c -= b;
  EXPECT_TRUE(approxEqual(float(c), 1.5f, 0.5f));

  c *= F8E4M3FN(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 3.0f, 0.5f));

  c /= F8E4M3FN(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 1.5f, 0.5f));
}

TEST(Float8Test, ComparisonOperators) {
  F8E4M3FN a(1.0f);
  F8E4M3FN b(2.0f);
  F8E4M3FN c(1.0f);

  // Equality
  EXPECT_TRUE(a == c);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != c);

  // Ordering
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(b <= a);

  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= b);

  // NaN comparisons
  F8E4M3FN nan = F8E4M3FN::quietNaN();
  EXPECT_FALSE(nan == nan);
  EXPECT_TRUE(nan != nan);
  EXPECT_FALSE(nan < a);
  EXPECT_FALSE(a < nan);

  // Zero comparisons
  F8E4M3FN pos_zero(0.0f);
  F8E4M3FN neg_zero(-0.0f);
  EXPECT_TRUE(pos_zero == neg_zero);
}

TEST(Float8Test, Overflow) {
  // Test overflow to max value (E4M3FN cannot represent infinity)
  F8E4M3FN large(1000.0f);
  EXPECT_FALSE(large.isNaN());
  EXPECT_EQ(large.getRawBits(), 0x7E); // Clamped to max

  // Test arithmetic overflow
  F8E4M3FN big = F8E4M3FN::max();
  F8E4M3FN result = big + big;
  EXPECT_EQ(result.getRawBits(), 0x7E); // Clamped to max
}

TEST(Float8Test, Underflow) {
  // Test underflow to zero
  F8E4M3FN tiny(1e-10f);
  EXPECT_TRUE(tiny.isZero());

  // Test gradual underflow (denormals)
  F8E4M3FN small(0.005f);
  EXPECT_FALSE(small.isZero());
  EXPECT_TRUE(float(small) > 0);
}

TEST(Float8Test, StreamOperator) {
  F8E4M3FN val(3.0f);
  std::stringstream ss;
  ss << val;
  float parsed;
  ss >> parsed;
  EXPECT_TRUE(approxEqual(parsed, float(val), 0.5f));
}

TEST(Float8Test, SaturatingIntegerConversions) {
  // Test normal conversions - Float8 has limited precision, so actual values
  // may differ Float8 with 3 mantissa bits cannot represent 10.7 exactly
  F8E4M3FN val_pos(10.7f);
  F8E4M3FN val_neg(-10.7f);
  int8_t expected_pos = static_cast<int8_t>(float(val_pos));
  int8_t expected_neg = static_cast<int8_t>(float(val_neg));

  EXPECT_EQ(val_pos.toInt8Sat(), expected_pos);
  EXPECT_EQ(val_neg.toInt8Sat(), expected_neg);
  EXPECT_EQ(val_pos.toInt16Sat(), static_cast<int16_t>(float(val_pos)));
  EXPECT_EQ(val_neg.toInt16Sat(), static_cast<int16_t>(float(val_neg)));
  EXPECT_EQ(val_pos.toInt32Sat(), static_cast<int32_t>(float(val_pos)));
  EXPECT_EQ(val_neg.toInt32Sat(), static_cast<int32_t>(float(val_neg)));
  EXPECT_EQ(val_pos.toInt64Sat(), static_cast<int64_t>(float(val_pos)));
  EXPECT_EQ(val_neg.toInt64Sat(), static_cast<int64_t>(float(val_neg)));

  // Test saturation to int8_t
  EXPECT_EQ(F8E4M3FN(200.0f).toInt8Sat(), 127);
  EXPECT_EQ(F8E4M3FN(-200.0f).toInt8Sat(), -128);
  EXPECT_EQ(F8E4M3FN::max().toInt8Sat(), 127);
  EXPECT_EQ((-F8E4M3FN::max()).toInt8Sat(), -128);

  // Test saturation to int16_t
  EXPECT_EQ(F8E4M3FN::max().toInt16Sat(), 448); // Float8 max is 448
  EXPECT_EQ((-F8E4M3FN::max()).toInt16Sat(), -448);

  // Test NaN conversion
  EXPECT_EQ(F8E4M3FN::quietNaN().toInt8Sat(), 0);
  EXPECT_EQ(F8E4M3FN::quietNaN().toInt16Sat(), 0);
  EXPECT_EQ(F8E4M3FN::quietNaN().toInt32Sat(), 0);
  EXPECT_EQ(F8E4M3FN::quietNaN().toInt64Sat(), 0);

  // Test zero
  EXPECT_EQ(F8E4M3FN(0.0f).toInt8Sat(), 0);
  EXPECT_EQ(F8E4M3FN(-0.0f).toInt8Sat(), 0);
}

//===----------------------------------------------------------------------===//
// Int4 Tests
//===----------------------------------------------------------------------===//

TEST(Int4Test, DefaultConstruction) {
  Int4 i;
  EXPECT_EQ(i.getRawBits(), 0);
  EXPECT_EQ(int(i), 0);
}

TEST(Int4Test, IntegerConstruction) {
  // Test construction within range
  EXPECT_EQ(int(Int4(0)), 0);
  EXPECT_EQ(int(Int4(7)), 7);
  EXPECT_EQ(int(Int4(-8)), -8);
  EXPECT_EQ(int(Int4(3)), 3);
  EXPECT_EQ(int(Int4(-3)), -3);

  // Test saturation on overflow
  EXPECT_EQ(int(Int4(8)), 7);     // Saturate to max
  EXPECT_EQ(int(Int4(100)), 7);   // Saturate to max
  EXPECT_EQ(int(Int4(-9)), -8);   // Saturate to min
  EXPECT_EQ(int(Int4(-100)), -8); // Saturate to min

  // Test wrap-around with fromRawBits
  EXPECT_EQ(int(Int4::fromRawBits(8)),
            -8); // 0x8 = -8 in 4-bit two's complement
  EXPECT_EQ(int(Int4::fromRawBits(9)),
            -7); // 0x9 = -7 in 4-bit two's complement
  EXPECT_EQ(int(Int4::fromRawBits(15)),
            -1); // 0xF = -1 in 4-bit two's complement
}

TEST(Int4Test, RawBitsConstruction) {
  // Test fromRawBits
  for (int i = 0; i < 16; ++i) {
    Int4 val = Int4::fromRawBits(i);
    EXPECT_EQ(val.getRawBits(), i);

    // Check sign extension
    int expected = (i < 8) ? i : (i - 16);
    EXPECT_EQ(int(val), expected);
  }
}

TEST(Int4Test, ArithmeticOperations) {
  Int4 a(3);
  Int4 b(2);
  Int4 c(-3);

  // Addition
  EXPECT_EQ(int(a + b), 5);
  EXPECT_EQ(int(a + c), 0);
  EXPECT_EQ(int(Int4(4) + Int4(4)), -8);  // 4 + 4 = 8 = -8 (wrap-around)
  EXPECT_EQ(int(Int4(6) + Int4(5)), -5);  // 6 + 5 = 11 = -5 (wrap-around)
  EXPECT_EQ(int(Int4(-6) + Int4(-5)), 5); // -6 + -5 = -11 = 5 (wrap-around)

  // Subtraction
  EXPECT_EQ(int(a - b), 1);
  EXPECT_EQ(int(b - a), -1);
  EXPECT_EQ(int(a - c), 6);
  EXPECT_EQ(int(Int4(7) - Int4(-5)), -4); // 7 - (-5) = 12 = -4 (wrap-around)
  EXPECT_EQ(int(Int4(-7) - Int4(5)), 4);  // -7 - 5 = -12 = 4 (wrap-around)

  // Multiplication
  EXPECT_EQ(int(a * b), 6);
  EXPECT_EQ(int(a * c), 7);              // 3 * -3 = -9 = 7 (wrap-around)
  EXPECT_EQ(int(Int4(4) * Int4(3)), -4); // 4 * 3 = 12 = -4 (wrap-around)
  EXPECT_EQ(int(Int4(7) * Int4(2)), -2); // 7 * 2 = 14 = -2 (wrap-around)

  // Division
  EXPECT_EQ(int(Int4(6) / Int4(2)), 3);
  EXPECT_EQ(int(Int4(7) / Int4(2)), 3);
  EXPECT_EQ(int(Int4(-6) / Int4(2)), -3);
  EXPECT_EQ(int(Int4(6) / Int4(0)), 0); // Division by zero returns 0

  // Modulo
  EXPECT_EQ(int(Int4(7) % Int4(3)), 1);
  EXPECT_EQ(int(Int4(-7) % Int4(3)), -1);
  EXPECT_EQ(int(Int4(6) % Int4(0)), 0); // Modulo by zero returns 0

  // Negation
  EXPECT_EQ(int(-a), -3);
  EXPECT_EQ(int(-c), 3);
  EXPECT_EQ(int(-Int4(-8)), -8); // -(-8) = 8 = -8 (wrap-around)
  EXPECT_EQ(int(-Int4(4)), -4);  // -(4) = -4

  // Compound assignment
  Int4 d(2);
  d += Int4(3);
  EXPECT_EQ(int(d), 5);

  d -= Int4(1);
  EXPECT_EQ(int(d), 4);

  d *= Int4(2);
  EXPECT_EQ(int(d), -8); // 4 * 2 = 8 = -8 (wrap-around)

  d /= Int4(2);
  EXPECT_EQ(int(d), -4); // -8 / 2 = -4

  d += Int4(5);
  EXPECT_EQ(int(d), 1); // -4 + 5 = 1

  d %= Int4(2);
  EXPECT_EQ(int(d), 1);
}

TEST(Int4Test, BitwiseOperations) {
  Int4 a = Int4::fromRawBits(0b0101); // 5
  Int4 b = Int4::fromRawBits(0b0011); // 3

  // AND
  EXPECT_EQ((a & b).getRawBits(), 0b0001);

  // OR
  EXPECT_EQ((a | b).getRawBits(), 0b0111);

  // XOR
  EXPECT_EQ((a ^ b).getRawBits(), 0b0110);

  // NOT
  EXPECT_EQ((~a).getRawBits(), 0b1010);
  EXPECT_EQ(int(~a), -6); // Sign extended

  // Shift left
  EXPECT_EQ(int(Int4(2) << 1), 4);
  EXPECT_EQ(int(Int4(3) << 1), 6);
  EXPECT_EQ(int(Int4(5) << 1), -6); // 5 << 1 = 10 = -6 (wrap-around)
  EXPECT_EQ(int(Int4(2) << 2), -8); // 2 << 2 = 8 = -8 (wrap-around)
  EXPECT_EQ(int(Int4(2) << 4), 0);  // Shift >= 4 returns 0
  EXPECT_EQ(int(Int4(7) << 1), -2); // 7 << 1 = 14 = -2 (wrap-around)

  // Shift right
  EXPECT_EQ(int(Int4(4) >> 1), 2);
  EXPECT_EQ(int(Int4(-4) >> 1), -2);
  EXPECT_EQ(int(Int4(1) >> 1), 0);
  EXPECT_EQ(int(Int4(-1) >> 1), -1); // Arithmetic shift
  EXPECT_EQ(int(Int4(5) >> 4), 0);
  EXPECT_EQ(int(Int4(-5) >> 4), -1); // Arithmetic shift
}

TEST(Int4Test, ComparisonOperators) {
  Int4 a(3);
  Int4 b(3);
  Int4 c(5);
  Int4 d(-3);

  // Equality
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a == d);

  // Inequality
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_TRUE(a != d);

  // Less than
  EXPECT_FALSE(a < b);
  EXPECT_TRUE(a < c);
  EXPECT_FALSE(a < d);
  EXPECT_TRUE(d < a);

  // Less than or equal
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(a <= d);

  // Greater than
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_TRUE(a > d);

  // Greater than or equal
  EXPECT_TRUE(a >= b);
  EXPECT_FALSE(a >= c);
  EXPECT_TRUE(a >= d);
}

TEST(Int4Test, SpecialValues) {
  // Test min and max
  EXPECT_EQ(int(Int4::min()), -8);
  EXPECT_EQ(int(Int4::max()), 7);
  EXPECT_EQ(int(Int4::zero()), 0);

  // Test isNegative
  EXPECT_TRUE(Int4(-1).isNegative());
  EXPECT_TRUE(Int4(-8).isNegative());
  EXPECT_FALSE(Int4(0).isNegative());
  EXPECT_FALSE(Int4(7).isNegative());
}

TEST(Int4Test, StreamOperator) {
  Int4 val(5);
  std::stringstream ss;
  ss << val;
  int parsed;
  ss >> parsed;
  EXPECT_EQ(parsed, 5);

  // Test negative value
  Int4 neg(-3);
  std::stringstream ss2;
  ss2 << neg;
  ss2 >> parsed;
  EXPECT_EQ(parsed, -3);
}

TEST(Int4Test, EdgeCases) {
  // Test all possible values
  for (int i = -8; i <= 7; ++i) {
    Int4 val(i);
    EXPECT_EQ(int(val), i);
  }

  // Test wraparound behavior through raw bits
  Int4 max_plus_one =
      Int4::fromRawBits(0x8); // This is -8 in 4-bit two's complement
  EXPECT_EQ(int(max_plus_one), -8);

  // Test that upper bits are ignored in fromRawBits
  EXPECT_EQ(Int4::fromRawBits(0xFF).getRawBits(), 0xF);
  EXPECT_EQ(Int4::fromRawBits(0x1F).getRawBits(), 0xF);

  // Verify specific wrap-around case requested by user
  EXPECT_EQ(int(Int4(4) + Int4(4)), -8); // 4 + 4 = 8 wraps to -8

  // Test constructor saturation
  for (int i = -20; i <= 20; ++i) {
    Int4 val(i);
    int expected = i;
    if (i > 7)
      expected = 7;
    if (i < -8)
      expected = -8;
    EXPECT_EQ(int(val), expected) << "Failed for constructor with i=" << i;
  }
}

TEST(Int4Test, BitcastConversions) {
  // Test bitcast conversions between Int4 and UInt4
  for (int bits = 0; bits < 16; ++bits) {
    Int4 i4 = Int4::fromRawBits(bits);
    UInt4 u4 = i4.toUInt4();
    Int4 i4_back = u4.toInt4();

    // Bitcast should preserve raw bits
    EXPECT_EQ(u4.getRawBits(), bits);
    EXPECT_EQ(i4_back.getRawBits(), bits);

    // But interpret them differently
    int signed_val = int(i4);
    int unsigned_val = int(u4);

    if (bits < 8) {
      // 0-7 are positive in both
      EXPECT_EQ(signed_val, unsigned_val);
    } else {
      // 8-15 are negative in Int4 but positive in UInt4
      EXPECT_EQ(signed_val, unsigned_val - 16);
    }
  }

  // Test specific cases
  Int4 neg_one(-1);
  UInt4 fifteen = neg_one.toUInt4();
  EXPECT_EQ(int(fifteen), 15);
  EXPECT_EQ(fifteen.getRawBits(), 0xF);

  UInt4 eight(8);
  Int4 neg_eight = eight.toInt4();
  EXPECT_EQ(int(neg_eight), -8);
  EXPECT_EQ(neg_eight.getRawBits(), 0x8);
}

//===----------------------------------------------------------------------===//
// UInt4 Tests
//===----------------------------------------------------------------------===//

TEST(Uint4Test, DefaultConstruction) {
  UInt4 u;
  EXPECT_EQ(u.getRawBits(), 0);
  EXPECT_EQ(int(u), 0);
}

TEST(Uint4Test, IntegerConstruction) {
  // Test construction within range
  EXPECT_EQ(int(UInt4(0)), 0);
  EXPECT_EQ(int(UInt4(15)), 15);
  EXPECT_EQ(int(UInt4(7)), 7);
  EXPECT_EQ(int(UInt4(10)), 10);

  // Test saturation on overflow
  EXPECT_EQ(int(UInt4(16)), 15);  // Saturate to max
  EXPECT_EQ(int(UInt4(100)), 15); // Saturate to max
  EXPECT_EQ(int(UInt4(-1)), 0);   // Saturate to min
  EXPECT_EQ(int(UInt4(-100)), 0); // Saturate to min
}

TEST(Uint4Test, RawBitsConstruction) {
  // Test fromRawBits
  for (int i = 0; i < 16; ++i) {
    UInt4 val = UInt4::fromRawBits(i);
    EXPECT_EQ(val.getRawBits(), i);
    EXPECT_EQ(int(val), i);
  }
}

TEST(Uint4Test, ArithmeticOperations) {
  UInt4 a(5);
  UInt4 b(3);
  UInt4 c(10);

  // Addition
  EXPECT_EQ(int(a + b), 8);
  EXPECT_EQ(int(a + c), 15);
  EXPECT_EQ(int(UInt4(12) + UInt4(5)), 15); // Saturates to max

  // Subtraction
  EXPECT_EQ(int(a - b), 2);
  EXPECT_EQ(int(b - a), 0); // Saturates to 0 (no negative)
  EXPECT_EQ(int(c - a), 5);
  EXPECT_EQ(int(UInt4(3) - UInt4(7)), 0); // Saturates to min

  // Multiplication
  EXPECT_EQ(int(a * b), 15); // 15 saturates to 15
  EXPECT_EQ(int(UInt4(2) * UInt4(3)), 6);
  EXPECT_EQ(int(UInt4(4) * UInt4(5)), 15); // 20 saturates to 15

  // Division
  EXPECT_EQ(int(UInt4(12) / UInt4(3)), 4);
  EXPECT_EQ(int(UInt4(15) / UInt4(2)), 7);
  EXPECT_EQ(int(UInt4(7) / UInt4(2)), 3);
  EXPECT_EQ(int(UInt4(6) / UInt4(0)), 0); // Division by zero returns 0

  // Modulo
  EXPECT_EQ(int(UInt4(15) % UInt4(4)), 3);
  EXPECT_EQ(int(UInt4(7) % UInt4(3)), 1);
  EXPECT_EQ(int(UInt4(6) % UInt4(0)), 0); // Modulo by zero returns 0

  // Compound assignment
  UInt4 d(4);
  d += UInt4(3);
  EXPECT_EQ(int(d), 7);

  d -= UInt4(2);
  EXPECT_EQ(int(d), 5);

  d *= UInt4(3);
  EXPECT_EQ(int(d), 15); // 15 saturates to 15

  d /= UInt4(3);
  EXPECT_EQ(int(d), 5);

  d %= UInt4(3);
  EXPECT_EQ(int(d), 2);
}

TEST(Uint4Test, BitwiseOperations) {
  UInt4 a = UInt4::fromRawBits(0b1010); // 10
  UInt4 b = UInt4::fromRawBits(0b0110); // 6

  // AND
  EXPECT_EQ((a & b).getRawBits(), 0b0010);

  // OR
  EXPECT_EQ((a | b).getRawBits(), 0b1110);

  // XOR
  EXPECT_EQ((a ^ b).getRawBits(), 0b1100);

  // NOT
  EXPECT_EQ((~a).getRawBits(), 0b0101);
  EXPECT_EQ(int(~a), 5);

  // Shift left
  EXPECT_EQ(int(UInt4(3) << 1), 6);
  EXPECT_EQ(int(UInt4(5) << 1), 10);
  EXPECT_EQ(int(UInt4(10) << 1), 15); // 20 saturates to 15
  EXPECT_EQ(int(UInt4(2) << 3), 15);  // 16 saturates to 15
  EXPECT_EQ(int(UInt4(2) << 4), 0);   // Shift >= 4 returns 0

  // Shift right (logical shift for unsigned)
  EXPECT_EQ(int(UInt4(8) >> 1), 4);
  EXPECT_EQ(int(UInt4(15) >> 1), 7);
  EXPECT_EQ(int(UInt4(1) >> 1), 0);
  EXPECT_EQ(int(UInt4(10) >> 2), 2);
  EXPECT_EQ(int(UInt4(15) >> 4), 0); // Shift >= 4 returns 0
}

TEST(Uint4Test, ComparisonOperators) {
  UInt4 a(5);
  UInt4 b(5);
  UInt4 c(10);
  UInt4 d(3);

  // Equality
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a == d);

  // Inequality
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_TRUE(a != d);

  // Less than
  EXPECT_FALSE(a < b);
  EXPECT_TRUE(a < c);
  EXPECT_FALSE(a < d);
  EXPECT_TRUE(d < a);

  // Less than or equal
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(a <= d);

  // Greater than
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_TRUE(a > d);

  // Greater than or equal
  EXPECT_TRUE(a >= b);
  EXPECT_FALSE(a >= c);
  EXPECT_TRUE(a >= d);
}

TEST(Uint4Test, SpecialValues) {
  // Test min and max
  EXPECT_EQ(int(UInt4::min()), 0);
  EXPECT_EQ(int(UInt4::max()), 15);
  EXPECT_EQ(int(UInt4::zero()), 0);
}

TEST(Uint4Test, StreamOperator) {
  UInt4 val(12);
  std::stringstream ss;
  ss << val;
  int parsed;
  ss >> parsed;
  EXPECT_EQ(parsed, 12);
}

TEST(Uint4Test, EdgeCases) {
  // Test all possible values
  for (int i = 0; i <= 15; ++i) {
    UInt4 val(i);
    EXPECT_EQ(int(val), i);
  }

  // Test that upper bits are ignored in fromRawBits
  EXPECT_EQ(UInt4::fromRawBits(0xFF).getRawBits(), 0xF);
  EXPECT_EQ(UInt4::fromRawBits(0x1F).getRawBits(), 0xF);

  // Test conversion differences with Int4
  // Raw bits 0x8 is -8 in Int4 but 8 in UInt4
  Int4 signed_val = Int4::fromRawBits(0x8);
  UInt4 unsigned_val = UInt4::fromRawBits(0x8);
  EXPECT_EQ(int(signed_val), -8);
  EXPECT_EQ(int(unsigned_val), 8);
}

TEST(Uint4Test, BitcastConversions) {
  // Test bitcast conversions from UInt4 perspective
  for (int i = 0; i <= 15; ++i) {
    UInt4 u4(i);
    Int4 i4 = u4.toInt4();
    UInt4 u4_back = i4.toUInt4();

    // Bitcast should preserve raw bits
    EXPECT_EQ(i4.getRawBits(), i);
    EXPECT_EQ(u4_back.getRawBits(), i);

    // Values 0-7 are the same in both representations
    // Values 8-15 in UInt4 become -8 to -1 in Int4
    if (i < 8) {
      EXPECT_EQ(int(i4), i);
    } else {
      EXPECT_EQ(int(i4), i - 16);
    }
  }
}

//===----------------------------------------------------------------------===//
// Float16 Tests
//===----------------------------------------------------------------------===//

TEST(Float16Test, DefaultConstruction) {
  Float16 f;
  EXPECT_EQ(f.getRawBits(), 0);
  EXPECT_EQ(float(f), 0.0f);
}

TEST(Float16Test, FloatConversion) {
  // Test basic values
  EXPECT_EQ(float(Float16(0.0f)), 0.0f);
  EXPECT_EQ(float(Float16(1.0f)), 1.0f);
  EXPECT_EQ(float(Float16(-1.0f)), -1.0f);

  // Test powers of 2
  EXPECT_EQ(float(Float16(2.0f)), 2.0f);
  EXPECT_EQ(float(Float16(0.5f)), 0.5f);
  EXPECT_EQ(float(Float16(0.25f)), 0.25f);

  // Test values that require rounding
  EXPECT_TRUE(approxEqual(float(Float16(3.14159f)), 3.14159f, 0.001f));
  EXPECT_TRUE(approxEqual(float(Float16(1.23456f)), 1.23456f, 0.001f));
}

TEST(Float16Test, SpecialValues) {
  // Test infinity
  Float16 pos_inf(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(pos_inf.isInf());
  EXPECT_FALSE(pos_inf.isNegative());
  EXPECT_TRUE(std::isinf(float(pos_inf)));
  EXPECT_EQ(pos_inf.getRawBits(), Float16::infinity().getRawBits());

  Float16 neg_inf(-std::numeric_limits<float>::infinity());
  EXPECT_TRUE(neg_inf.isInf());
  EXPECT_TRUE(neg_inf.isNegative());
  EXPECT_TRUE(std::isinf(float(neg_inf)));
  EXPECT_EQ(neg_inf.getRawBits(), Float16::negInfinity().getRawBits());

  // Test NaN
  Float16 nan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(nan.isNaN());
  EXPECT_TRUE(std::isnan(float(nan)));

  // Test zero
  Float16 zero(0.0f);
  EXPECT_TRUE(zero.isZero());
  EXPECT_FALSE(zero.isNegative());

  Float16 neg_zero(-0.0f);
  EXPECT_TRUE(neg_zero.isZero());
  EXPECT_TRUE(neg_zero.isNegative());
}

TEST(Float16Test, RangeAndPrecision) {
  // Test max value
  Float16 max_val = Float16::max();
  EXPECT_TRUE(approxEqual(float(max_val), 65504.0f, 1.0f));

  // Test min normalized value
  Float16 min_val = Float16::min();
  EXPECT_TRUE(approxEqual(float(min_val), 6.103515625e-5f, 1e-10f));

  // Test epsilon
  Float16 eps = Float16::epsilon();
  EXPECT_TRUE(approxEqual(float(eps), 9.765625e-4f, 1e-7f));

  // Test denormal values
  Float16 denorm(1e-5f);
  EXPECT_TRUE(float(denorm) > 0);
  EXPECT_TRUE(float(denorm) < float(Float16::min()));
}

TEST(Float16Test, ArithmeticOperations) {
  Float16 a(2.5f);
  Float16 b(1.5f);

  // Addition
  Float16 sum = a + b;
  EXPECT_TRUE(approxEqual(float(sum), 4.0f));

  // Subtraction
  Float16 diff = a - b;
  EXPECT_TRUE(approxEqual(float(diff), 1.0f));

  // Multiplication
  Float16 prod = a * b;
  EXPECT_TRUE(approxEqual(float(prod), 3.75f));

  // Division
  Float16 quot = a / b;
  EXPECT_TRUE(approxEqual(float(quot), 1.66667f, 0.001f));

  // Negation
  Float16 neg = -a;
  EXPECT_TRUE(approxEqual(float(neg), -2.5f));

  // Compound assignment
  Float16 c(1.0f);
  c += a;
  EXPECT_TRUE(approxEqual(float(c), 3.5f));

  c -= b;
  EXPECT_TRUE(approxEqual(float(c), 2.0f));

  c *= Float16(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 4.0f));

  c /= Float16(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 2.0f));
}

TEST(Float16Test, ComparisonOperators) {
  Float16 a(1.0f);
  Float16 b(2.0f);
  Float16 c(1.0f);

  // Equality
  EXPECT_TRUE(a == c);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != c);

  // Ordering
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(b <= a);

  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= b);

  // NaN comparisons
  Float16 nan = Float16::quietNaN();
  EXPECT_FALSE(nan == nan);
  EXPECT_TRUE(nan != nan);
  EXPECT_FALSE(nan < a);
  EXPECT_FALSE(a < nan);

  // Zero comparisons
  Float16 pos_zero(0.0f);
  Float16 neg_zero(-0.0f);
  EXPECT_TRUE(pos_zero == neg_zero);
}

TEST(Float16Test, Overflow) {
  // Test overflow to infinity
  Float16 large(100000.0f);
  EXPECT_TRUE(large.isInf());
  EXPECT_FALSE(large.isNegative());

  // Test arithmetic overflow
  Float16 big = Float16::max();
  Float16 result = big + big;
  EXPECT_TRUE(result.isInf());
}

TEST(Float16Test, Underflow) {
  // Test underflow to zero
  Float16 tiny(1e-10f);
  EXPECT_TRUE(tiny.isZero());

  // Test gradual underflow (denormals)
  Float16 small(1e-5f);
  EXPECT_FALSE(small.isZero());
  EXPECT_TRUE(float(small) > 0);
}

TEST(Float16Test, StreamOperator) {
  Float16 val(3.14f);
  std::stringstream ss;
  ss << val;
  float parsed;
  ss >> parsed;
  EXPECT_TRUE(approxEqual(parsed, float(val), 0.01f));
}

TEST(Float16Test, SaturatingIntegerConversions) {
  // Test normal conversions - C++ truncates towards zero when converting float
  // to int
  EXPECT_EQ(Float16(10.7f).toInt8Sat(), 10);   // 10.7 truncates to 10
  EXPECT_EQ(Float16(-10.7f).toInt8Sat(), -10); // -10.7 truncates to -10
  EXPECT_EQ(Float16(10.7f).toInt16Sat(), 10);
  EXPECT_EQ(Float16(-10.7f).toInt16Sat(), -10);
  EXPECT_EQ(Float16(10.7f).toInt32Sat(), 10);
  EXPECT_EQ(Float16(-10.7f).toInt32Sat(), -10);
  EXPECT_EQ(Float16(10.7f).toInt64Sat(), 10);
  EXPECT_EQ(Float16(-10.7f).toInt64Sat(), -10);

  // Test saturation to int8_t
  EXPECT_EQ(Float16(200.0f).toInt8Sat(), 127);
  EXPECT_EQ(Float16(-200.0f).toInt8Sat(), -128);

  // Test saturation to int16_t
  EXPECT_EQ(Float16(40000.0f).toInt16Sat(), 32767);
  EXPECT_EQ(Float16(-40000.0f).toInt16Sat(), -32768);
  EXPECT_EQ(Float16::max().toInt16Sat(), 32767); // Float16 max is 65504
  EXPECT_EQ((-Float16::max()).toInt16Sat(), -32768);

  // Test NaN and infinity conversion
  EXPECT_EQ(Float16::quietNaN().toInt8Sat(), 0);
  EXPECT_EQ(Float16::quietNaN().toInt16Sat(), 0);
  EXPECT_EQ(Float16::quietNaN().toInt32Sat(), 0);
  EXPECT_EQ(Float16::quietNaN().toInt64Sat(), 0);

  EXPECT_EQ(Float16::infinity().toInt8Sat(), 127);
  EXPECT_EQ(Float16::infinity().toInt16Sat(), 32767);
  EXPECT_EQ(Float16::infinity().toInt32Sat(), 2147483647);
  EXPECT_EQ(Float16::negInfinity().toInt8Sat(), -128);
  EXPECT_EQ(Float16::negInfinity().toInt16Sat(), -32768);
  EXPECT_EQ(Float16::negInfinity().toInt32Sat(), -2147483648);

  // Test zero
  EXPECT_EQ(Float16(0.0f).toInt8Sat(), 0);
  EXPECT_EQ(Float16(-0.0f).toInt8Sat(), 0);
}

//===----------------------------------------------------------------------===//
// BFloat16 Tests
//===----------------------------------------------------------------------===//

TEST(BFloat16Test, DefaultConstruction) {
  BFloat16 f;
  EXPECT_EQ(f.getRawBits(), 0);
  EXPECT_EQ(float(f), 0.0f);
}

TEST(BFloat16Test, FloatConversion) {
  // Test basic values
  EXPECT_EQ(float(BFloat16(0.0f)), 0.0f);
  EXPECT_EQ(float(BFloat16(1.0f)), 1.0f);
  EXPECT_EQ(float(BFloat16(-1.0f)), -1.0f);

  // Test powers of 2
  EXPECT_EQ(float(BFloat16(2.0f)), 2.0f);
  EXPECT_EQ(float(BFloat16(0.5f)), 0.5f);
  EXPECT_EQ(float(BFloat16(0.25f)), 0.25f);

  // Test values that require rounding
  EXPECT_TRUE(approxEqual(float(BFloat16(3.14159f)), 3.14159f, 0.01f));
  EXPECT_TRUE(approxEqual(float(BFloat16(1.23456f)), 1.23456f, 0.01f));
}

TEST(BFloat16Test, SpecialValues) {
  // Test infinity
  BFloat16 pos_inf(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(pos_inf.isInf());
  EXPECT_FALSE(pos_inf.isNegative());
  EXPECT_TRUE(std::isinf(float(pos_inf)));
  EXPECT_EQ(pos_inf.getRawBits(), BFloat16::infinity().getRawBits());

  BFloat16 neg_inf(-std::numeric_limits<float>::infinity());
  EXPECT_TRUE(neg_inf.isInf());
  EXPECT_TRUE(neg_inf.isNegative());
  EXPECT_TRUE(std::isinf(float(neg_inf)));
  EXPECT_EQ(neg_inf.getRawBits(), BFloat16::negInfinity().getRawBits());

  // Test NaN
  BFloat16 nan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(nan.isNaN());
  EXPECT_TRUE(std::isnan(float(nan)));

  // Test zero
  BFloat16 zero(0.0f);
  EXPECT_TRUE(zero.isZero());
  EXPECT_FALSE(zero.isNegative());

  BFloat16 neg_zero(-0.0f);
  EXPECT_TRUE(neg_zero.isZero());
  EXPECT_TRUE(neg_zero.isNegative());
}

TEST(BFloat16Test, RangeAndPrecision) {
  // BFloat16 has same exponent range as float32
  BFloat16 large(1e38f);
  EXPECT_FALSE(large.isInf());
  EXPECT_TRUE(float(large) > 1e37f);

  // Test min normalized value
  BFloat16 min_val = BFloat16::min();
  EXPECT_TRUE(
      approxEqual(float(min_val), std::numeric_limits<float>::min(), 1e-45f));

  // Test epsilon
  BFloat16 eps = BFloat16::epsilon();
  EXPECT_TRUE(approxEqual(float(eps), 0.0078125f, 1e-7f));
}

TEST(BFloat16Test, ArithmeticOperations) {
  BFloat16 a(2.5f);
  BFloat16 b(1.5f);

  // Addition
  BFloat16 sum = a + b;
  EXPECT_TRUE(approxEqual(float(sum), 4.0f, 0.1f));

  // Subtraction
  BFloat16 diff = a - b;
  EXPECT_TRUE(approxEqual(float(diff), 1.0f, 0.1f));

  // Multiplication
  BFloat16 prod = a * b;
  EXPECT_TRUE(approxEqual(float(prod), 3.75f, 0.1f));

  // Division
  BFloat16 quot = a / b;
  EXPECT_TRUE(approxEqual(float(quot), 1.66667f, 0.1f));

  // Negation
  BFloat16 neg = -a;
  EXPECT_TRUE(approxEqual(float(neg), -2.5f, 0.1f));

  // Compound assignment
  BFloat16 c(1.0f);
  c += a;
  EXPECT_TRUE(approxEqual(float(c), 3.5f, 0.1f));

  c -= b;
  EXPECT_TRUE(approxEqual(float(c), 2.0f, 0.1f));

  c *= BFloat16(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 4.0f, 0.1f));

  c /= BFloat16(2.0f);
  EXPECT_TRUE(approxEqual(float(c), 2.0f, 0.1f));
}

TEST(BFloat16Test, ComparisonOperators) {
  BFloat16 a(1.0f);
  BFloat16 b(2.0f);
  BFloat16 c(1.0f);

  // Equality
  EXPECT_TRUE(a == c);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != c);

  // Ordering
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(b <= a);

  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= b);

  // NaN comparisons
  BFloat16 nan = BFloat16::quietNaN();
  EXPECT_FALSE(nan == nan);
  EXPECT_TRUE(nan != nan);
  EXPECT_FALSE(nan < a);
  EXPECT_FALSE(a < nan);

  // Zero comparisons
  BFloat16 pos_zero(0.0f);
  BFloat16 neg_zero(-0.0f);
  EXPECT_TRUE(pos_zero == neg_zero);
}

TEST(BFloat16Test, RoundingBehavior) {
  // Test round-to-nearest-even

  // This value has bits set that will be truncated
  float val1 = 1.0f + (1.0f / 256.0f) + (1.0f / 512.0f); // 1.00390625
  BFloat16 bf1(val1);
  // Should round up because sticky bits are set
  EXPECT_TRUE(approxEqual(float(bf1), 1.0f + (1.0f / 128.0f), 1e-6f));

  // Test tie case - rounds to even
  float val2 = 1.0f + (1.0f / 256.0f); // Exactly halfway
  BFloat16 bf2(val2);
  // Should round to nearest even (down in this case)
  EXPECT_TRUE(approxEqual(float(bf2), 1.0f, 1e-6f));
}

TEST(BFloat16Test, StreamOperator) {
  BFloat16 val(3.14f);
  std::stringstream ss;
  ss << val;
  float parsed;
  ss >> parsed;
  EXPECT_TRUE(approxEqual(parsed, float(val), 0.1f));
}

TEST(BFloat16Test, SaturatingIntegerConversions) {
  // Test normal conversions - C++ truncates towards zero when converting float
  // to int
  EXPECT_EQ(BFloat16(10.7f).toInt8Sat(), 10);   // 10.7 truncates to 10
  EXPECT_EQ(BFloat16(-10.7f).toInt8Sat(), -10); // -10.7 truncates to -10
  EXPECT_EQ(BFloat16(10.7f).toInt16Sat(), 10);
  EXPECT_EQ(BFloat16(-10.7f).toInt16Sat(), -10);
  EXPECT_EQ(BFloat16(10.7f).toInt32Sat(), 10);
  EXPECT_EQ(BFloat16(-10.7f).toInt32Sat(), -10);
  EXPECT_EQ(BFloat16(10.7f).toInt64Sat(), 10);
  EXPECT_EQ(BFloat16(-10.7f).toInt64Sat(), -10);

  // Test saturation to int8_t
  EXPECT_EQ(BFloat16(200.0f).toInt8Sat(), 127);
  EXPECT_EQ(BFloat16(-200.0f).toInt8Sat(), -128);

  // Test saturation to int16_t
  EXPECT_EQ(BFloat16(40000.0f).toInt16Sat(), 32767);
  EXPECT_EQ(BFloat16(-40000.0f).toInt16Sat(), -32768);

  // Test saturation to int32_t
  EXPECT_EQ(BFloat16(3e9f).toInt32Sat(), 2147483647);
  EXPECT_EQ(BFloat16(-3e9f).toInt32Sat(), -2147483648);

  // Test NaN and infinity conversion
  EXPECT_EQ(BFloat16::quietNaN().toInt8Sat(), 0);
  EXPECT_EQ(BFloat16::quietNaN().toInt16Sat(), 0);
  EXPECT_EQ(BFloat16::quietNaN().toInt32Sat(), 0);
  EXPECT_EQ(BFloat16::quietNaN().toInt64Sat(), 0);

  EXPECT_EQ(BFloat16::infinity().toInt8Sat(), 127);
  EXPECT_EQ(BFloat16::infinity().toInt16Sat(), 32767);
  EXPECT_EQ(BFloat16::infinity().toInt32Sat(), 2147483647);
  EXPECT_EQ(BFloat16::infinity().toInt64Sat(), 9223372036854775807LL);
  EXPECT_EQ(BFloat16::negInfinity().toInt8Sat(), -128);
  EXPECT_EQ(BFloat16::negInfinity().toInt16Sat(), -32768);
  EXPECT_EQ(BFloat16::negInfinity().toInt32Sat(), -2147483648);
  EXPECT_EQ(BFloat16::negInfinity().toInt64Sat(), (-9223372036854775807LL - 1));

  // Test zero
  EXPECT_EQ(BFloat16(0.0f).toInt8Sat(), 0);
  EXPECT_EQ(BFloat16(-0.0f).toInt8Sat(), 0);
}

// Cross-type comparison tests
TEST(DataTypeTest, Float8VsFloat16VsBFloat16) {
  // Use a value that shows clear precision differences
  float test_val = 1.23456789f;

  F8E4M3FN f8(test_val);
  Float16 f16(test_val);
  BFloat16 bf16(test_val);

  // Float8 has the least precision
  float f8_error = std::abs(float(f8) - test_val);
  float f16_error = std::abs(float(f16) - test_val);
  float bf16_error = std::abs(float(bf16) - test_val);

  // For normal values in Float16's range:
  // Float16 has 10 mantissa bits, BFloat16 has 7, Float8 has 3
  // Generally Float16 is more accurate, but for some values BFloat16 might be
  // closer due to rounding. What we can guarantee is Float8 has much less
  // precision than both.

  // Float8 should definitely have more error than both Float16 and BFloat16
  EXPECT_LT(f16_error, f8_error * 0.5f);
  EXPECT_LT(bf16_error, f8_error * 0.5f);

  // Also verify the actual precision differences are reasonable
  EXPECT_LT(f16_error, 0.001f); // Float16 should be quite accurate
  EXPECT_LT(bf16_error, 0.01f); // BFloat16 less so
  EXPECT_LT(f8_error, 0.2f);    // Float8 much less accurate

  // Test range differences
  float large_val = 1e20f;
  F8E4M3FN f8_large(large_val);
  Float16 f16_large(large_val);
  BFloat16 bf16_large(large_val);

  // Float8 should clamp to max (no infinity)
  EXPECT_FALSE(f8_large.isNaN());
  EXPECT_EQ(f8_large.getRawBits(), 0x7E);

  // Float16 should overflow to infinity
  EXPECT_TRUE(f16_large.isInf());

  // BFloat16 can represent this value
  EXPECT_FALSE(bf16_large.isInf());
  EXPECT_TRUE(float(bf16_large) > 1e19f);
}

// Edge case tests
TEST(DataTypeTest, EdgeCases) {
  // Test smallest positive normal float16
  Float16 f16_min_normal = Float16::fromRawBits(0x0400);
  EXPECT_TRUE(approxEqual(float(f16_min_normal), 6.103515625e-5f, 1e-10f));

  // Test largest subnormal float16
  Float16 f16_max_subnormal = Float16::fromRawBits(0x03FF);
  EXPECT_TRUE(float(f16_max_subnormal) < float(f16_min_normal));
  EXPECT_TRUE(float(f16_max_subnormal) > 0);

  // Test smallest positive subnormal float16
  Float16 f16_min_subnormal = Float16::fromRawBits(0x0001);
  EXPECT_TRUE(float(f16_min_subnormal) > 0);
  EXPECT_TRUE(
      approxEqual(float(f16_min_subnormal), 5.960464477539063e-8f, 1e-15f));

  // Test NaN propagation
  F8E4M3FN nan_f8 = F8E4M3FN::quietNaN();
  F8E4M3FN result_f8 = nan_f8 + F8E4M3FN(1.0f);
  EXPECT_TRUE(result_f8.isNaN());

  Float16 nan_f16 = Float16::quietNaN();
  Float16 result_f16 = nan_f16 + Float16(1.0f);
  EXPECT_TRUE(result_f16.isNaN());

  BFloat16 nan_bf16 = BFloat16::quietNaN();
  BFloat16 result_bf16 = nan_bf16 * BFloat16(0.0f);
  EXPECT_TRUE(result_bf16.isNaN());
}

// Float8 specific edge cases
TEST(Float8Test, E4M3FNSpecificBehavior) {
  // Test that values above max get clamped, not turned to infinity
  F8E4M3FN beyond_max(500.0f);
  EXPECT_EQ(beyond_max.getRawBits(), 0x7E);
  EXPECT_FALSE(beyond_max.isNaN());

  // Test denormal range
  // Smallest positive denormal: 2^-9 = 0.001953125
  F8E4M3FN min_denorm = F8E4M3FN::fromRawBits(0x01);
  EXPECT_TRUE(approxEqual(float(min_denorm), 0.001953125f, 1e-9f));

  // Largest denormal: 7 * 2^-9 = 0.013671875
  F8E4M3FN max_denorm = F8E4M3FN::fromRawBits(0x07);
  EXPECT_TRUE(approxEqual(float(max_denorm), 0.013671875f, 1e-9f));

  // Test all exponent values
  for (int exp = 0; exp < 15; ++exp) {
    // Create a normal number with this exponent and mantissa 0
    int8_t bits = (exp << 3);
    F8E4M3FN val = F8E4M3FN::fromRawBits(bits);
    float result = float(val);

    if (exp == 0) {
      // Exponent 0 with mantissa 0 is zero
      EXPECT_EQ(result, 0.0f);
    } else if (exp == 15) {
      // Exponent 15 is reserved for NaN (0x78-0x7F)
      // With mantissa 0, 0x78 might be treated as a special case
      // E4M3FN doesn't have infinity, so this could be max value or NaN
      continue; // Skip this case as it's implementation-defined
    } else {
      // Normal values: 2^(exp-7)
      float expected = std::pow(2.0f, exp - 7);
      EXPECT_TRUE(approxEqual(result, expected, expected * 0.1f))
          << "Failed for exp=" << exp << ", bits=0x" << std::hex
          << (int)(unsigned char)bits << ", expected=" << expected
          << ", got=" << result;
    }
  }
}

} // anonymous namespace
