//===- Int4Tests.cpp  -----------------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Unit tests for INT4 runtime data type.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Common/Int4.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <type_traits>

using namespace mlirtrt::runtime;

template <typename T>
class Int4Test : public ::testing::Test {};

using Int4Types = ::testing::Types<nv_int4, nv_uint4>;
TYPED_TEST_SUITE(Int4Test, Int4Types);

TYPED_TEST(Int4Test, TestCreation) {
  using int4Type = TypeParam;
  int8_t i_i8{5};
  int16_t i_i16{5};
  int32_t i_i32{5};
  int64_t i_i64{5};
  int4Type num1(i_i8);
  EXPECT_EQ(num1, int4Type(i_i16));
  EXPECT_EQ(num1, int4Type(i_i32));
  EXPECT_EQ(num1, int4Type(i_i64));
  // copy constructor
  int4Type num2(num1);
  int4Type num3;
  num3 = num1;
  EXPECT_EQ(num1, num2);
  EXPECT_EQ(num1, num3);
  // move constructor
  int4Type num4(std::move(num1));
  int4Type num5;
  num5 = std::move(num1);
  EXPECT_EQ(num1, num4);
  EXPECT_EQ(num1, num5);
}

TEST(Int4Test, TestLimits) {
  nv_int4 a;
  EXPECT_EQ(a.getMaxValue(), nv_int4(7));
  EXPECT_EQ(a.getMinValue(), nv_int4(-8));
  nv_uint4 b;
  EXPECT_EQ(b.getMaxValue(), nv_uint4(15));
  EXPECT_EQ(b.getMinValue(), nv_uint4(0));
}

TYPED_TEST(Int4Test, TestTypeCast) {
  using int4Type = TypeParam;
  int4Type num1(4);
  EXPECT_EQ(static_cast<int8_t>(num1), 4);
  EXPECT_EQ(static_cast<int16_t>(num1), 4);
  EXPECT_EQ(static_cast<int32_t>(num1), 4);
  EXPECT_EQ(static_cast<int64_t>(num1), 4);
}

TYPED_TEST(Int4Test, TestOperators) {
  using int4Type = TypeParam;
  for (int i = static_cast<int>(int4Type::getMinValue());
       i <= static_cast<int>(int4Type::getMaxValue()); i++) {
    int4Type num1(i);
    EXPECT_EQ(-num1, int4Type(-i));
    // temp
    int4Type num2;
    EXPECT_EQ((num2 = num1, ++num2), int4Type(i + 1));
    EXPECT_EQ((num2 = num1, num2++), int4Type(i));
    EXPECT_EQ(num2, int4Type(i + 1));
    EXPECT_EQ((num2 = num1, --num2), int4Type(i - 1));
    EXPECT_EQ((num2 = num1, num2--), int4Type(i));
    EXPECT_EQ(num2, int4Type(i - 1));

    for (int j = static_cast<int>(int4Type::getMinValue());
         j <= static_cast<int>(int4Type::getMaxValue()); j++) {
      int4Type num3(j);
      EXPECT_EQ(num1 + num3, int4Type(i + j));
      EXPECT_EQ((num2 = num1, num2 += num3), int4Type(i + j));
      EXPECT_EQ(num1 - num3, int4Type(i - j));
      EXPECT_EQ((num2 = num1, num2 -= num3), int4Type(i - j));
      EXPECT_EQ(num1 * num3, int4Type(i * j));
      EXPECT_EQ((num2 = num1, num2 *= num3), int4Type(i * j));
      if (j != 0) {
        EXPECT_EQ(num1 % num3, int4Type(i % j));
        EXPECT_EQ(num1 / num3, int4Type(i / j));
        EXPECT_EQ((num2 = num1, num2 /= num3), int4Type(i / j));
      }
      EXPECT_EQ(num1 == num3, i == j);
      EXPECT_EQ(num1 != num3, i != j);
      EXPECT_EQ(num1 <= num3, i <= j);
      EXPECT_EQ(num1 >= num3, i >= j);
      EXPECT_EQ(num1 < num3, i < j);
      EXPECT_EQ(num1 > num3, i > j);
      EXPECT_EQ(num1 & num3, int4Type(i & j));
      EXPECT_EQ(num1 | num3, int4Type(i | j));
      EXPECT_EQ(num1 ^ num3, int4Type(i ^ j));
    }
    for (int8_t numBits = 0; numBits < 4; numBits++) {
      EXPECT_EQ(num1 >> numBits, int4Type(i >> numBits));
      EXPECT_EQ((num2 = num1, num2 >>= numBits), int4Type(i >> numBits));
      EXPECT_EQ(num1 << numBits, int4Type(i << numBits));
      EXPECT_EQ((num2 = num1, num2 <<= numBits), int4Type(i << numBits));
    }
  }
}

template <typename T>
class Int4PackedTest : public ::testing::Test {};

using Int4PackedTypes = ::testing::Types<nv_int4p, nv_uint4p>;
TYPED_TEST_SUITE(Int4PackedTest, Int4PackedTypes);

TYPED_TEST(Int4PackedTest, TestCreation) {
  using int4PackedType = TypeParam;
  // un = upper nibble (4 bits), ln = lower nibble
  int8_t i_i8un{3};
  int8_t i_i8ln{5};
  int16_t i_i16un{3};
  int16_t i_i16ln{5};
  int32_t i_i32un{3};
  int32_t i_i32ln{5};
  int64_t i_i64un{3};
  int64_t i_i64ln{5};
  int4PackedType num1(i_i8un, i_i8ln);
  EXPECT_EQ(num1, int4PackedType(i_i16un, i_i16ln));
  EXPECT_EQ(num1, int4PackedType(i_i32un, i_i32ln));
  EXPECT_EQ(num1, int4PackedType(i_i64un, i_i64ln));
  // copy constructor
  int4PackedType num2(num1);
  int4PackedType num3;
  num3 = num1;
  EXPECT_EQ(num1, num2);
  EXPECT_EQ(num1, num3);
  // move constructor
  int4PackedType num4(std::move(num1));
  int4PackedType num5;
  num5 = std::move(num1);
  EXPECT_EQ(num1, num4);
  EXPECT_EQ(num1, num5);
}

TEST(Int4PackedTest, TestLimits) {
  nv_int4p a;
  EXPECT_EQ(a.getMaxValue(), nv_int4p(7, 7));
  EXPECT_EQ(a.getMinValue(), nv_int4p(-8, -8));
  nv_uint4p b;
  EXPECT_EQ(b.getMaxValue(), nv_uint4p(15, 15));
  EXPECT_EQ(b.getMinValue(), nv_uint4p(0, 0));
}

TYPED_TEST(Int4PackedTest, TestOperators) {
  // For binary operations, first `i4p` is comprised of `a` and `b`, second
  // `i4p` is comprised of `x` and `y`.
  // Operations `op` is performed on (`a`, `b`) and (`x`, `y`) as (`a` op `x`,
  // `b` op `y`).
  using int4PackedType = TypeParam;
  std::ostringstream buffer;
  buffer << int4PackedType::getMinValue() << int4PackedType::getMaxValue();
  int int4PackedTypeMin = buffer.str()[0] - '0';
  int int4PackedTypeMax =
      std::is_same<int4PackedType, nv_int4p>()
          ? (buffer.str()[buffer.str().size() - 1] - '0')
          : stoi(buffer.str().substr(buffer.str().size() - 2, 2));

  for (int a = int4PackedTypeMin; a <= int4PackedTypeMax; a++) {
    for (int b = int4PackedTypeMin; b <= int4PackedTypeMax; b++) {
      int4PackedType num1(a, b);
      EXPECT_EQ(-num1, int4PackedType(-a, -b));
      int4PackedType num2;
      EXPECT_EQ((num2 = num1, ++num2), int4PackedType(a + 1, b + 1));
      EXPECT_EQ((num2 = num1, num2++), int4PackedType(a, b));
      EXPECT_EQ(num2, int4PackedType(a + 1, b + 1));
      EXPECT_EQ((num2 = num1, --num2), int4PackedType(a - 1, b - 1));
      EXPECT_EQ((num2 = num1, num2--), int4PackedType(a, b));
      EXPECT_EQ(num2, int4PackedType(a - 1, b - 1));

      for (int x = int4PackedTypeMin; x <= int4PackedTypeMax; x++) {
        for (int y = int4PackedTypeMin; y <= int4PackedTypeMax; y++) {
          int4PackedType num3(x, y);
          EXPECT_EQ(num1 + num3, int4PackedType(a + x, b + y));
          EXPECT_EQ((num2 = num1, num2 += num3), int4PackedType(a + x, b + y));
          EXPECT_EQ(num1 - num3, int4PackedType(a - x, b - y));
          EXPECT_EQ((num2 = num1, num2 -= num3), int4PackedType(a - x, b - y));
          EXPECT_EQ(num1 * num3, int4PackedType(a * x, b * y));
          EXPECT_EQ((num2 = num1, num2 *= num3), int4PackedType(a * x, b * y));
          if (x != 0 && y != 0) {
            EXPECT_EQ(num1 % num3, int4PackedType(a % x, b % y));
            EXPECT_EQ(num1 / num3, int4PackedType(a / x, b / y));
            EXPECT_EQ((num2 = num1, num2 /= num3),
                      int4PackedType(a / x, b / y));
          }
          EXPECT_EQ(num1 == num3,
                    (((b & 0x0F) | (a << 4)) == ((y & 0x0F) | (x << 4))));
          EXPECT_EQ(num1 != num3,
                    (((b & 0x0F) | (a << 4)) != ((y & 0x0F) | (x << 4))));

          EXPECT_EQ(num1 & num3, int4PackedType(a & x, b & y));
          EXPECT_EQ(num1 | num3, int4PackedType(a | x, b | y));
          EXPECT_EQ(num1 ^ num3, int4PackedType(a ^ x, b ^ y));
        }
      }
      for (int8_t numBits = 0; numBits < 4; numBits++) {
        EXPECT_EQ(num1 >> numBits, int4PackedType(a >> numBits, b >> numBits));
        EXPECT_EQ((num2 = num1, num2 >>= numBits),
                  int4PackedType(a >> numBits, b >> numBits));
        EXPECT_EQ(num1 << numBits, int4PackedType(a << numBits, b << numBits));
        EXPECT_EQ((num2 = num1, num2 <<= numBits),
                  int4PackedType(a << numBits, b << numBits));
      }
    }
  }
}