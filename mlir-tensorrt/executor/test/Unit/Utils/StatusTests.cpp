//===- StatusTests.cpp  ---------------------------------------------------===//
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
/// Tests for the Status object.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/Support/Debug.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mtrt;

TEST(TestStatus, TestStatusMacros) {
  auto result = []() -> Status {
    auto internalErr =
        mtrt::getInternalErrorStatus("some error - {0}", "some explanation");
    RETURN_STATUS_IF_ERROR(internalErr);
    return getOkStatus();
  }();

  ASSERT_FALSE(result.isOk());
  ASSERT_TRUE(result.isError());
  ASSERT_EQ(result.getCode(), StatusCode::InternalError);
  EXPECT_THAT(
      result.getMessage(),
      testing::MatchesRegex(
          R"~((.*)/StatusTests\.cpp:([0-9]+) some error - some explanation)~"));
}

TEST(TestStatusOr, StatusOr) {
  StatusOr<int> s(123);
  EXPECT_TRUE(s.isOk());
  EXPECT_FALSE(s.isError());
  EXPECT_EQ(*s, 123);

  StatusOr<int> e = getInternalErrorStatus("some error");
  EXPECT_FALSE(e.isOk());
  EXPECT_TRUE(e.isError());
  EXPECT_EQ(e.getStatus().getCode(), StatusCode::InternalError);
  EXPECT_EQ(e.checkStatus().getMessage(), "some error");
}

struct CopyMoveCounter {
  static inline int copy_count = 0;
  static inline int move_count = 0;

  static void reset() {
    copy_count = 0;
    move_count = 0;
  }

  int value;
  CopyMoveCounter(int v) : value(v) {}
  CopyMoveCounter(const CopyMoveCounter &o) : value(o.value) { ++copy_count; }
  CopyMoveCounter(CopyMoveCounter &&o) : value(o.value) { ++move_count; }
};

TEST(TestStatusOr, RvalueOverloadReducesCopies) {
  CopyMoveCounter::reset();

  auto makeExpected = []() -> StatusOr<CopyMoveCounter> {
    return CopyMoveCounter(42);
  };

  // Extract from rvalue - should prefer moves
  CopyMoveCounter result = std::move(makeExpected()).getValue();

  // Should have moves, not copies (exact count depends on optimization)
  EXPECT_GT(CopyMoveCounter::move_count, 0);
  EXPECT_EQ(CopyMoveCounter::copy_count, 0); // No copies with &&!
}
