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
#include "mlir-executor/Support/Status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlirtrt;

TEST(TestStatus, TestStatusMacros) {
  auto result = []() -> Status {
    auto internalErr =
        mlirtrt::getInternalErrorStatus("some error - {0}", "some explanation");
    RETURN_STATUS_IF_ERROR(internalErr);
    return getOkStatus();
  }();

  ASSERT_FALSE(result.isOk());
  ASSERT_TRUE(result.isError());
  ASSERT_EQ(result.getCode(), StatusCode::InternalError);
  EXPECT_THAT(
      result.getString(),
      testing::MatchesRegex(
          R"~(InternalError: (.*)/StatusTests\.cpp:([0-9]+) InternalError: some error - some explanation)~"));
}
