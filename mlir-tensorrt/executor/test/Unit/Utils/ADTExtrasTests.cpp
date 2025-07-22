//===- ADTExtrasTests.cpp ------------------------------------------------===//
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
/// Tests for extra ADT expansions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Support/ADTExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(FormatVariadicTests, RangeFormatting) {
  EXPECT_EQ("0, 1, 2", formatv("{0}", llvm::ArrayRef<int>{0, 1, 2}).str());
  EXPECT_EQ("0, 1, 2", formatv("{0}", std::array<int, 3>{0, 1, 2}).str());
  EXPECT_EQ("0, 1, 2", formatv("{0}", llvm::SmallVector<int>{0, 1, 2}).str());

  llvm::SmallVector<int64_t> myVec{0, 1, 2};
  EXPECT_EQ("0, 1, 2 0, 1, 2", formatv("{0} {1}", myVec, myVec).str());
}

TEST(FormatVariadicTests, TupleFormatting) {
  EXPECT_EQ("Tuple<0, 1, 2>", formatv("{0}", std::make_tuple(0, 1, 2)).str());
  EXPECT_EQ("Tuple<0x1x2>",
            formatv("{0:$[x]}", std::make_tuple(0, 1, 2)).str());
  EXPECT_EQ("Tuple<0x0, 1, 0x2>",
            formatv("{0:@<x>@<n>@<x>]}", std::make_tuple(0, 1, 2)).str());
  EXPECT_EQ("Tuple<0x0; 1; 0x2>",
            formatv("{0:$[; ]@<x>@<n>@<x>]}", std::make_tuple(0, 1, 2)).str());
}

TEST(FormatVariadicTests, ZipFormatting) {
  auto list1 = std::array<int, 2>{0, 3};
  auto list2 = std::array<int, 2>{1, 4};
  auto list3 = std::array<int, 2>{2, 5};
  auto zip = llvm::zip(list1, list2, list3);
  EXPECT_EQ("[Tuple<0, 1, 2>, Tuple<3, 4, 5>]",
            formatv("[{0}]", llvm::iterator_range(zip)).str());
  EXPECT_EQ("[Tuple<0, 1, 2>, Tuple<3, 4, 5>]", formatv("[{0}]", zip).str());
  EXPECT_EQ("[Tuple<0, 1, 2>; Tuple<3, 4, 5>]",
            formatv("[{0:$[; ]@[]}]", llvm::iterator_range(zip)).str());
  EXPECT_EQ("[Tuple<0x1x2>;Tuple<3x4x5>]",
            formatv("[{0:$[;]@[$<x>]}]", llvm::iterator_range(zip)).str());
}
