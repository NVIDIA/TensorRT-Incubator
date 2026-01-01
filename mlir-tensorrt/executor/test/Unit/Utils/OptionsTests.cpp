//===- OptionsTests.cpp ---------------------------------------------------===//
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
/// Tests for mlir-tensorrt-common Support/Options.{h,cpp}.
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt-common/Support/Options.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

struct TestOptions : mlir::CLOptionScope {
  using CLOptionScope::CLOptionScope;

  Option<bool> foo{*this, "foo", llvm::cl::init(false)};
  Option<int> bar{*this, "bar", llvm::cl::init(0)};
  Option<std::string> str{*this, "str", llvm::cl::init("")};
  ListOption<std::string> includeDirs{*this, "I", llvm::cl::CommaSeparated};
};

TEST(OptionSetTest, EmptyStringSucceedsNoChange) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  EXPECT_TRUE(mlir::succeeded(opts.parseFromString("")));
  EXPECT_FALSE(opts.foo);
  EXPECT_EQ(opts.bar, 0);
  EXPECT_EQ(opts.str, "");
  EXPECT_TRUE(opts.includeDirs.empty());
}

TEST(OptionSetTest, BracedBundleAndDashFormsSucceed) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  EXPECT_TRUE(
      mlir::succeeded(opts.parseFromString("{ --foo -bar=7 -I=a,b,c }")));
  EXPECT_TRUE(opts.foo);
  EXPECT_EQ(opts.bar, 7);
  EXPECT_THAT(llvm::ArrayRef<std::string>(opts.includeDirs),
              testing::ElementsAre("a", "b", "c"));
}

TEST(OptionSetTest, QuotesAndBackslashesTokenize) {
  {
    TestOptions opts(mlir::CLOptionScope::LocalScope{});
    EXPECT_TRUE(mlir::succeeded(opts.parseFromString("--str=\"hello world\"")));
    EXPECT_EQ(opts.str, "hello world");
  }
  {
    // GNU-style escaping should keep this as one token with a space in the
    // value.
    TestOptions opts(mlir::CLOptionScope::LocalScope{});
    EXPECT_TRUE(mlir::succeeded(opts.parseFromString("--str=hello\\ world")));
    EXPECT_EQ(opts.str, "hello world");
  }
}

TEST(OptionSetTest, UnknownOptionFailsAndReportsName) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  std::string err;
  auto onError = [&](llvm::StringRef msg) { err = msg.str(); };
  EXPECT_TRUE(
      mlir::failed(opts.parseFromString("--does_not_exist=1", onError)));
  EXPECT_EQ(err, "option not found: does_not_exist");
}

TEST(OptionSetTest, PositionalArgumentsRejected) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  std::string err;
  auto onError = [&](llvm::StringRef msg) { err = msg.str(); };
  EXPECT_TRUE(mlir::failed(opts.parseFromString("foo", onError)));
  EXPECT_EQ(err, "positional arguments not supported (prefix with '--')");
}

TEST(OptionSetTest, MissingOptionNameRejected) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  std::string err;
  auto onError = [&](llvm::StringRef msg) { err = msg.str(); };
  EXPECT_TRUE(mlir::failed(opts.parseFromString("--=1", onError)));
  EXPECT_EQ(err, "empty option name");
}

TEST(OptionSetTest, PrintBundleRoundTripsParse) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  EXPECT_TRUE(mlir::succeeded(opts.parseFromString(
      "{ --foo --bar=7 --str=\"hello world\" --I=a --I=b }")));

  std::string printed;
  llvm::raw_string_ostream os(printed);
  opts.print(os);
  os.flush();

  EXPECT_EQ(printed, "{--I=a --I=b --bar=7 --foo --str=\"hello world\"}");

  TestOptions reparsed(mlir::CLOptionScope::LocalScope{});
  EXPECT_TRUE(mlir::succeeded(reparsed.parseFromString(printed)));

  EXPECT_TRUE(reparsed.foo);
  EXPECT_EQ(reparsed.bar, 7);
  EXPECT_EQ(reparsed.str, "hello world");
  EXPECT_THAT(llvm::ArrayRef<std::string>(reparsed.includeDirs),
              testing::ElementsAre("a", "b"));
}

TEST(OptionSetTest, ListAssign) {
  TestOptions opts(mlir::CLOptionScope::LocalScope{});
  EXPECT_TRUE(mlir::succeeded(opts.parseFromString(
      "{ --foo --bar=7 --str=\"hello world\" --I=a --I=b }")));
  EXPECT_THAT(llvm::ArrayRef<std::string>(opts.includeDirs),
              testing::ElementsAre("a", "b"));

  // Try to assign/override includeDirs.
  opts.includeDirs.assign({"c", "d"});
  EXPECT_THAT(llvm::ArrayRef<std::string>(opts.includeDirs),
              testing::ElementsAre("c", "d"));
}

} // namespace
