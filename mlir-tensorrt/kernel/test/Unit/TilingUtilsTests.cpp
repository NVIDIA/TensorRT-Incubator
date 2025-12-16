//===- TilingUtilsTests.cpp -----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Unit tests for tiling utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Utils/TilingUtils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using ::testing::ElementsAre;

TEST(TilingUtils, TestTileSizeSelection) {

  MLIRContext context;
  MLIRContext *ctx = &context;
  context.loadAllAvailableDialects();
  auto getMap = [&](StringRef str) { return mlir::parseAffineMap(str, ctx); };
  Type f32Type = Float32Type::get(ctx);

  auto matmulMaps = llvm::SmallVector<AffineMap>{
      getMap("(d0, d1, d2) -> (d0, d2)"), getMap("(d0, d1, d2) -> (d1, d2)"),
      getMap("(d0, d1, d2) -> (d0, d1)")};

  {
    std::optional<double> tau = tiling_utils::getWeightedTileBaseViaPolynomial(
        {1, 1, 2}, matmulMaps, {f32Type, f32Type, f32Type}, 4096 * 4,
        {1., 1., 1.});
    EXPECT_TRUE(tau.has_value());
    if (tau.has_value())
      EXPECT_THAT(tiling_utils::getTileShapeFromBaseAndWeights(*tau, {1, 1, 2}),
                  ElementsAre(28, 28, 56));
  }

  {
    std::optional<double> tau = tiling_utils::getWeightedTileBaseViaPolynomial(
        {1, 1, 1}, matmulMaps, {f32Type, f32Type, f32Type}, 4096 * 4,
        {1., 1., 1.});
    EXPECT_TRUE(tau.has_value());
    if (tau.has_value())
      EXPECT_THAT(tiling_utils::getTileShapeFromBaseAndWeights(*tau, {1, 1, 1}),
                  ElementsAre(36, 36, 36));
  }

  {
    std::optional<double> tau = tiling_utils::getWeightedTileBaseViaPolynomial(
        {16, 8, 8},
        {getMap("(d0, d1, d2) -> (d1, d2)"),
         getMap("(d0, d1, d2)-> (d0, d1, d2)"), getMap("(d0, d1, d2) -> (d0)")},
        {f32Type, f32Type, f32Type}, 65e3, {1., 1., 1.});
    EXPECT_TRUE(tau.has_value());
    if (tau.has_value())
      EXPECT_THAT(
          tiling_utils::getTileShapeFromBaseAndWeights(*tau, {16, 8, 8}),
          ElementsAre(112, 56, 56));
  }

  {
    std::optional<double> tau = tiling_utils::getWeightedTileBaseViaPolynomial(
        {1},
        {getMap("(d0) -> (d0)"), getMap("(d0) -> (d0)"),
         getMap("(d0) -> (d0)")},
        {f32Type, f32Type, f32Type}, 4096 * 4, {1., 1., 1.});
    EXPECT_TRUE(tau.has_value());
    if (tau.has_value())
      EXPECT_THAT(tiling_utils::getTileShapeFromBaseAndWeights(*tau, {1}),
                  ElementsAre(1365));
  }

  {
    std::optional<double> tau = tiling_utils::getWeightedTileBaseViaPolynomial(
        {1, 2, 1, 3},
        {getMap("(d0, d1, d2, d3) -> (d0, d1)"),
         getMap("(d0, d1, d2, d3) -> (d2)"),
         getMap("(d0, d1, d2, d3) -> (d0, d3)")},
        {f32Type, f32Type, f32Type}, 4096 * 4, {1., 1., 1.});
    EXPECT_TRUE(tau.has_value());
    if (tau.has_value())
      EXPECT_THAT(
          tiling_utils::getTileShapeFromBaseAndWeights(*tau, {1, 2, 1, 3}),
          ElementsAre(28, 56, 28, 84));
  }

  {
    std::optional<double> tau =
        tiling_utils::getTileBaseUsingWorkVolumeConstraint(
            {1, 1, 1, 1},
            {utils::IteratorType::parallel, utils::IteratorType::parallel},
            utils::IteratorType::parallel, 4);
    ASSERT_TRUE(tau.has_value());
    EXPECT_EQ(*tau, 2.0);
  }

  {
    std::optional<tiling_utils::GridLevelTilingResult> result =
        tiling_utils::getGridLevelTiling(
            matmulMaps,
            {utils::IteratorType::parallel, utils::IteratorType::parallel,
             utils::IteratorType::reduction},
            {f32Type, f32Type, f32Type}, {4096, 4096, 4096}, 1, 256e3, 65536);
    EXPECT_TRUE(result.has_value());
    if (result) {
      EXPECT_THAT(result->ctaTileShape, ElementsAre(64, 64, 64));
      EXPECT_THAT(result->gridShape, ElementsAre(4096 / 64, 4096 / 64, 1));
      EXPECT_THAT(result->ctaWorkShape, ElementsAre(64, 64, 4096));
    }
  }

  {
    // Test dynamic batch dimension
    auto bmmMaps = llvm::SmallVector<AffineMap>{
        getMap("(d0, d1, d2, d3) -> (d0, d1, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d2, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d1, d2)")};
    std::optional<tiling_utils::GridLevelTilingResult> result =
        tiling_utils::getGridLevelTiling(
            bmmMaps,
            {utils::IteratorType::parallel, utils::IteratorType::parallel,
             utils::IteratorType::parallel, utils::IteratorType::reduction},
            {f32Type, f32Type, f32Type},
            {ShapedType::kDynamic, 4096, 4096, 4096}, 1, 256e3, 65536);
    EXPECT_TRUE(result.has_value());
    if (result) {
      EXPECT_THAT(result->gridShape,
                  ElementsAre(ShapedType::kDynamic, 4096 / 32, 4096 / 32, 1));
      EXPECT_THAT(result->ctaWorkShape, ElementsAre(1, 32, 32, 4096));
      EXPECT_THAT(result->ctaTileShape, ElementsAre(1, 32, 32, 32));
    }
  }

  {
    // Test dynamic outer-product dimension
    auto bmmMaps = llvm::SmallVector<AffineMap>{
        getMap("(d0, d1, d2, d3) -> (d0, d1, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d2, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d1, d2)")};
    std::optional<tiling_utils::GridLevelTilingResult> result =
        tiling_utils::getGridLevelTiling(
            bmmMaps,
            {utils::IteratorType::parallel, utils::IteratorType::parallel,
             utils::IteratorType::parallel, utils::IteratorType::reduction},
            {f32Type, f32Type, f32Type},
            {ShapedType::kDynamic, ShapedType::kDynamic, 4096, 4096}, 1, 256e3,
            65536);
    EXPECT_TRUE(result.has_value());
    if (result) {
      EXPECT_THAT(result->gridShape,
                  ElementsAre(ShapedType::kDynamic, ShapedType::kDynamic,
                              4096 / 32, 1));
      EXPECT_THAT(result->ctaWorkShape, ElementsAre(1, 1, 32, 4096));
      EXPECT_THAT(result->ctaTileShape, ElementsAre(1, 1, 32, 32));
    }
  }

  {
    // Test getPowerOfTwoTiles = true with non-power-of-2 staticLoopRanges
    // Expect the ctaTileShape dimensions to be the largest power-of-2 factor
    // of the corresponding loop range, subject to tau/weight constraints.
    std::optional<tiling_utils::GridLevelTilingResult> result =
        tiling_utils::getGridLevelTiling(
            matmulMaps, // Reuse matmul maps for simplicity
            {utils::IteratorType::parallel, utils::IteratorType::parallel,
             utils::IteratorType::reduction},
            {f32Type, f32Type, f32Type}, {96, 100, 75}, 1, 256e3, 65536, true);
    EXPECT_TRUE(result.has_value());
    if (result) {
      // Largest PoT factors: 96 -> 32, 100 -> 4, 75 -> 1
      // Assuming tau/weights allow these power-of-two smaller than
      // staticLoopRanges.
      EXPECT_THAT(result->ctaTileShape, ElementsAre(64, 64, 64));
      // Grid shape: ceil(96/64)=2, ceil(100/64)=2, reduction=1
      EXPECT_THAT(result->gridShape, ElementsAre(2, 2, 1));
      // CTA work shape: Tiles for parallel, round up to cover the full range
      // for reduction
      EXPECT_THAT(result->ctaWorkShape, ElementsAre(64, 64, 128));
    }
  }

  {
    // Test getPowerOfTwoTiles = true with different non-power-of-2 ranges and
    // maps.
    auto bmmMaps = llvm::SmallVector<AffineMap>{
        getMap("(d0, d1, d2, d3) -> (d0, d1, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d2, d3)"),
        getMap("(d0, d1, d2, d3) -> (d0, d1, d2)")};
    std::optional<tiling_utils::GridLevelTilingResult> result =
        tiling_utils::getGridLevelTiling(
            bmmMaps,
            {utils::IteratorType::parallel, utils::IteratorType::parallel,
             utils::IteratorType::parallel, utils::IteratorType::reduction},
            {f32Type, f32Type, f32Type}, {100, 200, 300, 49}, 1, 256e3, 65536,
            true);
    EXPECT_TRUE(result.has_value());
    if (result) {
      // Largest PoT factors: 100 -> 4, 200 -> 8, 300 -> 4, 49 -> 1
      // Assuming tau/weights allow these largest factors.
      EXPECT_THAT(result->ctaTileShape, ElementsAre(16, 16, 16, 16));
      // Grid shape: ceil(100/16)=7, ceil(200/16)=13, ceil(300/16)=19,
      // reduction=1
      EXPECT_THAT(result->gridShape, ElementsAre(7, 13, 19, 1));
      // CTA work shape: Tiles for parallel (since no reduction)
      EXPECT_THAT(result->ctaWorkShape, ElementsAre(16, 16, 16, 64));
    }
  }
}
