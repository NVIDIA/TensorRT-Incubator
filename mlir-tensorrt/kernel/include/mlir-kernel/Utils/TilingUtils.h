//===- TilingUtils.h ------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Utilities related to tiling of shapes and distribution calculations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_TILINGUTILS_H
#define MLIR_TENSORRT_UTILS_TILINGUTILS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include <numeric>

namespace mlir::tiling_utils {

/// Return factors of `num` ordered from largest divisor to smallest divisor.
/// If `numDivisors` is set, then only that many of the largest divisors are
/// returned.
SmallVector<int64_t> getFactors(int64_t num);

/// TileShapeInfo encapsulates the result of calling
/// `enumerateProcessorDistributionShapes`. The information in this struct is
/// specific to an iteration space shape and dimension ordering that is provided
/// when calling that function.
struct TileShapeInfo {
  /// A shape of the tile whose volume is smaller than the max volume given to
  /// `enumerateTileShapes`.
  SmallVector<int64_t> tileShape;

  /// The amount of tiles along each dimension of the original iteration space.
  /// If the tile shape actually represents a shape of processors, then this is
  /// the amount of work required per processor along each dimension of the
  /// iteration space is given by this vector. The total number of tiles (total
  /// amount of work) required is the product of this shape.
  SmallVector<int64_t> workShape;
};

/// This function attempts to enumerate the possibilities for dividing up an
/// iteration space using a tile shape whose volume is no larger than some upper
/// bound. The tile shape can also be thought of
/// as a distribution of processors to an iteration space where only the upper
/// bound on number of processors is given. The returned vectors
/// describe the potential arrangement of the processors by describing their
/// shape (the tile shape). The caller gives the maximum tile volume
/// allowed. One potential tile shape is provided for each value for
/// `maxTileVolumes`. The `dimensionOrder` specifies the order of the
/// iteration space dimensions over which the tile volume is distributed. This
/// function tries to distribute as much of the shape budget as possible to each
/// dimension greedily without creating residual tiles.
SmallVector<TileShapeInfo>
getTileShapesUsingVolumeBudgets(ArrayRef<int64_t> iterationSpaceShape,
                                ArrayRef<int64_t> dimensionOrder,
                                ArrayRef<int64_t> maxTileVolumes);

struct TileShapeSelectionConfig {
  int64_t deviceNumSMs;
  uint64_t registersPerBlockBytes;
  uint64_t sharedMemoryPerBlockBytes;
  bool getPowerOfTwoTiles = false;
  // Number of stages for reduction dimension strip mining.
  // This multiplies the shared memory usage for tiles with reduction
  // dimensions.
  uint64_t numStages = 1;
};

struct TileShapeSelectionResult {
  SmallVector<int64_t> ctaWorkloadShape;
  SmallVector<int64_t> ctaBlockingShape;
  SmallVector<int64_t> threadTileShape;

  SmallVector<int64_t> gridShape;
  SmallVector<int64_t> threadBlockShape;
};

/// This function uses a simple model to produce a set of tile shapes
/// that should be applied to the linalg op to guide the production of a GPU
/// kernel. The different tile shapes returned are as follows. Each of the
/// tile shapes have the same rank as the iteration space of the linalg op:
///
///   1. The "CTA workload shape". This is the entire work performed by
///      the CTA, including accounting for all iterations of inner
///      strip-mining loops in the kernel. Also returned is the "grid shape",
///      which is exactly the iteration space of the linalg op divided
///      by the CTA workload shape.

///   2. The "CTA blocking shape". This is the shape of the work performed
///      by the CTA inside the strip-mining loop nest within the kernel.
///      If this shape is equal to the "CTA workload shape", then the
///      kernel does not have any inner strip-mining loops.
///
///   3. The "thread workload shape". This is the shape of the work performed
///      by a single thread. Note that this shape does not specify exactly
///      how threads should be arranged over the CTA blocking tile in order
///      to achieve this workload. That is up to the implementation (e.g.
///      threads may be distributed cyclically in one dimension but blocked
///      in another).
///
/// ### Implementation
///
/// The algorithm is inspired by [1], which describes a simple tile size
/// selection model for tiling affine loop nests when targeted toward CPUs.
/// The core idea itself is very simple and can repurposed for formulating
/// tile size selection in different scenarios.
///
/// At each level, we assume that the tile shape `S` is parameterized by
///
/// ```
/// (1) S = [ ⌊𝞽⌋ w₀ , ⌊𝞽⌋ w₁, ..., ⌊𝞽⌋ wₙ]
/// ```
///
/// Where each `wi` is a constant multiplicative factor chosen by heuristic.
/// The only unknown is 𝞽, a positive real number, which is called the
/// "tile base". Therefore, determining 𝞽
/// will determine the shape. Differently from [1], we enforce `wi` to be
/// integers and the floor function wraps only 𝞽. Typically `wi` are chosen
/// to be "nice" small numbers like 2, 4, 8 in order to yield good tile shapes.
/// The weights are chosen relative to one another to place emphasis on
/// different dimensions, e.g. for selecting the CTA tile shape we prefer to
/// weight parallel dimensions more heavily.
///
/// To find 𝞽, we formulate a system of constraints and solve it.
///
/// The insight of [1] is that some important constraints can be formulated
/// as simple polynomials which can easily be solved to find
/// a good initial tile shape, which may be further refined.
///
/// A good example is the matmul problem, which has an iteration space
/// `[M, N, K]`. Let `[m, n, k]` be the CTA blocking shape. All input
/// tiles must fit into shared memory of size `C`, so  we have the
/// following constraint (where `a` is element byte size).
///
/// ```
///  The tile shape is [𝞽m, 𝞽n, 𝞽k].
/// (3)  𝞽² · (a · (m·k + n·k)) <= C
/// (4)  𝞽²  <= C /  (a · (m·k + n·k))
/// ```
///
/// All the terms in `C /  (a · (m·k + n·k))` are constants, so the square
/// root of that number yields the desired upper bound on 𝞽.
///
/// The same technique can be used to lower bound 𝞽 or apply other bounds.
/// Lower bounds can be formulated to ensure that a minimum volume
/// of warps/threads can be used in the next level of tiling).
/// Additional upper bounds can be formulated to represent constraints
/// on the number of registers used (although this is very loose).
///
/// Once we solve for each of the bounds, it is straightforward to select
/// a 𝞽 that lies within the feasible range. Finally, we calculate
/// `S` according to equation (1). The last step is to do some post-processing.
/// In our case, we reduce each tile extent to be the next smallest factor
/// that evenly divides the corresponding iteration space extent.
///
/// ### References
///
/// [1] A practical tile size selection model for affine loop nests.
///     Kumudha Narasimhan, Aravind Acharya, Abhinav Baid, and Uday Bondhugula.
///     2021.  In Proceedings of the 35th ACM International Conference on
///     Supercomputing (ICS
///     '21). Association for Computing Machinery, New York, NY, USA, 27–39.
///     https://doi.org/10.1145/3447818.3462213
FailureOr<TileShapeSelectionResult>
simpleGpuTileShapeSelection(ArrayRef<AffineMap> indexingMaps,
                            ArrayRef<utils::IteratorType> iteratorTypes,
                            TypeRange operandTypes, int64_t numDpsInits,
                            ArrayRef<int64_t> loopRanges,
                            TileShapeSelectionConfig config);

/// Same as above, but retrieves the necessary information from the LinalgOp.
FailureOr<TileShapeSelectionResult>
simpleGpuTileShapeSelection(linalg::LinalgOp op,
                            TileShapeSelectionConfig config);

/// Returns the value of the "tile base" (tau parameter, see docs for
/// `simpleGpuTileShapeSelection` above) based on storage constraints (sum of
/// all bytes required for operands). We assume the weights for each iterator
/// are given by `weights` and `indexingMaps` and `operandTypes` contain the
/// metadata for the linalg op's operands. Note that currently only indexing
/// maps with "projected permutations" are supported. The function solves for
/// tau based on the constraint `f(tau) = constraint` where `f` is a polynomial
/// that computes the total storage required for all operands.
std::optional<double>
getWeightedTileBaseViaPolynomial(ArrayRef<int64_t> weights,
                                 ArrayRef<AffineMap> indexingMaps,
                                 TypeRange operandTypes, int64_t constraint,
                                 ArrayRef<double> operandWeights);

/// Returns the value of the "tile base" (tau parameter, see docs for
/// `simpleGpuTileShapeSelection` above) by solving a constraint on the volume
/// of the tiled iteration space. The `targetType` controls whether we are
/// considering parallel dimension volume or reduction volume. The function
/// solves for tau based on the constraint `f(tau) = constraint` where `f` is a
/// monomial that computes the total iteration space volume.
std::optional<double> getTileBaseUsingWorkVolumeConstraint(
    ArrayRef<int64_t> weights, ArrayRef<utils::IteratorType> iteratorTypes,
    utils::IteratorType targetType, double constraint);

/// Returns the the shape S by computing
///
/// ```
/// (1) S = [ ⌊𝞽⌋ w₀ , ⌊𝞽⌋ w₁, ..., ⌊𝞽⌋ wₙ]
/// ```
///
/// Where `base` corresponds to tau in the equation above and `w_i` are elements
/// of the weights vector.
SmallVector<int64_t> getTileShapeFromBaseAndWeights(double base,
                                                    ArrayRef<int64_t> weights);

/// Encapsulates the result of an initial CTA-level tiling by specifying the
/// grid shape, CTA work tile shape, and CTA work tile shape within strip-mining
/// loops (e.g. the CTA blocking shape).
struct GridLevelTilingResult {
  SmallVector<int64_t> gridShape;
  SmallVector<int64_t> ctaWorkShape;
  SmallVector<int64_t> ctaTileShape;
};

/// Return the grid-level tiling based on a heuristic set of constraints
/// formulated using the linalg operation metadata as well as the shared memory
/// and number of registers available. If getPowerOfTwoTiles is true, then this
/// function ensures to return a power of two tiling for ctaWorkShape and
/// ctaTileShape. But gridShape is not guaranteed to be power of 2.
std::optional<GridLevelTilingResult>
getGridLevelTiling(ArrayRef<AffineMap> indexingMaps,
                   ArrayRef<utils::IteratorType> iteratorTypes,
                   TypeRange operandTypes, ArrayRef<int64_t> staticLoopRanges,
                   unsigned numDPSInits, uint64_t smemMax, uint64_t registerMax,
                   bool getPowerOfTwoTiles = false, uint64_t numStages = 1);

template <typename RangeTy>
int64_t ctaVolume(RangeTy &&input) {
  auto range = make_filter_range(std::forward<RangeTy>(input),
                                 [](int64_t x) { return x != 0; });
  return std::accumulate(range.begin(), range.end(), 1, std::multiplies<>());
}

inline SmallVector<int64_t> fixupTileShape(ArrayRef<int64_t> iterSpaceShape,
                                           ArrayRef<int64_t> tileShape) {
  SmallVector<int64_t> result;
  for (auto [l, r] : llvm::zip_equal(iterSpaceShape, tileShape))
    result.push_back(l == r ? 0 : r);
  return result;
}

template <typename T>
T roundDownToPowerOf2(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return (x <= 1.0) ? 1.0 : std::pow(2.0, std::floor(std::log2(x)));
  } else if constexpr (std::is_integral_v<T>) {
    if (x <= 1)
      return 1;
    T result = 1;
    while (result <= x / 2) {
      result <<= 1;
    }
    return result;
  }
}

template <typename T>
T roundUpToPowerOf2(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return (x <= 1.0) ? 1.0 : std::pow(2.0, std::ceil(std::log2(x)));
  } else if constexpr (std::is_integral_v<T>) {
    if (x <= 1)
      return 1;
    // Check if x is already a power of 2
    if ((x & (x - 1)) == 0)
      return x;
    T result = 1;
    while (result < x) {
      result <<= 1;
    }
    return result;
  }
}

} // namespace mlir::tiling_utils

#endif // MLIR_TENSORRT_UTILS_TILINGUTILS_H
