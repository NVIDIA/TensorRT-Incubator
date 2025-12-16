//===- TilingUtils.cpp ----------------------------------------------------===//
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
///===- TilingUtils.cpp ---------------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of tiling utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Utils/TilingUtils.h"
#include "mlir-tensorrt-common/Support/ADTExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <numeric>

#define DEBUG_TYPE "tiling-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")
#define DBGV(x, ...) LLVM_DEBUG(DBGS() << llvm::formatv(x "\n", __VA_ARGS__))

using namespace mlir;
using namespace mlir::tiling_utils;

static int64_t getElementBytes(Type t) {
  Type elementType = mlir::getElementTypeOrSelf(t);
  if (isa<IndexType>(elementType))
    return 8;
  if (elementType.isIntOrFloat())
    return llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
  if (auto complexType = dyn_cast<ComplexType>(elementType))
    return getElementBytes(complexType.getElementType());
  llvm::report_fatal_error("unhandled type encountered in getElementBytes");
}

/// Get divisors sorted largest to smallest.
SmallVector<int64_t> tiling_utils::getFactors(int64_t num) {
  assert(num > 0 && "expected positive number");
  SmallVector<int64_t> divisors{{num, 1}};
  for (int64_t i = 2; i * i < num; i++) {
    if (num % i == 0)
      divisors.append({num / i, i});
  }
  llvm::sort(divisors, std::greater<int64_t>());
  return divisors;
}

// Get the next factor of `num` that is greater than `start`.
static std::optional<int64_t> getNextFactorOf(int64_t start, int64_t num) {
  for (int64_t i = start + 1; i <= num; i++) {
    if (num % i == 0)
      return i;
  }
  return {};
}

// Get the next power-of-two that is greater than `start`.
static std::optional<int64_t> getNextPow2Of(int64_t start, int64_t num) {
  for (int64_t i = start + 1; i <= num; i++) {
    if (llvm::isPowerOf2_64(i))
      return i;
  }
  return {};
}

template <typename Callable>
std::optional<double> reduce(ArrayRef<std::optional<double>> values,
                             Callable &&compare) {
  std::optional<double> acc = {};
  for (std::optional<double> v : values) {
    if (!v)
      continue;
    if (!acc) {
      acc = *v;
      continue;
    }
    acc = compare(*acc, *v) ? *acc : *v;
  }
  return acc;
}

static SmallVector<int64_t>
roundToNextLargestFactorOf(ArrayRef<int64_t> toRound,
                           ArrayRef<int64_t> dividend) {
  SmallVector<int64_t> result;
  for (auto [l, r] : llvm::zip_equal(toRound, dividend)) {
    if (ShapedType::isDynamic(r)) {
      result.push_back(1);
      continue;
    }

    for (int64_t i = l; i >= 1; i--) {
      if (r % i == 0) {
        result.push_back(i);
        break;
      }
    }
  }
  return result;
}

// This function rounds down toRound to the next largest power of 2. If toRound
// is larger than dividend, it round down to the next largest power of 2 of
// dividend smaller than dividend.
static SmallVector<int64_t> roundToNextLargestPow2(ArrayRef<int64_t> toRound,
                                                   ArrayRef<int64_t> dividend) {
  SmallVector<int64_t> result;
  for (auto [l, r] : llvm::zip_equal(toRound, dividend)) {
    if (ShapedType::isDynamic(r)) {
      result.push_back(1);
      continue;
    }
    int64_t nextPow2 = roundDownToPowerOf2(std::min(l, r));
    result.push_back(nextPow2);
  }
  return result;
}

namespace {
class AffineMapUsesDimInResults : private llvm::SmallBitVector {
public:
  AffineMapUsesDimInResults(AffineMap map)
      : isFunctionOfDim(map.getNumDims(), 0) {
    for (AffineExpr expr : map.getResults())
      calculateRecursively(expr);
  }

  bool usesDim(unsigned pos) const { return isFunctionOfDim[pos]; }

private:
  void calculateRecursively(AffineExpr e) {
    if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(e)) {
      isFunctionOfDim.set(dimExpr.getPosition());
      return;
    }
    if (auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(e)) {
      calculateRecursively(binExpr.getLHS());
      calculateRecursively(binExpr.getRHS());
      return;
    }
  }
  llvm::SmallBitVector isFunctionOfDim;
};
} // namespace

// Evaluate the derivative of a polynomial given by `coef` on the value `x`.
static double evalPolynomialDeriv(double x, ArrayRef<double> coef) {
  double output = 0;
  for (auto [idx, c] : llvm::enumerate(llvm::reverse(coef.drop_front())))
    output += (idx + 1) * c * std::pow(x, idx);
  return output;
}

/// Evaluate a polynomial given by `coef` on the value `x`.
static double evalPolynomial(double x, ArrayRef<double> coef) {
  double output = 0;
  for (auto [idx, c] : llvm::enumerate(llvm::reverse(coef.drop_front(1))))
    output += c * std::pow(x, idx + 1);
  if (!coef.empty())
    output += coef.front();
  return output;
}

// Newton's method to find a root of a polynomial. We only fallback to this if
// we can't solve the polynomial using a simpler analytical method.
static std::optional<double> newton(ArrayRef<double> coef, double initialGuess,
                                    double tolerance = 1e-3,
                                    unsigned maxIterations = 100) {
  double x = initialGuess;
  unsigned step = 0;
  for (step = 0; step < maxIterations; ++step) {
    DBGV("newton's method on {0}, step {1}, value {2}, f(x) = {3}, f'(x) = {4}",
         coef, step, x, evalPolynomial(x, coef), evalPolynomialDeriv(x, coef));
    double fVal = evalPolynomial(x, coef);
    double fPrimeVal = evalPolynomialDeriv(x, coef);
    // Prevent division by zero
    if (std::abs(fPrimeVal) < std::numeric_limits<double>::epsilon())
      break;
    // Update the guess
    double xNext = x - fVal / fPrimeVal;
    // Check for convergence
    if (std::abs(xNext - x) < tolerance) {
      x = xNext;
      break;
    }
    x = xNext;
  }
  DBGV("newton's method on {0}, step {1}, value {2}, f(x) = {3}, f'(x) = {4}",
       coef, step, x, evalPolynomial(x, coef), evalPolynomialDeriv(x, coef));
  if (x < 0 || std::abs(evalPolynomial(x, coef)) > tolerance)
    return {};
  return x;
}

std::optional<double> tiling_utils::getWeightedTileBaseViaPolynomial(
    ArrayRef<int64_t> weights, ArrayRef<AffineMap> indexingMaps,
    TypeRange operandTypes, int64_t tileVolumeUB,
    ArrayRef<double> operandWeights) {
  if (indexingMaps.empty())
    return {};

  assert(weights.size() == indexingMaps.front().getNumDims());

  // Initialize the coefficients with 0-th power constant term.
  SmallVector<double, 4> coefficients = {-1.0 * tileVolumeUB};

  for (auto [map, operandType, operandWeight] :
       llvm::zip_equal(indexingMaps, operandTypes, operandWeights)) {
    if (!map.isProjectedPermutation(/*allowZeroInResults=*/true))
      return {};

    /// For operands whose maps are projected permutations, the
    /// volume `v` of a tile is equal to
    /// ```
    /// v = 𝞽ⁿ · ∏ W[d]
    /// ```
    /// where `n` is the number of results in the expression
    /// (less constant zeros) and `d` ranges over
    /// the dimension indices in the result.
    unsigned power = 0;
    double coef = 1.0;
    for (AffineExpr e : map.getResults()) {
      if (e.isSymbolicOrConstant())
        continue;
      power++;
      coef *= weights[cast<AffineDimExpr>(e).getPosition()];
    }
    coef *= getElementBytes(operandType);
    coef *= operandWeight;

    // Accumulate into the coefficients vector.
    coefficients.resize(std::max<size_t>(coefficients.size(), power + 1), 0);
    coefficients[power] += coef;
  }

  // Solve the polynomial. For no coefficients above 0, there is no constraint.
  if (coefficients.size() <= 1)
    return {};

  LLVM_DEBUG(DBGS() << llvm::formatv("coefficients = {0}\n", coefficients););

  // Degree 1 (linear).
  if (coefficients.size() == 2) {
    double tau = -coefficients[0] / coefficients[1];
    assert(tau >= 0 && "invalid tile base");
    return tau;
  }

  // Degree 2 (use quadratic equation).
  if (coefficients.size() == 3) {
    double c = coefficients[0];
    double b = coefficients[1];
    double a = coefficients[2];
    double discriminant = (b * b) - 4 * a * c;
    if (discriminant < 0)
      return {};
    double sqRoot = std::sqrt(discriminant);
    double root1 = (-b + sqRoot) / (2 * a);
    double root2 = (-b - sqRoot) / (2 * a);
    return (root1 > root2) ? root1 : root2;
  }

  // For any degree, if all other coefficients (besides constant) are 0, then we
  // can just use N'th root.
  if (llvm::all_of(ArrayRef(coefficients).drop_back(1).drop_front(1),
                   [](int64_t coef) { return coef == 0; }))
    return std::pow(-static_cast<double>(coefficients.front()) /
                        coefficients.back(),
                    1.0 / (coefficients.size() - 1));

  // Fallback to newton's method. Typically this takes ~10 steps.
  return newton(coefficients, 1.0, /*tolerance=*/1e-3, /*maxIterations=*/50);
}

std::optional<double> tiling_utils::getTileBaseUsingWorkVolumeConstraint(
    ArrayRef<int64_t> weights, ArrayRef<utils::IteratorType> iteratorTypes,
    utils::IteratorType targetType, double constraint) {

  // Initialize the coefficients with 0-th power constant term.
  unsigned power = 0;
  int64_t coef = 1;
  for (auto [idx, iterType] : llvm::enumerate(iteratorTypes)) {
    if (iterType != targetType)
      continue;
    power++;
    coef *= weights[idx];
  }

  // Not constrained enough, no bound.
  if (power == 0)
    return {};

  return std::pow(constraint / static_cast<double>(coef),
                  1.0 / static_cast<double>(power));
}

std::optional<GridLevelTilingResult> tiling_utils::getGridLevelTiling(
    ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, TypeRange operandTypes,
    ArrayRef<int64_t> staticLoopRanges, unsigned numDPSInits, uint64_t smemMax,
    uint64_t regMax, bool getPowerOfTwoTiles, uint64_t numStages) {
  // Give an initial set of weights. Each tile shape dim extent will be a
  // multiple of the weight assigned to it.
  // As an initial heuristic, we give reduction dimensions 1/2 the weight of a
  // parallel dimension.
  //
  // Here weights initializations are always power of 2 (they are 1, 8, 16). So
  // we can use the same code for getting initial weights when
  // getPowerOfTwoTiles is true.
  SmallVector<int64_t> weights;
  for (auto [idx, iterType] : llvm::enumerate(iteratorTypes)) {
    if (utils::IteratorType::reduction == iterType) {
      if (ShapedType::isDynamic(staticLoopRanges[idx]))
        return {};
      weights.push_back(8);
      continue;
    }
    if (ShapedType::isDynamic(staticLoopRanges[idx])) {
      weights.push_back(1);
      continue;
    }
    weights.push_back(16);
  }

  if (weights.empty())
    return {};

  // Increase weights for iterator indices that index into trailing dimensions
  // of operands. Tile sizes for iterator A should be weighted more than
  // iterator B if iterator A indexes into more trailing tensor dimensions than
  // B.
  SmallVector<int64_t> coalescingFactors(weights.size(), 0);
  for (AffineMap map : indexingMaps) {
    for (unsigned d : llvm::seq<unsigned>(0, iteratorTypes.size())) {
      // If the size of this iterator is unknown, then don't do anything. We
      // will be forcing the tile size to 1 later on.
      if (ShapedType::isDynamic(staticLoopRanges[d]))
        continue;
      if (map.getNumResults() > 0 &&
          map.getResults().back() == getAffineDimExpr(d, map.getContext()))
        coalescingFactors[d] += 1;
    }
  }

  // If we are getting power of two tiles, then we need to round the coalesce
  // factor down to the nearest power of two. This will ensure the weights are
  // still power of 2.
  for (auto [w, coalesce] : llvm::zip_equal(weights, coalescingFactors))
    if (coalesce > 0)
      w *= getPowerOfTwoTiles ? roundDownToPowerOf2(coalesce) : coalesce;

  // Compute weights for bound on Tau via register file size. We know
  // for sure that all output/acc tiles must fit into registers at the same
  // time. We discount the reduction dimensions since those do not all need to
  // fit into registers at once.
  //
  // If we are getting power of two tiles, essentially we let 𝞽 = 2^t for some
  // interger t. We can still use the same polynomial solver to get the value of
  // 𝞽.

  // For shared memory, multiply weight by numStages for input operands that
  // access reduction dimensions. This accounts for strip mining with multiple
  // stages loaded into shared memory simultaneously.
  SmallVector<double> dpsInputOperandWeights;
  dpsInputOperandWeights.reserve(indexingMaps.size() - numDPSInits);
  for (size_t i = 0; i < indexingMaps.size() - numDPSInits; ++i) {
    const AffineMap &map = indexingMaps[i];
    // Check if this operand accesses any reduction dimensions
    bool accessesReduction = false;
    for (auto [idx, iterType] : llvm::enumerate(iteratorTypes)) {
      if (iterType == utils::IteratorType::reduction) {
        // Check if this map uses this reduction dimension
        for (AffineExpr result : map.getResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(result)) {
            if (dimExpr.getPosition() == idx) {
              accessesReduction = true;
              break;
            }
          }
        }
        if (accessesReduction)
          break;
      }
    }
    // If this input accesses reduction dims, multiply weight by numStages
    // to account for multiple stages in shared memory
    dpsInputOperandWeights.push_back(accessesReduction ? numStages : 1.0);
  }

  SmallVector<double> operandWeightsForRFBound(indexingMaps.size(), 1.0);

  SmallVector<std::optional<double>> upperBounds = {
      // The first bound is based on the volume of input tiles. This volume is
      // bounded by the shared memory capacity. Volume of output/accumulator
      // tiles are usually held in registers. For inputs that access reduction
      // dimensions, we multiply by numStages to account for pipelined stages.
      getWeightedTileBaseViaPolynomial(
          weights, indexingMaps.drop_back(numDPSInits),
          operandTypes.drop_back(numDPSInits), smemMax, dpsInputOperandWeights),
      // This second bound is for the volume of all tiles. This relates tile
      // size to register file capacity.
      getWeightedTileBaseViaPolynomial(weights, indexingMaps, operandTypes,
                                       regMax, operandWeightsForRFBound),
  };

  // Choose the smallest of the upper bounds.
  std::optional<double> tauUb = reduce(upperBounds, std::less<double>());

  // This specifies the tau required to give at least volume of 128 in parallel
  // dimensions.
  std::optional<double> tauLb = getTileBaseUsingWorkVolumeConstraint(
      weights, iteratorTypes, utils::IteratorType::parallel, 128);
  if (!tauUb && !tauLb)
    return {};

  // Use a helper function to get the result.
  auto getResult = [&](double tau, ArrayRef<int64_t> weights) {
    // If we are getting power of two tiles, here both tau and weights should be
    // power of two already, and initial tile shape should also be power of two.
    SmallVector<int64_t> initialTileShape =
        getTileShapeFromBaseAndWeights(tau, weights);
    assert(!getPowerOfTwoTiles ||
           llvm::all_of(initialTileShape, llvm::isPowerOf2_64) &&
               "expected (tau * weight) gives power of two tile shape");

    SmallVector<int64_t> tileShape;
    if (getPowerOfTwoTiles) {
      tileShape = roundToNextLargestPow2(initialTileShape, staticLoopRanges);
    } else {
      tileShape =
          roundToNextLargestFactorOf(initialTileShape, staticLoopRanges);
    }

    // The grid shape will be filled with 'staticLoopRanges / ctaWorkloadShape'.
    SmallVector<int64_t> gridShape;
    SmallVector<int64_t> ctaWorkloadShape;

    for (auto [iterType, tileDim, iterSpaceSize] :
         llvm::zip_equal(iteratorTypes, tileShape, staticLoopRanges)) {
      // We don't tile the reduction across the grid.
      if (iterType == utils::IteratorType::reduction) {
        gridShape.push_back(1);
        if (getPowerOfTwoTiles) {
          ctaWorkloadShape.push_back(roundUpToPowerOf2(iterSpaceSize));
        } else {
          ctaWorkloadShape.push_back(iterSpaceSize);
        }
        continue;
      }

      // Parallel dimensions are tiled by the tileDim.a
      assert((!ShapedType::isDynamic(iterSpaceSize) || tileDim == 1) &&
             "expected to tile-by-1 for dynamic dimensions");
      gridShape.push_back(llvm::divideCeil(iterSpaceSize, tileDim));
      ctaWorkloadShape.push_back(tileDim);
    }
    // For isPowerOfTwoTiles, gridShape may not be power of two;
    // ctaWorkloadShape/tileShape may not be a factor of staticLoopRanges but
    // they are power of two.
    return GridLevelTilingResult{gridShape, ctaWorkloadShape, tileShape};
  };

  // If we are getting power of two tiles, then we need to round the 𝞽 down to
  // the nearest power of two.
  if (tauUb) {
    if (getPowerOfTwoTiles)
      tauUb = roundDownToPowerOf2(*tauUb);
    return getResult(*tauUb, weights);
  }

  if (tauLb) {
    if (getPowerOfTwoTiles)
      tauLb = roundDownToPowerOf2(*tauLb);
    return getResult(*tauLb, weights);
  }

  return {};
}

SmallVector<int64_t>
tiling_utils::getTileShapeFromBaseAndWeights(double base,
                                             ArrayRef<int64_t> weights) {
  return llvm::map_to_vector(weights, [base](int64_t w) -> int64_t {
    return std::max<int64_t>(std::floor(base), 1) * w;
  });
}

template <typename T>
static T volume(ArrayRef<T> ar) {
  T num = 1;
  for (T dim : ar) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}

/// Given the current number of processors, the maximum number of processors,
/// and the upper bound for the number of threads, this function returns
/// potential tuples of (newNumProcessors, currDimSize / numProcsAlongDim,
/// numProcsAlongDim).
static std::tuple<int64_t, int64_t, int64_t>
factorDimension(const int64_t currNumProcessors, const int64_t currDimSize,
                const int64_t numProcessorsLimit) {
  assert(currDimSize >= 1 && "expected valid currDimSize");
  for (int64_t divisor : getFactors(currDimSize)) {
    int64_t hypotheticalNumProcessors = currNumProcessors * divisor;
    if (currDimSize % divisor == 0 &&
        hypotheticalNumProcessors <= numProcessorsLimit) {
      return {hypotheticalNumProcessors, currDimSize / divisor, divisor};
    }
  }
  return {currNumProcessors, currDimSize, 1};
}

SmallVector<TileShapeInfo> tiling_utils::getTileShapesUsingVolumeBudgets(
    ArrayRef<int64_t> iterationSpaceShape, ArrayRef<int64_t> distributionOrder,
    ArrayRef<int64_t> maxNumProcessors) {
  assert(iterationSpaceShape.size() >= distributionOrder.size() &&
         "expected iteration space rank to equal distribution order size");
  SmallVector<TileShapeInfo> result;
  for (int64_t numProcUB : maxNumProcessors) {
    int64_t currNumThreads = 1;
    result.emplace_back(
        TileShapeInfo{SmallVector<int64_t>(iterationSpaceShape.size(), 1),
                      llvm::to_vector(iterationSpaceShape)});
    for (auto idx : distributionOrder) {
      int64_t loopDim = iterationSpaceShape[idx];
      auto [newNumThreads, newDimSize, divisor] =
          factorDimension(currNumThreads, loopDim, numProcUB);
      currNumThreads = newNumThreads;
      result.back().tileShape[idx] = divisor;
      result.back().workShape[idx] = newDimSize;
    }
  }
  return result;
}

FailureOr<TileShapeSelectionResult>
mlir::tiling_utils::simpleGpuTileShapeSelection(
    ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, TypeRange operandTypes,
    int64_t numDpsInits, ArrayRef<int64_t> loopRanges,
    TileShapeSelectionConfig config) {
  if (iteratorTypes.empty())
    return TileShapeSelectionResult{{}, {}, {}, {}, {}};

  std::optional<GridLevelTilingResult> gridTiling = getGridLevelTiling(
      indexingMaps, iteratorTypes, operandTypes, loopRanges, numDpsInits,
      config.sharedMemoryPerBlockBytes, config.registersPerBlockBytes,
      config.getPowerOfTwoTiles, config.numStages);
  if (!gridTiling)
    return failure();

  DBGV("grid_shape = {0}, cta_tile_shape = {1}",
       llvm::iterator_range(gridTiling->gridShape),
       llvm::iterator_range(gridTiling->ctaTileShape));

  // Compute weights for bound on Tau via register file size. We know for sure
  // that all output/acc tiles must fit into registers. The input tiles are
  // discounted since their corresponding SIMT variables are likely to have much
  // smaller live ranges.
  SmallVector<int64_t> threadTileWeights(iteratorTypes.size(), 1);
  threadTileWeights.back() = 2;

  int64_t parallelDimVolume = 1;
  for (auto [dimType, size] :
       llvm::zip_equal(iteratorTypes, gridTiling->ctaTileShape)) {
    if (dimType == utils::IteratorType::parallel)
      parallelDimVolume *= size;
  }

  SmallVector<double> operandWeights(indexingMaps.size(), 1.0);

  // Get an initial tile size for the thread. This is done by supposing that the
  // number of threads should be ~128 and then dividing the register file based
  // on that. The resulting thread tile size could be too small (because it
  // results in an excess number of threads). If the number of threads is too
  // big, then we can just scale it down later by growing the thread tile size
  // in one of the parallel dimensions.
  SmallVector<std::optional<double>> ubs = {
      getWeightedTileBaseViaPolynomial(
          threadTileWeights, indexingMaps, operandTypes,
          static_cast<uint64_t>(config.registersPerBlockBytes) * 1024 / 128,
          operandWeights),
      // This upper bound limits is to limit the size of Tau as a function of
      // the required parallelism. A bigger Tau (and therefore bigger thread
      // tile shape), will imply fewer number of threads. This is OK until the
      // number of threads drops too low. This constraint specifies at least one
      // warp (which will be further scaled back later if parallel volume is
      // smaller than that).
      getTileBaseUsingWorkVolumeConstraint(
          threadTileWeights, iteratorTypes, utils::IteratorType::parallel,
          static_cast<double>(parallelDimVolume) / 128)};

  SmallVector<std::optional<double>> lbs = {
      // This upper bound limits is to limit the size of Tau as a function of
      // the required parallelism. A bigger Tau (and therefore bigger thread
      // tile shape), will imply fewer number of threads. This is OK until the
      // number of threads drops too low. This constraint specifies at least one
      // warp (which will be further scaled back later if parallel volume is
      // smaller than that).
      getTileBaseUsingWorkVolumeConstraint(
          threadTileWeights, iteratorTypes, utils::IteratorType::parallel,
          static_cast<double>(parallelDimVolume) / 128)};

  // Choose the smallest of the upper bounds.
  std::optional<double> ub = reduce(ubs, std::less<double>());
  if (!ub)
    return {};

  // If we are getting power of two tiles, then we need to round the 𝞽 down to
  // the nearest power of two.
  if (config.getPowerOfTwoTiles)
    ub = roundDownToPowerOf2(*ub);

  DBGV("thread tile shape base ubs = {0}, ub = {1}", llvm::iterator_range(ubs),
       ub);

  // If requires getting power of two tiles, then the above
  // gridTiling->ctaTileShape should be power of two already.
  assert(!config.getPowerOfTwoTiles ||
         llvm::all_of(gridTiling->ctaTileShape, llvm::isPowerOf2_64) &&
             "expected gridTiling->ctaTileShape to be power of two");

  SmallVector<int64_t> initialThreadTileShape =
      getTileShapeFromBaseAndWeights(*ub, threadTileWeights);

  SmallVector<int64_t> threadTileShape;
  if (config.getPowerOfTwoTiles) {
    threadTileShape = roundToNextLargestPow2(initialThreadTileShape,
                                             gridTiling->ctaTileShape);
  } else {
    threadTileShape = roundToNextLargestFactorOf(initialThreadTileShape,
                                                 gridTiling->ctaTileShape);
  }

  DBGV("thread tile shape = {0}", llvm::iterator_range(threadTileShape));

  constexpr int64_t kMaxThreadsPerBlock = 1024;

  SmallVector<int64_t> threadBlockShape(iteratorTypes.size(), 1);
  for (auto [idx, threadTileDim, iterType] :
       llvm::enumerate(threadTileShape, iteratorTypes)) {
    if (iterType == utils::IteratorType::reduction) {
      threadBlockShape[idx] = 1;
      threadTileShape[idx] = gridTiling->ctaTileShape[idx];
      continue;
    }
    threadBlockShape[idx] = gridTiling->ctaTileShape[idx] / threadTileDim;
  }

  // Helper used to update the thread block shape in-place as we expand the
  // tile shape below. It doesn't break the power of two constraint if there's
  // such constraint on the inputs.
  auto updateThreadBlockShape =
      [&](llvm::MutableArrayRef<int64_t> threadBlockShape,
          llvm::MutableArrayRef<int64_t> threadTileShape,
          ArrayRef<int64_t> ctaBlockingShape, unsigned idx) {
        if (iteratorTypes[idx] == utils::IteratorType::reduction) {
          threadBlockShape[idx] = 1;
          threadTileShape[idx] = ctaBlockingShape[idx];
          return;
        }
        threadBlockShape[idx] = ctaBlockingShape[idx] / threadTileShape[idx];
      };

  // Expand the thread tile shape in order to reduce the total number of
  // threads required by the CTA below the maximum allowed.
  // This loop keeps increasing the threadTileShape. When threadTileShape is
  // larger, the threadBlockShape will be smaller. And the loop stops when
  // threadBlockShape is <= kMaxThreadsPerBlock.
  while (volume<int64_t>(threadBlockShape) > kMaxThreadsPerBlock) {
    DBGV("trying to grow thread tile shape {0} to reduce thread block shape "
         "{1} (volume {2}) below maximum {3}",
         llvm::iterator_range(threadTileShape),
         llvm::iterator_range(threadBlockShape),
         volume<int64_t>(threadBlockShape), kMaxThreadsPerBlock);

    // Increase tile dimensions sizes in a round-robin fashion.
    bool changed = false;
    for (unsigned d = 0; d < threadTileShape.size(); d++) {
      if (ShapedType::isDynamic(loopRanges[d])) {
        threadTileShape[d] = 1;
        continue;
      }
      if (iteratorTypes[d] == utils::IteratorType::reduction)
        continue;
      std::optional<int64_t> nextFactor;
      if (config.getPowerOfTwoTiles)
        nextFactor = getNextPow2Of(threadTileShape[d], loopRanges[d]);
      else
        nextFactor = getNextFactorOf(threadTileShape[d], loopRanges[d]);

      if (nextFactor) {
        threadTileShape[d] = *nextFactor;
        changed = true;

        // This function doesn't break the power of two constraint if there's
        // such constraint on threadTileShape and ctaTileShape.
        updateThreadBlockShape(threadBlockShape, threadTileShape,
                               gridTiling->ctaTileShape, d);

        if (volume<int64_t>(threadBlockShape) <= kMaxThreadsPerBlock)
          break;
      }
    }
    if (changed)
      continue;

    DBGV("failed to grow thread tile shape {0} to reduce thread block shape "
         "{1} below maximum {2} ",
         llvm::iterator_range(threadTileShape),
         llvm::iterator_range(threadBlockShape), kMaxThreadsPerBlock);
    return failure();
  }

  TileShapeSelectionResult result{
      gridTiling->ctaWorkShape, gridTiling->ctaTileShape,
      std::move(threadTileShape), gridTiling->gridShape,
      std::move(threadBlockShape)};

  return result;
}

FailureOr<TileShapeSelectionResult>
mlir::tiling_utils::simpleGpuTileShapeSelection(
    linalg::LinalgOp op, TileShapeSelectionConfig config) {
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  int64_t numDpsInits = op.getNumDpsInits();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  return simpleGpuTileShapeSelection(indexingMaps, iteratorTypes,
                                     op->getOperandTypes(), numDpsInits,
                                     loopRanges, config);
}
