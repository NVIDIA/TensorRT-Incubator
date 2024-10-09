//===- Patterns.h -----------------------------------------------*- C++ -*-===//
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
/// Declarations for clustering rewrite patterns and pattern drivers.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_CLUSTERING_PATTERNS_H
#define MLIR_TENSORRT_TRANSFORMS_CLUSTERING_PATTERNS_H

#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include <limits>
#include <optional>

namespace mlir {

/// Returns an identity OneToNTypeConverter.
OneToNTypeConverter getIdentityTypeConverter();

/// A type of a function that can filter clusters.
using ClusterFilterFn = std::function<bool(const Cluster &)>;

/// A type of a function that determines if the op is already in a clustered
/// region.
using IsOpInClusterRegionFn = std::function<bool(Operation *)>;

/// A default cluster filter the filters cluster by (a) not being all constants
/// and (b) being of minimum size.
ClusterFilterFn getDefaultClusterFilter(unsigned minClusterSize = 1);

/// Given the "main" `func.func` operation, this class describes a base pattern
/// for doing a "cluster-and-outline" transformation.
class ClusteringRewriter {
public:
  explicit ClusteringRewriter(
      ClusteringOpts opts, ClusterRegionOpBuilderFunc regionBuilderFunc,
      IsOpInClusterRegionFn isInClusterRegionFunc,
      ClusterFilterFn clusterFilter = getDefaultClusterFilter(),
      PatternBenefit benefit = PatternBenefit(1));
  ClusteringRewriter(ClusteringOpts opts,
                     ClusterRegionOpBuilderFunc regionBuilderFunc,
                     IsOpInClusterRegionFn isInClusterRegionFunc,
                     PatternBenefit benefit = PatternBenefit(1));

  virtual ~ClusteringRewriter() {}

  FailureOr<SmallVector<Cluster>> findClusters(func::FuncOp mainFunc) {
    return analyzeAndClusterOperations(mainFunc, opts);
  }

  FailureOr<SmallVector<Operation *>>
  findClusterAndCreateRegionOp(func::FuncOp mainFunc, RewriterBase &rewriter);

  const PatternBenefit &getBenefit() const { return benefit; }

  ClusterFilterFn getClusterFilter() const { return clusterFilter; }

  Attribute getTarget() const { return opts.clusterTarget; }

protected:
  /// Clustering options that clustering will be based on
  ClusteringOpts opts;

  /// A function that constructs the "region op" to encapsulate clusters.
  ClusterRegionOpBuilderFunc regionBuilderFunc;

  /// A function that determines if an op is already in a clustered region op.
  IsOpInClusterRegionFn isInClusterRegionFunc;

  /// Function that filters clusters before outlining.
  ClusterFilterFn clusterFilter;

  /// PatternBenefit which determins the clustering order
  PatternBenefit benefit;
};

/// A set of patterns for clustering.
template <typename RewriteType>
class ClusteringPatternSet {
public:
  auto begin() const { return patterns.begin(); }
  auto end() const { return patterns.end(); }
  auto begin() { return patterns.begin(); }
  auto end() { return patterns.end(); }
  template <typename... Args>
  ClusteringPatternSet &add(Args &&...args) {
    std::unique_ptr<RewriteType> pattern =
        std::make_unique<RewriteType>(std::forward<Args>(args)...);
    patterns.emplace_back(std::move(pattern));
    return *this;
  }

private:
  SmallVector<std::unique_ptr<RewriteType>> patterns;
};

/// Apply a set of clustering patterns to the function. Patterns are sorted and
/// applied in decreasing order by benefit.
LogicalResult
applyClusteringPatterns(func::FuncOp mainFunc,
                        ClusteringPatternSet<ClusteringRewriter> &patterns);

/// A type of a function that can filter cluster region operations.
using RegionOpFilterFn = std::function<bool(Operation *)>;

/// Create a cluster region op filter using the specified parameters.
RegionOpFilterFn getRegionOpFilter(
    Attribute target,
    unsigned operationCnt = std::numeric_limits<unsigned>::max(),
    IsClusterableOpFn canOpCluster = [](Operation *op) { return true; });

/// Given a `func.func` operation, this class describes a base
/// pattern for doing a "scf::ExecuteRegion" based transformation
class RegionOpFusionRewriter {
public:
  RegionOpFusionRewriter(const SmallVector<RegionOpFilterFn> &filters,
                         Attribute newTarget,
                         ClusterRegionOpBuilderFunc regionOpBuilderFunc)
      : filters(filters), target(newTarget),
        regionOpBuilderFunc(regionOpBuilderFunc) {}

  /// This function walks on the mainFunc graph and  finds any matched
  /// patterns according to filters. After it finds matched consecutive
  /// Operation* in the graph and it will try to merge them into 1
  /// single Operation* and rewrite it into the graph with a new
  /// clustering target
  void run(func::FuncOp mainFunc, RewriterBase &rewriter);

private:
  /// A list of filter functions that identify scf.execute_region operations of
  /// interest
  SmallVector<RegionOpFilterFn> filters;

  /// the target of the merged region operation will be set to
  Attribute target;

  ClusterRegionOpBuilderFunc regionOpBuilderFunc;
};

/// Return the "target" of a particular cluster represented by the
/// `scf.execute_region` operation. This currently returns the StringAttr
/// named `__cluster_target__` if present, failure otherwise.
FailureOr<Attribute> getClusterTarget(Operation *regionOp);

/// Apply a set of region-op rewriter patterns to the function.
LogicalResult applyRegionOpRewritePatterns(
    func::FuncOp mainFunc,
    ClusteringPatternSet<RegionOpFusionRewriter> &patterns);

} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_CLUSTERING_PATTERNS_H
