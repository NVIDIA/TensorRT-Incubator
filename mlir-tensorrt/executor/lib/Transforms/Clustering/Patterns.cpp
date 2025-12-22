//===- Patterns.cpp -------------------------------------------------------===//
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
/// Implementation of clustering patterns and clustering pattern driver.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "clustering-patterns"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

using namespace mlir;

ClusterFilterFn mlir::getDefaultClusterFilter(unsigned minClusterSize) {
  return [minClusterSize](Cluster cluster) {
    // Disregard the cluster if it is all constant ops.
    if (std::all_of(cluster.begin(), cluster.end(), [](Operation *op) {
          return op->hasTrait<OpTrait::ConstantLike>();
        }))
      return false;
    return cluster.size() >= minClusterSize;
  };
}

TypeConverter mlir::getIdentityTypeConverter() {
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type t) { return t; });
  return typeConverter;
}

ClusteringRewriter::ClusteringRewriter(
    ClusteringOpts opts, ClusterRegionOpBuilderFunc regionBuilderFunc,
    IsOpInClusterRegionFn isInClusterRegionFunc, ClusterFilterFn clusterFilter,
    PatternBenefit benefit)
    : opts(opts), regionBuilderFunc(std::move(regionBuilderFunc)),
      isInClusterRegionFunc(std::move(isInClusterRegionFunc)),
      clusterFilter(std::move(clusterFilter)), benefit(benefit) {}

ClusteringRewriter::ClusteringRewriter(
    ClusteringOpts opts, ClusterRegionOpBuilderFunc regionBuilderFunc,
    IsOpInClusterRegionFn isInClusterRegionFunc, PatternBenefit benefit)
    : opts(opts), regionBuilderFunc(std::move(regionBuilderFunc)),
      isInClusterRegionFunc(std::move(isInClusterRegionFunc)),
      clusterFilter(getDefaultClusterFilter(1)), benefit(benefit) {}

FailureOr<SmallVector<Operation *>>
ClusteringRewriter::findClusterAndCreateRegionOp(FunctionOpInterface mainFunc,
                                                 RewriterBase &rewriter) {
  std::function<bool(Operation *)> isClusterableOp =
      std::move(opts.isClusterableOp);
  opts.isClusterableOp = [&](Operation *op) {
    return !isInClusterRegionFunc(op) && isClusterableOp(op);
  };
  FailureOr<SmallVector<Cluster>> clusters =
      analyzeAndClusterOperations(mainFunc, opts);
  if (failed(clusters))
    return failure();
  LLVM_DEBUG(DBGS() << "num clusters before filtering: " << clusters->size()
                    << "\n");
  SmallVector<Cluster> filteredClusters =
      llvm::to_vector(llvm::make_filter_range(*clusters, clusterFilter));
  LLVM_DEBUG(DBGS() << "num clusters after filtering: "
                    << filteredClusters.size() << "\n");

  SmallVector<Operation *> result;
  result.reserve(filteredClusters.size());
  for (const Cluster &cluster : filteredClusters) {
    auto regionOp = cast<Operation *>(
        createRegionOpFromCluster(cluster, rewriter, regionBuilderFunc));
    if (!regionOp)
      return mainFunc->emitError("failed to create Operation* from cluster");
    result.push_back(regionOp);
  }
  return result;
}
