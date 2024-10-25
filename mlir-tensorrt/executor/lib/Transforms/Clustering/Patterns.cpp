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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

OneToNTypeConverter mlir::getIdentityTypeConverter() {
  OneToNTypeConverter typeConverter;
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
ClusteringRewriter::findClusterAndCreateRegionOp(func::FuncOp mainFunc,
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

/// Check if two scf::ExecuteRegion Op can merge, assumes producer is
/// producing results used by consumer `previousOps` records all the ops that
/// can be merged into producer, need to check if merging producer into
/// consumer will break the dominance property of `previousOps` users
static bool canMergeRegionOps(Operation *producer, Operation *consumer,
                              ArrayRef<Operation *> opsToMergeIntoProducer) {
  if (producer->getBlock() != consumer->getBlock())
    return false;

  DominanceInfo domInfo(consumer->getParentOp());
  for (Value yieldValue : producer->getResults()) {
    for (Operation *user : yieldValue.getUsers()) {
      // if the user is not a Operation*
      if (user->getBlock() == producer->getBlock() && !isa<Operation *>(user) &&
          domInfo.properlyDominates(user, consumer))
        return false;

      auto userRegionOp = user->getParentOp();
      // if the user is a Operation*
      if (userRegionOp->getBlock() == producer->getBlock() &&
          userRegionOp != consumer &&
          domInfo.properlyDominates(userRegionOp, consumer))
        return false;
    }
  }

  for (auto iter = opsToMergeIntoProducer.begin();
       iter < opsToMergeIntoProducer.end() - 1; ++iter) {
    Operation *prevRegionOp = *iter;
    for (Value yieldValue : prevRegionOp->getResults()) {
      for (Operation *user : yieldValue.getUsers()) {
        // if user is not in a Operation*
        if (user->getBlock() == producer->getBlock() &&
            !isa<Operation *>(user) &&
            domInfo.properlyDominates(user, consumer))
          return false;

        // if the user is in a Operation*
        auto prevUserRegionOp = user->getParentOp();
        if (prevUserRegionOp->getBlock() == consumer->getBlock() &&
            prevUserRegionOp != producer && prevUserRegionOp != consumer &&
            domInfo.properlyDominates(prevUserRegionOp, consumer))
          return false;
      }
    }
  }

  return true;
}

static Operation *mergeRegionOps(Operation *producer, Operation *consumer,
                                 Attribute newTarget, RewriterBase &rewriter,
                                 ClusterRegionOpBuilderFunc createRegionOp) {
  Block *producerBody = &producer->getRegion(0).getBlocks().front();
  Block *consumerBody = &consumer->getRegion(0).getBlocks().front();
  Operation *producerYieldOp = producerBody->getTerminator();

  // In Operation*, we have:

  // %3 = scf.execute_region() {
  // %1 = op(%in0)
  // %2 = op(%in1)
  // scf.yield(%1, %2)
  // }
  // %4 = scf.execute_region() {
  // %5 = op(%3#0)
  // }

  // All operations that consumes producer's result are actually using
  // Operation*'s yield output (%3 here) rather than results of the
  // operations (%1, %2 here) in that Operation*. A mapping from
  // (ExecuteRegionOp's result value) => (the operation result in that block) is
  // required
  IRMapping yieldedValueMap;
  for (unsigned i = 0, e = producerYieldOp->getOperands().size(); i < e; ++i)
    yieldedValueMap.map(producer->getResult(i), producerYieldOp->getOperand(i));

  // some producer's result is only used in consumer while others not, need to
  // yield those values again after merging.

  // This holds all values to yield after merge, notes these values are produced
  // by operations in Operation*
  SmallVector<Value> yieldValuesAfterMerge;
  // This holds all result values that is used by later operations, %3#0 above
  // for instance
  /// TODO: we should rearrange this so that the new yields are ordered by use
  /// order.
  SmallVector<Value> resultValuesAfterMerge;
  for (Value val : producer->getResults()) {
    for (auto user : val.getUsers()) {
      if (user->getParentOp() != consumer) {
        yieldValuesAfterMerge.push_back(yieldedValueMap.lookup(val));
        resultValuesAfterMerge.push_back(val);
      }
    }
  }
  rewriter.setInsertionPointToStart(consumerBody);

  // move all operations from producer to consumer
  rewriter.eraseOp(producerYieldOp);
  rewriter.inlineBlockBefore(producerBody, consumerBody, consumerBody->begin());

  // replace all usages in consumer that consumes producer's output
  for (auto [oldValue, newValue] : yieldedValueMap.getValueMap())
    replaceAllUsesInRegionWith(oldValue, newValue, consumer->getRegion(0));

  // create the new yield op after merge
  Operation *consumerYieldOp = consumerBody->getTerminator();
  for (Value consumerYieldVal : consumerYieldOp->getOperands())
    yieldValuesAfterMerge.push_back(consumerYieldVal);

  rewriter.setInsertionPointToEnd(consumerBody);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(consumerYieldOp,
                                            yieldValuesAfterMerge);

  // create a new merged Operation*
  SmallVector<Type> yieldTypesAfterMerge;
  for (auto &val : yieldValuesAfterMerge)
    yieldTypesAfterMerge.push_back(val.getType());

  rewriter.setInsertionPoint(consumer);
  Operation *mergedOp = createRegionOp(
      rewriter, consumer->getLoc(), TypeRange(yieldTypesAfterMerge), newTarget);

  rewriter.setInsertionPointToStart(&mergedOp->getRegion(0).emplaceBlock());

  Block *mergedBody = &mergedOp->getRegion(0).front();

  rewriter.mergeBlocks(consumerBody, mergedBody);

  // replace all usages of producer/consumer's output with mergedOp's output
  for (auto val : consumer->getResults())
    resultValuesAfterMerge.push_back(val);

  for (auto [oldVal, newVal] :
       llvm::zip(resultValuesAfterMerge, mergedOp->getResults())) {
    rewriter.replaceUsesWithIf(oldVal, newVal, [&](OpOperand &operand) {
      return !mergedOp->isProperAncestor(operand.getOwner());
    });
  }

  rewriter.eraseOp(producer);
  rewriter.eraseOp(consumer);

  return mergedOp;
}

FailureOr<Attribute> mlir::getClusterTarget(Operation *regionOp) {
  if (!regionOp->hasAttr(Cluster::kRegionTargetAttrName))
    return failure();
  return regionOp->getAttr(Cluster::kRegionTargetAttrName);
}

RegionOpFilterFn mlir::getRegionOpFilter(Attribute target,
                                         unsigned operationCnt,
                                         IsClusterableOpFn canOpCluster) {
  return [=](Operation *regionOp) {
    auto curTarget = getClusterTarget(regionOp);
    if (failed(curTarget))
      return false;
    if (curTarget.value() != target)
      return false;

    if (regionOp->getRegion(0).getBlocks().front().getOperations().size() >
        operationCnt)
      return false;

    /// last op is yieldOp, which should be excluded
    Block &body = regionOp->getRegion(0).front();

    for (auto &innerOp : body.without_terminator()) {
      if (!canOpCluster(&innerOp)) {
        return false;
      }
    }
    return true;
  };
}

void RegionOpFusionRewriter::run(func::FuncOp mainFunc,
                                 RewriterBase &rewriter) {
  /// walk on the graph and find any pieces of nodes in the graph which
  /// matches to the matcher

  /// Note here we cannot modify/erase the regionOps directly in walking
  /// process, for example, if we are going to merge A=>B=>C, merging them when
  /// we are visiting A would lead to segmentation fault since B and C are
  /// visited later which are actually erased. A vector opsToMerge is used to
  /// address this issue, which records all the ops that we are going to merge
  SmallVector<SmallVector<Operation *>> opsToMerge;

  DenseSet<Operation *> alreadyInMerge;
  auto funcWalkResult = mainFunc->walk([&](Operation *regionOp) {
    if (alreadyInMerge.contains(regionOp) || filters.empty() ||
        !filters.front()(regionOp))
      return WalkResult::advance();

    auto curFilterIter = filters.begin();

    Operation *curRegionOp = regionOp;
    SmallVector<Operation *> stack{curRegionOp};

    // use DFS here to find matched pattern
    while (!stack.empty() && stack.size() < filters.size()) {
      DenseSet<Operation *> visitedRegionConsumers;

      bool userMatch = false;

      for (Operation *user : curRegionOp->getUsers()) {
        /// users are operations in another Operation*, so need
        /// to check their parents
        auto nextRegionOp =
            llvm::dyn_cast_or_null<Operation *>(user->getParentOp());

        if (!nextRegionOp || nextRegionOp->getBlock() != regionOp->getBlock() ||
            visitedRegionConsumers.contains(nextRegionOp))
          continue;

        visitedRegionConsumers.insert(nextRegionOp);

        /// A->B or B->C and be merged doesn't necessary mean
        /// A->B->C can be merged into 1
        if ((*(curFilterIter + 1))(nextRegionOp) &&
            canMergeRegionOps(curRegionOp, nextRegionOp, stack)) {
          stack.push_back(nextRegionOp);
          curRegionOp = nextRegionOp;
          userMatch = true;
          break;
        }
      }

      if (userMatch) {
        ++curFilterIter;
        continue;
      }

      curRegionOp = stack.back();
      stack.pop_back();
      --curFilterIter;
    }

    // if find matched pattern, merge those ops
    if (stack.size() == filters.size()) {
      opsToMerge.push_back(stack);
      alreadyInMerge.insert(stack.begin(), stack.end());
    }

    return WalkResult::advance();
  });

  for (auto ops : opsToMerge) {
    Operation *producer = ops.front();

    LLVM_DEBUG(DBGS() << "merge following Operation* into 1 "
                         "Operation* : \n");
    LLVM_DEBUG(DBGS() << "  --  " << producer << "\n");
    for (auto iter = ops.begin() + 1; iter != ops.end(); ++iter) {
      LLVM_DEBUG(DBGS() << "  --  " << *iter << "\n");
      producer = mergeRegionOps(producer, *iter, target, rewriter,
                                regionOpBuilderFunc);
    }
    LLVM_DEBUG(DBGS() << "  --  New merged Operation* has target: " << target
                      << "\n");
  }

  if (funcWalkResult.wasInterrupted())
    emitError(mainFunc->getLoc())
        << "failed to merge some consecutive Operation* into one";
}

/// Apply a set of clustering patterns to the function.
LogicalResult mlir::applyClusteringPatterns(
    func::FuncOp mainFunc, ClusteringPatternSet<ClusteringRewriter> &patterns) {
  llvm::sort(patterns, [](const std::unique_ptr<ClusteringRewriter> &lhs,
                          const std::unique_ptr<ClusteringRewriter> &rhs) {
    return lhs->getBenefit() > rhs->getBenefit();
  });
  // Execute.
  IRRewriter rewriter(mainFunc->getContext());

  for (const std::unique_ptr<ClusteringRewriter> &rewrite : patterns) {
    FailureOr<SmallVector<Operation *>> regionOps =
        rewrite->findClusterAndCreateRegionOp(mainFunc, rewriter);

    if (failed(regionOps))
      return emitError(mainFunc.getLoc())
             << "clustering rewrite " << rewrite->getTarget() << " failed ";

    LLVM_DEBUG(DBGS() << "clustering pattern created " << regionOps->size()
                      << " Operation* with tag " << rewrite->getTarget()
                      << " (benefit=" << rewrite->getBenefit().getBenefit()
                      << ")\n");
  }
  LLVM_DEBUG(DBGS() << "After clustering and Operation* creation: " << mainFunc
                    << "\n");

  return success();
}

LogicalResult mlir::applyRegionOpRewritePatterns(
    func::FuncOp mainFunc,
    ClusteringPatternSet<RegionOpFusionRewriter> &patterns) {
  IRRewriter rewriter(mainFunc->getContext());

  for (const std::unique_ptr<RegionOpFusionRewriter> &rewrite : patterns) {
    rewrite->run(mainFunc, rewriter);
  }

  return success();
}
