
//===- Clustering.cpp -------------------------------------------*- C++ -*-===//
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
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include <algorithm>
#include <queue>

#define DEBUG_TYPE "clustering"
#define DBGS() (llvm::dbgs() << "// [" DEBUG_TYPE "]: ")

using namespace mlir;

//===----------------------------------------------------------------------===//
// ClusteringState
//===----------------------------------------------------------------------===//

/// Print out the cluster by printing out all Operation* pointers belonging
/// to the cluster.
// void debugPrintClus
static void debugPrintCluster(const ClusteringState &state, Operation *op,
                              llvm::raw_ostream &os) {
  unsigned idx = 0;
  state.runOnEquivalenceClass(op, [&](Operation *member, Operation *) {
    os << " - " << idx++ << ": " << *member << "\n";
  });
}

/// Check if merging `src_root` cluster into `dst_root` cluster will obey
/// dominance property. Here `src_root` is before `dst_root`, so what this
/// function does is to check if all users of `src_root` cluster are below
/// `dst_root`
static LogicalResult obeyDominanceProperty(ClusteringState &state,
                                           Operation *srcRoot,
                                           Operation *dstRoot) {
  assert(srcRoot->getBlock() == dstRoot->getBlock() &&
         "expected src and dst to be in the same block");
  const DominanceInfo &domInfo = state.domInfo;
  llvm::EquivalenceClasses<Operation *> &ec = state.ec;
  bool obeysDominanceResult = true;
  state.runOnEquivalenceClass(srcRoot, [&](Operation *op, Operation *root) {
    for (Operation *user : op->getUsers()) {
      // Doesn't violate dominance property if the user is in either one of the
      // 2 clusters once they are merged into single one cluster
      llvm::EquivalenceClasses<Operation *>::member_iterator userRootIt =
          ec.findLeader(user);
      if (userRootIt != ec.member_end() &&
          (*userRootIt == srcRoot || *userRootIt == dstRoot))
        continue;

      // If the user is in a different cluster whose root is dominated by the
      // destination cluster root, then we don't have to worrry about creating a
      // cycle since the user will be moved below all ops in the dest cluster.
      if (userRootIt != ec.member_end() &&
          domInfo.properlyDominates(dstRoot, *userRootIt))
        continue;

      // TODO: the below condition below assumes that the insertion point of the
      // cluster will be at the root (rather than at the position of the first
      // user).

      // If the user is in the same block as src/dst, then merging will create a
      // cycle if the user is before the last operation in the `dst` cluster,
      // which is the `dst` root.
      // If the user in in a different block, there there is no issue if the dst
      // root dominates that block.
      if (!domInfo.properlyDominates(dstRoot, user)) {
        LLVM_DEBUG(DBGS() << "  --  " << *dstRoot << "\n";
                   DBGS() << "  --  does not dominate\n";
                   DBGS() << "  --  " << *user << "\n");
        obeysDominanceResult = false;
        break;
      }
    }
  });
  return success(obeysDominanceResult);
}

ClusteringState::ClusteringState(Operation *op, ClusteringOpts opts)
    : opts(std::move(opts)), domInfo(op) {}

void ClusteringState::runOnEquivalenceClass(
    Operation *op,
    llvm::function_ref<void(Operation *member, Operation *leader)> func) const {
  auto leaderIt = ec.findLeader(op);
  for (auto mi = leaderIt, end = ec.member_end(); mi != end; ++mi)
    func(*mi, *leaderIt);
}

bool ClusteringState::contains(Operation *op) const {
  return ec.findLeader(op) != ec.member_end();
}

void ClusteringState::addCluster(Operation *op) {
  assert(!contains(op) && "tried to insert re-insert an operation");
  ec.insert(op);
}

bool ClusteringState::canUnionClusters(Operation *x, Operation *y) {
  Operation *xRootOp = ec.getLeaderValue(x);
  Operation *yRootOp = ec.getLeaderValue(y);

  // If the roots are in different blocks, we cannot cluster. If they are the
  // same, we cannot do anything.
  if (xRootOp == yRootOp || xRootOp->getBlock() != yRootOp->getBlock())
    return false;

  bool xBeforeY = xRootOp->isBeforeInBlock(yRootOp);

  Operation *producerRoot = xBeforeY ? xRootOp : yRootOp;
  Operation *consumerRoot = xBeforeY ? yRootOp : xRootOp;

  LLVM_DEBUG({
    DBGS() << "checking whether can union cluster:\n";
    debugPrintCluster(*this, x, llvm::dbgs());
    llvm::dbgs() << "with cluster:\n";
    debugPrintCluster(*this, y, llvm::dbgs());
  });

  // Check if merging `src_root` to `dst_root` will obey SSA dominance property
  return succeeded(obeyDominanceProperty(*this, producerRoot, consumerRoot));
}

LogicalResult ClusteringState::unionClusters(Operation *x, Operation *y) {
  if (!canUnionClusters(x, y))
    return failure();
  Operation *xRootOp = ec.getLeaderValue(x);
  Operation *yRootOp = ec.getLeaderValue(y);
  bool xBeforeY = xRootOp->isBeforeInBlock(yRootOp);
  Operation *producerRoot = xBeforeY ? xRootOp : yRootOp;
  Operation *consumerRoot = xBeforeY ? yRootOp : xRootOp;
  ec.unionSets(consumerRoot, producerRoot);
  return success();
}

/// Find the region that is the closest common ancestor between the two ops.
static Region *findCommonAncestor(Operation *lhs, Operation *rhs) {
  Region *region = lhs->getParentRegion();
  while (region) {
    if (region->findAncestorOpInRegion(*rhs))
      break;
    region = region->getParentRegion();
  }
  return region;
}

static bool compareOps(Operation *lhs, Operation *rhs,
                       const DominanceInfo &domInfo) {
  bool lhsDomRhs = domInfo.properlyDominates(lhs, rhs);
  bool rhsDomLhs = domInfo.properlyDominates(rhs, lhs);
  if (lhsDomRhs ^ rhsDomLhs)
    return lhsDomRhs;

  Region *common = findCommonAncestor(lhs, rhs);
  assert(common && "expected common ancestor");

  SmallVector<Block *> blocks;
  common->walk<WalkOrder::PreOrder>(
      [&](Block *block) { blocks.push_back(block); });
  return llvm::find(blocks, lhs->getBlock()) <
         llvm::find(blocks, rhs->getBlock());
}

static SmallVector<Operation *> getSortedTrackedOperations(
    const ClusteringState &state,
    ClusteringRootTraversalDirection rootTraversalDirection,
    bool rootOnly = false) {
  const llvm::EquivalenceClasses<Operation *> &ec = state.ec;
  SmallVector<Operation *> ops;
  for (llvm::EquivalenceClasses<Operation *>::iterator it = ec.begin();
       it != ec.end(); ++it) {
    if (rootOnly && !it->isLeader())
      continue;
    ops.push_back(it->getData());
  }
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    bool lhsBefore = compareOps(lhs, rhs, state.domInfo);
    return rootTraversalDirection == ClusteringRootTraversalDirection::PreOrder
               ? lhsBefore
               : !lhsBefore;
  });
  return ops;
}

SmallVector<Operation *> ClusteringState::getClusterRoots(bool sorted) const {
  return getSortedTrackedOperations(
      *this, ClusteringRootTraversalDirection::PreOrder, /*rootOnly=*/true);
}

SmallVector<Cluster> ClusteringState::getClusters() const {
  SmallVector<Cluster> clusters;
  SmallVector<Operation *> roots = getClusterRoots(/*sorted=*/true);
  for (Operation *root : roots) {
    clusters.emplace_back(
        llvm::make_range(ec.findLeader(root), ec.member_end()),
        opts.clusterTarget);
    Cluster &cluster = clusters.back();
    // Topological sort won't handle disconnected cases correctly. Since a
    // cluster must all be in the same block, we can just sort them this way.
    llvm::sort(cluster, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
  }
  return clusters;
}

//===----------------------------------------------------------------------===//
// Clustering methods
//===----------------------------------------------------------------------===//

void mlir::runBFSClustering(
    ClusteringState &state, ShouldGrowClusterFn shouldGrowClusterFn,
    ClusteringRootTraversalDirection rootTraversalDirection) {
  llvm::EquivalenceClasses<Operation *> &ec = state.ec;
  SmallVector<Operation *> ops = getSortedTrackedOperations(
      state, rootTraversalDirection, /*rootOnly=*/false);

  for (Operation *op : ops) {
    std::queue<Operation *> opQueue;
    opQueue.push(op);
    llvm::DenseSet<Operation *> visitedOp{op};

    while (!opQueue.empty()) {
      Operation *curOp = opQueue.front();
      opQueue.pop();
      for (Operation *consumer : curOp->getUsers()) {
        if (ec.findLeader(consumer) == ec.member_end() ||
            visitedOp.contains(consumer))
          continue;
        visitedOp.insert(consumer);
        if (shouldGrowClusterFn &&
            !shouldGrowClusterFn(
                curOp, llvm::make_range(ec.findLeader(curOp), ec.member_end()),
                consumer,
                llvm::make_range(ec.findLeader(consumer), ec.member_end())))
          continue;
        if (succeeded(state.unionClusters(curOp, consumer)))
          opQueue.push(consumer);
      }
    }
  }
}

void mlir::mergeIndependentClusters(
    ClusteringState &state,
    ShouldMergeIndependentClustersFn shouldTryMergeClusters) {
  assert(shouldTryMergeClusters &&
         "expected valid function shouldTryMergeClusters");
  // This function loops over all clusters and tries to merges them, then does
  // it again until a fixed point is reached. In certain situations, one trip
  // through all pairs won't merge all clusters (ie. after trip through all
  // pairs of clusters, the changes made will open up new merging
  // opportunities), which is why we need to do it repeatedly.
  bool numClustersChanged = true;
  while (numClustersChanged) {
    numClustersChanged = false;
    SmallVector<Operation *> allRoots = state.getClusterRoots(/*sorted=*/true);
    LLVM_DEBUG(DBGS() << "[Target: " << state.opts.clusterTarget
                      << "] Attempting to merge all pairs of clusters\n");
    for (size_t i = 0; i < allRoots.size(); ++i) {
      // Roots may have changed in previous iteration.
      if (state.ec.getLeaderValue(allRoots[i]) != allRoots[i])
        continue;
      for (size_t j = i + 1; j < allRoots.size(); ++j) {
        // Roots may have changed in previous iteration.
        if (state.ec.getLeaderValue(allRoots[j]) != allRoots[j])
          continue;
        if (shouldTryMergeClusters &&
            !shouldTryMergeClusters(
                allRoots[i],
                llvm::make_range(state.ec.findLeader(allRoots[i]),
                                 state.ec.member_end()),
                allRoots[j],
                llvm::make_range(state.ec.findLeader(allRoots[j]),
                                 state.ec.member_end())))
          continue;
        if (succeeded(state.unionClusters(allRoots[i], allRoots[j]))) {
          numClustersChanged = true;
          LLVM_DEBUG(DBGS()
                     << "[Target: " << state.opts.clusterTarget
                     << "] combined clusters " << i << ", " << j << "\n");
        }
      }
    }
  }
}

void mlir::annotateClustersWithClusterIdAttribute(
    const ClusteringState &state) {
  for (const auto &[idx, op] :
       llvm::enumerate(state.getClusterRoots(/*sorted=*/false))) {
    MLIRContext *ctx = op->getContext();
    state.runOnEquivalenceClass(op, [&](Operation *member, Operation *leader) {
      member->setAttr("cluster.root",
                      IntegerAttr::get(IntegerType::get(ctx, 64),
                                       reinterpret_cast<intptr_t>(leader)));
    });
  }
}

FailureOr<SmallVector<Cluster>>
mlir::analyzeAndClusterOperations(ClusteringState &clusterer) {
  const ClusteringOpts &opts = clusterer.opts;

  // If there are no initial clusters, then there is nothing to do.
  if (std::distance(clusterer.ec.begin(), clusterer.ec.end()) == 0)
    return SmallVector<Cluster>{};

  // Get the module for debug printing.
  ModuleOp op = (*clusterer.ec.begin()).getData()->getParentOfType<ModuleOp>();

  LLVM_DEBUG({
    annotateClustersWithClusterIdAttribute(clusterer);
    DBGS() << "[Target: " << opts.clusterTarget << "] After ID assignment:\n"
           << *op << "\n";
  });

  // apply BFS to find connected clusterable ops and union them
  runBFSClustering(clusterer, opts.shouldGrowClusterFn, opts.bfsRootTraversal);

  LLVM_DEBUG({
    annotateClustersWithClusterIdAttribute(clusterer);
    DBGS() << "[Target: " << opts.clusterTarget
           << "] After running initial BFS clustering:\n"
           << *op << "\n";
  });
  if (opts.mergeIndependentClusters)
    mergeIndependentClusters(clusterer, opts.mergeIndependentClusters);

  LLVM_DEBUG({
    annotateClustersWithClusterIdAttribute(clusterer);
    DBGS() << "[Target: " << opts.clusterTarget
           << "] After merging independent clusters:\n"
           << *op << "\n";
  });

  // Update root attributes to be all the same in 1 cluster
  SmallVector<Cluster> result = clusterer.getClusters();
  return result;
}

void mlir::populateSizeOneClusters(ClusteringState &state,
                                   Operation *toppLevelOp,
                                   IsClusterableOpFn isClusterableOp) {
  // set cluster_op_id and cluster_root_id attribute to each clusterable op
  toppLevelOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == toppLevelOp)
      return WalkResult::advance();
    if (state.contains(op))
      return WalkResult::skip();
    if (isClusterableOp(op)) {
      state.addCluster(op);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

FailureOr<SmallVector<Cluster>>
mlir::analyzeAndClusterOperations(Operation *op, const ClusteringOpts &opts) {
  ClusteringState state(op, opts);
  LLVM_DEBUG(
      DBGS()
      << "[Target: " << opts.clusterTarget
      << "] Initializing all clusterable operations to single-op clusters.\n");

  populateSizeOneClusters(state, op, opts.isClusterableOp);

  return analyzeAndClusterOperations(state);
}

Operation *
mlir::createRegionOpFromCluster(const Cluster &cluster, RewriterBase &rewriter,
                                ClusterRegionOpBuilderFunc createRegionOp) {
  // insert the region to the last Op to because of dominance property
  assert(cluster.getTarget() && "expected a valid cluster target attribute");
  Operation *insertionOp = cluster.getRoot();

  // find all the values that are used outside of the cluster. These values
  // will be yield from the created `scf.execute_region`
  SetVector<Value> yieldValues;
  SmallVector<Type> yieldTypes;
  DenseSet<Operation *> clusterOpSet;
  for (Operation *op : cluster)
    clusterOpSet.insert(op);

  for (Operation *op : cluster) {
    for (OpOperand &use : op->getUses()) {
      // skip if the user is also in the cluster
      if (clusterOpSet.contains(use.getOwner()))
        continue;
      // skip if the value is already in yield value set
      if (yieldValues.contains(use.get()))
        continue;

      yieldValues.insert(use.get());
      yieldTypes.emplace_back(use.get().getType());
    }
  }

  rewriter.setInsertionPoint(insertionOp);
  Operation *regionOp = createRegionOp(rewriter, insertionOp->getLoc(),
                                       yieldTypes, cluster.getTarget());
  assert(regionOp->getRegion(0).getBlocks().size() == 1 &&
         "expected single-block region");
  Operation *term = regionOp->getRegion(0).front().getTerminator();
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&regionOp->getRegion(0).front());

    // Move in ops in order. We do not need to remap operands since
    // we are unlinking the ops from the original Block and moving them into
    // the Block of the region body.
    for (Operation *opInCluster : cluster)
      opInCluster->moveBefore(term);

    // Add the terminator.
    term->setOperands(yieldValues.getArrayRef());
  }

  // Replace uses of the operations outside of the region op body.
  for (auto [oldVal, newVal] : llvm::zip(yieldValues, regionOp->getResults())) {
    rewriter.replaceUsesWithIf(oldVal, newVal, [&](OpOperand &operand) {
      return !regionOp->isProperAncestor(operand.getOwner());
    });
  }
  regionOp->setAttr(Cluster::kRegionTargetAttrName, cluster.getTarget());

  return regionOp;
}

//===----------------------------------------------------------------------===//
// Region Outlining to Function Utilities
//===----------------------------------------------------------------------===//

/// Create a `func.func` operation that has the specified arg/result types and
/// insert it into the moduleSymbolTable.
static FunctionOpInterface
createOutlinedFunc(RewriterBase &rewriter, Location loc,
                   SymbolTable &moduleSymbolTable, StringRef nameBase,
                   TypeRange funcArgTypes, TypeRange funcResultTypes) {
  OpBuilder::InsertionGuard g(rewriter);

  // Create the func for outlining the region body.
  FunctionType type =
      FunctionType::get(rewriter.getContext(), funcArgTypes, funcResultTypes);
  auto outlinedFunc = mlir::func::FuncOp::create(loc, nameBase, type, {});
  Block *funcBody = outlinedFunc.addEntryBlock();

  // Add an empty terminator.
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(loc);

  // Insert into the module.
  moduleSymbolTable.insert(outlinedFunc);

  return cast<FunctionOpInterface>(outlinedFunc.getOperation());
}

OutlineRegionOptions::CreateFuncAndCallStubsFunc
OutlineRegionOptions::getDefaultCreateFuncAndCallStubFunc(
    SymbolTable &moduleSymbolTable, ArrayRef<NamedAttribute> extraFuncAttrs,
    StringRef namePrefix) {
  // Create the func.func
  std::string prefixStr = namePrefix.str();
  std::vector<NamedAttribute> extraFuncAttrsCopy(extraFuncAttrs);
  return [prefixStr, &moduleSymbolTable, extraFuncAttrs = extraFuncAttrsCopy](
             RewriterBase &rewriter, Location loc, Operation *regionOp,
             ArrayRef<Value> callOperands, ArrayRef<Type> convertedOperandTypes,
             ArrayRef<Type> results)
             -> FailureOr<std::pair<FunctionOpInterface, SmallVector<Value>>> {
    FunctionOpInterface func =
        createOutlinedFunc(rewriter, loc, moduleSymbolTable, prefixStr,
                           convertedOperandTypes, results);
    func.setPrivate();

    for (const NamedAttribute &attr : extraFuncAttrs)
      func->setAttr(attr.getName(), attr.getValue());

    rewriter.setInsertionPoint(regionOp);
    SmallVector<Value> callReplacements =
        rewriter
            .create<func::CallOp>(loc, llvm::cast<func::FuncOp>(*func),
                                  callOperands)
            .getResults();
    return std::make_pair(func, callReplacements);
  };
}

/// Given the `args` that have "source" types (which should each be converted to
/// N types), materialize the converted values and return them.
static FailureOr<SmallVector<Value>>
convertSourceToTarget(RewriterBase &rewriter,
                      const OneToNTypeConverter &typeConverter,
                      ValueRange args) {
  SmallVector<Value> returnedValues;
  for (Value originalValue : args) {
    SmallVector<Type> convertedTypes;
    if (failed(
            typeConverter.convertType(originalValue.getType(), convertedTypes)))
      return failure();
    // If no conversion required, just append the value.
    if (convertedTypes.size() == 1 &&
        convertedTypes.front() == originalValue.getType()) {
      returnedValues.push_back(originalValue);
      continue;
    }
    // Otherwise, call target materialization.
    std::optional<SmallVector<Value>> materialized =
        typeConverter.materializeTargetConversion(
            rewriter, originalValue.getLoc(), convertedTypes, originalValue);
    if (!materialized)
      return failure();
    returnedValues.append(*materialized);
  }
  return returnedValues;
}

using SignatureConversion = TypeConverter::SignatureConversion;

/// Given the `targetTypedResults` of size N, we need to convert them to a total
/// of M <= N source-typed values. This function materializes the correct source
/// values by invoking the `materializeSourceConversion` function. A populated
/// `SignatureConversion` must be provided to map the target types back to
/// source types.
static FailureOr<SmallVector<Value>>
convertTargetToSource(RewriterBase &rewriter, ValueRange targetTypedResults,
                      const OneToNTypeConverter &typeConverter,
                      TypeRange srcTypes,
                      const SignatureConversion &sigConversion) {
  SmallVector<Value> replacements;
  for (auto [idx, originalType] : llvm::enumerate(srcTypes)) {
    std::optional<SignatureConversion::InputMapping> inputMapping =
        sigConversion.getInputMapping(idx);
    if (!inputMapping)
      return failure();
    ValueRange valsToConvert =
        targetTypedResults.slice(inputMapping->inputNo, inputMapping->size);
    // If no conversion required, just append the value.
    if (valsToConvert.size() == 1 &&
        valsToConvert.front().getType() == originalType) {
      replacements.push_back(valsToConvert.front());
      continue;
    }
    // Otherwise, call the source materialization.
    Value materialized = typeConverter.materializeSourceConversion(
        rewriter, valsToConvert.front().getLoc(), originalType, valsToConvert);
    if (!materialized)
      return failure();
    replacements.push_back(materialized);
  }
  return replacements;
}

static LogicalResult getUsedValuesDefinedAboveOrClone(
    RewriterBase &rewriter, Region &body, SetVector<Value> &operands,
    std::function<bool(Value, Region &)> shouldCloneProducer) {
  if (body.getBlocks().size() != 1)
    return failure();
  OpBuilder::InsertionGuard g(rewriter);
  mlir::getUsedValuesDefinedAbove(body, operands);

  SetVector<Value> keptOperands;
  for (Value v : operands) {
    Operation *producer = v.getDefiningOp();
    if (!producer || producer->getNumOperands() > 0 ||
        !shouldCloneProducer(v, body)) {
      keptOperands.insert(v);
      continue;
    }
    rewriter.setInsertionPointToStart(&body.front());
    Operation *clonedProducer = rewriter.clone(*producer);
    rewriter.replaceOpUsesWithinBlock(producer, clonedProducer->getResults(),
                                      &body.front());
  }
  std::swap(keptOperands, operands);
  return success();
}

FailureOr<std::pair<FunctionOpInterface, SetVector<Value>>>
mlir::outlineRegionOp(RewriterBase &rewriter, Operation *op,
                      OutlineRegionOptions &opts) {

  using SignatureConversion = TypeConverter::SignatureConversion;
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = op->getLoc();

  SetVector<Value> operands;
  Region &body = op->getRegion(0);
  if (failed(getUsedValuesDefinedAboveOrClone(rewriter, body, operands,
                                              opts.shouldCloneProducer)))
    return failure();

  // Create the signature conversion and the types of the results.
  SignatureConversion sigConverter(operands.size());
  if (failed(opts.typeConverter.convertSignatureArgs(
          TypeRange(operands.getArrayRef()), sigConverter)))
    return failure();
  SmallVector<Type> convertedResultTypes;
  if (failed(opts.typeConverter.convertTypes(op->getResultTypes(),
                                             convertedResultTypes)))
    return failure();

  // Materialize the converted call operands if required.
  rewriter.setInsertionPoint(op);
  FailureOr<SmallVector<Value>> convertedOperands = convertSourceToTarget(
      rewriter, opts.typeConverter, operands.getArrayRef());
  if (failed(convertedOperands))
    return failure();

  // Call user callback to get func and call op.
  FailureOr<std::pair<FunctionOpInterface, SmallVector<Value>>> callbackResult =
      opts.createFunc(rewriter, loc, op, *convertedOperands,
                      sigConverter.getConvertedTypes(), convertedResultTypes);
  if (failed(callbackResult))
    return failure();

  auto [outlinedFunc, callResults] = *callbackResult;
  assert(outlinedFunc.getFunctionBody().getBlocks().size() == 1 &&
         "expected body with one block");
  Block *outlinedFuncBlock = &outlinedFunc.getFunctionBody().front();
  assert(outlinedFuncBlock->getOperations().size() == 1 &&
         "expected function body block to contain empty terminator");

  // Populate the function entry block.
  {
    // Create remapped operands.
    rewriter.setInsertionPointToStart(outlinedFuncBlock);
    FailureOr<SmallVector<Value>> remappedArgs = convertTargetToSource(
        rewriter, outlinedFuncBlock->getArguments(), opts.typeConverter,
        TypeRange(operands.getArrayRef()), sigConverter);
    if (failed(remappedArgs))
      return failure();
    for (auto [orig, replacement] : llvm::zip(operands, *remappedArgs))
      mlir::replaceAllUsesInRegionWith(orig, replacement, body);

    // Move region op operations to the func body.
    Operation *regionYieldOp = body.front().getTerminator();
    rewriter.inlineBlockBefore(&body.front(),
                               outlinedFuncBlock->getTerminator());

    // Convert the yielded values to the required type and update terminator.
    rewriter.setInsertionPoint(regionYieldOp);
    FailureOr<SmallVector<Value>> returnedValues = convertSourceToTarget(
        rewriter, opts.typeConverter, regionYieldOp->getOperands());
    if (failed(returnedValues))
      return failure();
    outlinedFuncBlock->getTerminator()->setOperands(*returnedValues);
    rewriter.eraseOp(regionYieldOp);
  }

  // Create the call operation.
  {
    rewriter.setInsertionPointAfter(op);
    SignatureConversion resultTypeMapping(op->getNumResults());
    if (failed(opts.typeConverter.computeTypeMapping(op->getResultTypes(),
                                                     resultTypeMapping)))
      return failure();

    FailureOr<SmallVector<Value>> replacements =
        convertTargetToSource(rewriter, callResults, opts.typeConverter,
                              op->getResultTypes(), resultTypeMapping);
    if (failed(replacements))
      return failure();

    rewriter.replaceOp(op, *replacements);
  }

  return std::pair<FunctionOpInterface, SetVector<Value>>{outlinedFunc,
                                                          operands};
}
