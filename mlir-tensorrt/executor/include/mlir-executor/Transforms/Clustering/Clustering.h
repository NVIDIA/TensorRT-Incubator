//===- Clustering.h ---------------------------------------------*- C++ -*-===//
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
/// Generic clustering and analysis.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_CLUSTERING_CLUSTERING_H
#define MLIR_TENSORRT_TRANSFORMS_CLUSTERING_CLUSTERING_H

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {

class OneToNTypeConverter;

/// A callable type that determines whether the operation is clusterable.
using IsClusterableOpFn = std::function<bool(Operation *)>;

using ClusterIterator = llvm::EquivalenceClasses<Operation *>::member_iterator;
using ClusterRange = llvm::iterator_range<ClusterIterator>;

/// A type for functions that determines whether to union two clusters during
/// BFS clustering.
using ShouldGrowClusterFn =
    std::function<bool(Operation *producer, ClusterRange producerCluster,
                       Operation *consumer, ClusterRange consumerCluster)>;

/// A type for functions that determine whether to try to combine two clusters
/// not connected by a producer-consumer relationship. This does not need to
/// test dependence criteria since that is handled by the caller.
using ShouldMergeIndependentClustersFn =
    std::function<bool(Operation *lhsRoot, ClusterRange lhs, Operation *rhsRoot,
                       ClusterRange rhs)>;

/// An enum indicating whether clusters should be traversed in 'pre'
/// (producer-to-consumer, forward iteration over ops in Block) or 'post'
/// (consumer-to-producer, reverse iteration over block). This is used to
/// determine iteration order in certain algorithms like BFS clustering.
enum class ClusteringRootTraversalDirection { PreOrder, PostOrder };

/// Encapsulates options required for determining how to cluster a group of
/// operations.
struct ClusteringOpts {
  /// Optional callback is used to determine whether an operation is of interest
  /// for the type of clusters being created.
  IsClusterableOpFn isClusterableOp{nullptr};

  /// Optional callback that determines whether to union two clusters during
  /// the depth-wise cluster extension step.
  ShouldGrowClusterFn shouldGrowClusterFn{nullptr};

  /// Whether to merge independent clusters (as long as possible without
  /// creating cycles). It is always safe to return either true or false here
  /// without testing dependence criteria since the caller will always check
  /// that a cycle is not created before attempting to union the clusters.
  ShouldMergeIndependentClustersFn mergeIndependentClusters =
      [](Operation *lhsRoot, ClusterRange lhs, Operation *rhsRoot,
         ClusterRange rhs) { return true; };

  // Specifies the translation target for this cluster
  Attribute clusterTarget;

  /// Determines the order in which clusters are grown during BFS clustering.
  ClusteringRootTraversalDirection bfsRootTraversal =
      ClusteringRootTraversalDirection::PreOrder;
};

/// A cluster is a vector of connected operations that can legally be outlined
/// to a function without doing any reordering of the IR other than inserting
/// the "call" operation at the position of the last operation in the vector.
/// The cluster may use values defined outside the cluster. Results of any of
/// the operations may or may not be used outside of the cluster.
class Cluster {
public:
  Cluster(Operation *op, Attribute target) : members{op}, target(target) {}

  template <typename Range>
  Cluster(Range &&r, Attribute target)
      : members(llvm::to_vector(std::forward<Range>(r))), target(target) {}

  auto begin() const { return members.begin(); }
  auto end() const { return members.end(); }
  auto begin() { return members.begin(); }
  auto end() { return members.end(); }
  Operation *front() const { return members.front(); }
  Operation *back() const { return members.back(); }

  auto size() const { return members.size(); }

  /// Return the root of the cluster, which is the operation which is not
  /// dominated by any other ops in the cluster. Since all Operations are in the
  /// same block, this is the operation that comes after the others in the
  /// cluster.
  Operation *getRoot() const { return members.back(); }

  Attribute getTarget() const { return target; }

  static constexpr StringRef kRegionTargetAttrName = "__cluster_target__";

  /// Implicitly cast to an ArrayRef over the underlying members of the cluster.
  operator ArrayRef<Operation *>() const {
    return ArrayRef<Operation *>(members);
  }

private:
  SmallVector<Operation *> members;
  /// A cluster will be translated into different target at the end (This target
  /// indicates what this cluster will be running on, CPU/GPU.) A variable is
  /// used here to indicate what every cluster instance will be translated into.
  Attribute target;
};

/// ClusteringState tracks the division of operations into disjoint sets.
struct ClusteringState {
  /// Create clustering state given the `op` containing the Regions to cluster
  /// and the set of options that guide clustering.
  ClusteringState(Operation *op, ClusteringOpts opts);

  /// Attempts to union two clusters (the cluster containing x and the cluster
  /// containing y). If this is impossible (because it would create a cycle),
  /// return failure. Asserts that `x` and `y` each belong to a cluster. If they
  /// belong to the same cluster or if the union is successful, then `success`
  /// is returned.
  LogicalResult unionClusters(Operation *x, Operation *y);

  /// Return `true` if it is possible to union clusters `x` and `y`, otherwise
  /// false.
  bool canUnionClusters(Operation *x, Operation *y);

  /// Execute function `func` on each member of the cluster that `op` belongs to
  /// (including `op` itself).
  void runOnEquivalenceClass(
      Operation *op,
      llvm::function_ref<void(Operation *member, Operation *leader)> func)
      const;

  /// Returns true iff the `op` is tracked in some cluster.
  bool contains(Operation *op) const;

  /// Adds op as a new cluster containing a single Operation. Asserts that the
  /// op is not already in a cluster.
  void addCluster(Operation *op);

  /// Returns the cluster roots. If `sorted` is true, then they are returned in
  /// topological order.
  SmallVector<Operation *> getClusterRoots(bool sorted = false) const;

  /// Returns the set of clusters represented by the current state. The clusters
  /// are topologically sorted by root. The oeprations within each cluster are
  /// also topologically sorted.
  SmallVector<Cluster> getClusters() const;

  /// Return the options associated with this clustering state.
  // const ClusteringOpts &getOptions() const { return opts; }
  const ClusteringOpts opts;

  /// Contains dominance info for the regions of interest.
  DominanceInfo domInfo;

  /// Contains the set of clusters. Each cluster is an equivalence class. Allows
  /// iterating over all members in a cluster.
  llvm::EquivalenceClasses<Operation *> ec;
};

/// Populates a set of size-1 clusters in `state` by walking `op` in pre-order
/// and using the given callback to determine if a nested op should be used as a
/// cluster. The walk has logic to avoid creating nested clusters.
void populateSizeOneClusters(ClusteringState &state, Operation *op,
                             IsClusterableOpFn isClusterableOp);

/// Uses BFS to union the existing clusters tracked by `state`, resulting in
/// fewer clusters if any union is successful. The algorithm attempts to union
/// clusters by iterating over operations tracked by `state` in a breadth-first
/// manner and attempting to union the cluster of the current op with the
/// clusters of its users. The `shouldGrowClusterFn` callback, if provided, lets
/// the caller control whether when the algorithm should attempt to perform a
/// union between a producer/consumer cluster pair.  At worst case the running
/// time of this is O(N^2) where N is the number of operations tracked in
/// `state`, but realistically it depends on cluster sizes and program topology.
///
/// ### About Traversal Direction
///
/// In the BFS clustering algorithm, we loop over all existing operations that
/// are tracked in some cluster and attempt to combine the cluster of the
/// current op by munioning with other clusters. Pre-order iteration can
/// sometimes result in sub-optimal clustering. For example, when the input
/// clusters have a diamond pattern as shown below (in the diagram, "C0→C1"
/// means C1 consumes C0 results), and the logic in `shouldGrowClusterFn`
/// enforces the additional restriction that a producer cluster should union
/// with a consumer cluster only if all the users of the producer are in the
/// potential consumer cluster, then the algorithm will output two clusters: C0
/// and (C1, C2, C3).
///
/// ```
///    C0
///   ↙ ↘
///  C1  C2
///    ↘↙
///    C3
/// ```
///
/// The pre-order iteration prevents C0 from unioning with C1 and C2 unless a
/// horizontal unioning step is performed. However, a post order iteration would
/// union C3 with C1 and C2, then C0 will union with (C1, C2, C3).
void runBFSClustering(ClusteringState &state,
                      ShouldGrowClusterFn shouldGrowClusterFn,
                      ClusteringRootTraversalDirection rootTraversalDirection =
                          ClusteringRootTraversalDirection::PreOrder);

/// Merge independent clusters.
void mergeIndependentClusters(
    ClusteringState &state,
    ShouldMergeIndependentClustersFn shouldTryMergeClusters);

/// Annotates all ops tracked by the state with an attribute containing the
/// cluster ID.
void annotateClustersWithClusterIdAttribute(const ClusteringState &state);

/// This function expects `state` to contain an initial set of clusters. It uses
/// BFS to grow and union clusters, where two clusters with by a
/// producer-consumer relationship are merged based on the optional
/// `opts.shouldGrowClusterFn` callback as well as dependence criteria. Finally,
/// "horizontal" merging (unioning clusters without producer-consumer
/// relationship if dependence constraints allow) is performed using the
/// `opts.shouldMergeIndependentClusters` callback.
FailureOr<SmallVector<Cluster>>
analyzeAndClusterOperations(ClusteringState &state);

/// Same as `analyzeAndClusterOperations`, but it seeds the initial clustering
/// by walking the regions in `op` and using the `opts.isClusterableOp` create
/// initial single-operation clusters.
FailureOr<SmallVector<Cluster>>
analyzeAndClusterOperations(Operation *op, const ClusteringOpts &opts);

/// A handle to a function that constructs a "region op" for representing
/// clusters. A "region" operation is an operation with no arguments and a
/// single one-block region. The region op must also store the `target`
/// Attribute that indicates what type of backend or compilation path is
/// targeted for the cluster.
using ClusterRegionOpBuilderFunc = std::function<Operation *(
    OpBuilder &, Location loc, TypeRange types, Attribute target)>;

/// Creates a "region op" from the given cluster. See above for the
/// definition of "region op". When an operation located outside of the cluster
/// uses an SSA value produced by an operation in the cluster, the use is
/// replaced by the result of the region op. It is assumed that the root is
/// located at the back of the cluster.
Operation *createRegionOpFromCluster(const Cluster &cluster,
                                     RewriterBase &rewriter,
                                     ClusterRegionOpBuilderFunc createRegionOp);

template <typename OpType, typename Terminator>
OpType createRegionOpFromCluster(const Cluster &cluster,
                                 RewriterBase &rewriter) {
  return cast<OpType>(createRegionOpFromCluster(
      cluster, rewriter,
      [](OpBuilder &b, Location loc, TypeRange types, Attribute target) {
        OpType op = b.create<OpType>(loc, types, target);
        b.setInsertionPointToStart(&op->getRegion(0).emplaceBlock());
        b.create<Terminator>(loc);
        return op;
      }));
}

struct OutlineRegionOptions {
  /// The type-converter allows for converting the signature/results of a
  /// cluster during the outlining in a many-to-one manner. For example,
  /// this can be used to implement scalarization of tensor arguments. If not
  /// needed, the caller should set this to the identity converter.
  OneToNTypeConverter typeConverter;

  /// Function that returns `true` if the given Value defined above and used
  /// within the region should be cloned into the region instead of passed as an
  /// argument. The `Value` will typically be a constant or result of an
  /// operation with no operands.
  std::function<bool(Value, Region &)> shouldCloneProducer{};

  /// A 'CreateFuncAndCallStubsFunc' is a function that takes in the signature
  /// of a function and constructs a 'FunctionOpInterface' stub function (body
  /// only contains 'return' terminator). In addition, it should construct a
  /// CallOpInterface operation immediately before `regionOp`.
  /// The CallOpInterface calls the created stub function. The arguments of the
  /// call are specified by `callOperands`.
  using CreateFuncAndCallStubsFunc = std::function<
      FailureOr<std::pair<FunctionOpInterface, SmallVector<Value>>>(
          RewriterBase &, Location, Operation *regionOp,
          ArrayRef<Value> callOperands, ArrayRef<Type> funcArgTypes,
          ArrayRef<Type> funcResultTypes)>;

  /// Return a default implementation for a 'CreateFuncAndCallStubsFunc'. The
  /// caller must provide the symbol table in which the created function will be
  /// inserted. The function is named by using 'namePrefix', but it may have a
  /// suffix attached when the SymbolTable insertion uniques the symbol name.
  /// Additional discardable attributes `extraFuncAttrs` will be added to the
  /// created function if provided.
  static CreateFuncAndCallStubsFunc getDefaultCreateFuncAndCallStubFunc(
      SymbolTable &moduleSymbolTable,
      ArrayRef<NamedAttribute> extraFuncAttrs = {},
      StringRef namePrefix = "cluster");

  /// Creates and returns the outlined function (with an empty body region)
  /// using the specified name and operands.
  CreateFuncAndCallStubsFunc createFunc;
};

/// Outline the given `scf.execute_region` operation to a function-like
/// operation. The `scf.execute_region` operation gets replaced with a
/// call-like operation. The creation of the function and call-like operation
/// are performed by the caller through a callback (see `OutlineRegionOptions`
/// above).
FailureOr<std::pair<FunctionOpInterface, SetVector<Value>>>
outlineRegionOp(RewriterBase &rewriter, Operation *op,
                OutlineRegionOptions &opts);

} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_CLUSTERING_CLUSTERING_H
