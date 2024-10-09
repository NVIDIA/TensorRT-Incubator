//===- TestClustering.cpp  ------------------------------------------------===//
//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Clustering transforms test pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-executor/Transforms/Clustering/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

scf::ExecuteRegionOp createScfRegionOpFromCluster(const Cluster &cluster,
                                                  RewriterBase &rewriter) {
  return cast<scf::ExecuteRegionOp>(mlir::createRegionOpFromCluster(
      cluster, rewriter,
      [](OpBuilder &b, Location loc, TypeRange types, Attribute target) {
        scf::ExecuteRegionOp op = b.create<scf::ExecuteRegionOp>(loc, types);
        b.setInsertionPointToStart(&op->getRegion(0).emplaceBlock());
        b.create<scf::YieldOp>(loc);
        return op;
      }));
}

namespace {
class TestClusteringPass
    : public PassWrapper<TestClusteringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestClusteringPass)
  TestClusteringPass() {}
  TestClusteringPass(const TestClusteringPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-clustering"; }
  StringRef getDescription() const final {
    return "Test cluster-and-outline transformation";
  }

  void runOnOperation() override {
    // This pass must run on a module because it creates new functions. We will
    // run the cluster-and-outline on the first function in the module, which is
    // how are tests are setup.
    ModuleOp op = getOperation();
    func::FuncOp target = *op.getOps<func::FuncOp>().begin();

    ClusteringOpts opts;
    opts.shouldGrowClusterFn =
        [](Operation *producer, ClusterRange producerCluster,
           Operation *consumer, ClusterRange consumerCluster) {
          // Check if the producer's users are all in the consumer cluster.
          for (Operation *user : producer->getUsers()) {
            if (!llvm::is_contained(consumerCluster, user))
              return false;
          }
          return true;
        };
    opts.mergeIndependentClusters = [&](Operation *lhsRoot, ClusterRange lhs,
                                        Operation *rhsRoot, ClusterRange rhs) {
      // For testing, we use an arbitrary criteria of "no more than eight arith
      // operations in each cluster".
      unsigned countLhs = llvm::count_if(lhs, [](Operation *op) {
        return isa_and_nonnull<arith::ArithDialect>(op->getDialect());
      });
      unsigned countRhs = llvm::count_if(rhs, [](Operation *op) {
        return isa_and_nonnull<arith::ArithDialect>(op->getDialect());
      });
      return countLhs + countRhs <= horizontalMergeArithOpLimit;
    };
    opts.isClusterableOp = [](Operation *op) -> bool {
      // Constants are handled at the outline step.
      if (op->hasTrait<OpTrait::ConstantLike>())
        return false;
      return isa_and_nonnull<arith::ArithDialect>(op->getDialect());
    };
    opts.clusterTarget = StringAttr::get(op->getContext(), "test-target");

    /// Initialize the clustering state using the attributes. This allows us to
    /// inject initial conditions that can check edge cases in the horizontal
    /// merge step that would otherwise be very difficult to recreate using a
    /// small example. It is important to create the clusters in a particular
    /// order so that we can have predictable results. Therefore, first identify
    /// the cluster steps, then add them in order of cluster id.
    ClusteringState state(target, opts);
    std::map<int64_t, SmallVector<Operation *>> seedMap;
    for (Operation &op : target.getOps()) {
      IntegerAttr seedAttr = op.getAttrOfType<IntegerAttr>("__cluster_id__");
      if (!seedAttr)
        continue;
      int64_t clusterId = seedAttr.getInt();
      if (!llvm::is_contained(seedMap, clusterId)) {
        seedMap[clusterId] = SmallVector<Operation *>{&op};
        continue;
      }
      seedMap[clusterId].push_back(&op);
    }

    for (auto [clusterId, ops] : seedMap) {
      state.addCluster(ops.back());
      for (Operation *op : ArrayRef(ops).drop_back(1)) {
        state.addCluster(op);
        if (failed(state.unionClusters(op, ops.back()))) {
          emitError(op->getLoc())
              << "failed to setup initial clusters due to dependency violation";
          return signalPassFailure();
        }
      }
    }

    // If there are no initial seeds, then use clustering callback
    // `isClusterableOp` to perform the clustering.
    if (seedMap.empty())
      mlir::populateSizeOneClusters(state, target, opts.isClusterableOp);

    if (!disableBFSClustering)
      runBFSClustering(state, opts.shouldGrowClusterFn,
                       bfsClusteringDirection == "pre"
                           ? ClusteringRootTraversalDirection::PreOrder
                           : ClusteringRootTraversalDirection::PostOrder);

    if (doMergeIndependentClusters)
      mergeIndependentClusters(state, opts.mergeIndependentClusters);

    SmallVector<Cluster> clusters = state.getClusters();

    // Perform outlining to functions.
    IRRewriter rewriter(&getContext());
    OneToNTypeConverter typeConverter = getIdentityTypeConverter();
    auto shouldCloneProducer = [](Value definedAbove, Region &cluster) {
      Operation *producer = definedAbove.getDefiningOp();
      if (!producer || !producer->hasTrait<OpTrait::ConstantLike>())
        return false;
      return llvm::all_of(producer->getUsers(), [&cluster](Operation *user) {
        return cluster.isAncestor(user->getParentRegion());
      });
    };
    for (Cluster &cluster : clusters) {
      auto regionOp = createScfRegionOpFromCluster(cluster, rewriter);
      SetVector<Value> operands;

      OutlineRegionOptions outlineOpts{
          /*typeConverter=*/getIdentityTypeConverter(),
          /*shouldCloneProducer=*/shouldCloneProducer,
          /*createFunc=*/
          [](RewriterBase &rewriter, Location loc, Operation *regionOp,
             ArrayRef<Value> callOperands, ArrayRef<Type> convertedOperandTypes,
             ArrayRef<Type> results)
              -> FailureOr<std::pair<FunctionOpInterface, SmallVector<Value>>> {
            // Create the func for outlining the region body.
            FunctionType type = FunctionType::get(
                rewriter.getContext(), convertedOperandTypes, results);
            auto outlinedFunc =
                mlir::func::FuncOp::create(loc, "cluster", type, {});
            outlinedFunc.setPrivate();
            Block *funcBody = outlinedFunc.addEntryBlock();

            // Add an empty terminator.
            rewriter.setInsertionPointToEnd(funcBody);
            rewriter.create<func::ReturnOp>(loc);

            // Insert into the module.
            auto module = regionOp->getParentOfType<ModuleOp>();
            SymbolTable(module).insert(
                outlinedFunc, module->getRegions().front().front().end());

            rewriter.setInsertionPoint(regionOp);
            auto callOp =
                rewriter.create<func::CallOp>(loc, outlinedFunc, callOperands);

            return std::make_pair(cast<FunctionOpInterface>(*outlinedFunc),
                                  SmallVector<Value>(callOp.getResults()));
          }};

      FailureOr<std::pair<FunctionOpInterface, SetVector<Value>>> func =
          outlineRegionOp(rewriter, regionOp, outlineOpts);
      if (failed(func)) {
        emitError(regionOp->getLoc())
            << "failed to outline cluster to function";
        return signalPassFailure();
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect>();
  }
  Option<int64_t> horizontalMergeArithOpLimit{
      *this, "horizontal-merge-arith-op-limit",
      llvm::cl::desc("the limit on number of arith ops per cluster during "
                     "horizontal merge"),
      llvm::cl::init(4)};
  Option<bool> disableBFSClustering{
      *this, "disable-bfs-clustering",
      llvm::cl::desc("disable the BFS clustering step"), llvm::cl::init(false)};
  Option<bool> doMergeIndependentClusters{
      *this, "merge-independent-clusters",
      llvm::cl::desc("perform horizontal merge"), llvm::cl::init(false)};

  Option<std::string> bfsClusteringDirection{
      *this, "bfs-root-traversal",
      llvm::cl::desc("direction of root traversal during the BFS clustering "
                     "step, 'pre' or 'post'"),
      llvm::cl::init("pre")};
};
} // namespace

namespace mlir::executor {
void registerTestClusteringTransformPass() {
  PassRegistration<TestClusteringPass>();
}
} // namespace mlir::executor