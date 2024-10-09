//===- CreateShapeFuncs.cpp -----------------------------------------------===//
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
/// Implementation of the `plan-create-shape-funcs` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "plan-create-shape-funcs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

static constexpr llvm::StringLiteral kShapeFuncArgAttrName =
    "plan.shape_func_arg";

namespace mlir::plan {
#define GEN_PASS_DEF_PLANCREATESHAPEFUNCSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// Shape function clustering
// The shape function clustering functions below were inspired by the upstream
// `outline-shape-computations-pass` defined by the Shape dialect. However,
// our implementation differs significantly in the types of operations which
// can be clustered and the overall logic around outlining. We may transition
// the DFS clustering code to use our internal DFS clustering algorithm, which
// has better test coverage.
//===----------------------------------------------------------------------===//

// Returns true if `op` is a shape.with_shape that uses the scalar `prevOutput`
// or if `op` is a scalar calculation where all the users' of `op` eventually
// point to the shape operand of plan.with_shape op. In the second case, `op` is
// added to the `shapeOperations` set.
static bool populateShapeOperationsDFS(Operation *op, Value prevOutput,
                                       DenseSet<Operation *> &shapeOperations) {
  if (shapeOperations.contains(op))
    return true;

  if (auto withOp = llvm::dyn_cast<plan::WithShapeOp>(op))
    return llvm::is_contained(withOp.getShape(), prevOutput);

  // When CSE is applied after the shape calculation materialization, we may
  // have computations that feed into both `plan.with_shape` operations as well
  // as `plan.with_values` ops. If a scalar is used in a `with_values` op, this
  // does not mean its producer is not a shape func.
  if (auto withValuesOp = llvm::dyn_cast<plan::WithValuesOp>(op))
    return llvm::is_contained(withValuesOp.getElements(), prevOutput);

  if (!isa<arith::ArithDialect>(op->getDialect()) &&
      !isa<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>(op))
    return false;

  if (op->use_empty())
    return false;

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (!constOp.getType().isSignlessIntOrIndexOrFloat())
      return false;
    if (llvm::any_of(constOp->getUsers(), [&](Operation *user) {
          return populateShapeOperationsDFS(user, constOp.getResult(),
                                            shapeOperations);
        })) {
      shapeOperations.insert(op);
      return true;
    }
    return false;
  }

  if (llvm::none_of(op->getResultTypes(),
                    [](Type t) { return t.isSignlessIntOrIndexOrFloat(); }))
    return false;

  for (Value res : op->getResults()) {
    for (Operation *user : res.getUsers()) {
      // If some operation in the forward slice is not a scalar arithmetic
      // op or a `plan.with_shape` op, then `op` is not a shape
      // calculation operation.
      if (!populateShapeOperationsDFS(user, res, shapeOperations))
        return false;
    }
  }

  shapeOperations.insert(op);
  return true;
}

/// Returns all the operations in `op` that are purely used to calculate
/// a shape used by `shape.with_shape`.  All operations in `op` can be
/// clustered and outlined to shape functions.
static DenseSet<Operation *> getShapeCalculationOps(Operation *op) {
  DenseSet<Operation *> shapeOperations;
  op->walk([&](Operation *op) {
    (void)populateShapeOperationsDFS(op, /*prevOutput=*/nullptr,
                                     shapeOperations);
  });
  return shapeOperations;
}

static bool needsShapeFunc(func::FuncOp func) {
  auto isDynamic = [](Type t) {
    auto rtt = dyn_cast<RankedTensorType>(t);
    if (rtt && !rtt.hasStaticShape())
      return true;
    return false;
  };

  return llvm::any_of(func.getFunctionType().getResults(), isDynamic);
}

// The output of a cluster is the `shape`, and the inputs are the outputs of
// operations who are not in `onlyUsedByWithShapes`
static DenseSet<Operation *>
getClusterFromShapeOp(plan::WithShapeOp withOp,
                      const DenseSet<Operation *> shapeOperations) {
  DenseSet<Operation *> cluster;
  DenseSet<Operation *> visited;
  std::queue<Operation *> queue;

  for (Value dim : withOp.getShape()) {
    if (Operation *defOp = dim.getDefiningOp()) {
      visited.insert(defOp);
      queue.push(defOp);
    }
    while (!queue.empty()) {
      Operation *op = queue.front();
      queue.pop();
      if (!shapeOperations.contains(op))
        continue;
      cluster.insert(op);
      for (Value inp : op->getOperands()) {
        Operation *inpDefOp = inp.getDefiningOp();
        if (!inpDefOp)
          continue;
        if (inpDefOp && !visited.contains(inpDefOp)) {
          visited.insert(inpDefOp);
          queue.push(inpDefOp);
        }
      }
    }
  }
  return cluster;
}

static DenseMap<plan::WithShapeOp, SmallVector<Operation *>>
createShapeClusters(Operation *op,
                    const DenseSet<Operation *> shapeOperations) {
  // collect all the plan.with_shape ops.
  SmallVector<plan::WithShapeOp> allWithOps;
  op->walk([&](plan::WithShapeOp withOp) { allWithOps.push_back(withOp); });

  DenseMap<plan::WithShapeOp, DenseSet<Operation *>> clusters;
  for (plan::WithShapeOp withOp : allWithOps) {
    if (!clusters.contains(withOp))
      clusters[withOp] = getClusterFromShapeOp(withOp, shapeOperations);
  }

  DenseMap<Operation *, SmallVector<plan::WithShapeOp>> op2shapes;
  for (auto [withOp, shapeOpSet] : clusters) {
    for (Operation *shapeOp : shapeOpSet)
      op2shapes[shapeOp].push_back(withOp);
  }

  // This ends up doing the topological sort.
  DenseMap<plan::WithShapeOp, SmallVector<Operation *>> result;
  op->walk([&](Operation *op) {
    auto it = op2shapes.find(op);
    if (it == op2shapes.end())
      return;
    const auto &[shapeOp, withOps] = *it;
    for (auto withOp : withOps)
      result[withOp].push_back(shapeOp);
  });

  return result;
}

struct ShapeClusterIO {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

static ShapeClusterIO getClusterBoundary(plan::WithShapeOp withOp,
                                         ArrayRef<Operation *> cluster) {
  ShapeClusterIO result;
  llvm::SmallDenseSet<Value> inputSet;
  llvm::SmallDenseSet<Operation *> opSet(cluster.begin(), cluster.end());
  assert(opSet.size() == cluster.size() &&
         "cluster contains duplicate operations");

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (opSet.contains(operandOp))
        continue;
      if (inputSet.insert(operand).second)
        result.inputs.push_back(operand);
    }
  }

  for (Value shapeDim : withOp.getShape()) {
    Operation *def = shapeDim.getDefiningOp();
    if ((!def && !llvm::is_contained(result.inputs, shapeDim)) ||
        (def && !opSet.contains(def)))
      result.inputs.push_back(shapeDim);
    result.outputs.push_back(shapeDim);
  }

  return result;
}

// Create a shape.func representing the shape computation for `shape`.
static std::pair<func::FuncOp, SmallVector<Value>>
createFuncFromCluster(RewriterBase &b, Location loc,
                      ArrayRef<Operation *> cluster,
                      plan::WithShapeOp withShapeOp, StringRef fnName) {
  OpBuilder::InsertionGuard g(b);
  ShapeClusterIO clusterIO = getClusterBoundary(withShapeOp, cluster);
  TypeRange fnResultTypes = TypeRange(withShapeOp.getShape());

  auto fnType =
      cluster.empty()
          ? b.getFunctionType(fnResultTypes, fnResultTypes)
          : b.getFunctionType(TypeRange(clusterIO.inputs), fnResultTypes);

  func::FuncOp fnOp = func::FuncOp::create(loc, fnName, fnType);
  Block *block = fnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  IRMapping bvm;

  for (auto [inp, arg] : llvm::zip(clusterIO.inputs, fnOp.getArguments())) {
    bvm.map(inp, arg);

    // If the function argument mapped to a scalar input or dimension of an
    // argument of the original func, then append that metadata as to the arg
    // attributes.
    if (auto dimOp = inp.getDefiningOp<tensor::DimOp>()) {
      std::optional<int64_t> dim = getConstantIntValue(dimOp.getDimension());
      auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
      if (blockArg && isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        fnOp.setArgAttr(
            arg.getArgNumber(), kShapeFuncArgAttrName,
            b.getDictionaryAttr(
                {b.getNamedAttr("argument",
                                b.getIndexAttr(blockArg.getArgNumber())),
                 b.getNamedAttr(
                     "dimension",
                     b.getIndexAttr(dim ? *dim : ShapedType::kDynamic))}));
      }
    }

    // If the function argument mapped to a shape tensor argument...
    if (auto extractOp = inp.getDefiningOp<tensor::ExtractOp>()) {
      SmallVector<int64_t> indices;
      for (Value v : extractOp.getIndices()) {
        APInt constIndex;
        if (matchPattern(v, m_ConstantInt(&constIndex))) {
          indices.push_back(constIndex.getSExtValue());
          continue;
        }
        indices.push_back(ShapedType::kDynamic);
      }

      auto blockArg = dyn_cast<BlockArgument>(extractOp.getTensor());
      if (blockArg && isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        fnOp.setArgAttr(
            arg.getArgNumber(), kShapeFuncArgAttrName,
            b.getDictionaryAttr(
                {b.getNamedAttr("argument",
                                b.getIndexAttr(blockArg.getArgNumber())),
                 b.getNamedAttr("indices", b.getDenseI64ArrayAttr(indices))}));
      }
    }

    if (auto blockArg = dyn_cast<BlockArgument>(inp)) {
      if (isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        fnOp.setArgAttr(
            arg.getArgNumber(), kShapeFuncArgAttrName,
            b.getDictionaryAttr({b.getNamedAttr(
                "argument", b.getIndexAttr(blockArg.getArgNumber()))}));
      }
    }
  }

  for (Operation *op : cluster)
    b.clone(*op, bvm);

  // Get the outputs.
  SmallVector<Value> results = llvm::map_to_vector(
      clusterIO.outputs, [&](Value v) { return bvm.lookup(v); });
  b.create<func::ReturnOp>(loc, results);
  fnOp.setPrivate();
  return std::make_pair(fnOp, clusterIO.inputs);
}

static func::ReturnOp getUniqueReturn(func::FuncOp op) {
  assert(op.getBody().getBlocks().size() == 1 &&
         "expected single-block function body region");
  return cast<func::ReturnOp>(op.getFunctionBody().front().getTerminator());
}

static MemorySpaceAttr getHostSpace(RewriterBase &rewriter) {
  return MemorySpaceAttr::get(rewriter.getContext(), MemorySpace::host);
}

static Value getHostConstantTensor(RewriterBase &rewriter, Location loc,
                                   ArrayRef<int64_t> values) {
  auto rtt =
      RankedTensorType::get({static_cast<int64_t>(values.size())},
                            rewriter.getIndexType(), getHostSpace(rewriter));

  // Create the DenseIntElementsAttr with host space
  auto attr = DenseIntElementsAttr::get(rtt, values);
  return rewriter.create<arith::ConstantOp>(loc, rtt, attr);
}

static SmallVector<Value> createConstantIndices(RewriterBase &rewriter,
                                                Location loc,
                                                ArrayRef<int64_t> indices) {
  return llvm::map_to_vector(indices, [&](int64_t i) -> Value {
    return rewriter.create<arith::ConstantIndexOp>(loc, i);
  });
}

/// Creates a `func.call` operation (assuming insertion point is somewhere in
/// `currFunc`'s body), where the `shapeFunc` should be a shape calculation
/// func where arguments are decorated with `plan.shape_func_arg` information.
/// It is assumed that `currFunc` contains arguments containing either shape
/// tensors, original host tensor, or original scalar arguments corresponding
/// to the original function's argument types.
static FailureOr<func::CallOp> createCallForShape(RewriterBase &rewriter,
                                                  Location loc,
                                                  func::FuncOp shapeFunc,
                                                  func::FuncOp currFunc) {
  // Create the arguments.
  SmallVector<Value> callArgs;
  for (unsigned i = 0; i < shapeFunc.getNumArguments(); i++) {
    auto argMetadata =
        shapeFunc.getArgAttrOfType<DictionaryAttr>(i, kShapeFuncArgAttrName);
    // No mapping to input args.
    if (!argMetadata)
      return failure();

    if (argMetadata.contains("dimension") && argMetadata.contains("argument")) {

      int64_t dimension =
          cast<IntegerAttr>(argMetadata.getNamed("dimension")->getValue())
              .getInt();
      int64_t argIdx =
          cast<IntegerAttr>(argMetadata.getNamed("argument")->getValue())
              .getInt();

      callArgs.push_back(rewriter.create<tensor::ExtractOp>(
          loc, currFunc.getArgument(argIdx),
          createConstantIndices(rewriter, loc, dimension)));

      continue;
    }

    if (argMetadata.contains("indices") && argMetadata.contains("argument")) {
      ArrayRef<int64_t> indices =
          cast<DenseI64ArrayAttr>(argMetadata.getNamed("indices")->getValue())
              .asArrayRef();
      int64_t argIdx =
          cast<IntegerAttr>(argMetadata.getNamed("argument")->getValue())
              .getInt();
      callArgs.push_back(rewriter.create<tensor::ExtractOp>(
          loc, currFunc.getArgument(argIdx),
          createConstantIndices(rewriter, loc, indices)));
      continue;
    }

    if (argMetadata.contains("argument")) {
      int64_t argIdx =
          cast<IntegerAttr>(argMetadata.getNamed("argument")->getValue())
              .getInt();
      callArgs.push_back(currFunc.getArgument(argIdx));
      continue;
    }

    return failure();
  }

  return rewriter.create<func::CallOp>(loc, shapeFunc, callArgs);
}

static RankedTensorType getShapeTensorType(RankedTensorType sourceType) {
  return RankedTensorType::get(
      {std::max<int64_t>(1, sourceType.getRank())},
      IndexType::get(sourceType.getContext()),
      MemorySpaceAttr::get(sourceType.getContext(), MemorySpace::host));
}

/// Creates an function called `[func_name]_get_shape` for the specified
/// `func`. For each argument of `func`, it accepts:
/// - If the argument is a non-host tensor arg, it accepts an index tensor
/// containing the shape
///   It does not matter whether the arg is dynamically shaped or not.
/// - Otherwise, the arg of the new func has the same type as the arg from
/// `func`
//
/// The created function may perform assert checks that static dimensions
/// match where appropriate. It returns a set of host tensors containing the
/// shape of each result.
static FailureOr<func::FuncOp> createAggregateShapeFunc(
    RewriterBase &rewriter, func::FuncOp func,
    ArrayRef<std::optional<func::FuncOp>> shapeComputationFuncs,
    const DataFlowSolver &solver) {

  SmallVector<Type> argTypes;
  for (auto [idx, t] : llvm::enumerate(func.getArgumentTypes())) {
    auto rtt = dyn_cast<RankedTensorType>(t);
    if (!rtt)
      argTypes.push_back(t);
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(func.getArgument(idx));
    if (!lattice || lattice->getValue().isUninitialized())
      return failure();
    if (lattice->getValue().isHostVisible()) {
      argTypes.push_back(t);
      continue;
    }
    argTypes.push_back(getShapeTensorType(rtt));
  }

  SmallVector<Type> resultTypes;
  SmallVector<unsigned> funcTensorResultIndices;
  resultTypes.reserve(func.getNumResults());
  for (auto [idx, t] : llvm::enumerate(func.getResultTypes())) {
    auto rtt = dyn_cast<RankedTensorType>(t);
    if (!rtt)
      continue;
    funcTensorResultIndices.push_back(idx);
    resultTypes.push_back(getShapeTensorType(rtt));
  }

  assert(shapeComputationFuncs.size() == func.getNumResults() &&
         "expected one optional shape computation function per func result");

  func::FuncOp aggregateShapeFunc = func::FuncOp::create(
      func.getLoc(), llvm::formatv("{0}_get_shapes", func.getName()).str(),
      FunctionType::get(func->getContext(), argTypes, resultTypes));
  aggregateShapeFunc.setPublic();
  Block *block = aggregateShapeFunc.addEntryBlock();
  rewriter.setInsertionPoint(block, block->end());

  func::ReturnOp term = getUniqueReturn(func);
  SmallVector<Value> shapeFuncReturns;
  for (unsigned retIdx : funcTensorResultIndices) {
    Value retValue = term->getOperand(retIdx);
    auto rtt = cast<RankedTensorType>(retValue.getType());
    if (rtt.hasStaticShape()) {
      shapeFuncReturns.push_back(
          getHostConstantTensor(rewriter, retValue.getLoc(), rtt.getShape()));
      continue;
    }

    std::optional<func::FuncOp> shapeFunc = shapeComputationFuncs[retIdx];
    if (!shapeFunc)
      return failure();

    FailureOr<func::CallOp> call = createCallForShape(
        rewriter, retValue.getLoc(), *shapeFunc, aggregateShapeFunc);
    if (failed(call))
      return failure();

    // Make sure all scalars are 'index type'. This ensures that the resulting
    // API generated is always uniform, i.e. that shape functions will yield
    // tensor type equivalent to the data model's representation of IndexType
    // regardless of what the input IR used to represent shapes (currently
    // StableHLO is fixed to i32 for legacy reasons).
    SmallVector<Value> fromElementsOperands(call->getResults());
    for (Value &val : fromElementsOperands) {
      if (val.getType().isIndex())
        continue;
      val = rewriter.create<arith::IndexCastOp>(val.getLoc(),
                                                rewriter.getIndexType(), val);
    }

    shapeFuncReturns.push_back(rewriter.create<tensor::FromElementsOp>(
        retValue.getLoc(),
        RankedTensorType::get({call->getNumResults()}, rewriter.getIndexType(),
                              getHostSpace(rewriter)),
        fromElementsOperands));
  }

  // Make sure to mark that the shape function arg and results as host tensors.
  // The TensorKindAnalysis currently doesn't inspect encoding attributes.
  for (unsigned i = 0; i < aggregateShapeFunc.getNumResults(); i++)
    aggregateShapeFunc.setResultAttr(i, getHostTensorArgAttrName(),
                                     rewriter.getUnitAttr());
  for (unsigned i = 0; i < aggregateShapeFunc.getNumArguments(); i++)
    aggregateShapeFunc.setArgAttr(i, getHostTensorArgAttrName(),
                                  rewriter.getUnitAttr());

  rewriter.create<func::ReturnOp>(func.getLoc(), shapeFuncReturns);

  return aggregateShapeFunc;
}

namespace {
class CreateShapeFuncsPass
    : public plan::impl::PlanCreateShapeFuncsPassBase<CreateShapeFuncsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp op = getOperation();

    // Identify the set of shape operations.
    llvm::DenseSet<Operation *> shapeOps = getShapeCalculationOps(op);

    // Create a map from tensor operands of `plan.with_shape` ops to
    // the set of operations that are required to calculate the shape.
    DenseMap<plan::WithShapeOp, SmallVector<Operation *>> shapeClusterMap =
        createShapeClusters(op, shapeOps);

    IRRewriter rewriter(op->getContext());

    SymbolTable symbolTable(op);

    llvm::DenseMap<WithShapeOp, func::FuncOp> createdFuncs;

    // A map from a function to the function that computes the result shapes
    // for each result. The func in the list may be `nullopt` if either the
    // result type is static or the result cannot be computed from inputs.
    llvm::DenseMap<func::FuncOp, SmallVector<std::optional<func::FuncOp>>>
        shapeComputationMap;

    // Before we start creating new functions, get a list of the existing
    // public functions.
    SmallVector<func::FuncOp> publicFuncs = llvm::to_vector(
        llvm::make_filter_range(op.getOps<func::FuncOp>(),
                                [](func::FuncOp op) { return op.isPublic(); }));

    for (auto func : publicFuncs) {

      // Skip all functions with static shape args/results.
      if (!needsShapeFunc(func))
        continue;

      func::ReturnOp term = getUniqueReturn(func);

      SmallVector<std::optional<func::FuncOp>> funcs;
      funcs.reserve(term->getNumOperands());

      for (OpOperand &operand : term->getOpOperands()) {
        auto withOp = operand.get().getDefiningOp<WithShapeOp>();
        if (!withOp) {
          funcs.push_back(std::nullopt);
          continue;
        }

        // Check if the func already exists (may happen for values that
        // share the same shape).
        auto existingFunc = createdFuncs.find(withOp);
        if (existingFunc != createdFuncs.end()) {
          funcs.push_back(existingFunc->getSecond());
          continue;
        }

        std::string shapeFuncName = llvm::formatv(
            "shape_{0}_result_{1}", func.getName(), operand.getOperandNumber());

        auto [shapeFunc, inputs] = createFuncFromCluster(
            rewriter, withOp.getLoc(), shapeClusterMap[withOp], withOp,
            shapeFuncName);

        mlir::Attribute shapeFuncMarker = mlir::StringAttr::get(
            rewriter.getContext(), PlanDialect::kShapeFuncMarkerAttrName);

        shapeFunc->setAttr(PlanDialect::kShapeFuncMarkerAttrName,
                           shapeFuncMarker);

        symbolTable.insert(shapeFunc, op.end());
        funcs.push_back(shapeFunc);
      }

      shapeComputationMap[func] = std::move(funcs);
    }

    DataFlowSolver solver;
    SymbolTableCollection symbolTableCollection;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTableCollection);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    for (auto func : publicFuncs) {

      // Skip all functions with static shape args/results.
      if (!needsShapeFunc(func))
        continue;

      FailureOr<func::FuncOp> aggShapeFunc = createAggregateShapeFunc(
          rewriter, func, shapeComputationMap[func], solver);

      // Only data-dependent shape function is expected to fail here.
      if (failed(aggShapeFunc))
        continue;

      mlir::Attribute shapeFuncMarker = mlir::StringAttr::get(
          rewriter.getContext(), PlanDialect::kShapeFuncMarkerAttrName);

      (*aggShapeFunc)
          ->setAttr(PlanDialect::kShapeFuncMarkerAttrName, shapeFuncMarker);
      if (failed(aggShapeFunc))
        continue;

      // Add the symbol to the original func.
      symbolTable.insert(*aggShapeFunc);
      func->setAttr(PlanDialect::kShapeFuncAttrName,
                    SymbolRefAttr::get(*aggShapeFunc));
    }
  }
};
} // namespace
