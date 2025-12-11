//===- TensorRTBackend.cpp ------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Definitions for TensorRT backend extensions to the Plan dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.h"
#include "mlir-executor/Transforms/Clustering/Clustering.h"
#include "mlir-tensorrt-common/Utils/TensorRTVersion.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Utils/ShapeInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace mlir;
using namespace mlir::plan;

#define DEBUG_TYPE "tensorrt-backend"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

/// Include the Tablegen'd C++ code describing the backend attribute. We will
/// attach this to the plan dialect as an extension.
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackendAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorRTBackendAttr
//===----------------------------------------------------------------------===//

static ShapeInfoCallbacks getShapeInfoCallbacks() {
  ShapeInfoCallbacks shapeInfoCallbacks{};
  shapeInfoCallbacks.isElementValueEqualToConstant =
      [](TensorElementValue elementValue,
         Attribute constValue) -> std::optional<bool> {
    auto withValuesOp =
        elementValue.getTensor().getDefiningOp<plan::WithValuesOp>();
    if (!withValuesOp)
      return {};
    Value element = withValuesOp.getElements()[elementValue.getLinearIndex()];

    Attribute intAttr = {};
    if (!matchPattern(element, m_Constant(&intAttr)))
      return {};
    return intAttr == constValue;
  };
  shapeInfoCallbacks.isElementValueEqualToShapeDimExtent =
      [](TensorElementValue elementValue,
         TensorShapeDimExtent dimExtent) -> std::optional<bool> {
    assert(elementValue.getTensor().getType().getElementType().isIntOrIndex() &&
           "expected int or integer tensor");
    auto withValuesOp =
        elementValue.getTensor().getDefiningOp<plan::WithValuesOp>();
    if (!withValuesOp)
      return {};

    /// Scalar value will be of type equivalent to `elementValue.tensor` element
    /// type.
    Value scalarValue =
        withValuesOp.getElements()[elementValue.getLinearIndex()];

    /// Check if it is statically known to be equal to the `dimExtent`.
    IntegerAttr constInt = {};
    if (std::optional<int64_t> staticSize = dimExtent.getConstantSize()) {
      if (matchPattern(scalarValue, m_Constant(&constInt)))
        return constInt.getValue().getSExtValue() == *staticSize;
    }

    /// Otherwise, we need to check equivalence of the dynamic values.
    /// There are two cases to consider: either both have the same type, or
    /// `plan.with_shape` may have index type scalars and `plan.with_values`
    /// will have a more specific integer type that matches the shape tensor.
    /// We can try to handle the later case where the conversion is done by
    /// `arith.index_cast`.
    /// TODO: we should change the shape materialization pass so that we infer
    /// the desired shape tensor element type and have all `plan.with_shape`
    /// materialize with that scalar type using casts.
    if (auto withShape = dimExtent.tensor.getDefiningOp<plan::WithShapeOp>()) {
      Value dimExtentValue = withShape.getShape()[dimExtent.dim];
      if (dimExtentValue == scalarValue)
        return true;
      if (auto indexCastOp =
              dyn_cast<arith::IndexCastOp>(scalarValue.getDefiningOp())) {
        if (indexCastOp.getOperand() == dimExtentValue)
          return true;
      }
    }

    return {};
  };
  return shapeInfoCallbacks;
}

/// Return true if the op is an input dialect operation.
static bool isStableHloOrChloOp(Operation *op) {
  return llvm::isa_and_present<stablehlo::StablehloDialect, chlo::ChloDialect,
                               tensorrt::TensorRTDialect>(op->getDialect());
}

/// Return true if the op is a loop-carried dependency that likely should be
/// bufferized in-place while re-using an input buffer for the output buffer.
/// Such operations should not be offloaded to TensorRT due to I/O aliasing
/// constraints.
static bool isLikelyYieldedFromLoopAndBufferizeInPlace(Operation *op) {
  if (!llvm::any_of(op->getUsers(),
                    llvm::IsaPred<scf::YieldOp, scf::ConditionOp>))
    return false;
  return true;
}

/// Return true if the value is likely to be a shape tensor that resides in the
/// host memory space.
static bool isLikelyShapeTensor(Value v, const DataFlowSolver &solver) {
  const auto *lattice = solver.lookupState<TensorKindLattice>(v);
  return lattice && !lattice->getValue().isUninitialized() &&
         lattice->getValue().isHostVisible();
}

/// Return the types of all operands that are likely to be shape tensors.
static SmallVector<RankedTensorType, 2>
checkForShapeTensorOperands(Operation *op, const DataFlowSolver &solver) {
  SmallVector<RankedTensorType, 2> shapeTensorsTypes;
  llvm::for_each(op->getOperands(), [&](Value v) {
    auto tensorType = dyn_cast<RankedTensorType>(v.getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return;
    if (isLikelyShapeTensor(v, solver))
      shapeTensorsTypes.push_back(tensorType);
  });
  return shapeTensorsTypes;
}

/// Return the types of all operands that are likely to be shape tensors.
static SmallVector<RankedTensorType, 2>
checkForShapeTensorResults(Operation *op, const DataFlowSolver &solver) {
  SmallVector<RankedTensorType, 2> shapeTensorsTypes;
  llvm::for_each(op->getResults(), [&](Value v) {
    auto tensorType = dyn_cast<RankedTensorType>(v.getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return;
    if (isLikelyShapeTensor(v, solver))
      shapeTensorsTypes.push_back(tensorType);
  });
  return shapeTensorsTypes;
}

/// Don't cluster operations which are slices of a block input input. These
/// are unlikely to have good performance and often will materialize
/// copies.
static bool isSliceOfBlockArgument(Operation *op) {
  if (!isa<stablehlo::SliceOp, stablehlo::DynamicSliceOp,
           stablehlo::DynamicUpdateSliceOp, stablehlo::RealDynamicSliceOp>(op))
    return false;
  return isa<BlockArgument>(op->getOperand(0));
}

/// ClusteringOpts that identifies groups of TensorRT operations and will be
/// clustered into one TensorRT function (which is eventually translated to a
/// engine).
FailureOr<ClusteringOpts>
TensorRTBackendAttr::getClusterKindOptions(InputKind inputKind, Operation *op,
                                           DataFlowSolver &solver) const {
  // Any properties used in the returned lambdas must be copied by value,
  // otherwise it will not work correctly.
  bool disallowShapeTensorCalculations = getDisallowShapeTensorCalculations();

  ClusteringOpts opts;
  opts.mergeIndependentClusters = [](Operation *, ClusterRange, Operation *,
                                     ClusterRange) { return true; };
  opts.clusterTarget = *this;

  std::optional<int64_t> tensorrtMajorVersion = getTensorrtMajorVersion();
  if (!tensorrtMajorVersion)
    tensorrtMajorVersion = NV_TENSORRT_MAJOR;

  MLIRContext *ctx = op->getContext();
  RewritePatternSet patterns(ctx);
  LowerToTensorRTOptions loweringOptions;
  loweringOptions.setTensorRTVersion(*tensorrtMajorVersion);
  TensorRTTypeConverter typeConverter(ctx, loweringOptions);
  TensorRTConversionTarget target(*ctx, typeConverter);
  populateStablehloToTensorRtConversionPattern(typeConverter, patterns,
                                               getShapeInfoCallbacks());
  populateChloToTensorRtLegalityAndPatterns(typeConverter, target, patterns);

  // Analyze the convertible operations.
  ConversionConfig conversionConfig;
  DenseSet<Operation *> legalizedOps;
  conversionConfig.legalizableOps = &legalizedOps;
  if (failed(applyAnalysisConversion(op, target, std::move(patterns),
                                     conversionConfig)))
    return emitError(op->getLoc())
           << "failed to apply TensorRT conversion analysis";

  opts.isClusterableOp = [solver = &solver, disallowShapeTensorCalculations,
                          legalizedOps](Operation *op) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      return false;
    if (llvm::isa<plan::WithShapeOp>(op))
      return true;
    if (llvm::isa<plan::WithValuesOp>(op))
      return !disallowShapeTensorCalculations;
    if (llvm::isa<tensorrt::TensorRTOpInterface>(op))
      return true;
    if (!isStableHloOrChloOp(op))
      return false;
    if (!legalizedOps.contains(op))
      return false;
    // Don't cluster operations inside of stablehlo ops with regions.
    // For example, if we set `disallowShapeTensorCalculations`, then
    // a parent `stablehlo.reduce` might not be clustered even though it was
    // converted. The operations inside the `stablehlo.reduce` are considered
    // legalized since the parent was legalized, but we don't want to cluster
    // them since they weren't directly replaced.
    Operation *parent = op->getParentOp();
    if (parent && isStableHloOrChloOp(parent) && legalizedOps.contains(parent))
      return false;

    /// Don't cluster operations which are slices of a block input input. These
    /// are unlikely to have good performance and often will materialize
    /// copies.
    if (isSliceOfBlockArgument(op))
      return false;

    // Check for operations that would not benefit from TensorRT offloading due
    // to TensorRT's I/O aliasing constraints.
    if (isLikelyYieldedFromLoopAndBufferizeInPlace(op))
      return false;

    SmallVector<RankedTensorType, 2> shapeTensorOperandTypes =
        checkForShapeTensorOperands(op, *solver);
    SmallVector<RankedTensorType, 2> shapeTensorResultTypes =
        checkForShapeTensorResults(op, *solver);

    // Don't cluster operations that have boolean shape tensor operands.
    // Otherwise clustering may create clusters with boolean shape tensors as
    // inputs to the cluster; we don't have a way to reject such cases
    // currently.
    if (llvm::any_of(shapeTensorOperandTypes, [](RankedTensorType type) {
          return type.getElementType().isInteger(1);
        }))
      return false;

    if (disallowShapeTensorCalculations && !shapeTensorResultTypes.empty())
      return false;

    return true;
  };

  return opts;
}

int64_t TensorRTBackendAttr::getClusterBenefit(InputKind inputKind) const {
  return getBenefit();
}

std::function<bool(const Cluster &)>
TensorRTBackendAttr::getClusterFilter(InputKind inputKind) const {
  // Disregard the cluster if it is all constant ops.
  return [](const Cluster &cluster) -> bool {
    return !llvm::all_of(cluster, [](Operation *op) {
      return op->hasTrait<OpTrait::ConstantLike>() ||
             llvm::isa<plan::WithShapeOp, plan::WithValuesOp>(op);
    });
  };
}

std::optional<OutlineRegionOptions>
TensorRTBackendAttr::getClusterOutliningOptions(
    InputKind inputKind, MLIRContext *ctx,
    SymbolTable &moduleSymbolTable) const {
  return {};
}

static FailureOr<tensorrt::ShapeProfileAttr>
getTensorRTShapeProfile(plan::BoundsAttr attr, Value v) {
  // The TensorRT group formation pass doesn't give any information about "opt"
  // profiles, so we just choose the midpoint of the upper and lower bounds.
  auto getProfileAttr = [&](ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
    SmallVector<int64_t> opt;
    opt.reserve(lb.size());
    for (auto [l, u] : llvm::zip_equal(lb, ub))
      opt.push_back((l + u) / 2);
    return tensorrt::ShapeProfileAttr::get(attr.getContext(), lb, opt, ub);
  };

  RankedTensorType rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt || attr.isNone())
    return failure();

  auto apIntConvertToI64 = [](const APInt &apInt) {
    return apInt.getSExtValue();
  };
  if (attr.isValueBound())
    return getProfileAttr(
        llvm::map_to_vector(attr.getMinValues().getValues<APInt>(),
                            apIntConvertToI64),
        llvm::map_to_vector(attr.getMaxValues().getValues<APInt>(),
                            apIntConvertToI64));

  if (rtt.hasStaticShape())
    return getProfileAttr(rtt.getShape(), rtt.getShape());

  assert(attr.isShapeBound() && "expected shape bound");
  return getProfileAttr(attr.getMinShape(), attr.getMaxShape());
}

template <typename RegionOpType>
static LogicalResult populateFunctionAttributes(RewriterBase &rewriter,
                                                RegionOpType op,
                                                FunctionOpInterface func) {

  // Populate the function arguments attributes.
  for (unsigned i = 0; i < func.getNumArguments(); i++) {
    BoundsAttr srcAttr = cast<BoundsAttr>(op.getInputAttrs()[i]);
    // We may have scalar (index|signless int)-typed values since we haven't
    // eliminated `plan.(with_shape|with_values)` ops yet.
    if (!op.argHasTensorType(i) || srcAttr.isNone())
      continue;
    FailureOr<tensorrt::ShapeProfileAttr> boundAttr =
        getTensorRTShapeProfile(srcAttr, op.getInputs()[i]);
    if (failed(boundAttr))
      return op->emitOpError("failed to create TensorRT shape profile "
                             "attribute from Plan BoundsAttr for argument #")
             << i << " (" << srcAttr << ")";
    if (srcAttr.isShapeBound()) {
      func.setArgAttr(i,
                      tensorrt::TensorRTDialect::getShapeProfileArgAttrName(),
                      *boundAttr);
      continue;
    }
    assert(srcAttr.isValueBound() && "expected value bound or shape bound");
    func.setArgAttr(
        i, tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName(),
        *boundAttr);
    func.setArgAttr(i, mlir::getHostTensorArgAttrName(),
                    rewriter.getUnitAttr());
  }

  // Populate the function result attributes if the call was a DPS kind.
  if constexpr (!std::is_same_v<RegionOpType, plan::InlineClosedAllocGroupOp>) {
    for (unsigned i = 0; i < func.getNumResults(); i++) {
      BoundsAttr srcAttr = cast<BoundsAttr>(op.getResAttrs()[i]);
      if (srcAttr.isNone())
        continue;
      FailureOr<tensorrt::ShapeProfileAttr> boundsAttr =
          getTensorRTShapeProfile(srcAttr, op.getResults()[i]);
      if (failed(boundsAttr))
        return op->emitOpError("failed to create TensorRT shape profile "
                               "attribute from Plan BoundsAttr for result #")
               << i << " (" << srcAttr << ")";
      if (srcAttr.isShapeBound()) {
        func.setResultAttr(
            i, tensorrt::TensorRTDialect::getShapeProfileArgAttrName(),
            *boundsAttr);
        continue;
      }
      assert(srcAttr.isValueBound() && "expected value bound or shape bound");
      func.setResultAttr(
          i, tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName(),
          *boundsAttr);
      func.setResultAttr(i, mlir::getHostTensorArgAttrName(),
                         rewriter.getUnitAttr());
    }
  }

  return success();
}

/// Create a `func.func` operation that represents `regionOp` and inserts into
/// the `module` SymbolTable. The function is given a name starting with
/// `nameBase` but may have numbers appended in order to unique the name. The
/// created function has argument/result types as indicated by the parameters.
static FailureOr<FunctionOpInterface>
createOutlinedFunc(RewriterBase &rewriter, Location loc, Operation *regionOp,
                   Operation *module, StringRef nameBase, StringRef tagName,
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
  SymbolTable(module).insert(outlinedFunc,
                             module->getRegions().front().front().end());

  // Tag the function with a UnitAttr for identifying the different kinds of
  // functions based on the cluster type.
  outlinedFunc->setAttr(tagName, rewriter.getUnitAttr());
  return cast<FunctionOpInterface>(outlinedFunc.getOperation());
}

/// Given the `op`, find the closest ModuleOp and check if the module has a
/// `tensorrt.module` operation in it. If it does, then return the existing
/// `tensorrt.module` operation. Otherwise, create a new `tensorrt.module`.
static tensorrt::TensorRTModuleOp getOrCreateTensorRTModuleOp(Operation *op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;
  SymbolTable symbolTable(moduleOp);
  tensorrt::TensorRTModuleOp result = nullptr;
  for (auto kernelModuleOp :
       moduleOp.getBody()->getOps<tensorrt::TensorRTModuleOp>()) {
    result = kernelModuleOp;
    break;
  }
  if (result)
    return result;

  // Create the function. Symbol name de-duplication occurs with insert into the
  // symbol table.
  result = tensorrt::TensorRTModuleOp::create(moduleOp.getLoc(), "trt_engines");
  symbolTable.insert(result, op->getParentOp() == moduleOp ? Block::iterator(op)
                                                           : Block::iterator{});
  return result;
}

/// Given a closed group operation, outline the body and replace with a call
/// operation (variant of call is specified using the template parameter). For
/// DPS variants, we explicitly allocate the create the DPS result tensors prior
/// to the call invocation.
template <typename RegionOpType = plan::InlineClosedGroupOp>
static LogicalResult outlineTensorRTRegion(RewriterBase &rewriter,
                                           RegionOpType op) {
  tensorrt::TensorRTModuleOp trtModuleOp = getOrCreateTensorRTModuleOp(op);
  auto funcArgTypes = llvm::to_vector(TypeRange(op.getInputs()));
  FailureOr<FunctionOpInterface> func = createOutlinedFunc(
      rewriter, op.getLoc(), op, trtModuleOp, "tensorrt_cluster",
      "cluster.tensorrt", TypeRange(op.getInputs()),
      op.getYield()->getOperandTypes());
  if (failed(func))
    return failure();
  assert(func->getFunctionBody().getBlocks().size() == 1 &&
         "expected body with one block");
  func->setPublic();

  rewriter.setInsertionPoint(op);
  CallOpInterface callOp;

  constexpr bool isNonDPSVariant =
      std::is_same_v<RegionOpType, plan::InlineClosedAllocGroupOp>;

  if constexpr (!isNonDPSVariant)
    callOp = cast<CallOpInterface>(*rewriter.create<tensorrt::CallOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), op.getOuts(),
        SymbolRefAttr::get(trtModuleOp.getNameAttr(),
                           {FlatSymbolRefAttr::get(*func)})));
  else
    callOp = cast<CallOpInterface>(*rewriter.create<tensorrt::CallAllocOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(),
        SymbolRefAttr::get(trtModuleOp.getNameAttr(),
                           {FlatSymbolRefAttr::get(*func)})));

  // Populate the function arguments attributes.
  if (failed(populateFunctionAttributes(rewriter, op, *func)))
    return failure();

  // Populate the function entry block.
  rewriter.eraseBlock(&func->getFunctionBody().front());

  // Move private decomposition funcs associated with all `stablehlo.composite`
  // ops to the `tensorrt.module` op. This is needed since `tensorrt.module` op
  // has its own symbol table.
  SymbolTableCollection symbolTable;
  for (auto compositeOp :
       op.getBody().template getOps<stablehlo::CompositeOp>()) {
    auto decompositionFunc = dyn_cast_if_present<func::FuncOp>(
        symbolTable.lookupSymbolIn(op->template getParentOfType<ModuleOp>(),
                                   compositeOp.getDecompositionAttr()));
    if (!decompositionFunc)
      return emitError(compositeOp.getLoc())
             << "failed to lookup stablehlo.composite decomposition "
                "function: "
             << compositeOp.getDecompositionAttr();
    rewriter.moveOpAfter(decompositionFunc, func->getOperation());
  }

  // Move region op operations to the func body.
  Operation *regionYieldOp = op.getYield();
  rewriter.inlineRegionBefore(op.getRegion(), func->getFunctionBody(),
                              func->getFunctionBody().end());
  rewriter.setInsertionPoint(regionYieldOp);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(regionYieldOp,
                                              regionYieldOp->getOperands());

  // Erase the DPS arugments, which now should be unused.
  if constexpr (!isNonDPSVariant) {
    if (llvm::any_of(func->getArguments().take_back(op.getOuts().size()),
                     [](BlockArgument arg) { return !arg.use_empty(); }))
      return failure();
    func->getFunctionBody().front().eraseArguments(op.getInputs().size(),
                                                   op.getOuts().size());
  }

  // replace the original region results.
  rewriter.replaceOp(op, callOp);
  return success();
}

bool TensorRTBackendAttr::requiresClosure(InputKind) const { return true; }

LogicalResult TensorRTBackendAttr::outlineClosedCluster(
    InputKind inputKind, RewriterBase &rewriter, Operation *op,
    SymbolTable &moduleSymbolTable) const {
  if (auto group = llvm::dyn_cast<plan::InlineClosedGroupOp>(op)) {
    if (failed(outlineTensorRTRegion<InlineClosedGroupOp>(rewriter, group)))
      return failure();
    return success();
  }
  if (auto allocGroup = llvm::dyn_cast<plan::InlineClosedAllocGroupOp>(op)) {
    if (failed(outlineTensorRTRegion<InlineClosedAllocGroupOp>(rewriter,
                                                               allocGroup)))
      return failure();
    return success();
  }
  return failure();
}

bool TensorRTBackendAttr::supportsInputKind(InputKind inputKind) const {
  if (inputKind == InputKind::Stablehlo) {
#ifdef MLIR_TRT_ENABLE_HLO
    return true;
#else
    return false;
#endif
  }
  return inputKind == InputKind::TensorRT;
}

//===----------------------------------------------------------------------===//
// Extension Registration
//===----------------------------------------------------------------------===//

namespace {
class PlanDialectTensorRTBackend
    : public plan::PlanDialectExtension<PlanDialectTensorRTBackend> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanDialectTensorRTBackend)

  void init() {
    (void)&generatedAttributeParser;
    (void)&generatedAttributePrinter;
    registerAttributes<plan::TensorRTBackendAttr>();
  }
};
} // namespace

void mlir::plan::registerTensorRTBackend(DialectRegistry &registry) {
  registry.addExtensions<PlanDialectTensorRTBackend>();
}
