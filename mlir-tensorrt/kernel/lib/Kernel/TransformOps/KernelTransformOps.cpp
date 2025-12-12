//===- KernelTransformOps.cpp ---------------------------------------------===//
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
/// Definitions for Kernel dialect extension ops for the Transform dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir-kernel/Utils/OutliningUtils.h"
#include "mlir-kernel/Utils/TensorUtils.h"
#include "mlir-tensorrt-common/Utils/RegionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// KernelVectorizeChildrenAndApplyPatternsOp
// TODO: the code for this operation is a clone of usptream code.
// prior to Upstream PR 111349. Remove KernelVectorizeChildrenAndApplyPatternsOp
// when the issue 966 is resolved.
//===----------------------------------------------------------------------===//

void transform::KernelVectorizeChildrenAndApplyPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    bool vectorizePadding, bool vectorizeExtract, bool flatten1DDepthwiseConv) {
  result.addOperands(target);
  if (vectorizePadding) {
    result.addAttribute(
        KernelVectorizeChildrenAndApplyPatternsOp::getVectorizePaddingAttrName(
            result.name),
        builder.getUnitAttr());
  }
  if (vectorizeExtract) {
    result.addAttribute(KernelVectorizeChildrenAndApplyPatternsOp::
                            getVectorizeNdExtractAttrName(result.name),
                        builder.getUnitAttr());
  }
  if (flatten1DDepthwiseConv) {
    result.addAttribute(KernelVectorizeChildrenAndApplyPatternsOp::
                            getFlatten_1dDepthwiseConvAttrName(result.name),
                        builder.getUnitAttr());
  }
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

namespace {
/// This is an helper only to call vectorize via a pattern inside of
/// VectorizeChildrenAndApplyPatternsOp::applyToOne.
struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context,
                                bool vectorizeExtract = false,
                                bool flattenConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeExtract),
        flatten1DDepthwiseConv(flattenConv) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::hasVectorizationImpl(op) || isa<tensor::InsertSliceOp>(op) ||
        isa<linalg::ConvolutionOpInterface>(op))
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Op, cannot vectorize");
    if (isa_and_present<memref::MemRefDialect>(op->getDialect()))
      return failure();
    return mlir::linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                                   /*inputScalableVecDims=*/{},
                                   vectorizeNDExtract, flatten1DDepthwiseConv);
  }

private:
  /// Controls whether to vectorize `tensor.extract` when the input tensor is
  /// rank >= 2.
  bool vectorizeNDExtract = false;
  /// Controls whether to "flatten" the channel dimension when vectorising 1D
  /// depthwise convolutions. This should lead to bette vectorization for
  /// tensors with a low number of channel dimensions.
  bool flatten1DDepthwiseConv = false;
};
} // namespace

DiagnosedSilenceableFailure
transform::KernelVectorizeChildrenAndApplyPatternsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<VectorizationPattern>(ctx, getVectorizeNdExtract(),
                                     getFlatten_1dDepthwiseConv());

  if (!getDisableTransferPermutationMapLoweringPatterns())
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  if (!getDisableMultiReductionToContractPatterns())
    vector::populateVectorReductionToContractPatterns(patterns);

  vector::populateSinkVectorOpsPatterns(patterns);

  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

  if (getVectorizePadding())
    linalg::populatePadOpVectorizationPatterns(patterns);
  vector::populateVectorStepLoweringPatterns(patterns);

  TrackingListener listener(state, *this);
  GreedyRewriteConfig config{};
  config.listener = &listener;
  config.maxIterations = 2;

  // We don't care about convergence failure here. Some upstream vectorization
  // patterns are not well-behaved in not creating new ops if returning failure.
  (void)applyPatternsGreedily(target, std::move(patterns), config);

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// NestScalarLinalgInForallOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure NestScalarLinalgInForallOp::applyToOne(
    transform::TransformRewriter &rewriter, DestinationStyleOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  OperandRange outputs = target.getDpsInits();
  SmallVector<SmallVector<Range>> insertionParams;
  for (Value initArg : outputs) {
    auto tensorType = dyn_cast<RankedTensorType>(initArg.getType());
    if (!tensorType)
      return emitDefaultDefiniteFailure(target);
    SmallVector<Range> &ranges = insertionParams.emplace_back();
    ranges.reserve(tensorType.getRank());
    for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
      if (!ShapedType::isDynamic(dim)) {
        ranges.push_back(Range{rewriter.getIndexAttr(0),
                               rewriter.getIndexAttr(dim),
                               rewriter.getIndexAttr(1)});
        continue;
      }
      Value size = rewriter.create<tensor::DimOp>(
          initArg.getLoc(), initArg,
          rewriter.create<arith::ConstantIndexOp>(initArg.getLoc(), idx));
      ranges.push_back(
          Range{rewriter.getIndexAttr(0), size, rewriter.getIndexAttr(1)});
    }
  }

  IRMapping mapping;
  DestinationStyleOpInterface innerOp{};
  auto forallOp = rewriter.create<scf::ForallOp>(
      target.getLoc(), ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
      outputs, /*mapping=*/std::nullopt,
      [&](OpBuilder &b, Location loc, ValueRange regionArgs) {
        const unsigned numThreadIdRegionArgs = 1;
        unsigned numOutputRegionArgs =
            regionArgs.size() - numThreadIdRegionArgs;
        ValueRange outputArgs = regionArgs.take_back(numOutputRegionArgs);
        mapping.map(outputs, outputArgs);
        innerOp = cast<DestinationStyleOpInterface>(
            b.clone(*target.getOperation(), mapping));
        auto termOp = b.create<scf::InParallelOp>(loc);
        b.setInsertionPointToStart(termOp.getBody());
        for (unsigned idx = 0; idx < innerOp->getNumResults(); idx++)
          b.create<tensor::ParallelInsertSliceOp>(
              loc, Value(innerOp->getResult(idx)), outputArgs[idx],
              insertionParams[idx]);
      });
  forallOp.setMappingAttr(
      rewriter.getArrayAttr({rewriter.getAttr<mlir::gpu::GPUBlockMappingAttr>(
          mlir::gpu::MappingId::DimX)}));
  rewriter.replaceOp(target, forallOp->getResults());
  results.push_back(forallOp);
  results.push_back(innerOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// VerifyPostTilingOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
VerifyPostTilingOp::applyToOne(transform::TransformRewriter &rewriter,
                               scf::ForallOp target,
                               transform::ApplyToEachResultList &results,
                               transform::TransformState &state) {

  auto getForall = [&](Block *block) {
    llvm::SmallSetVector<scf::ForallOp, 4> result;
    block->walk<WalkOrder::PreOrder>([&](scf::ForallOp nested) {
      result.insert(nested);
      return WalkResult::skip();
    });
    return result;
  };

  auto nestedForall = getForall(target.getBody());
  if (nestedForall.size() >= 2) {
    for (auto nested : nestedForall.getArrayRef()) {
      nested.emitError("generated kernel is not thread safe");
      return emitDefaultDefiniteFailure(target);
    }
  }

  if (!nestedForall.empty()) {
    auto walkResult =
        target.getBody()->walk<WalkOrder::PreOrder>([&](Operation *nested) {
          if (isa<scf::ForallOp>(nested))
            return WalkResult::skip();
          if (isa<linalg::LinalgOp, vector::TransferReadOp,
                  vector::TransferWriteOp, tensor::InsertOp, tensor::ExtractOp>(
                  nested)) {
            auto diag = nested->emitError("kernel constains undistributed "
                                          "compute/memory ops after tiling");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return emitDefaultDefiniteFailure(target);
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::VerifyPostTilingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ForallToKernelOp
//===----------------------------------------------------------------------===//

/// Find or insert a `gpu.module` into the module.
/// If `reuseExistingGpuModule` is true, then the kernel function will
/// be inserted into the first existing `gpu.module` that is found. If
/// no existing `gpu.module` is found or if `reuseExistingGpuModule` is
/// false, then a new `gpu.module` will be created.
static mlir::gpu::GPUModuleOp
getOrCreateGpuModuleOp(ModuleOp moduleOp, StringRef moduleNamePrefix,
                       bool reuseExistingGpuModule) {
  SymbolTable symbolTable(moduleOp);
  OpBuilder builder(moduleOp->getContext());

  // Try to find an existing `gpu.module` operation.
  if (reuseExistingGpuModule) {
    for (auto kernelModuleOp :
         moduleOp.getBody()->getOps<mlir::gpu::GPUModuleOp>())
      return kernelModuleOp;
  }

  // Create the function. Symbol name de-duplication occurs with insert into the
  // symbol table.
  auto result = builder.create<mlir::gpu::GPUModuleOp>(
      moduleOp.getLoc(), moduleNamePrefix, ArrayAttr{});
  symbolTable.insert(result);
  return result;
}

/// Return the prefix name for the kernel. This is "[parent func name]_kernel".
/// TODO: We can add a more complicated naming scheme by looking at the contents
/// and deducing the kind of operation (e.g. matmul, reduction, etc).
static std::string getKernelBaseName(scf::ForallOp op) {
  auto parentSymbol = op->getParentOfType<SymbolOpInterface>();
  if (!parentSymbol)
    return "kernel";
  return (parentSymbol.getName() + "_kernel").str();
}

/// Compose and apply an affine map that computes the product of a set of basis
/// values. This is used to compute the total number of elements in a
/// multi-dimensional grid, e.g., for flattening multi-dimensional indices into
/// a single linear index.
static OpFoldResult
makeComposedApplyReduceProduct(RewriterBase &rewriter, Location loc,
                               ArrayRef<OpFoldResult> basis) {
  MLIRContext *ctx = rewriter.getContext();
  // Create a list of affine symbols, one for each basis element.
  auto symbolExprs = llvm::to_vector(
      llvm::map_range(llvm::seq<unsigned>(0, basis.size()), [&](unsigned idx) {
        return getAffineSymbolExpr(idx, ctx);
      }));
  // Compute the product expression using the max linear index utility.
  AffineExpr productExpr = mlir::computeMaxLinearIndex(ctx, symbolExprs);
  // Apply the composed affine map to the basis values.
  return affine::makeComposedFoldedAffineApply(rewriter, loc, productExpr,
                                               basis);
}

/// Create a remapping of the basis elements to 3D grid dimensions. This is used
/// to create the block/grid dimensions for the kernel launch. The remapping is
/// created by distributing each basis element into one of three segments. We do
/// this greedily, trying to reduce the maximum volume of the segments when
/// there are >3 basis elements. For dynamic sizes, we treat the size as "1024"
/// for the purposes of the segmentation. This could be improved with better
/// bounds information.
/// The reason we try to reduce the maximum volume of the segments is because
/// we don't want to overflow any one particular launch dimension.
/// This is all heuristic and ideally we can hoist this up to force callers to
/// explicitly specify the mapping of the `scf.forall` induction variables to
/// the 3D grid dimensions.
static FailureOr<std::array<SmallVector<unsigned>, 3>>
createBasisTo3DRemaping(ArrayRef<OpFoldResult> basis) {
  std::array<SmallVector<unsigned>, 3> mapping;
  std::array<SmallVector<OpFoldResult>, 3> segments;
  SmallVector<std::optional<int64_t>> constVals =
      llvm::map_to_vector(basis, getConstantIntValue);

  auto getSegmentVolume = [&](ArrayRef<OpFoldResult> segment) -> uint64_t {
    uint64_t product = 1;
    for (OpFoldResult ofr : segment) {
      if (auto constVal = getConstantIntValue(ofr))
        product *= static_cast<uint64_t>(*constVal);
      else
        product *= 1024;
    }
    return product;
  };

  for (auto [i, ofr] : llvm::enumerate(basis)) {
    unsigned segmentIdx = 0;
    uint64_t minVolume = std::numeric_limits<uint64_t>::max();
    for (unsigned i = 0; i < segments.size(); ++i) {
      uint64_t volume = getSegmentVolume(segments[i]);
      if (volume < minVolume) {
        minVolume = volume;
        segmentIdx = i;
      }
    }
    segments[segmentIdx].push_back(ofr);
    mapping[segmentIdx].push_back(i);
  }

  return mapping;
}

namespace {

/// Helper struct to map block/grid dimensions for GPU kernel launches.
/// This is used to generate the correct block IDs and grid sizes for
/// outlining `scf.forall` operations to GPU kernels.
struct BlockIdMapper {
  /// Construct a BlockIdMapper with the given upper bounds for each dimension.
  BlockIdMapper(ArrayRef<OpFoldResult> mixedUBs) : forallUBs(mixedUBs) {
    FailureOr<std::array<SmallVector<unsigned>, 3>> mapping =
        createBasisTo3DRemaping(forallUBs);
    if (failed(mapping))
      return;
    basisTo3DMap = *mapping;
    constForallUBs = llvm::map_to_vector(forallUBs, getConstantIntValue);
  }

  /// The upper bounds for each dimension of the forall.
  SmallVector<OpFoldResult> forallUBs;
  SmallVector<std::optional<int64_t>> constForallUBs;
  std::array<SmallVector<unsigned>, 3> basisTo3DMap;

  /// Create the replacement induction variable values for the kernel body.
  /// For up to 3 dimensions, this maps to GPU block IDs (x, y, z).
  /// For more than 3 dimensions, a single delinearized block ID is used.
  SmallVector<Value> createIVReplacements(RewriterBase &rewriter,
                                          Location loc) {
    static constexpr std::array<gpu::Dimension, 3> dimensions = {
        gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};

    SmallVector<Value> results(forallUBs.size(), Value{});
    for (auto [i, indices] : llvm::enumerate(basisTo3DMap)) {
      SmallVector<AffineExpr> strides =
          mlir::computeSuffixProduct(llvm::map_to_vector(
              llvm::seq<unsigned>(indices.size()),
              [&](unsigned i) { return rewriter.getAffineSymbolExpr(i + 1); }));
      SmallVector<AffineExpr> delinearized =
          mlir::delinearize(rewriter.getAffineSymbolExpr(0), strides);
      SmallVector<OpFoldResult> operands = {
          rewriter
              .create<mlir::gpu::BlockIdOp>(loc, rewriter.getIndexType(),
                                            dimensions[i])
              .getResult()};
      for (unsigned i : indices)
        operands.push_back(forallUBs[i]);
      SmallVector<OpFoldResult> ids =
          affine::makeComposedFoldedMultiResultAffineApply(
              rewriter, loc,
              AffineMap::get(0, indices.size() + 1, delinearized,
                             rewriter.getContext()),
              operands);
      for (auto [j, ofr] : llvm::zip_equal(indices, ids))
        results[j] = getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
    }

    return results;
  }

  /// Create the grid dimensions for the kernel launch.
  /// For up to 3 dimensions, returns the grid size for each dimension.
  /// For more than 3 dimensions, returns a single value representing the total
  /// number of blocks.
  SmallVector<Value> createGridDimensions(RewriterBase &rewriter,
                                          Location loc) {
    SmallVector<Value> result;
    SmallVector<AffineExpr> exprs = llvm::map_to_vector(
        llvm::seq<unsigned>(forallUBs.size()),
        [&](unsigned i) { return rewriter.getAffineSymbolExpr(i); });
    for (auto [i, indices] : llvm::enumerate(basisTo3DMap)) {
      AffineExpr expr = rewriter.getAffineConstantExpr(1);
      for (unsigned j : indices)
        expr = expr * exprs[j];
      result.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc,
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, AffineMap::get(0, exprs.size(), expr),
              forallUBs)));
    }
    if (result.size() < 3)
      result.append(3 - result.size(),
                    rewriter.create<arith::ConstantIndexOp>(loc, 1));
    return result;
  }

  /// Get the known grid size as a DenseI32ArrayAttr, if all upper bounds are
  /// constant. For up to 3 dimensions, returns the grid shape padded to 3D. For
  /// more than 3 dimensions, returns {product, 1, 1}. Returns an empty
  /// attribute if any bound is dynamic.
  DenseI32ArrayAttr getKnownGridSizeAttr(RewriterBase &rewriter) {
    if (!llvm::all_of(constForallUBs,
                      [](std::optional<int64_t> ub) { return ub.has_value(); }))
      return {};
    SmallVector<int32_t> knownGridShape;
    for (auto [i, indices] : llvm::enumerate(basisTo3DMap)) {
      uint32_t acc = 1;
      for (unsigned j : indices)
        acc *= static_cast<uint32_t>(*constForallUBs[j]);
      knownGridShape.push_back(static_cast<int32_t>(acc));
    }
    // Pad to 3D if necessary.
    if (knownGridShape.size() < 3)
      knownGridShape.append(3 - knownGridShape.size(), 1);
    return rewriter.getDenseI32ArrayAttr(knownGridShape);
  }
};

} // namespace

/// This function determines whether to clone producer operations when outlining
/// an `scf.forall` operation to a kernel function. We choose to clone
/// `tensor.empty` (which otherwise may create unnecessary kernel arguments)
/// and any scalar/vector-typed constants into the kernel region.
static bool shouldCloneIntoKernelRegion(Operation *op) {
  if (isa<tensor::EmptyOp>(op))
    return true;
  if (op->hasTrait<OpTrait::ConstantLike>() &&
      isa<IntegerType, FloatType, IndexType, ComplexType, VectorType>(
          op->getResult(0).getType()))
    return true;
  return false;
}

/// Outlines the body of an `scf.forall` operation to a new `func.func` nested
/// under a `gpu.module` operation. The `scf.forall` operation is replaced by
/// a `kernel.call` operation that represents "launching" the kernel. The block
/// arguments for the `scf.forall` body representing the indices of the
/// processing elements are replaced by `gpu.block_id` operations inserted at
/// the start of the body block. Up to three indices are replaced by
/// `gpu.block_id` with the position arguments `x`, `y`, and `z` corresponding
/// to the first, second, and third block arguments representing processing
/// element indices. Having more than three distribution indices is considered a
/// definite error. All `shared_outs` parameters and well as used values defined
/// above become arguments to the created `func.func`. The only exception is
/// scalar and vector constants, which are inlined into the body The result is a
/// `func.func` that represents a thead-block level SPMD program.
static FailureOr<ForallOutliningResult>
forallToCTAs(RewriterBase &rewriter, scf::ForallOp forall,
             mlir::gpu::GPUModuleOp gpuModule, StringRef kernelNamePrefix,
             int64_t numThreads) {

  // Get the total linear grid size.
  BlockIdMapper blockIdMapper(forall.getMixedUpperBound());
  SmallVector<Value> gridShape =
      blockIdMapper.createGridDimensions(rewriter, forall.getLoc());

  SymbolTable gpuModuleSymbolTable(gpuModule);
  FailureOr<ForallOutliningResult> forallOutlineResult = mlir::outlineForall(
      rewriter, forall, kernelNamePrefix, gpuModuleSymbolTable,
      /*ivReplacementBuilder=*/
      [&](RewriterBase &rewriter, Location loc, ValueRange ivs,
          ArrayRef<OpFoldResult> ubs, std::optional<ArrayAttr> attr) {
        return blockIdMapper.createIVReplacements(rewriter, loc);
      },
      /*callBuilder=*/
      [&](RewriterBase &rewriter, scf::ForallOp forall, ValueRange args,
          func::FuncOp callee) -> Operation * {
        Value blockSize = rewriter.create<arith::ConstantIndexOp>(
            forall.getLoc(), numThreads);
        return rewriter.create<kernel::CallOp>(
            forall.getLoc(), forall->getResultTypes(), gridShape,
            ValueRange{blockSize}, args.drop_back(forall->getNumResults()),
            args.take_back(forall->getNumResults()),
            SymbolRefAttr::get(
                gpuModule.getSymNameAttr(),
                {FlatSymbolRefAttr::get(callee.getSymNameAttr())}));
      },
      shouldCloneIntoKernelRegion);
  if (failed(forallOutlineResult))
    return failure();

  // Mark the number of warps as an attribute of the new kernel function.
  forallOutlineResult->outlinedBody->setAttr(
      kernel::KernelDialect::getKernelFunctionNumThreadsAttrName(),
      rewriter.getI64IntegerAttr(numThreads));
  forallOutlineResult->outlinedBody->setAttr(
      gpu::GPUDialect::getKernelFuncAttrName(),
      UnitAttr::get(forallOutlineResult->outlinedBody->getContext()));

  // The GPU dialect defines discardable attributes `gpu.known_block_size` and
  // `gpu.known_grid_size` that can be used to specify the block and grid size
  // on the outlined function. These attributes are used by the implementations
  // of InferIntRangeOpInterface for all the `gpu.[block|grid]_dim` and
  // `gpu.[thread|block]_id` opserations.
  mlir::gpu::GPUDialect::KnownBlockSizeAttrHelper(rewriter.getContext())
      .setAttr(forallOutlineResult->outlinedBody,
               rewriter.getDenseI32ArrayAttr(
                   {static_cast<int32_t>(numThreads), 1, 1}));

  if (DenseI32ArrayAttr knownGridSizeAttr =
          blockIdMapper.getKnownGridSizeAttr(rewriter)) {
    mlir::gpu::GPUDialect::KnownGridSizeAttrHelper(rewriter.getContext())
        .setAttr(forallOutlineResult->outlinedBody, knownGridSizeAttr);
  }

  return forallOutlineResult;
}

DiagnosedSilenceableFailure
ForallToKernelOp::applyToOne(transform::TransformRewriter &rewriter,
                             scf::ForallOp forall,
                             transform::ApplyToEachResultList &results,
                             transform::TransformState &state) {
  // Top-level forall become kernels.
  if (!forall.isNormalized()) {
    forall.emitOpError() << "not compatible with SPMD transformation";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  if (forall->getParentOfType<scf::ForallOp>()) {
    results.assign(3, nullptr);
    return DiagnosedSilenceableFailure::success();
  }
  if (!isa<func::FuncOp>(forall->getParentOp())) {
    Operation *parent = forall->getParentOp();
    emitError(
        "cannot convert an scf.forall operation into a kernel function if "
        "the parent is not a function")
            .attachNote(parent->getLoc())
        << "see parent " << *parent;
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  // If we are allowing re-use of GPU modules, then then name should be like
  // ```
  // gpu.module @kernels {
  //    func.func @func1_kernel1 {... }
  //    func.func @func2_kernel2 {... }
  // }
  // ```
  //
  // If we are not allowing re-use, then the name should be like
  //
  // ```
  // gpu.module @func1_kernel1 {
  //    func.func @kernel {... }
  // }
  // gpu.module @func2_kernel2 {
  //    func.func @kernel {... }
  // }
  // ```
  std::string gpuModuleNamePrefix =
      getReuseExistingGpuModule() ? "kernels" : getKernelBaseName(forall);
  std::string kernelNamePrefix =
      getReuseExistingGpuModule() ? getKernelBaseName(forall) : "kernel";

  mlir::gpu::GPUModuleOp gpuModule =
      getOrCreateGpuModuleOp(forall->getParentOfType<ModuleOp>(),
                             gpuModuleNamePrefix, getReuseExistingGpuModule());

  if (getExtraModuleAttrsAttr() && !getReuseExistingGpuModule())
    for (NamedAttribute attr : getExtraModuleAttrsAttr())
      gpuModule->setAttr(attr.getName(), attr.getValue());

  FailureOr<ForallOutliningResult> result = forallToCTAs(
      rewriter, forall, gpuModule, kernelNamePrefix, getNumThreads());
  if (failed(result)) {
    forall->emitOpError() << "failed to create device kernel/SPDM transform";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  results.push_back(result->forallReplacement);
  results.push_back(gpuModule);
  results.push_back(result->outlinedBody);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ForallToSubgroupsOp
//===----------------------------------------------------------------------===//

/// Given the `scf.in_parallel` terminator, create and returns some equivalent
/// `tensor.insert_slice` operations to replace it.
static SmallVector<Value>
forallTerminatorToInsertSlices(RewriterBase &rewriter,
                               scf::InParallelOp terminator) {
  SmallVector<Value> returnedValues;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(terminator);
  for (Operation &yieldingOp : terminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(yieldingOp);
    auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        insertOp.getLoc(), insertOp.getSource(), insertOp.getDest(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());
    returnedValues.push_back(insertSliceOp);
  }
  return returnedValues;
}

static FailureOr<Operation *> forallToSubgroups(RewriterBase &rewriter,
                                                scf::ForallOp forall,
                                                int64_t subgroupSize) {
  rewriter.setInsertionPoint(forall);

  // We assume that the CUDA threadblocks created are of the shape (NumThreads,
  // 1, 1).
  Value threadIdx = rewriter.create<mlir::gpu::ThreadIdOp>(
      forall.getLoc(), mlir::gpu::Dimension::x);
  // Get the current warp/subgroup/thread id by
  // (x mod (prod(loop_upper_bound) * group_size)) floor_div group_size
  Location loc = forall->getLoc();
  SmallVector<Value> basis = forall.getUpperBound(rewriter);
  OpFoldResult basisProd =
      makeComposedApplyReduceProduct(rewriter, loc, getAsOpFoldResult(basis));

  OpFoldResult groupId = affine::makeComposedFoldedAffineApply(
      rewriter, loc,
      (rewriter.getAffineSymbolExpr(0) %
       (rewriter.getAffineSymbolExpr(1) * subgroupSize))
          .floorDiv(subgroupSize),
      {threadIdx, basisProd});
  // Then delinearize into the correct basis for the forall variables.
  auto delinOp = rewriter.create<affine::AffineDelinearizeIndexOp>(
      forall.getLoc(), getValueOrCreateConstantIndexOp(rewriter, loc, groupId),
      getAsOpFoldResult(basis));

  // Get returned values and fixup terminator.
  SmallVector<Value> returnedValues =
      forallTerminatorToInsertSlices(rewriter, forall.getTerminator());
  rewriter.eraseOp(forall.getTerminator());

  // Inline the body.
  Block *parentBlock = forall->getBlock();
  parentBlock->getOperations().splice(Block::iterator(forall),
                                      forall.getBody()->getOperations());

  // Remap induction vars.
  rewriter.setInsertionPointToStart(parentBlock);
  for (auto [idx, v] : llvm::enumerate(forall.getInductionVars()))
    rewriter.replaceAllUsesWith(v, delinOp->getResult(idx));
  for (auto [idx, v] : llvm::enumerate(forall.getOutputsMutable()))
    rewriter.replaceAllUsesWith(forall.getTiedBlockArgument(&v), v.get());
  assert(forall->getNumResults() == returnedValues.size());
  rewriter.replaceOp(forall, returnedValues);
  return delinOp.getOperation();
}

DiagnosedSilenceableFailure
ForallToSubgroupsOp::applyToOne(transform::TransformRewriter &rewriter,
                                scf::ForallOp forall,
                                transform::ApplyToEachResultList &results,
                                transform::TransformState &state) {
  // Top-level forall become kernels.
  if (!forall.isNormalized()) {
    forall.emitOpError() << "not compatible with SPMD transformation";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // Don't change the nested foralls
  if (forall->getParentOfType<scf::ForallOp>()) {
    results.push_back(nullptr);
    return DiagnosedSilenceableFailure::success();
  }
  FailureOr<Operation *> result =
      forallToSubgroups(rewriter, forall, getSubgroupSize());
  if (failed(result))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(*result);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FuseGreedilyOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
FuseGreedilyOp::applyToOne(transform::TransformRewriter &rewriter,
                           Operation *target, ApplyToEachResultList &results,
                           TransformState &state) {
  rewriter.setInsertionPoint(target);
  LogicalResult fuseResult = tensor_ext::fuseGreedily(target, rewriter);

  if (failed(fuseResult))
    return emitDefaultSilenceableFailure(target);
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void FuseGreedilyOp::build(OpBuilder &builder, OperationState &state,
                           Value target) {
  return FuseGreedilyOp::build(builder, state, target.getType(), target);
}

//===----------------------------------------------------------------------===//
// LowerToLoopsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
LowerToLoopsOp::applyToOne(transform::TransformRewriter &rewriter,
                           ToLoopsOpInterface target,
                           transform::ApplyToEachResultList &results,
                           transform::TransformState &state) {
  if (!target)
    return emitSilenceableError() << "no handle";

  FailureOr<LowerToLoopsResult> result = target.lowerToLoops(rewriter);
  if (failed(result))
    return emitDefaultSilenceableFailure(target) << "failed to lower to loops";
  rewriter.replaceOp(target.getOperation(), result->replacements);
  if (!result->loops.empty())
    results.push_back(result->loops.front());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ApplyRewriteVectorTransferReadToConstantPatternOp
//===----------------------------------------------------------------------===//

namespace {
struct RewriteVectorTransferReadToConstantPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    return kernel::replaceVectorTransferReadWithConstant(rewriter, op);
  }
};
} // namespace

void ApplyRewriteVectorTransferReadToConstantPatternOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.add<RewriteVectorTransferReadToConstantPattern>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// InterchangeForallAndForOp
//===----------------------------------------------------------------------===//

namespace {
struct InterchangeForAndForallPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {

    return tensor_ext::interchangeForallAndFor(rewriter, op);
  }
};
} // namespace

void ApplyInterchangeForAndForallPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.add<InterchangeForAndForallPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ApplyAffineBoundsOptimizationPatternsOp
//===----------------------------------------------------------------------===//

void ApplyAffineBoundsOptimizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  kernel::populateAffineBoundsOptimizationPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Tablegen'd op definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//
namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class KernelTransformDialectExtension
    : public transform::TransformDialectExtension<
          KernelTransformDialectExtension> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelTransformDialectExtension)

  void init() {

    // Because we allow running various passes needed for codegen via
    // `transform.run_pipeline`, declare all the dialects used in codegen
    // here.
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<nvgpu::NVGPUDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<bufferization::BufferizationDialect>();
    declareGeneratedDialect<kernel::KernelDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    declareGeneratedDialect<NVVM::NVVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir-kernel/Kernel/TransformOps/KernelTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void kernel::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<KernelTransformDialectExtension>();
}
