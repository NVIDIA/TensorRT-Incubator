//===- Attributes.cpp -----------------------------------------------------===//
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
/// Definition of Kernel dialect attributes.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Attributes.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "kernel-dialect"
#define DBGS() (llvm::dbgs() << __FILE__ << ":" << __LINE__ << ": ")

using namespace mlir;
using namespace mlir::kernel;

//===----------------------------------------------------------------------===//
// Parsing Directives
//===----------------------------------------------------------------------===//

/// Parses tensor core information from the assembly format.
///
/// Expected format: `aType, bType, cType, dType, mxnxk[, sparse]`
///
/// @param parser The AsmParser instance
/// @param aType Output parameter for the A matrix type
/// @param bType Output parameter for the B matrix type
/// @param cType Output parameter for the C matrix type (accumulator input)
/// @param dType Output parameter for the D matrix type (result)
/// @param m Output parameter for the M dimension (rows of A and C/D)
/// @param n Output parameter for the N dimension (columns of B and C/D)
/// @param k Output parameter for the K dimension (columns of A, rows of B)
/// @param sparse Output parameter indicating if sparse tensor cores are used
/// @return Success if parsing succeeds, failure otherwise
static LogicalResult parseTensorCoreInfo(AsmParser &parser, Type &aType,
                                         Type &bType, Type &cType, Type &dType,
                                         int64_t &m, int64_t &n, int64_t &k,
                                         bool &sparse) {
  SmallVector<int64_t> dim;

  // Parse the four tensor types in order: A, B, C, D
  if (parser.parseType(aType) || parser.parseComma() ||
      parser.parseType(bType) || parser.parseComma() ||
      parser.parseType(cType) || parser.parseComma() ||
      parser.parseType(dType) || parser.parseComma())
    return failure();

  // Parse the MxNxK dimensions
  if (parser.parseDimensionList(dim, /*allowDynamic=*/false,
                                /*withTrailingX=*/false) ||
      dim.size() != 3)
    return failure();
  m = dim[0];
  n = dim[1];
  k = dim[2];

  // Parse optional sparse flag
  sparse = false;
  if (failed(parser.parseOptionalComma()))
    return success();
  if (parser.parseKeyword("sparse"))
    return failure();
  sparse = true;
  return success();
}

/// Prints tensor core information to the assembly format.
///
/// Output format: `aType, bType, cType, dType, mxnxk[, sparse]`
static void printTensorCoreInfo(AsmPrinter &printer, Type aType, Type bType,
                                Type cType, Type dType, int64_t m, int64_t n,
                                int64_t k, bool sparse) {
  // Print types
  for (Type t : {aType, bType, cType, dType})
    printer << t << ", ";

  // Print dimensions
  printer << m << "x" << n << "x" << k;

  // Print sparse flag if enabled
  if (sparse)
    printer << ", sparse";
}

/// Parses a comma-separated list of integers enclosed in square brackets.
///
/// Expected format: `[int1, int2, ..., intN]`
///
/// @param parser The AsmParser instance
/// @return A vector of parsed integers on success, failure otherwise
static FailureOr<SmallVector<int64_t>> parseIntArray(AsmParser &parser) {
  SmallVector<int64_t> ints;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square,
          [&]() {
            ints.push_back(0);
            return parser.parseInteger(ints.back());
          },
          "comma separated list of integers"))
    return failure();
  return ints;
}

/// Prints an array of integers in square bracket notation.
///
/// Output format: `[int1, int2, ..., intN]`
static void printIntArray(AsmPrinter &printer, ArrayRef<int64_t> ints) {
  printer << '[';
  llvm::interleaveComma(ints, printer);
  printer << ']';
}

//===----------------------------------------------------------------------===//
// MatMulShapeAttr
//===----------------------------------------------------------------------===//

// clang-format off
/// Verify the tensor core instruction info. This follows the verification used
/// by the NVGPU dialect. The verification is up-to-date through at least Ampere
/// instruction requirements. This verification does not support Volta MMA ops.
LogicalResult
TensorCoreInfoAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           Type aType, Type bType, Type cType, Type dType,
                           int64_t m, int64_t n, int64_t k, bool sparse) {
  // clang-format on
  // Tensor Core M/N dims should be either 8 or 16. They should form between 1
  // and 4 8x8 tiles (see above table). In the K dimension, they should have 1
  // or 2 128b tiles (or just 1 256 bit tile for f64).
  int operandBitwidth = aType.getIntOrFloatBitWidth();
  const int64_t shapeK = aType.isF64() ? 4 : 128 / operandBitwidth;
  auto validateSize = [&](int64_t m, int64_t n, int64_t k) {
    if ((m != 8 && m != 16) || (n != 8 && n != 16))
      return false;
    int64_t numTiles = (m / 8) * (n / 8);
    if (numTiles < 1 || numTiles > 4)
      return false;
    if (k % shapeK != 0)
      return false;
    numTiles = k / shapeK;
    if (k != shapeK && aType.isF64())
      return false;
    return numTiles <= 2 && numTiles >= 1;
  };
  if (!validateSize(m, n, k))
    return emitError()
           << "invalid matrix multiply-and-accumulate instruction shape";

  // F64 is not allowed in sparse mode.
  if (sparse && aType.isF64())
    return emitError() << "f64 is not supported for sparse mode";

  auto isValidMmaElType = [](Type t) {
    return t.isF32() || t.isBF16() || t.isF16() || t.isInteger(8) ||
           t.isInteger(4) || t.isInteger(1) || t.isF64();
  };

  for (Type t : {aType, bType, cType, dType}) {
    if (!isValidMmaElType(t))
      return emitError() << "valid data types are (i4,i8,f16,bf16,tf32,f64)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MatMulScheduleParametersAttr
//===----------------------------------------------------------------------===//

LogicalResult kernel::MatMulScheduleParametersAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    [[maybe_unused]] gpu::TargetAttrInterface gpuTargetInfo,
    llvm::ArrayRef<int64_t> CTATileSizes, int64_t numOfWarps,
    llvm::ArrayRef<int64_t> warpTileSizes,
    [[maybe_unused]] std::optional<int64_t> numOfStages,
    TensorCoreInfoAttr tensorCoreInfo) {
  // Check if the given num of warps is valid
  if (!(llvm::isPowerOf2_64(numOfWarps) && numOfWarps <= 8)) {
    emitError() << "Num of warps must be one of: 1, 2, 4, or 8, but got "
                << numOfWarps;
    return failure();
  }
  // Check if the given CTA tile sizes match the warp tile sizes dimension
  int64_t CTATileSizesLen = CTATileSizes.size();
  int64_t warpTileSizesLen = warpTileSizes.size();
  if (warpTileSizesLen > 0 && CTATileSizesLen > 0 &&
      CTATileSizesLen != warpTileSizesLen) {
    emitError() << "The warp tile size has dimension " << warpTileSizesLen
                << ", which doesn't match the CTA tile size of dimension "
                << CTATileSizesLen;
    return failure();
  }

  if (tensorCoreInfo &&
      failed(TensorCoreInfoAttr::verify(
          emitError, tensorCoreInfo.getAType(), tensorCoreInfo.getBType(),
          tensorCoreInfo.getCType(), tensorCoreInfo.getDType(),
          tensorCoreInfo.getM(), tensorCoreInfo.getN(), tensorCoreInfo.getK(),
          tensorCoreInfo.getSparse())))
    return failure();

  return success();
}

SmallVector<int64_t>
kernel::MatMulScheduleParametersAttr::getCTATileShape(linalg::LinalgOp) const {
  return llvm::to_vector(this->getCtaTileSizes());
}

//===----------------------------------------------------------------------===//
// ElementwiseScheduleParametersAttr
//===----------------------------------------------------------------------===//

LogicalResult kernel::ElementwiseScheduleParametersAttr::verify(
    function_ref<InFlightDiagnostic()>, gpu::TargetAttrInterface,
    ArrayRef<int64_t>, ArrayRef<int64_t>, ArrayRef<int64_t>) {

  return success();
}

SmallVector<int64_t> kernel::ElementwiseScheduleParametersAttr::getCTATileShape(
    linalg::LinalgOp linalgOp) const {

  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  ArrayRef<int64_t> gridShape = getGridShape();

  SmallVector<int64_t> ctaWorkloadShape;
  ctaWorkloadShape.reserve(gridShape.size());
  for (auto [idx, gridSize] : llvm::enumerate(gridShape)) {
    if (gridSize == 1) {
      ctaWorkloadShape.push_back(0);
      continue;
    }
    assert(loopRanges[idx] % ctaWorkloadShape.back() == 0 &&
           "The loop range is not divisible by the tile shape");
    ctaWorkloadShape.push_back(llvm::divideCeil(loopRanges[idx], gridSize));
  }
  return ctaWorkloadShape;
}

//===----------------------------------------------------------------------===//
// FallbackScheduleParametersAttr
//===----------------------------------------------------------------------===//

SmallVector<int64_t> kernel::FallbackScheduleParametersAttr::getCTATileShape(
    linalg::LinalgOp) const {
  return llvm::to_vector(getCtaWorkloadShape());
}

//===----------------------------------------------------------------------===//
// ScatterScheduleParametersAttr
//===----------------------------------------------------------------------===//

SmallVector<int64_t>
kernel::ScatterScheduleParametersAttr::getCTATileShape(linalg::LinalgOp) const {
  return llvm::to_vector(getCtaWorkloadShape());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void KernelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-kernel/Kernel/IR/Attributes.cpp.inc"
      >();

  declarePromisedInterface<kernel::GPUModuleLoweringAttrInterface,
                           kernel::DefaultGPUModuleKindAttr>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#define GET_ATTRDEF_CLASSES
#include "mlir-kernel/Kernel/IR/Attributes.cpp.inc"
#pragma GCC diagnostic pop
