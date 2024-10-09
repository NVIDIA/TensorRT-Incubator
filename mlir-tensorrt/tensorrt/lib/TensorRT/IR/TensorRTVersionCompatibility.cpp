#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tensorrt;

namespace {
/// Return the disjunction of the callables invoked with the element type of
/// `t`.
template <typename... Callable>
bool isType(Type t, Callable... funcs) {
  return (funcs(t) || ...);
}

auto I1 = [](Type t) { return t.isSignlessInteger(1); };
auto I4 = [](Type t) { return t.isSignlessInteger(4); };
auto UI8 = [](Type t) { return t.isUnsignedInteger(8); };
auto I8 = [](Type t) { return isTensorRTInt8Type(t); };
auto I32 = [](Type t) { return t.isSignlessInteger(32); };
auto I64 = [](Type t) { return t.isSignlessInteger(64); };
auto F8 = [](Type t) { return t.isFloat8E4M3FN(); };
auto F16 = [](Type t) { return t.isF16(); };
auto F32 = [](Type t) { return t.isF32(); };
auto BF16 = [](Type t) { return t.isBF16(); };
} // namespace

//===----------------------------------------------------------------------===//
// ActivationOp
//===----------------------------------------------------------------------===//
bool tensorrt::ActivationOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  auto inputElementType = getInput().getType().getElementType();
  auto activationType = getActivationType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I8, F16, F32);
  case 9:
  case 10:
    if (activationType == ActivationType::kRELU)
      return isType(inputElementType, I8, I32, I64, F16, BF16, F32);
    return isType(inputElementType, I8, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

/// Validates element type of unary op operator, given type of unary operation
/// and tensorrt version.
static bool isValidForTensorRTVersionUnaryOpImpl(Type inputElementType,
                                                 UnaryOperation operationType,
                                                 int64_t trtMajorVersion) {
  switch (operationType) {
  case UnaryOperation::kABS:
  case UnaryOperation::kNEG:
  case UnaryOperation::kSIGN:
    if (trtMajorVersion == 10 || trtMajorVersion == 9)
      return isType(inputElementType, I8, I32, I64, F16, BF16, F32);
    return isType(inputElementType, I8, I32, F16, F32);
  case UnaryOperation::kACOS:
  case UnaryOperation::kACOSH:
  case UnaryOperation::kASIN:
  case UnaryOperation::kASINH:
  case UnaryOperation::kATAN:
  case UnaryOperation::kATANH:
  case UnaryOperation::kCOSH:
  case UnaryOperation::kSINH:
  case UnaryOperation::kTAN:
    return isType(inputElementType, F16, F32);
  case UnaryOperation::kCEIL:
  case UnaryOperation::kCOS:
  case UnaryOperation::kERF:
  case UnaryOperation::kEXP:
  case UnaryOperation::kFLOOR:
  case UnaryOperation::kLOG:
  case UnaryOperation::kRECIP:
  case UnaryOperation::kROUND:
  case UnaryOperation::kSIN:
  case UnaryOperation::kSQRT:
    if (trtMajorVersion == 10 || trtMajorVersion == 9)
      return isType(inputElementType, F16, BF16, F32);
    return isType(inputElementType, F16, F32);
  case UnaryOperation::kNOT:
    return isType(inputElementType, I1);
  }
  llvm::report_fatal_error(
      "unhandled elementwise operation type for TensorRT 8.x/9.x/10.x");
}

bool tensorrt::UnaryOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  UnaryOperation operationType = getUnaryOperation();
  switch (trtMajorVersion) {
  case 8:
    return isValidForTensorRTVersionUnaryOpImpl(inputElementType, operationType,
                                                trtMajorVersion);
  case 9:
  case 10:
    return isValidForTensorRTVersionUnaryOpImpl(inputElementType, operationType,
                                                trtMajorVersion);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ElementwiseOp
//===----------------------------------------------------------------------===//

static bool
isValidForTensorRTVersionElementwiseOpImpl(Type elementType,
                                           ElementWiseOperation operationType,
                                           int64_t trtMajorVersion) {
  switch (operationType) {
  case ElementWiseOperation::kSUM:
  case ElementWiseOperation::kPROD:
  case ElementWiseOperation::kMAX:
  case ElementWiseOperation::kMIN:
  case ElementWiseOperation::kSUB:
  case ElementWiseOperation::kDIV:
  case ElementWiseOperation::kFLOOR_DIV:
    if (trtMajorVersion == 10 || trtMajorVersion == 9)
      return isType(elementType, I8, I32, I64, F16, BF16, F32);
    return isType(elementType, I8, F16, F32);
  case ElementWiseOperation::kEQUAL:
  case ElementWiseOperation::kLESS:
  case ElementWiseOperation::kGREATER:
    if (trtMajorVersion == 10 || trtMajorVersion == 9)
      return isType(elementType, I32, I64, F16, BF16, F32);
    return isType(elementType, F16, F32);
  case ElementWiseOperation::kPOW:
    if (trtMajorVersion == 10 || trtMajorVersion == 9)
      return isType(elementType, I8, F16, F32, BF16);
    return isType(elementType, I8, F16, F32);
  case ElementWiseOperation::kAND:
  case ElementWiseOperation::kOR:
  case ElementWiseOperation::kXOR:
    return isType(elementType, I1);
  }
  llvm::report_fatal_error(
      "unhandled elementwise operation type for TensorRT 8.x/9.x/10.x");
}

bool tensorrt::ElementWiseOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput1().getType().getElementType();
  ElementWiseOperation operationType = getElementwiseOperation();
  switch (trtMajorVersion) {
  case 8:
    return isValidForTensorRTVersionElementwiseOpImpl(
        inputElementType, operationType, trtMajorVersion);
  case 9:
  case 10:
    return isValidForTensorRTVersionElementwiseOpImpl(
        inputElementType, operationType, trtMajorVersion);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

bool tensorrt::ConstantOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getWeightsAttr().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I4, I8, I32, I64, F8, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//

bool tensorrt::ConcatenationOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType =
      cast<RankedTensorType>(getInputs().getType().front()).getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I8, I32, I64, F8, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

bool tensorrt::ConvolutionOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType =
      cast<RankedTensorType>(getInput().getType()).getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// EinsumOp
//===----------------------------------------------------------------------===//

bool tensorrt::EinsumOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType =
      cast<RankedTensorType>(getResult().getType()).getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionGatherOpImpl(int64_t trtMajorVersion,
                                                  Type dataElementType,
                                                  Type indicesElementType) {
  switch (trtMajorVersion) {
  case 8:
    return isType(dataElementType, I1, I8, I32, I64, F16, BF16, F32) &&
           isType(indicesElementType, I32, I64);
  case 9:
  case 10:
    return isType(dataElementType, I1, I8, I32, F16, F32) &&
           isType(indicesElementType, I32);
  default:
    return false;
  }
}

bool tensorrt::GatherOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type dataElementType = getData().getType().getElementType();
  Type indicesElementType = getIndices().getType().getElementType();
  return isValidForTensorRTVersionGatherOpImpl(trtMajorVersion, dataElementType,
                                               indicesElementType);
}

bool tensorrt::GatherNdOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type dataElementType = getData().getType().getElementType();
  Type indicesElementType = getIndices().getType().getElementType();
  return isValidForTensorRTVersionGatherOpImpl(trtMajorVersion, dataElementType,
                                               indicesElementType);
}

bool tensorrt::GatherElementsOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type dataElementType = getData().getType().getElementType();
  Type indicesElementType = getIndices().getType().getElementType();
  return isValidForTensorRTVersionGatherOpImpl(trtMajorVersion, dataElementType,
                                               indicesElementType);
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

bool tensorrt::IdentityOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type srcElementType = getInput().getType().getElementType();
  Type targetElementType = getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(srcElementType, I1, UI8, I8, I32, F16, F32) &&
           isType(targetElementType, I1, UI8, I8, I32, F16, F32);
  // TODO: Identity is deprecated. Return false once migrated to cast.
  case 9:
  case 10:
    return isType(srcElementType, I1, UI8, I8, I32, F16, BF16, F32) &&
           isType(targetElementType, I1, UI8, I8, I32, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// LinspaceOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionFillOpImpl(int64_t trtMajorVersion,
                                                Type resultElementType,
                                                FillOperation operationType) {
  switch (trtMajorVersion) {
  case 8:
    switch (operationType) {
    case FillOperation::kLINSPACE:
    case FillOperation::kRANDOM_UNIFORM:
      return isType(resultElementType, F16, F32);
    // TensorRT 8.6 doesn't have `kRANDOM_NORMAL` mode.
    default:
      return false;
    }
  case 9:
  case 10:
    switch (operationType) {
    case FillOperation::kLINSPACE:
      return isType(resultElementType, I32, I64, F32);
    case FillOperation::kRANDOM_UNIFORM:
    case FillOperation::kRANDOM_NORMAL:
      return isType(resultElementType, F16, F32);
    }
  default:
    return false;
  }
}

bool tensorrt::LinspaceOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type resultElementType = getResult().getType().getElementType();
  FillOperation operationType = FillOperation::kLINSPACE;
  return isValidForTensorRTVersionFillOpImpl(trtMajorVersion, resultElementType,
                                             operationType);
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

bool tensorrt::RandomUniformOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type resultElementType = getResult().getType().getElementType();
  FillOperation operationType = FillOperation::kRANDOM_UNIFORM;
  return isValidForTensorRTVersionFillOpImpl(trtMajorVersion, resultElementType,
                                             operationType);
}

//===----------------------------------------------------------------------===//
// RandomNormOp
//===----------------------------------------------------------------------===//

bool tensorrt::RandomNormalOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type resultElementType = getResult().getType().getElementType();
  FillOperation operationType = FillOperation::kRANDOM_NORMAL;
  return isValidForTensorRTVersionFillOpImpl(trtMajorVersion, resultElementType,
                                             operationType);
}

//===----------------------------------------------------------------------===//
// NormalizationOp
//===----------------------------------------------------------------------===//

bool tensorrt::NormalizationOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

bool tensorrt::PoolingOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
  case 9:
  case 10:
    return isType(inputElementType, I8, F16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

bool tensorrt::ReduceOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I8, I32, I64, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

bool tensorrt::SelectOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getElseInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I32, I64, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

bool tensorrt::SliceOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  SliceMode sliceMode = getMode();
  // TensorRT 8.x has `kDEFAULT` slice mode which is
  // renamed to `kSTRICT_BOUNDS` in TensorRT 9.x onwards.
  // `kSTRICT_BOUNDS` emits error if out of bound which did
  // nit happen in `kDEFAULT`. For validation case, `kDEFAULT`
  // is treated as `kSTRICT_BOUNDS`.
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    switch (sliceMode) {
    case SliceMode::kCLAMP:
    case SliceMode::kFILL:
    case SliceMode::kREFLECT:
      return isType(inputElementType, I1, I8, I32, F16, F32);
    default:
      return isType(inputElementType, I1, I8, I32, I64, F16, BF16, F32);
    }
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

bool tensorrt::SoftMaxOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// OneHotOp
//===----------------------------------------------------------------------===//

bool tensorrt::OneHotOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type valuesElementType = getValues().getType().getElementType();
  switch (trtMajorVersion) {
  // TensorRT 8.x doesn't have OneHot op
  case 8:
    return false;
  case 9:
  case 10:
    return isType(valuesElementType, I8, I32, I64, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// RaggedSoftmaxOp
//===----------------------------------------------------------------------===//

bool tensorrt::RaggedSoftMaxOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// MatrixMultiplyOp
//===----------------------------------------------------------------------===//

bool tensorrt::MatrixMultiplyOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput0().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionTopKOpImpl(int64_t trtMajorVersion,
                                                Type inputElementType) {
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I32, I64, F16, BF16, F32);
  default:
    return false;
  }
}

bool tensorrt::TopKOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionTopKOpImpl(trtMajorVersion, inputElementType);
}

//===----------------------------------------------------------------------===//
// PaddingOp
//===----------------------------------------------------------------------===//

bool tensorrt::PaddingOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  // TODO: Padding is deprecated from TensorRT 8.2
  // Replace uses with slice.
  switch (trtMajorVersion) {
  case 8:
  case 9:
  case 10:
    return isType(inputElementType, I8, I32, F16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// NonZeroOp
//===----------------------------------------------------------------------===//

bool tensorrt::NonZeroOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I8, I32, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

bool tensorrt::IfOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type resultElementType =
      cast<RankedTensorType>(getResultTypes().front()).getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(resultElementType, I1, I32, F16, F32);
  case 9:
  case 10:
    return isType(resultElementType, I1, I32, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

bool tensorrt::ShapeOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I8, I32, F8, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ParametricReLUOp
//===----------------------------------------------------------------------===//

bool tensorrt::ParametricReLUOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I8, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I8, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionShuffleOpImpl(int64_t trtMajorVersion,
                                                   Type inputElementType) {
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I1, I8, I32, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I1, I8, I32, I64, F8, F16, BF16, F32);
  default:
    return false;
  }
}

bool tensorrt::ShuffleOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// DeconvolutionOp
//===----------------------------------------------------------------------===//

bool tensorrt::DeconvolutionOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
  case 9:
  case 10:
    return isType(inputElementType, I8, F16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

bool tensorrt::QuantizeOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  Type resultElementType = getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, F16, F32) && isType(resultElementType, I8);
  case 9:
  case 10:
    return isType(inputElementType, F16, BF16, F32) &&
           isType(resultElementType, I8, F8, I4);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

bool tensorrt::DequantizeOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  Type resultElementType = getType().getElementType();
  switch (trtMajorVersion) {
  case 8:
    return isType(inputElementType, I8) && isType(resultElementType, F16, F32);
  case 9:
  case 10:
    return isType(inputElementType, I4, I8, F8) &&
           isType(resultElementType, F16, BF16, F32);
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

bool tensorrt::TransposeOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// ExpandRankOp
//===----------------------------------------------------------------------===//

bool tensorrt::ExpandRankOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// CollapseRankOp
//===----------------------------------------------------------------------===//

bool tensorrt::CollapseRankOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

bool tensorrt::BroadcastOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// ArgMinOp / ArgMaxOp
//===----------------------------------------------------------------------===//

bool tensorrt::ArgMinOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionTopKOpImpl(trtMajorVersion, inputElementType);
}

bool tensorrt::ArgMaxOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionTopKOpImpl(trtMajorVersion, inputElementType);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool tensorrt::ReshapeOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionShuffleOpImpl(trtMajorVersion,
                                                inputElementType);
}

//===----------------------------------------------------------------------===//
// ResizeOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionResizeOpImpl(int64_t trtMajorVersion,
                                                  Type inputElementType,
                                                  ResizeMode resizeMode) {
  switch (trtMajorVersion) {
  case 8:
  case 9:
  case 10:
    return isType(inputElementType, I8, F16, F32);
  default:
    return false;
  }
}

bool tensorrt::ResizeNearestOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionResizeOpImpl(
      trtMajorVersion, inputElementType, ResizeMode::kNEAREST);
}

bool tensorrt::ResizeLinearOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionResizeOpImpl(
      trtMajorVersion, inputElementType, ResizeMode::kLINEAR);
}

bool tensorrt::ResizeCubicOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type inputElementType = getInput().getType().getElementType();
  return isValidForTensorRTVersionResizeOpImpl(
      trtMajorVersion, inputElementType, ResizeMode::kCUBIC);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

static bool isValidForTensorRTVersionScatterOpImpl(int64_t trtMajorVersion,
                                                   Type dataElementType,
                                                   Type indicesElementType) {
  switch (trtMajorVersion) {
  case 8:
    return isType(dataElementType, I8, I32, F16, F32) &&
           isType(indicesElementType, I32);
  case 9:
  case 10:
    return isType(dataElementType, I8, I32, I64, F16, BF16, F32) &&
           isType(indicesElementType, I32, I64);
  default:
    return false;
  }
}

bool tensorrt::ScatterOp::isValidForTensorRTVersion(int64_t trtMajorVersion) {
  Type dataElementType = getData().getType().getElementType();
  Type indicesElementType = getIndices().getType().getElementType();
  return isValidForTensorRTVersionScatterOpImpl(
      trtMajorVersion, dataElementType, indicesElementType);
}

//===----------------------------------------------------------------------===//
// ScatterElementsOp
//===----------------------------------------------------------------------===//

bool tensorrt::ScatterElementsOp::isValidForTensorRTVersion(
    int64_t trtMajorVersion) {
  Type dataElementType = getData().getType().getElementType();
  Type indicesElementType = getIndices().getType().getElementType();
  return isValidForTensorRTVersionScatterOpImpl(
      trtMajorVersion, dataElementType, indicesElementType);
}