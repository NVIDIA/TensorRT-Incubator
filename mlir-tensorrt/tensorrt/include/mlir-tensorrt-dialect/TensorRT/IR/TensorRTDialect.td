#ifndef TENSORRT_MLIR_TENSORRT_DIALECT
#define TENSORRT_MLIR_TENSORRT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTInterfaces.td"

//===----------------------------------------------------------------------===//
// TensorRT Dialect
//===----------------------------------------------------------------------===//

def TensorRT_Dialect : Dialect {
  let name = "tensorrt";
  let summary = "A dialect that models the TensorRT API";
  let description = [{
      This dialect is automatically generated from the TensorRT API
      header `NvInfer.h`.
  }];
  let cppNamespace = "::mlir::tensorrt";

  // Enable when the dialect gains custom types.
  // let useDefaultTypePrinterParser = 1;

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    /// Return the name of the function arg attr that encodes
    /// host tensor value bounds. It should have a type `ShapeProfileAttr`.
    static StringRef getShapeTensorValueBoundsArgAttrName() {
      return "tensorrt.value_bounds";
    }

    /// Return the name of the function arg attr that encodes
    /// the shape bounds. It should have a type `ShapeProfileAttr`.
    static StringRef getShapeProfileArgAttrName() {
      return "tensorrt.shape_profile";
    }

    /// TensorRT quantization and dequantization mode markers.
    static constexpr StringRef kTensorRTPerTensorQuantizationMarker = "tensorrt.pt_q";
    static constexpr StringRef kTensorRTPerChannelQuantizationMarker = "tensorrt.pc_q";
    static constexpr StringRef kTensorRTBlockQuantizationMarker = "tensorrt.block_q";
    static constexpr StringRef kTensorRTPerTensorDequantizationMarker = "tensorrt.pt_dq";
    static constexpr StringRef kTensorRTPerChannelDequantizationMarker = "tensorrt.pc_dq";
    static constexpr StringRef kTensorRTBlockDequantizationMarker = "tensorrt.block_dq";
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    // Arith dialect is required since OpaquePluginOp can use arith ops in its
    // shape calculation region.
    "::mlir::arith::ArithDialect",
    "::mlir::quant::QuantizationDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TensorRT Dialect Op Traits
//===----------------------------------------------------------------------===//

class TensorRT_OpTrait<string cppClassName, string extraOpDefinition> :
    NativeOpTrait<cppClassName, /*traits=*/[],
                  /*extraOpDeclaration=*/[{}],
                  /*extraOpDefinition=*/extraOpDefinition> {
  let cppNamespace = "::mlir::tensorrt";
}

def TensorRTInferPartialTensorTypeTrait :
      TensorRT_OpTrait<"TensorRTInferPartialTensorTypeTrait", "">;

def TensorRTInferCompleteTensorTypeTrait :
  TensorRT_OpTrait<"TensorRTInferCompleteTensorTypeTrait", [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                    std::optional<Location> location,
                    ValueRange operands, DictionaryAttr attributes,
                    OpaqueProperties properties, RegionRange regions,
                    SmallVectorImpl<Type> &inferredReturnTypes) {
      SmallVector<ShapedTypeComponents, 2> retComponents;
      if (failed($cppClass::inferReturnTypeComponents(context, location,
                              operands, attributes, properties, regions,
                              retComponents)))
        return failure();
      return ::mlir::detail::inferReturnTensorTypes(retComponents,
                                              inferredReturnTypes);
    }
  }]
>;

// A trait that encapsulates both InferShapedTypeOpInterface an
// InferTypeOpInterface. This is used for when the result type(s) of an operation
// can be completely inferred (both shape and element type). The op must implement
// "inferReturnTypeComponents".
def TensorRTInferTensorResultTypes :
  TraitList<[
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
      ["inferReturnTypeComponents"]>,
    TensorRTInferCompleteTensorTypeTrait
  ]>;

// A trait used when only the tensor type (shape, element type) can only be
// partially determined from the op's attributes and input types. This trait
// will ensure that the result types are automatically verified, when possible.
// The op still must implement "inferReturnTypeComponents".
def TensorRTPartiallyInferTensorResultTypes
  : TraitList<[
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
      ["inferReturnTypeComponents"]>,
    TensorRTInferPartialTensorTypeTrait
  ]>;

//===----------------------------------------------------------------------===//
// TensorRT Dialect Type Restrictions
//===----------------------------------------------------------------------===//
def TensorRT_ShapeTensor: TensorOf<[I32],
      [HasStaticShapePred,
       HasAnyRankOfPred<[1]>],
      "1D static i32 tensor representing a shape">;

def TensorRT_QuantizedI8 : Type<
      And<[CPred<"llvm::isa<mlir::quant::UniformQuantizedType>($_self)">,
           CPred<"llvm::cast<mlir::quant::UniformQuantizedType>($_self).getStorageType().isSignlessInteger(8)">,
           Or<[
            CPred<"llvm::cast<mlir::quant::UniformQuantizedType>($_self).getExpressedType().isF32()">,
            CPred<"llvm::cast<mlir::quant::UniformQuantizedType>($_self).getExpressedType().isF16()">
           ]>,
           CPred<"llvm::cast<mlir::quant::UniformQuantizedType>($_self).getZeroPoint() == 0">
      ]>,
      "i8 approximation of f16 or f32"> {
}

def TensorRT_I8 : AnyTypeOf<[I8, TensorRT_QuantizedI8],
                            "allowed TensorRT tensor i8 element types">;

def TensorRT_F8 : AnyTypeOf<[F8E4M3FN]>;

def TensorRT_I4 : I<4>;

// We only support up to 8 ranks for TensorRT Tensors
class TensorRT_RankedTensorOf<list<Type> allowedTypes>
  : TensorRankOf<allowedTypes, [0, 1, 2, 3, 4, 5, 6, 7, 8]>;

def TensorRT_Tensor : TensorRT_RankedTensorOf<
      [I1, TensorRT_I8, I32, I64, F16, BF16, F32]>;

def TensorRT_DimensionListAttr : ConfinedAttr<DenseI64ArrayAttr,[
      DenseArrayStrictlySorted<DenseI64ArrayAttr>,
      DenseArrayNonNegative<DenseI64ArrayAttr>
    ]>;

#endif // TENSORRT_MLIR_TENSORRT_DIALECT
