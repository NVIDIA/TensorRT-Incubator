//===- MlirTensorRtTblgen.cpp -----------------------------------*- C++ -*-===//
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
/// This file contains the main function for mlir-tensorrt-tblgen.
///
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

using ::llvm::Record;

using ::mlir::tblgen::Argument;
using ::mlir::tblgen::FmtContext;
using ::mlir::tblgen::Operator;
using ::mlir::tblgen::strfmt;

/// Emit a set of C++ functions that convert from TensorRT dialect enums
/// to the equivalent enums under the nvinfer1 namespace. This is just a 1-1
/// conversion using the names of the enums. The TensorRT enum names were
/// originally generated from the NvInfer.h header file. The generated enum
/// takes the following form. For the enum `mlir::tensorrt::SliceMode`, the
/// following function is generated:
///
/// ```C++
/// std::optional<nvinfer1::SliceMode>
/// convertSliceModeToNvInferEnum(::mlir::tensorrt::SliceMode value) {
///   switch (value) {
///   case SliceMode::kDEFAULT:
///     return nvinfer1::SliceMode::kDEFAULT;
///   case SliceMode::kWRAP:
///     return nvinfer1::SliceMode::kWRAP;
///   case SliceMode::kCLAMP:
///     return nvinfer1::SliceMode::kCLAMP;
///   case SliceMode::kFILL:
///     return nvinfer1::SliceMode::kFILL;
///   case SliceMode::kREFLECT:
///     return nvinfer1::SliceMode::kREFLECT;
///   }
/// }
/// ```
static bool emitEnumConverters(const llvm::RecordKeeper &recordKeeper,
                               raw_ostream &outputStream) {
  raw_indented_ostream os(outputStream);
  ArrayRef<const Record *> defs =
      recordKeeper.getAllDerivedDefinitions("TensorRT_I32EnumAttr");

  std::string enumDeclsStr;
  llvm::raw_string_ostream enumDeclsStream(enumDeclsStr);

  {
    tblgen::IfDefScope ifdefScope("GEN_TRT_ENUM_CONVERTER_DEFS", os);
    for (const Record *def : defs) {
      if (def->getValueAsBit("skipNvInferEnumConverterGeneration"))
        continue;

      StringRef dialectEnumCppType = def->getValueAsString("cppType");
      StringRef dialectEnumClassName = def->getValueAsString("className");

      std::string nvinferClassName =
          strfmt("nvinfer1::{0}", dialectEnumClassName);

      std::vector<StringRef> enumerants;
      for (const Record *enumerant : def->getValueAsListOfDefs("enumerants"))
        enumerants.push_back(enumerant->getValueAsString("symbol"));

      std::string funcSignature =
          strfmt("std::optional<{0}> convert{1}ToNvInferEnum({2} value)",
                 nvinferClassName, dialectEnumClassName, dialectEnumCppType);
      outputStream << funcSignature << " {\n";
      os.indent();
      os << "switch(value) {\n";
      os.indent();
      for (StringRef enumerant : enumerants) {
        os << "case " << dialectEnumClassName << "::" << enumerant << ":\n";
        os.indent();
        os << strfmt("return {0}::{1};\n", nvinferClassName, enumerant);
        os.unindent();
      }
      os.unindent();
      os << "}\n";
      os << "llvm_unreachable(\"unknown TRT enum conversion from "
         << dialectEnumCppType << "\");\n";
      os.unindent();
      os << "}\n\n";

      // Append to the enum declarations string.
      enumDeclsStream << funcSignature << ";\n";
    }
  }

  {
    tblgen::IfDefScope ifdefScope("GEN_TRT_ENUM_CONVERTER_DECLS", os);
    enumDeclsStream.flush();
    os << enumDeclsStr << "\n";
  }

  return false;
}

/// Emit replacement for `$<attribute-name>` when the attribute is a F32/F64
/// (and possibly optional) attribute. See below for more explanation of
/// the replacement rules and context during mlir-to-trt-api translation.
static void
emitScalarFloatAttrReplacement(raw_indented_ostream &os,
                               const tblgen::Operator &op,
                               const tblgen::NamedAttribute &namedAttr) {
  const tblgen::Attribute &attr = namedAttr.attr;
  std::string getter = strfmt("op.{0}()", op.getGetterName(namedAttr.name));
  std::string scalarType =
      attr.getAttrDefName() == "F32Attr" ? "float" : "double";
  std::string suffix = scalarType;
  suffix[0] = llvm::toUpper(suffix[0]);
  if (attr.isOptional())
    scalarType = strfmt("std::optional<{0}>", scalarType);
  os << scalarType << " " << namedAttr.name << " = ";
  if (attr.isOptional()) {
    os << getter << " ? " << scalarType << "(" << getter << "->convertTo"
       << suffix << "()) : std::nullopt;\n";
    return;
  }
  os << getter << ".convertTo" << suffix << "();\n";
}

/// Emit replacement for `$<attribute-name>`. See below for more explanation of
/// the replacement rules and context.
static void emitAttributeReplacement(FmtContext &ctx, raw_indented_ostream &os,
                                     const tblgen::Operator &op,
                                     const tblgen::NamedAttribute &namedAttr) {
  StringRef name = namedAttr.name;
  std::string getter = strfmt("op.{0}()", op.getGetterName(namedAttr.name));
  ctx.addSubst(namedAttr.name, namedAttr.name);

  // Emit remaps for attributes to values types that can go to TRT API.
  const tblgen::Attribute &attr = namedAttr.attr;
  const StringRef attrRecordName = attr.getAttrDefName();
  const llvm::Record &attrDef =
      !attr.hasDefaultValue() ? attr.getDef() : attr.getBaseAttr().getDef();
  if (attrDef.isSubClassOf("TensorRT_EnumAttr")) {
    StringRef enumClassName =
        attrDef.getValueAsDef("enum")->getValueAsString("className");
    std::string remapEnumFuncName =
        strfmt("::mlir::tensorrt::convert{0}ToNvInferEnum", enumClassName);
    os << "auto " << namedAttr.name << " = " << remapEnumFuncName << "("
       << getter << ");\n";
    os << strfmt(
        "if(!{0}) return emitError(op->getLoc()) << \"failed to "
        "convert TensorRT dialect attribute {1} to 'nvinfer' enum\";\n",
        namedAttr.name, enumClassName);
    return;
  }
  if (attrRecordName == "ElementsAttr") {
    os << "FailureOr<nvinfer1::Weights> " << namedAttr.name
       << " = encoder.getNvInferWeights(" << getter << ");\n";
    os << "if (failed(" << namedAttr.name << ")) return failure();\n";
    return;
  }
  if (attrRecordName == "DenseI64ArrayAttr" ||
      attrRecordName == "DenseI32ArrayAttr") {
    if (!attr.isOptional()) {
      os << "nvinfer1::Dims " << namedAttr.name
         << " = ::mlir::tensorrt::getNvInferDims(" << getter << ");\n";
      return;
    }

    os << "std::optional  <nvinfer1::Dims> " << name << " = "
       << "::mlir::tensorrt::getOptionalNvInferDims(" << getter << ");\n";

    return;
  }
  if (attrRecordName == "F32Attr" || attrRecordName == "F64Attr") {
    emitScalarFloatAttrReplacement(os, op, namedAttr);
    return;
  }
  if (attrRecordName == "TensorRT_DimensionListAttr") {
    os << "// Convert the dimension list to a bitmask.\n";
    os << "uint32_t " << name << " = getBitMaskFromDimensionList(" << getter
       << ");\n";
    return;
  }

  os << "// generically handled attr : type = " << attr.getAttrDefName()
     << ", retType = " << attr.getReturnType() << "\n";
  os << "auto " << name << " = " << getter << ";\n";
}

/// Emit replacements for `$<operand-name>`. See below for more explanation of
/// context and replacement rules.
static void emitOperandReplacement(FmtContext &ctx, raw_indented_ostream &os,
                                   const tblgen::Operator &op,
                                   const tblgen::NamedTypeConstraint &operand) {
  StringRef name = operand.name;
  if (operand.constraint.getCppType() == "::mlir::TensorType" ||
      operand.constraint.getCppType() == "::mlir::RankedTensorType") {
    std::string getter = strfmt("op.{0}()", op.getGetterName(operand.name));
    ctx.addSubst(operand.name, name);
    if (!operand.isVariadic() && !operand.isOptional()) {
      os << "assert(encoder.contains(op." << op.getGetterName(operand.name)
         << "()) > 0 );\n";
      os << "nvinfer1::ITensor* " << name << " = ";
      os << "encoder.lookup(op." << op.getGetterName(operand.name) << "());\n";
      return;
    }
    if (!operand.isVariadic() && operand.isOptional()) {
      os << "nvinfer1::ITensor* " << name << " = ";
      os << "op." << op.getGetterName(operand.name) << "() != nullptr ? ";
      os << "encoder.lookup(" << getter << ") : nullptr;\n";
      return;
    }
    if (!operand.isOptional() && operand.isVariadic()) {
      os << "SmallVector<nvinfer1::ITensor*> " << name << ";\n";
      os << "for(unsigned i = 0; i < " << getter << ".size(); i++) {\n";
      os.indent();
      os << strfmt("{0}.push_back(encoder.lookup({1}[i]));\n", name, getter);
      os.unindent();
      os << "}\n";
      return;
    }
  }

  // Otherwise, remap to the default getter.
  ctx.addSubst(operand.name,
               strfmt("op.{0}()", op.getGetterName(operand.name)));
}

/// Generates a C++ function that encodes TensorRT dialect operations
/// into TensorRT layers via the `nvinfer1::INetworkDefinition` API.
/// This uses the `trtLayerAdd` string that is attached to each TensorRT
/// operation ODS specification (except for extension operations, which
/// obviously don't have an equivalent TensorRT API call).
///
/// ### Generated Signature
///
/// The generated function has a signature as follows.
///
/// ```
/// LogicalResult encodeTensorRTOp(TensorRTOpInterface tensorrtOp,
///                            NvInferNetworkEncoder &encoder,
///                            SmallVector<nvinfer1::ITensor *> &results)
/// ```
///
/// The function accepts an NvInferNetworkEncoder handle, which contains
/// a handle for the `nvinfer1::INetworkDefinition*` as well as methods
/// to map `Value`s to `nvinfer::ITensor*`'s during the conversion process.
/// After the `tensorrtOp` is encoded as an ILayer, the ITensor* results
/// corresponding to the `tensorrtOp` Value results should be added to the
/// `results` vector. The mappings will automatically be inserted by the caller.
///
/// ### ODS `trtLayerAdd` Convention and Substitutions
///
/// Each ODS spec will contain a string under the key `trtLayerAdd`.
/// For example, ActivationOp has the following:
///
/// ```tablegen
/// let trtLayerAdd = [{
///   nvinfer1::IActivationLayer *layer = $net->addActivation(
///                                                   *$input,
///                                                   $activationType);
///   if($alpha)
///     layer->setAlpha($alpha->convertToFloat());
///   if($beta)
///     layer->setBeta($beta->convertToFloat());
///   $results.push_back(layer->getOutput(0));
/// }];
/// ```
///
/// Names prefixed by `$` will be automatically substituted by the code
/// generator. The following substitutions are available:
///
/// - "$net": will be replaced by the appropriate
///          `nvinfer::INetworkDefinition*` variable name.
///
///  - "$e": will be replaced by the variable name for the
///          `NvInferNetworkEncoder` function argument (see signature above).
///
///  - "$op": will be replaced  by the variable name for the operation
///           with the same type as the ODS spec (in the example, $op would
///           be replaced with a variable name that has type ActivationOp.
///
/// - "$<attribute-name>":
///        - If the attribute is derived from a TensorRT_EnumAttr,
///          then this will automatically be substituted with the nvinfer1
///          equivalent enum value.
///        - If the attribute is an DenseI(64|32)ArrayAttr, then this will be
///          converted to an nvinfer1::Dims object.
///        - ElementsAttr and subclasses and optional versions of ElementsAttr
///          will be converted to an nvinfer1::Weights object.
///        - Otherwise, the `op.getAttrName()` value is substituted.
///
/// - "$results": replaced by the variable name for the results vector.
///
/// - "$<operand-name>": If the operand is a RankedTensor, this is automatically
///        substituted with the correct `nvinfer::ITensor*` value corresponding
///        to this value during the translation process. If the value is
///        variadic, the variable has the type `SmallVector<nvinfer::ITensor*>`.
///        If the Value is optional and not present, then it will have `nullptr`
///        value and `nvinfer1::ITensor*` type.
static bool emitLayerAddDefinitions(const llvm::RecordKeeper &recordKeeper,
                                    raw_ostream &outputStream) {
  raw_indented_ostream os(outputStream);
  ArrayRef<const Record *> defs = recordKeeper.getAllDerivedDefinitions("Op");

  /// The signature for the encoding.
  StringRef encodeOpSignature =
      "LogicalResult encodeOp(::mlir::Operation *tensorrtOp,"
      "::mlir::tensorrt::NvInferNetworkEncoder &encoder, "
      "::mlir::SmallVector<nvinfer1::ITensor*> &results)";
  std::string attachInterfaceStr;
  llvm::raw_string_ostream attachInterfaceStream(attachInterfaceStr);

  {
    tblgen::IfDefScope ifdefScope("GEN_TRT_ENCODE_OP_IMPL", os);
    for (const Record *def : defs) {
      Operator op(def);

      std::string implClassName = op.getCppClassName().str() + "EncodingImpl";

      os << "struct " << implClassName
         << " : public "
            "::mlir::tensorrt::TensorRTEncodingOpInterface::ExternalModel<"
         << implClassName << ", " << op.getCppClassName() << "> {\n";

      os.indent();

      os << encodeOpSignature << " const {\n";
      os.indent();

      StringRef expr = def->getValueAsString("trtLayerAdd");
      StringRef cppClass = op.getCppClassName();

      os << strfmt("auto op = dyn_cast<{0}>(tensorrtOp);\n", cppClass);

      // if (expr.empty()) {
      //   os << "return op->emitOpError(\"unhandled tensorrt op during
      //   translation "
      //         "to tensorrt engine\");\n";
      //   os.unindent();
      //   os << "}\n\n";
      //   continue;
      // }

      os << "nvinfer1::INetworkDefinition *network = "
            "encoder.getNetworkDefinition();\n";

      // Add substitutions for variables that apply to all ops.
      FmtContext ctx;
      ctx.addSubst("e", "encoder");
      ctx.addSubst("op", "op");
      ctx.addSubst("net", "network");
      ctx.addSubst("results", "results");
      ctx.addSubst("resultShape",
                   "::mlir::tensorrt::getNvInferDims(op.getType().getShape())");
      for (const Argument &arg : op.getArgs()) {
        if (auto *namedAttr = arg.dyn_cast<tblgen::NamedAttribute *>()) {
          emitAttributeReplacement(ctx, os, op, *namedAttr);
          continue;
        }
        if (auto *operand = arg.dyn_cast<tblgen::NamedTypeConstraint *>()) {
          emitOperandReplacement(ctx, os, op, *operand);
          continue;
        }
      }

      // Add substitution for regions. These are used in control flow ops like
      // `IfOp`.
      for (auto region : op.getRegions()) {
        ctx.addSubst(region.name,
                     strfmt("op.{0}()", op.getGetterName(region.name)));
      }

      // Emit the body with substitutions.
      os << "auto const numLayersBefore = network->getNbLayers();\n";
      os << tblgen::tgfmt(expr, &ctx);
      os << "auto const numLayersAfter = network->getNbLayers();\n";
      os << "for (int64_t i = numLayersBefore; i < numLayersAfter; ++i) "
            "encoder.map(tensorrtOp, network->getLayer(i));\n";
      os << "return success();\n";
      os.unindent();
      os << "}\n";
      os.unindent();
      os << "};\n\n";

      // Append to the attach interface stream.
      attachInterfaceStream << op.getCppClassName() << "::attachInterface<"
                            << implClassName << ">(*ctx);\n";
    }
  }

  {

    tblgen::IfDefScope ifdefScope("GEN_TRT_ENCODE_OP_IMPL_ATTACH_INTERFACE",
                                  os);
    attachInterfaceStream.flush();
    os << attachInterfaceStr;
  }

  return false;
}

static void printMultilineDocString(llvm::raw_ostream &os,
                                    llvm::StringRef str) {
  for (llvm::StringRef line : llvm::split(str, "\n")) {
    line = line.drop_while(llvm::isSpace);
    if (!line.empty())
      os << "/// " << line << "\n";
  }
}

/// Emit a header file that declares/defines the enums specified in the file
/// along with associated printer and parser utilities. Declarations will be
/// guarded by `GEN_ENUM_DECLS` and definitions guarded by `GEN_ENUM_DEFS`.
static bool emitEnumDefs(const llvm::RecordKeeper &recordKeeper,
                         raw_ostream &outputStream, bool useC) {
  raw_indented_ostream os(outputStream);
  ArrayRef<const Record *> defs =
      recordKeeper.getAllDerivedDefinitions("EnumSpec");

  {
    tblgen::IfDefScope ifdefScope("GEN_ENUM_DECLS", os);

    // Generate all the concrete class declarations.
    for (const Record *def : defs) {
      StringRef className = def->getValueAsString("symbol");

      //===----------------------------------------------------------------------===//
      // Emit the declaration:
      // `///` documentationString
      // `enum` SymbolName `{` (CaseName `=` CaseValue)... `}`
      //===----------------------------------------------------------------------===//

      // Emit the constructor. All the fields get populated in the constructor.
      printMultilineDocString(os, def->getValueAsString("documentationString"));

      if (useC)
        os << "typedef enum MTRT_" << className << "{\n";
      else
        os << "enum class " << className << "{\n";

      os.indent();

      llvm::interleave(
          def->getValueAsListOfDefs("cases"), os,
          [&](const Record *caseDef) {
            if (useC)
              os << "MTRT_" << className << "_"
                 << caseDef->getValueAsString("symbol");
            else
              os << caseDef->getValueAsString("symbol");
            os << " = " << caseDef->getValueAsInt("value");
          },
          ",\n");
      os << "\n";
      os.unindent();
      os << "}";

      if (useC)
        os << " MTRT_" << className << ";\n\n";
      else
        os << ";\n\n";

      if (useC)
        continue;

      os << "/// Returns a " << className
         << " value corresponding to `str` if possible, otherwise returns "
            "nullopt.\n";
      os << "std::optional<" << className << "> parse" << className
         << "(std::string_view str);\n";

      os << "/// Returns a string representation of " << className << ".\n";
      os << "std::string_view stringify" << className << "(" << className
         << " val);\n\n";
    }
  }

  if (useC)
    return false;

  {
    tblgen::IfDefScope ifdefScope("GEN_ENUM_DEFS", os);
    // Generate all the concrete class declarations.
    for (const Record *def : defs) {
      StringRef className = def->getValueAsString("symbol");

      //===----------------------------------------------------------------------===//
      // Emit the string -> enum ("symbolizer") definitions.
      //===----------------------------------------------------------------------===//
      os << "std::optional<" << className << "> parse" << className
         << "(std::string_view str) {\n";
      os.indent();

      for (const Record *enumCase : def->getValueAsListOfDefs("cases")) {
        if (enumCase->getValueAsString("symbol").contains("_SIZE"))
          continue;
        llvm::StringRef name = enumCase->getValueAsString("symbol");
        os << "if(str == \"" << name << "\") {\n";
        os.indent();
        os << "return " << className << "::" << name << ";\n";
        os.unindent();
        os << "}\n";
      }

      // All other cases: return nullopt for failure to parse.
      os << "return std::nullopt;\n";
      os.unindent();
      os << "}\n";

      //===----------------------------------------------------------------------===//
      // Emit enum -> string ("stringify") definitions
      //===----------------------------------------------------------------------===//
      os << "std::string_view stringify" << className << "(" << className
         << " val) {\n";
      os.indent();

      os << "switch(val) {\n";
      for (const Record *enumCase : def->getValueAsListOfDefs("cases")) {
        if (enumCase->getValueAsString("symbol").contains("_SIZE"))
          continue;
        os << "case " << className
           << "::" << enumCase->getValueAsString("symbol") << ":\n";
        os.indent();
        os << "return \"" << enumCase->getValueAsString("symbol") << "\";\n";
        os.unindent();
      }

      os << "default:\n";
      os.indent();
      os << "llvm_unreachable(\"invalid enum value\");\n";
      os.unindent();
      os << "}\n";

      os.unindent();
      os << "}\n";
    }
  }

  return false;
}

int main(int argc, char **argv) {
  // Generator that prints records.
  mlir::GenRegistration printRecords(
      "print-records", "Print all records to stdout",
      [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
        os << records;
        return false;
      });

  static GenRegistration genApiLayerAddDefs(
      "gen-tensorrt-layer-add-defs",
      "Generates TensorRT Dialect definitions for TRT API layer-add "
      "functions "
      "(.cpp)",
      [](const llvm::RecordKeeper &records, raw_ostream &os) {
        return emitLayerAddDefinitions(records, os);
      });

  static GenRegistration genEnumConverterDefs(
      "gen-tensorrt-enum-converter-defs",
      "Generates converter functions that convert from TensorRT dialect "
      "enums "
      "to equivalent enums in nvinfer1 namespace enums (.cpp)",
      [](const llvm::RecordKeeper &records, raw_ostream &os) {
        return emitEnumConverters(records, os);
      });

  static GenRegistration genEnumDefs(
      "gen-custom-enum-defs",
      "Generates enum definitions and utilities (printers, parsers, etc)",
      [&](const llvm::RecordKeeper &records, raw_ostream &os) {
        return emitEnumDefs(records, os, /*useC=*/false);
      });

  static GenRegistration genEnumCDefs(
      "gen-custom-enum-c-defs",
      "Generates enum definitions and utilities (printers, parsers, etc)",
      [&](const llvm::RecordKeeper &records, raw_ostream &os) {
        return emitEnumDefs(records, os, /*useC=*/true);
      });

  return mlir::MlirTblgenMain(argc, argv);
}
