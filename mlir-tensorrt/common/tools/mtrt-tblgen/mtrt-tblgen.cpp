//===- executor-tblgen.cpp ------------------------------------------------===//
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
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/CodeGenHelpers.h"
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
      os.unindent();
      os << "}\n";

      os << "llvm_unreachable(\"invalid enum value\");\n";

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