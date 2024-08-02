//===- EinsumHelper.cpp ---------------------------------------------------===//
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
#include "EinsumHelper.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include <regex>

using namespace mlir;
using namespace mlir::tensorrt;
using namespace mlir::tensorrt::einsum;

/// Validates einsum equation string. It uses regular expressions to match
/// incoming equation to exactly what TensorRT expects.
static LogicalResult validateEquation(StringRef equation,
                                      std::optional<Location> loc,
                                      ErrorFn emitErrorFn) {
  if (equation.empty())
    return emitErrorFn(loc, "einsum equation string is empty");
  // TensorRT supports only lower case letters with NO ellipses
  // Valid input regex pattern can be broken down as follows,
  // einsumLabels = "[ a-z]*"
  // einsumSubscript = einsumLabels + "(?:" + einsumLabels + ")?"
  // einsumInputs = "(" + einsumSubscript + "(?:," + einsumSubscript + ")*)"
  // einsumArrow = "- *>"
  // einsumOutput = "(" + einsumSubscript + ")"
  // einsumValidInputRegex = einsumInputs + "(?:" + einsumArrow.str() +
  // einsumOutput + ")?"

  StringRef einsumValidInputRegex = "([ a-z]*(?:[ a-z]*)?(?:,[ a-z]*(?:[ "
                                    "a-z]*)?)*)(?:- *>([ a-z]*(?:[ a-z]*)?))?";
  std::regex pattern(einsumValidInputRegex.str());
  std::cmatch match;
  if (!std::regex_match(equation.begin(), equation.end(), match, pattern))
    return emitErrorFn(
        loc, "einsum equation syntax is invalid. TensorRT "
             "only supports ASCII lower-case letters with no ellipses.");
  return success();
}

/// Computes output subscript, given a list of input subscripts. Simple
/// einsum rule which states `labels repeated across input are reduced in
/// the output` is followed.
static Subscript computeOutputSubscript(ArrayRef<Subscript> inputSubscripts) {
  // When label is repeated between inputs, that dimension is reduced
  std::map<char, int64_t> uniqueLabels;
  for (auto &inputSubscript : inputSubscripts)
    for (auto &label : inputSubscript)
      if (uniqueLabels.count(label) == 0) {
        uniqueLabels.insert(std::pair<char, int64_t>(label, 1));
      } else {
        auto itr = uniqueLabels.find(label);
        itr->second += 1;
      }
  Subscript out;
  for (const auto &[label, count] : uniqueLabels)
    if (count == 1)
      out.push_back(label);
  return out;
}

/// Parses input and output subscripts from user provided equation string.
/// This functions assumes that equation string is validated using
/// `validateEquation` function before it is parsed.
static FailureOr<IOSubscripts> parseInputOutputSubscripts(StringRef equation) {
  SmallVector<StringRef> inputOutputSplit =
      llvm::to_vector(llvm::split(equation, "->"));
  // parse all input subscripts by ignoring spaces
  SmallVector<StringRef> inputSubscripts =
      llvm::to_vector(llvm::split(inputOutputSplit[0], ","));
  IOSubscripts ioSubscripts;
  for (const auto &inputSubscript : inputSubscripts) {
    Subscript s;
    for (auto &label : inputSubscript)
      if (label != ' ')
        s.push_back(label);
    ioSubscripts.inputs.push_back(s);
  }
  // if `inputOutputSplit` has size one when -> is not passed.
  if (inputOutputSplit.size() == 1) {
    ioSubscripts.outputs = computeOutputSubscript(ioSubscripts.inputs);
  } else {
    Subscript s;
    if (!inputOutputSplit[1].empty()) {
      // If only `->` is provided but output subscript is empty and all
      // dimensions are reduced leading to a scalar output.
      // When provided, we parse output subscript by ignoring space
      for (const auto &label : inputOutputSplit[1])
        if (label != ' ')
          s.push_back(label);
    }
    ioSubscripts.outputs = s;
  }
  return ioSubscripts;
}

/// Validate input subscript using parsed subscript string for each
/// input and type information for that input.
static LogicalResult validateInputsSubscript(const IOSubscripts &subscripts,
                                             TypeRange inputOperands,
                                             std::optional<Location> loc,
                                             ErrorFn emitErrorFn) {
  if (inputOperands.size() != 1 && inputOperands.size() != 2)
    return emitErrorFn(loc, Twine("einsum op may only have 1 or 2 inputs"));
  if (subscripts.inputs.size() != inputOperands.size())
    return emitErrorFn(
        loc, Twine("each tensor input should have a subscript. Received ") +
                 Twine(inputOperands.size()) + Twine(" tensor operands and ") +
                 Twine(subscripts.inputs.size()) + Twine(" input subscripts"));
  llvm::SmallMapVector<char, int64_t, 8> /*<label, dimension, 8>*/ allLabelDims;
  for (const auto &[inputIdx, it] :
       llvm::enumerate(llvm::zip(subscripts.inputs, inputOperands))) {
    auto [subscript, operand] = it;
    // Check if each dimension has assigned label
    if (static_cast<int64_t>(subscript.size()) !=
        cast<RankedTensorType>(operand).getRank())
      return emitErrorFn(
          loc, Twine("each tensor dimension must have a label. Tensor input ") +
                   Twine(inputIdx) + Twine(" has rank of ") +
                   Twine(cast<RankedTensorType>(operand).getRank()) +
                   Twine(" but subscript size is ") + Twine(subscript.size()));
    for (const auto &[label, dimension] :
         llvm::zip(subscript, cast<RankedTensorType>(operand).getShape())) {
      // check if label to dimension mapping is unique between all inputs.
      // If label is shared between the inputs, corresponding dimension must
      // match. for example, ('ij,jk->ik', a, b) is valid for a =
      // tensor<4x5xf32>, b = tensor<5x6xf32> but invalid for a =
      // tensor<4x6xf32>, b = tensor<5x6xf32>
      if (allLabelDims.count(label) == 0) {
        allLabelDims.insert(std::pair<char, int64_t>(label, dimension));
      } else {
        if (allLabelDims[label] != dimension)
          return emitErrorFn(loc, Twine("label `") + Twine(label) +
                                      Twine("` is repeated between inputs but "
                                            "dimensions are not same"));
      }
    }
  }
  return success();
}

/// Validate output subscript using parsed output substring and optional
/// type information.
/// Output subscript validation contains both labels validation and checking
/// if label dimensions match with input if label is present in the input.
/// However, this function doesn't check if output type is correct since that
/// work is done by `inferOutputShape`. Parameter `output` is passed when this
/// function is used within `verify`.
static LogicalResult validateOutputSubscript(const IOSubscripts &subscript,
                                             std::optional<Location> loc,
                                             ErrorFn emitErrorFn,
                                             TensorType output = nullptr) {
  if (output &&
      static_cast<int64_t>(subscript.outputs.size()) != output.getRank()) {
    return emitErrorFn(loc, Twine("output tensor has rank ") +
                                Twine(output.getRank()) +
                                Twine(" but subscript has size ") +
                                Twine(subscript.outputs.size()));
  }
  llvm::SmallSet<char, 8> inputLabels;
  for (auto &inp : subscript.inputs)
    inputLabels.insert(inp.begin(), inp.end());
  llvm::SmallSet<char, 4> outputLabels;
  for (const char &l : subscript.outputs) {
    if (!inputLabels.contains(l))
      return emitErrorFn(
          loc, Twine("output label `") + Twine(l) +
                   Twine("` does not appear in the input subscript string"));
    if (!outputLabels.contains(l))
      outputLabels.insert(l);
    else
      return emitErrorFn(loc,
                         Twine("label `") + Twine(l) +
                             Twine("` is repeated in the output substring"));
  }
  return success();
}

static LogicalResult inferOutputShapeImpl(const IOSubscripts &ioSubscripts,
                                          TypeRange inputOperands,
                                          SmallVector<int64_t> &outputShape,
                                          std::optional<Location> loc,
                                          ErrorFn emitErrorFn) {
  llvm::SmallMapVector<char, int64_t, 16> inputLabelsDims;
  for (const auto &[subscript, operand] :
       llvm::zip((ioSubscripts).inputs, inputOperands)) {
    for (const auto &[label, dims] :
         llvm::zip(subscript, cast<RankedTensorType>(operand).getShape()))
      if (inputLabelsDims.count(label) == 0)
        inputLabelsDims.insert(std::pair<char, int64_t>(label, dims));
  }

  for (const auto &label : (ioSubscripts).outputs) {
    if (inputLabelsDims.count(label) == 0)
      return emitErrorFn(loc,
                         "missing output label from input label map is not "
                         "expected error in shape inference");
    outputShape.push_back(inputLabelsDims[label]);
  }
  return success();
}

FailureOr<SmallVector<int64_t>>
einsum::inferOutputShape(StringRef equation, TypeRange inputOperands,
                         std::optional<Location> loc, ErrorFn emitErrorFn) {
  if (failed(validateEquation(equation, loc, emitErrorFn)))
    return failure();
  auto ioSubscripts = parseInputOutputSubscripts(equation);
  if (failed(ioSubscripts))
    return failure();
  if (failed(validateInputsSubscript(*ioSubscripts, inputOperands, loc,
                                     emitErrorFn)))
    return failure();
  if (failed(validateOutputSubscript(*ioSubscripts, loc, emitErrorFn)))
    return failure();
  SmallVector<int64_t> outputShape;
  if (failed(inferOutputShapeImpl(*ioSubscripts, inputOperands, outputShape,
                                  loc, emitErrorFn))) {
    return failure();
  }
  return outputShape;
}
