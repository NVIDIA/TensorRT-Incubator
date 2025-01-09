//===- Options.cpp --------- ----------------------------------------------===//
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
/// Implementation of CL option parsing utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

LogicalResult OptionsContext::parse(llvm::ArrayRef<llvm::StringRef> argv,
                                    std::string &error) {
  for (llvm::StringRef arg : argv) {
    llvm::StringRef term = arg.trim();
    if (!term.consume_front("--") && !term.consume_front("-")) {
      error = llvm::formatv("could not parse: {0}", arg);
      return failure();
    }

    auto [key, value] = term.split("=");
    llvm::cl::Option *option = this->OptionsMap.lookup(key);
    if (!option) {
      error = llvm::formatv("no option registered with name {0}", key);
      return failure();
    }

    if (llvm::cl::ProvidePositionalOption(option, value, argv.size())) {
      error = llvm::formatv("could not parse {0}", arg);
      return failure();
    }
  }
  return success();
}

void OptionsContext::print(llvm::raw_ostream &os) const {
  llvm::interleave(
      this->OptionsMap, os,
      [&](const auto &it) {
        os << "--" << it.getKey() << "=";
        llvm::cl::Option *opt = it.getValue();
        auto printer = this->printers.lookup(opt);
        if (printer)
          printer(os);
      },
      " ");
}

SmallVector<std::string> OptionsContext::serialize() const {
  assert(getHash() && "cannot serialize non-hashable options");
  SmallVector<std::string> result;
  for (const auto &[key, option] : this->OptionsMap) {
    std::string val;
    {
      llvm::raw_string_ostream ss(val);
      auto printer = this->printers.lookup(option);
      if (printer)
        printer(ss);
    }
    result.push_back(llvm::formatv("--{0}={1}", key, val));
  }
  return result;
}

std::optional<llvm::hash_code> OptionsContext::getHash() const {
  // We hash by just hashing the string representation.
  llvm::SmallString<128> str;
  {
    llvm::raw_svector_ostream os(str);
    print(os);
  }
  return llvm::hash_value(str);
}
