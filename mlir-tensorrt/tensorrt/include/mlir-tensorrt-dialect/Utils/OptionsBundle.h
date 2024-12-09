//===- OptionsBundle.h ------------------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_DIALECT_UTILS_OPTIONS_BUNDLE
#define MLIR_TENSORRT_DIALECT_UTILS_OPTIONS_BUNDLE

#include "mlir-tensorrt-dialect/Utils/Options.h"
#include "llvm/Support/Error.h"
#include <tuple>

namespace mlir {

/// This class allows us to have options classes subscribe to one or more option
/// providers.
template <typename... OptionProviders>
class OptionsBundle : public OptionsContext {
public:
  OptionsBundle() {
    std::apply(
        [&](auto &...optionProvider) {
          (optionProvider.addToOptions(*this), ...);
        },
        optionProviders);
  }

  template <typename OptionsProviderT>
  const OptionsProviderT &get() const {
    return std::get<OptionsProviderT>(optionProviders);
  }

  template <typename OptionsProviderT>
  OptionsProviderT &get() {
    return std::get<OptionsProviderT>(optionProviders);
  }

  llvm::Error finalize() override {
    llvm::Error result = llvm::Error::success();
    std::apply(
        [&](auto &...optionProvider) {
          ((result = std::move(llvm::joinErrors(std::move(result),
                                                optionProvider.finalize()))),
           ...);
        },
        optionProviders);

    return result;
  }

private:
  std::tuple<OptionProviders...> optionProviders{};
};
} // namespace mlir

#endif /* MLIR_TENSORRT_DIALECT_UTILS_OPTIONS_BUNDLE */
