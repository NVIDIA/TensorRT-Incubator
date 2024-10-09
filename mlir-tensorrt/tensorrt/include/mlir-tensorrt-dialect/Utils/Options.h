//===- Options.h ------------------------------------------------*- C++ -*-===//
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
#ifndef MLIR_TENSORRT_DIALECT_UTILS_OPTIONS
#define MLIR_TENSORRT_DIALECT_UTILS_OPTIONS

#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// This is a very simple class that wraps the parsing/printing of a set of
/// textual flags/options. It only supports named options (not positional
/// arguments). This class is loosely inspired by both the 'option binder'
/// utility in IREE as well as the `mlir::detail::PassOptions` class while
/// trying to overcome the limitations. In MLIR PassOptions and PipelineOptions,
/// the `llvm::cl::SubCommand` is used to bind options directly on a struct like
///
/// ```
/// class MyClass : PassOptions {
///    Option<bool> myFlag{*this, "my-flag", llvm::cl::init(false), ...};
/// };
/// ```
///
/// This has a number of annoying consequences, including:
/// 1. `MyClass` now has implicitly deleted copy-constructors
/// 2. You can't nest `MyClass` in another class and have the options be
///    by automatically combined
///
/// So to work around this, we can aggregate groups of related options into
/// structs like this:
///
/// ```
/// struct MyOptions {
///   bool myFlag;
///
///   void addToOptions(OptionsContext &ctx) {
///     ctx.addOption("my-flag", myFlag, llvm::cl::init(false), ...);
///  }
/// };
/// ```
///
/// This allows for binding to OptionsContext with external storage (in
/// MyOptions) and thus care must be take to have suitable lifetimes. This
/// pattern is very useful when combining different options structs:
///
/// ```
/// struct TopOptions : public OptionsContext {
///    MyOptions subOptions;
///    int anotherOpt;
///
///    TopOptions() {
///      subOptions.addToOptions(*this);
///      addOption("another-opt", anotherOpt, ...);
///    }
/// };
/// ...
/// TopOptions opts;
/// std::string err;
/// if(failed(opts.parse(argv, err))) {
///   return emitError(...) << err;
/// }
/// ```
class OptionsContext : public llvm::cl::SubCommand {
public:
  OptionsContext() = default;
  OptionsContext(const OptionsContext &) = delete;
  OptionsContext(OptionsContext &&) = default;
  virtual ~OptionsContext() = default;

protected:
  /// Add an option to this context. The storage `value` must outlive the
  /// OptionsContext.
  template <typename DataType, typename ParserClass, typename... Mods>
  void addOptionImpl(llvm::StringRef name, DataType &value, Mods &&...mods) {
    auto opt =
        std::make_unique<llvm::cl::opt<DataType, /*ExternalStorage=*/true,
                                       /*ParserClass=*/ParserClass>>(
            llvm::cl::sub(*this), name, llvm::cl::location(value),
            std::forward<Mods>(mods)...);
    printers[opt.get()] = [opt = opt.get()](llvm::raw_ostream &os) {
      mlir::detail::pass_options::printOptionValue<decltype(opt->getParser())>(
          os, opt->getValue());
    };
    options.push_back(OptionInfo{std::move(opt)});
  }

public:
  /// Add an option to this context. The storage `value` must outlive the
  /// OptionsContext.
  template <typename DataType, typename... Mods>
  void addOption(llvm::StringRef name, DataType &value, Mods &&...mods) {
    addOptionImpl<DataType, llvm::cl::parser<DataType>, Mods...>(
        name, value, std::forward<Mods>(mods)...);
  }

  /// Add an option to this context using a custom parser class (given as
  /// template argument). The storage `value` must outlive the OptionsContext.
  template <typename ParserClass, typename DataType, typename... Mods>
  void addOptionWithParser(llvm::StringRef name, DataType &value,
                           Mods &&...mods) {
    addOptionImpl<DataType, ParserClass, Mods...>(name, value,
                                                  std::forward<Mods>(mods)...);
  }

  /// Add a list options to this context. This context will have duplicated
  /// storage, but that's OK.
  /// Remember to pass `llvm::cl::CommaSeparated` for comma separated lists.
  template <typename DataType, typename ValueType, typename... Mods>
  void addList(llvm::StringRef name, ValueType &value, Mods... mods) {
    auto item = std::make_unique<llvm::cl::list<DataType>>(
        name, llvm::cl::sub(*this), std::forward<Mods>(mods)...);
    item->setCallback(
        [&value](const DataType &newVal) { value.push_back(newVal); });
    printers[item.get()] = [opt = item.get()](llvm::raw_ostream &os) {
      auto printItem = [&](const DataType &value) {
        mlir::detail::pass_options::printOptionValue<
            decltype(opt->getParser())>(os, value);
      };
      llvm::interleave(*opt, os, printItem, ",");
    };
    options.push_back(OptionInfo{std::move(item)});
  }

  /// Parse the options from an array of arguments.
  LogicalResult parse(llvm::ArrayRef<llvm::StringRef> argv, std::string &error);

  /// Print the options to the stream.
  void print(llvm::raw_ostream &os) const;

  /// Get a hash derived from the string representation of the options.
  /// Derived classes can use this method to incorporate additional factors
  /// which cannot be captured by the options string representation. Returning
  /// nullopt indicates that the options cannot or should not be hashed and used
  /// as a cache key.
  virtual std::optional<llvm::hash_code> getHash() const;

private:
  struct OptionInfo {
    std::unique_ptr<llvm::cl::Option> option;
  };
  /// Storage for the options.
  std::vector<OptionInfo> options;
  llvm::DenseMap<llvm::cl::Option *, std::function<void(llvm::raw_ostream &)>>
      printers;
};
} // namespace mlir

#endif /* MLIR_TENSORRT_DIALECT_UTILS_OPTIONS */
