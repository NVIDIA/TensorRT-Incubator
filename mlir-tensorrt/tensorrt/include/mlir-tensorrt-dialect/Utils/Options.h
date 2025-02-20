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
#include <type_traits>

namespace mlir {

namespace detail {

/// MlirOptionAdaptor is derived from the MLIR base option class used to define
/// pass and pipeline options. It is used to override the parent `llvm::cl::opt`
/// callback to also populate an external storage reference.
template <typename DataType,
          typename OptionParser =
              mlir::detail::PassOptions::OptionParser<DataType>>
class MlirOptionAdaptor
    : public mlir::detail::PassOptions::Option<DataType, OptionParser> {
public:
  using Base = mlir::detail::PassOptions::Option<DataType, OptionParser>;

  template <typename... Args>
  MlirOptionAdaptor(mlir::detail::PassOptions &parent, StringRef arg,
                    DataType &external, Args &&...args)
      : Base(parent, arg, std::forward<Args>(args)...) {
    this->setCallback([&](const auto &v) {
      external = v;
      this->optHasValue = true;
    });
  }
};
} // namespace detail

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

  /// Convenience type for declaring options as class/struct member without
  /// having to explicitly write `addOption` in the constructor of the options
  /// container class.
  template <typename T, typename... Mods>
  struct Option {

    T value;
    operator const T &() const { return value; }
    Option &operator=(const T &rhs) {
      value = rhs;
      return *this;
    }

    template <typename U = T>
    std::enable_if_t<std::is_same_v<std::string, U>, bool> empty() const {
      return value.empty();
    }

    // Implicit conversion operator to StringRef, enabled only if T is
    // std::string
    template <typename U = T>
    operator typename std::enable_if_t<std::is_same_v<U, std::string>,
                                       llvm::StringRef>() const {
      return value;
    }

    template <typename... Args>
    Option(OptionsContext *ctx, llvm::StringRef name, Args &&...args) {
      ctx->addOption<T, Mods...>(name, value, std::forward<Args>(args)...);
    }

    Option() = delete;
    Option(const Option &) = delete;
    Option(Option &&) = default;
    Option &operator=(const Option &) = delete;
  };

  /// Convenience type for declaring vector class member as an option without
  /// having to explicitly write `addList` in the constructor of the options
  /// container class.
  template <typename T, typename... Mods>
  struct ListOption {
    std::vector<T> value;
    operator const std::vector<T> &() const { return value; }

    auto empty() const { return value.empty(); }
    auto begin() const { return value.begin(); }
    auto end() const { return value.end(); }
    auto front() const { return value.front(); }
    auto back() const { return value.back(); }
    auto emplace_back(T &&item) {
      return value.emplace_back(std::forward<T>(item));
    }
    auto push_back(T &&item) { return value.push_back(std::forward<T>(item)); }

    template <typename... Args>
    ListOption(OptionsContext *ctx, llvm::StringRef name, Args &&...args) {
      ctx->addList<T, Mods...>(name, value, std::forward<Args>(args)...);
    }
  };

  /// A tag which can be passed to an option constructed to the constructor
  /// argument list using the `Option|ListOption` classes above or the
  /// `addOption|addList` functions defined below. The tag indicates that the
  /// option should be omitted from CLI parameters when converting the derived
  /// OptionsContext class to a `mlir::PassPipelineOptions|PassOptions` class
  /// when using a templated adaptor class defined below.
  struct OmitFromCLI {};

protected:
  // Helper function to exclude all elements of type `OmitFromCLI` from the
  // tuple `t`.
  template <typename Tuple, std::size_t I>
  static constexpr auto filterCLIOmissionTag(const Tuple &t,
                                             bool &omitFromMlirPipeline) {
    if constexpr (!std::is_same_v<OmitFromCLI,
                                  std::tuple_element_t<I, Tuple>>) {
      return std::make_tuple(std::get<I>(t));
    } else {
      omitFromMlirPipeline = true;
      return std::tuple<>();
    }
  }

  // Helper to collect valid arguments into a new tuple
  template <typename Tuple, std::size_t... I>
  constexpr auto filterCLIOmissionTag(bool &omitFromMlirPipeline,
                                      const Tuple &t,
                                      std::index_sequence<I...>) {
    return std::tuple_cat(
        filterCLIOmissionTag<Tuple, I>(t, omitFromMlirPipeline)...);
  }

  // Helper to construct llvm::cl::opt values.
  template <typename OptType, typename... Args>
  auto buildOpt(bool &omitFromMlirPipeline, Args &&...args) {
    // Pack all arguments into a tuple.
    auto t = std::make_tuple(std::forward<Args>(args)...);
    // Create an index sequence based on the tuple size
    constexpr std::size_t size = std::tuple_size<decltype(t)>::value;
    constexpr auto indices = std::make_index_sequence<size>{};
    // Collect only the arguments that are not of type `OmitFromCLI`, which is
    // our custom tag. Arguments of all other types are forwarded to the option
    // construction function.
    auto valid_args = filterCLIOmissionTag(omitFromMlirPipeline, t, indices);
    return std::apply(
        [&](auto &&...valid_args_inner) {
          return std::make_unique<OptType>(
              std::forward<decltype(valid_args_inner)>(valid_args_inner)...);
        },
        valid_args);
  }

  /// Add an option to this context. The storage `value` must outlive the
  /// OptionsContext.
  template <typename DataType, typename ParserClass, typename... Mods>
  void addOptionImpl(llvm::StringRef name, DataType &value, Mods &&...mods) {
    bool omitFromMlirPipeline = false;
    auto opt = buildOpt<llvm::cl::opt<DataType, true, ParserClass>>(
        omitFromMlirPipeline, llvm::cl::sub(*this), name,
        llvm::cl::location(value), std::forward<Mods>(mods)...);

    // Add to the map of printers.
    printers[opt.get()] = [opt = opt.get()](llvm::raw_ostream &os) {
      mlir::detail::pass_options::printOptionValue<decltype(opt->getParser())>(
          os, opt->getValue());
    };

    // Populate the callback which allows for converting this option
    // into a MLIR pipeline option on-the-fly.
    if (!omitFromMlirPipeline)
      mlirOptionConverters[opt.get()] =
          [opt = opt.get(), name = name.str(),
           &value](mlir::detail::PassOptions &parent,
                   std::vector<std::unique_ptr<llvm::cl::Option>> &storage) {
            auto converted =
                std::make_unique<detail::MlirOptionAdaptor<DataType>>(
                    parent, name, value);
            const llvm::cl::OptionValue<DataType> &V = opt->getDefault();
            converted->setInitialValue(V.hasValue() ? V.getValue()
                                                    : DataType());
            converted->setDescription(opt->HelpStr);
            storage.emplace_back(
                std::unique_ptr<llvm::cl::Option>(std::move(converted)));
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
    bool omitFromMlirPipeline = false;
    auto item = buildOpt<llvm::cl::list<DataType>>(omitFromMlirPipeline, name,
                                                   llvm::cl::sub(*this),
                                                   std::forward<Mods>(mods)...);
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

  SmallVector<std::string> serialize() const;

  /// Get a hash derived from the string representation of the options.
  /// Derived classes can use this method to incorporate additional factors
  /// which cannot be captured by the options string representation. Returning
  /// nullopt indicates that the options cannot or should not be hashed and used
  /// as a cache key.
  virtual std::optional<llvm::hash_code> getHash() const;

  /// Update options, if needed, after parsing. This method can be used to
  /// modify options based on the values of other options or can be used to
  /// populate options that were not provided using arbitrarily complex logic
  /// (instead of just a default value).
  virtual llvm::Error finalize() = 0;

  using MlirOptionConverter =
      std::function<void(mlir::detail::PassOptions &,
                         std::vector<std::unique_ptr<llvm::cl::Option>> &)>;

  /// Return the callbacks used to populate a PassOptions or
  /// PassPipelineOptions adaptor.
  const llvm::DenseMap<llvm::cl::Option *, MlirOptionConverter> &
  getMlirOptionsConverters() const {
    return mlirOptionConverters;
  }

private:
  struct OptionInfo {
    std::unique_ptr<llvm::cl::Option> option;
  };
  /// Storage for the options.
  std::vector<OptionInfo> options;
  llvm::DenseMap<llvm::cl::Option *, std::function<void(llvm::raw_ostream &)>>
      printers;

  /// Holds a set of callbacks for populating options of a MLIR PassOptions.
  /// This is only used to implement conversion to and from a MLIR pipeline or
  /// pass options.
  llvm::DenseMap<llvm::cl::Option *, MlirOptionConverter> mlirOptionConverters;
};

/// This type is used to declare an adaptor object which allows using the
/// "AdaptedType", which is derived from OptionsContext, as a MLIR
/// PassPipelineOptions. Note that currently only regular options (and not list
/// options) are passed through. Any list options will retain their defaults.
/// TODO: support list options
template <typename Derived, typename AdaptedType>
class PassPipelineOptionsAdaptor : public mlir::PassPipelineOptions<Derived> {
public:
  static_assert(std::is_base_of_v<mlir::OptionsContext, AdaptedType>,
                "expected AdaptedType to be derived from OptionsContext");

  PassPipelineOptionsAdaptor() {
    storage = std::make_unique<AdaptedType>();
    mlirOptionStorage.reserve(storage->getMlirOptionsConverters().size());
    for (auto &[key, converter] : storage->getMlirOptionsConverters())
      converter(*this, mlirOptionStorage);
  }

  operator const AdaptedType &() const { return *storage; }

private:
  std::vector<std::unique_ptr<llvm::cl::Option>> mlirOptionStorage;
  std::unique_ptr<AdaptedType> storage{nullptr};
};

} // namespace mlir

#endif /* MLIR_TENSORRT_DIALECT_UTILS_OPTIONS */
