//===- Options.h ----------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Utilities for defining and round-tripping bundles of `llvm::cl` options.
///
/// This header provides:
/// - `mlir::CLOptionScope`: an owning scope for `llvm::cl` option objects that
///   can be registered either globally (process command line) or locally (a
///   private `llvm::cl::SubCommand` for request-local parsing).
/// - `mlir::OptionsGroup` / `mlir::LocalScopedOptionsGroup`: small helpers for
///   grouping options and constructing locally-scoped bundles.
///
/// Comparison to other LLVM/MLIR option mechanisms:
/// - **MLIR pass/pipeline options** (e.g. `PassPipelineOptions` and per-pass
///   option parsing) are primarily designed around pass manager and pipeline
///   syntax. In contrast, this utility is pass-agnostic: it uses `llvm::cl`
///   directly and focuses on parsing/printing stand-alone option bundles (see
///   `CLOptionScope::parseFromString` and `CLOptionScope::print`).
/// - The idea for the “local `llvm::cl::SubCommand` + parse/print” and option
///   to attach to the global top-level subcommand pattern is
///   inspired by IREE's `OptionsBinder` (see
///  `https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Utils/OptionUtils.h`).
/// - The IREE `OptionsBinder` provides a similar “global vs local” concept, but
///   binds flags onto external storage and includes additional features such as
///   change tracking and optimization-level default overrides.
///   This header instead uses an owning options struct (the struct members are
///   the `llvm::cl` options) and provides a small, self-contained parser/pretty
///   printer for round-tripping. This enables the class to be much more compact
///   but has the disadvantage that the LLVM CommandLine.h header is required to
///   be included everywhere where these utilities are used.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_SUPPORT_OPTIONS
#define MLIR_TENSORRT_COMMON_SUPPORT_OPTIONS

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir {

/// A scoped, owning container for `llvm::cl` command-line options.
///
/// An instance of `CLOptionScope` owns the underlying `llvm::cl` option objects
/// and binds them to a particular `llvm::cl::SubCommand`. This supports two
/// common usage patterns:
/// - **GlobalScope**: options are registered with the process-wide top-level
///   subcommand and participate in normal `llvm::cl::ParseCommandLineOptions`
///   parsing.
/// - **LocalScope**: options are registered with a private subcommand owned by
///   the `CLOptionScope` instance, enabling request-local parsing from an
///   option string.
///
/// Local parsing intentionally supports a reduced syntax (see
/// `parseFromString`).
class CLOptionScope : protected llvm::cl::SubCommand {
public:
  struct GlobalScope {};
  struct LocalScope {};

  using ErrorCallback = std::function<void(llvm::StringRef message)>;
  using PrintTokenList = llvm::SmallVector<std::string, 8>;

  /// If `DataType` is parsed by an llvm::cl generic parser (e.g. enums with
  /// `cl::values(...)`), wrap the parser so we can print the current value
  /// using its symbolic option spelling rather than a numeric value.
  template <typename DataType>
  struct GenericOptionParser : public llvm::cl::parser<DataType> {
    using llvm::cl::parser<DataType>::parser;

    std::optional<llvm::StringRef> findArgStrForValue(const DataType &value) {
      for (auto &it : this->Values)
        if (it.V.compare(value))
          return it.Name;
      return std::nullopt;
    }
  };

  template <typename DataType>
  using OptionParser =
      std::conditional_t<std::is_base_of_v<llvm::cl::generic_parser_base,
                                           llvm::cl::parser<DataType>>,
                         GenericOptionParser<DataType>,
                         llvm::cl::parser<DataType>>;

  CLOptionScope(GlobalScope, llvm::cl::OptionCategory *category = nullptr)
      : bindingSub(&llvm::cl::SubCommand::getTopLevel()), category(category) {}
  CLOptionScope(LocalScope, llvm::cl::OptionCategory *category = nullptr)
      : bindingSub(static_cast<llvm::cl::SubCommand *>(this)),
        category(category) {}

  CLOptionScope(const CLOptionScope &) = delete;
  CLOptionScope &operator=(const CLOptionScope &) = delete;
  CLOptionScope(CLOptionScope &&) = delete;
  CLOptionScope &operator=(CLOptionScope &&) = delete;

  /// Returns true if this scope is bound to the global top-level
  /// `llvm::cl::SubCommand`.
  bool isGlobalScope() const;

  /// Parse an option string into this scope (works for LocalScope; also
  /// works for GlobalScope but does not integrate with env/rsp/config).
  ///
  /// Supported syntax:
  /// - GNU tokenization (quotes/backslashes) via `cl::TokenizeGNUCommandLine`.
  /// - `--name`, `--name=value`, `-name`, `-name=value`
  /// - optional outer `{ ... }` wrapper
  ///
  /// Not supported:
  /// - response files/config/env merging
  /// - positional args
  /// - `--name value` (value stealing)
  /// - `-o value` / `-O2` style prefix/grouping semantics
  LogicalResult parseFromString(llvm::StringRef optionString,
                                const ErrorCallback &onError = ErrorCallback());

  /// Print the current option values in a form that can be parsed by
  /// `parseFromString` (a space-separated bundle wrapped in `{...}`).
  ///
  /// If `includeDefaults` is false, only options that differ from their
  /// defaults (or non-empty lists) are printed.
  void print(llvm::raw_ostream &os, bool includeDefaults = false) const;

protected:
  llvm::cl::SubCommand &subcommandForOptions() { return *bindingSub; }
  const llvm::cl::SubCommand &subcommandForOptions() const {
    return *bindingSub;
  }

  template <typename ParserT, typename ValueT, typename = void>
  struct HasFindArgStrForValue : std::false_type {};
  template <typename ParserT, typename ValueT>
  struct HasFindArgStrForValue<
      ParserT, ValueT,
      std::void_t<decltype(std::declval<ParserT &>().findArgStrForValue(
          std::declval<const ValueT &>()))>> : std::true_type {};

  /// Register an option printer for a named option.
  void registerPrinter(
      llvm::StringRef name,
      std::function<void(PrintTokenList &out, bool includeDefaults)> emit);

  static void printQuoted(llvm::raw_ostream &os, llvm::StringRef value);

  static void printValueToken(llvm::raw_ostream &os, llvm::StringRef value) {
    printQuoted(os, value);
  }
  static void printValueToken(llvm::raw_ostream &os, const std::string &value) {
    printQuoted(os, value);
  }
  template <typename T>
  static void printValueToken(llvm::raw_ostream &os, const T &value) {
    if constexpr (std::is_enum_v<T>)
      os << static_cast<std::underlying_type_t<T>>(value);
    else
      os << value;
  }

  template <typename ParserT, typename T>
  static void printValueTokenWithParser(llvm::raw_ostream &os, ParserT &parser,
                                        const T &value) {
    if constexpr (HasFindArgStrForValue<ParserT, T>::value) {
      if (auto arg = parser.findArgStrForValue(value)) {
        os << *arg;
        return;
      }
    }
    printValueToken(os, value);
  }

public:
  /// A typed scalar option that is scoped to this scope's chosen
  /// subcommand.
  template <typename T, typename Parser = OptionParser<T>>
  class Option : public llvm::cl::opt<T, /*ExternalStorage=*/false, Parser> {
  public:
    template <typename... Mods>
    Option(CLOptionScope &parent, llvm::StringRef name, Mods &&...mods)
        : llvm::cl::opt<T, /*ExternalStorage=*/false, Parser>(
              name, llvm::cl::sub(parent.subcommandForOptions()),
              std::forward<Mods>(mods)...) {
      parent.registerPrinter(
          name, [this](PrintTokenList &out, bool includeDefaults) {
            const bool isDefault = this->getDefault().compare(this->getValue());
            if (!includeDefaults && isDefault)
              return;

            std::string token;
            llvm::raw_string_ostream os(token);
            os << "--" << this->ArgStr;

            // For boolean options, prefer `--flag` when true. Print `=false`
            // when explicitly emitting false (e.g. includeDefaults or
            // non-default false).
            if constexpr (std::is_same_v<T, bool>) {
              if (this->getValue()) {
                out.push_back(os.str());
                return;
              }
              os << "=false";
              out.push_back(os.str());
              return;
            }

            // ValueDisallowed options can't be printed with `=...` (their value
            // is part of the option name).
            if (this->getValueExpectedFlag() != llvm::cl::ValueDisallowed) {
              os << "=";
              CLOptionScope::printValueTokenWithParser(os, this->getParser(),
                                                       this->getValue());
            }
            out.push_back(os.str());
          });

      if (parent.category)
        this->addCategory(*parent.category);
    }
  };

  /// A typed list option scoped to this scope's chosen subcommand.
  template <typename T, typename Parser = OptionParser<T>>
  class ListOption : public llvm::cl::list<T, /*StorageClass=*/bool, Parser> {
  public:
    template <typename... Mods>
    ListOption(CLOptionScope &parent, llvm::StringRef name, Mods &&...mods)
        : llvm::cl::list<T, /*StorageClass=*/bool, Parser>(
              name, llvm::cl::sub(parent.subcommandForOptions()),
              std::forward<Mods>(mods)...) {
      parent.registerPrinter(
          name, [this](PrintTokenList &out, bool includeDefaults) {
            if (this->empty() && !includeDefaults)
              return;
            for (const auto &elt : *this) {
              std::string token;
              llvm::raw_string_ostream os(token);
              os << "--" << this->ArgStr;
              if (this->getValueExpectedFlag() != llvm::cl::ValueDisallowed) {
                os << "=";
                CLOptionScope::printValueTokenWithParser(os, this->getParser(),
                                                         elt);
              }
              out.push_back(os.str());
            }
          });
      if (parent.category)
        this->addCategory(*parent.category);
    }

    /// Override the list contents with the provided values.
    void assign(llvm::ArrayRef<T> values) {
      this->clear();
      for (const T &v : values)
        this->push_back(v);
    }

    /// Override the list contents with the provided values.
    void assign(std::initializer_list<T> values) {
      assign(llvm::ArrayRef(values));
    }

    /// Override the list contents with values convertible to `T` (e.g.
    /// string literals for `T=std::string`).
    template <typename U>
    void assign(std::initializer_list<U> values) {
      this->clear();
      for (const U &v : values)
        this->push_back(T(v));
    }
  };

private:
  struct PrintEntry {
    llvm::StringRef name;
    std::function<void(PrintTokenList &out, bool includeDefaults)> emit;
  };

  llvm::cl::SubCommand *bindingSub = nullptr;
  std::vector<PrintEntry> printers;
  llvm::cl::OptionCategory *category = nullptr;
};

struct OptionsGroup {
  OptionsGroup(mlir::CLOptionScope &scope) : scope(scope), ctx(scope) {}
  virtual ~OptionsGroup() = default;

  template <typename T, typename Parser = mlir::CLOptionScope::OptionParser<T>>
  using Option = mlir::CLOptionScope::Option<T, Parser>;

  template <typename T, typename Parser = mlir::CLOptionScope::OptionParser<T>>
  using ListOption = mlir::CLOptionScope::ListOption<T, Parser>;

  /// The underlying `llvm::cl::SubCommand` scope that options are registered
  /// against.
  mlir::CLOptionScope &scope;

  /// Backward-compatibility alias: older code refers to the scope as `ctx`.
  mlir::CLOptionScope &ctx;
};

/// Adapt an OptionsGroup into a standalone locally-scoped CLOptionScope
/// wrapper. It implicitly converts to a const reference to the provider type.
template <typename GroupT>
class LocalScopedOptionsGroup {
private:
  static_assert(std::is_base_of_v<OptionsGroup, GroupT>,
                "GroupT must derive from OptionsGroup");
  CLOptionScope scope{CLOptionScope::LocalScope{}};
  GroupT group{scope};

public:
  LocalScopedOptionsGroup() = default;

  GroupT &get() { return group; }
  const GroupT &get() const { return group; }

  operator const GroupT &() const { return get(); }
};

//===----------------------------------------------------------------------===//
// options_group_tuple
//
// This helper template is used to create a tuple of (unique pointers to)
// OptionsGroup types.
//===----------------------------------------------------------------------===//

namespace detail {
template <typename TupleType>
struct OptionsGroupTupleHelper;

template <typename... Ts>
struct OptionsGroupTupleHelper<std::tuple<std::unique_ptr<Ts>...>> {
  using type = std::tuple<std::unique_ptr<Ts>...>;
  static auto get(mlir::CLOptionScope &ctx) {
    return std::make_tuple(std::make_unique<Ts>(ctx)...);
  }
};
} // namespace detail

template <typename... Ts>
using options_group_tuple = typename detail::OptionsGroupTupleHelper<
    std::tuple<std::unique_ptr<Ts>...>>::type;

template <typename TupleType>
auto make_options_group_tuple(mlir::CLOptionScope &ctx) {
  return detail::OptionsGroupTupleHelper<TupleType>::get(ctx);
}

} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_SUPPORT_OPTIONS
