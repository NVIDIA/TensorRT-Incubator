//===- ADTExtras.h ----------------------------------------------*- C++ -*-===//
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
/// Expansions of template or support tools defined under 'llvm/ADT'.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_UTILS_ADTEXTRAS_H
#define MLIR_EXECUTOR_UTILS_ADTEXTRAS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FormatVariadicDetails.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

namespace llvm {

namespace support::detail {

// Helper alias to check if T has begin() and end()
template <typename T, typename = void>
struct is_range : std::false_type {};

template <typename T>
struct is_range<T, std::void_t<decltype(std::begin(std::declval<T>())),
                               decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

} // namespace support::detail

//===----------------------------------------------------------------------===//
// Teach the 'llvm::formatv' some sensible default way of handling types that
// can be used with range-style for loops. This includes any template which
// doesn't have trailing "size" template variable at the end (e.g. std::array
// and llvm::SmallVector). Those are handled below.
//===----------------------------------------------------------------------===//

template <template <typename...> class Container, typename... U>
class format_provider<
    Container<U...>,
    std::enable_if_t<
        support::detail::is_range_v<Container<U...>> &&
        !std::is_same_v<llvm::iterator_range<U...>, Container<U...>>>> {
public:
  using ValueType = Container<U...>;
  using ElementType = typename ValueType::value_type;
  static_assert(!support::detail::uses_missing_provider<ElementType>::value,
                "no format provider for container element type");

  static void format(const ValueType &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    auto adapter = support::detail::build_format_adapter(
        llvm::make_range(V.begin(), V.end()));
    adapter.format(Stream, Style);
  }
};

//===----------------------------------------------------------------------===//
// Teach the 'llvm::formatv' how to handle std::array
//===----------------------------------------------------------------------===//

template <template <typename, std::size_t> class Container, typename T,
          std::size_t N>
class format_provider<Container<T, N>> {
public:
  using ValueType = Container<T, N>;
  using ElementType = typename ValueType::value_type;
  static_assert(!support::detail::uses_missing_provider<ElementType>::value,
                "no format provider for container element type");
  static void format(const ValueType &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    auto adapter = support::detail::build_format_adapter(
        llvm::make_range(V.begin(), V.end()));
    adapter.format(Stream, Style);
  }
};

//===----------------------------------------------------------------------===//
// Teach the 'llvm::formatv' how to handle llvm::SmallVector.
//===----------------------------------------------------------------------===//

template <template <typename, unsigned> class Container, typename T, unsigned N>
class format_provider<Container<T, N>> {
public:
  using ValueType = Container<T, N>;
  static void format(const ValueType &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    auto adapter = support::detail::build_format_adapter(
        llvm::make_range(V.begin(), V.end()));
    adapter.format(Stream, Style);
  }
};

//===----------------------------------------------------------------------===//
// Teach the 'llvm::formatv' how to handle tuple types. This also will
// automatically enable it to hande iterators that produce tuples, e.g.
// 'llvm::zip'.
//===----------------------------------------------------------------------===//

template <typename... Ts>
class format_provider<std::tuple<Ts...>> {
  using ValueType = std::tuple<Ts...>;

  template <typename T>
  static std::enable_if_t<support::detail::is_range_v<T>, int>
  dispatchFormatting(const T &Ve, StringRef ElStyle,
                     llvm::raw_ostream &Stream) {
    Stream << "[";
    auto Adapter = support::detail::build_format_adapter(
        llvm::make_range(Ve.begin(), Ve.end()));
    Adapter.format(Stream, ElStyle);
    Stream << "]";
    return 1;
  }

  template <typename T>
  static std::enable_if_t<!support::detail::is_range_v<T>, int>
  dispatchFormatting(const T &Ve, StringRef ElStyle,
                     llvm::raw_ostream &Stream) {
    auto Adapter = support::detail::build_format_adapter(Ve);
    Adapter.format(Stream, ElStyle);
    return 1;
  }

  template <typename T>
  static StringRef consumeOneOption(const T &, StringRef &Style, char Indicator,
                                    StringRef Default) {
    if (Style.empty())
      return Default;
    if (Style.front() != Indicator)
      return Default;
    Style = Style.drop_front();
    if (Style.empty()) {
      assert(false && "Invalid range style");
      return Default;
    }

    for (const char *D : std::array<const char *, 3>{"[]", "<>", "()"}) {
      if (Style.front() != D[0])
        continue;
      size_t End = Style.find_first_of(D[1]);
      if (End == StringRef::npos) {
        assert(false && "Missing range option end delimeter!");
        return Default;
      }
      StringRef Result = Style.slice(1, End);
      Style = Style.drop_front(End + 1);
      return Result;
    }
    assert(false && "Invalid range style!");
    return Default;
  }

public:
  static void format(const ValueType &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    StringRef seperator = consumeOneOption<void *>(nullptr, Style, '$', ", ");

    auto styles = std::apply(
        [&](auto... x) {
          return std::make_tuple(consumeOneOption(x, Style, '@', "")...);
        },
        V);

    unsigned iter = 0;
    auto op = [&](const auto &Ve, StringRef ElStyle) {
      if (iter++ > 0)
        Stream << seperator;
      return dispatchFormatting(Ve, ElStyle, Stream);
    };
    Stream << "Tuple<";
    std::apply(
        [&](auto &&...x) {
          return std::apply(
              [&](auto &&...y) { return std::make_tuple(op(x, y)...); },
              styles);
        },
        V);
    Stream << ">";
  }
};

//===----------------------------------------------------------------------===//
// Teach the 'llvm::formatv' how to handle raw zip-iterators without
// 'llvm::make_range' wrapper.
//===----------------------------------------------------------------------===//

template <template <typename...> class ItType, typename T, typename U,
          typename... Args>
class format_provider<detail::zippy<ItType, T, U, Args...>> {
public:
  using ValueType = detail::zippy<ItType, T, U, Args...>;
  using ElementType = typename ValueType::value_type;
  static_assert(!support::detail::uses_missing_provider<ElementType>::value,
                "no format provider for container element type");

  static void format(const ValueType &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    auto adapter = support::detail::build_format_adapter(
        llvm::make_range(V.begin(), V.end()));
    adapter.format(Stream, Style);
  }
};

} // namespace llvm

#endif // MLIR_EXECUTOR_UTILS_ADTEXTRAS_H
