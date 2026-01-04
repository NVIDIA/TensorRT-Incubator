#include "mlir-tensorrt-common/Support/Options.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace mlir {

bool CLOptionScope::isGlobalScope() const {
  return bindingSub == &llvm::cl::SubCommand::getTopLevel();
}

void CLOptionScope::registerPrinter(
    llvm::StringRef name,
    std::function<void(PrintTokenList &out, bool includeDefaults)> emit) {
  printers.push_back(PrintEntry{name, std::move(emit)});
}

void CLOptionScope::printQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
  // Emit a GNU-tokenizable double-quoted string when necessary.
  //
  // We keep this conservative: quote anything with whitespace or quotes or
  // backslashes, and escape common control characters.
  const bool needsQuotes =
      value.empty() ||
      value.find_first_of(" \t\n\r\"\\") != llvm::StringRef::npos;
  if (!needsQuotes) {
    os << value;
    return;
  }

  os << '"';
  for (char c : value) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << '"';
}

void CLOptionScope::print(llvm::raw_ostream &os, bool includeDefaults) const {
  PrintTokenList tokens;

  // Stable ordering by option name.
  llvm::SmallVector<const PrintEntry *, 16> entries;
  entries.reserve(printers.size());
  for (const auto &e : printers)
    entries.push_back(&e);
  llvm::sort(entries, [](const PrintEntry *a, const PrintEntry *b) {
    return a->name < b->name;
  });

  for (const PrintEntry *e : entries)
    e->emit(tokens, includeDefaults);

  os << "{";
  llvm::interleave(tokens, os, [&](const std::string &t) { os << t; }, " ");
  os << "}";
}

LogicalResult CLOptionScope::parseFromString(llvm::StringRef optionString,
                                             const ErrorCallback &onError) {
  llvm::cl::SubCommand &sub = subcommandForOptions();

  optionString = optionString.trim();
  if (optionString.empty())
    return success();

  // Allow MLIR-style option bundles: "{a=b c=d}".
  if (optionString.consume_front("{") && optionString.consume_back("}"))
    optionString = optionString.trim();

  llvm::SmallVector<const char *, 16> argv;
  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver saver(alloc);
  llvm::cl::TokenizeGNUCommandLine(optionString, saver, argv,
                                   /*MarkEOLs=*/false);

  for (size_t i = 0, e = argv.size(); i < e; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.empty())
      continue;

    llvm::StringRef nameVal;
    if (arg.starts_with("--")) {
      nameVal = arg.drop_front(2);
    } else if (arg.starts_with("-")) {
      nameVal = arg.drop_front(1);
    } else {
      if (onError)
        onError("positional arguments not supported (prefix with '--')");
      return failure();
    }

    llvm::StringRef name;
    llvm::StringRef value; // default => nullptr data() (no explicit value)
    size_t eqPos = nameVal.find('=');
    if (eqPos == llvm::StringRef::npos) {
      name = nameVal;
    } else {
      name = nameVal.take_front(eqPos);
      value = nameVal.drop_front(eqPos + 1);
    }

    if (name.empty()) {
      if (onError)
        onError("empty option name");
      return failure();
    }

    auto foundIt = sub.OptionsMap.find(name);
    if (foundIt == sub.OptionsMap.end()) {
      if (onError) {
        std::string msg("option not found: ");
        msg.append(name.begin(), name.end());
        onError(msg);
      }
      return failure();
    }

    llvm::cl::Option *opt = foundIt->second;
    if (llvm::cl::ProvidePositionalOption(opt, value,
                                          static_cast<int>(i + 1))) {
      if (onError) {
        std::string msg("option parse error for: ");
        msg.append(name.begin(), name.end());
        if (value.data()) {
          msg.push_back('=');
          msg.append(value.begin(), value.end());
        }
        onError(msg);
      }
      return failure();
    }
  }

  return success();
}

} // namespace mlir
