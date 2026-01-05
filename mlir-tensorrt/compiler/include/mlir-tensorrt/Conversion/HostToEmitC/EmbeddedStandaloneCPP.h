//===- EmbeddedStandaloneCPP.h ----------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Provides access to StandaloneCPP runtime source/header contents embedded in
/// the compiler binary at build time.
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_EMBEDDEDSTANDALONECPP_H
#define MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_EMBEDDEDSTANDALONECPP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mtrt::compiler::emitc_support {

struct EmbeddedTextFile {
  llvm::StringRef path;
  llvm::StringRef contents;
};

/// Returns the list of embedded StandaloneCPP runtime files.
llvm::ArrayRef<EmbeddedTextFile> getEmbeddedStandaloneCPPFiles();

/// Returns the embedded contents for the given path, or an empty StringRef if
/// not found.
llvm::StringRef getEmbeddedStandaloneCPPFileContents(llvm::StringRef path);

} // namespace mtrt::compiler::emitc_support

#endif // MLIR_TENSORRT_CONVERSION_HOSTTOEMITC_EMBEDDEDSTANDALONECPP_H
