//===- XlaAdaptor.h ------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_TENSORRT_PJRT_XLAADAPTOR
#define INCLUDE_MLIR_TENSORRT_PJRT_XLAADAPTOR

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#pragma GCC diagnostic ignored "-Wc++98-compat-extra-semi"
#endif
#include "xla/pjrt/c/pjrt_c_api.h"       // IWYU pragma: export
#include "xla/pjrt/compile_options.pb.h" // IWYU pragma: export
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif // INCLUDE_MLIR_TENSORRT_PJRT_XLAADAPTOR
