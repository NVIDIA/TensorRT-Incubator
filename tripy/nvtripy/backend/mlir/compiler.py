#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional, Tuple

import mlir_tensorrt.compiler.api as compiler
from mlir_tensorrt.compiler import ir

import nvtripy.config as cfg
from nvtripy import config, utils
from nvtripy.backend.mlir.utils import (
    make_ir_context,
    map_error_to_user_code_and_raise,
    redirect_stderr,
)
from nvtripy.logging import logger

G_COMPILER_CLIENT = None
G_TIMING_CACHE_FILE = cfg.timing_cache_file_path


# Avoid instantiating the compiler more than once.
def _get_compiler_objects() -> Tuple[ir.Context, compiler.CompilerClient]:
    global G_COMPILER_CLIENT, G_TIMING_CACHE_FILE
    if G_TIMING_CACHE_FILE != cfg.timing_cache_file_path:
        # Reinitialize the compiler if the timing cache file path has changed.
        global G_COMPILER_CLIENT
        G_COMPILER_CLIENT = None
        G_TIMING_CACHE_FILE = cfg.timing_cache_file_path

    ctx = make_ir_context()
    if G_COMPILER_CLIENT is None:
        G_COMPILER_CLIENT = compiler.CompilerClient(ctx)
    return ctx, G_COMPILER_CLIENT


class Compiler:
    def __init__(self, trt_builder_opt_level=0) -> None:
        self.mlir_context, self.compiler_client = _get_compiler_objects()
        self.trt_builder_opt_level = trt_builder_opt_level

    def _make_mlir_opts(self, trt_builder_opt_level):
        opts = [
            f"--tensorrt-timing-cache-path={G_TIMING_CACHE_FILE}",
            f"--tensorrt-builder-opt-level={trt_builder_opt_level}",
            "--tensorrt-strongly-typed=True",
        ]
        if config.enable_mlir_debug or config.enable_tensorrt_debug:
            opts.append("--debug=true")
            if config.enable_mlir_debug:
                opts.append(f"--debug-only={config.mlir_debug_types}")
                opts.append(f"--mlir-print-ir-tree-dir={config.mlir_debug_tree_path}")
            if config.enable_tensorrt_debug:
                opts.append(f"--tensorrt-layer-info-dir={config.tensorrt_debug_path}")
                opts.append(f"--tensorrt-engines-dir={config.tensorrt_debug_path}")
        return compiler.StableHLOToExecutableOptions(self.compiler_client, opts)

    def compile_stabehlo_program(self, code: str) -> compiler.Executable:
        with self.mlir_context:
            module = ir.Module.parse(code)
            opts = self._make_mlir_opts(self.trt_builder_opt_level)
            return compiler.compiler_stablehlo_to_executable(self.compiler_client, module.operation, opts)

    # The optional flat_ir parameter is used to generate nicer error messages.
    @utils.log_time
    def compile(self, mlir_module: ir.Module, flat_ir: Optional["FlatIR"] = None) -> compiler.Executable:
        logger.mlir(lambda: f"{mlir_module.operation.get_asm(large_elements_limit=32)}\n")
        opts = self._make_mlir_opts(self.trt_builder_opt_level)

        try:
            with redirect_stderr() as outfile:
                executable = compiler.compiler_stablehlo_to_executable(
                    self.compiler_client, mlir_module.operation, opts
                )
        except Exception as exc:
            outfile.flush()
            outfile.seek(0)
            stderr = outfile.read()
            map_error_to_user_code_and_raise(flat_ir, exc, stderr.decode())
        else:
            return executable
