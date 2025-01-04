# RUN: %PYTHON %s 2>&1
# REQUIRES: host-has-at-least-1-gpus
import os
import tempfile

import mlir_tensorrt.compiler.api as api
from mlir_tensorrt.compiler.ir import *

ASM = """
func.func @main(%arg0: tensor<?x3x4xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 3, 4], opt = [5, 3, 4], max = [10, 3, 4]>}) -> tensor<?x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<?x3x4xf32>, tensor<?x3x4xf32>) -> tensor<?x3x4xf32>
  func.return %1 : tensor<?x3x4xf32>
}
"""


def _check_debug_files(mlir_tree_path, trt_path):
    assert os.path.exists(mlir_tree_path), "MLIR debug directory is missing."
    assert os.path.exists(trt_path), "TensorRT debug directory is missing."

    # check trt engine and json file
    trt_file_types = {file.split(".")[-1] for file in os.listdir(trt_path)}
    assert trt_file_types == {"engine", "json"}


def compile_asm(ASM):
    with Context() as context:
        m = Module.parse(ASM)
        client = api.CompilerClient(context)

        mlir_tree_path = tempfile.TemporaryDirectory()
        trt_path = tempfile.TemporaryDirectory()

        opts = api.StableHLOToExecutableOptions(
            client,
            [
                "--tensorrt-builder-opt-level=3",
                "--tensorrt-strongly-typed=false",
                "--debug=true",
                "--debug-only=translate-to-tensorrt,stablehlo-clustering",
                f"--mlir-print-ir-tree-dir={mlir_tree_path.name}",
                f"--tensorrt-layer-info-dir={trt_path.name}",
                f"--tensorrt-engines-dir={trt_path.name}",
            ],
        )

        api.compiler_stablehlo_to_executable(client, m.operation, opts)
        _check_debug_files(mlir_tree_path.name, trt_path.name)


compile_asm(ASM)
