# REQUIRES: tensorrt-version-ge-10.0
# REQUIRES: host-has-at-least-1-gpus
# REQUIRES: debug-print
# RUN: %PYTHON %s 2>&1 | FileCheck %s

import mlir_tensorrt.compiler.api as api
from mlir_tensorrt.compiler.ir import *
import tempfile
import glob
import os
import json
import gc

STATIC_ASM = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1 : tensor<2x3x4xf32>
}
"""


def layer_metadata_callback(op) -> str:
    print("layer_metadata_callback CALLED")
    return "TEST_CUSTOM_METADATA"


def compile_asm():
    with Context() as context:
        m = Module.parse(STATIC_ASM)
        client = api.CompilerClient(context)

        with tempfile.TemporaryDirectory() as tmp:
            opts = api.StableHLOToExecutableOptions(
                client,
                [
                    "--tensorrt-builder-opt-level=3",
                    "--tensorrt-strongly-typed=false",
                    "--debug=true",
                    "--debug-only=translate-to-tensorrt,stablehlo-clustering",
                    f"--tensorrt-layer-info-dir={tmp}",
                ],
            )

            opts.set_tensorrt_translation_metadata_callback(layer_metadata_callback)

            api.compiler_stablehlo_to_executable(client, m.operation.clone(), opts)

            json_file = glob.glob(os.path.join(tmp, "*"))[0]

            metadata = json.load(open(json_file, "r"))["Layers"][-1]["Metadata"]
            print(metadata)


print("Compiling ASM")
compile_asm()
# CHECK-LABEL: Compiling ASM
# CHECK: layer_metadata_callback CALLED
# CHECK: TEST_CUSTOM_METADATA


def layer_metadata_callback2(op) -> str:
    print("layer_metadata_callback2 CALLED")
    return "TEST_CUSTOM_METADATA2"


def compile_multiple():
    # Compile multiple times with different callbacks to ensure pass manager caching doesn't
    # cause issues.
    with Context() as context:
        m = Module.parse(STATIC_ASM)
        client = api.CompilerClient(context)
        opts0 = api.StableHLOToExecutableOptions(
            client,
            ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
        )
        opts0.set_tensorrt_translation_metadata_callback(layer_metadata_callback)
        api.compiler_stablehlo_to_executable(client, m.operation.clone(), opts0)

        del opts0
        gc.collect()

        opts1 = api.StableHLOToExecutableOptions(
            client,
            ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
        )
        opts1.set_tensorrt_translation_metadata_callback(layer_metadata_callback2)
        api.compiler_stablehlo_to_executable(client, m.operation.clone(), opts1)


print("Checking multiple compile calls")
compile_multiple()

# CHECK-LABEL: Checking multiple compile calls
# CHECK: layer_metadata_callback CALLED
# CHECK: layer_metadata_callback2 CALLED
