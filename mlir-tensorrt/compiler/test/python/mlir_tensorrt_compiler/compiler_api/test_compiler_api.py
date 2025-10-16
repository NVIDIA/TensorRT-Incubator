# REQUIRES: tensorrt-version-ge-10.0
# REQUIRES: host-has-at-least-1-gpus
# REQUIRES: debug-print
# RUN: %pick-one-gpu %PYTHON %s 2>&1 | FileCheck %s

# This test requires TensorRT >= 10.0 since we are testing ability
# to set the 'tensorrt-strongly-typed' flag.

import sys
from typing import List

import mlir_tensorrt.compiler.api as api
from mlir_tensorrt.compiler.ir import *
from mlir_tensorrt.compiler.ir import _GlobalDebug

STATIC_ASM = """
func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %1 : tensor<2x3x4xf32>
}
"""

DYNAMIC_ASM = """
func.func @main(%arg0: tensor<?x3x4xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 3, 4], opt = [5, 3, 4], max = [10, 3, 4]>}) -> tensor<?x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<?x3x4xf32>, tensor<?x3x4xf32>) -> tensor<?x3x4xf32>
  func.return %1 : tensor<?x3x4xf32>
}
"""


def print_shape_or_strides(array):
    # We use i64 min to represent "?".
    return ", ".join((str(x) if x != -9223372036854775808 else "?") for x in array)


def _test_memref(memref: api.MemRefType):
    memref = api.MemRefType(memref)
    assert isinstance(memref, api.MemRefType)
    print(f"type: {type(memref)}")
    print(f"shape: [{print_shape_or_strides(memref.shape)}]")
    print(f"strides: [{print_shape_or_strides(memref.strides)}]")
    print(f"element_type: {memref.dtype}")
    print(f"address_space: {memref.address_space}")


def _test_func_signature(sig: api.PyFunctionSignature):
    print(sig)
    print(f"Num of args: {sig.get_num_args()}")
    print(f"Num of results: {sig.get_num_results()}")
    print(f"Num of input args: {sig.get_num_input_args()}")
    print(f"Num of output args: {sig.get_num_output_args()}")
    print(f"Num of arg bounds: {sig.get_num_arg_bounds()}")
    for i in range(sig.get_num_arg_bounds()):
        print(
            f"Arg {i} Bound: min({sig.get_arg_bound(i).min()}), max({sig.get_arg_bound(i).max()})"
        )
    print(f"Num of res bounds: {sig.get_num_res_bounds()}")
    for i in range(sig.get_num_res_bounds()):
        print(
            f"Arg {i} Bound: min({sig.get_res_bound(i).min()}), max({sig.get_res_bound(i).max()})"
        )
    print(f"Shape function name: {sig.get_shape_func_name()}")
    _test_memref(sig.get_arg(0))
    _test_memref(sig.get_arg(1))


def flush():
    # Flush buffers before runnign the compilation so as not to mess up
    # the CHECK ordering requirements.
    sys.stdout.flush()
    sys.stderr.flush()


with Context() as context:
    client = api.CompilerClient(context)
    _GlobalDebug.flag = True
    _GlobalDebug.set_types("translate-to-tensorrt")

    def compile(m, options: List[str]):
        task = client.get_compilation_task(
            "stablehlo-to-executable",
            options,
        )
        task.run(m)
        return api.translate_mlir_to_executable(m)

    def compile_asm(ASM):
        m = Module.parse(ASM)

        opts = [
            "--tensorrt-builder-opt-level=0",
            "--tensorrt-strongly-typed=false",
            "--tensorrt-workspace-memory-pool-limit=1gb",
        ]

        print("running compilation (1)")
        flush()
        exe = compile(m.operation.clone(), opts)
        # Options don't change, so the cached pipeline should be re-used.
        print("running compilation (2)")
        flush()
        exe = compile(m.operation.clone(), opts)

        sig = exe.get_signature("main")
        _test_func_signature(sig)

        # Verify that "strongly typed" flag is passed through, but
        # catch any exceptions since that is only supported on TensorRT 10.

        # Changing the options should cause new pipeline to be generated, creating new builder.
        opts = [
            "--tensorrt-builder-opt-level=1",
            "--tensorrt-strongly-typed=true",
            "--tensorrt-workspace-memory-pool-limit=1024kiB",
            "--mlir-timing",
        ]

        try:
            print("running compilation (3)")
            flush()
            exe = compile(m.operation.clone(), opts)
        except:
            pass

    print("Compiling static asm")
    compile_asm(STATIC_ASM)
    # CHECK-LABEL: Compiling static asm
    # CHECK-LABEL: running compilation (1)
    # CHECK: [translate-to-tensorrt] TranslateToTensorRTEnginePass is generating a new TensorRT builder
    # CHECK: [translate-to-tensorrt] timing cache path was not specified, creating a fresh timing cache
    # CHECK: [translate-to-tensorrt] Setting builder optimization level to 0
    # CHECK: [translate-to-tensorrt] setting TensorRT builder workspace memory pool limit = 1073741824 bytes
    # CHECK-LABEL: running compilation (2)
    # CHECK-NOT: {{.*}} generating a new TensorRT builder {{.*}}
    # CHECK-NOT: {{.*}} timing cache path was not specified {{.*}}
    # CHECK: FunctionSignature(Signature<args=[MemRef<2x3x4xf32, strides=[12, 4, 1], device>, MemRef<2x3x4xf32, strides=[12, 4, 1], device>],
    # CHECK-SAME: results=[], num_output_args=1, arg_bounds=[UNK, UNK], result_bounds=[], cconv=unpacked, undef=[], abi_version=0>)
    # CHECK: Num of args: 2
    # CHECK: Num of results: 0
    # CHECK: Num of input args: 1
    # CHECK: Num of output args: 1
    # CHECK: Num of arg bounds: 2
    # CHECK: Arg 0 Bound: min([]), max([])
    # CHECK: Arg 1 Bound: min([]), max([])
    # CHECK: Num of res bounds: 0
    # CHECK: Shape function name: None
    # CHECK: type: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
    # CHECK: shape: [2, 3, 4]
    # CHECK: strides: [12, 4, 1]
    # CHECK: element_type: ScalarTypeCode.f32
    # CHECK: address_space: PointerType.device
    # CHECK: type: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
    # CHECK: shape: [2, 3, 4]
    # CHECK: strides: [12, 4, 1]
    # CHECK: element_type: ScalarTypeCode.f32
    # CHECK: address_space: PointerType.device

    # The first time executing a pipeline with the second set of options results
    # in a cache miss:

    # CHECK-LABEL: running compilation (3)
    # CHECK: TranslateToTensorRTEnginePass is generating a new TensorRT builder
    # CHECK: timing cache path was not specified, creating a fresh timing cache
    # CHECK: [translate-to-tensorrt] enabling 'strongly-typed' mode in TensorRT translation
    # CHECK: [translate-to-tensorrt] Setting builder optimization level to 1
    # CHECK: [translate-to-tensorrt] setting TensorRT builder workspace memory pool limit = 1048576 bytes

    print("Compiling dynamic asm")
    compile_asm(DYNAMIC_ASM)
# CHECK-LABEL: Compiling dynamic asm
# CHECK: running compilation (1)
# CHECK-NOT: TranslateToTensorRTEnginePass is generating a new TensorRT builder
# CHECK-NOT  timing cache path was not specified, creating a fresh timing cache
# CHECK: [translate-to-tensorrt] Setting builder optimization level to 0
# CHECK: [translate-to-tensorrt] setting TensorRT builder workspace memory pool limit = 1073741824 bytes
# CHECK: running compilation (2)
# CHECK-NOT: {{.*}} generating a new TensorRT builder {{.*}}
# CHECK-NOT: {{.*}} timing cache path was not specified {{.*}}
# CHECK: FunctionSignature(Signature<args=[MemRef<?x3x4xf32, strides=[12, 4, 1], device>, MemRef<?x3x4xf32, strides=[12, 4, 1], device>],
# CHECK-SAME: results=[], num_output_args=1,
# CHECK-SAME: arg_bounds=[dim_bounds<min = [1,3,4], max = [10,3,4]>, dim_bounds<min = [1,3,4], max = [10,3,4]>], result_bounds=[]
# CHECK-SAME: cconv=unpacked, undef=[], abi_version=0>
# CHECK: Num of args: 2
# CHECK: Num of results: 0
# CHECK: Num of input args: 1
# CHECK: Num of output args: 1
# CHECK: Num of arg bounds: 2
# CHECK: Arg 0 Bound: min([1, 3, 4]), max([10, 3, 4])
# CHECK: Arg 1 Bound: min([1, 3, 4]), max([10, 3, 4])
# CHECK: Num of res bounds: 0
# CHECK: Shape function name: main_get_shapes
# CHECK: type: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
# CHECK: shape: [?, 3, 4]
# CHECK: strides: [12, 4, 1]
# CHECK: element_type: ScalarTypeCode.f32
# CHECK: address_space: PointerType.device
# CHECK: type: <class 'mlir_tensorrt.compiler._mlir_libs._api.MemRefType'>
# CHECK: shape: [?, 3, 4]
# CHECK: strides: [12, 4, 1]
# CHECK: element_type: ScalarTypeCode.f32
# CHECK: address_space: PointerType.device
# CHECK: running compilation (3)
# CHECK-NOT: TranslateToTensorRTEnginePass is generating a new TensorRT builder
# CHECK-NOT: timing cache path was not specified, creating a fresh timing cache
# CHECK: [translate-to-tensorrt] enabling 'strongly-typed' mode in TensorRT translation
# CHECK: [translate-to-tensorrt] Setting builder optimization level to 1
# CHECK: [translate-to-tensorrt] setting TensorRT builder workspace memory pool limit = 1048576 bytes

# CHECK: Execution time report
