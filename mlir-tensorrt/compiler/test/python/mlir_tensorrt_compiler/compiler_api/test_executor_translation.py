# RUN: %PYTHON %s | FileCheck %s
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir

with ir.Context() as ctx:
    client = compiler.CompilerClient(ctx)
    ASM = """
    func.func @main(%arg0: i32, %arg1: i32) -> i32 attributes{
      executor.function_metadata=#executor.func_meta<[i32, i32],[i32], num_output_args = 0>
    }{
      %0 = executor.sremi %arg0, %arg1 : i32
      return %0 : i32
    }
    """

    m = ir.Module.parse(ASM)
    exe = compiler.translate_mlir_to_executable(m.operation)

    sig = exe.get_signature("main")
    print(sig)
    # CHECK: FunctionSignature(Signature<args=[i32, i32], results=[i32], num_output_args=0, arg_bounds=[UNK, UNK],
    # CHECK-SAME: result_bounds=[UNK], cconv=unpacked, undef=[], abi_version=0>
