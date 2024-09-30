# RUN: %PYTHON %s | FileCheck %s
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np
import torch


main_memref_io = """
func.func @main(%arg0: tensor<?x3x4xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 3, 4], opt = [5, 3, 4], max = [10, 3, 4]>}) -> tensor<?x3x4xf32> {
  %1 = stablehlo.add %arg0, %arg0 : (tensor<?x3x4xf32>, tensor<?x3x4xf32>) -> tensor<?x3x4xf32>
  func.return %1 : tensor<?x3x4xf32>
}
"""

main_scalar_io = """
func.func @main(%arg0: f32) -> f32 {
  func.return %arg0 : f32
}
"""


class Test:
    def __init__(self, program: str):
        # Build/parse the main function.
        with ir.Context() as context:
            m = ir.Module.parse(program)

            # Use the compiler API to compile to executable.
            client = compiler.CompilerClient(context)
            opts = compiler.StableHLOToExecutableOptions(
                client,
                ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
            )
            opts.set_debug_options(False, [], "tmp")
            self.exe = compiler.compiler_stablehlo_to_executable(
                client, m.operation, opts
            )

        self.client = runtime.RuntimeClient()
        self.stream = self.client.create_stream()
        self.devices = self.client.get_devices()
        self.session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)

        self.host_data = np.ones(shape=(1, 3, 4), dtype=np.float32)

    def create_memref(self, shape, type):
        return self.client.create_memref(
            np.ones(shape, dtype=type).data,
            device=self.devices[0],
            stream=self.stream,
        )

    def create_memref_from_dlpack(self, shape, type):
        arr = torch.ones(shape, dtype=type)
        memref = self.client.create_memref_view_from_dlpack(arr.__dlpack__())
        memref = self.client.copy_to_device(memref, device=self.client.get_devices()[0])
        print(f"Memref stride: {memref.strides}")
        return memref

    def create_memref_host(self, shape, type):
        h_memref = self.client.create_host_memref_view(
            ptr=int(self.host_data.ctypes.data),
            dtype=runtime.ScalarTypeCode.f32,
            shape=shape,
        )
        return h_memref

    def create_scalar(self, value):
        return self.client.create_scalar(value, runtime.ScalarTypeCode.i64)

    def execute(self, arg: runtime.RuntimeValue):
        session = runtime.RuntimeSession(self.session_options, self.exe)
        try:
            session.execute_function(
                "main",
                in_args=[arg],
                out_args=[arg],
                stream=self.stream,
                client=self.client,
            )
            print("Test passed succesfully")
        except runtime.MTRTException as e:
            print(f"MTRTException: {e}")


if __name__ == "__main__":
    t = Test(main_memref_io)
    print("TEST: runtime shape mismatch")
    t.execute(t.create_memref((5, 4, 2), np.float32))
    print("TEST: runtime rank mismatch")
    t.execute(t.create_memref((5, 3), np.float32))
    print("TEST: runtime memref element type mismatch")
    t.execute(t.create_memref((5, 3), np.int32))
    print("TEST: unit stride dimension")
    t.execute(t.create_memref_from_dlpack((1, 3, 4), torch.float32))
    print("TEST: runtime memref address space mismatch")
    t.execute(t.create_memref_host((1, 3, 4), np.float32))
    print("TEST: runtime type mismatch")
    t.execute(t.create_scalar(5))
    t = Test(main_scalar_io)
    print("TEST: function with no output arguments")
    t.execute(t.create_memref((5, 3), np.int32))

# CHECK-LABEL: TEST: runtime shape mismatch
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: Input argument 0 validation failed against corresponding function signature arg 0. Reason: InvalidArgument: Runtime shape mismatch. Expected [-9223372036854775808, 3, 4] but received [5, 4, 2]
# CHECK-LABEL: TEST: runtime rank mismatch
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: Input argument 0 validation failed against corresponding function signature arg 0. Reason: InvalidArgument: function expects a memref type with rank 3 but receieved 2
# CHECK-LABEL: TEST: runtime memref element type mismatch
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: Input argument 0 validation failed against corresponding function signature arg 0. Reason: InvalidArgument: function expects a memref type with element type f32 but receieved i32
# CHECK-LABEL: TEST: unit stride dimension
#       CHECK: Memref stride: [1, 4, 1]
#       CHECK: Test passed succesfully
# CHECK-LABEL: TEST: runtime memref address space mismatch
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: Input argument 0 validation failed against corresponding function signature arg 0. Reason: InvalidArgument: function expects a memref type with address space device but receieved host
# CHECK-LABEL: TEST: runtime type mismatch
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: Input argument 0 validation failed against corresponding function signature arg 0. Reason: InvalidArgument: function expects a memref type but received scalar type
# CHECK-LABEL: TEST: function with no output arguments
#       CHECK: MTRTException: InvalidArgument: InvalidArgument: function expects 0 output args (destination args) but received 1
