from typing import List, Optional

import mlir_tensorrt.runtime.api as runtime

from tripy.backend.utils import TensorInfo
from tripy.common import Array, datatype
from tripy.common.exception import raise_error
from tripy.frontend import Tensor
from tripy.utils import default, Result, from_dims, log_time

G_RUNTIME_CLIENT = None


def _get_runtime_client() -> runtime.RuntimeClient:
    global G_RUNTIME_CLIENT
    if G_RUNTIME_CLIENT is None:
        G_RUNTIME_CLIENT = runtime.RuntimeClient()

    return G_RUNTIME_CLIENT


def _convert_to_runtime_dtype(dtype: datatype.dtype) -> runtime.ScalarTypeCode:
    NUMPY_TO_MLIR_TRT = {
        datatype.int8: runtime.ScalarTypeCode.i8,
        datatype.int32: runtime.ScalarTypeCode.i32,
        datatype.int64: runtime.ScalarTypeCode.i64,
        datatype.uint8: runtime.ScalarTypeCode.i8,
        datatype.float16: runtime.ScalarTypeCode.f16,
        datatype.float32: runtime.ScalarTypeCode.f32,
        datatype.bool: runtime.ScalarTypeCode.i8,
    }

    if dtype not in NUMPY_TO_MLIR_TRT:
        raise_error(f"Data type: '{dtype}' does not have a corresponding runtime data type")
    return NUMPY_TO_MLIR_TRT.get(dtype)


def _convert_to_memref(
    inp: Array, runtime_client: runtime.RuntimeClient, stream: Optional[runtime.Stream] = None
) -> Result[runtime.MemRef]:
    devices = runtime_client.get_devices()

    if inp.device.kind != "gpu":
        # TODO (#136): In the CPU case, just omit the device. MLIR-TRT should perform the host->device copy.
        return Result.ok(
            runtime_client.create_memref(
                inp.byte_buffer,
                shape=inp.shape,
                dtype=_convert_to_runtime_dtype(inp.dtype),
                device=devices[0],
                stream=stream,
            )
        )

    device_index: int = inp.device.index
    if device_index >= len(devices):
        Result.err([f"Requested CUDA device: {device_index} but the only {len(devices)} devices are present"])

    device = devices[device_index]
    return Result.ok(
        runtime_client.create_device_memref_view(
            ptr=inp.byte_buffer.data.ptr,
            dtype=_convert_to_runtime_dtype(inp.dtype),
            shape=inp.shape,
            device=device,
        )
    )


class Executor:
    def __init__(self, executable: runtime.Executable, out_tensor_info: List[TensorInfo] = None) -> None:
        self.runtime_client = _get_runtime_client()
        self.stream = self.runtime_client.create_stream()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)

        self.out_tensor_info = out_tensor_info

    # Slice the output buffer (allocated with max shapes) to get the data with runtime shapes.
    def slice_outputs(self, outputs):
        def slice_to_match(output_shape, runtime_shape):
            slice_spec = [slice(None)] * len(runtime_shape)  # Default slice is to include everything
            for index, (dim1, dim2) in enumerate(zip(output_shape, runtime_shape)):
                if dim1 > dim2:
                    slice_spec[index] = slice(0, dim2)
            return slice_spec

        for out_index, (output, out_info) in enumerate(zip(outputs, self.out_tensor_info)):
            runtime_shape = tuple([s.runtime_value for s in out_info.shape])
            output_shape = output.shape
            if output_shape != runtime_shape:
                t = Tensor(output)
                slice_index = tuple(slice_to_match(output_shape, runtime_shape))
                sliced_t = t[slice_index].eval()
                outputs[out_index] = sliced_t

    @log_time
    def execute(self, inputs: List[Tensor] = []) -> List[Array]:
        from tripy.frontend.trace.ops import Storage

        in_args = []
        for inp in inputs:
            assert isinstance(inp.op, Storage)
            memref = _convert_to_memref(inp.op.data, self.runtime_client)
            if not memref:
                raise_error(
                    "Could not convert tensor to memref",
                    details=[f"Tensor was: ", inp, "Error was: ", memref.error_details],
                )
            in_args.append(memref.value)

        # Allocate output memory and store buffer pointers.
        # mlir-tensorrt requires the output buffer to be of the shape with max bounds.
        outputs = [
            Array(None, shape=from_dims(info.shape, use_max_value=True), dtype=info.dtype, device=info.device)
            for info in self.out_tensor_info
        ]

        out_args = []
        for out in outputs:
            memref = _convert_to_memref(out, self.runtime_client)
            if not memref:
                raise_error("Could not allocate output memref", details=memref.error_details)
            out_args.append(memref.value)

        # Execute and populate device pointers.
        self.session.execute_function("main", in_args=in_args, out_args=out_args, stream=self.stream)
        self.stream.sync()

        # For outputs that were on the host, do the copy back
        for idx, out_info in enumerate(self.out_tensor_info):
            if out_info.device.kind != "gpu":
                host_out = outputs[idx]
                self.runtime_client.copy_to_host(
                    device_memref=out_args[idx],
                    existing_host_memref=self.runtime_client.create_host_memref_view(
                        ptr=int(host_out.byte_buffer.ctypes.data),
                        dtype=_convert_to_runtime_dtype(host_out.dtype),
                        shape=host_out.shape,
                    ),
                    stream=None,
                )

        self.slice_outputs(outputs)

        return outputs
