from typing import List, Optional

import mlir_tensorrt.runtime.api as runtime
import mlir_tensorrt.compiler.api as compiler

from tripy.backend.utils import TensorInfo
from tripy.common import Array, datatype, device
from tripy.common.exception import raise_error
from tripy.frontend import Tensor, dynamic_dim
from tripy.utils import Result, from_dims, log_time

G_RUNTIME_CLIENT = None


def _get_runtime_client() -> runtime.RuntimeClient:
    global G_RUNTIME_CLIENT
    if G_RUNTIME_CLIENT is None:
        G_RUNTIME_CLIENT = runtime.RuntimeClient()

    return G_RUNTIME_CLIENT


NUMPY_TO_MLIR_TRT = {
    datatype.int8: runtime.ScalarTypeCode.i8,
    datatype.int32: runtime.ScalarTypeCode.i32,
    datatype.int64: runtime.ScalarTypeCode.i64,
    datatype.uint8: runtime.ScalarTypeCode.ui8,
    datatype.float16: runtime.ScalarTypeCode.f16,
    datatype.float32: runtime.ScalarTypeCode.f32,
    datatype.bool: runtime.ScalarTypeCode.i1,
    datatype.float8: runtime.ScalarTypeCode.f8e4m3fn,
    datatype.bfloat16: runtime.ScalarTypeCode.bf16,
}

MLIR_TRT_TO_NUMPY = {v: k for k, v in NUMPY_TO_MLIR_TRT.items()}


def _convert_to_runtime_dtype(dtype: datatype.dtype) -> runtime.ScalarTypeCode:
    if dtype not in NUMPY_TO_MLIR_TRT:
        raise_error(f"Data type: '{dtype}' does not have a corresponding runtime data type")
    return NUMPY_TO_MLIR_TRT.get(dtype)


def _convert_to_tripy_dtype(dtype: runtime.ScalarTypeCode) -> datatype.dtype:
    if dtype not in MLIR_TRT_TO_NUMPY:
        raise_error(f"Data type: '{dtype}' does not have a corresponding numpy data type")
    return MLIR_TRT_TO_NUMPY.get(dtype)


def _convert_to_memref(
    inp: Array, runtime_client: runtime.RuntimeClient, stream: Optional[runtime.Stream] = None
) -> Result[runtime.MemRefValue]:
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


def _get_output_tensor_info(
    executable: runtime.Executable, output_runtime_shapes: List[int], output_devices=List[device]
):
    signature = executable.get_signature("main")
    offset = signature.get_num_input_args()
    output_info = []

    for output_index in range(signature.get_num_output_args()):
        arg_index = output_index + offset
        arg = signature.get_arg(arg_index)
        assert compiler.MemRefType.isinstance(arg) or compiler.ScalarType.isinstance(
            arg
        ), "Argument must be either MemRefType or ScalarType"
        assert compiler.MemRefType.isinstance(
            arg
        ), "ScalarType argument are not yet supported"  # 158: Add scalar type output argument support.
        memref = compiler.MemRefType(arg)
        dtype = _convert_to_tripy_dtype(memref.dtype)
        device_type = "gpu" if memref.address_space == runtime.PointerType.device else "cpu"
        if output_devices[output_index]:
            device_type = output_devices[output_index].kind
        is_static_shape = all(dim >= 0 for dim in memref.shape)
        if is_static_shape:
            output_info.append(TensorInfo(tuple([dynamic_dim(s) for s in memref.shape]), dtype, device(device_type)))
        else:
            upper_bounds = signature.get_arg_bound(arg_index).max()
            assert len(upper_bounds) == len(memref.shape), "Upper bounds and shape length must match"

            max_shape = [upper if dim < 0 else dim for dim, upper in zip(memref.shape, upper_bounds)]
            output_info.append(
                TensorInfo(
                    tuple(
                        [
                            dynamic_dim(runtime, min=None, opt=None, max=max)
                            for max, runtime in zip(max_shape, output_runtime_shapes[output_index])
                        ]
                    ),
                    dtype,
                    device(device_type),
                )
            )

    return output_info


class Executor:
    def __init__(self, executable: runtime.Executable) -> None:
        self.executable = executable
        self.runtime_client = _get_runtime_client()
        self.stream = self.runtime_client.create_stream()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)

    # Slice the output buffer (allocated with max shapes) to get the data with runtime shapes.
    def slice_outputs(self, outputs, out_tensor_info):
        def slice_to_match(output_shape, runtime_shape):
            slice_spec = [slice(None)] * len(runtime_shape)  # Default slice is to include everything
            for index, (dim1, dim2) in enumerate(zip(output_shape, runtime_shape)):
                if dim1 > dim2:
                    slice_spec[index] = slice(0, dim2)
            return slice_spec

        for out_index, (output, out_info) in enumerate(zip(outputs, out_tensor_info)):
            runtime_shape = tuple([s.runtime_value for s in out_info.shape])
            output_shape = output.shape
            if output_shape != runtime_shape:
                t = Tensor(output)
                slice_index = tuple(slice_to_match(output_shape, runtime_shape))
                sliced_t = t[slice_index].eval()
                outputs[out_index] = sliced_t

    @log_time
    def execute(
        self, output_runtime_shapes: List[int], output_devices=List[device], inputs: List[Tensor] = []
    ) -> List[Array]:
        from tripy.frontend.trace.ops import Storage

        # HACK (#109): Remove `get_runtime_shapes` once we can infer runtime shapes from executable.
        # HACK (#155): Remove `get_devices` once executable output tensor location matches Trace IR.
        out_tensor_info = _get_output_tensor_info(self.executable, output_runtime_shapes, output_devices)

        in_args = []
        for inp in inputs:
            assert isinstance(inp.trace_tensor.producer, Storage)
            memref = _convert_to_memref(inp.trace_tensor.producer.data, self.runtime_client)
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
            for info in out_tensor_info
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
        for idx, out_info in enumerate(out_tensor_info):
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

        self.slice_outputs(outputs, out_tensor_info)

        return outputs
