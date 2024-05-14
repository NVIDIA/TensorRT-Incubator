from typing import List, Optional

import mlir_tensorrt.runtime.api as runtime
import mlir_tensorrt.compiler.api as compiler

from tripy.backend.utils import TensorInfo
from tripy.backend.mlir.utils import get_max_upper_bounds
from tripy.common import Array, datatype, device
from tripy.common.exception import raise_error
from tripy.frontend import Tensor, dynamic_dim
from tripy.utils import Result, from_dims, log_time, make_tuple


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


class Executor:
    def __init__(self, executable: runtime.Executable) -> None:
        self.runtime_client = _get_runtime_client()
        self.stream = self.runtime_client.create_stream()
        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        self.session = runtime.RuntimeSession(session_options, executable)
        self.device = self.runtime_client.get_devices()[0]  # Assume a single device is available.
        self.signature = executable.get_signature("main")

    def _get_inputs_shape_memref(self, inputs):
        inputs_shape_memref = []
        for input in inputs:
            # (#155) Shape function require input arguments to be on device.
            input_shape = Array(
                from_dims(input.trace_tensor.producer.data.shape),
                shape=make_tuple(input.trace_tensor.rank),
                dtype=datatype.int64,
                device=device("cpu"),
            )
            input_shape_memref = self.runtime_client.create_memref(
                input_shape.byte_buffer, shape=make_tuple(input.trace_tensor.rank), dtype=runtime.ScalarTypeCode.i64
            )
            inputs_shape_memref.append(input_shape_memref)
        return inputs_shape_memref

    def _get_outputs_shape_memref(self):
        offset = self.signature.get_num_input_args()
        outputs_shape_memref = []
        for output_index in range(self.signature.get_num_output_args()):
            arg_index = output_index + offset
            arg = self.signature.get_arg(arg_index)
            assert compiler.MemRefType.isinstance(arg)
            memref = runtime.MemRefType(arg)
            rank = len(memref.shape)
            if len(memref.shape) > 0:
                output_shape = Array(memref.shape, shape=make_tuple(rank), dtype=datatype.int64, device=device("cpu"))
                output_shape_memref = self.runtime_client.create_memref(
                    output_shape.byte_buffer, shape=make_tuple(rank), dtype=runtime.ScalarTypeCode.i64
                )
                outputs_shape_memref.append(output_shape_memref)
            else:
                outputs_shape_memref.append(None)
        return outputs_shape_memref

    def _execute_shape_inference(self, inputs_shape_memref, outputs_shape_memref):
        # Only execute shape inference if shape function name is valid.
        if self.signature.get_shape_func_name() is None:
            for memref in outputs_shape_memref:
                if memref is not None:
                    assert (
                        all(dim >= 0 for dim in memref.shape)
                        and f"Output shape {memref.shape} must be statically known if shape inference function is missing. "
                    )
            return None

        self.session.execute_function(
            self.signature.get_shape_func_name(), in_args=inputs_shape_memref, out_args=outputs_shape_memref
        )

        outputs_runtime_shape = [memoryview(s).tolist() for s in outputs_shape_memref]
        return outputs_runtime_shape

    def _get_output_tensor_info(self, outputs_runtime_shape, output_devices):
        offset = self.signature.get_num_input_args()
        outputs_tensor_info = []
        for output_index in range(self.signature.get_num_output_args()):
            arg_index = output_index + offset
            arg = self.signature.get_arg(arg_index)
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
                outputs_tensor_info.append(
                    TensorInfo(tuple([dynamic_dim(s) for s in memref.shape]), dtype, device(device_type))
                )
            else:
                assert outputs_runtime_shape
                upper_bounds = self.signature.get_arg_bound(arg_index).max()
                assert len(upper_bounds) == len(memref.shape), "Upper bounds and shape length must match"
                max_shape = [upper if dim < 0 else dim for dim, upper in zip(memref.shape, upper_bounds)]
                for idx, dim in enumerate(memref.shape):
                    if dim > 0:
                        assert (
                            outputs_runtime_shape[output_index][idx] == dim
                            and f"Inferred runtime shape must be same as static output shape. Expected {dim}, received {outputs_runtime_shape[output_index][idx]}"
                        )
                for idx, dim in enumerate(memref.shape):
                    if dim < 0:
                        assert (
                            upper_bounds[idx] >= outputs_runtime_shape[output_index][idx]
                            and f"Upper bound {upper_bounds[idx]} for a dim at {idx} not be less than runtime shape {outputs_runtime_shape[output_index][idx]}"
                        )
                        # TODO: Improve this check to be stricter.
                        assert (
                            upper_bounds[idx] < outputs_runtime_shape[output_index][idx] + get_max_upper_bounds()
                            and f"Upper bound {upper_bounds[idx]} for a dim at {idx} must not exceed runtime shapes {outputs_runtime_shape[output_index][idx]} by {get_max_upper_bounds()}"
                        )
                runtime_shape = [
                    rs if dim < 0 else dim for dim, rs in zip(memref.shape, outputs_runtime_shape[output_index])
                ]
                outputs_tensor_info.append(
                    TensorInfo(
                        tuple(
                            [
                                dynamic_dim(runtime, min=None, opt=None, max=max)
                                for runtime, max in zip(runtime_shape, max_shape)
                            ]
                        ),
                        dtype,
                        device(device_type),
                    )
                )
        return outputs_tensor_info

    def get_output_tensor_runtime_info(self, inputs, output_devices=List[device]):
        inputs_shape_memref = self._get_inputs_shape_memref(
            inputs
        )  # Can we use executable signature inputs to retrieve this information?
        outputs_shape_memref = self._get_outputs_shape_memref()
        outputs_runtime_shape = self._execute_shape_inference(inputs_shape_memref, outputs_shape_memref)
        output_tensor_info = self._get_output_tensor_info(outputs_runtime_shape, output_devices)
        return output_tensor_info

    @log_time
    def execute(self, output_devices=List[device], inputs: List[Tensor] = []) -> List[Array]:
        from tripy.frontend.trace.ops import Storage

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

        # HACK (#155): Remove `get_devices` once executable output tensor location matches Trace IR.
        out_tensor_info = self.get_output_tensor_runtime_info(inputs, output_devices)

        # Allocate output memory and store buffer pointers.
        # mlir-tensorrt requires the output buffer to be of the shape with max bounds.
        outputs = [
            Array(None, shape=from_dims(info.shape), dtype=info.dtype, device=info.device) for info in out_tensor_info
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

        return outputs
