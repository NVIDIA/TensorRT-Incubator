from dataclasses import dataclass
from typing import Sequence, Set

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy import utils
from tripy.common.array import Array
from tripy.flat_ir.ops.base import BaseFlatIROp

import mlir_tensorrt.runtime.api as runtime


@dataclass(repr=False)
class ConstantOp(BaseFlatIROp):

    data: Array

    def str_skip_fields(self) -> Set[str]:
        if utils.should_omit_constant_in_str(self.data.shape):
            return {"data"}
        return set()

    def to_mlir(self, operands):
        import array
        import tripy.common.datatype as datatype
        from tripy.backend.mlir import utils as mlir_utils

        # TODO(#189): Remove explicit copy to host for constants
        assert isinstance(self.data, Array)
        memref_value = self.data.memref_value
        if self.data.device.kind == "gpu":
            memref_value = runtime.RuntimeClient().copy_to_host(
                device_memref=memref_value,
                stream=None,
            )

        # Workaround (#208): bools are represented as i1 in MLIR-TRT but they cannot be used for DenseElementsAttr
        # so we have to represent them as ints and then cast the result
        if self.outputs[0].dtype == datatype.bool:
            # need to use memoryview.cast to ensure that the view will be flattened
            int_memref = self.data.runtime_client.create_memref(
                array.array("i", memoryview(memref_value).cast("b").tolist()),
                shape=self.data.shape,
                dtype=mlir_utils.convert_tripy_dtype_to_runtime_dtype(datatype.int32),
                device=None,
            )
            attr = ir.DenseElementsAttr.get(
                array=int_memref, type=mlir_utils.get_mlir_dtype(datatype.int32), shape=self.data.shape
            )
            cast_output = mlir_utils.make_mlir_tensor(utils.to_dims(self.data.shape), datatype.bool)
            constant_op = stablehlo.ConstantOp(attr)
            return [stablehlo.ConvertOp(result=cast_output, operand=constant_op)]

        attr = ir.DenseElementsAttr.get(
            array=memref_value, type=mlir_utils.get_mlir_dtype(self.outputs[0].dtype), shape=self.data.shape
        )

        return [stablehlo.ConstantOp(attr)]
