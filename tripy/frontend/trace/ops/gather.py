from dataclasses import dataclass

import tripy.frontend.trace.ops.utils as op_utils
from tripy import export, utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Gather(BaseTraceOp):
    axis: int

    # the output is a shape if the value input is a shape
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.infer_from_first_input_only

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank + self.inputs[1].rank - 1

    def infer_dtypes(self):
        from tripy import int32

        if self.inputs[1].dtype != int32:
            utils.raise_error_io_info(
                self,
                "Index tensor for gather operation should be of int32 type.",
                details=[
                    f"Input tensor 'index' for operation: 'gather' must be of int32 type, but 'index' has type: {self.inputs[1].dtype}"
                ],
            )

        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicGatherOp, DynamicSliceOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import int32

        input_shape = op_utils.get_shape_of_tensor(inputs[0])
        zero_1d = op_utils.add_constant_tensor_from_list([0], inputs[0].device)
        one_1d = op_utils.add_constant_tensor_from_list([1], inputs[0].device)
        second_half_start = op_utils.add_constant_tensor_from_list([self.axis + 1], inputs[0].device)

        # Gather slice_sizes is the same as size of input with dim at self.axis replaced with 1.
        # Code below performs input_shape[0:self.axis], 1, input_shape[self.axis + 1 : ]
        size_partial_tensors = []
        if self.axis > 0:
            slice_len = op_utils.add_constant_tensor_from_list([self.axis], inputs[0].device)
            axis_first_half = FlatIRTensor.build(
                shape=utils.to_dims([self.axis]),
                dtype=int32,
                device=inputs[0].device,
                reason_details=["slice the input shape ", input_shape, " to get input_shape[0:self.axis]."],
            )
            DynamicSliceOp.build([input_shape, zero_1d, slice_len, one_1d], [axis_first_half])
            size_partial_tensors.append(axis_first_half)

        size_partial_tensors.append(one_1d)
        slice_len = op_utils.add_constant_tensor_from_list([inputs[0].rank - self.axis], inputs[0].device)
        axis_second_half = FlatIRTensor.build(
            rank=1,
            dtype=int32,
            device=inputs[0].device,
            reason_details=["slice the input shape ", input_shape, " to get input_shape[self.axis + 1 :]."],
        )
        DynamicSliceOp.build([input_shape, second_half_start, slice_len, one_1d], [axis_second_half])
        size_partial_tensors.append(axis_second_half)

        slice_sizes = op_utils.concatenate_tensors(size_partial_tensors, dim=0)

        DynamicGatherOp.build([inputs[0], inputs[1], slice_sizes], outputs, self.axis)


@export.public_api(document_under="tensor_operations")
def gather(input: "tripy.Tensor", dim: int, index: "tripy.Tensor") -> "tripy.Tensor":
    """
    Gather values from the input tensor along the specified axis based on the specified indices.
    This behaves similarly to ``numpy.take()``.

    Args:
        input: The input tensor
        dim: Axis along which data is gathered.
        index: The indices of elements to gather.

    Returns:
        A new tensor of the same data type as the input tensor and same shape along every
        dimension except ``dim``, which will have a size equal to ``len(index)``.

    .. code-block:: python
        :linenos:
        :caption: Example

        data = tp.iota((3, 2, 2))
        indices = tp.Tensor([0, 2], dtype=tp.int32)
        output = tp.gather(data, 0, indices)

        assert np.array_equal(cp.from_dlpack(output).get(), np.take(cp.from_dlpack(data).get(), cp.from_dlpack(indices).get(), axis=0))
    """
    return Gather.build([input, index], dim)
