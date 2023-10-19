from tripy.flat_ir.flat_ir import FlatIR
from tripy.frontend.parameters import BinaryElementwiseParameters


@FlatIR.str_from_params(BinaryElementwiseParameters)
def to_str(params, input_names, output_name):
    assert params.operation == BinaryElementwiseParameters.Operation.SUM, "Only SUM is supported for now!"
    return f"{output_name} = {' + '.join(input_names)}"


@FlatIR.shape_inference(BinaryElementwiseParameters)
def infer_shapes(params, input_shapes):
    assert params.operation == BinaryElementwiseParameters.Operation.SUM, "Only SUM is supported for now!"
    return input_shapes[0]
