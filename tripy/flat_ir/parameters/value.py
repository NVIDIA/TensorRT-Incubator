from tripy.flat_ir.flat_ir import FlatIR
from tripy.frontend.parameters import ValueParameters


@FlatIR.str_from_params(ValueParameters)
def to_str(params, input_names, output_names):
    assert not input_names, "ValueParameters should have no inputs!"
    assert len(output_names) == 1, "ValueParameters should have exactly one output!"

    return f"{output_names[0]} : values=({params.values}), shape=(), stride=(), loc=()"


@FlatIR.shape_inference(ValueParameters)
def infer_shapes(params, input_shapes):
    assert not input_shapes, "ValueParameters should have no inputs!"
    return [params.shape()]
