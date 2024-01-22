# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
def insert_broadcast(cls, flat_ir, input_tensor, out_shape):
    from tripy.flat_ir.ops import BroadcastOp
    from tripy.frontend.ops.utils import get_broadcast_in_dim

    output_tensor = flat_ir.add_tensor(shape=out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    flat_ir.add_op(
        cls,
        BroadcastOp,
        [input_tensor],
        [output_tensor],
        broadcast_dim=get_broadcast_in_dim(input_tensor.shape, out_shape),
    )
    return output_tensor


def insert_reshape(cls, flat_ir, input_tensor, out_shape):
    from tripy.flat_ir.ops import ReshapeOp

    output_tensor = flat_ir.add_tensor(shape=out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    flat_ir.add_op(cls, ReshapeOp, [input_tensor], [output_tensor])
    return output_tensor
