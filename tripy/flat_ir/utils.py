# Insert a broadcast op into the flat_ir which broadcasts input tensor to output shape.
def insert_broadcast(cls, flat_ir, inp, out_shape):
    from tripy.flat_ir.ops import BroadcastOp
    from tripy.frontend.ops.utils import get_broadcast_in_dim

    output_tensor = flat_ir.add_tensor(shape=out_shape, dtype=inp.dtype, device=inp.device)
    flat_ir.add_op(cls, BroadcastOp, [inp], [output_tensor], broadcast_dim=get_broadcast_in_dim(inp.shape, out_shape))
    return output_tensor
