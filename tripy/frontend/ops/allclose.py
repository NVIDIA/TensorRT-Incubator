from tripy import export


@export.public_api(document_under="tensor_operations")
def allclose(a: "tripy.Tensor", b: "tripy.Tensor", rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Returns True if the following equation is elementwise True:
    absolute(a - b) <= (atol + rtol * absolute(b))

    Args:
        a: The LHS tensor.
        b: The RHS tensor.
        rtol: The relative tolerance
        atol: The absolute tolerance

    Returns:
        A boolean value

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.Tensor([1e10,1e-7])
        b = tp.Tensor([1e10,1e-7])
        assert tp.allclose(a, b) == True
    """
    from tripy.frontend.trace.ops.unary_elementwise import abs
    from tripy.frontend.trace.ops.reduce import all

    compare = abs(a - b) <= (atol + rtol * abs(b))
    return all(compare).data().data()
