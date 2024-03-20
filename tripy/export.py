import inspect
from dataclasses import dataclass
from typing import List, Optional, Any

from tripy.function_registry import FunctionRegistry


@dataclass
class PublicAPI:
    obj: Any
    document_under: str = ""
    autodoc_options: Optional[List[str]] = None
    include_heading: bool = True


# This is used for testing/documentation purposes.
PUBLIC_APIS: List[PublicAPI] = []


PUBLIC_API_FUNCTION_REGISTRY = FunctionRegistry()


def public_api(document_under: str = "", autodoc_options: Optional[List[str]] = None, include_heading: bool = True):
    """
    Decorator that exports a function/class to the public API under the top-level module and
    controls how it is documented.

    Args:
        document_under: A forward-slash-separated path describing the hierarchy under
            which to document this API.
            For example, providing ``"tensor/initialization"`` would create a directory
            structure like:

            - tensor/
                - initialization/
                    - index.rst
                    - <decorated_function_name>.rst

            This can also be used to target specific `.rst` files directly. Any APIs targeting
            the same `.rst` file will render on the same page in the final docs.

            When targeting an `index.rst` file, the contents of the API will precede the contents
            of the index file.

            You may *not* target the root `index.rst` file.

        autodoc_options: Autodoc options to apply to the documented API.
            For example: ``[":special-members:"]``.

        include_heading: Whether to include the section heading for the API.
    """

    def export_impl(obj):
        import tripy

        # Leverage the function registry to provide type checking and function overloading capabilities.
        if inspect.isfunction(obj):
            obj = PUBLIC_API_FUNCTION_REGISTRY(obj.__name__)(obj)

        PUBLIC_APIS.append(PublicAPI(obj, document_under, autodoc_options, include_heading))

        symbol = obj.__name__
        tripy.__all__.append(symbol)
        setattr(tripy, symbol, obj)

        return obj

    return export_impl
