from typing import Any


class FunctionRegistry(dict):
    """
    Maintains a mapping of keys to functions.

    Besides a specialized __call__ function, this behaves exactly like a dictionary.
    """

    def __call__(self, key: Any):
        """
        Registers a function with this function registry.
        This function allows instances of the class to be used as decorators.

        Args:
            key: The key under which to register the function.
        """

        def impl(func):
            self[key] = func
            return func

        return impl
