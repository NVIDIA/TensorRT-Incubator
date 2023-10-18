from tripy.util.logging import G_LOGGER
import time


def default(value, default):
    """
    Returns a specified default value if the provided value is None.

    Args:
        value : The value.
        default : The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


def log_time(func):
    """
    Provides a wrapper for any arbitrary function to measure and log time to execute this function.
    """

    def wrapper(*args, **kwargs):
        # Get textual representation of args/kwargs.
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        start_time = time.time()
        result = func(*args, **kwargs)
        G_LOGGER.timing(f"{func.__name__}({signature}) executed in {time.time() - start_time:.4f} seconds")
        return result

    return wrapper
