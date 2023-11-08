from dataclasses import dataclass

from tripy.exception import TripyException


@dataclass
class Device:
    # TODO: Improve docstrings here. Unclear what other information we'd want to include.
    """
    Describes a device
    """

    def __init__(self, kind: str, index: int) -> None:
        VALID_KINDS = {"cpu", "gpu"}
        if kind not in VALID_KINDS:
            raise TripyException(f"Unrecognized device kind: {kind}. Choose from: {list(VALID_KINDS)}")

        self.kind = kind
        self.index = index

    kind: str
    index: int


def device(device: str) -> Device:
    """
    Creates a Device object from the given string.

    Args:
        device: A string consisting of the device kind and an optional index.
                The device kind may be one of: ["cpu", "gpu"].
                If the index is provided, it should be separated from the device kind
                by a colon (':').

    Example:
    ::

        import tripy

        device = tripy.device("cpu")
        assert device.kind == "cpu"
        assert device.index == 0


        device = tripy.device("gpu:1")
        assert device.kind == "gpu"
        assert device.index == 1
    """
    kind, _, index = device.partition(":")
    kind = kind.lower()

    if index:
        try:
            index = int(index)
        except ValueError:
            raise TripyException(f"Could not interpret: {index} as an integer")
    else:
        index = 0

    if index < 0:
        raise TripyException(f"Device index must be a non-negative integer, but was: {index}")

    return Device(kind, index)
