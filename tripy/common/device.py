from dataclasses import dataclass

from tripy.common.exception import TripyException


@dataclass
class device:
    # TODO: Improve docstrings here. Unclear what other information we'd want to include.
    """
    Describes a device.
    """

    def __init__(self, device) -> None:
        """
        Args:
            device: A string consisting of the device kind and an optional index.
                    The device kind may be one of: ["cpu", "gpu"].
                    If the index is provided, it should be separated from the device kind
                    by a colon (':'). If the index is not provided, it defaults to 0.

        Example:
        ::

            device = tp.device("cpu")
            assert device.kind == "cpu"
            assert device.index == 0

            device = tp.device("gpu:1")
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

        VALID_KINDS = {"cpu", "gpu"}
        if kind not in VALID_KINDS:
            raise TripyException(f"Unrecognized device kind: {kind}. Choose from: {list(VALID_KINDS)}")

        self.kind = kind
        self.index = index

    kind: str
    index: int
