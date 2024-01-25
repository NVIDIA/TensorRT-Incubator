from dataclasses import dataclass

from tripy.common.exception import TripyException


@dataclass
class device:
    # TODO: Improve docstrings here. Unclear what other information we'd want to include.
    """
    Describes a device.
    """

    kind: str
    index: int

    def __init__(self, device) -> None:
        r"""
        Args:
            device: A string consisting of the device kind and an optional index.
                    The device kind may be one of: ["cpu", "gpu"].
                    If the index is provided, it should be separated from the device kind
                    by a colon (':'). If the index is not provided, it defaults to 0.

        Example:

        .. code:: python
            :number-lines:

            cpu = tp.device("cpu")
            print(f"cpu: {cpu}")
            assert cpu.kind == "cpu"
            assert cpu.index == 0

            gpu = tp.device("gpu:1")
            print(f"gpu: {gpu}")
            assert gpu.kind == "gpu"
            assert gpu.index == 1
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

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}"
