from typing import Any, List, Union

from colored import Fore, Style

from tripy.common.exception import raise_error
from tripy.utils.utils import default


# Like raise_error but adds information about the inputs and output.
def raise_error_io_info(
    op: Union["BaseTraceOp", "BaseFlatIROp"], summary: str, details: List[Any] = None, include_inputs: bool = True
) -> None:
    details = default(details, ["This originated from the following expression:"])
    details += [":"] + op.outputs
    if include_inputs:
        for index, inp in enumerate(op.inputs):
            details.extend([f"{Fore.magenta}Input {index} was:{Style.reset}", inp])

    raise_error(summary, details)
