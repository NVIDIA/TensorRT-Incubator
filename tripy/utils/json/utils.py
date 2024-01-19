import json
import typing
from typing import Any, Union

from tripy import utils
from tripy.utils.json.enc_dec import Decoder, Encoder


def to_json(obj: Any) -> str:
    return json.dumps(obj, cls=Encoder, indent=" " * 4)


def from_json(src: str) -> Any:
    return json.loads(src, object_pairs_hook=Decoder())


# TODO: Add examples here once we're able to serialize something.
def save(obj: Any, dest: Union[str, typing.IO]):
    """
    Saves an object to the specified destination.

    Args:
        obj: The object to save
        dest: A path or file-like object
    """
    utils.save_file(to_json(obj), dest, mode="w")


def load(src: Union[str, typing.IO]) -> Any:
    """
    Loads an object from the specified source.

    Args:
        src: A path or file-like object

    Returns:
        The loaded object
    """
    return from_json(utils.load_file(src, mode="r"))
