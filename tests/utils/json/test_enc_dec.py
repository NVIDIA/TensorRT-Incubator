import pytest

from tripy.common.exception import TripyException
from tripy.utils.json import Decoder, Encoder, from_json, to_json
from tripy.utils.json.enc_dec import TYPE_MARKER


class Dummy:
    def __init__(self, x):
        self.x = x


@Encoder.register(Dummy)
def encode_dummy(dummy):
    return {"x": dummy.x}


@Decoder.register(Dummy)
def decode_dummy(dct):
    assert len(dct) == 1  # Custom type markers should be removed at this point
    return Dummy(x=dct["x"])


class NoDecoder:
    def __init__(self, x):
        self.x = x


@Encoder.register(NoDecoder)
def encode_nodecoder(no_decoder):
    return {"x": no_decoder.x}


class TestEncoder:
    def test_registered(self):
        d = Dummy(x=-1)
        d_json = to_json(d)
        assert encode_dummy(d) == {"x": d.x, TYPE_MARKER: "Dummy"}
        expected = f'{{\n    "x": {d.x},\n    "{TYPE_MARKER}": "Dummy"\n}}'
        assert d_json == expected


class TestDecoder:
    def test_object_pairs_hook(self):
        d = Dummy(x=-1)
        d_json = to_json(d)

        new_d = from_json(d_json)
        assert new_d.x == d.x

    def test_error_on_no_decoder(self):
        d = NoDecoder(x=1)
        d_json = to_json(d)

        with pytest.raises(TripyException, match="Could not decode serialized type: NoDecoder."):
            from_json(d_json)

    def test_names_correct(self):
        # If the name of a class changes, then we need to specify an `alias` when registering
        # to retain backwards compatibility.
        assert set(Decoder.REGISTERED.keys()) == {
            "CachedExecutable",
            "device",
            "Dim",
            "dtype",
            "Dummy",
            "ndarray",
            "TensorInfo",
        }
