"""
This file registers JSON serialization methods for foreign types.
For Tripy types, serialization methods are defined alongside the type definition.
"""
import base64
import io

import numpy as np

from tripy.utils.json.enc_dec import Decoder, Encoder


@Encoder.register(np.ndarray)
def encode(array):
    outfile = io.BytesIO()
    np.save(outfile, array, allow_pickle=False)
    outfile.seek(0)
    data = base64.b64encode(outfile.read()).decode()
    return {"array": data}


@Decoder.register(np.ndarray)
def decode(dct):
    data = base64.b64decode(dct["array"].encode(), validate=True)
    infile = io.BytesIO(data)
    return np.load(infile, allow_pickle=False)
