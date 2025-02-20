#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from collections import OrderedDict

from nvtripy.common.exception import raise_error

TYPE_MARKER = "tripy_class"


def str_from_type(typ):
    return typ.__name__


class BaseCustomImpl:
    """
    Base class for Tripy's JSON encoder/decoder.
    """

    @classmethod
    def register(cls, typ: type, alias: str = None):
        """
        Decorator that registers JSON encoding/decoding functions for types.

        Args:
            typ: The type to register
            alias:
                    An alias under which to also register the decoder function.
                    This can be used to retain backwards-compatibility when a class
                    name changes.
        """
        # For the documentation that follows, assume we have a class:
        #
        #     class Dummy:
        #         def __init__(self, x):
        #             self.x = x

        # ========
        # Encoders
        # ========
        #
        # Encoder functions should accept instances of the specified type and
        # return dictionaries.
        #
        # For example:
        #
        #     @Encoder.register(Dummy)
        #     def encode(dummy):
        #         return {"x": dummy.x}
        #
        # To use the custom encoder, use the `to_json` helper:
        #
        #     d = Dummy(x=1)
        #     d_json = to_json(d)

        # ========
        # Decoders
        # ========
        #
        # Decoder functions should accept dictionaries, and return instances of the
        # type.
        #
        # For example:
        #
        #     @Decoder.register(Dummy)
        #     def decode(dct):
        #         return Dummy(x=dct["x"])
        #
        # To use the custom decoder, use the `from_json` helper:
        #
        #     from_json(d_json)

        def register_impl(func):
            def add(key, val):
                if key in cls.REGISTERED:
                    raise_error(
                        f"Duplicate serialization function for type: {key}.\n"
                        f"Note: Existing function: {cls.REGISTERED[key]}, New function: {func}"
                    )
                cls.REGISTERED[key] = val

            assert cls in [Encoder, Decoder], f"Cannot register for unrecognized class type: {cls}"
            if cls == Encoder:

                def wrapped(obj):
                    dct = func(obj)
                    dct[TYPE_MARKER] = str_from_type(typ)
                    return dct

                add(typ, wrapped)
                return wrapped
            elif cls == Decoder:

                def wrapped(dct):
                    if TYPE_MARKER in dct:
                        del dct[TYPE_MARKER]

                    return func(dct)

                add(str_from_type(typ), wrapped)
                if alias is not None:
                    add(alias, wrapped)

        return register_impl


class Encoder(BaseCustomImpl, json.JSONEncoder):
    REGISTERED = {}

    def default(self, o):
        if type(o) in self.REGISTERED:
            return self.REGISTERED[type(o)](o)
        return super().default(o)


class Decoder(BaseCustomImpl):
    REGISTERED = {}

    def __call__(self, pairs):
        # The encoder will insert special key-value pairs into dictionaries encoded from
        # custom types. If we find one, then we know to decode using the corresponding custom
        # type function.
        dct = OrderedDict(pairs)

        type_name = dct.get(TYPE_MARKER)
        if type_name is None:
            return dct

        if type_name not in self.REGISTERED:
            raise_error(f"Could not decode serialized type: {type_name}.")
        return self.REGISTERED[type_name](dct)
