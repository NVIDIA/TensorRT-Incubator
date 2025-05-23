# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy import export
from nvtripy.common.device import device as tp_device
from nvtripy.common.exception import raise_error

# Global variable to store instances
_default_stream_instances = {}


# Stream.synchronize is important for performance, so we would prefer to avoid overhead.
@export.public_api(document_under="compiling_code", bypass_dispatch=["synchronize"])
class Stream:
    """
    Represents a CUDA stream that can be used to manage concurrent operations.

    .. note:: Streams can only be used with compiled functions.

    This class is a wrapper around the underlying stream object, allowing management of CUDA streams.
    """

    def __init__(self) -> None:
        """
        .. code-block:: python
            :linenos:
            :caption: Creating New Streams

            stream_a = tp.Stream()
            stream_b = tp.Stream()

            assert stream_a != stream_b

        .. code-block:: python
            :linenos:
            :caption: Using Streams With Compiled Functions

            # doc: no-print-locals func

            func = tp.compile(tp.relu, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

            # Run the compiled linear function on a custom stream:
            stream = tp.Stream()
            func.stream = stream

            input = tp.ones((2, 2), dtype=tp.float32).eval()
            output = func(input)

            assert tp.equal(output, tp.relu(input))
        """
        from nvtripy.backend.mlir.utils import MLIRRuntimeClient

        self._active_cuda_stream = MLIRRuntimeClient().create_stream()

    def synchronize(self) -> None:
        """
        Synchronize the stream, blocking until all operations in this stream are complete.

        .. code-block:: python
            :linenos:
            :caption: Using Synchronize For Benchmarking

            # doc: no-print-locals
            import time

            func = tp.compile(tp.relu, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

            input = tp.ones((2, 2), dtype=tp.float32).eval()

            func.stream = tp.Stream()

            num_iters = 10
            start_time = time.perf_counter()
            for _ in range(num_iters):
                _ = func(input)
            func.stream.synchronize()
            end_time = time.perf_counter()

            time = (end_time - start_time) / num_iters
            print(f"Execution took {time * 1000} ms")
        """
        self._active_cuda_stream.sync()

    def __eq__(self, other: "nvtripy.Stream"):
        if not isinstance(other, Stream):
            return False
        return self._active_cuda_stream == other._active_cuda_stream

    def __str__(self):
        return f"<Stream(id={id(self)})>"

    def __hash__(self):
        return hash(id(self))

    @property
    def ptr(self) -> int:
        """
        Returns a pointer to the underlying CUDA stream.

        Returns:
            A pointer to the underlying CUDA stream.

        .. code-block:: python
            :linenos:
            :caption: Retrieving The Default Stream

            stream = tp.Stream()
            stream_ptr = stream.ptr
        """
        return self._active_cuda_stream.ptr


@export.public_api(document_under="compiling_code/stream.rst")
def default_stream(device: tp_device = tp_device("gpu")) -> Stream:
    """
    Provides access to the default Tripy CUDA stream for a given device.
    There is only one default stream instance per device.

    Args:
        device: The device for which to get the default stream.

    Returns:
        The default stream for the specified device.

    Raises:
        :class:`nvtripy.TripyException`: If the device is not of type 'gpu' or if the device index is not 0.

    .. note:: Calling :func:`default_stream` with the same device always
        returns the same :class:`Stream` instance for that device.

    .. code-block:: python
        :linenos:
        :caption: Retrieving The Default Stream

        # Get the default stream for the current device.
        default = tp.default_stream()
    """
    global _default_stream_instances

    if device.kind != "gpu":
        raise_error(f"default_stream creation requires device to be of type gpu, got device={device}.")

    if device.index != 0:
        raise_error(f"Tripy stream only works with device index 0, got device={device}")

    if device.index not in _default_stream_instances:
        _default_stream_instances[device.index] = Stream()

    return _default_stream_instances[device.index]
