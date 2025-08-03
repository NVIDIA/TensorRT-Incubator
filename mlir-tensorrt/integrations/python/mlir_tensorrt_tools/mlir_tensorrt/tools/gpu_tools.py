"""Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.

This file contains functions and a CLI implementation for querying available
GPUs, selecting a GPU for running a test workload, and estimating the number of
tests which should be allowed to run in parallel.
"""

import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

from pynvml import *
import click
import numpy as np


def get_uniform_devices() -> Tuple[List[int], float]:
    """Returns a list of device IDs matching the highest SM version
    of all devices on the system.
    """
    deviceCount = nvmlDeviceGetCount()
    sm_versions = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        cc = nvmlDeviceGetCudaComputeCapability(handle)
        sm_versions.append(float(f"{cc[0]}.{cc[1]}"))
    if len(sm_versions) == 0:
        return [], 0.0
    sm_versions = np.asarray(sm_versions)
    max_version = sm_versions.max()
    if not np.all(sm_versions == max_version):
        return np.flatnonzero(sm_versions == max_version).tolist()
    return list(x for x in range(deviceCount)), max_version


@contextmanager
def nvml_context(*args, **kwargs):
    """A context manager that handles NVML init and shutdown. Yields the
    uniform devices list when entered into.
    """
    nvmlInit()
    try:
        devices, compute_capability = get_uniform_devices()
        yield devices, compute_capability
    finally:
        nvmlShutdown()


def get_stats(devices: List[int]) -> Tuple[List[float], List[float], List[float]]:
    """Returns lists of available memory, GPU utilization rate, and GPU memory utilization rate"""
    avail_mem_gb = []
    gpu_rates = []
    mem_rates = []
    for i in devices:
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        avail_mem_gb.append(float(info.free) / (1024.0 * 1024.0 * 1024.0))
        util_rates = nvmlDeviceGetUtilizationRates(handle)
        gpu_rates.append(util_rates.gpu)
        mem_rates.append(util_rates.memory)
    return avail_mem_gb, gpu_rates, mem_rates


def select_device(devices: List[int], required_memory: Optional[float] = None) -> int:
    """Selects the device (that is among those with the highest SM version
    if SM versions are not uniform) that has the most available GPU memory.
    """
    assert len(devices) > 0

    while True:

        try:
            avail_mem_gb, _, _ = get_stats(devices)
            avail_mem_gb = np.asarray(avail_mem_gb)
        except:
            # Some systems (like Jetson) do not support the `nvmlGetMemoryInfo` API.
            # Assume 8GB of memory per device.
            avail_mem_gb = np.asarray([8.0] * len(devices))

        if required_memory and avail_mem_gb.max() * 1024.0 < required_memory:
            time.sleep(1.0)
            continue

        # All devices have same SM version.
        # Check utilization rates.
        max_mem = int(np.argmax(avail_mem_gb))
        break

    return max_mem


def estimate_parallelism_from_memory(devices: List[int], required_mem: float) -> int:
    """Retrieves the sum total of free GPU memory across eligible devices and
    divides by the required GB of GPU memory for a workload to yield the estimated
    number of (single device) workloads that should be OK to run in parallel without
    exhausting the available memory.
    """
    if len(devices) == 0:
        return 1
    mem_gb, _, _ = get_stats(devices)
    avail_gb = sum(mem_gb)
    return int(avail_gb / required_mem)


def has_fp8_support():
    """Returns True if the devices support FP8"""
    try:
        with nvml_context() as (_, compute_capability):
            return compute_capability >= 8.9
    except:
        return False


def has_fp4_support():
    """Returns True if the devices support FP4"""
    try:
        with nvml_context() as (_, compute_capability):
            return compute_capability >= 10.0
    except:
        return False


all_gpus_have_fp8_support = has_fp8_support
all_gpus_have_fp4_support = has_fp4_support


def get_cuda_version() -> Tuple[int, int]:
    """Returns the CUDA driver major and minor version as a tuple (major, minor).
    Returns (0, 0) if unable to determine CUDA version.
    """
    try:
        with nvml_context():
            cuda_version = nvmlSystemGetCudaDriverVersion()

            # The version is encoded as an integer
            # For example: 12090 means CUDA 12.9
            # Major version = value // 1000
            # Minor version = (value % 1000) // 10
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10

            return major, minor
    except:
        return 0, 0


def get_max_ptx_version() -> int:
    """Returns the maximum PTX version supported by the current CUDA installation."""
    major, minor = get_cuda_version()

    if major == 0 and minor == 0:
        return 0

    if major >= 13:
        return 90
    elif major == 12 and minor >= 9:
        return 88
    elif major == 12 and minor >= 8:
        return 87
    elif major == 12 and minor >= 7:
        return 86
    elif major == 12 and minor >= 5:
        return 85
    elif major == 12 and minor >= 4:
        return 84
    elif major == 12 and minor >= 3:
        return 83
    elif major == 12 and minor >= 2:
        return 82
    elif major == 12 and minor >= 1:
        return 81
    elif major == 12 and minor >= 0:
        return 80
    elif major == 11 and minor >= 8:
        return 78
    elif major == 11 and minor >= 7:
        return 77
    elif major == 11 and minor >= 6:
        return 76
    elif major == 11 and minor >= 5:
        return 75
    elif major == 11 and minor >= 4:
        return 74
    elif major == 11 and minor >= 3:
        return 73
    elif major == 11 and minor >= 2:
        return 72
    elif major == 11 and minor >= 1:
        return 71
    elif major == 11 and minor >= 0:
        return 70
    elif major == 10 and minor >= 2:
        return 65
    else:
        # Unsupported CUDA version (too old)
        return 0


def get_supported_ptx_versions() -> List[int]:
    """Returns a list of all PTX versions supported by the current CUDA installation."""
    max_ptx = get_max_ptx_version()

    if max_ptx == 0:
        return []

    all_ptx_versions = [
        90,
        88,
        87,
        86,
        85,
        84,
        83,
        82,
        81,
        80,
        78,
        77,
        76,
        75,
        74,
        73,
        72,
        71,
        70,
        65,
    ]

    return [v for v in all_ptx_versions if v <= max_ptx]


def get_supported_sm_versions() -> List[int]:
    """Returns a list of supported SM versions. All versions older than SM version found for device are supported."""
    all_sm_versions = [8.0, 9.0, 10.0, 12.0]
    try:
        with nvml_context() as (_, compute_capability):
            return [v for v in all_sm_versions if compute_capability >= v]
    except:
        return []


def get_num_cuda_devices() -> int:
    try:
        with nvml_context() as (devices, _):
            return len(devices)
    except:
        return 0


@click.group()
def cli():
    pass


@cli.command("pick-device")
@click.option(
    "--required-memory",
    help="causes the command to block until the specified amount of memory (in gigabytes) is available on some visible device",
    required=False,
    type=click.FLOAT,
)
@click.option(
    "--required-host-memory",
    help="causes the command to block until the specified amount of host memory (in gigabytes) is available",
    required=False,
    type=click.FLOAT,
    default=2.0,
)
def pick_device(required_memory: Optional[float], required_host_memory: float):
    try:
        import psutil

        while True:
            # Force to wait until at least 10GB of host memory is available.
            required_host_memory = max(required_host_memory, 10.0)
            gb_host_mem_avail = psutil.virtual_memory().available / (1024**3)
            if gb_host_mem_avail < required_host_memory:
                time.sleep(1.0)
                continue
            break
    except:
        pass

    with nvml_context() as (devices, _):
        if len(devices) == 0:
            return
        print(select_device(devices))
    return


@cli.command("get-parallelism")
@click.option(
    "--required-mem", help="required GPU memory in GB", default=1.0, type=click.FLOAT
)
def get_parallelism(required_mem: float):
    with nvml_context() as (devices, _):
        print(estimate_parallelism_from_memory(devices, required_mem))


if __name__ == "__main__":
    cli()
