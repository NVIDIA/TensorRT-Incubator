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


def get_uniform_devices() -> List[int]:
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
        return []
    sm_versions = np.asarray(sm_versions)
    max_version = sm_versions.max()
    if not np.all(sm_versions == max_version):
        return np.flatnonzero(sm_versions == max_version).tolist()
    return list(x for x in range(deviceCount))


def get_sm_version() -> Tuple[int, int]:
    """Returns the largest/latest SM version among all devices on the host."""
    deviceCount = nvmlDeviceGetCount()
    version = (0, 0)
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        cc = nvmlDeviceGetCudaComputeCapability(handle)
        cc = (cc[0], cc[1])
        if cc > version:
            version = cc
    return version


@contextmanager
def nvml_context(*args, **kwargs):
    """A context manager that handles NVML init and shutdown. Yields the
    uniform devices list when entered into.
    """
    nvmlInit()
    try:
        devices = get_uniform_devices()
        yield devices
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
        avail_mem_gb, _, _ = get_stats(devices)
        avail_mem_gb = np.asarray(avail_mem_gb)

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
    return get_sm_version() >= (8, 9)


def get_num_cuda_devices() -> int:
    try:
        with nvml_context() as devices:
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

    with nvml_context() as devices:
        if len(devices) == 0:
            return
        print(select_device(devices))
    return


@cli.command("get-parallelism")
@click.option(
    "--required-mem", help="required GPU memory in GB", default=1.0, type=click.FLOAT
)
def get_parallelism(required_mem: float):
    with nvml_context() as devices:
        print(estimate_parallelism_from_memory(devices, required_mem))


if __name__ == "__main__":
    cli()
