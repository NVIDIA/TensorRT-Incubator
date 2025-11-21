#!/usr/bin/env python3

import argparse
import json
import sys

CUDA_VERSIONS_DICT = {
    "nightly": ["12.9.1", "13.0.0"],
    "test": ["12.9.1"],
    "release": ["12.9.1", "13.0.0"],
}

TRT_VERSIONS_DICT = {
    "nightly": ["10.12", "10.13"],
    "test": ["10.12"],
    "release": ["10.12", "10.13"],
}

DOCKER_IMAGE_DICT = {
    "nightly": {
        "12.9.1": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17",
        "13.0.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu",
    },
    "test": {
        "12.9.1": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17",
        "13.0.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu",
    },
    "release": {
        "12.9.1": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-rocky-gcc11",
        "13.0.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-rocky",
    },
}

GPU_RUNNER_DICT = {
    "aarch64": "linux-arm64-gpu-l4-latest-1",
    "x86_64": "linux-amd64-gpu-h100-latest-1",
}


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        help="channel",
        type=str,
        default="test",
    )

    options = parser.parse_args(args)
    if options.channel not in ("nightly", "test", "release"):
        raise Exception(
            "--channel is invalid, please choose from nightly, test or release"
        )
    channel = options.channel

    cuda_versions = CUDA_VERSIONS_DICT[channel]
    trt_versions = TRT_VERSIONS_DICT[channel]
    docker_images = DOCKER_IMAGE_DICT[channel]

    matrix_dict = {"include": []}
    for cuda_version in cuda_versions:
        for trt_version in trt_versions:
            for arch, gpu_runner in GPU_RUNNER_DICT.items():
                matrix_dict["include"].append(
                    {
                        "cuda": cuda_version,
                        "trt": trt_version,
                        "docker_image": docker_images[cuda_version],
                        "arch": arch,
                        "runner": gpu_runner,
                    }
                )
    print(json.dumps(matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
