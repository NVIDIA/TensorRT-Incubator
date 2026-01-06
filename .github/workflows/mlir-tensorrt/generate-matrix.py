#!/usr/bin/env python3

import argparse
import json
import sys

CUDA_TRT_VERSIONS_DICT = {
    "nightly": [
        {
            "cuda": "12.9",
            "trt": "10.12",
        },
        {
            "cuda": "13.0",
            "trt": "10.13",
        },
    ],
    "test": [
        {
            "cuda": "13.0",
            "trt": "10.13",
        },
    ],
    "release": [
        {
            "cuda": "12.9",
            "trt": "10.12",
        },
        {
            "cuda": "13.0",
            "trt": "10.13",
        },
    ],
    "pypi-release": [
        {
            "cuda": "13.0",
            "trt": "10.13",
        },
    ],
}

ARCH_LIST_DICT = {
    "test": ["x86_64"],
    "release": ["x86_64", "aarch64"],
    "nightly": ["x86_64", "aarch64"],
    "pypi-release": ["x86_64", "aarch64"],
}

GH_RUNNER_DICT = {
    "x86_64": "linux-amd64-gpu-h100-latest-1",
    "aarch64": "linux-arm64-gpu-l4-latest-1",
}

CMAKE_PRESET_DICT = {
    "nightly": "github-cicd",
    "test": "github-cicd",
    "release": "distribution-wheels",
    "pypi-release": "distribution-wheels",
}

DOCKER_IMAGE_DICT = {
    "nightly": {
        "12.9": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu24.04-0.1",
        "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu24.04-0.1",
    },
    "test": {
        "12.9": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu24.04-0.1",
        "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu24.04-0.1",
    },
    "release": {
        "aarch64": {
            "12.9": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu22.04-0.1",
            "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu22.04-0.1",
        },
        "x86_64": {
            "12.9": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-rockylinux9-0.1",
            "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-rockylinux9-0.1",
        },
    },
    "pypi-release": {
        "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-rockylinux8-0.1",
    },
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
    if options.channel not in ("nightly", "test", "release", "pypi-release"):
        raise Exception(
            "--channel is invalid, please choose from nightly, test, release or pypi-release"
        )

    channel = options.channel
    cuda_trt_versions = CUDA_TRT_VERSIONS_DICT[channel]
    docker_images = DOCKER_IMAGE_DICT[channel]
    cmake_preset = CMAKE_PRESET_DICT[channel]
    arch_list = ARCH_LIST_DICT[channel]
    matrix_dict = {"include": []}
    for arch in arch_list:
        gh_runner = GH_RUNNER_DICT[arch]
        for cuda_trt_version in cuda_trt_versions:
            cuda_version = cuda_trt_version["cuda"]
            trt_version = cuda_trt_version["trt"]
            if channel == "release":
                # release wheel build for aarch64 and x86_64 uses different docker images
                docker_image = docker_images[arch][cuda_version]
            else:
                docker_image = docker_images[cuda_version]
            matrix_dict["include"].append(
                {
                    "cuda": cuda_version,
                    "trt": trt_version,
                    "docker_image": docker_image,
                    "cmake_preset": cmake_preset,
                    "arch": arch,
                    "github_runner": gh_runner,
                }
            )

    sys.stdout.write(json.dumps(matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
