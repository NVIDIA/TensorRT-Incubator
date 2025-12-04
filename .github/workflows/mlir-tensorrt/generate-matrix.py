#!/usr/bin/env python3

import argparse
import json
import sys

CUDA_VERSIONS_DICT = {
    "nightly": ["12.9", "13.0"],
    "test": ["12.9", "13.0"],
    "release": ["12.9", "13.0"],
}

LATEST_CUDA_VERSION = "13.0"
LATEST_TRT_VERSION = "10.13"

TRT_VERSIONS_DICT = {
    "nightly": ["10.12", "10.13"],
    "test": ["10.13"],
    "release": ["10.12", "10.13"],
}

TRT_VERSIONS_CUDA_MIN_MAX_DICT = {
    "10.12": {"min_cuda_version": "12.9", "max_cuda_version": "12.9"},
    "10.13": {"min_cuda_version": "12.9", "max_cuda_version": "13.0"},
}

CMAKE_PRESET_DICT = {
    "nightly": "github-cicd",
    "test": "github-cicd",
    # release should use the release wheel build preset
    # TODO: add the release wheel build preset
    "release": "github-cicd",
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
        "12.9": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-rockylinux8-0.1",
        "13.0": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-rockylinux8-0.1",
    },
}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


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
    cmake_preset = CMAKE_PRESET_DICT[channel]

    matrix_dict = {"include": []}
    for cuda_version in cuda_versions:
        for trt_version in trt_versions:
            if trt_version not in TRT_VERSIONS_CUDA_MIN_MAX_DICT:
                raise Exception(
                    f"TRT version {trt_version} is not in TRT_VERSIONS_CUDA_MIN_MAX_DICT"
                )
            min_cuda_version = TRT_VERSIONS_CUDA_MIN_MAX_DICT[trt_version][
                "min_cuda_version"
            ]
            max_cuda_version = TRT_VERSIONS_CUDA_MIN_MAX_DICT[trt_version][
                "max_cuda_version"
            ]
            if not (
                _version_tuple(min_cuda_version)
                <= _version_tuple(cuda_version)
                <= _version_tuple(max_cuda_version)
            ):
                continue
            matrix_dict["include"].append(
                {
                    "cuda": cuda_version,
                    "trt": trt_version,
                    "docker_image": docker_images[cuda_version],
                    "cmake_preset": cmake_preset,
                    "latest_cuda": LATEST_CUDA_VERSION,
                    "latest_trt": LATEST_TRT_VERSION,
                }
            )
    print(json.dumps(matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
