#!/usr/bin/env python3

import argparse
import json
import sys

CUDA_VERSIONS_DICT = {
    "nightly": ["12.9", "13.0"],
    "test": ["12.9", "13.0"],
    "release": ["12.9", "13.0"],
}

TRT_VERSIONS_DICT = {
    "nightly": ["10.12", "10.13"],
    "test": ["10.13"],
    "release": ["10.12", "10.13"],
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
            matrix_dict["include"].append(
                {
                    "cuda": cuda_version,
                    "trt": trt_version,
                    "docker_image": docker_images[cuda_version],
                    "cmake_preset": cmake_preset,
                }
            )
    print(json.dumps(matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
