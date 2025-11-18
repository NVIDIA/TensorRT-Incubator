#!/usr/bin/env python3

import argparse
import json
import sys

# please update the cuda version/python version/tensorRT version you want to test
# TODO: add cu130 support
# # build and upload the ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu-llvm17
CUDA_VERSIONS_DICT = {
    "nightly": ["cu129"],
    "test": ["cu129"],
    "release": ["cu129"],
}

TRT_VERSIONS_DICT = {
    "nightly": ["10.12", "10.13"],
    "test": ["10.12"],
    "release": ["10.12", "10.13"],
}

DOCKER_IMAGE_DICT = {
    "nightly": {
        "cu129": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17",
    },
    "test": {
        "cu129": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17",
    },
    "release": {
        "cu129": "ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17",
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

    matrix_dict = {"include": []}
    for cuda_version in cuda_versions:
        for trt_version in trt_versions:
            matrix_dict["include"].append(
                {
                    "cuda": cuda_version,
                    "trt": trt_version,
                    "docker_image": docker_images[cuda_version],
                }
            )
    print(json.dumps(matrix_dict, indent=4))


if __name__ == "__main__":
    main(sys.argv[1:])
