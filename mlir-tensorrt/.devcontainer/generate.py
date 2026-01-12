import json
import sys
from pathlib import Path

# Bump the version when the Dockerfile or scripts under build_tools/docker/
# change and a new tag is created and pushed to the registry.
# This ensures that when people rebuild the container,
# they get the latest version.
#
# It's a two step workflow:
# 1. Update the version number in the GitHub workflows script.
# 2. After the new containers are built and tag is created, update this version
# number here, regenerate the devcontainer configurations and commit the changes.
VERSION = "0.1"


def make_customizations():
    return {
        "vscode": {
            "extensions": [
                "llvm-vs-code-extensions.vscode-clangd",
                "llvm-vs-code-extensions.vscode-mlir",
                "eamodio.gitlens",
                "ms-python.black-formatter",
                "ms-python.python",
            ],
            "settings": {
                "[python]": {"editor.defaultFormatter": "ms-python.black-formatter"},
                "mlir.pdll_compilation_databases": ["build/pdll_compile_commands.yml"],
                "mlir.server_path": "build/bin/mlir-tensorrt-lsp-server",
                "files.exclude": {
                    "**/.git": True,
                    "**/.cache": True,
                    "**/.venv*": True,
                },
                "files.watcherExclude": {
                    "**/.git/objects/**": True,
                    "**/.git/subtree-cache/**": True,
                    "**/.private*": True,
                    "**/.venv*/**": True,
                    "**/build/**": True,
                },
                "search.exclude": {
                    "**/.private*": True,
                    "**/.venv*": True,
                    "**/build": True,
                },
                "python.analysis.include": [
                    "integrations/python",
                    "integrations/python/internal",
                ],
                "python.analysis.typeCheckingMode": "basic",
                "python.analysis.extraPaths": [
                    "build/python_packages/mlir_tensorrt_compiler",
                    "build/python_packages/mlir_tensorrt_runtime",
                    "build/python_packages/tools",
                ],
                "python.analysis.exclude": [
                    "**/build/**",
                    "**/.cache.cpm/**",
                    "**/*bazel*/**",
                    "**/build_tools/**",
                    "third_party",
                ],
            },
        }
    }


def make_features(remote_user: str):
    return {
        "ghcr.io/devcontainers/features/common-utils:2": {
            "installZsh": True,
            "installOhMyZsh": True,
            "configureZshAsDefaultShell": False,
            "upgradePackages": False,
            "username": remote_user,
            "userUid": "automatic",
            "userGid": "automatic",
        },
        "ghcr.io/devcontainers/features/git:1": {},
    }


def get_llvm_toolchain_version():
    return 20


def make_build(base_image: str, linux_distro: str, name: str):
    return {
        "name": name,
        "build": {
            "context": "${localWorkspaceFolder}/build_tools/docker",
            "dockerfile": "${localWorkspaceFolder}/build_tools/docker/Dockerfile",
            "args": {
                "BASE_IMAGE": base_image,
                "LINUX_DISTRO": linux_distro,
                "LLVM_VERSION": get_llvm_toolchain_version(),
            },
        },
    }


def make_config(
    name: str, customizations: dict, features: dict, image: dict, remote_user: str
):
    return {
        "name": name,
        **image,
        "remoteUser": remote_user,
        "updateRemoteUserUID": True,
        "runArgs": [
            "--name",
            name + "-${localEnv:USER:" + remote_user + "}-${devcontainerId}",
            "--cap-add=SYS_PTRACE",
            "--security-opt",
            "seccomp=unconfined",
            "--shm-size=1g",
            "--ulimit",
            "memlock=-1",
            "--network=host",
        ],
        "hostRequirements": {"gpu": "optional"},
        "workspaceMount": "source=${localWorkspaceFolder}/..,target=/workspaces/TensorRT-Incubator,type=bind,consistency=cached",
        "workspaceFolder": "/workspaces/TensorRT-Incubator/mlir-tensorrt",
        "customizations": customizations,
        "features": features,
    }


def make_base_image(os: str, cuda_version: str):
    if "ubuntu22.04" in os:
        os = "ubuntu22.04"
        return f"nvcr.io/nvidia/cuda:{cuda_version}-devel-{os}"
    if "ubuntu24.04" in os:
        os = "ubuntu24.04"
        return f"nvcr.io/nvidia/cuda:{cuda_version}-devel-{os}"
    if "rockylinux8" in os:
        os = "rockylinux8"
        return f"nvcr.io/nvidia/cuda:{cuda_version}-devel-{os}"
    if "rockylinux9" in os:
        os = "rockylinux9"
        return f"nvcr.io/nvidia/cuda:{cuda_version}-devel-{os}"

    raise Exception("failed to determine base image name")


def make_devcontainer_name(os: str, cuda_version_short: str):
    return f"cuda{cuda_version_short}-{os}"


def make_prebuilt_image_name(name: str):
    return f"ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:{name}-{VERSION}"


def create_configs(os: str, cuda_version: str):
    cuda_short = ".".join(cuda_version.split(".")[0:-1])
    devcontainer_name = make_devcontainer_name(os, cuda_short)
    prebuilt_image_tag = make_prebuilt_image_name(devcontainer_name)
    customizations = make_customizations()

    # In newer Ubuntu NVIDIA base containers, there is already a non-root user named "ubuntu" added
    # with the default UID of 1000. For Rockylinux9, there is no such user, so we use "nvidia" as the
    # remote user and it will get created by the devcontainer common-utils feature script.
    if "ubuntu" in os:
        remote_user = "ubuntu"
    else:
        remote_user = "nvidia"

    features = make_features(remote_user)
    prebuilt_config = make_config(
        devcontainer_name + "-prebuilt",
        customizations,
        features,
        {
            "image": prebuilt_image_tag,
        },
        remote_user,
    )
    scratch_config = make_config(
        devcontainer_name,
        customizations,
        features,
        make_build(make_base_image(os, cuda_version), os, devcontainer_name),
        remote_user,
    )
    return prebuilt_config, scratch_config


def enumerate_configs():
    for os in ["ubuntu22.04", "ubuntu24.04", "rockylinux8", "rockylinux9"]:
        for cuda_version in ["12.9.1", "13.0.2"]:
            yield create_configs(os, cuda_version)


def main(out_dir: Path):

    def write_config(devcontainer_config: dict):
        dir = out_dir / devcontainer_config["name"]
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "devcontainer.json").write_text(
            json.dumps(devcontainer_config, indent=2)
        )

    for prebuilt_config, scratch_config in enumerate_configs():
        write_config(prebuilt_config)
        write_config(scratch_config)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
