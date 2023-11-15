import os
import subprocess


def main():
    # download stablehlo if it doesn't exist
    if os.path.exists("stablehlo"):
        print("Found stablehlo package, skipping download.")
    else:
        print("Downloading stablehlo package from CI...")
        subprocess.run(
            [
                'curl --header "PRIVATE-TOKEN: $TRIPY_GITLAB_API_TOKEN" "https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/stablehlo.zip" --output stablehlo.zip'
            ],
            shell=True,
        )
        subprocess.run(["unzip -q stablehlo.zip && rm stablehlo.zip"], shell=True)
    # download mlir-tensorrt
    if os.path.exists("mlir-tensorrt"):
        print("Found mlir-tensorrt package, skipping download.")
    else:
        diff_files = subprocess.run(["git diff origin/HEAD --name-only"], shell=True, capture_output=True, text=True)
        if "mlir-tensorrt.txt" not in diff_files.stdout.split("\n"):
            print("Downloading mlir-tensorrt package from CI...")
            subprocess.run(
                [
                    'curl --header "PRIVATE-TOKEN: $TRIPY_GITLAB_API_TOKEN" "https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/mlir-tensorrt.zip" --output mlir-tensorrt.zip'
                ],
                shell=True,
            )
            subprocess.run(["unzip -q mlir-tensorrt.zip && rm mlir-tensorrt.zip"], shell=True)
        else:
            print("mlir-tensorrt version has been changed, please build the package locally first!")


if __name__ == "__main__":
    main()
