import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--ci", action="store_true", help="Set true if this script is run in CI")


def main(args):
    # download stablehlo if it doesn't exist
    header = '--header "JOB-TOKEN: $CI_JOB_TOKEN"' if args.ci else '--header "PRIVATE-TOKEN: $TRIPY_GITLAB_API_TOKEN"'
    if os.path.exists("stablehlo"):
        print("Found stablehlo package, skipping download.")
    else:
        print("Downloading stablehlo package from CI...")
        url = '"https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/stablehlo.zip"'
        curl_cmd = " ".join(["curl", header, url, "--output stablehlo.zip"])
        subprocess.run(
            [curl_cmd],
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
            url = '"https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/mlir-tensorrt.zip"'
            curl_cmd = " ".join(["curl", header, url, "--output mlir-tensorrt.zip"])
            subprocess.run(
                [curl_cmd],
                shell=True,
            )
            subprocess.run(["unzip -q mlir-tensorrt.zip && rm mlir-tensorrt.zip"], shell=True)
        else:
            print("mlir-tensorrt version has been changed, please build the package locally first!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
