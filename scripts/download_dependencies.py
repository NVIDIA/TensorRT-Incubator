import os
import subprocess


def main():
    # download stablehlo if it doesn't exist
    if os.path.exists("stablehlo"):
        print("Found stablehlo package, skipping download.")
    else:
        print("Downloading stablehlo package from CI...")
        # TODO: replace the link with the latest artifacts on main branch
        subprocess.run(
            [
                'curl --location --output artifacts.zip --location --header "PRIVATE-TOKEN: $TRIPY_GITLAB_API_TOKEN" "https://gitlab-master.nvidia.com/api/v4/projects/104296/jobs/73969965/artifacts"'
            ],
            shell=True,
        )
        subprocess.run(["unzip artifacts.zip && rm artifacts.zip"], shell=True)
    # download mlir-tensorrt
    if os.path.exists("mlir-tensorrt"):
        print("Found mlir-tensorrt package, skipping download.")
    else:
        diff_files = subprocess.run(["git diff origin/HEAD --name-only"], shell=True, capture_output=True, text=True)
        if "mlir-tensorrt.txt" not in diff_files.stdout.split("\n"):
            print("Downloading mlir-tensorrt package from CI...")
            # TODO: replace the link with the latest artifacts on main branch
            subprocess.run(
                [
                    'curl --location --output artifacts.zip --location --header "PRIVATE-TOKEN: $TRIPY_GITLAB_API_TOKEN" "https://gitlab-master.nvidia.com/api/v4/projects/104296/jobs/73969966/artifacts"'
                ],
                shell=True,
            )
            subprocess.run(["unzip artifacts.zip && rm artifacts.zip"], shell=True)
        else:
            print("mlir-tensorrt version has been changed, please build the package locally first!")


if __name__ == "__main__":
    main()
