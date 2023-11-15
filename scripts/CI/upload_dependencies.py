import os
import subprocess


def main():
    # upload stablehlo if it is built in this pipeline
    if not os.path.exists("stablehlo"):
        print("Didn't find stablehlo package, skipping upload.")
    else:
        print("Compressing stablehlo package...")
        subprocess.run(["zip -r -q -y stablehlo.zip stablehlo"], shell=True)
        print("Uploading stablehlo package to package registry...")
        subprocess.run(
            [
                'curl --header "JOB-TOKEN: $CI_JOB_TOKEN" --upload-file stablehlo.zip "https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/stablehlo.zip"'
            ],
            shell=True,
        )
    # upload mlir-tensorrt, it should be built if the script is triggered
    if not os.path.exists("mlir-tensorrt"):
        print("mlir-tensorrt artifact is not found, please check the pipeline!")
    else:
        print("Compressing mlir-tensorrt package...")
        subprocess.run(["zip -r -q mlir-tensorrt.zip mlir-tensorrt/build/lib/Integrations"], shell=True)
        print("Uploading mlir-tensorrt package to package registry...")
        subprocess.run(
            [
                'curl --header "JOB-TOKEN: $CI_JOB_TOKEN" --upload-file mlir-tensorrt.zip "https://gitlab-master.nvidia.com/api/v4/projects/104296/packages/generic/dependencies/default/mlir-tensorrt.zip"'
            ],
            shell=True,
        )


if __name__ == "__main__":
    main()
