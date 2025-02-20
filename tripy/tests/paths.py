import glob
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def get_files_with_extension(ext):
    return [
        path
        for path in glob.glob(os.path.join(ROOT_DIR, "**", f"*{ext}"), recursive=True)
        if not path.startswith(
            (
                os.path.join(ROOT_DIR, "build"),
                os.path.join(ROOT_DIR, "mlir-tensorrt"),
                os.path.join(ROOT_DIR, "stablehlo"),
            )
        )
    ]


MARKDOWN_FILES = get_files_with_extension(".md")

PYTHON_FILES = get_files_with_extension(".py")
