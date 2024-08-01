
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import re

import numpy as np

from tripy.backend.mlir.compiler import Compiler


def preprocess_program(program):
    """
    Preprocesses the input stablehlo program.
    1. Removes prefix numbers.
    2. Replaces dense data sections that are hidden with <...> with random weights in a string.

    Args:
    text: The input string.

    Returns:
    The string with dense data replaced by random weights.
    """

    def extract_shape_and_type(tensor_description):
        """
        Extracts the shape and type from the tensor description string.
        """
        # Ensure the regex correctly captures both dimensions and floating-point precision type
        match = re.search(r"tensor<((\d+x)*\d*)x(f\d+)>", tensor_description)
        if match:
            dimensions = match.group(1)
            precision_type = match.group(3)  # Correctly capture the floating-point precision type
            shape = tuple(map(int, dimensions.split("x"))) if dimensions else ()  # Handle scalar case
            return shape, precision_type
        else:
            return None, None

    def generate_random_dense_data(shape):
        """
        Generates random float32 data for a given shape. Handles scalars and N-dimensional arrays.
        """

        def format_array(array):
            if array.ndim == 1:
                return "[" + ", ".join(f"{num:.6e}" for num in array) + "]"
            else:
                inner = ",\n" + " "
                return "[" + inner.join(format_array(subarray) for subarray in array) + "]"

        if shape:  # For tensors with dimensions
            random_data = np.random.rand(*shape).astype(np.float32)
            formatted_data = format_array(random_data)
        else:  # For scalars
            random_data = np.random.rand().astype(np.float32)
            formatted_data = f"{random_data:.6e}"

        return formatted_data

    def replace_placeholder_with_dense_data(modified_text):
        """
        Replaces the "..." in the modified string with random dense data based on tensor shape.

        Args:
        modified_text: The input string containing the placeholder and tensor shape information.

        Returns:
        The string with the placeholder replaced by generated dense data.
        """
        # Extract the tensor shape from the modified text
        tensor_shape_info = modified_text.split(" : ")[-1]
        shape, precision = extract_shape_and_type(tensor_shape_info)
        assert precision == "f32", f"Only f32 precision supported but found {precision}"
        assert shape, f"Shape information missing in {modified_text}"

        # Generate random dense data
        random_dense_data = generate_random_dense_data(shape)

        # Replace the placeholder with generated random dense data
        start_index = modified_text.find("<") + 1
        end_index = modified_text.find(">") - 1
        return modified_text[:start_index] + random_dense_data + modified_text[end_index + 1 :]

    # Remove prefix numbers
    program = re.sub(r"^\d+:\s+", "", program, flags=re.MULTILINE)

    # Split in lines and if line contains dense, replace ... with real random weights.
    lines = program.split("\n")
    replaced = [
        replace_placeholder_with_dense_data(line) if "stablehlo.constant dense<...>" in line else line for line in lines
    ]
    return "\n".join(replaced)


def read_program_from_file(file_path):
    """Read program from file and return as string."""
    with open(file_path, "r") as file:
        return file.read()


def compile_code(code):
    compiler = Compiler()
    compiler.compile_stabehlo_program(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs compilation of a given Stablehlo program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adding the arguments
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--program", type=str, help="The input stablehlo program. Note that the program can have prefix line numbers."
    )
    group.add_argument("--file", type=str, help="Path to the file containing the program.")

    # Parse the arguments
    args = parser.parse_args()

    # Process the input
    program_string = args.program if args.program else read_program_from_file(args.file)
    cleaned_example = preprocess_program(program_string)
    compile_code(cleaned_example)
