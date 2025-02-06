import os
import glob
import tempfile
import subprocess
import re
import shutil

def construct_test_names(output):
    relevant_output = output.split('\n\n=======================')[0]
    module_match = re.search(r'<Module (.*?)>', relevant_output)
    module_name = module_match.group(1) if module_match else ''
    class_match = re.search(r'<Class (.*?)>', relevant_output)
    class_name = class_match.group(1) if class_match else ''
    function_matches = re.findall(r'<Function (.*?)>', relevant_output)
    
    test_names = [
        f"/tripy/{module_name}::{class_name}::{func}" 
        for func in function_matches
    ]
    return test_names

def run_single_test(test_name):
    with tempfile.TemporaryDirectory() as test_dir:
        # Set environment for this test
        test_env = os.environ.copy()
        test_env['TRIPY_MLIR_DEBUG_ENABLED'] = '1'
        test_env['TRIPY_MLIR_DEBUG_PATH'] = test_dir
        
        # Run the test
        cmd = ['pytest', test_name, '-v']
        result = subprocess.run(cmd, env=test_env, capture_output=True, text=True)
        
        # Look for 0_ file and get content
        mlir_files = glob.glob(os.path.join(test_dir, '**', '0_*.mlir'), recursive=True)
        content = ""
        
        if mlir_files:
            with open(mlir_files[0], 'r') as source:
                content = source.read()
                content = re.sub(r'^// -----.*$', '', content, flags=re.MULTILINE)
                content = content.strip()
        
        return result.returncode == 0, content

def collect_and_format_tests(test_pattern='/tripy/tests/integration/*'):
    # Collect test files
    test_files = glob.glob(test_pattern)
    test_assemblies = {}
    
    for test_file in test_files:
        print(f"Collecting tests from {test_file}")
        cmd = ['pytest', '--collect-only', '--no-header', '--no-summary', test_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        test_names = construct_test_names(result.stdout)
        
        # Run tests and collect MLIR
        print("\nRunning tests and collecting MLIR:")
        for test_name in test_names:
            print(f"Running {test_name}")
            success, content = run_single_test(test_name)
            status = "PASSED" if success else "FAILED"
            print(f"{status} {test_name}")
            
            if content:
                test_key = f"test_{test_name.split('::')[-1]}"
                test_assemblies[test_key] = f"""\n{content}\n"""
    
    # Write collected assemblies to Python file
    with open('test_tripy_integration.py', 'w') as f:
        f.write("""# RUN: %PYTHON %s | FileCheck %s
import os
import mlir_tensorrt.compiler.api as api
from mlir_tensorrt.compiler.ir import *

# Store assemblies in a dictionary with test names as keys
TEST_ASSEMBLIES = {
""")
        
        for test_name, asm in test_assemblies.items():
            f.write(f'    "{test_name}": """{asm}""",\n')
        
        f.write("""}

def compile_asm(asm_str):
    with Context() as context:
        m = Module.parse(asm_str)
        client = api.CompilerClient(context)
        opts = api.StableHLOToExecutableOptions(
            client,
            [
                "--tensorrt-builder-opt-level=3",
                "--tensorrt-strongly-typed=true",
            ],
        )
        api.compiler_stablehlo_to_executable(client, m.operation, opts)

def run_tests():
    print(f"Running Tripy regression tests")
    for test_name, asm in TEST_ASSEMBLIES.items():
        try:
            compile_asm(asm)
            print(f"PASSED: {test_name}")
        except Exception as e:
            print(f"FAILED: {test_name}: {str(e)}")

if __name__ == "__main__":
    run_tests()
""")
    
        f.write(f"\n\n# CHECK-LABEL: Running Tripy regression tests\n")
        for test_name, _ in test_assemblies.items():
            f.write(f"#    CHECK: PASSED: {test_name}\n")

    return test_assemblies


def main():
    # Initialize integration file
    tests = collect_and_format_tests()
    print(f"\nTotal tests collected: {len(tests)}")

if __name__ == "__main__":
    main()
