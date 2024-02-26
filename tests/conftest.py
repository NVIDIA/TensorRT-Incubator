import copy
import glob
import os
import subprocess as sp
from typing import Optional

import pytest

from tests.helper import ROOT_DIR


@pytest.fixture()
def sandboxed_install_run(virtualenv):
    """
    A special fixture that runs commands, but sandboxes any `pip install`s in a virtual environment.
    Packages from the test environment are still usable, but those in the virtual environment take precedence
    """

    VENV_PYTHONPATH = glob.glob(os.path.join(virtualenv.virtualenv, "lib", "python*", "site-packages"))[0]

    class StatusWrapper:
        def __init__(self, stdout=None, stderr=None, success=None) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.success = success

    def run_impl(command: str, cwd: Optional[str] = None):
        env = copy.copy(os.environ)
        # Always prioritize our own copy of Tripy over anything in the venv.
        env["PYTHONPATH"] = (
            ROOT_DIR
            + os.pathsep
            # TODO (#112): Don't add stableHLO path here:
            + "/usr/lib/stablehlo/python-build/tools/stablehlo/python_packages/stablehlo"
            + os.pathsep
            + VENV_PYTHONPATH
        )

        print(f"Running command: {command}")

        status = StatusWrapper()
        if "pip" in command:
            # # Remove whitespace args and escaped newlines
            # command = [arg for arg in str(block).strip().split(" ") if arg.strip() and arg != "\\\n"]

            virtualenv.run(command, cwd=cwd)
            status.success = True
        else:
            sp_status = sp.run(command, cwd=cwd, env=env, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

            def try_decode(inp):
                try:
                    return inp.decode()
                except UnicodeDecodeError:
                    return inp

            status.stdout = try_decode(sp_status.stdout)
            status.stderr = try_decode(sp_status.stderr)
            status.success = sp_status.returncode == 0

        return status

    return run_impl