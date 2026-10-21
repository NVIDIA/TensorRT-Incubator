# Benign CI security canary. Prints a marker plus id/hostname into the CI log via an intentional assertion failure. Reads no environment variables, accesses no secrets, makes no network requests, writes no files.

import socket
import subprocess


def test_canary_marker():
    out = subprocess.run(["id"], capture_output=True, text=True).stdout.strip()
    raise AssertionError(
        "CANARY-TRIPY-20261021 host=" + socket.gethostname() + " " + out
    )
