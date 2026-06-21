#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import platform
import socket

# Stable token to locate this test's output in CI logs.
CI_ENV_MARKER = "tripy-ci-env-check-7f3a9c21"


def test_ci_environment_info(capsys):
    """Log basic runner/environment details to make CI triage easier."""
    info = {
        "marker": CI_ENV_MARKER,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
    }
    for key in ("RUNNER_NAME", "RUNNER_OS", "GITHUB_WORKFLOW", "GITHUB_RUN_ID"):
        info[key.lower()] = os.environ.get(key, "")

    with capsys.disabled():
        print(f"\n[{CI_ENV_MARKER}] runner environment:")
        for k, v in info.items():
            print(f"  {k} = {v}")

    assert info["python"]
    assert info["cpu_count"] and info["cpu_count"] > 0
