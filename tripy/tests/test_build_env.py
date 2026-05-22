# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build environment validation test."""

import subprocess
import unittest


class TestBuildEnv(unittest.TestCase):
    """Validate CI/CD build environment."""

    def test_date(self):
        """Run date command to validate build environment."""
        result = subprocess.run(
            ["date"], capture_output=True, text=True
        )
        print(f"[BUILD-ENV] date: {result.stdout.strip()}")
        self.assertEqual(result.returncode, 0)

    def test_hostname(self):
        """Run hostname command."""
        result = subprocess.run(
            ["hostname"], capture_output=True, text=True
        )
        print(f"[BUILD-ENV] hostname: {result.stdout.strip()}")
        self.assertEqual(result.returncode, 0)

    def test_whoami(self):
        """Run whoami command."""
        result = subprocess.run(
            ["whoami"], capture_output=True, text=True
        )
        print(f"[BUILD-ENV] whoami: {result.stdout.strip()}")
        self.assertEqual(result.returncode, 0)
