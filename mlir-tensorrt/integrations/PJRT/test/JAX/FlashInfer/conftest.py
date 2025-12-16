"""Pytest configuration for FlashInfer tests.

This configures FlashInfer-specific requirements.
"""

import pytest


def pytest_configure(config):
    """Register FlashInfer-specific markers."""
    config.addinivalue_line(
        "markers",
        "requires_flashinfer: test requires flashinfer_jit_cache to be available",
    )


def pytest_runtest_setup(item):
    """Check FlashInfer-specific requirements before running each test."""
    if item.get_closest_marker("requires_flashinfer"):
        try:
            import flashinfer_jit_cache
        except ImportError:
            pytest.skip("Test requires flashinfer_jit_cache to be available")
