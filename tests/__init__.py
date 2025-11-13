"""Test package initialization."""

import pytest


def test_package_import():
    """Test that package can be imported."""
    import bookdatamaker
    assert bookdatamaker.__version__ == "0.1.0"
