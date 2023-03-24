"""test pycontrails init."""

from __future__ import annotations

import pytest

import pycontrails


def test_metadata() -> None:
    """Test package data loading."""

    assert pycontrails.__metadata__
    assert pycontrails.__license__ == "Apache-2.0"
    assert pycontrails.__url__
