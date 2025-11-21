"""Test coroutine utilities."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from pycontrails.utils import coroutines


@pytest.mark.parametrize("value", [1, "foo", None])
def test_run(value: Any) -> None:
    """Test sychronous running of coroutine."""

    async def _coro(value: Any) -> Any:
        return value

    assert coroutines.run(_coro(value)) == value


@pytest.mark.parametrize("items", [[], [1, 2], ["a", "b", "c"], [None, 2, "four", "six"]])
def test_materialize(items: list[Any]) -> None:
    """Test synchronous materialization from async iterator."""

    async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
        for item in items:
            yield item

    assert coroutines.materialize(_aiter(items)) == items
