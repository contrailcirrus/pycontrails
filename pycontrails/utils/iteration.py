"""Utilites for iterating of sequences."""

from __future__ import annotations

from typing import Any, Iterator


def chunk_list(lst: list, n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from list."""

    for i in range(0, len(lst), n):
        yield lst[i : i + n]
