"""Multitasking utilities."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run(task: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously.

    If an event loop is already running in the main thread (e.g., in Jupyter),
    work is offloaded to a separate thread to avoid errors.

    Parameters
    ----------
    task : Coroutine[Any, Any, T]
        Coroutine object, typically created by calling an async function

    Returns
    -------
    T
        Value returned by the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # Must run the coroutine in a separate thread if an
    # event loop is already running in the main thread.
    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, task)
            return future.result()

    return asyncio.run(task)


def run_all(tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
    """Run multiple coroutines sychronously.

    Parameters
    ----------
    tasks : list[Coroutine[Any, Any, T][]
        List of coroutine objects

    Returns
    -------
    list[T]
        List of values returned by each coroutine
    """

    async def _run_all(tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
        return await asyncio.gather(*tasks)

    return run(_run_all(tasks))


def materialize(aiter: AsyncIterator[T]) -> list[T]:
    """Materialize a list from an async iterator.

    Parameters
    ----------
    aiter : AsyncIterator
        Async iterator object, typically created by calling
        an async function that yields rather than returning.

    Returns
    -------
    list[T]
        List of values yielded by the generator.
    """

    async def _materialize_async(aiter: AsyncIterator[T]) -> list[T]:
        out = []
        async for item in aiter:
            out.append(item)
        return out

    return run(_materialize_async(aiter))
