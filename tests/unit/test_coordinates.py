"""Test pycontrails.core.coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycontrails.core import coordinates


def test_slice_domain() -> None:
    """Test coordinates.slice_domain()."""
    domain = np.arange(-180, 180, 1.0)

    # normal request within bounds
    request = np.arange(-20, 20, 1)
    sl = coordinates.slice_domain(domain, request)
    assert isinstance(sl, slice)
    assert sl.start == 160
    assert sl.stop == 200
    assert sl.step is None

    # normal request within bounds with buffer
    request = np.arange(-20, 20, 1)
    sl = coordinates.slice_domain(domain, request, buffer=(5, 10))
    assert isinstance(sl, slice)
    assert sl.start == 155
    assert sl.stop == 210
    assert sl.step is None

    # on the min edge of the bounds
    request = np.arange(-200, 20, 1)
    sl = coordinates.slice_domain(domain, request)
    assert sl.start == 0
    assert sl.stop == 200
    assert sl.step is None

    # complete outside min edge of the bounds
    request = np.arange(-220, -190, 1)
    sl = coordinates.slice_domain(domain, request)
    assert sl.start == 0
    assert sl.stop == 2
    assert sl.step is None

    # on the max edge of the bounds
    request = np.arange(-20, 200, 1)
    sl = coordinates.slice_domain(domain, request)
    assert sl.start == 160
    assert sl.stop == len(domain)
    assert sl.step is None

    # completely outside min edge of the bounds
    request = np.arange(190, 220, 1)
    sl = coordinates.slice_domain(domain, request)
    assert sl.start == len(domain) - 2
    assert sl.stop == len(domain)
    assert sl.step is None

    # handle nans
    request = np.arange(-20.0, 20.0, 1)
    request[3] = np.nan
    sl = coordinates.slice_domain(domain, request)
    assert sl.start == 160
    assert sl.stop == 200
    assert sl.step is None

    # fake domain that is short should just be returned
    domain = np.array([0, 1])
    request = np.arange(100, 200, 1)
    sl = coordinates.slice_domain(domain, request)
    assert sl.start is None
    assert sl.stop is None
    assert sl.step is None

    # unsorted domain should throw an error
    request = np.arange(-20, 20, 1)
    with pytest.raises(ValueError, match="Domain must be sorted"):
        sl = coordinates.slice_domain(np.array([3, 1, 2, 4, 5, 6]), request)


def test_buffer_domain_time_like():
    """Test coordinates.slice_domain on time-like domain."""
    domain = pd.date_range("2022-02-01", "2022-02-02", 321)
    request = np.datetime64("2022-02-01T06"), np.datetime64("2022-02-01T12")
    buffer = np.timedelta64(1, "h"), np.timedelta64(1, "h")
    sl = coordinates.slice_domain(domain, request, buffer)
    assert sl.start == 66
    assert sl.stop == 175

    with pytest.warns(UserWarning):
        buffer = buffer[0], -buffer[1]
        coordinates.slice_domain(domain, request, buffer)

    with pytest.warns(UserWarning):
        buffer = -buffer[0], buffer[1]
        coordinates.slice_domain(domain, request, buffer)


def test_buffer_domain_warning():
    """Confirm that slice_domain gives a warning for a negative buffer.

    (This has burned me in the past!)
    """
    domain = np.arange(-180, 180, 1.0)
    request = -100.5, 100.5
    buffer = 0, 0
    sl = coordinates.slice_domain(domain, request, buffer)
    np.testing.assert_array_equal(domain[sl], domain[(domain <= 101) & (domain >= -101)])

    with pytest.warns(UserWarning):
        buffer = -1, 0
        coordinates.slice_domain(domain, request, buffer)

    with pytest.warns(UserWarning):
        buffer = 3, -1
        coordinates.slice_domain(domain, request, buffer)


def test_intersect_domain() -> None:
    """Test coordinates.intersect_domain()."""
    request = np.arange(-20, 20, 1.0)

    # all request points are within domain
    domain = np.arange(-30, 30, 1.0)
    mask = coordinates.intersect_domain(domain, request)
    assert isinstance(mask, np.ndarray)
    assert mask.all()

    # some domain points are within request
    domain = np.arange(-30, 10, 1.0)
    mask = coordinates.intersect_domain(domain, request)
    assert mask.any()
    assert not mask[31:40].any()

    domain = np.arange(-10, 10, 1.0)
    mask = coordinates.intersect_domain(domain, request)
    assert mask.any()
    assert not mask[0:10].any() and not mask[31:40].any()

    domain = np.arange(10, 30, 1.0)
    mask = coordinates.intersect_domain(domain, request)
    assert mask.any()
    assert not mask[0:30].any()

    # no domain points are within request
    domain = np.arange(-40, -25, 1.0)
    mask = coordinates.intersect_domain(domain, request)
    assert not mask.any()

    # supports unsorted requests
    request_unsorted = np.array([3, 1, 2, 4, 5, 10, -1, -5, 3])
    domain = np.arange(-30, 30, 1.0)
    mask = coordinates.intersect_domain(domain, request_unsorted)
    assert mask.all()

    domain = np.arange(-30, 0, 1.0)
    mask = coordinates.intersect_domain(domain, request_unsorted)
    assert mask[6:8].all() and not mask[0:6].any() and not mask[8].any()
