"""Subcolumn generation for partially-cloudy columns."""

import numpy as np
import numpy.typing as npt


def generate_subcolumns(
    lwc: npt.NDArray[np.float64],
    iwc: npt.NDArray[np.float64],
    cf: npt.NDArray[np.float64],
    columns: npt.NDArray[np.float64],
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]], list[float]]:
    """Generate cloud water content profiles for subcolumns."""

    # Maximum in-cloud water content
    lwc0 = _safe_divide(lwc, cf, fill=0.0)
    iwc0 = _safe_divide(iwc, cf, fill=0.0)

    # Cloud mask
    x = np.where(cf[:, np.newaxis] > columns[np.newaxis, :], 1.0, 0.0)

    # Generate subcolumn profiles
    lwc_sub = [lwc0 * x[:, i] for i in range(x.shape[1])]
    iwc_sub = [iwc0 * x[:, i] for i in range(x.shape[1])]

    # Compute weights
    if columns.size == 1:
        bounds = np.array([0, 1])
    else:
        bounds = np.pad(0.5 * (columns[1:] + columns[:-1]), (1, 1))
        bounds[-1] = 1.0
    weights = np.diff(bounds).tolist()

    return lwc_sub, iwc_sub, weights


def _safe_divide(
    arr1: npt.NDArray[np.float64], arr2: npt.NDArray[np.float64], fill: float = 1.0
) -> npt.NDArray[np.float64]:
    """Safely divide array."""
    out = np.full(arr1.shape, fill)
    mask = arr2 != 0
    out[mask] = arr1[mask] / arr2[mask]
    return out
