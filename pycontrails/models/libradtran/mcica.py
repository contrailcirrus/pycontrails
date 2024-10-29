"""Monte Carlo independent column approximation for partially-cloudy columns."""

import numpy as np
import numpy.typing as npt


def generate_subcolumns(
    lwc: npt.NDArray[np.float64],
    iwc: npt.NDArray[np.float64],
    cf: npt.NDArray[np.float64],
    ncol: int,
    *,
    random_seed: int = 12345,
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]], list[float]]:
    """Generate cloud water content profiles for subcolumns."""

    # Early return for cloud-free profiles
    if not np.any((cf * lwc > 0) | (cf * iwc > 0)):
        return [np.zeros_like(lwc)], [np.zeros_like(iwc)], [1.0]

    # In-cloud water content
    lwc_cld = _safe_divide(lwc, cf, fill=0)
    iwc_cld = _safe_divide(iwc, cf, fill=0)

    rng = np.random.default_rng(random_seed)

    # Compute cumulative cloud fraction from TOA
    pstar = _safe_divide(1.0 - np.maximum(cf[1:], cf[:-1]), 1.0 - cf[:-1])
    C = np.zeros((cf.size + 1,))
    C[1] = cf[0]
    C[2:] = 1 - (1 - C[1]) * np.cumprod(pstar, axis=0)

    # Determine level of highest cloud layer
    rstar = rng.uniform(0, 1, size=(1, ncol))
    itop = np.nonzero(
        (rstar > C[:-1, np.newaxis] / C[-1:, np.newaxis])
        & (rstar <= C[1:, np.newaxis] / C[-1:, np.newaxis])
    )[0]

    # Determine binary cloud fraction for each subcolumn
    x = np.zeros((cf.size, ncol))
    for i in range(cf.size):
        top = i == itop
        x[i, top] = 1

        lower = i > itop
        if lower.sum() == 0:
            continue

        r = rng.uniform(0, 1, size=lower.sum())
        r0 = np.minimum(cf[i], cf[i - 1]) / cf[i]
        r1 = _safe_divide(
            np.maximum(cf[i], cf[i - 1]) - cf[i - 1] - (C[i + 1] - C[i]), C[i] - cf[i - 1]
        )
        x[i, lower] = np.where(x[i - 1, lower] == 1, (r < r0).astype(int), (r < r1).astype(int))

    # Always return a clear-sky subcolumn unless column is completely cloudy
    lwc_sub = [x[:, i] * lwc_cld for i in range(x.shape[1])]
    iwc_sub = [x[:, i] * iwc_cld for i in range(x.shape[1])]
    weights = [C.max() / ncol] * ncol
    if C.max() < 1.0:
        lwc_sub = [np.zeros_like(lwc)] + lwc_sub
        iwc_sub = [np.zeros_like(iwc)] + iwc_sub
        weights = [1 - C.max()] + weights

    return lwc_sub, iwc_sub, weights


def _safe_divide(
    arr1: npt.NDArray[np.float64], arr2: npt.NDArray[np.float64], fill: float = 1.0
) -> npt.NDArray[np.float64]:
    """Safely divide array."""
    if np.any((arr1 > 0) & (arr2 == 0)):
        msg = "Cannot safely divide finite number by zero."
        raise ValueError(msg)
    out = np.full(arr1.shape, fill)
    mask = arr2 != 0
    out[mask] = arr1[mask] / arr2[mask]
    return out
