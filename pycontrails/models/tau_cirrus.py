"""Calculate tau cirrus on Met data."""

import dask.array
import xarray as xr

from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import MetVariable
from pycontrails.physics import constants, thermo
from pycontrails.utils.types import ArrayLike

TauCirrus = MetVariable(
    short_name="tau_cirrus",
    standard_name="tau_cirrus",
    long_name="Cirrus optical depth",
    units="dimensionless",
)


def _geopotential_height(met: MetDataset) -> xr.DataArray:
    """Extract geopotential height from MetDataset."""

    # Attempt 1: Use geopotential height if available
    try:
        return met.data["geopotential_height"]
    except KeyError:
        pass

    # Attempt 2: Use geopotential if available
    try:
        return met.data["geopotential"] / constants.g
    except KeyError:
        pass

    # Attempt 3: Approximate geopotential height from altitude
    # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.height_to_geopotential.html
    altitude = met.data["altitude"]
    return altitude * constants.radius_earth / (constants.radius_earth + altitude)


def tau_cirrus(met: MetDataset) -> xr.DataArray:
    """Calculate the optical depth of NWP cirrus around each pressure level.

    Parameters
    ----------
    met : MetDataset
        A MetDataset with the following variables:

        - "air_temperature"
        - "mass_fraction_of_cloud_ice_in_air", "specific_cloud_ice_water_content",
          or "ice_water_mixing_ratio"

    Returns
    -------
    xr.DataArray
        Array of tau cirrus values. Has the same dimensions as the input data.

    Notes
    -----
    Implementation differs from original Fortran implementation in computing the
    vertical derivative of geopotential height. In particular, the finite difference
    at the top-most and bottom-most layers different from original calculation by a
    factor of two. The implementation here is consistent with a numerical approximation
    of the derivative.
    """

    geopotential_height = _geopotential_height(met)

    # TODO: these are not *quite* the same, though we treat them the same for now
    # The generic "mass_fraction_of_cloud_ice_in_air" and ECMWF "specific_cloud_ice_water_content"
    # are mass ice per mass of *moist* air,
    # whereas GFS "ice_water_mixing_ratio" is mass ice per mass of *dry* air
    #
    # The method `cirrus_effective_extinction_coef` uses input of mass ice per mass of *dry* air,
    # so only the GFS data is exactly right.
    if "mass_fraction_of_cloud_ice_in_air" in met.data:
        ciwc = met.data["mass_fraction_of_cloud_ice_in_air"]
    elif "specific_cloud_ice_water_content" in met.data:
        ciwc = met.data["specific_cloud_ice_water_content"]
    elif "ice_water_mixing_ratio" in met.data:
        ciwc = met.data["ice_water_mixing_ratio"]
    else:
        msg = "Could not find cloud ice variable"
        raise KeyError(msg)

    beta_e = cirrus_effective_extinction_coef(
        ciwc,
        met.data["air_temperature"],
        met.data["air_pressure"],
    )

    # dask.array.gradient expects at least 2 elements in each chunk
    level_axis = geopotential_height.get_axis_num("level")
    if geopotential_height.chunks:
        level_chunks = geopotential_height.chunks[level_axis]  # type: ignore[call-overload, index]
        if any(chunk < 2 for chunk in level_chunks):
            geopotential_height = geopotential_height.chunk(level=-1)

    dz = -dask.array.gradient(geopotential_height, axis=level_axis)
    dz = xr.DataArray(dz, dims=geopotential_height.dims)

    da = beta_e * dz

    da = da.cumsum(dim="level")

    return _assign_attrs(da)


def cirrus_effective_extinction_coef(
    ciwc: ArrayLike,
    T: ArrayLike,
    p: ArrayLike,
) -> ArrayLike:
    r"""Calculate the effective extinction coefficient for spectral range 0.2-0.69 um.

    Parameters
    ----------
    ciwc : ArrayLike
        Cloud ice water content, [:math:`kg_{ice} kg_{dry \ air}^{-1}`].
        Note that ECMWF provides specific ice water content per mass *moist* air.
    T : ArrayLike
        Air temperature, [:math:`K`]
    p : ArrayLike
        Air pressure, [:math:`Pa`]

    Returns
    -------
    ArrayLike
        Effective extinction coefficient for spectral range 0.2-0.69 um, [:math:`m^{-1}`]

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`sunParametrizationEffectiveSizes1999`

    Notes
    -----
    References as noted in :cite:`schumannContrailCirrusPrediction2012`:

    - Sun and Rikus QJRMS (1999), 125, 3037-3055
    - Sun QJRMS (2001), 127, 267-271
    - McFarquhar QJRMS (2001), 127, 261-265
    """
    # ciwc has some negative values
    # these become NaN when we raise them to powers down below
    # explicitly clipping at 0
    ciwc = ciwc.clip(min=0.0)

    # Coefficients to calculate beta_e
    a_0_beta = -1.30817e-4
    a_1_beta = 2.52883e0

    rho_air = thermo.rho_d(T, p)
    riwc = ciwc * rho_air * 1000.0

    # calculates the ice particle effective diameter in the NWP cirrus (in um)
    # Computing these exponentials is expensive. If this is a bottleneck, numexpr
    # should be used.
    tiwc = T + constants.absolute_zero + 190.0
    d_eff = 45.8966 * riwc**0.2214 + 0.7957 * tiwc * riwc**0.2535

    # explicitly clipping at 10
    d_eff = d_eff.clip(min=10.0)

    return riwc * (a_0_beta + a_1_beta / d_eff)


def _assign_attrs(da: xr.DataArray) -> xr.DataArray:
    """Assign name and attrs to xr.DataArray.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to assign attributes to

    Returns
    -------
    xr.DataArray
        DataArray with assigned attributes
    """
    da.name = TauCirrus.standard_name
    da.attrs = TauCirrus.attrs

    return da
