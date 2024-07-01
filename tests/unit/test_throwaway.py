import h5netcdf  # noqa
import xarray as xr


def test():
    ds = xr.Dataset(
        {
            "a": (["x"], [1, 2, 3]),
            "b": (["x"], [4, 5, 6]),
        },
        coords={"x": [10, 20, 30]},
    )
    ds.to_netcdf("test.nc")
