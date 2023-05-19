
# Changelog

## v0.42.1

### Features

- Add new `HistogramMatchingWithEckel` experimental humidity scaling model. This is still a work in progress.
- Add new `Flight.fit_altitude` method which uses piecewise linear fitting to smooth a flight profile.
- Add new `pycontrails.core.flightplan` module for parsing ATC flight plans between string and dictionary representations.
- Add new [airports](docs/examples/airports.ipynb) and [flightplan](docs/examples/flightplan.ipynb) examples.

### Breaking changes

- No longer attach empty fields "sdr", "rsr", "olr", "rf_sw", "rf_lw", "rf_net" onto the `source` parameter in `Cocip.eval` when the flight doesn't generate any persistent contrails.
- Remove params `humidity_scaling`, `rhi_adj_uncertainty`, and `rhi_boost_exponent_uncertainty` from `CocipUncertaintyParams`.
- Change the default value for `parallel` from True to False in `xr.open_mfdataset`. This can be overridden by setting the `xr_kwargs` parameter in `ERA5.open_metdataset`.

### Fixes

- Fix a unit test (`test_dtypes.py::test_issr_sac_grid_output`) that occasionally hangs. There may be another test in `test_ecmwf.py` that suffers from the same issue.
- Fix issue encountered in `Cocip.eval` when concatenating contrails with inconsistent values for `_out_of_bounds`. This is only relevant when running the model with the experimental parameter `interpolation_use_indices=True`.
- Add a `Fleet.max_distance_gap` property. The previous property on the `Flight` class was not applicable to `Fleet` instances.
- Fix warning in `Flight` class to correctly suggest adding kwarg `drop_duplicated_times`.
- Fix an issue in the `VectorDataset` constructor with a `data` parameter of type `pd.DataFrame`. Previously, time data was rewritten to the underlying DataFrame. This could cause copy-on-write issues if the DataFrame was a view of another DataFrame. This is now avoided.

### Internals

- When possible, replace type hints `np.ndarray` -> `np.typing.NDArray[np.float_]` in the `cocip`, `cocip_params`, `cocip_uncertainty`, `radiative_forcing`, and `wake_vortex` modules.
- Slight performance enhancements in the `radiative_forcing` module.
- Change the default value of `u_wind` and `v_wind` from None to 0 in `Flight.segment_true_airspeed`. This makes more sense semantically.

## v0.42.0

Phase 1 of the Spire datalib, which contains functions to identify unique flight trajectories from the raw Spire ADS-B data.

#### Features

- Add a `pycontrails.core.airport` module to read and process the global airport database, which can be used to identify the nearest airport to a given coordinate.
- Add a `pycontrails.datalib.spire.clean` function to remove and address erroneous waypoints in the raw Spire ADS-B data.
- Add a `pycontrails.datalib.spire.filter_altitude` function to remove noise in cruise altitude.
- Add a `pycontrails.datalib.spire.identify_flights` function to identify unique flight trajectories from ADS-B messages.
- Add a `pycontrails.datalib.spire.validate_trajectory` function to check the validity of the identified trajectories from ADS-B messages.
- Add a `FlightPhase` integer `Enum` in the `flight` module. This includes a new `level_flight` flight phase.

#### Internals

- Add unit tests providing examples to identify unique flights.
- Rename `flight._dt_waypoints` -> `flight.segment_duration`.
- Move `jet.rate_of_climb_descent` -> `flight.segment_rocd`.
- Move `jet.identify_phase_of_flight` -> `flight.segment_phase`.
- Update `FlightPhase` to be a dictionary enumeration of flight phases.
- Add references to [`traffic` library](https://traffic-viz.github.io/).

## v0.41.0

Improve polygon algorithms.

#### Features

- Rewrite the `polygon` module to run computation with [opencv](https://docs.opencv.org/4.x/index.html) in place of [scikit-image](https://scikit-image.org/) for finding contours. This change improves the algorithm runtime and fixes some previous unstable behavior in finding nested contours. For an introduction to the methodology, see the [OpenCV contour tutorial](https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html).

#### Breaking changes

- Completely rewrite the `polygon` module. Replace the "main" public function `polygon.find_contours_to_depth` with `polygon.find_multipolygon`. Replace the `polygon.contour_to_lat_lon` function with `polygon.multipolygon_to_geojson`. Return `shapely` objects when convenient to do so.
- Convert continuous data to binary for polygon computation.
- Remove parameters `min_area_to_iterate` and `depth` in the `MetDataArray.to_polygon_feature` method. The `depth` parameter has been replaced by the boolean `interiors` parameter. Add a `properties` parameter for adding properties to the `Polygon` and `MultiPolygon` features. The `max_area` and `epsilon` parameters are now expressed in terms of latitude-longitude degrees.

#### Internals

- Add `opencv-python-headless>=4.5` as an optional "vis" dependency. Some flavor of `opencv` is required for the updated polygon algorithms.

## v0.40.1

#### Fixes

- Use [oldest-supported-numpy](https://pypi.org/project/oldest-supported-numpy/) for building pycontrails wheels. This allows pycontrails to be compatible with environments that use old versions of numpy. The [pycontrails v0.40.0 wheels](https://pypi.org/project/pycontrails/0.40.0/#files) are not compatible with numpy 1.22.

## v0.40.0

Support scipy 1.10, improve interpolation performance, and fix many windows issues.

#### Features

- Improve interpolation performance by cythonizing linear interpolation. This extends the approach taken in [scipy 1.10](https://github.com/scipy/scipy/pull/17291). The pycontrails [cython routines](pycontrails/core/rgi_cython.pyx) allow for both float64 and float32 grids via cython fused types (the current scipy implementation assumes float64). In addition, interpolation up to dimension 4 is supported (the scipy implementation supports dimension 1 and 2).
- Officially support [scipy 1.10](https://scipy.github.io/devdocs/release/1.10.0-notes.html).
- Officially test on windows in the GitHub Actions CI.
- Build custom wheels for python 3.9, 3.10, and 3.11 for the following platforms:
  - Linux (x86_64)
  - macOS (arm64 and x86_64)
  - Windows (x86_64)

#### Breaking changes

- Change `MetDataset` and `MetDataArray` conventions: underlying dimension coordinates are automatically promoted to float64.
- Change how datetime arrays are converted to floating values for interpolation. The new approach introduces small differences compared with the previous implementation. These differences are significant enough to see relative differences in CoCiP predictions on the order of 1e-4.

#### Fixes

- Unit tests no longer raise errors when the `pycontrails-bada` package is not installed. Instead, some tests are skipped.
- Fix many numpy casting issues encountered on windows.
- Fix temp file issues encountered on windows.
- Officially support changes in `xarray` 2023.04 and `pandas` 2.0.

#### Internals

- Make the `interpolation` module more aligned with [scipy 1.10 enhancements](https://docs.scipy.org/doc/scipy/release.1.10.0.html#scipy-interpolate-improvements) to the `RegularGridInterpolator`. In particular, grid coordinates now must be float64.
- Use [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/) to build wheels for Linux, macOS (arm64 and x86_64), and Windows on [release](.github/workflows/release.yaml) in Github Actions. Allow this workflow to be triggered manually to test the release process without actually publishing to PyPI.
- Simplify interpolation with pre-computed indices (invoked with the model parameter `interpolation_use_indices`) via a `RGIArtifacts` interface.
- Overhaul much of the interpolation module to improve performance.
- Slight performance enhancements to the `met` module.

## v0.39.6

#### Features

- Add `geo.azimuth` and `geo.segment_azimuth` functions to calculate the azimuth between coordinates. Azimuth is the angle between coordinates relative to true north on the interval `[0, 360)`.

#### Fixes

- Fix edge case in polygon algorithm by utilizing the `fully_connected` parameter in `measure.find_contours`. This update leads to slight changes in interior contours in some cases.
- Fix hard-coded POSIX path in `conftest.py` for windows compatibility.
- Address `PermissionError` raised by `shutil.copy` when the destination file is open in another thread. The copy is skipped and a warning is logged.
- Fix some unit tests in `test_vector.py` and `test_ecmwf.py` for windows compatibility. There are still a few tests that fail on windows (unrelated to changes in this release) that will be fixed in v0.40.0.
- Allow `cachestore=None` to skip caching in `ERA5`, `HRES`, and `GFS` interfaces. Previously a default `DiskCacheStore` was created even when `cachestore=None`. By default, caching is still enabled.

#### Internals

- Clean up `geo` module docstrings.
- Add Wikipedia reference for [Azimuth](https://en.wikipedia.org/wiki/Azimuth).
- Convert `MetBase._load` to from a class method to a function.

## v0.39.5

#### Fixes

- Fix `docs/examples/CoCiP.ipynb` example demonstrating aircraft performance integration
- Fix unit test caused by breaking change in [pyproj 3.5.0](https://github.com/pyproj4/pyproj/releases/tag/3.5.0)

#### Internals

- Add additional Zenodo metadata
- Execute notebook examples in [Docs Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml)

## v0.39.4

#### Internals

- Add additional Zenodo metadata
- Add [Doc / Notebook test Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/doctest.yaml) to run notebook test (`make nb-test`) and doctests (`make doctest`) on pull requests.
- Update [Docs Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml) to use python 3.11.

## v0.39.3

#### Fixes

- Update links in the [README](README.md).
- Update the [release guide](RELEASE.md) to include a checklist of steps to follow when cutting a release.

## v0.39.2

#### Fixes

- Fix links in documentation website.
- Deploy build to [PyPI](https://pypi.org/project/pycontrails/) (in addition to Test PyPI) on release.

## v0.39.1

#### Features

- Use [setuptools_scm](https://github.com/pypa/setuptools_scm) to manage the `pycontrails` version.

## v0.39.0

#### Features

- Add Apache-2 LICENSE and NOTICE files (#21)
- Add [CONTRIBUTING](https://github.com/contrailcirrus/pycontrails/blob/main/CONTRIBUTING.md) document
- Update documentation website [py.contrails.org](https://py.contrails.org).
  Includes `install` and `develop` guides, update citations
  and many other small improvements (#19).
- Initiate the [Github Discussion](https://github.com/contrailcirrus/pycontrails/discussions) forum (#22).

#### Fixes

- Fix erroneous docstrings in `emissions.py` (#25)

#### Internals

- Add Github action to push to `pypi` on tag (#3)
- Replace `flake8` with `ruff` linter
- Add `nb-clean` to pre-commit hooks for example notebooks
- Add `doc8` rst linter and pre-commit hook

## v0.38.0

#### Breaking changes

- Change default value of `epsilon` parameter in method `MetDataArray.to_polygon_feature` from 0.15 to 0.0.
- Change the polygon simplification algorithm. The new implementation uses `shapely.buffer` and doesn't try to preserve the topology of the simplified polygon. This change may result in slightly different polygon geometries.

#### Internals

- Add `depth` parameter to `MetDataArray.to_polygon_feature` to control the depth of the contour searching.
- Add experimental `convex_hull` parameter to `MetDataArray.to_polygon_feature` to control whether to take the convex hull of each contour.
- Warn if `iso_value` is not specified in `MetDataArray.to_polygon_feature`.

#### Fixes

- Consolidate three redundant implementations of standardizing variables into a single `met.standardize_variables`.
- Ensure simplified polygons returned by `MetDataArray.to_polygon_feature` are disjoint. While non-disjoint polygons don't violate the GeoJSON spec, they can cause problems in some applications.

## v0.37.3

#### Internals

- Add citations for ISA calculations
- Abstract functionality to convert a Dataset or DataArray to longitude coordinates `[-180, 180)` into `core.met.shift_longitude`. Add tests for method.
- Add auto-formatting checks to CI testing

## v0.37.2

ACCF integration updates

#### Fixes

- Fixes ability to evaluate ACCF model over a `MetDataset` grid by passing in tuple of (dims, data) when assigning data
- Fixes minor issues with ACCF configuration and allows more configuration options
- Updates example ACCF notebook with example of how to set configuration options when evaluating ACCFs over a grid

## v0.37.1

#### Features

- Include "rhi" and "iwc" variables in `CocipGrid` verbose outputs.

## v0.37.0

#### Breaking changes

- Update CoCiP unit test static results for breaking changes in tau cirrus calculation. The relative difference in pinned energy forcing values is less than 0.001%.

#### Fixes

- Fix geopotential height gradient calculation in the `tau_cirrus` module. When calculating finite differences along the vertical axis, the tau cirrus model previously divided the top and bottom differences by 2. To numerically approximate the derivative at the top and bottom levels, these differences should have actually been divided by 1. The calculation now uses `np.gradient` to calculate the derivative along the vertical axis, which handles this correctly.
- Make tau cirrus calculation slightly more performant.
- Include a warning in the suspect `IFS._calc_geopotential` implementation.

#### Internals

- Remove `_deprecated.tau_cirrus_alt`. This function is now the default `tau_cirrus` calculation. The original `tau_cirrus` calculation is still available in the `_deprecated` module.
- Run `flake8` over test modules.
