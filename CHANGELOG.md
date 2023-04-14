
# Changelog

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
