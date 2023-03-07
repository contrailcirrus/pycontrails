
# Changelog

## 0.37.1

#### Features

- Include "rhi" and "iwc" variables in `CocipGrid` verbose outputs.

## 0.37.0

#### Breaking changes

- Update CoCiP unit test static results for breaking changes in tau cirrus calculation. The relative difference in pinned energy forcing values is less than 0.001%.

#### Fixes

- Fix geopotential height gradient calculation in the `tau_cirrus` module. When calculating finite differences along the vertical axis, the tau cirrus model previously divided the top and bottom differences by 2. To numerically approximate the derivative at the top and bottom levels, these differences should have actually been divided by 1. The calculation now uses `np.gradient` to calculate the derivative along the vertical axis, which handles this correctly.
- Make tau cirrus calculation slightly more performant.
- Include a warning in the suspect `IFS._calc_geopotential` implementation.

#### Internals

- Remove `_deprecated.tau_cirrus_alt`. This function is now the default `tau_cirrus` calculation. The original `tau_cirrus` calculation is still available in the `_deprecated` module.
- Run `flake8` over test modules.

## 0.36.0

#### Features

- Include new `Cocip` benchmark data in `/tests/benchmark/cocip/`.

#### Fixes

- Include clipping threshold for SDR based on the cosine of solar zenith angle. In CoCiP, when the cosine of solar zenith angle < 0.01, the SDR is set to 0.

#### Internals

- Fix references and documentation in `geo` and `jet` modules.
- Add testing for solar calculation sin the `geo` module.
- Renames directory `/tests/regression/` to `/tests/benchmark`.
- Includes automatic benchmark test workflow to compare with benchmark test results.

## 0.35.2

#### Breaking changes

- No longer broadcast all numeric `source` attributes in `Cocip.eval`. Instead, if a numeric variable could be present either on `source.data` or `source.attrs`, the new `source.get_data_or_attr` method should be used. This change may break some downstream code that relies on the old behavior. This updates makes our `numpy` calculations slightly more performant and reduces some memory overhead.

#### Fixes

- Avoid dangerous default values on `VectorDataset.setdefault` and `VectorDataDict.setdefault`.

#### Features

- Include new `VectorDataset.get_data_or_attr` method to get data from `source.data` or `source.attrs` if it exists.

## 0.35.1

#### Internals

- Update documentation of `jet.py`.
- Rearrange `jet.py` module so relevant functions are categorised and close to each other.

## 0.35.0

#### Breaking changes

- Removes all datalib classes from top level `pycontrails.datalib` module. Classes must be imported directly from their submodule (e.g. `from pycontrails.datalib.ecmwf import ERA5`, `from pycontrails.datalib.gfs import GFSForecast`)
- Rename module helper functions (#36):
    + `geo.haversine_segments` -> `geo.segment_haversine`
    + `geo.longitude_angles` -> `geo.longitude_angle`
    + `geo.segment_angles` -> `geo.segment_angle`. Includes `Flight.segment_angle`.
    + `geo.segment_lengths` -> `geo.segment_length`. Includes `Flight.segment_length`.

#### Features

- Officially support Python 3.11 (#22)
- Add `Flight.segment_duration` helper method (#15)

#### Internals

- Move cirium support out to another extension [pycontrails-cirium](https://github.com/contrailcirrus/pycontrails-cirium) (#34)
- Remove deprecations (Raise on `GCPCacheStore.clear()`, remove `Cocip.eval(flight=...)`, remove `removeMetDataArray.to_polygons()`) (#23)
- Consolidate static ECMWF test data for unit tests
- Remove `pd.Series` from `ArrayLike` type. Pandas series sometimes result in unforseen errors when using like a `np.ndarray`. Use `Series.to_numpy()` before using `ArrayLike` inputs. (#25)
- Renames `met_data_source` to `met_source` on `MetDataset` and `MetDataArray` classes. (#16)

## 0.34.1

#### Fixes

- Replace the officially deprecated [appdirs package](https://github.com/ActiveState/appdirs) with its successor [platformdirs](https://github.com/platformdirs/platformdirs).
- Defer import of `platformdirs` until it is needed. This avoids an unnecessary dependency for users who do not need the ability to cache data with to a default user-specific directory.

## 0.34.0

Removes BADA-specific interface into private adjacent module

#### Breaking changes

- Removes explicit support and testing for Python 3.8 
- Removes `pycontrails.models.bada` model and associated units tests
- Includes optional `pycontrails.ext.bada` dependency for running BADA models using private [pycontrails-bada](https://github.com/contrailcirrus/pycontrails-bada) extension.
- Removes all models from top level `pycontrails.models` module. Models must be imported directly from their submodule (e.g. `from pycontrails.models.issr import ISSR`, `from pycontrails.models.cocip import Cocip`)
- Removes BADA-specific parameters from `CocipParams`. Now, the `Cocip` model includes an `"aircraft_performance"` param that must be an instance of an `AircraftPerformance` model. Note that the parameters to the `CocipGrid` model have not been changed.

#### Features

- Includes aircraft performance functions in the `physics.jet` module that were previously included with the `BADA` model.

#### Fixes

#### Internals

- Maintain package version in `pyproject.toml` and assign to `pycontrails.__version__` in __init__.py
- Import core packages in `pycontrails.core.__init__.py`
- Removes published docker images and associated Github action
- Removes `extensions` module namespace in favor of shorter `ext` module
- Adds `models.aircraft_performance` as abstract base class for aircraft performance models. Currently used as the base class for `ext.bada` models
- Adds helper functions to evaluate emissions and aircraft performance models in `Cocip`
- Fully deprecated `ContrailGrid` reference
- Fixed circular dependency between `Emissions` and `Cocip`. The `Emissions` model is now fully upstream of `Cocip` and no longer depends on `CocipParams`.

## 0.33.8

#### Fixes

- Remove OS dependencies from README

#### Internals

- Make `pip-install` no longer installs dev dependencies. Use make `dev-install` recipe instead.
- Update `CoCiP.ipynb` notebook example for clarity

## 0.33.7

### Fixes

- Don't promote the `dtype` when calling the `Flight.segment_azimuth` method. Now the `dtype` of the returned azimuth is the same as longitude array.

## 0.33.6

### Fixes

- Fix a bug `CocipGrid` in which running the model in "vector" mode with explicit segments lengths causes an indexing error in the `result_merge_source` function.

## 0.33.5

### Fixes

- Fix a bug in `BADAGrid` in which a conflicting shape was used to reshape 1D array data into a 4D gridded output.

## 0.33.1

Remove Github deploy key from docker images

### Internals

- Removes `docker-deploy` key from docker image builds

## 0.33.0

Fixes bug computing fuel flow during slow ascent and descent.

### Internals

An aircraft is in climb / descent if it is near the nominal rocd for the specified altitude as specified by the BADA PTF data. If the aircraft has a ROCD < -100 or ROCD > 100, and it is not near the nominal climb / descend value, the nominal flights phase value is left as `nan`, and no fuel correction is applied. In this case, we are assuming that the aircraft is doing something that is not "nominal" like a powered descent or a step climb.

The original `common.FlightPhase` still gets used in a couple places, mainly in the BADA4 actions model.

## 0.32.2

Implement experimental "segment-free" `CocipGrid` model.

### Features

- Allow `CocipGrid` to run in an experimental "segment-free" mode by setting parameters `azimuth=None` and `segment_length=None`. This mode disables all evolution effects of wind shear normal to the underlying contrail segment. It also eliminates any energy forcing scaling by the segment ratio used in the standard `CocipGrid` model. This mode is intended to be used for a directionless contrail plume, thereby eliminating the need to specify an azimuth parameter.
- Include experimental `dsn_dz_factor` parameter in `CocipGrid`. This is used to approximate `dsn_dz` with `dsn_dz_factor * ds_dz`. It is only considered when `azimuth=None` and `segment_length=None`. Setting a nonzero value here outside of the experimental segment-free mode will raise an error.

### Internals

- Move the `wind_shear` function from the `thermo.py` module into the `wind_shear.py` module.
- Remove outdated integration tests for `CocipGrid` model.
- Clean up the `cocip_grid.run_interpolators` function. Change function signatures for `cocip_grid.calc_wind_shear` and `cocip_grid.advect`.
- Exclude `scipy` 1.10 from the dependency list. This is temporary and will be fixed in January 2023.

## 0.32.1

### Breaking changes

- Require `shapely` for polygon constructions. This can be installed with `pip install pycontrails[vis]` or `pip install shapely`.

### Fixes

- In the `MetDataArray.to_polygon_feature` method, include `precision` parameter.
- Simplify polygons in `polygon.clean_contours` using `shapely` instead of homegrown algorithms. This ensures the resulting polygons are valid.
- Include `min_area_to_iterate` parameter in `polygon.find_contours_to_depth`. This helps avoid extra recursion when exterior contours are very small.
- Fix `polygon.contour_to_lon_lat` to ensure that the returned polygon is valid. Previously, rounding the coordinates could cause a polygon to become invalid.

### Internals

- Include new `polygon_bug.nc` static test file for testing polygon simplification.

## 0.32.0

### Fixes

- Fix CoCiP radiative heating unit test failing because comparison was too precise

### Internals

- Migrate repository to Github at remote <https://github.com/contrailcirrus/pycontrails.git>
- Update issue templates, pull request templates for Github
- Translate Gitlab CI jobs into Github Actions. Unit tests run on fresh linux instances as well as pre-built pycontrails docker images.
- Move static BADA test files to remote bucket `gs://contrails-301217-bada/bada-outputs`.
- Update sphinx docs url to `https://py.contrails.earth`

## 0.31.7

### Features

- Add option to disable checks involving `max_altitude_m` and `min_altitude_m` in `contrail_properties.contrail_persistent`. These parameters can be set to None to disable the checks.

## 0.31.6

### Fixes

- Fix depth clipping in `contrail_properties.plume_effective_depth`. This was partially fixed in 6e918cf8, and this version fixes the remaining issue. Rather than clipping `sigma_zz` only, the `diffuse_v` is now clipped to avoid contrail depths exceeding the `max_contrail_depth` parameter.
- Remove the extra `dt_integration` used to compute the `CocipGrid` max time in the `cocip_time_handling` module. This was causing the model to run for a single extra unnecessary evolution step.
- Provide a more accurate estimate in the `CocipGrid._estimate_runtime` method.
- Fix a `float32` -> `float64` dtype conversion in `Model.set_source_met` for `MetDataset` source.

### Internals

- Minor cleanups and optimizations in the `contrail_properties` and `models` modules.
- Configure `isort` with `--profile "black".

## 0.31.5

### Features

- Adds example notebook to demonstrate use of `ACCF` model

### Fixes

- Suppresses warning message about humidity scaling in `ACCF` model
- Allows `ERA5` data source to download variables required for `ACCF` model

## 0.31.4

Update radiative heating methodology.

#### Fixes

- Include additional radiative heating equations and improved documentation on their explanation.
- When radiative heating is enabled, the temperature of the contrail plume changes with the cumulative heating. Previously CoCiP assumes that the temperature in the contrail plume is identical to the ambient temperature.
- Include additional `CocipParam` to constrain maximum contrail depth, currently set to 1.5 km.

## 0.31.3

#### Fixes

- Fix bug in HRES where forecast time was not being set
- Add check that `forecast_time` is equal to `time` when using `paths` input.

#### Internals

- Make cachestore optional for MetDataSource when using `paths` input. If `cachestore` is set to None, the MetDataSource will ignore it.
- Convert `step_offset` into a property.
- Add utility to round hours down to the nearest X hour.

## 0.31.2

#### Internals

- Abstract opening and caching local paths in `HRES` and `ERA5` interfaces.
- Add `HRES` dissemination filename generator helper function

## 0.31.1

#### Features

- Enable cache access when using local file paths with `ERA5` and `HRES`. This can save time in loading grib source files.

#### Fixes

- Fix loading multiple HRES grib files at once from local paths
- Set default xr_kwarg `"parallel"` to False for grib files or else python crashes
- Fix `GCPCacheStore` tests so each test run has its own `cache_dir`, avoiding random test clashes.

#### Internals

- Add helper `datalib.MetDataSource` methods to list datafiles cached or not cached
- Add cache override to conftest to use with met files provided as paths. This avoids inadvertently loading met files from a local cache during tests.

## 0.31.0

A wrapper around the `climaccf` Python library from DLR to allow ACCFs to be evaluated over a flight trajectory of meteorology grid.

### Features

- Adds a class `ACCF` which implements `Model` and allows ACCFs to be computed over a `Flight`, `MetDataset`, or `GeoVectorDataset`.

### Internals

- The `ACCF` constructor does little but sets up internal variables for the libary and ensures prerequisit pacakges are installed.
- `ACCF.eval` downselects the met data and performs the actual computations of the ACCF functions.
- Adds additional `MetVaraibles` within the ECMWF code to define PotentialVorticity and SurfaceSolarDownwardRadiation.
- Adds a helper function `standardize_variables` to `met` which converts ECMWF short names to standard names.
- The `pip-install` recipe doesn't fail if the `climaccf` package is not available. Note that the `casadi` package (a `climaccf` dependency) isn't available for Python 3.10.
- Try to import `netCDF4` in `pycontrails.__init__` as a workaround for [a recent issue](https://github.com/pydata/xarray/issues/7259). This was handled differently in `pycontrails 0.28.8`.

## 0.30.0

Better support for local source files in `datalib` classes.

#### Features

- Updates `MetDataSource` constructor to take `paths` input. The `paths` input corresponds to any local file  in the original source format.
- Enable `MetDataSource` to open local `paths` input without specifying `time` range to select. In this case, `time` is set to None.

#### Fixes

- Fix bug in `GFS.step_offset` for forecast retrieval

#### Internals

**Datalib**

- Consolidates `MetDataSourceWithDownload` into `MetDataSource`. Constructor `grid` input is now generic.
- Renames `MetDataSource.surface_variables` to `MetDataSource.single_level_variables`
- Renames `MetDataSource.cachepath` method to `MetDatasSource.create_cachepath`
- Renames `MetDataSource.cachepaths` property to `MetDatasSource._cachepaths`
- Add `MetDataSource.cache_dataset` abstract method to allow the source to be cached after loading files.

**EMCMWF**

- Splits ERA5, HRES, and IFS classes into independent modules
- Removes as much abstraction as possible from `Ecmwfapi` class

**Cache**

- Remove support for putting objects into a `CacheStore`
- Support `pathlib.Path` in `CacheStore.put`

**Tests**

- Use separate cache for test fixture setup scripts.
- Updates met data test fixtures. Old test fixtures had more level dimensions than the script should have created, so there are some test changes to reduce the number of available pressure levels.

## 0.29.1

Bugfixes for `pycontrails 0.29.0`.

#### Fixes

- Annotate `engine_type_edb` and `engine_uid_edb` as `str | None` in `AircraftProperties`. This accurately reflects the type of the data.
- Fill missing values with None when reading `bada-synonym-list-20221102.csv` and `cirium-edb-cleaned-20221026.csv`. This is consistent with the assumed types of this data in downstream usage.
- In `BADAModel`, only attach `source.attrs` variables `engine_type_edb` and `engine_uid_edb` when they are not None.
- Rename `thrust_settings` to `thrust_setting` in the `emissions.py` module.
- Copy the `fuel` attribute when calling the `Flight.copy` method. Previously this attribute was dropped.

## 0.29.0

Support for Cirium queries and emissions upgrades.

#### Breaking changes

- The `Emissions` model attributes `edb_engine_gaseous` and `edb_engine_nvpm` are now indexed by `engine_uid`. Previously the keys of these dictionaries were `edb_atyp`. The `Emissions.eval` method now looks for an "engine_uid" attribute on the source parameter. Previously, only the aircraft type was considered.
- Update `emissions/static` files. The CSVs now include 557 unique engines for gaseous data (was 117 engines previously) and 178 unique engines for nvPM data (was 47 engines previously).
- The `BADA4` model now uses both aircraft and engine types in determining aircraft properties. Previously, a default engine was assumed.
- Update `bada/static` files with new default aircraft - engine pairs.
- Include "engine_uid" parameter in `CocipGrid` and `BADAGrid`.
- Rename `raise_errors` parameter to `raise_error` and always use a default value of True. Previously, these were used inconsistently throughout pycontrails.

#### Features

- Support Cirium queries to extract specialized aircraft properties (registered engine type and mass).
- Emissions T4/T2 methodology now accounts for step change in nvPM emissions from staged combustors (double annular combustors and twin annular premixing swirler combustors).
- Add `HydrogenFuel` fuel class.
- Improvements to reserve fuel methodology according to Wasiuk et al. Previous methodology to estimate the reserve fuel was too low, causing the aircraft mass and fuel consumption to be underestimated.

#### Fixes

- The ICAO EDB nvPM emissions profile no longer depend on the fuel inputs. Instead, the value from the EDB is used.

#### Internals

- Refactor the `emissions.py` module to be more functional. The `Emissions.eval` implementation now avoids redundant calculation of `thrust_settings`.
- `BADA` classes now load a "aircraft_engine_dataframe" attribute.

## 0.28.9

New experimental `radiative_heating_effects` `Cocip` parameter.

#### Features

- Contrails absorb incoming solar radiation and outgoing longwave radiation, causing it to heat up. The additional heating energy drives a local updraft (but this is negligible and not included in CoCiP); and the  differential heating rate drives local turbulence. As radiative emission from the contrail is limited by its low temperature, the net result is deposition of radiative energy in the contrail. Turning on the `CocipParam.radiative_heating_effects` parameter simulates this process in `Cocip`.

## 0.28.8

Improve `ExponentialBoostLatitudeCorrectionHumidityScaling` with a variable upper limit based on thermodynamics.

#### Features

- Replace `ExponentialBoostLatitudeCorrectionHumidityScaling` parameter `clip_upper` with a dynamically calculated `rhi_max`. This new upper limit is calculated at each waypoint as a function of air temperature based on thermodynamics.

#### Fixes

- Upgrade to `mypy 0.99` and fix lingering mypy issue in `xml_bada4`. Enforce `warn_unused_ignores` in `setup.cfg`.

#### Internals

- Exclude `xarray` version `2022.11.0` in `setup.cfg`. This is a workaround for [a recent issue](https://github.com/pydata/xarray/issues/7259).
- Consolidate package information and tool configuration into `pyproject.toml`.

## 0.28.7

Fix dtype promotion in BADA and Emissions models.

#### Fixes

- Fix issue in `BADA` and `Emissions` models in which `float32` dtypes were promoted to `float64`.

#### Internals

- Avoid using `np.append` or `np.diff(..., append=...)` when possible. These patterns automatically promote the underlying array dtype and require an extra copy.

## 0.28.6

Update pycontrails for python 3.11 support.

#### Fixes

- Upgrade pycontrails `dataclasses` to use `default_factory` patterns for [mutable fields](https://docs.python.org/3/library/dataclasses.html#mutable-default-values).
- Fix bug in `Cocip` fleet calculation in which "global" `Fleet.attrs` were lost.

#### Internals

- Remove `attr_keys` from `Fleet` construction. This key was not used.

## 0.28.5

#### Fixes

- Fix inconsistency by renaming `thrust_settings` to `thrust_setting`.

## 0.28.4

#### Fixes

- Fix bug in `Cocip._fill_empty_flight_results`. This bug only cropped up when the `source` parameter is a `Fleet` instance.

## 0.28.3

More memory efficient implementation of contrail -> flight aggregation in `Cocip`.

#### Breaking changes

- In `Cocip.eval`, only calculate the `contrail_dataset` attribute if `params["verbose_outputs"]` is True. The `contrail` attribute remains the same.
- Rename `sdr` -> `sdr_mean`, `olr` -> `olr_mean`, `rsr` -> `rsr_mean`, `rw_sw` -> `rw_sw_mean`, `rw_lw` -> `rw_lw_mean`, `rf_net` -> `rf_net_mean` on the `Cocip.contrail` instance attribute.

#### Features

- Implement a more memory efficient `Cocip._bundle_results` method. In particular, the aggregation from contrail data to flight data now bypasses the `contrail_dataset` construction, and instead uses a pandas `groupby` directly on the pre-computed `contrail` attribute. When running `Cocip` over a large fleet, the new implementation uses 10x less memory than the previous implementation.

## 0.28.2

Integrate SAF work from Teoh et al. (2022) (ES&T) into pycontrails.

#### Features

- Add new `SAFBlend` subclass of `Fuel`. This takes a single `pct_blend` parameter to construct a Jet A-1 / SAF blend. It can be accessed from the top level pycontrails API (`from pycontrails import SAFBlend`).
- Change nvPM mass and number emissions calculations if a SAF is used.

#### Fixes

- Rename `bc_ei_m` to `nvpm_ei_m` (note the "m" as opposed to "n") in several remaining places (see 0.28.0 release notes for the source of this change).
- Correct formula used in the `jet.compressor_inlet_pressure` function.

## 0.28.1

#### Fixes

- Fix bug in `MetDataset.set_source_met` by allowing attributes `source` and `met` to have different coordinate resolution in the case when `source` is also a `MetDataset` instance. The previous implementation assumed both grids had identical coordinate skeletons. The fix allows for arbitrary grids to be used as `source`. In particular, when the `source` coordinate grid is not a subset of the `met` coordinate grid, the `xr.DataArray.interp` method is used.
- Flatten humidity scaling metadata when attaching to `MetDataset.attrs`. This fixes a bug in which nested dictionaries in `MetDataset.attrs` could not be consumed by `xr.Dataset.to_netcdf`. In particular, the `SAC`, `ISSR`, `PCR`, and `CocipGrid` model gridded output can now seamlessly be saved to netCDF.
- Fix `Cocip` "fleet computation" issue in which NaT values in `contrail_age` were handled differently from the "flight computation". These NaT values are now 0 filled in `Cocip._bundle_results`.

#### Internals

- Add `Fleet` class to `pycontrails/__init__.py`.
- The `Cocip.eval` method now returns a `Fleet` instance if a fleet was originally provided in the `source` parameter. This method still maps `Sequence[Flight]` to `list[Flight]`.

## 0.28.0

Update class and variable names

#### Breaking changes

- Rename `ContrailGrid` to `CocipGrid` and place in module `pycontrails.core.cocipgrid.cocip_grid`
- Rename any field with `bc_ei_n` to `nvpm_ei_n`. This includes in `CocipParams`, which will cause backwards incompatibility.
- Changes `fuel_data_source` output key in `BADA`, `Emissions` and `Cocip models` to `bada_model`

#### Fixes

- Fix `CocipGrid` implementation when no `humidity_scaling` parameter is set

## 0.27.0

Overhaul humidity scaling.

#### Breaking changes

- Change `Cocip` and `ContrailGrid` to NOT use any humidity scaling by default.
- Add new model parameter `HumidityScalingParams.humidity_scaling` to nearly all models (`Cocip`, `ContrailGrid`, `ISSR`, `SAC`, `PCR`, `PCC`, and `Emissions`). Each of these models automatically scaled humidity if a nontrivial `humidity_scaling` parameter is provided in the associated `ModelParams`. Previously, only the `Cocip` and `ContrailGrid` models considered humidity scaling.
- Issue warning the aforementioned models if met data is ERA5 and humidity scaling is NOT enabled.
- The `Cocip.eval` method now no longer applies a constant `rhi_adj` scaling to the specific humidity variable of the `met` attribute. In particular, `met` input data is no longer mutated apart from included the `tau_cirrus` variable if it is not already present.
- Remove the `process_met_datasets` class method from `Cocip`. This is now a standalone function in `cocip.py`.
- Remove the `process_met` `Cocip` model parameter. This is now handled automatically in the `Cocip.__init__` method. Much of the logic around processing the `met` and `rad` parameters is simplified and overhauled in this release.
- Remove `rhi_adj` and `rhi_boost_exponent` parameters from `Cocip`. These are replaced with the high level `humidity_scaling` parameter.

#### Features

- Add new humidity scaling method (Roger Teoh's *global humidity correction*) which extends `ExponentialBoostHumidityScaling` with a new latitude correction term.
- Overhaul all humidity scaling schemes with a common `HumidityScaling` base class, which itself subclasses `Model`. This class features both an abstract `scale` method as well as the usual high-level `eval` method.
- Add new experimental `originates_from_ecmwf` function in `met` module. This property is True if the dataset appears to be derived from an ECMWF source.

#### Fixes

- The exponential boosting humidity scaler `ExponentialBoostHumidityScaling` (the default humidity scaling scheme for `Cocip`) now agrees with Roger Teoh's calculation: both specific humidity and RHi apply an exponential boosting step in the humidity correction.

#### Internals

- Enable updating model parameters on `Model.eval(source, **params)` by updating the base `Model.eval()` class method signature. This must be implemented by `Model` classes using the `self.update_params(params)` helper.
- Remove `_eval_vector` and `_eval_met_grid` methods from `SAC`, `ISSR`, and `PCR`. Because the interfaces for `MetDataset` and `GeoVectorDataset` are highly similar, these models can now apply common logic in the `eval` method. This change ensures better maintainability moving forward and better parity when calling `eval` with distinct source types.
- Update some logic for running `Cocip` with uncertainty. Some aspects of this may still be outdated.
- Rename the `humidity_enhancement` module to `humidity_scaling` and place in the `models/` directory. Include in sphinx documentation.
- The `issr.issr` function now accepts an `rhi` parameter (default None) in case `rhi` is already computed (for example, when humidity scaling is applied).
- The `PCR` model now sequentially calls the `ISSR` and `SAC` models.
- Running `mypy` now warns if `type: ignore` is unnecessary.

## 0.26.0

Low level performance enhancements with an eye toward [Zarr](https://zarr.readthedocs.io/en/stable/) support.

#### Breaking changes

- Change the definition / convention for `MetDataset(wrap_longitude=True)` patterns. The previous implementation duplicated and shifted the minimum **and** maximum longitude values of the underlying grid. The current implementation ensures the longitude dimension **covers** the closed interval ``[-180, 180]`` (typically this can be achieved by duplicating the minimum longitude value only). This is more in-line with how this feature is applied for longitude-advection, and helps avoid some unnecessarily redundant data copying.
- Rename the `MetDataArray.is_loaded` property to `MetDataArray.in_memory`. This is better aligned with the `xr.DataArray._in_memory` property (which is called under the hood).
- Warn if loading met data into memory consumes more than 4GB of memory. Raise a `ValueError` if loading consumes more than 32GB of memory.
- `Cocip` no longer uses the `downselect_met` parameter. This now occurs automatically (see description in Features below).

#### Features

- Implement new experimental interpolation enhancement in which previously computed interpolation indices are saved for repeated interpolations on the same underlying coordinate grid. This provides a nontrivial speed up when interpolating a `GeoVectorDataset` against multiple variables in a `MetDataset`. Essentially, this experimental interpolation mode caches the the outputs of the [`RegularGridInterpolator._find_indices` method](https://github.com/scipy/scipy/blob/v1.9.1/scipy/interpolate/_rgi.py#L418) method to avoid redundant calculation. This new experimental mode can be activated by setting the `interpolation_use_indices` model parameter to True.
- Include a `MetDataset.is_single_level` property to conveniently check if the gridded data is satisfies the surface-level convention (single level item with value -1).
- Include a `MetDataset.is_zarr` property to check if underlying dataset is based on a Zarr store. Implementation is brittle and may be removed in the future.
- Include `MetDataset.from_zarr` class method to instantiate a `MetDataset` from a Zarr store, with an option to wrap in a `zarr.LRUStoreCache` instance. This method is purely for convenience, and should be removed if maintenance becomes non-trivial.
- `Cocip` automatically downselects met data for contrail initialization and contrail evolution. This helps to avoid loading unused met data and ensures a much smaller memory footprint throughout evolution.

#### Fixes

- Reduce default values for `met_longitude_buffer`, `met_latitude_buffer`, and `met_level_buffer`. Only models utilizing an evolution methodology (ie, `Cocip`) require a non-zero buffer.
- `Cocip` now warns if flight trajectory segment lengths are within 90% of the `max_seg_length_m` parameter. Previously, flights with large gaps between waypoints would silently generate non-persistent contrails (ie, contrails reaching their end-of-life too soon).

#### Internals

- The private `Cocip._process_flight`, `Cocip._simulate_wake_vortex_downwash`, and `Cocip._find_initial_persistent_contrails` methods now take a `met` parameter.
- Move the private `Cocip._calc_timestep_contrail_evolution` method to its own standalone function.
- Move the `cocip.py` module function `downselect_met` into `contrail_grid.py`. It is now only used by `ContrailGrid`.
- Remove the `pycontrails.utils.cache.py` module. This is no longer relevant.

## 0.25.12

Improve grid -> polygon conversion

#### Features

- Improve `MetDataset` to GeoJSON conversion through new `to_polygon_feature` method. Instead of comparing all pairs of polygons to determine interior - exterior relationships, the `polygon.py` module implements a recursive procedure for finding nested polygons. The new implementation is linear in the number of polygons (previous implementation was quadratic). The runtime performance improved by 10 - 20x for large complicated grids arising from `ContrailGrid` output.
- The `to_polygon_feature` takes a `min_area` parameter to optionally exclude polygons with insufficient area. This is useful for downstream applications in which very tiny regions of contrail impact can be ignored.
- Optionally include an altitude value as the z-coordinate in the GeoJSON output from `to_polygon_feature`.
- Implement `MetDataArray.to_polygon_feature_collection` to calculate polygons on each vertical level.
- Implement new `find_contours_to_depth` function for low-level polygon calculation. This function recursively calculates contour relationships to an arbitrary nesting depth. This function is called with `depth=2` in `MetDataArray.to_polygon_feature` to calculate GeoJSON polygons with exterior and interior linear rings.

#### Fixes

- Explicitly fill nan values and padded values with parameter `fill_value` in `MetDataArary.to_polygon_feature`. Previously, nan values were not handled, leading to unexpected corrupt polygon output from `skimage.measure.find_contours`.
- Fix `contour_to_lon_lat` implementation. Previous function assumed the underlying array data passed into `skimage.measure.find_contours` was binary (and so all contour indices were either integer or half integer). This is not the case for continuous gridded data (ie, energy forcing predictions from `ContrailGrid`).

#### Internals

- Deprecate `MetDataArray.to_polygons`. The `MetDataArray.to_polygon_feature` method should be used instead.
- Hard-code `to_polygon_feature` polygon orientation to be consistent with GeoJSON spec.
- Use a wider tolerance in `polygon.simplify_contour`. This parameter (`epsilon` in the code) is hard-coded, and should be refactored to a function parameter if more control is needed.
- Change doctests requiring `ERA5` instances to use the same ECMWF data as used by the example notebooks. Include `ensure-era5-cached` make recipe to ensure data is found locally before tests are run.

## 0.25.11

`Cocip` now assigns 0 to `energy_forcing` and `contrail_age` for waypoints *not* producing contrails. Previously, these entries were nan-filled. Values corresponding to outside of the met domain remain nan-filled.

#### Internals

- `Cocip` output variables `ef` and `cocip` are set to 0 at waypoints which do not produce initial contrail formation.
- Intermediate `Cocip.contrail_list` entries are assigned an `age` of 0 for waypoints at which `ef` is also 0. Consequently, waypoints not producing contrails have a `contrail_age` of 0 in the model output.
- Static test fixtures only changed in `ef`, `cocip`, `contrail_age`, `age`, and `age_hours` fields. Any other changes there are a dtype flip from `int` to `float` or vice versa. Completely regenerate `tests/unit/static/cocip-flight-statistics.json`.
- Use an intermediate `_met_intersection` key in `Cocip.source` to keep track of waypoints inside of the met domain. This is removed in the `Cocip._bundle_results` method.
- Slight cleanup of `CoCiP.ipynb` notebook.

## 0.25.10

Check for contrails approaching north and south pole in CoCiP persistence step.

#### Internals

- The function `contrail_properties.contrail_persistent` now implements a check involving contrail latitude values. If the latitudes are outside of the interval `[-89, 89]`, the corresponding contrail waypoint has reached its end of life.
- The private function `contrail_properties._within_range` now takes upper and lower bounds optionally (previously both were required). This mirrors the patterns of `np.clip`.

## 0.25.9

Update emissions `t4_t2` estimate based on comparison of nvPM with ECLIF measurements.

#### Features

- Update non-dimensional engine thrust setting (`t4_t2`) methodology after reviewing data from ground and cruise nvPM measurements from the ECLIF II campaign.
- Update calculation of air-to-fuel ratio.
- `BADA3` now computes the fuel mass flow rate for Piston aircraft. Previously a `NotImplementedError` was raised.

#### Internals

- Avoid a divide-by-zero error previously encountered with zero fuel flow rates during descent. Fuel flow in the descent stage is now clipped to some nominal positive value.

#### Fixes

- Make mass unit consistent in `BADA3` OPF and PTF data. Both now uses kilograms.

## 0.25.8

Fix `MetDataArray.to_polygons()` method

#### Fixes

- Fix `MetDataArray.to_polygons()` implementation to properly meet [GeoJSON Polygon spec](https://www.rfc-editor.org/rfc/rfc7946.html#section-3.1.6). In particular, we add an algorithm to identify and keep track of polygons within other polygons. Note this algorithm does not yet support filled polygons nested within polygon rings.

## 0.25.7

Small build system improvement

#### Internals

- Prefix git tag version with `v`
- Fix CI job to built artifacts on tag

## 0.25.6

Improve CI infrastructure into `/docker` directory.

#### Internals

- Remove `Dockerfile` in favor of `/docker` directory with sub environments.
- Install python dependencies in Docker image
- Add `ci-` make recipes for creating CI images
- New CI images will be created on each pushed tag
- Set default local docker image to `python39`
- Upgrade interpolation patterns to use new `scipy` 1.9 features (let `scipy` RGI remove singleton dimensions instead of doing it ourselves).
- Require `scipy>=1.9` as a dependency

## 0.25.5

Improve CI infrastructure, minor dependency cleanup.

#### Internals

- Removes `linux-setup` Makefile recipe in favor of specifying dependencies in specific docker images.
- Point CI scripts to slightly customized Python container images on GCP Artifact Repository for 3.8, 3.9, 3.10
- Make `dask` a required `pycontrails` dependency.
- Remove `joblib` from the required dependencies.
- Remove the `utils.progress` and the `utils.profiling` modules. These were leftovers from previous `ContrailGrid` implementations.
- Alphabetically sort dependencies in `setup.cfg`.

## 0.25.4

Re-organize `thermo` modules, improve more documentation.

#### Fixes

- Fix `HRES` step_offset calculation when requesting an input `forecast_time`

#### Internals

- Add new `physics.jet` module for jet propulsion relationships. Move existing `thermo` functions into `jet`.
- Add molecular mass of dry air, water vapor, and add liquid vapor pressure calculation to `thermo` module.
- Improve docstring in `Cocip` support modules/functions
- Move wind shear functions to `Cocip` to its own `cocip.wind_shear` module

## 0.25.3

Add new regression test, clean up existing test fixtures, other small documentation improvements.

#### Fixes

- Fix `Cocip` model to re-calculate continuity after filtering for persistent waypoints. Set the "ef" of discontinuous waypoints to 0 after re-calculating continuity.

#### Internals

- Adds `cocip-fortran` regression test. See `tests/regression/cocip-fortran/README.md` for more information.
- Rename 'thermo.T_potential_grad' to 'thermo.T_potential_gradient'.
- Rename unit test fixtures and static filenames to clarify assets.
- Abstract `Cocip.calc_timestep_meteorology` into its own functional method `calc_timestep_meteorology`
- Update minimum package dependencies for `numpy`, `xarray`, and `dask`
- Call `np.clip` in place rather than copying to new array.

## 0.25.2

Fix bug in which `BADAFlight` overwrites aircraft performance attributes.

#### Fixes

- BADAFlight uses any aircraft performance variables defined on `source.attrs`. This is achieved by using `model.get_source_param` instead of `model.source.get`.
- Favor the method `xr.DataArray.where` instead of the function `xr.where` for nan-filling procedures in SAC calculations. This fixes an inconsistency in which attrs are lost in some versions of xarray. In particular, the upgrade from xarray 2022.3 to 2022.6 breaks some of our previous patterns for `xr.where` usage.

#### Internals

- The method `Model.get_source_param` now accepts an optional default value.
- Upgrade type hinting for `xarray` version 2022.6.0
- Remove `XArrayLike` type hint and replace with more restrictive `XArrayType` defined and used in `met.py`.
- Include `variables` property on `MetBase` for slightly faster access to coordinate variables.
- Replace `xr.where` with `clip` in the `tau_cirrus` module. This is semantically more correct and provides slightly better performance.

## 0.25.1

Fix minor bugs and inconsistencies. Better support for attaching model metadata (parameters) to output attributes.

#### Internals

- On `Model` subclasses, change the container type of `met_variables`, `processed_met_variables`, and `optional_met_variables` from `list` to `tuple`. This helps avoid mutate these class variables, which we think of as "read-only".
- `Model` no longer sets a default empty value for `met_variables`. Instead, this attribute is not set.
- Delay construction of `cdsapi.Client` instance until needed in `ERA5` methods. Previously, the `cdsapi` client was constructed in the `__init__` method. Now it is constructed only if data is actually requested from the ECMWF server. This helps avoid unnecessary dependencies on the `cdsapi` package if `ERA5` only downloads from a cache.
- No longer attach the parameter `time` to `ERA5` instances. Now the `timesteps` variable is the only way to access the time dimension of the ERA5 request. This helps avoid some ambiguity.
- The `pcr.pcr` function now returns a triple of PCR, SAC, ISSR data.
- Rename `ERA5._datafile_cached` to `ERA5.is_datafile_cached`. This allows checking if a datafile is cached to be in the "public" side of the `ERA5` class.

#### Fixes

- Remove "dangerous default" value of `pressure_level=[-1]` in `ERA5.__init__`. Instead, use `-1` as the default value.
- The `contrail_properties.initial_persistent` function now return an array of floats. The previous `dtype` was `bool`. This is needed when the downwash data is reattached to the `Cocip.source` attribute. The reindexing introduces nans, and the previous implementation gave rise to a `pd.Series` with both boolean values and `nan`s. Consequently, pandas set the dtype to `object`. This is now fixed, and the `pd.Series` now has floating `dtype`.
- The `PCR` model maintains a consistent `dtype`. Previously the `dtype` was hard-coded to `np.float64`.
- Keep `xr.Dataset` attributes in calls to `xr.where`. Previously, attributes (such as `level` metadata) were dropped.
- Change `MetDataset` level unit attribute from `mb` to `hPa`.
- Update the `Emissions` model to set air temperature and specific humidity on the source parameter via met data. This makes the model more consistent with other pycontrails patterns and allows more seamless chaining of the `BADAFlight` and `Emissions` models.

#### Features

- Set attributes `pycontrails_version` and `met_data_source` on gridded model `xr.Dataset.attrs`.

## 0.25.0

Rewrite much of the `ContrailGrid` model for better grid-vector alignment.

#### Features

- Simply logic of `ContrailGrid`. Consolidate the `_eval_vector` and `_eval_grid` methods into a single `eval` method.
- The `ContrailGrid` model now strictly adheres to the `max_age` parameter. The previous implementation evolved contrails up to `met_slice_dt` beyond the `max_age` parameter.
- Change `ContrailGrid` time evolution to match `Cocip`. This is best explained by an example. Suppose a persistent contrail is initialized at time 8:12, and the model `dt_integration` is 10 minutes. The previous `ContrailGrid` implementation would evolve the contrail to times 8:22, 8:32, 8:42, .... The `Cocip` model evolves the contrail to times 8:20, 8:30, 8:40, .... The current `ContrailGrid` implementation now follows the `Cocip` model.
- Better `ContrailGrid` output variable handling. There are now two model parameters that control the verbosity of the output:
  - verbose_outputs_formation: This attaches additional variables to the object returned by the `eval` method. For example, this parameter can be used to attach "sac", "fuel_flow", "initially_persistent", and other formation-specific variables to the model output.
  - verbose_outputs_evolution: This attaches attributes `contrail_list` and `contrail` to the underlying `ContrailGrid` instance. This mirrors the corresponding `Cocip` implementation (which always attaches these attributes). (In the future, we may want to write the intermediate artifacts to disk, or use a "streaming" model.)

#### Breaking changes

- Change the default `ModelParam.met_time_buffer` from `(1 hour, 1 hour)` to `(0 hours, 0 hours)`. There didn't seem to be a good reason to use a nonzero buffer for time data. Both `Cocip` and `ContrailGrid` use the `cocip.downselect_met` function, which already utilizes the `max_age` and `dt_integration` parameters to precisely downselect met data to the exact time domain needed for CoCiP evolution. In light of this, the previous buffer of `(1 hour, 1 hour)` is not necessary.
- Remove `Flight` support from `ContrailGrid`. The model now treats `Flights` as if they are simply `GeoVectorDataset` objects. Most relevantly, unit tests did not regress after this change.

#### Internals

- Change docstrings for some functions returning `tuple` to follow [common numpy convention](https://numpydoc.readthedocs.io/en/latest/format.html#returns).
- Change parameter order in `GeoVectorDataset.downselect_met` and require buffer parameters to be keyword-only.
- In `Cocip.process_emissions`, only call `BADAFlight` and `Emissions` models is flight does not have the necessary variables. The same is true for the `calc_emissions` function in `contrail_grid`.
- The `cocip.downselect_met` function accepts `MetDataset` source (in addition to `GeoVectorDataset` source).
- Create `CocipTimeHandlingMixin` for `ContrailGrid` time handling. In this future, this could be used by the `Cocip` model for better met time handling control. This class attaches the `timedict` attribute to `ContrailGrid` and also loads `met` and `rad` data according to `met_slice_dt`. This class also includes better time validation for parameters that were implicitly assumed to take a certain form (e.g., `dt_integration` must be a divisor of `met_slice_dt`).
- Remove `log_call` decorators from `ContrailGrid`. These were somewhat clunky and make debugging harder.
- Remove `_grid_coords` and `_init_grid` methods from `ContrailGrid`. The first is essentially identical to `contrail_grid.source.coords`, and the second is handled by other helper functions that run with vector and grid `source`.
- Rename `mesh` variables to `vector` in the `contrail_grid` module.
- Reimplement the `combine_vectors` (previously `combine_meshes`) helper function. Contrails no longer need to have a common end of life in order to be concatenated. Moreover, this function now returns a generator. This allows for a slightly smaller memory footprint.
- `ContrailGrid` contrails now include "age" and "formation_time" variables.

## 0.24.1

Address corner case in `ContrailGrid._load_met_slices`.

#### Fixes

- The `ContrailGrid._load_met_slices` method no longer assumes `met` and `rad` data coincident with `met_slice_dt`. The new implementation is more robust to custom time domains on the met and rad data. NOTE: This may get overhauled once more in 0.25.0 or 0.26.0.

#### Breaking changes

- Change the `coordinates.slice_domain` implementation to not unnecessarily extend outside of the `domain` variable. See the elaborate doctest there.

#### Internals

- Warn the file does not exist in `DiskCacheStore.clear`.

## 0.24.0

Support `float32` dtypes in CoCiP models and make interpolation faster.

This release includes roughly half a dozen improvements, each of which provide a 3 - 10% speedup. The table below shows `ContrailGrid` benchmarks on 1440 x 641 x 13 x 1 source grid.

| version                  |  runtime (s) |
| ------------------------ | ------------ |
| 0.23.2, 64-bit precision |     1045.472 |
| 0.24.0, 64-bit precision |      890.066 |
| 0.24.0, 32-bit precision |      723.237 |

As a second means of comparison (less precise), runtime of the unit test are given below.

| version | n tests |  runtime (s) |
| ------- | ------- | ------------ |
|  0.23.2 |     675 |        94.97 |
|  0.24.0 |     715 |        74.01 |

#### Features

- Allow `Cocip`, `ContrailGrid`, `SAC`, `ISSR`, and `BADA` models to use and maintain the `dtype` as specified by the `source` parameter. The intention here is to allow `np.float32` or `np.float64` precision for model calculations. In practice, `np.float32` precision is slightly faster than `np.float64` precision, but the difference is not as large as I'd hoped. (Aside: the slowest step in running a large CoCiP simulation is a fancy indexing operation in interpolation (more than half of the runtime is spent on one line of code), and this indexing bottleneck is, in essence, independent of `dtype).
- Include a new `interpolation_localize` parameter on `ModelParams` to allow for automatic downselection in gridded interpolation. This seems to give a slight speed improvement in some contexts, but it is still somewhat experimental (turned off by default). I'm still not sure if this is always faster than the default behavior. When enabled, it shouldn't break existing models.
- Include custom implementation of the `scipy.interpolate.RegularGridInterpolator._evaluate_linear` method for faster interpolation.
- Allow `MetDataArray` interpolation on grids with singleton dimensions. Previously interpolation over any grid with a singleton dimension (ie, a single time value) would output `nan` values. Currently implementation squeezes the underlying grid as necessary to avoid this.
- Add `listdir` method to `CacheStore`. Only implement for `DiskCacheStore`.

#### Fixes

- Fix bug in handling corrupt data caused by partially downloaded netCDF from `GCPCacheStore`. Current implementation will now remove the corrupt file and redownload from `GCPCacheStore` (this was my intended implementation when I first encountered this).

#### Internals

- `ERA5.open_metdataset` maintains the `dtype` of the underlying netCDF grid. Previously all coordinate data was converted to `float64`. One difference introduced here is with pressure level data. The netCDF files downloaded from ECMWF have pressure level dtype ``int32``. These are now converted to ``float32``. No precision is lost in this conversion.
- Reimplement the `bada4.drag_coefficient_below_mach_threshold` function to avoid exponentiation. Previous implementation was far and away the bottleneck in BADA4 calculations. The new implementation is between 100x and 1000x faster for large input.
- Include slightly more performant implementation of `contrail_properties.ice_particle_terminal_fall_speed`. Previous implementation used one-sided bounds for `particle_mass`, resulting in some unnecessary calculations. New implementation uses two-sided bounds in constructing `alpha`.
- Consolidate all interpolation procedures into the `pycontrails.core.interpolation` module. Change `NoValidateRegularGridInterpolator` to `_PycontrailsRegularGridInterpolator` which is private to the `interpolation` module itself. Include a "public" `interp` function (which is only used in `MetDataArray.interpolate`).
- Overhaul the conversion of `datetime64` to float in gridded interpolation patterns. Previously there was an implicit `np.int64` -> `np.float64` cast which was not lossless. New implementation is more explicit with respect to the underlying `dtype` and documents lossy operations. (Aside: For the most part, there may be some lossy operations in gridded interpolation due to inherent differences in spatial and time coordinates (e.g. `np.int64` vs. `np.float64`). In practice, we end up discretizing the `time` coordinate to a millisecond resolution, which is sufficiently precise for our purposes.)
- Include additional warning if all interpolated values in `tau_cirrus` are `nan` (ie, all values are outside of the grid). This additional check is fast and helps avoid common mistakes that I've consistently encountered.
- Avoid `np.append` when possible. Instead, allocate an array of the correct size and fill it with data via slicing.
- When broadcasting numeric attributes to vector variables in `GeoVectorDataset`, try to convert to ``float32`` if possible.
- Read all BADA data into `numpy` as `float32`. This is controlled by a module level `DTYPE` variable that could be overwritten before `BADA` instances and initialized.

## 0.23.2

Remove `convert` module in favor of attaching functions as methods on relevant class.

#### Internals

- `convert.met_to_vector` -> `MetDataset.to_vector` (instance method)
- `convert.vector_splits` -> `VectorDataset.generate_splits` (instance method)
- `convert.met` -> `MetDataset.from_coords` (class method)
- Replace some `vector.py` and `met.py` function and class imports with module-level imports. This avoids circular imports and keeps a cleaner namespace in our core modules.
- Remove `convert.py` module.

## 0.23.1

#### Internals

- Separate `open3d` into its own optional installation. This is the only package that doesn't work with Python 3.10.

## 0.23.0

Simplify `ContrailGrid` implementation and improve `Emissions` interpolation.

#### Features

- Add "fuel_flow" as a supported variable in `ContrailGrid` verbose outputs.
- Add `shape` property to `MetDataArray` and `MetDataset`. `xarray` itself has no notion of `shape` on a dataset, and so the usage may be a bit confusing.
- Create `convert.py` module for converting between grid-like and vector-like data structures. This abstracts away some of the logic previously in `ContrailGrid.create_source`.
- Improve `Emissions` runtime performance by using `np.interp` instead of `xr.DataArray.interp`. This improves interpolation performance by about 500x. Overall, the full emissions pipeline on a large 25000 flight fleet decreased from about 8 minutes to 30 seconds in profiling of the changes. This is a significant improvement for running large simulations and pipelines.
- Allow `GeoVectorDataset` instantiation from `altitude_ft` column.

#### Internals

- Add `__contains__` method to `MetDataset` and remove `__iter__` method from `MetDataArray`. The desired behavior of `MetDataArray.__iter__` is not clear.
- Avoid raising a `ValueError` in `ContrailsGrid.__init__` when the `met` and `rad` dataset are already preprocessed. The class method `process_met_datasets` automatically determines if the datasets need processing, so this `ValueError` only created friction.
- Remove intermediate variable `fuel_dist` in both `Cocip` and `ContrailGrid`. This variable is only needed once in the construction of the downwash contrail, and it is redundant with `fuel_flow` and `true_airspeed`, which are more commonly used, more interpretable, and already attached to contrail `GeoVectorDataset` instances.
- Simplify logic in `BADAGrid.eval` by explicitly converting `MetDataset` source to `GeoVectorDataset`. This also avoided overwriting existing variables.
- Simplify logic in `ContrailGrid._generate_new_grid_vectors`.
- `ContrailGrid` no longer raises an error if the met data includes a "geopotential" variable but no "ciwc" or "tau_cirrus" variable.
- Continue to improve code quality in `emissions.py` and `ffm2.py` by replacing object-oriented code with functional code.
- Fix some TODOs in `test_met.py` related to `MetDataset.__setitem__`.

## 0.22.4

#### Breaking changes

- Deprecate the `GCPCacheStore.clear()` method

#### Internals

- Use `pyproject.toml` for `pytest` configuration instead of `setup.cfg`
- Add `filterwarnings error::UserWarning` to `pyproject.toml` configuration instead of command line arguments

## 0.22.3

#### Internals

- Update docstring and citations in `Cocip` class docstring.
- Rename `tests/validation` to `tests/regression` which is more descriptive.
- Removes `file` key from `contrails.bib` bibliography
- Bump Dockerfile python version to 3.9

## 0.22.2

Improve flight time validation in order to catch surprise errors occurring in `Flight.segment_groundspeed`.

#### Features

- Include `drop_duplicated_times` parameter in `Flight` class. If this parameter is enabled (default is `False`), duplicate times will be removed from the flight. Specifically, for each group of duplicate times, the first time will be kept and the rest will be removed.

#### Internals

- When instantiating a `Flight` object, warn if:
  - Any time value is `NaT`
  - Consecutive time values are not strictly increasing (this includes waypoints with duplicated times)

## 0.22.1

Possibly closes #64 by enabling `xarray.open_dataset` keyword arguments to propagate through a call to `era5.open_metdataset`. For example:

```python
>>> from pycontrails.datalib.ecmwf import ERA5
>>> era5 = ERA5(("2019-01-01", "2019-01-01T06"), "tsr")
>>> xr_kwargs = {"chunks": {"longitude": 360}, "parallel": False}
>>> rad = era5.open_metdataset(xr_kwargs)
>>> rad.data.chunks
Frozen({'longitude': (360, 360, 360, 360), 'latitude': (721,), 'level': (1,), 'time': (1, 1, 1, 1, 1, 1, 1)})
```

#### Breaking changes

- `ERA5.open_metdataset` (and similar methods) no longer take a `chunks` argument. Instead, these take an optional `xr_kwargs` arguments which allow more keyword arguments to be passed into `xarray.open_mfdataset`. Relevant to #64, this allows a "parallel" keyword argument to be passed into the `open_metdataset` method as a fix (see example above).

#### Internals

- Use a new `MetDataSourceWithDownload` class to avoid about 50 lines of repeated code in `ECMWFAPI` and `GFS`.

## 0.22.0

Major refactor to consolidate `GridModel` and `FlightModel` into one `Model`.

#### Breaking changes

- `GridModel` and `FlightModel` have been consolidated into a single `Model` type. `FlightModelParams` and `GridModelParams` are consolidated into `ModelParams`.
- All model `.eval()` methods take a single keyword `source`. Each model supports different input types for `source`, but in general models accept `Flight`, `MetDataset`, or `GeoVectorDataset`.

#### Features

**Met**

- Adds `get`, **setitem**` and `update()` methods to `MetDataset`. These methods will warn when overwriting keys in the underlying dataset. Both methods support assignment with a`MetDataArray` for convenience.
- Adds `size` property to `MetDataset` and `MetDataArray` to return the number of grid points in the underlying data.
- Adds `values` property to `MetDataArray` as a shortcut to the underlying `data.values` property.

**Models**

- Models have a new property `met_required` the determines if `met` input is required to model constructor.
- Adds `Model.set_source(source)` method that standardizes how input sources are attached to attribute `self.source`. In particular, all `Flight` sources will be assigned a broadcast `flight_id` if no `flight_id` is present.
- Consolidates `Model.intersect_met_variables` and `Model._intersect_met_variable` into `Model.set_source_met(optional=..., variable=...)`
- Add method `Model.get_source_param` that allows the model to look for source data in the following order (1) `source.data`, (2) `source.attrs` (3) `Model.params`
- Add module method `model.interpolate_met` as a shortcut to interpolate `GeoVectorDataset` vector data against `MetDataset` gridded data.
- Ensure all models use input source data variables *before* calculated or param data. The user will be warned if source data is updated unexpectedly.

**Utils**

- Adds `typing.type_guard` util to type guard a variable with a standard error message.

#### Fixes

- Update `GeoVectorDataset.downselect_met` to properly handle single level (`level` = [-1]) met datasources.

#### Internals

- Renames `VectorAttrDict` -> `AttrDict`
- Remove support for `n_rollouts` and `uncertainty_params` in `ContrailGrid`.
- Add `-W error::UserWarning` flag to `pytest` make recipe to ensure that all tests emitting a `UserWarning` are caught.
- Add `__contains__` method to `VectorDataset`. This allows key lookups in constant time (as opposed to the linear time method implicitly used through `__iter__`).
- Change order in which `VectorDataset.__add__` handles empty values. The left-hand addend now takes priority.

## 0.21.0

Small updates to `GeoVectorDataset` and `VectorDataset` to warn users when overwriting data.

#### Breaking changes

- Remove `Contrail` data model from the repository

#### Features

- Adds new types for `VectorDataset.data: VectorDataDict` and `VectorDataset.attr: VectorAttrDict` that will warn when overwriting attributes. These dictionaries override `setdefault` to have slightly more ergonomic behavior for the `data` and `attrs` respectively.
- Adds `VectorDataset.update()` method to override a `data` value without warning. `VectorDataset.attrs.update()` and `VectorDataset.data.update()` works the same.
- Adds `VectorDataset.setdefault()` method as shortcut to `VectorDataset.data.setdefault` and validate array size after setting.
- Add `wingspan` param to `ContrailGrid`
- Allow direct conversion between `VectorDataset` types (e.g., `fl = Flight(...); vector = GeoVectorDataset(fl)`).
- Allow equality of `nan` values in `VectorDataset.__eq__`.
- Add `VectorDataset.sum` class method as a shortcut to `np.concatenate` over data variables.

#### Internals

- Update `Cocip` warning for non-sequential "waypoint" keys to be an `ValueError`
- Allow `Cocip` models to be run multiple times with different flights by comparing met/rad modifications
- Allow update to `VectorDataDict` when all underlying arrays have length 0. This could be error prone in a `GeoVectorDataset` if an empty dataset is created, then only specific coordinate keys are set.
- Elevate warnings in tests to errors using `@pytest.mark.filterwarnings("error")`.
- Warn if `Flight` data is sorted in constructor. It's rare for flight data not be be sorted by time, and it can lead to difficult to debug errors when working with different `VectorDataset` types.

## 0.20.4

Support for disabling contrail filtering in CoCiP models.

#### Breaking changes

- The cocip module `calc_timestep_geometry` now sets any nan values in `segment_length`, `cos_a`, and `sin_a` to 0. This ensures that they do not contribute to energy forcing and prevents the "propagation" of nan values in an evolving contrails.
- Set missing values to 1 in segment ratio calculation in the `contrail_properties` module.

#### Features

- Add `filter_initially_persistent` parameter to `CocipParams`. This works analogously to `filter_sac` in filtering the `downwash_flight` variable.
- Add `persistent_buffer` parameter to `CocipParams`. This allows `Cocip` to continue to evolve non-persistent waypoints beyond their end of life. Not (yet) implemented in `ContrailGrid`. When running `Cocip` with a non-default `persistent_buffer`, an additional `end_of_life` variable is attached to each evolving `Contrail` instance.

#### Fixes

- Fix tiny `mypy` issue caused by new version of `dask`.

#### Internals

- Rename `_sg_filter` parameter from `period` to `window_length` to be consistent with `scipy.signal.savgol_filter` implementation.
- `CocipFlightParams` derives from `BADAFlightParams` instead of `FlightParams`. This is consistent with `CocipParams` deriving from `BADAModelParams`.
- Smoothing parameters used in computing true airspeed are now included in `CocipFlightParams`. Default values are consistent with the default values on the `Flight.segment_true_airspeed` method.
- The cocip module `calc_continuous` function now checks for identical flight ID in addition to consecutive waypoints. This check is necessary in "fleet mode" and has no effect in "flight mode".
- Implement slighty more efficient `dz_max` calculation by computing weakly stratified displacement only on waypoints that are weakly stratified.

## 0.20.3

Hot fix for bug introduced in 0.20.2.

#### Fixes

- Fix bug in `contrail_grid.result_to_metdataset` in which a `pd.Series` was not instantiated with the required index.

#### Internals

- Clean up patterns in function `contrail_grid.calc_intermediate_results`.

## 0.20.2

Fix emissions interpolation bug, fix `ContrailGrid` `VectorDataset` alignment issue, and allow `VectorDataset` constructor more flexibility with array-like input.

#### Fixes

- The `Emissions.get_nvpm_emissions_index_edb` method previously raised a `ValueError` when the `t4_t2` variable contained all nan values. Such a situation arises when any of the input arrays `true_airspeed`, `fuel_flow_per_engine`, or `air_temperature` themselves contained all nan values. Instead of raising a `ValueError`, this method now returns arrays of all nan for `bc_ei_n` and `bc_ei_m`.
- Fix subtle bug related to the output of `contrail_grid.calc_intermediate_results` function. Previously, this function instantiated a `VectorDataset` with both `np.array` and `pd.Series` data variables. The `pd.Series` variables give rise to unnoticed alignment issues when the `ContrailGrid` model re-shaped 1-dimensional output to a 4d grid. I'm not sure of the extent to which bug impacts model output without more integration-style testing.

#### Features

- `VectorDataset`s and subclasses can be constructed from dictionaries with array-like values rather than strict `np.array` values. For example, `vector = VectorDataset({"a": [3, 4, 5], "b": [3.3, 2.2, 1.1]})` is now possible. Internally, all type annotations remain strict, requesting `dict[str, np.ndarray]` for the `data` variable. This change mainly allows for more seamless construction of `VectorDataset` instances in unit tests and example notebooks.
- Similarly, the `VectorDataset.__setitem__` now handles array-like input correctly. For example, `vector["c"] = (0.5, np.nan, 0.3)`.

#### Internals

- Remove the `VectorDataset._copy_data` method.
- It is no longer possible to construct to a `VectorDataset` with `data` attribute of the form `dict[str, pd.Series]`. Such a data structure will be converted to a `dict[str, np.ndarray]` upon initialization.

## 0.20.1

Enable better support for running `ContrailGrid` with verbose outputs.

#### Fixes

- The `ContrailGrid` model now uses the "shift_radiation_time" parameter for in its `process_met_datasets` method. Previously, the default value of 30 minutes was hard-coded in various places. This changes supports running `ContrailGrid` with ERA5 met ensemble members.

#### Features

- The `ContrailGrid` model now supports a `"verbose_outputs"` parameter. This parameter can be a`bool` or a `list` of `str`s. If `True`, all supported verbose variables will be kept. If`False`, no verbose variables will be kept (default behavior). If a`list[str]` is passed, only variables from the `list` are kept. Currently, `ContrailGrid` supports verbose outputs for the following variables:
  - "sac"
  - "persistent"
  - "T_crit_sac"
- The `MetDataset` returned by `ContrailGrid.eval` now contains between 2 and 5 variables, depending on the `"verbose_outputs"` parameter.

#### Internals

- The `contrail_grid` functions `find_initial_contrail_regions` and `find_initial_persistent_contrails` now return `tuple`s whose first entry is the previously returned `Contrail` instance and whose second entry is raw output from an intermediate calculation of interest. This second entry is used when running in "verbose" mode.
- A new `_run_downwash` helper function is added to `contrail_grid` to support the calculation of downwash. All verbose outputs are now handled within this function.
- Additional tests for verbose outputs.

## 0.20.0

Update docstrings to be fully `pydocstyle` compliant and mostly `darglint` compliant.

#### Fixes

- Find and fix bug in `pycontrails.models.emissions.ffm2._estimate_specific_humidity`. Previously, the `denom` variable was defined by `denom = air_pressure_hpa - rh - P_sat`. The second `-` was change to `*` as in equation (45) in DuBois & Paynter (2006).

#### Internals

- Refactor `pycontrails.models.emissions.ffm2` to take a more functional approach, avoiding repeated code. Use existing functions from `pycontrails.models.bada.common` to calculate temperature and pressure ratios rather than supporting two identical parallel implementations.
- Remove classes `Hydrocarbon` and `CarbonMonoxide` from `pycontrails.models.emissions.ffm2`. Both were identical to the base class `COorHC`.
- Require additional 3rd party library [`overrides`](https://github.com/mkorpela/overrides) to programmatically copy a docstring from the method from a parent class to a child class in the case that the child's method does not itself have a docstring. This change allows us to avoid repeating identical docstrings in the child classes while maintaining `pydocstyle` compliance. See the [discussion surrounding this issue](https://github.com/PyCQA/pydocstyle/issues/309). Another option is to add `# noqa: D102` to any method on a child class for which the parent class docstring is to be used.
- ~Remove `autodoc_inherit_docstrings = False` in `pycontrails/docs/conf.py`. Child methods now having a docstring will now show the docstring from the parent class in the built documentation.~ This happens automatically with the `@overrides` decorator.
- Remove identical or nearly identical docstrings that were duplicated in child classes.
- Address many `darglint` suggestions in `pycontrails.core` and `pycontrails.models`. The `darglint` checker is slow to run and somewhat opinionated, and so I think it should not be added as a dev-dependency.
- Move `thermo.mach_number` to `units.tas_to_mach_number`. Implement inverse function `units.mach_number_to_tas`.
- Address all internal warnings seen when running `pytest` over `pycontrails`. Specifically, acknowledge such warnings with patterns such as `with pytest.warns(...):`.

## 0.19.1

Implement `verbose_outputs`, `process_emissions`, and `filter_sac` model parameters on `ContrailGrid`. These parameters are needed for two-stage prediction prototyping and for constructing `ContrailGrid` animations with intermediate artifacts. These updates are likely not yet finalized, but they allow for greater `ContrailGrid` flexibility.

#### Breaking changes

- The `GeoVectorDataset` constructor now raises an error if instantiating from a `DataFrame` whose `"time"` column is timezone aware. Such data is not cast to `np.datetime64` with the `to_numpy` method. The user should address this upstream of the `GeoVectorDataset` data structure. The error messages are informative, giving suggestion on how to address the issue on the `DataFrame` object itself.

#### Features

- Implement `process_emissions` flag in `ContrailGrid`. Processing emissions can be skipped in "path mode" when the input data has all required emissions variables. This mirrors the implementation in `Cocip`. In "grid mode", the `"process_emissions"` flag applies the `default_bc_ei_n` value to each gridpoint.
- Implement `verbose_outputs` flag in `ContrailGrid` when running in "path mode". This is not yet implemented in the "grid mode". Memory usage is a limiting factor when running gridded models with verbose outputs and a small `dt_integration`. To alleviate this bottleneck, we could use a tool like `dask` to cache intermediate artifacts to disk (instead of maintaining in memory). This effort would be very involved.
Implement `filter_sac` flag in `ContrailGrid`. This is enabled for both "path mode" and "grid mode". This mirrors the implementation in `Cocip`.
- The [`docs/examples/flight.csv`](docs/examples/flight.csv) now contains waypoints showing both ISSR and SAC. This flight data was pulled from the OpenSky database. Reproducible instructions can be found in the [REAMDE](docs/examples/README.md). This README now includes more informative documentation for the example notebooks.
- Include a new `ContrailGrid.ipynb` example notebook showing core uses cases for `ContrailGrid`.

#### Fixes

- Fix bug in the `ecmwf` module in which the function `any` was forgotten. This caused a `ValueError` to be raised when data was not previously cached. Interestingly, because we don't actually test `ecmwf` requests in our unit test, I was only able to see this by running the notebook tests. If `pycontrails` has wider use, we may want to include a lightweight request similar to the [`cdsapi` example](https://github.com/ecmwf/cdsapi/blob/master/example-era5.py).
- Before running any notebooks, the `nbtest` `Makefile` recipe downloads all ERA5 meteorology data needed throughout the notebooks. This ensures that repeated calls to our `ERA5` API don't overwrite cached data needed elsewhere. Calls to `ERA5` within each notebook can now be "narrow" (ie, only requesting variables that are needed for the particular use-case) because all data is cached on disk up front.
- All example notebooks have been re-run to address API changes. The only nontrivial failure I noticed is that the `ecmwf.ipynb` MARS HRES request fails for historical data involving the `"ciwc"` variable. However, the corresponding request for more recent data (last 48 hours) seems to work. This is documented in the notebook itself.
- Fix some situations in which the `copy` flag was ignored in `VectorDataset` patterns. In short, numpy arrays were copied despite setting `copy=False`.
- Extra attention is taken with `"time"` data in `VectorDataset` patterns. In particular, the `create_empty` method now creates a `"time"` variable with the "correct" `dtype` of `np.datetime64`. Previously, the default `dtype` of `np.float64` was used.

#### Internals

- Remove default value for `supported` in `datalib.parse_variables`. This wasn't being used, and the default value would lead to a degenerate situation in which no variables are supported.
- Update and document implementation of the `ice_particle_activation_rate` without changing the external behavior.
- Replace `ContrailGrid` met handling errors with warnings. Similarly, warn user if the `MetDatasets` should be downselected in the horizontal domain (longitude, latitude) when running `ContrailGrid.`
- Additional logging in memory intensive places in `ContrailGrid.path_eval`.
- Use `np.concatenate` instead of `pd.concat` in `ContrailGrid.path_eval` for the sake of memory.

## 0.19.0

Add support for [GFS Forecast](https://registry.opendata.aws/noaa-gfs-bdp-pds/)

#### Breaking changes

- Informal ECMWF variable names are no longer supported. Previously, `temperature` worked as a variable input. Now the full `standard_name` must be used `air_temperature`.
- Rename `HRES` attribute `basetime` to `forecast_time`.

#### Features

- Add support for [GFS Forecast](https://registry.opendata.aws/noaa-gfs-bdp-pds/) data management
- Add `MetVariable` dataclass to describe meteorology variables used throughout the repository.
- Add `pycontrails.core.met_var` module that contains all common met variables.
- Add `MetDataSource` template for all datalib meteorology datasources.
- Support alternate `GFSForecast` variables interchangeably in `Cocip` and `ContrailGrid` model
- Add `hash()` method for `MetDataSource` classes

#### Fixes

- Update `Dataset.drop` -> `Dataset.drop_vars` to fix API change in `xarray`
- Import optional imports within `__init__()` methods to actually make optional.

#### Internals

- Abstracts calculation of shortwave, longwave, and radiation to functions in `cocip` module. ECMWF and GFS provide radiation data differently, so these new functions will handle data based on the variables present. See `cocip.calc_shortwave_radiation` and `cocip.calc_outgoing_longwave_radiation`.
- Abstracts `Cocip._interpolate` method into a `cocip._interpolate` function in the module.
- Refactor `MetDataset.ensure_vars(vars)` to operate on a `list[MetVariable]` or a `list[list[MetVariable]]` instead of a `list[str]`. If an element in `list[list[MetVariables]]`, only one variable will be required. The method now returns a list of string keys that are found in the `MetDataset`.
- Refactor `Models.met_variables` attribute to be a list[`MetVariable`] or a list[list[`MetVariable`]]. If an element in list[list[`MetVariables`]], only one variable will be required.
- Refactor all references to `_parameter` or `_PARAMETER` to `_variable` or `_VARIABLE`
- Update `mypy` version
- Run `black` against notebook examples, add `make nbblack` to check if notebooks are formatted correctly. Notebooks are added to the `black` commit hook for reformatting. The `nbblack` recipe is run in `test`.

## 0.18.1

Generalize `Fuel` support for future additional of alternative fuel types.

#### Features

- Add a `Fuel` dataclass with `JetA` implementation to handle different fuel types. `Emissions` and `BADA` models updated to use new `Fuel` class, either from the `flight.fuel` attribute, or from a grid model parameter.

#### Internals

- Refactor `test_sac_issr` to include all `FlightModel` tests related to these modules. Removes `test_flight_intersect` file.

## 0.18.0

Ensure tighter parity between the `Cocip` and `ContrailGrid` models.

### Breaking changes

- The CoCiP continuity convention for segment angles is slightly adjusted. Previously, discontinuous waypoints were assigned segment angles with `sin_a = 0` and `cos_a = 1`. (Such a segment points due-east, and there is no intrinsic reason to prefer a east-pointing segment over any other direction.) With this update, discontinuous waypoints are assigned `sin_a = 0` and `cos_a = 0`. These new values are more closely aligned with the rational of underpinning the CoCiP continuity convention: The purpose of this convention is to avoid applying any segment-derived quantity on a partially initialized contrail segment (ie, if the initial waypoint of a contrail segment has begun evolution but the terminal waypoint has not). Note that trig values `sin_a` and `cos_a` are only used to compute the wind shear normal `dsn_dz`. When `sin_a` and `cos_a` are both 0, the corresponding wind shear `dsn_dz` is also 0. Consequently, wind shear from a partially initialized segment no longer contributes to plume geometry. Note that this change only has a very slight impact on `Cocip` outputs (see changes in `tests/unit/static/cocip*.json`), and the impact diminishes as `dt_integration` tends toward 0 or as the flight is resampled with additional waypoints. Once both segment endpoints have "come online", the continuity convention is no longer relevent and `sin_a` and `cos_a` are computed as usual.
- Remove the `rf_net_sum` variable from `ContrailGrid` outputs. When running in *grid* mode (the default mode), `ContrailGrid` returns a `MetDataset` with `contrail_age` and `ef_per_m` variables. When running in *mesh* or *flight* mode, `ContrailGrid` maintains a copy of all intermediate artifacts for post-mortem analysis.
- Remove `CONTINUITY_CONVENTION_PARAM` from the `Cocip` model. Simplify the logic involved in `continuous & persistent` calculations. This was done without changing the `tests/unit/static/cocip*.json` outputs.

### Features

- The `Cocip` model is somewhat more functional, now calling the functions `calc_contrail_properties` and `calc_radiative_properties` in the `cocip` module. These functions were brought over from the `contrail_grid` module. This approach ensures closer parity between `Cocip` and `ContrailGrid`, avoids the possibility of object-oriented side effects (less stateful), and avoids the logic that was previously repeated separately in the `cocip` and `contrail_grid` module. In the same vein, the new functions `calc_continuous` and `calc_timestep_geometry` are based on the previous `Cocip` methods of the same name. This move to a more functional approach may lead to easier to maintain models. Presently, the `Cocip` methods `_calc_timestep_meteorology` and `_calc_timestep_contrail_evolution` are not as closely aligned with functions in `contrail_grid`, and so these have not been reorganized.
- The `ContrailGrid` model now employs a shifted `dt` for the first wind advection step of the initial contrail segment. The purpose of this change is create closer parity with advection methodology in `Cocip`. In the `ContrailGrid` implementation, each infinitesimal segment includes both a "head" and "tail" longitude-latitude pair. If we imagine a flight traversing the segment between the head and the tail, there is a small but meaningful time delta as the flight moves. While this time delay is baked into the current `Cocip` implementation (roughly, as new contrail waypoints "come online", the `dt` for the first advection step is `advection_time - formation_time`, which is NOT constant), it was lacking in the `ContrailGrid` implementation. This update eliminates the main source of disagreement between `Cocip` and `ContrailGrid`.

### Fixes

- Stricter checking for `specific_humidity` adjustment on the `met` parameter in `Cocip` and `ContrailGrid`. Each model deal with humidity enhancement slightly differently (`Cocip` scales the entire specific humidity `MetDataArray`, whereas `ContrailGrid` applies the enhancement only to the grid points under consideration). Additional checks are applied in `Cocip.__init__` and `ContrailGrid.__init__` to ensure the user does not mistakenly adjust humidity twice.
- Fix a bug in `ContrailGrid` in which specific humidity was overwritten without any `rhi_adj`-enhancement. Now the `run_interpolators` function will not override any existing variables on the parameter `mesh`.
- Fix bugs in `contrail_grid.find_initial_persistent_contrails` (previously `contrail_grid.calc_first_contrail`) in which variables from the downwash contrail were mistakenly used instead of corresponding variables on the grid points. The implementation is now completely aligned with the analogous `Cocip` implementation.
- The initial stages of `Cocip` and `ContrailGrid` are now strictly unit tests in `test_cocip_contrail_grid_parity` to ensure identical initialization of initial contrails. Now the main source of model divergence is the CoCiP continuity convention.
- Reverse the segment head and tail in `contrail_grid.calc_wind_shear`.
- Fix a bug in `MetDataset.downselect` arising when the underlying dataset included negative pressure levels (ie, single level or surface level datasets).

### Internals

- More consistent variable, function, and method names between `Cocip` and `ContrailGrid`.
- `ContrailGrid` now defines a `contrail_list` and `contrail` attribute when running in *mesh* or *flight* mode. This allows for better examination of intermediate artifacts, mirroring the analogous logic from `Cocip`. The `ContrailGrid.from_flight` methods should be preferred over `Cocip` when the user seeks to disregard CoCiP continuity conventions.
- Avoid redundant computation of `segment_length` in `ContrailGrid`. Now `segment_length` is attached as a variable on the contrail undergoing evolution.
- Additional reliance of `BADAFlight` methodologies when running `ContrailGrid` in *flight* mode. This ensures identical BADA-derived quantities such as fuel flow, aircraft mass, and engine efficiency between the `Cocip` and `ContrailGrid.from_flight` models.
- Slightly enhance logging in `cocip` and `contrail_grid` modules.
- Generalize `create_empty` method from `GeoVectorDataset` to `VectorDataset`.
- `ContrailGrid` now calculates the `T_crit_sac` variable after filtering by SAC. This is slightly more performant than computing it before filtering.

## 0.17.1

#### Fixes

- Fix bug in `Emissions` model when certain aircraft types have duplicate `log_fuel_flow` coordinates in the `co_hc_emissions_index_profile`
- Fix `mypy` issue arising from upgrade to version 0.94.

#### Internals

- `BADAGrid.eval` proceeds in batches rather than pumping (possibly) large numpy arrays through the low level `BADA` functions all at once.
- `ContrailGridParams` include `nominal_engine_efficiency` and `nominal_fuel_flow`. These values override those computed through `BADAGrid`.
- Clip uncomfortably long attribute values in `VectorDataset.__repr__`.
- Attach ROCD to `AircraftPerformance`, allowing BADA methods to return ROCD data.

## 0.17.0

#### Breaking changes

- Update default CocipParams `rhi_adj` and `rhi_boost_exponent` for the latest results from Teoh 2022. Static data updated in (6f4934f) to reflect the parameter changes.

#### Features

- Enable `r_eff` scaling to be overridden in `radiative_forcing` methods. This allows for `r_eff` sensitivity exploration in `radiative_forcing` methods. The `Cocip` model does not yet have hooks to change the `r_eff` parameterization.
- Add Cocip parameter `filter_sac` to stop the Cocip model from filtering waypoints that don't satisfy SAC. SAC data is still calculated, but the waypoints are not filtered out.
- Add Cocip parameter `habit_distributions` to customize the habit weight distributions used in the radiative forcing parameterizations. This is primarily used for testing sensitivity. Moves `RFConstants.habit_weights`, `RFConstants.radius_threshold_um`, `RFConstants.habits` into `CocipParams`.
- Allow CocipUncertaintyParams to be set to `None` to keep the parameter set to its default value while using the uncertainty generator.

#### Fixes

- Clean up `Cocip` uncertainty parameters and update default distributions per the most recent iteration.
- Remove `r_ice_vol` scaling factor

#### Internals

- Add bibliography instructions to documentation
- Update MR template with tests. Comment out `pydocstyle` test for now.
- Include `D202` in pydocstyle ignore (no new line after docstring)

## 0.16.0

Fine tune low level BADA models and create BADA interface for gridded models.

### Breaking changes

- Many of the low-level BADA functions and methods have changed.
- `ContrailGrid` interacts with the `BADAGrid` API.
- `CocipParams` inherits from `BADAParams`.
- `PTFParams` now derives nominal values for climb and descent fuel flow values.

### Features

- The `BADA3` and `BADA4` method `calculate_aircraft_performance` (this should be thought of as the core BADA method) now takes in additional parameters: `correct_fuel_flow` and `model_choice`. These allow for fine-grained control over the BADA internal methodology. High level interfaces to BADA (such as `BADAFlight` and `BADAGrid`) include reasonable default values for these parameters (`True` and `"total_energy_model"` respectively).
- `BADAGrid` is a new grid-specific interface to low level BADA computations. Both `BADAFlight` and `BADAGrid` call `calculate_aircraft_performance`. The critical difference here is that `BADAFlight` makes use of the `time` parameter whereas `BADAGrid` does not. This dual-use of `time` is now explicitly (previously this was a hack).
- Enhanced fuel flow correct in BADA models.

### Fixes

- BADA3 fuel flow predictions significantly improved when validated against flight recorder data.
- `BADA3` and `BADA4` models avoid repeated computation of frequently used arrays.
- `BADA4` is now fully documented.

### Internals

- BADA4 matrix multiplications previously done by hand are now vectorized with `np.einsum` and `np.vander`. There are many places in the BADA4 formula in which 6th degree Taylor polynomials in two or three variables are used. The function `_calc_vandermonde_product_sum` in the `bada4` module achieves this and is unit tested to ensure agreement with previous implementation.
- Low level BADA computations are now completely functional. Previously the `bada3` module included a `TotalEnergyModel` class and the `bada4` module included a `JetActions` class. The object oriented patterns was ripped out to avoid unintended side-effects and to make the BADA models more explicit.
- Additional fields on some of the lightweight bada-specific `dataclasses`.
- Closer parity between low level `bada3` and `bada4` methods.

## 0.15.3

Avoid overwriting flight segment-specific variables when `Cocip.eval` is called.

### Features

- The flight variables `"true_airspeed"`, `"segment_length"`, and `"cos_a", "sin_a"` are no longer overwritten if they already exist on the parameter `flight` in `Cocip.eval`.

### Internals

- Found and fixed a bug in `test_cocip.py` with the `cocip_no_ef` fixture.
- Remove `fl.copy` in several places `test_cocip.py`. The `fl` fixture has `"function"` scope.

## 0.15.2

Fix small bug arising from incompatible array shapes in radiative forcing calculations.

### Fixes

- Resize `tau_cirrus` in the `shortwave_radiative_forcing` function in order for compatibility with other arrays.

### Internals

- Attach azimuth to `ContrailGrid` output.

## 0.15.1

Add support for radiation uncertainty parameters in `Cocip` and `ContrailGrid` models.

### Features

- Adds `r_ice_vol_enhancement_factor`, `rf_sw_enhancement_factor`, `rf_lw_enhancement_factor` parameters to `Cocip` model to support uncertainty investigations. The `r_ice_vol_enhancement_factor` scales the mean ice particle radius before calculating habit weight regime and effective radius `r_eff`. The `rf_sw_enhancement_factor`, `rf_lw_enhancement_factor` scale shortwave and longwave radiative forcing, respectively.

#### Internals

- Refactor the `radiative_forcing` module for performance. SW/LW radiation calculations are masked to run only where habit weights > 0 and SDR > 0 (for shortwave forcing).
- Add `solar_constant` to `constants` module with [equation reference](http://solardat.uoregon.edu/SolarRadiationBasics.html)
- Adds a Cocip performance profiling test that is always skipped. Included as a reference implementation.
- Removes `contrail_albedo` test since these are all now 1D vector calculations
- Updates `contrails.bib` bibliography for references in documentation

## 0.15.0

Enable BADA3 support in `ContrailGrid` and define the priority of BADA3 and BADA4 within `Cocip` and `ContrailGrid`.

### Features

- Enable `BADA3` support in `ContrailGrid`.
- Include `bada_priority` model parameter in `BADAFlightParams` and `CocipParams`. This parameter has default value `4`, indicating that model should query the BADA4 database for aircraft data before falling back to the BADA3 database.

### Fixes

- The BADA energy model method for calculating Mach number has abstracted to the function `pycontrails.models.bada.common.calculate_mach_number`. This new function consistently clips the maximum Mach number at the BADA-derived theoretical limit. The purpose of these changes is to avoid unrealistic engine efficiency values, which greatly impact SAC and CoCiP model output. This changes the previous limits for unrealistic Mach numbers, which were substantially larger than the BADA maximum.
- Fix a small bug in `ContrailGrid` in which the model exits early in `"path"` mode when no grid point satisfies the SAC or forms initial contrails. In this case, the `evolve_mesh_path_mode` now returns a `VectorDataset` with the correct keys.
- Correctly scale energy forcing from a nominal segment length to a trajectory segment length when `ContrailGrid` runs in "flight" mode.

### Internals

- The gitlab CI downloads BADA files from GCS for running tests.
- Include `seaborn` in "vis" dependencies. This is needed to ensure the `nbval` tests pass.
- Allow the default `BADA3_PATH` and `BADA4_PATH` in the `tests` directory to be overridden by the environment variable `"BADA_CACHE_DIR"`. This is used by the gitlab CI.
- The `contrail_grid` module now directly interacts with the `BADA.energy_model` method, avoiding the need to pass in the `n_iter` parameter. This should be replaced by a general purpose grid-oriented energy model.

## 0.14.0

Small fixes to codebase, documentation, build, and CI based on initial tutorials.

### Breaking changes

- Drop support for Python 3.7

### Fixes

- `Cocip`: Rename `rf_net` -> `rf_net_1` in contrail evolution for consistency
- `Cocip`: Remove `thrust` from required flight variables. This is only required for aircraft performance processing, not in the Cocip algorithm.

### Internals

- Start using `CHANGELOG.md` as MD file in the top of the repository. Referenced in sphinx using [myst-parser](https://www.sphinx-doc.org/en/master/usage/markdown.html).
- Run black formatting over whole repository using black version 22
- Upgrade minimum dependency versions setup.cfg primarily for typing compatibility
- Fix CI tests and temporarily suspend doctests in CI pipeline.
- Fix doctest precision issues by using the [pytest `NUMBER` option](https://docs.pytest.org/en/6.2.x/doctest.html#using-doctest-options)
- Remove Python 3.10 from CI testing for now

<!-- 
FORMAT

### Breaking changes

### Features

### Fixes

### Internals
-->
