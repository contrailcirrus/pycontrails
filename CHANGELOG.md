# Changelog

## 0.54.10

### Features

- Output from the `DryAdvection` model will include a `flight_id` column when run with a `Flight` or a `Fleet`.

### Fixes

- GOES data download for GOES-East now supports the GOES-16 -> GOES-19 transition. The default GCS bucket is automatically selected based on the requested date (using 2025-04-04 as the cutoff). For pre-transition data, this change is backwards compatible with previous versions of pycontrails.
- Pass the `GOES.fs` instance to the `gcs_goes_path` function from the `GOES.gcs_goes_path` method to avoid repeated GCS file system client instantiation.

### Internals

- Fix errors in "See Also" sections of `Model` docstrings.

## 0.54.9

### Features

- Include new `goes.correct_parallax` function in the `GOES` datalib to correct for parallax artifacts in GOES imagery. This function can be used to better superimpose flight and contrail data over GOES imagery.

### Breaking changes

- Air temperature is now an optional met variable in `PSFlight`. Code that uses `PSFlight.met_variables` may need to be updated.
- The `ensure_true_airspeed_on_source` method of the `AircraftPerformance` class now operates fully in-place and no longer returns a value. Code that relies on a return value may need to be updated.
- The `PSFlight.eval_flight` method now retrieves all inputs to performance calculations from the `Flight` parameter rather than the `PSFlight` model source. Code may need to be updated if constants for `aircraft_mass`, `thrust`, `engine_efficiency`, or `fuel_flow` were stored as attributes in a `Fleet`, as these attributes are lost when a `Fleet` is converted to a `Flight` list prior to calling `eval_flight`.

### Internals

- Define `default_parameters`, `met_variables`, and `optional_met_variables` for the `AircraftPerformance` abstract base class.
- Remove the definitions of `met_variables` and `optional_met_variables` from `PSFlight`. These are now inherited from `AircraftPerformance`.
- Retrieve aircraft mass, thrust, engine efficiency, and fuel flow from `Flight` parameter rather than model source in `PSFlight.eval_flight`.
- Automatically run the [pycontrails-bada test](https://github.com/contrailcirrus/pycontrails-bada/actions/workflows/test.yml) workflow after publishing pycontrails release to PyPI.

## 0.54.8

### Features

- Update the Poll-Schumann Aircraft Performance model:
  - Support the Support Boeing 737 MAX 10 aircraft type.
  - Update aircraft maximum landing weight, maximum zero fuel weight, operating empty weight, and maximum payload within the static CSVs for existing aircraft types as recommended by Ian Poll.
- Add `ExponentialBoostLatitudeCorrectionHumidityScaling` calibrated for model-level ERA5 data.

### Internals

- Update the `pycontrails.physics.static` CSV files to include newly released global and regional passenger and cargo load factor data from IATA (Oct-2024 to Dec-2024).
- Attach `n_ice_per_m_0` and `f_surv` to the downwash flight computed in the `Cocip` runtime. This data is now saved as part of the `Cocip` output.
- Rename and modify `contrail_properties.ice_particle_number` to `contrail_properties.initial_ice_particle_number`.
- Rename `ValidateTrajectoryHandler.CRUISE_ROCD_THRESHOLD_FPS` -> `ValidateTrajectoryHandler.ROCD_THRESHOLD_FPS`. Update its value from 4.2 ft/sec ->  83.25 ft/sec.
- Remove the altitude filter on the `ValidateTrajectoryHandler` ROCD check. Now all waypoints are checked for ROCD violations.
- Correctly parse "DOF" (departure date) field from flight plan in the `flightplan` module.

## v0.54.7

### Features

- Add helper `classmethods` to `Model`, `Cocip`, and `CocipGrid` for generating lists of required variables from specific data sources.
- Add a `ValidateTrajectoryHandler` to the `spire` module to validate spire ADS-B data. This work is experimental and will be improved in future releases.
- Update Unterstrasser (2016)'s parameterised model of the contrail ice crystal survival fraction to the latest version (Lottermoser & Unterstrasser, 2025). This update:
  - improves the goodness of fit between the parameterised model and LES, and
  - expands the parameter space for application to very low and very high nvPM inputs, different fuel types (where the EI H2Os are different), and higher ambient temperatures (up to 235 K) to accommodate for contrails formed by liquid hydrogen aircraft.

### Breaking changes

- The `MetDataset.standardize_variables` method now returns a new `MetDataset` rather than modifying the existing dataset in place. To retain the previous behavior, use `MetDataset.standardize_variables(..., inplace=True)`.

### Fixes

- Change naming convention for eastward and northward wind fields in `AircraftPerformance` models for consistency with the `Cocip` and `DryAdvection` models. Fields on the `source` are now named `u_wind` and `v_wind` instead of `eastward_wind` and `northward_wind`. Under some paths of computation, this avoids a redundant interpolation.
- Fix the `AircraftPerformance.ensure_true_airspeed_on_source` method in the case when the `met` attr is None and the `fill_with_groundspeed` parameter is enabled.

### Internals

- Make `pycontrails` compatible with `pandas 2.0` and `pandas 2.1`.
- Avoid auto-promotion of float32 to float64 within the `Emissions` model run-time.
- Add convenience `VectorDataset.get_constant` method.

## v0.54.6

### Features

- Add support for generic (model-agnostic) meteorology data to `Cocip` and `CocipGrid`.

- Add two new parameters to the `DryAdvection` model.
  - If the `verbose_outputs` parameter is enabled, additional wind-shear data is included in the output.
  - If the `include_source_in_output` parameter is enabled, the source data with any of the intermediate artifacts (e.g., interpolated met data, wind-shear data, etc.) is included in the output.

  Both parameters are disabled by default.

### Fixes

- Update the CDS URL referenced throughout pycontrails from ``https://cds-beta.climate.copernicus.eu`` to ``https://cds.climate.copernicus.eu``.

### Internals

- Suppress mypy `return-value` errors for functions in `geo.py` where mypy fails to correctly infer return types of numpy ufuncs applied to xarray objects.
- Change `AircraftPerformance` and downstream implementations for better support in running over `Fleet` sources. The runtime of `PSFlight` remains the same.

## v0.54.5

### Features

- This release brings a number of very minor performance improvements to the low-level pycontrails data structures (`VectorDataset` and `MetDataset`). Cumulatively, these changes should bring in a small but nontrivial speedup (~5%) when running a model such as `Cocip` or `DryAdvection` on a single `Flight` source.
  - Core `Flight` methods such as `copy`, `filter`, and `downselect_met` are now ~10x faster for typical use cases.
  - Converting between `Fleet` and `Flight` instances via `Fleet.from_seq` and `Fleet.to_flight_list` are also ~5x faster.
- Implement low-memory met-downselection logic in `DryAdvection`. This is the same logic used in `CocipGrid` to reduce memory consumption by only loading the necessary time slices of the `met` data into memory. If `met` is already loaded into memory, this change will have no effect.

### Breaking Changes

- Remove the `copy` parameter from `GeovectorDataset.downselect_met`. This method always returns a view of the original dataset.
- Remove the `validate` parameter in  the `MetDataArray` constructor. Input data is now always validated.

### Fixes

- Make slightly more explicit when data is copied in the `VectorDataset` constructor: data is now always shallow-copied, and the `copy` parameter governs whether to copy the underlying arrays.
- Call `downselect_met` in `DryAdvection.eval`. (This was previously forgotten.)
- Fix minor bug in `CocipGrid` downselect met logic introduced in v0.54.4. This bug may have caused some met slices to be reloaded when running `CocipGrid.eval` with lazily-loaded `met` and `rad` data.

### Internals

- Add internal `VectorDataset._from_fastpath` and `MetDataset._from_fastpath` class methods to skip data validation.
- Define `__slots__` on `MetBase`, `MetDataset`, `MetDataArray`, and `AttrDict`.
- When `MetDataset` and `MetDataArray` shared a common implementation, move the implementation to `MetBase`. This was the case for the `copy`, `downselect`, and `wrap_longitude` methods.

## v0.54.4

### Features

- Improve the `_altitude_interpolation` function used within `Flight.resample_and_fill` and ensure that it is consistent with the [existing GAIA publication](https://acp.copernicus.org/articles/24/725/2024/) The function `_altitude_interpolation` now accounts for various scenarios. For example:

  1. Flight will automatically climb to an assumed cruise altitude if the first and next known waypoints are at very low altitudes with a large time gap.
  1. If there are large time gaps between known waypoints with a small altitude difference, then the flight will climb at the mid-point of the segment.
  1. If there are large time gaps and positive altitude difference, then the flight will climb at the start of its interpolation until the known cruising altitude and start its cruise phase.
  1. If there are large time gaps and negative altitude difference, then the flight will continue cruising and only starts to descend towards the end of the interpolation.
  1. If there is a shallow climb (ROCD < 500 ft/min), then always assume that the flight will climb at the next time step.
  1. If there is a shallow descent (-250 < ROCD < 0 ft/min), then always assume that the flight will descend at the final time step.

  Conditions (3) to (6) are based on the logic that the aircraft will generally prefer to climb to a higher altitude as early as possible, and descend to a lower altitude as late as possible, because a higher altitude can reduce drag and fuel consumption.

### Breaking changes

- Remove the optional input parameter `climb_descend_at_end` in `Flight.resample_and_fill`. See the description of the new `_altitude_interpolation` function for the rationale behind this change.
- Remove the `copy` argument from `Fleet.from_seq`. This argument was redundant and not used effectively in the implementation. The `Fleet.from_seq` method always returns a copy of the input sequence.

### Fixes

- Fix the `ERA5` interface when making a pressure-level request with a single pressure level. This change accommodates CDS-Beta server behavior. Previously, a ValueError was raised in this case.
- Bypass the ValueError raised by `dask.array.gradient` when the input array is not correctly chunk along the level dimension. Previously, `Cocip` would raise an error when computing tau cirrus in the case that the `met` data had single chunks along the level dimension.
- Fix the `CocipGrid` met downselection process to accommodate cases where `dt_integration` is as large as the time step of the met data. Previously, due to out-of-bounds interpolation, the output of `CocipGrid(met=met, rad=rad, dt_integration="1 hour")` was zero everywhere when the `met` and `rad` data had a time step of 1 hour.
- By default, don't interpolate air temperature when running the `DryAdvection` model in a point-wise manner (no wind-shear simulation).
- Use native python types (as opposed to `numpy` scalars) in the `PSAircraftEngineParams` dataclass.
- Ensure the `PSGrid` model maintains the precision of the `source`. Previously, float32 precision was lost per [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html).
- Fix `Fleet.resample_and_fill` when the the "flight_id" field is included on `Fleet.data` (as opposed to `Fleet.fl_attrs`). Previously, this would raise a ValueError.
- Use the supplied `nominal_rocd` parameter in `Flight.resample_and_fill` rather than `constants.nominal_rocd` (the default value of this parameter).

### Internals

- Add new `AdvectionBuffers` dataclass to override the zero-like values used in `ModelParams` with the buffer previously used in `CocipParams`. This is now a base class for `CocipParams` and `DryAdvectionParams`. In particular, the `DryAdvection` now uses nonzero values for the met downselect buffers.
- Change the order of advected points returned by `DryAdvection` to be consistent with the input order at each time step.
- Add the `RUF` ruleset for linting and formatting the codebase.
- Update type hints for `numpy` 2.2 compatibility. Additional changes may be required after the next iteration of the `numpy` 2.2 series.
- Relax the tolerance passed into `scipy.optimize.newton` in `ps_nominal_grid` to avoid some convergence warnings. (These warnings were more distracting than informative.)
- Remove the `_verify_altitude` check in `Flight.resample_and_fill`. This was often triggered by a flight with corrupt waypoints (ie, independent from the logic in `Flight.resample_and_fill`).

## v0.54.3

### Breaking changes

- Update the default load factor from 70% to 83% to be consistent with historical data. This is used whenever an aircraft performance model is run without a specified load factor.
- By default, the `CocipGrid.create_source` static method will return latitude values from -90 to 90 degrees. This change is motivated by the new advection scheme used near the poles. Previously, this method returned latitude values from -80 to 80 degrees.

### Features

- Create new function `ps_grid.ps_nominal_optimize_mach` which computes the optimal mach number given a set of operating conditions.
- Add a new `jet.aircraft_load_factor` function to estimate aircraft (passenger/cargo) load factor based on historical monthly and regional load factors provided by IATA. This improves upon the default load factor assumption. Historical load factor databases will be continuously updated as new data is released.
- Use a 2D Cartesian-like plane to advect particles near the poles (>80° in latitude) to avoid numerical instabilities and singularities caused by convergence of meridians. This new advection scheme is used for contrail advection in the `Cocip`, `CocipGrid`, and `DryAdvection` models. See the `geo.advect_horizontal` function for more details.

### Fixes

- Ensure the fuel type is preserved when calling `Flight.resample_and_fill`.
- Update the CLIMaCCF dependency to pull the head of the main branch in [CLIMaCCF](https://github.com/dlr-pa/climaccf). Update the installation instructions.
- Update the `ACCFParams.forecast_step` to None, which allows CLIMaCCF to automatically determine the forecast step based on the `met` data.
- Update the `ACCF` NOx parameter for the latest CLIMaCCF version.
- Ensure a custom "horizontal_resolution" param passed into `ACCF` is not overwritten.
- Remove duplicated variable in `ACCF.met_variables`.
- Allow the `ACCF` model to accept relative humidity as a percentage or as a proportion.
- Include `ecmwf.RelativeHumidity` in `ACCF.met_variables` so that `ERA5(..., variables=ACCF.met_variables)` no longer raises an error.

### Internals

- Improve computation of mach limits to accept vectorized input/output.
- Test against python 3.13 in the GitHub Actions CI. Use python 3.13 in the docs and doctest workflows.
- Publish to PyPI using [trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
- Update `pycontrails-bada` installation instructions. Install `pycontrails-bada` from GCP artifact repository in the test workflow.
- Floor the pycontrails version when running the docs workflow. This ensures that the [hosted documentation](https://py.contrails.org) references the last stable release.
- Update literature and bibliography in the documentation.
- Move the `engine_deterioration_factor` from `PSFlightParams` to `AircraftPerformanceParams` so it can be used by both the PS model and BADA.
- Include `engine_deterioration_factor` in `AircraftPerformanceGridParams`.

## v0.54.2

### Features

- Add `cache_download` parameter to the `GFSForecast` interface. When set to `True`, downloaded grib data is cached locally. This is consistent with the behavior of the `ERA5ModelLevel` and `HRESModelLevel` interfaces.

### Fixes

- Update GFS variable names "uswrf" -> "suswrf" and "ulwrf" -> "sulwrf". This accommodates a breaking change introduced in [eccodes 2.38](https://confluence.ecmwf.int/display/MTG2US/Changes+in+ecCodes+version+2.38.0+compared+to+the+previous+version#ChangesinecCodesversion2.38.0comparedtothepreviousversion-Changedshortnames).

### Internals

- Remove `overrides` dependency. Require `typing-extensions` for python < 3.12.
- Upgrade some type hints for more modern python language features.

## v0.54.1

### Features

- Add [CoCiP Grid notebook](https://py.contrails.org/notebooks/CoCiPGrid.html) example to documentation.
- Implement `PSFlight.eval` on a `Fleet` source.

### Breaking changes

- Remove `attrs["crs"]` usage from `GeoVectorDataset` and child classes (`Flight`, `Fleet`). All spatial data is assumed to be EPSG:4326 (WGS84). This was previously assumed implicitly, but now the `crs` attribute is removed from the `attrs` dictionary.
- Change the return type of `GeoVectorDataset.transform_crs` to a pair of numpy arrays representing `x` and `y` coordinates in the target CRS.
- Remove deprecated `MetDataset.variables` property in favor of `MetDataset.indexes`.
- Remove `**kwargs` in `MetDataArray` constructor.
- Rename `ARCOERA5` to `ERA5ARCO` for consistency with the `ERA5` and `ERA5ModelLevel` interfaces.

### Fixes

- Fix the integration time step in `CocipGrid.calc_evolve_one_step`. The previous implementation assumed a time interval of `params["dt_integration"]`. This may not be the case for all `source` parameters (for example, this could occur if running `CocipGrid` over a collection of ADS-B waypoints).
- Raise an exception in constructing `MetDataset(ds, copy=False)` when `ds["level"]` has float32 dtype. Per interpolation conventions, all coordinate variables must have float64 dtype. (This was previously enforced in longitude and latitude, but was overlooked in the level coordinate.)
- Allow `AircraftPerformance.ensure_true_airspeed_on_source` to use `eastward_wind` and `northward_wind` fields on the `source` if available. This is useful when the `source` has already been interpolated to met data.

## v0.54.0

### Features

- Perform model-level to pressure-level conversion in-house instead of relying on `metview`. This update has several advantages:
  - The `ARCOERA5` and `ERA5ModelLevel` interfaces no longer require `metview` to be installed. Similarly, grib files and the associated tooling can also largely be avoided.
  - The computation is performed using `xarray` and `dask` tooling, which means the result can be computed lazily if desired.
  - The computation is defined using `numpy` operations (some of which release the GIL) and can be parallelized using threading through `dask` (this is the default behavior).
  - The computation is generally a bit faster than the `metview` implementation (this depends on the exact chunking of the model level meteorology data). This chunking can be tuned by the user to optimize runtime performance or memory usage.

  See the `ml_to_pl` function for low-level details.

- Update the `ARCOERA5` and `ERA5ModelLevel` interfaces to use the new model-level to pressure-level conversion. The ERA5 model level data is now downloaded as the netcdf format instead of grib. This format change decreases the download size.

### Breaking changes

- Rename `levels` -> `model_levels` in the `ARCOERA5` and `ERA5ModelLevel` constructors. Rename `cache_grib` -> `cache_download`.
- Rename `pressure_levels_at_model_levels` -> `model_level_reference_pressure`. Add a new `model_level_pressure` method that requires surface pressure data.

### Fixes

- Update `ERA5ModelLevel` for the new CDS-Beta server.

### Internals

- Use `ruff` to format the codebase in place of `black`.
- Run `ruff` linting and formatting over the notebook examples in the documentation.
- Update development documentation with new links.

## v0.53.1

### Features

- Support `ERA5` downloads from [CDS-Beta](https://cds-beta.climate.copernicus.eu/). The updated interface is backwards compatible with the legacy CDS server. The choice of CDS server is governed by the `url` parameter in the `ERA5` constructor.

## v0.53.0

### Breaking changes

- Drop python 3.9 support per [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

### Features

- Build wheels for [python 3.13](https://peps.python.org/pep-0719/). (These wheels are available on PyPI, but other pycontrails dependencies may not yet support python 3.13.)

### Fixes

- Fix `PycontrailsRegularGridInterpolator` for compatibility with the latest scipy version.

### Internals

- Defer import of `skimage` and `rasterio`.

## v0.52.3

### Features

- Add experimental `preprocess_lowmem` parameter to `Cocip`. When set to `True`, `Cocip` will attempt to reduce memory consumption during flight preprocessing and initial formation/persistence calculations by using an alternate implementation of `MetDataArray.interpolate` (see below).
- Add `lowmem` keyword-only argument to `MetDataArray.interpolate`. When `True`, attempt to reduce memory consumption by using an alternative interpolation strategy that loads at most two time steps of meteorology data into memory at a time.

### Fixes

- Defer import of `matplotlib` in `models.cocip.output_formats`.
- Fix bug in `PycontrailsRegularGridInterpolator` that caused errors when dispatching to 1-d linear interpolation from in `rgi_cython.pyx`.

### Internals

- Implement low-memory paths in `Cocip.eval` and `MetDataArray.interpolate`.

## v0.52.2

### Breaking changes

- Flight antimeridian crossings are now detected based on individual flight segments rather than minimum and maximum longitude values. This allows for correct detection of antimeridian crossings for flights that span more than 180 degrees longitude, and may change the output of `Flight.resample_and_fill` for such flights.

### Features

- Add experimental `fill_low_alt_with_isa_temperature` parameter on the `AircraftPerformance` base class. When set to `True`, aircraft performance models with `Flight` sources will fill points below the lowest altitude in the ``met["air_temperature]`` data with the ISA temperature. This is useful when the met data does not extend to the surface. In this case, we can still estimate fuel flow and other performance metrics through the entire climb phase. By default, this parameter is set to `False`.
- Add experimental `fill_low_altitude_with_zero_wind` parameter on the `AircraftPerformance` base class. When set to `True`, aircraft performance models will estimate the true airspeed at low altitudes by assuming the wind speed is zero.
- Add convenience `Flight.plot_profile` method.

### Fixes

- Fix missing `Fuel Flow Idle (kg/sec)` value in the `1ZM001` engine in the `edb-gaseous-v29b-engines.csv`.
- Fix the `step_threshold` in `Flight._altitude_interpolation`. This correction is only relevant when passing in a non-default value for `freq` in `Flight.resample_and_fill`.
- Fix the `VectorDataset.__eq__` method to check for the same keys. Previously, if the other dataset had a superset of the instance keys, the method may still return `True`.
- Fix minor bug in `cocip.output_formats.radiation_time_slice_statistics` in which the function previously threw an error if `t_end` did not directly align with the time index in the `rad` dataset.
- Remove the residual constraint in `cocip.output_formats.contrails_to_hi_res_grid` used during debugging.
- Improve detection of antimeridian crossings for flights that span more than 180 degrees longitude.

### Internals

- Improve the runtime performance of `Fleet.to_flight_list`. For a large `Fleet`, this method is now 5x faster.
- Improve the runtime performance and memory footprint of `Cocip._bundle_results`. When running `Cocip` with a large `Fleet` source, `Cocip.eval` is now slightly faster and uses much less memory.

## v0.52.1

### Breaking changes

- Remove `lock=False` as a default keyword argument to `xr.open_mfdataset` in the `MetDataSource.open_dataset` method. This reverts a change from [v0.44.1](https://github.com/contrailcirrus/pycontrails/releases/tag/v0.44.1) and prevents segmentation faults when using recent versions of [netCDF4](https://pypi.org/project/netCDF4/) (1.7.0 and above).
- GOES imagery is now loaded from a temporary file on disk rather than directly from memory when using `GOES.get` without a cachestore.

### Internals

- Remove upper limits on netCDF4 and numpy versions.
- Remove h5netcdf dependency.
- Update doctests with numpy 2 scalar representation (see [NEP 51](https://numpy.org/neps/nep-0051-scalar-representation.html)). Doctests will now fail when run with numpy 1.
- Run certain tests in `test_ecmwf.py` and `test_met.py` using the single-threaded dask scheduler to prevent tests from hanging while waiting for a lock that is never released. (This issue was [encountered previously](https://github.com/contrailcirrus/pycontrails/pull/68), and removing `lock=False` in `MetDataSource.open_dataset` reverts the fix.)
- Document pycontrails installation from conda-forge.

### Fixes

- Ensure the `MetDataset` vertical coordinates `"air_pressure"` and `"altitude"` have the correct dtype.

## v0.52.0

### Breaking changes

- The `_antimeridian_index` helper method in the `flight` module now returns a list of integers rather than an integer. This allows `Flight.to_geojson_multilinestring` to support multiple antimeridian crossings (see below).

### Features

- Add tools for running [APCEMM](https://github.com/MIT-LAE/APCEMM) from within pycontrails. This includes:
  - utilities for generating APCEMM input files and running APCEMM (`pycontrails.models.apcemm.utils`)
  - an interface (`pycontrails.models.apcemm.apcemm`) that allows users to run APCEMM as a pycontrails `Model`.
- Add [APCEMM tutorial notebook](https://py.contrails.org/integrations/APCEMM.html).
- Add prescribed sedimentation rate to `DryAdvection` model.
- Add `Landsat` and `Sentinel` datalibs for searching, retrieving, and visualizing Landsat 8-9 and Sentinel-2 imagery. The datalibs include:
  - Tools for querying Landsat and Sentinel-2 imagery for intersections with user-defined regions (`landsat.query`, `sentinel.query`) or flights (`landsat.intersect`, `sentinel.intersect`). These tools use BigQuery tables and require a Google Cloud Platform account with access to the BigQuery API.
  - Tools for downloading and visualizing imagery from Landsat (`Landsat`) and Sentinel-2 (`Sentinel`). These tools retrieve data anonymously from Google Cloud Platform storage buckets and can be used without a Google Cloud Platform account.
- Add tutorial notebooks demonstrating how to use `Landsat` and `Sentinel` datalibs to find flights in high-resolution satellite imagery.
- Modify `Flight.to_geojson_multilinestring` to make grouping key optional and to support multiple antimeridian crossings.
- Update the `pycontrails` build system to require `numpy 2.0` per the [official numpy guidelines](https://numpy.org/devdocs/dev/depending_on_numpy.html#numpy-2-0-specific-advice). Note that the runtime requirement for `pycontrails` remains `numpy>=1.22`.
- Update `pycontrails` for breaking changes introduced in `numpy 2.0` (e.g., [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)). All changes are backward compatible with `numpy>=1.22`.

### Fixes

- Ensure width and depth are never attached to `DryAdvection` source when running pointwise-only model.
- Ensure that azimuth is not dropped when present in `DryAdvection` input source.
- Exclude netCDF version 1.7.0, which causes non-deterministic failures in unit tests primarily due to segmentation faults.
- Pin numpy to 1.x in runtime dependencies until a working numpy 2.x-compatible netCDF4 release is available.

### Internals

- Create unit tests for the `APCEMM` interface that run if an `APCEMM` executable is found on the `PATH` inside a clean APCEMM git repository with a pinned git hash. These tests will be skipped unless users carefully configure their local test environment and will not run in CI.
- Exclude APCEMM tutorial notebook from notebook tests.
- Add unit tests for Landsat and Sentinel search tools and datalibs, but disable any tests that retrieve imagery data when running tests in GitHub Actions to limit resource consumption in GitHub runners. Users can disable these units tests locally by setting the environment variable `GITHUB_ACTIONS=true`.
- Ensure `GITHUB_ACTIONS` environment variable is available when building and testing wheels on linux.
- Skip cells that retrieve imagery data when running tests on Landsat and Sentinel tutorial notebooks.
- Add tests for `Flight.to_geojson_multilinestring` with grouping key omitted and update tests with multiple antimeridian crossings.
- Minimum pandas version is bumped to 2.2 to ensure the the `include_groups` keyword argument used in `Flight.to_geojson_multilinestring` is available.
- Upgrade minimum `mypy` dependencies

## v0.51.2

### Features

- Add functionality to automatically compare  simulated contrails from `cocip.Cocip` with GOES satellite imagery (`compare_cocip_with_goes`).

### Internals

- Fix documentation build in CI.

## v0.51.1

### Breaking changes

- Average fuel burn in the PS model is increased by 2.5\% unless `engine_deterioration_factor` is overriden.

### Features

- PS model: Support four aircraft types, including `E75L`, `E75S`, `E290`, and `E295`.
- PS model: Integrate `ps_synonym_list` to increase PS model aircraft type coverage to 102.
- PS model: Account for increase in fuel consumption due to engine deterioration between maintenance cycle.

### Internals

- Calculate rate of climb and descent (ROCD) using hydrostatic equation to improve accuracy.
- PS model: Move engine performance buffer from `c_t_available` to `tr_max`.
- Set `tr_max` buffer to +20%, as recommended by Ian Poll.

## v0.51.0

### Breaking changes

- Geodesic interpolation is now used in `Flight.resample_and_fill` when the great circle distance between waypoints (rather than the total segment length including vertical displacement) exceeds a threshold. This may change the interpolation method used when resampling flight segments with lengths close to the geodesic interpolation threshold.
- Fixed typo in `thermo.c_pm`  will decrease computed values of moist heat capacity with non-zero specific humidity. We expect the downstream impact on contrail predictions by `ISSR`, `SAC`, `PCR`, and `Cocip` models to be minimal.
- `np.nan` is now used as the default `fill_value` in `MetDataArray.to_polygon_feature` and `MetDataArray.to_polygon_feature_collection`. This ensures that NaN values are never included in polygon interiors unless a non-NaN `fill_value` is explicitly passed as a keyword argument.

### Features

- Add `ERA5ModelLevel` and `HRESModelLevel` interfaces for accessing ERA5 and HRES data on model levels.
- Update [ECMWF tutorial notebook](https://py.contrails.org/notebooks/ECMWF.html) with instructions for using model-level datalibs.
- Add `HistogramMatching` humidity scaling calibrated for model-level ERA5 data.
- Modify `polygon.find_multipolygon`, `MetDataArray.to_polygon_feature`, `MetDataArray.to_polygon_feature_collection`, and `MetDataArray.to_polyhedra` to permit finding regions with values above or below a threshold.

### Fixes

- Use horizontal great circle distance to determine whether geodesic interpolation is used in `Flight.resample_and_fill`. This ensures geodesic interpolation is used between sufficiently distant waypoints even when one or both waypoints contain NaN altitude data.
- Fix typo in moist heat capacity equation `thermo.c_pm`.

### Internals

- Create static copy of dataframe for determining pressure at ECMWF model levels.
- Extract model-level utilities in `ARCOERA5` to their own module for reuse in `ERA5ModelLevel` and `HRESModelLevel`.
- Update Makefile so `make ensure-era5-cached` uses default cache directory when run locally.
- Bump pinned black and ruff versions.
- Disable mypy type checking on `functools.wraps` in `support_arraylike` decorator to avoid error that appears starting on mypy 1.10.0.
- Update pinned `Cocip` test output values after moist heat capacity bugfix.
- Add static files with ECMWF model levels and model-level ERA5 RHI quantiles to packaged data.
- Pass `exc_type=ImportError` to `pytest.importorskip` in test fixtures that use pycontrails extensions to suppress pytest warning when extensions are not installed.
- Bump minimum pytest version 8.2 to ensure the `exc_type` kwarg is available.

## v0.50.2

### Breaking changes

- Replaces engine-uid `01P10IA024` with `04P10IA027` (superseded in the IACO EDB).

### Features

- Adds support for the `E190` aircraft type in the PS model
- Adds `Flight.distance_to_coords` which takes in a distance along a flight trajectory in meters and returns geodesic coordinates.
- Adds methods to `ps_operational_limits` to find the maximum and minimum mach numbers for a given set of operating conditions.
- Updates ICAO aircraft engine emissions databank (EDB) from v28c to v29b.

### Internals

- New data for gaseous and nvPM emissions for PW812D, PW812GA, RR Trent 7000 with improved nvPM combustor
- Update nvPM emissions for IAE V2530-A5, PW1500G and PW1900G
- Update PS model coeffiencts to match the latest version provided by Ian Poll

## v0.50.1

### Breaking changes

- Updates to flight resampling logic now ensure that resampled waypoints include any and all times between flight start and end times that are a multiple of the resampling frequency. This may add an additional waypoint to some flights after resampling, and may result in `Flight.resample_and_fill` returning a flight with a single waypoint rather than an empty flight.

### Features

- Refine CoCiP contrail initialization model based on the work of Unterstrasser (2016, doi:10.5194/acp-16-2059-2016) and Karcher (2018, doi:10.1038/s41467-018-04068-0).
  - This update implements a refined parameterization of the survival fraction of contrail ice crystal number after the wake vortex phase (`f_surv`). The parameterised model was developed by Unterstrasser (2016) based on outputs provided by large eddy simulations, and improves agreement with LES output relative to the default survival fraction parameterization in CoCiP.
  - These changes replicate Fig. 4 of Karcher (2018), where `f_surv` now depends on the initial number of ice crystals. These effects are particularly important, especially in the "soot-poor" scenario where the number fraction of contrail ice crystals that survives the wake vortex phase could be larger than the mass fraction, because the particles are larger in size.
  - This also improves upon the existing assumption in CoCiP, where the survival fraction is estimated as the change in contrail ice water content (by mass) before and after the wake vortex phase.
The Unterstrasser (2016) parameterization can be used in CoCiP by setting a new parameter, `unterstrasser_ice_survival_fraction`, to `True`.
- Adds optional ATR20 to CoCiPGrid model.

### Fixes

- Update flight resampling logic to align with expected behavior for very short flights, which is now detailed in the `Flight.resample_and_fill` docstring.

### Internals

- Adds a parameter to `CoCipParams`, `unterstrasser_ice_survival_fraction`, that activates the Unterstrasser (2016) survival parameterization when set to `True`. This is disabled by default, and only implemented for `CoCiP`. `CoCiPGrid` will produce an error if run with `unterstrasser_ice_surival_fraction=True`.
- Modifies `CoCiPGrid` so that setting `compute_atr_20` (defined in `CoCipParams`) to `True` adds `global_yearly_mean_rf` and `atr20` to CoCiP-grid output.
- Replaces `pycontrails.datalib.GOES` ash convention label "MIT" with "SEVIRI"
- Modifies meteorology time step selection logic in `CoCiPGrid` to reduce duplicate chunk downloads when reading from remote zarr stores.
- Updates unit tests for xarray v2024.03.0, which introduced changes to netCDF decoding that slightly alter decoded values. Note that some unit tests will fail for earlier xarray versions.
- Updates `RegularGridInterpolator` to fall back on legacy scipy implementations of tensor-product spline methods when using scipy versions 1.13.0 and later.

## v0.50.0

### Features

- Add `ARCOERA5` interface for accessing ARCO ERA5 model level data. This interface requires the [metview](https://metview.readthedocs.io/en/latest/python.html) python package.
- Add [ARCO ERA5 tutorial notebook](https://py.contrails.org/notebooks/ARCO-ERA5.html) highlighting the new interface.
- Add support to output contrail warming impact in ATR20

### Breaking changes

- Reduce `CocipParams.met_level_buffer` from `(200, 200)` to `(40, 40)`. This change is motivated by the observation that the previous buffer was unnecessarily large and caused additional memory overhead. The new buffer is more in line with the typical vertical advection path of a contrail.

### Fixes

- Raise ValueError when `list[Flight]` source is provided to `Cocip` and the `copy_source` parameter is set to `False`. Previously the source was copied in this case regardless of the `copy_source` parameter.
- Fix broken link in the [model level notebook](https://py.contrails.org/notebooks/model-levels.html).

### Internals

- The `datalib.parse_pressure_levels` now sorts the pressure levels in ascending order and raises a ValueError if the input pressure levels are duplicated or have mixed signs.
- Add new `MetDataSource.is_single_level` property.
- Add `ecmwf.Divergence` (a subclass of `MetVariable`) for accessing ERA5 divergence data.
- Update the [specific humidity interpolation notebook](https://py.contrails.org/notebooks/specific-humidity-interpolation.html) to use the new `ARCOERA5` interface.
- Adds two parameters to `CoCipParams`, `compute_atr20` and `global_rf_to_atr20_factor`. Setting the former to `True` will add both `global_yearly_mean_rf` and `atr20` to the CoCiP output.
- Bump minimum pytest version to 8.1 to avoid failures in release workflow.

## v0.49.5

### Fixes

- Fix bug in which `Cocip._process_rad` dropped radiation dataset attributes introduced in v0.49.4.

## v0.49.4

### Breaking changes

- Remove the `CocipGridParams.met_slice_dt` parameter. Now met downselection is handled automatically during contrail evolution. When the `met` and `rad` data passed into `CocipGrid` are not already loaded into memory, this update may make `CocipGrid` slightly more performant.
- No longer explicitly load `met` and `rad` time slices into memory in `CocipGrid`. This now only occurs downstream when interpolation is performed. This change better aligns `CocipGrid` with other pycontrails models.
- Remove the `cocipgrid.cocip_time_handling` module. Any useful tooling has been moved directly to the `cocipgrid.cocip_grid` module.
- Remove the `CocipGrid.timedict` attribute. Add a `CocipGrid.timesteps` attribute. This is now applied in the same manner that the `Cocip` model uses its `timesteps` attribute.
- Simplify the runtime estimate used in constructing the `CocipGrid` `tqdm` progress bar. The new estimate is less precise than the previous estimate and should not be trusted for long-running simulations.
- Deprecate `MetBase.variables` in favor of `MetBase.indexes`.

### Features

- Add support for 9 additional aircraft types in the [Poll-Schumann](https://py.contrails.org/notebooks/AircraftPerformance.html) (PS) aircraft performance model. The new aircraft types are:
  - A338
  - A339
  - A35K
  - B37M
  - B38M
  - B39M
  - B78X
  - BCS1
  - BCS3
- Modify PS coefficients for B788, B789, and A359.
- Support running `CocipGrid` on meteorology data without a uniformly-spaced time dimension. The `CocipGrid` implementation now no longer assumes `met["time"].diff()` is constant.
- Add a `MetDataset.downselect_met` method. This performs a met downselection in analogy with `GeoVectorDataset.downselect_met`.

### Fixes

- Improve clarity of warnings produced when meteorology data doesn't cover the time range required by a gridded CoCiP model.
- No longer emit `pandas` warning when `Flight.resample_and_fill(..., drop=True, ...)` is called with non-float data.
- Correctly handle `CocipGrid` `rad` data with non-uniform time steps.

## v0.49.3

### Features

- Re-organize notebooks in documentation.
- Add new [model level](https://py.contrails.org/examples/model-levels.html) tutorial notebook.
- Add new high-level `Flight.clean_and_resample` method. This method parallels the `Flight.resample_and_fill` method but performs additional altitude filtering. In essence, this method is a combination of `Flight.filter_altitude` and `Flight.resample_and_fill`.

### Breaking changes

- Remove `Flight.fit_altitude` method in favor of `Flight.filter_altitude`. The new method now only applies a median filter during cruise flight phase.

### Fixes

- Remove opaque warning issued when all tau_contrail values are nan in `Cocip` evolution.
- Emit warning in `Cocip.eval` if the advected contrail is blown outside of the domain of the met data.
- Remove empty flights in `Fleet.from_seq`. Issue warning if an empty flight is encountered.
- Emit warning when `Flight.resample_and_fill` returns an empty flight.

### Internals

- Modify test workflow to use Makefile recipes and ensure early failures are detected in CI.
- Pin `black` and `ruff` versions for consistency between local and CI/CD environments.
- Improve development documentation.
- Improve handling of missing credentials in tests (`make nb-test`, `make doctest`).
- Update time frequency aliases for `pandas` 2.2 compatibility.
- Update cython annotations for `scipy` 1.12 compatibility.
- Improve notebook output testing capabilities (`make nb-test`).
- Add new convenience Make recipe to execute all notebooks in the docs (`make nb-execute`).
- Add new Make recipe to cleanup notebooks (`make nb-clean`).
- Add pre-commit hook to check if notebooks are *clean*.
- Re-organize notebooks in documentation.
- Clean up contributing and develop documentation.
- Automatically parse `np.timedelta64`-like model params in `Model.update_params`.

## v0.49.2

### Features

- Support [pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html). This can be enabled with `pd.set_option("mode.copy_on_write", True)` or by setting the `PANDAS_COPY_ON_WRITE` environment variable.

### Fixes

- Ensure the `Flight.fuel` attribute is preserved for the `Flight.filter` method.
- Ensure the `Fleet.fl_attrs` attribute is preserved for the `Fleet.filter` method.
- Raise `ValueError` when `Flight.sort` or `Fleet.sort` is called. Both of these subclasses assume a custom sorting order that is enforced in their constructors.
- Always correct intermediate thrust coefficients computed in the `PSGrid` model. This correction is already enabled by default in the `PSFlight` model.
- Include `attr` fields in the ValueError message raised in `CocipGrid` when not all aircraft performance variables are present on the `source` parameter.
- Allow `mach_number` as a replacement for `true_airspeed` in `CocipGrid` aircraft performance determination.

### Internals

- Make `Fuel` and its subclasses `JetA`, `SAFBlend`, and `HydrogenFuel` frozen.
- No longer copy `met` when `Models.downselect_met` is called. In some modes of operation, this reduces the memory footprint of the model.
- Update codebase for more harmony with [PDEP 8](https://jorisvandenbossche.github.io/pandas-website-preview/pdeps/0008-inplace-methods-in-pandas.html) and Copy-on-Write semantics.
- Add `default` parameter to the `VectorDataset.get_data_or_attr` method.

## v0.49.1

### Fixes

- Fix memory bottleneck in `CocipGrid` simulation by avoiding expensive call to `pd.concat`.
- Require `oldest-supported-numpy` for python 3.12. Remove logic for `numpy` 1.26.0rc1 in the `pyproject.toml` build system.

## v0.49.0

This release updates the Poll-Schumann (PS) aircraft performance model to version 2.0. It also includes a number of bug fixes and internal improvements.

### Features

- Add convenience `Fleet.resample_and_fill`.
- Update the PS model aircraft-engine parameters.
- Improve PS model accuracy in fuel consumption estimates.
- Improve PS model accuracy in overall propulsive efficiency estimates.
- Include additional guardrails in the PS model to constrain the operational limits of different aircraft-engine types, i.e., the maximum permitted Mach number by altitude, maximum available thrust coefficient, maximum lift coefficient, and maximum allowable aircraft mass.

### Fixes

- Update polygon algorithm to use `shapely.Polygon` instead of `shapely.LinearRing` for contours with at least 4 vertices.
- Fix `Fleet.to_flight_list` to avoid duplicating global attributes on the child `Flight` instances.
- Add `__slots__` to `GeoVectorDataset`, `Flight`, and `Fleet`. The base `VectorDataset` class already uses `__slots__`.
- Add `Fleet.copy` method.
- Improve `Fleet.__init__` implementation.
- Ensure `source` parameter is mutated in `CocipGrid.eval` when the model parameter `copy_source=False`.

## v0.48.1

### Features

- Generalize `met.shift_longitude()` to translate longitude coordinates onto any domain bounds.
- Add `VectorDataset.to_dict()` methods to output Vector data as dictionary. This method enables `Flight.to_dict()` objects to be serialized for input to the [Contrails API](https://api.contrails.org).
- Add `VectorDataset.from_dict()` class method to create `VectorDataset` class from dictionary.
- Support more time formats, including timezone aware times, in `VectorDataset` creation. All timezone aware `"time"`` coordinates are converted to UTC and stripped of timezone identifier.

### Fixes

- Fix issue in the `wake_vortex.max_downward_displacement` function in which float32 dtypes were promoted to float64 dtypes in certain cases.
- Ignore empty vectors in `VectorDataset.sum`.

### Internals

- Set `frozen=True` on the `MetVariable` dataclass.
- Test against python 3.12 in the GitHub Actions CI. Use python 3.12 the docs and doctest workflows.

## v0.48.0

This release includes a number of breaking changes and new features. If upgrading from a previous version of `pycontrails`, please read the changelog carefully. Open an [issue](https://github.com/contrailcirrus/pycontrails/issues) if you experience problems.

### Breaking changes

- When running `Cocip` and other `pycontrails` models, the `met` and `rad` parameter must now contain predefined metadata attributes `provider`, `dataset`, and `product` describing the met source. An error will now be raised in `Cocip` if these attributes are not present.
- Deprecate passing arbitrary `kwargs` into the `MetDataArray` constructor.
- No longer convert accumulated radiation data to average instantaneous data in `ERA5` and `HRES` interfaces. This logic is now handled downstream by the model (e.g., `Cocip`). This change allows for more flexibility in the `rad` data passed into the model and avoids unnecessary computation in the `MetDataSource` interfaces.
- Add new `MetDataSource.set_met_source_metadata` abstract method. This should be called within the implementing class `open_metdataset` method.
- No longer take a finite difference in the time dimension for HRES radiation data. This is now also handled natively in `Cocip`.
- No longer convert relative humidity from a percentage to a fraction in `ERA5` and `HRES` interfaces.
- Require the `HRES` `stream` parameter to be one of `["oper", "enfo"]`. Require the `field_type` parameter to be one of `["fc", "pf", "cf", "an"]`.
- Remove the `steps` and `step_offset` properties in the `GFSForecast` interface. Now the `timesteps` attribute is the only source of truth for determining AWS S3 keys. Change the `filename` method to take in a `datatime` timestep instead of an `int` step. No longer assign the first step radiation data to the zeroth step.
- Change the return type of `ISSR.eval`, `SAC.eval`, and `PCR.eval` from `MetDataArray` to `MetDataset`. This is more consistent with the return type of other `pycontrails` models and more closely mirrors the behavior of vector models. Set output `attrs` metadata on the global `MetDataset` instead of the individual `MetDataArray` in each case.

### Features

- Rewrite parts of the `pycontrails.core.datalib` module for higher performance and readability.
- Add optional `attrs` and `attrs_kwargs` parameters to `MetDataset` constructor. This allows the user to customize the attributes on the underlying `xarray.Dataset` object. This update makes `MetDataset` more consistent with `VectorDataset`.
- Add three new properties `provider_attr`, `dataset_attr`, and `product_attr` to `MetDataset`. These properties give metadata describing the underlying meterological data source.
- Add new `Model.transfer_met_source_attrs` method for more consistent handling of met source metadata on the `source` parameter passed into `Model.eval`.
- No longer require `geopotential` data when computing `tau_cirrus`. If neither `geopotential` nor `geopotential_height` are available, geopotential is approximated from the geometric height. No longer require geopotential on the `met` parameter in `Cocip` or `CocipGrid`.
- Remove the `Cocip` `shift_radiation_time` parameter. This is now inferred directly from the `rad` metadata. An error is raised if the necessary metadata is not present.
- Allow `Cocip` to run with both instantaneous (`W m-2`) and accumulated (`J m-2`) radiation data.
- Allow `Cocip` to run with accumulated ECMWF HRES radiation data.

### Fixes

- Correct radiation unit in the `ACCF` wrapper model [#64]. Both instantaneous (`W m-2`) and accumulated (`J m-2`) radiation data are now supported, and the `ACCF` wrapper will handle each appropriately.
- Avoid unnecessary writing and reading of temporary files in `ERA5.cache_dataset` and `HRES.cache_dataset`.
- Fix timestep resolution bug in `GFSForecast`. When the `grid` parameter is 0.5 or 1.0, forecasts are only available every 3 hours. Previously, the `timesteps` property would define an hourly timestep.

### Internals

- Include `name` parameter in `MetDataArray` constructor.
- Make the `coordinates.slice_domain` function slightly more performant by explicitly dropping nan values from the `request` parameter.
- Round unwieldy floating point numbers in `GeoVectorDataset._display_attrs`.
- Remove the `ecmwflibs` package from the `ecmwf` optional dependencies.
- Add NPY to `ruff` rules.
- Add convenience `MetDataset.standardize_variables` method.
- Remove the `p_settings` attribute on the `ACCF` interface. This is now constructed internally within `ACCF.eval`. Replace the `ACCF._update_accf_config` method with a `_get_accf_config` function.

## v0.47.3

### Fixes

- Strengthen `correct_fuel_flow` in the `PSmodel` to account for descent conditions.
- Clip the denominator computed in `pycontrails.physics.jet.equivalent_fuel_flow_rate_at_cruise`.
- Ensure the token used within GitHub workflows has the fewest privileges required. Set top-level permissions to `none` in each workflow file. Remove unnecessary permissions previously used in the `google-github-actions/auth` action.
- Fix bug in `radiative_forcing.effective_tau_contrail` identified in [#99](https://github.com/contrailcirrus/pycontrails/issues/99).
- Fix the unit for `vertical_velocity` in `geo.advect_level`.
- Fix bug appearing in `Flight._geodesic_interpolation` in which a single initial large gap was not interpolated with a geodesic path.

### Internals

- Add `FlightPhase` to the `pycontrails` namespace.

## v0.47.2

### Features

- New experimental `GOES` interface for downloading and visualizing GOES-16 satellite imagery.
- Add new [GOES example notebook](https://py.contrails.org/examples/GOES.html) highlighting the interface.
- Build python 3.12 wheels for Linux, macOS, and Windows on release. This is in addition to the existing python 3.9, 3.10, and 3.11 wheels.

### Fixes

- Use the experimental version number parameter `E` in `pycontrails.ecmwf.hres.get_forecast_filename`. Update the logic involved in setting the dissemination data stream indicator `S`.
- Change the behavior of `_altitude_interpolation` method that is called within  `resample_and_fill`. Step climbs are now placed in the middle of long flight segments. Descents continue to occur at the end of segments.

### Internals

- Provide consistent `ModuleNotFoundError` messages when optional dependencies are not installed.
- Move the `synthetic_flight` module into the `pycontrails.ext` namespace.

## v0.47.1

### Fixes

- Fix bug in `PSGrid` in which the `met` data was assumed to be already loaded into memory. This caused errors when running `PSGrid` with a `MetDataset` source.
- Fix bug (#86) in which `Cocip.eval` loses the `source` fuel type. Instead of instantiating a new `Flight` or `Fleet` instance with the default fuel type, the `Cocip._bundle_results` method now overwrites the `self.source.data` attribute with the bundled predictions.
- Avoid a memory explosion when running `Cocip` on a large non-dask-backed `met` parameter. Previously the `tau_cirrus` computation would be performed in memory over the entire `met` dataset.
- Replace `datetime.utcfromtimestamp` (deprecated in python 3.12) with `datetime.fromtimestamp`.
- Explicitly support python 3.12 in the `pyproject.toml` build system.

### Internals

- Add `compute_tau_cirrus_in_model_init` parameter to `CocipParams`. This controls whether to compute the cirrus optical depth in `Cocip.__init__` or `Cocip.eval`. When set to `"auto"` (the default), the `tau_cirrus` is computed in `Cocip.__init__` if and only if the `met` parameter is dask-backed.
- Change data requirements for the `EmpiricalGrid` aircraft performance model.
- Consolidate `ERA5.cache_dataset` and `HRES.cache_dataset` onto common `ECMWFAPI.cache_dataset` method. Previously the child implementations were identical.
- No longer require the `pyproj` package as a dependency. This is now an optional dependency, and can be installed with `pip install pycontrails[pyproj]`.

## v0.47.0

Implement a Poll-Schumann (`PSGrid`) theoretical aircraft performance over a grid.

### Breaking changes

- Move the `pycontrails.models.aircraft_performance` module to `pycontrails.core.aircraft_performance`.
- Rename `PSModel` -> `PSFlight`.

### Fixes

- Use the instance `fuel` attribute in the `Fleet.to_flight_list` method. Previously, the default `JetA` fuel was always used.
- Ensure the `Fleet.fuel` attribute is inferred from the underlying sequence of flights in the `Fleet.from_seq` method.

### Features

- Implement the `PSGrid` model. For a given aircraft type and position, this model computes optimal aircraft performance at cruise conditions. In particular, this model can be used to estimate fuel flow, engine efficiency, and aircraft mass at cruise. In particular, the `PSGrid` model can now be used in conjunction with `CocipGrid` to simulate contrail formation over a grid.
- Refactor the `Emissions` model so that `Emissions.eval` runs with `source: GeoVectorDataset`. Previously, the `eval` method required a `Flight` instance for the `source` parameter. This change allows the `Emissions` model to run more seamlessly as a sub-model of a gridded model (ie, `CocipGrid`),
- No longer require `pycontrails-bada` to import or run the `CocipGrid` model. Instead, the `CocipGridParams.aircraft_performance` parameter can be set to any `AircraftPerformanceGrid` instance. This allows the `CocipGrid` model to run with any aircraft performance model that implements the `AircraftPerformanceGrid` interface.
- Add experimental `EmpiricalAircraftPerformanceGrid` model.
- Add convenience `GeoVectorDataset.T_isa` method to compute the ISA temperature at each point.

### Internals

- Add optional `climb_descend_at_end` parameter to the `Flight.resample_and_fill` method. If True, the climb or descent will be placed at the end of each segment rather than the start.
- Define `AircraftPerformanceGridParams`, `AircraftPerformanceGrid`, and `AircraftPerformanceGridData` abstract interfaces for gridded aircraft performance models.
- Add `set_attr` parameter to `Models.get_source_param`.
- Better handle `source`, `source.attrs`, and `params` customizations in `CocipGrid`.
- Include additional classes and functions in the `pycontrails.models.emissions` module.
- Hardcode the paths to the static data files used in the `Emissions` and `PSFlight` models. Previously these were configurable by model parameters.
- Add `altitude_ft` parameter to the `GeoVectorDataset` constructor. Warn if at least two of `altitude_ft`, `altitude`, and `level` are provided.
- Allow instantiation of `Model` instances with `params: ModelParams`. Previously, the `params` parameter was required to be a `dict`. The current implementation checks that the `params` parameter is either a `dict` or has type `default_params` on the `Model` class.

## v0.46.0

Support "dry advection" simulation.

### Features

- Add new `DryAdvection` model to simulate sediment-free advection of an aircraft's exhaust plume. This model is experimental and may change in future releases. By default, the current implementation simulates plume geometry as a cylinder with an elliptical cross section (the same geometry assumed in CoCiP). Wind shear perturbs the ellipse azimuth, width, and depth over the plume evolution. The `DryAdvection` model may also be used to simulate advection without wind-shear effects by setting the model parameters `azimuth`, `width`, and `depth` to None.
- Add new [Dry Advection example notebook](https://py.contrails.org/examples/advection.html) highlighting the new `DryAdvection` model and comparing it to the `Cocip` model.
- Add optional `fill_value` parameter to `VectorDataset.sum`.

### Fixes

- (#80) Fix unit error in `wake_vortex.turbulent_kinetic_energy_dissipation_rate`. This fix affects the estimate of wake vortex max downward displacement and slightly changes `Cocip` predictions.
- Change the implementation of `Flight.resample_and_fill` so that lat-lon interpolation is linear in time. Previously, the timestamp associated to a waypoint was floored according to the resampling frequency without updating the position accordingly. This caused errors in segment calculations (e.g., true airspeed).

### Internals

- Add optional `keep_original_index` parameter to the `Flight.resample_and_fill` method. If `True`, the time original index is preserved in the output in addition to the new time index obtained by resampling.
- Improve docstrings in `wake_vortex` module
- Rename `wake_vortex` functions to remove `get_` prefix at the start of each function name.
- Add pytest command line parameter `--regenerate-results` to regenerate static test fixture results. Automate in make recipe `make pytest-regenerate-results`.
- Update handling of `GeoVectorDataset.required_keys` the `GeoVectorDataset` constructor. Add class variable `GeoVectorDataset.vertical_keys` for handing the vertical dimension.
- Rename `CocipParam.max_contrail_depth` -> `CocipParam.max_depth`.
- Add `units.dt_to_seconds` function to convert `np.timedelta64` to `float` seconds.
- Rename `thermo.p_dz` -> `thermo.pressure_dz`.

## v0.45.0

Add experimental support for simulating radiative effects due to contrail-contrail overlap.

### Features

- Support simulating contrail contrail overlap when running the `Cocip` model with a `Fleet` source. The `contrail_contrail_overlapping` and `dz_overlap_m` parameters govern the overlap calculation. This mode of calculation is still experimental and may change in future releases.
- Rewrite the `pycontrails.models.cocip.output` modules into a single `pycontrails.cocip.output_formats` module. The new module supports flight waypoint summary statistics, contrail waypoints summary statistics, gridded outputs, and time-slice outputs.
- Add new `GeoVectorDataset.to_lon_lat_grid` method. This method can be thought of as a partial inverse to the `MetDataset.to_vector` method. The current implementation is brittle and may be revised in a future release.

### Fixes

- Extend `Models.set_source_met` to allow `interpolation_q_method="cubic-spline"` when working with `MetDataset` source (ie, so-called gridded models). Previously a `NotImplementedError` was raised.
- Ensure the `Flight.copy` implementation works with `Fleet` instances as well.
- Avoid looping over `keys` twice in `VectorDataset.broadcast_attrs`. This is a slight performance enhancement.
- Fix `Fleet` signature for compatibility with `Flight`.
- Fix a few hard-coded assumptions in broadcasting aircraft performance and emissions when running `Cocip` with a `Fleet` source. The previous implementation did not consider the possibility of aircraft performance variables on `Flight.data` and `Flight.attrs` separately.

### Internals

- Add optional `raise_error` parameter to the `VectorDataset.broadcast_attrs` method.
- Update `Fleet` internals.

## v0.44.2

### Fixes

- Narrow type hints on the ABC `AircraftPerformance` model. The `AircraftPerformance.eval` method requires a `Flight` object for the `source` parameter.
- In `PSFlight.eval`, explicitly set any aircraft performance data at waypoints with zero true airspeed to `np.nan`. This avoids numpy `RuntimeWarning`s without affecting the results.
- Fix corner-case in the `polygon.buffer_and_clean` function in which the polygon created by buffering the `opencv` contour is not valid. Now a second attempt to buffer the polygon is made with a smaller buffer distance.
- Ignore RuntimeError raised in `scipy.optimize.newton` if the maximum number of iterations is reached before convergence. This is a workaround for occasional false positive convergence warnings. The pycontrails use-case may be related to [this GitHub issue](https://github.com/scipy/scipy/issues/8904).
- Update the `Models.__init__` warnings when `humidity_scaling` is not provided. The previous warning provided an outdated code example.
- Ensure the `interpolation_q_method` used in a parent model is passed into the `humidity_scaling` child model in the `Models.__init__` method. If the two `interpolation_q_method` values are different, a warning is issued. This could be extended to other model parameters in the future.

### Features

- Enable `ExponentialBoostLatitudeCorrectionHumidityScaling` humidity scaling for the model parameter `interpolation_q_method="cubic_spline"`.
- Add [GFS notebook](https://py.contrails.org/examples/GFS.html) example.

### Breaking changes

- Remove `ExponentialBoostLatitudeCorrectionHumidityScalingParams`. These parameters are now hard-coded in the `ExponentialBoostLatitudeCorrectionHumidityScaling` model.

## v0.44.1

### Breaking changes

- By default, call `xr.open_mfdataset` with `lock=False` in the `MetDataSource.open_dataset` method. This helps alleviate a `dask` threading issue similar to [this GitHub issue](https://github.com/pydata/xarray/issues/4406).

### Fixes

- Support `MetDataset` source in the `HistogramMatching` humidity scaling model. Previously only `GeoVectorDataset` sources were explicitly supported.
- Replace `np.gradient` with `dask.array.gradient` in the `tau_cirrus` module. This ensures that the computation is done lazily for dask-backed arrays.
- Round to 6 digits in the `polygon.determine_buffer` function. This avoid unnecessary warnings for rounding errors.
- Fix type hint for `opencv-python` 4.8.0.74.

### Internals

- Take more care with `float` and `int` types in the `contrail_properties` module. Prefer `np.clip` to `np.where` or `np.maximum` for clipping values.
- Support `air_temperature` in `CocipGrid` verbose formation outputs.
- Remove `pytest-timeout` dev dependency.

## v0.44.0

Support for the [Poll-Schumann aircraft performance model](https://doi.org/10.1017/aer.2020.62).

### Features

- Implement a basic working version of the Poll-Schumann (PS) aircraft performance model. This is experimental and may undergo revision in future releases. The PS Model currently supports the following 53 aircraft types:
  - A30B
  - A306
  - A310
  - A313
  - A318
  - A319
  - A320
  - A321
  - A332
  - A333
  - A342
  - A343
  - A345
  - A346
  - A359
  - A388
  - B712
  - B732
  - B733
  - B734
  - B735
  - B736
  - B737
  - B738
  - B739
  - B742
  - B743
  - B744
  - B748
  - B752
  - B753
  - B762
  - B763
  - B764
  - B77L
  - B772
  - B77W
  - B773
  - B788
  - B789
  - E135
  - E145
  - E170
  - E195
  - MD82
  - MD83
  - GLF5
  - CRJ9
  - DC93
  - RJ1H
  - B722
  - A20N
  - A21N

  The "gridded" version of this model is not yet implemented. This will be added in a future release.
- Improve the runtime of instantiating the `Emissions` model by a factor of 10-15x. This translates to a time savings of several hundred milliseconds on modern hardware. This improvement is achieved by more efficient parsing of the underlying static data and by deferring the construction of the interpolation artifacts until they are needed.
- Automatically use a default engine type from the aircraft type in the `Emissions` model if an `engine_uid` parameter is not included on the `source`. This itself is configurable via the `use_default_engine_uid` parameter on the `Emissions` model. The default mappings from aircraft types to engines is included in `pycontrails/models/emissions/static/default-engine-uids.csv`.

### Breaking changes

- Remove the `Aircraft` dataclass from `Flight` instantiation. Any code previously using this should instead directly pass additional `attrs` to the `Flight` constructor.
- The `load_factor` is now required in `AircraftPerformance` models. The global `DEFAULT_LOAD_FACTOR` constant in `pycontrails.models.aircraft_performance` provides a reasonable default. This is currently set to 0.7.
- Use a `takeoff_mass` parameter in `AircraftPerformance` models if provided in the `source.attrs` data.
- No longer use a reference mass `ref_mass` in `AircraftPerformance` models. This is replaced by the `takeoff_mass` parameter if provided, or calculated from operating empty operating mass, max payload mass, total fuel consumption mass, reserve fuel mass, and the load factor.

### Fixes

- Remove the `fuel` parameter from the `Emissions` model. This is inferred directly from the `source` parameter in `Emissions.eval`.
- Fix edge cases in the `jet.reserve_fuel_requirements` implementation. The previous version would return `nan` for some combinations of `fuel_flow` and `segment_phase` variables.
- Fix a spelling mistake: `units.kelvin_to_celcius` -> `units.kelvin_to_celsius`.

### Internals

- Use `ruff` in place of `pydocstyle` for linting docstrings.
- Use `ruff` in place of `isort` for sorting imports.
- Update the `AircraftPerformance` template based on the patterns used in the new `PSFlight` class. This may change again in the future.

## v0.43.0

Support experimental interpolation against gridded specific humidity. Add new data-driven humidity scaling models.

### Features

- Add new experimental `interpolation_q_method` field to the `ModelParams` data class. This parameter controls the interpolation methodology when interpolation against gridded specific humidity. The possible values are:
  - `None`: Interpolate linearly against specific humidity. This is the default behavior and is the same as the previous behavior.
  - `"cubic-spline"`: Apply cubic-spline scaling to the interpolation table vertical coordinate before interpolating linearly against specific humidity.
  - `"log-q-log-p"`: Interpolate in the log-log domain against specific humidity and pressure.

  This interpolation parameter is used when calling `pycontrails.core.models.interpolate_met`. It can also be used directly with the new lower-level `pycontrails.core.models.interpolate_gridded_specific_humidity` function.
- Add new experimental `HistogramMatching` humidity scaling model to match RHi values against IAGOS observations. The previous `HistogramMatchingWithEckel` scaling is still available when working with ERA5 ensemble members.
- Add new [tutorial](https://py.contrails.org/tutorials/interpolating-specific-humidity.html) discussing the new specific humidity interpolation methodology.

### Breaking changes

- Add an optional `q_method` parameter to the `pycontrails.core.models.interpolate_met` function. The default value `None` agrees with the previous behavior.
- Change function signatures in the `cocip.py` module for consistency. The `interp_kwargs` parameter is now unpacked in the `calc_timestep_meterology` signature. Rename `kwargs` to `interp_kwargs` where appropriate.
- Remove the `cache_size` parameter in `MetDataset.from_zarr`. Previously this parameter allowed the user to wrap the Zarr store in a `LRUCacheStore` to improve performance. Changes to Zarr internals have broken this approach. Custom Zarr patterns should now be handled outside of `pycontrails`.

### Fixes

- Recompute and extend quantiles for histogram matching humidity scaling. Quantiles are now available for each combination of `q_method` and the following ERA5 data products: reanalysis and ensemble members 0-9. This data is available as a parquet file and is packaged with `pycontrails`.
- Fix the precomputed Eckel coefficients. Previous values where computed for different interpolation assumptions and were not correct for the default interpolation method.
- Clip the scaled humidity values computed by the `humidity_scaling.eckel_scaling` function to ensure that they are non-negative. Previously, both relative and specific humidity values arising from Eckel scaling could be negative.
- Handle edge case of all NaN values in the `T_critical_sac` function in the `sac.py` module.
- Avoid extraneous copy when calling `VectorDataset.sum`.
- Officially support `numpy` v1.25.0.
- Set a `pytest-timeout` limit for tests in `tests/unit/test_ecmwf.py` to avoid hanging tests.
- Add `forecast_step` parameter to the `ACCF` model.

### Internals

- Refactor auxillary functions used by `HistogramMatchingWithEckel` to better isolated histogram matching from Eckel scaling.
- Refactor `intersect_met` method in `pycontrails.core.models` to handle experimental `q_method` parameter.
- Include a `q_method` field in `Model.interp_kwargs`.
- Include precomputed humidity lapse rate values in the new `pycontrails.core.models._load_spline` function.
- Move the `humidity_scaling.py` module into its own subdirectory within `pycontrails/models`.

## v0.42.2

Re-release of [v0.42.1](#v0421).

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

### Features

- Add a `pycontrails.core.airport` module to read and process the global airport database, which can be used to identify the nearest airport to a given coordinate.
- Add a `pycontrails.datalib.spire.clean` function to remove and address erroneous waypoints in the raw Spire ADS-B data.
- Add a `pycontrails.datalib.spire.filter_altitude` function to remove noise in cruise altitude.
- Add a `pycontrails.datalib.spire.identify_flights` function to identify unique flight trajectories from ADS-B messages.
- Add a `pycontrails.datalib.spire.validate_trajectory` function to check the validity of the identified trajectories from ADS-B messages.
- Add a `FlightPhase` integer `Enum` in the `flight` module. This includes a new `level_flight` flight phase.

### Internals

- Add unit tests providing examples to identify unique flights.
- Rename `flight._dt_waypoints` -> `flight.segment_duration`.
- Move `jet.rate_of_climb_descent` -> `flight.segment_rocd`.
- Move `jet.identify_phase_of_flight` -> `flight.segment_phase`.
- Update `FlightPhase` to be a dictionary enumeration of flight phases.
- Add references to [`traffic` library](https://traffic-viz.github.io/).

## v0.41.0

Improve polygon algorithms.

### Features

- Rewrite the `polygon` module to run computation with [opencv](https://docs.opencv.org/4.x/index.html) in place of [scikit-image](https://scikit-image.org/) for finding contours. This change improves the algorithm runtime and fixes some previous unstable behavior in finding nested contours. For an introduction to the methodology, see the [OpenCV contour tutorial](https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html).

### Breaking changes

- Completely rewrite the `polygon` module. Replace the "main" public function `polygon.find_contours_to_depth` with `polygon.find_multipolygon`. Replace the `polygon.contour_to_lat_lon` function with `polygon.multipolygon_to_geojson`. Return `shapely` objects when convenient to do so.
- Convert continuous data to binary for polygon computation.
- Remove parameters `min_area_to_iterate` and `depth` in the `MetDataArray.to_polygon_feature` method. The `depth` parameter has been replaced by the boolean `interiors` parameter. Add a `properties` parameter for adding properties to the `Polygon` and `MultiPolygon` features. The `max_area` and `epsilon` parameters are now expressed in terms of latitude-longitude degrees.

### Internals

- Add `opencv-python-headless>=4.5` as an optional "vis" dependency. Some flavor of `opencv` is required for the updated polygon algorithms.

## v0.40.1

### Fixes

- Use [oldest-supported-numpy](https://pypi.org/project/oldest-supported-numpy/) for building pycontrails wheels. This allows pycontrails to be compatible with environments that use old versions of numpy. The [pycontrails v0.40.0 wheels](https://pypi.org/project/pycontrails/0.40.0/#files) are not compatible with numpy 1.22.

## v0.40.0

Support scipy 1.10, improve interpolation performance, and fix many windows issues.

### Features

- Improve interpolation performance by cythonizing linear interpolation. This extends the approach taken in [scipy 1.10](https://github.com/scipy/scipy/pull/17291). The pycontrails [cython routines](pycontrails/core/rgi_cython.pyx) allow for both float64 and float32 grids via cython fused types (the current scipy implementation assumes float64). In addition, interpolation up to dimension 4 is supported (the scipy implementation supports dimension 1 and 2).
- Officially support [scipy 1.10](https://scipy.github.io/devdocs/release/1.10.0-notes.html).
- Officially test on windows in the GitHub Actions CI.
- Build custom wheels for python 3.9, 3.10, and 3.11 for the following platforms:
  - Linux (x86_64)
  - macOS (arm64 and x86_64)
  - Windows (x86_64)

### Breaking changes

- Change `MetDataset` and `MetDataArray` conventions: underlying dimension coordinates are automatically promoted to float64.
- Change how datetime arrays are converted to floating values for interpolation. The new approach introduces small differences compared with the previous implementation. These differences are significant enough to see relative differences in CoCiP predictions on the order of 1e-4.

### Fixes

- Unit tests no longer raise errors when the `pycontrails-bada` package is not installed. Instead, some tests are skipped.
- Fix many numpy casting issues encountered on windows.
- Fix temp file issues encountered on windows.
- Officially support changes in `xarray` 2023.04 and `pandas` 2.0.

### Internals

- Make the `interpolation` module more aligned with [scipy 1.10 enhancements](https://docs.scipy.org/doc/scipy/release.1.10.0.html#scipy-interpolate-improvements) to the `RegularGridInterpolator`. In particular, grid coordinates now must be float64.
- Use [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/) to build wheels for Linux, macOS (arm64 and x86_64), and Windows on [release](.github/workflows/release.yaml) in Github Actions. Allow this workflow to be triggered manually to test the release process without actually publishing to PyPI.
- Simplify interpolation with pre-computed indices (invoked with the model parameter `interpolation_use_indices`) via a `RGIArtifacts` interface.
- Overhaul much of the interpolation module to improve performance.
- Slight performance enhancements to the `met` module.

## v0.39.6

### Features

- Add `geo.azimuth` and `geo.segment_azimuth` functions to calculate the azimuth between coordinates. Azimuth is the angle between coordinates relative to true north on the interval `[0, 360)`.

### Fixes

- Fix edge case in polygon algorithm by utilizing the `fully_connected` parameter in `measure.find_contours`. This update leads to slight changes in interior contours in some cases.
- Fix hard-coded POSIX path in `conftest.py` for windows compatibility.
- Address `PermissionError` raised by `shutil.copy` when the destination file is open in another thread. The copy is skipped and a warning is logged.
- Fix some unit tests in `test_vector.py` and `test_ecmwf.py` for windows compatibility. There are still a few tests that fail on windows (unrelated to changes in this release) that will be fixed in v0.40.0.
- Allow `cachestore=None` to skip caching in `ERA5`, `HRES`, and `GFS` interfaces. Previously a default `DiskCacheStore` was created even when `cachestore=None`. By default, caching is still enabled.

### Internals

- Clean up `geo` module docstrings.
- Add Wikipedia reference for [Azimuth](https://en.wikipedia.org/wiki/Azimuth).
- Convert `MetBase._load` to from a class method to a function.

## v0.39.5

### Fixes

- Fix `docs/examples/CoCiP.ipynb` example demonstrating aircraft performance integration
- Fix unit test caused by breaking change in [pyproj 3.5.0](https://github.com/pyproj4/pyproj/releases/tag/3.5.0)

### Internals

- Add additional Zenodo metadata
- Execute notebook examples in [Docs Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml)

## v0.39.4

### Internals

- Add additional Zenodo metadata
- Add [Doc / Notebook test Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/doctest.yaml) to run notebook test (`make nb-test`) and doctests (`make doctest`) on pull requests.
- Update [Docs Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml) to use python 3.11.

## v0.39.3

### Fixes

- Update links in the [README](README.md).
- Update the [release guide](RELEASE.md) to include a checklist of steps to follow when cutting a release.

## v0.39.2

### Fixes

- Fix links in documentation website.
- Deploy build to [PyPI](https://pypi.org/project/pycontrails/) (in addition to Test PyPI) on release.

## v0.39.1

### Features

- Use [setuptools_scm](https://github.com/pypa/setuptools_scm) to manage the `pycontrails` version.

## v0.39.0

### Features

- Add Apache-2 LICENSE and NOTICE files (#21)
- Add [CONTRIBUTING](https://github.com/contrailcirrus/pycontrails/blob/main/CONTRIBUTING.md) document
- Update documentation website [py.contrails.org](https://py.contrails.org).
  Includes `install` and `develop` guides, update citations
  and many other small improvements (#19).
- Initiate the [Github Discussion](https://github.com/contrailcirrus/pycontrails/discussions) forum (#22).

### Fixes

- Fix erroneous docstrings in `emissions.py` (#25)

### Internals

- Add Github action to push to `pypi` on tag (#3)
- Replace `flake8` with `ruff` linter
- Add `nb-clean` to pre-commit hooks for example notebooks
- Add `doc8` rst linter and pre-commit hook

## v0.38.0

### Breaking changes

- Change default value of `epsilon` parameter in method `MetDataArray.to_polygon_feature` from 0.15 to 0.0.
- Change the polygon simplification algorithm. The new implementation uses `shapely.buffer` and doesn't try to preserve the topology of the simplified polygon. This change may result in slightly different polygon geometries.

### Internals

- Add `depth` parameter to `MetDataArray.to_polygon_feature` to control the depth of the contour searching.
- Add experimental `convex_hull` parameter to `MetDataArray.to_polygon_feature` to control whether to take the convex hull of each contour.
- Warn if `iso_value` is not specified in `MetDataArray.to_polygon_feature`.

### Fixes

- Consolidate three redundant implementations of standardizing variables into a single `met.standardize_variables`.
- Ensure simplified polygons returned by `MetDataArray.to_polygon_feature` are disjoint. While non-disjoint polygons don't violate the GeoJSON spec, they can cause problems in some applications.

## v0.37.3

### Internals

- Add citations for ISA calculations
- Abstract functionality to convert a Dataset or DataArray to longitude coordinates `[-180, 180)` into `core.met.shift_longitude`. Add tests for method.
- Add auto-formatting checks to CI testing

## v0.37.2

ACCF integration updates

### Fixes

- Fixes ability to evaluate ACCF model over a `MetDataset` grid by passing in tuple of (dims, data) when assigning data
- Fixes minor issues with ACCF configuration and allows more configuration options
- Updates example ACCF notebook with example of how to set configuration options when evaluating ACCFs over a grid

## v0.37.1

### Features

- Include "rhi" and "iwc" variables in `CocipGrid` verbose outputs.

## v0.37.0

### Breaking changes

- Update CoCiP unit test static results for breaking changes in tau cirrus calculation. The relative difference in pinned energy forcing values is less than 0.001%.

### Fixes

- Fix geopotential height gradient calculation in the `tau_cirrus` module. When calculating finite differences along the vertical axis, the tau cirrus model previously divided the top and bottom differences by 2. To numerically approximate the derivative at the top and bottom levels, these differences should have actually been divided by 1. The calculation now uses `np.gradient` to calculate the derivative along the vertical axis, which handles this correctly.
- Make tau cirrus calculation slightly more performant.
- Include a warning in the suspect `IFS._calc_geopotential` implementation.

### Internals

- Remove `_deprecated.tau_cirrus_alt`. This function is now the default `tau_cirrus` calculation. The original `tau_cirrus` calculation is still available in the `_deprecated` module.
- Run `flake8` over test modules.
