
# Changelog

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
