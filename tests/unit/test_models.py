"""Test pycontrails/core/models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pycontrails import Flight, MetDataArray, MetDataset, Model, ModelParams
from pycontrails.core.met_var import AirTemperature, MetVariable, SpecificHumidity


@dataclass
class ModelTestGridParams(ModelParams):
    """Grid test model params."""

    param1: str = "param1"
    param2: int = 1


@dataclass
class ModelTestModelParams(ModelParams):
    """Flight test model."""

    param1: str = "param1"
    param2: int = 1
    met_latitude_buffer: tuple[float, float] = (5, 5)


class ModelTestGrid(Model):
    """Test grid-like model."""

    name = "grid"
    long_name = "my grid model"
    default_params = ModelTestGridParams
    met: MetDataset
    met_required = True
    met_variables = (AirTemperature,)
    source: MetDataset

    def eval(self, source: None = None, **params: Any) -> MetDataArray:
        self.set_source()
        self.update_params(params)
        self.source.data["temp"] = self.met.data["air_temperature"]
        return self.source["temp"]


class ModelTestFlight(Model):
    """Test flight-like model."""

    name = "flight"
    long_name = "my flight model"
    default_params = ModelTestModelParams
    met_variables = (AirTemperature,)
    source: Flight

    def eval(self, source: Flight = None, **params: Any) -> Flight:
        self.set_source(source)
        self.update_params(params)
        self.downselect_met()
        return self.source


class ModelTestPresets(Model):
    """Test grid-like model with params."""

    name = "grid"
    long_name = "my grid model with param sets"
    default_params = ModelTestModelParams

    def eval(self, source: None = None, **params: Any) -> MetDataArray:
        pass


class ModelTestNoDefaultParams(Model):
    """Test grid-like model with no params."""

    name = "grid"
    long_name = "no default params"

    def eval(self, source: None = None, **params: Any) -> MetDataArray:
        pass


class ModelTestNoName(Model):
    """Test grid-like model with no name."""

    myname = "grid"

    def eval(self, source: None = None, **params: Any) -> MetDataArray:
        pass


OtherVar = MetVariable(short_name="o", standard_name="other")


class ModelTestGridMissingRequiredVariables(Model):
    """Test grid-like model missing required variables."""

    name = "grid"
    long_name = "missing required met variables"
    met_variables = [AirTemperature, OtherVar]

    def eval(self, source: None = None) -> MetDataArray:
        pass


class ModelTestFlightMissingRequiredVariables(Model):
    """Test flight-like model missing required variables."""

    name = "flight"
    long_name = "missing required met variables"
    met_variables = [AirTemperature, OtherVar]
    default_params = ModelParams

    def eval(self, source: None = None) -> MetDataArray:
        pass


# ----------
# ModelBase Tests
# ----------


def test_model_needs_names(met_era5_fake: MetDataset) -> None:
    """Model requires name properties defined."""
    with pytest.raises(TypeError, match="abstract methods long_name, name"):
        ModelTestNoName(met_era5_fake)  # type: ignore[abstract]


def test_model_bad_params() -> None:
    """Ensure an error is raised when a model sees an unknown parameter."""
    model = ModelTestFlight(interpolation_method="nearest")
    assert isinstance(model, Model)

    with pytest.raises(KeyError, match="interp_method"):
        ModelTestFlight(interp_method="nearest")


def test_model_copy_source(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Model copy_source."""
    flight_model = ModelTestFlight(met_era5_fake, copy_source=False)
    flight_model.eval(source=flight_fake)
    assert flight_model.source is flight_fake

    flight_model2 = ModelTestFlight(met_era5_fake)
    flight_model2.eval(source=flight_fake)
    assert flight_model2.source is not flight_fake
    assert flight_model.source.data == flight_fake.data


def test_model_type_guards(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Model require_met and require_source_type."""

    model = ModelTestFlight(met=None)
    with pytest.raises(ValueError, match="Meteorology"):
        model.require_met()

    model = ModelTestFlight(met=met_era5_fake)
    met = model.require_met()
    assert met is model.met

    with pytest.raises(ValueError, match="pycontrails.core.flight.Flight"):
        model.require_source_type(Flight)

    model.set_source(flight_fake)
    model.require_source_type(Flight)


# ----------------
# Grid-like Tests
# ----------------


def test_model_grid_params() -> None:
    """Model with parameters defined."""
    params = ModelTestGridParams(param1="test")
    params_dict = params.as_dict()
    assert params_dict["param1"] == "test"
    assert params_dict["param2"] == 1


def test_model_no_default_params(met_era5_fake: MetDataset) -> None:
    """Model with no parameters defined."""
    model = ModelTestNoDefaultParams(met=met_era5_fake)
    default_params = ModelParams().as_dict()
    assert model.params == default_params


def test_model_grid_init(met_era5_fake: MetDataset) -> None:
    """Model __init__."""
    model = ModelTestGrid(met=met_era5_fake)
    assert model.name == "grid"
    assert model.long_name == "my grid model"
    assert "param1" in model.params and model.params["param1"] == "param1"
    assert "param2" in model.params and model.params["param2"] == 1

    # override params via `params` dict and kwargs
    model = ModelTestGrid(met=met_era5_fake, params={"param1": "new_param1"}, param2=2)
    assert model.params["param1"] == "new_param1"
    assert model.params["param2"] == 2

    # update grid params
    model.update_params({"param1": "test2"})
    assert model.params["param1"] == "test2"


def test_model_grid_required_met_variables(met_era5_fake: MetDataset) -> None:
    """Model with required variables."""
    model = ModelTestGrid(met=met_era5_fake)
    assert isinstance(model.met, MetDataset)

    with pytest.raises(KeyError, match="other"):
        ModelTestGridMissingRequiredVariables(met_era5_fake)


def test_model_grid_hash(met_era5_fake: MetDataset) -> None:
    """Model hash."""
    grid_model = ModelTestGrid(met=met_era5_fake)
    assert grid_model.hash == "28648ed00ce10e05aaa198325a8991ef79910d47"


def test_model_met_not_copied(met_era5_fake: MetDataset) -> None:
    """Model met is never copied."""
    grid_model = ModelTestGrid(met=met_era5_fake)
    assert grid_model.met is met_era5_fake

    met_era5_fake.data = met_era5_fake.data.rename({"air_temperature": "renamed"})
    assert "air_temperature" not in grid_model.met.data and "renamed" in grid_model.met.data


def test_model_met_grid_eval(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Model met is required for ModelTestGrid."""
    grid_model = ModelTestGrid(met=met_era5_fake)
    out = grid_model.eval()

    # source is based on met, but does not contain met data vars
    assert grid_model.source is not grid_model.met
    assert grid_model.source.coords is not grid_model.met.coords
    for k in grid_model.source.coords:
        np.testing.assert_array_equal(grid_model.source.coords[k], grid_model.met.coords[k])

    assert "temp" in grid_model.source and "air_temperature" not in grid_model.source
    assert np.all(grid_model.source["temp"].values == met_era5_fake.data["air_temperature"].values)
    assert np.all(out.data.values == met_era5_fake.data["air_temperature"].values)

    with pytest.raises(
        ValueError, match="Meteorology is required for this model. Specify with ModelTestGrid"
    ):
        ModelTestGrid()

    # override params on eval
    assert grid_model.params["param1"] == "param1"
    grid_model.eval(param1="override")
    assert grid_model.params["param1"] == "override"


# -----------------
# Flight-like Tests
# -----------------


def test_model_flight_params() -> None:
    """Model params."""
    params = ModelTestModelParams(param1="test")
    params_dict = params.as_dict()
    assert params_dict["param1"] == "test"
    assert params_dict["param2"] == 1

    # default interpolation parameters
    assert "interpolation_method" in params_dict and params_dict["interpolation_method"] == "linear"
    assert (
        "interpolation_bounds_error" in params_dict
        and params_dict["interpolation_bounds_error"] is False
    )
    assert "interpolation_fill_value" in params_dict and np.isnan(
        params_dict["interpolation_fill_value"]
    )

    # default met params
    assert params_dict["downselect_met"] is True
    assert params_dict["met_longitude_buffer"] == (0, 0)
    assert params_dict["met_level_buffer"] == (0, 0)
    assert params_dict["met_time_buffer"] == (np.timedelta64(0, "h"), np.timedelta64(0, "h"))

    # override default params from FlighParams
    assert params_dict["met_latitude_buffer"] == (5, 5)


def test_model_flight_init(met_era5_fake: MetDataset) -> None:
    """Model __init__."""
    model = ModelTestFlight(met_era5_fake)
    assert model.name == "flight"
    assert model.long_name == "my flight model"
    assert "param1" in model.params and model.params["param1"] == "param1"
    assert "param2" in model.params and model.params["param2"] == 1

    # default params get set
    assert (
        "interpolation_method" in model.params and model.params["interpolation_method"] == "linear"
    )

    # override params via `params` dict and kwargs
    model = ModelTestFlight(
        met_era5_fake, params={"param1": "new_param1"}, param2=2, interpolation_method="nearest"
    )
    assert model.params["param1"] == "new_param1"
    assert model.params["param2"] == 2
    assert model.params["interpolation_method"] == "nearest"

    model.update_params({"param1": "test2"})
    assert model.params["param1"] == "test2"


def test_model_flight_required_met_variables(met_era5_fake: MetDataset) -> None:
    """Model required met variables."""
    model = ModelTestFlight(met_era5_fake)
    assert isinstance(model.met, MetDataset)

    with pytest.raises(KeyError, match="other"):
        ModelTestFlightMissingRequiredVariables(met_era5_fake)


def test_model_flight_hash(met_era5_fake: MetDataset) -> None:
    """Ensure pinned hash matches as check for model degradation."""
    flight_model = ModelTestFlight(met_era5_fake)
    assert flight_model.hash == "2789e8bef2606322984d6733e8072f501507f114"


def test_model_flight_met_not_copied(met_era5_fake: MetDataset) -> None:
    """Model met never copied."""
    flight_model = ModelTestFlight(met_era5_fake)
    assert flight_model.met is met_era5_fake

    met_era5_fake.data = met_era5_fake.data.rename({"air_temperature": "renamed"})
    assert "air_temperature" not in flight_model.met.data and "renamed" in flight_model.met.data


def test_model_flight_eval(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Model eval."""
    flight_model = ModelTestFlight(met_era5_fake)
    assert np.all(flight_model.eval(source=flight_fake)["latitude"] == flight_fake["latitude"])

    # override params on eval
    assert flight_model.params["param1"] == "param1"
    flight_model.eval(source=flight_fake, param1="override")
    assert flight_model.params["param1"] == "override"


def test_model_flight_downselect_met(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Model downselect_met."""
    # clip flight domain so it is definitely smaller than met
    mask = (flight_fake.level < 300) & (flight_fake.level > 200)
    fl2 = flight_fake.filter(mask)

    flight_model = ModelTestFlight(
        met=met_era5_fake,
        met_longitude_buffer=(15, 15),
        met_latitude_buffer=(5, 10),
        met_level_buffer=(0, 0),
        met_time_buffer=(0, np.timedelta64(1, "h")),
    )
    # When eval is called, the model will downselect met
    _ = flight_model.eval(source=fl2)

    assert flight_model.met is not None
    assert flight_model.met is not met_era5_fake
    assert flight_model.met.data["longitude"].max() == 153
    assert flight_model.met.data["latitude"].max() == 88
    assert flight_model.met.data["level"].max() == 300
    assert flight_model.met.data["time"].max() == np.datetime64("2020-01-01T03:00:00")

    assert flight_model.met.data["longitude"].min() == 61
    assert flight_model.met.data["latitude"].min() == 39
    assert flight_model.met.data["level"].min() == 200
    assert flight_model.met.data["time"].min() == np.datetime64("2020-01-01T01:00:00")

    # don't downselect met if param is False
    flight_model = ModelTestFlight(met_era5_fake, downselect_met=False)
    _ = flight_model.eval(source=fl2)
    assert flight_model.met is met_era5_fake


###########################
# Test `intersect_met_variables`` method
###########################


@dataclass
class VerifyMetParams(ModelParams):
    """Model params to test `verify_met`."""

    verify_met: bool = False


class ModelTestVerifyMet(Model):
    """Model to test `verify_met`."""

    name = "flight"
    long_name = "two variable flight model"
    default_params = VerifyMetParams
    met_variables = [AirTemperature, SpecificHumidity]

    source: Flight | MetDataset

    def eval(self, source: Flight | MetDataset) -> Flight | MetDataset:
        self.set_source(source)
        self.set_source_met()
        return self.source


def test_verify_met_all_vars_on_met(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test `intersect_met_variables` method on `ModelTestVerifyMet`."""
    model = ModelTestVerifyMet(met=met_era5_fake)
    model._verify_met()
    out = model.eval(source=flight_fake)
    assert isinstance(out, Flight)

    out = model.eval(source=met_era5_fake)
    assert isinstance(out, MetDataset)


@pytest.mark.parametrize("var", ModelTestVerifyMet.met_variables)
def test_verify_met_one_vars_on_met(
    met_era5_fake: MetDataset, flight_fake: Flight, var: MetVariable
) -> None:
    """Test `set_source_met` method on `ModelTestVerifyMet` with single met variable."""
    met = met_era5_fake.copy()
    del met.data[var.standard_name]
    assert var.standard_name not in met
    model = ModelTestVerifyMet(met=met)
    with pytest.raises(KeyError, match=var.standard_name):
        model._verify_met()

    flight_fake[var.standard_name] = np.ones(flight_fake.size)
    out = model.eval(flight_fake)
    assert isinstance(out, Flight)

    out = model.eval(source=met_era5_fake)
    assert isinstance(out, MetDataset)


def test_verify_met_when_met_none(flight_fake: Flight) -> None:
    """Test `intersect_met_variables` method on `ModelTestVerifyMet` with `met=None`."""
    model = ModelTestVerifyMet(met=None)

    with pytest.raises(KeyError):
        model.eval(flight_fake)
    flight_fake["air_temperature"] = np.ones(flight_fake.size)

    with pytest.raises(KeyError, match="specific_humidity"):
        model.eval(flight_fake)
    flight_fake["specific_humidity"] = np.zeros(flight_fake.size)

    out = model.eval(flight_fake)
    assert isinstance(out, Flight)
