"""Support for the Poll-Schumann (PS) aircraft performance model."""

from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.models.ps_model.ps_model import PSModel, PSModelParams

__all__ = ["PSModel", "PSModelParams", "PSAircraftEngineParams"]
