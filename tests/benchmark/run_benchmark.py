"""Run the CoCiP benchmark with the installed pycontrails.

    python tests/benchmark/run_benchmark.py

Write ``summary.json`` and ``flight_summary.csv`` to ``OUTPUT``.
"""

from __future__ import annotations

import collections
import collections.abc
import functools
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pathlib
import platform
import re
import resource
import shutil
import sys
import tarfile
import tempfile
import time
import types
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

import pycontrails

INPUTS = "gs://contrails-301217-benchmark-data/inputs"

# Track versions of these core dependencies
DEPENDENCIES = ("numpy", "scipy", "pandas", "xarray", "pycontrails-bada")
PYCONTRAILS_VERSION = tuple(int(x) for x in pycontrails.__version__.split(".")[:3])

# Package of the geo, units and constants modules, which are top-level before v0.5.0
PHYSICS = "pycontrails.physics" if PYCONTRAILS_VERSION >= (0, 5, 0) else "pycontrails"

# Number of batches for Cocip Fleet-mode (workaround for early pycontrails memory explosion)
N_BATCHES = int(os.environ.get("N_BATCHES", "1"))

# Fixes from later releases applied to the installed pycontrails or pycontrails-bada
PATCHES: list[str] = []

# Releases that raise for every flight (or nearly every flight) and that the benchmark leaves out
BROKEN_RELEASES = {(0, 9, 0), (0, 15, 1)}

# Number of flights, chosen at random with seed SAMPLE_SEED, to run for versions without Fleet,
# and for versions with Fleet if SAMPLE_FLIGHTS is set to 1
SAMPLE_SIZE = 2000
SAMPLE_SEED = 0
SAMPLE_FLIGHTS = os.environ.get("SAMPLE_FLIGHTS", "0") == "1"

OUTPUT = f"gs://contrails-301217-benchmark-data/results/{pycontrails.__version__}"
if SAMPLE_FLIGHTS:
    OUTPUT += "/sample"

# For a run on a sample: the sample size, the number of staged flights and the met processing
# time in seconds, which summarize uses to extrapolate the totals to all staged flights
SAMPLE: dict[str, Any] = {}

HUMIDITY_SCALING_PARAMS: dict[str, Any] = {"level_type": "model"}
AP_PARAMS: dict[str, Any] = {
    "fill_low_altitude_with_isa_temperature": True,
    "fill_low_altitude_with_zero_wind": True,
    "max_mach_buffer": -0.03,
}
COCIP_PARAMS: dict[str, Any] = {
    "preprocess_lowmem": True,
    "min_altitude_m": 6500.0,
    "max_altitude_m": 15000.0,
    "vpm_activation": True,
    "unterstrasser_ice_survival_fraction": True,
    "radiative_heating_effects": True,
    "dt_integration": np.timedelta64(5, "m"),
    "max_age": np.timedelta64(12, "h"),
    "max_depth": None,
    "max_horizontal_diffusivity": 100.0,
    "max_vertical_diffusivity": 10.0,
}

# First release with each parameter of AP_PARAMS, COCIP_PARAMS and HUMIDITY_SCALING_PARAMS that
# earlier releases do not have. Earlier releases run without the parameter.
PARAM_SINCE: dict[str, tuple[int, int, int]] = {
    "max_mach_buffer": (0, 63, 0),
    "fill_low_altitude_with_isa_temperature": (0, 52, 2),
    "fill_low_altitude_with_zero_wind": (0, 52, 2),
    "max_horizontal_diffusivity": (0, 60, 2),
    "max_vertical_diffusivity": (0, 60, 2),
    "vpm_activation": (0, 55, 0),
    "preprocess_lowmem": (0, 52, 3),
    "unterstrasser_ice_survival_fraction": (0, 50, 1),
    "max_depth": (0, 46, 0),
    "radiative_heating_effects": (0, 28, 9),
    "level_type": (0, 51, 0),
}


def available_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return ``params`` without the parameters that the installed pycontrails does not have."""
    return {
        k: v
        for k, v in params.items()
        if k not in PARAM_SINCE or PARAM_SINCE[k] <= PYCONTRAILS_VERSION
    }


def add_empty_plotting_modules() -> None:
    """Add empty matplotlib and cartopy modules to ``sys.modules`` if they are not installed.

    Older pycontrails import ``matplotlib.pyplot`` or ``cartopy.crs``. Neither is used.
    """
    for name, submodule in (("matplotlib", "matplotlib.pyplot"), ("cartopy", "cartopy.crs")):
        if importlib.util.find_spec(name) is None:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[submodule] = types.ModuleType(submodule)


def skip_overrides_signature_check() -> None:
    """Monkeypatch the overrides package to skip its signature check, if it is installed."""
    overrides_module: Any
    try:
        overrides_module = importlib.import_module("overrides.overrides")
    except ImportError:
        return
    overrides_module.ensure_signature_is_compatible = lambda *args, **kwargs: None


def open_met(fs: gcsfs.GCSFileSystem, name: str) -> pycontrails.MetDataset:
    """Download ``name`` from ``INPUTS`` and load it as a MetDataset."""
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        fs.download(f"{INPUTS}/{name}", tmp.name)
        ds = xr.open_dataset(tmp.name).load()
    return pycontrails.MetDataset(ds)


def emissions_engine_uid(em: Any, fl: pycontrails.Flight) -> str | None:
    """Return the engine UID that the emissions model ``em`` uses for ``fl``."""
    engine_uid = fl.attrs.get("engine_uid")
    if engine_uid is None and hasattr(em, "default_engines"):
        try:
            engine_uid = em.default_engines.at[fl.attrs.get("aircraft_type"), "engine_uid"]
        except KeyError:
            return None
    return engine_uid


def edb_fuel_flow_not_increasing(em: Any, fl: pycontrails.Flight) -> bool:
    """Return True if the EDB fuel flows for ``fl`` are not strictly increasing.

    A NaN fuel flow counts as not increasing. Return False if the EDB has no entry for the
    engine that the emissions model ``em`` uses for ``fl``.
    """
    engine_uid = emissions_engine_uid(em, fl)
    edb = getattr(em, "edb_engine_gaseous", {}).get(engine_uid) if engine_uid else None
    if edb is None:
        return False
    fuel_flow = np.array([edb.ff_7, edb.ff_30, edb.ff_85, edb.ff_100], dtype=float)
    return not np.all(np.diff(fuel_flow) > 0.0)


def convert_rad_to_flux(rad: pycontrails.MetDataset) -> None:
    """Convert the hourly ERA5 radiation accumulations in ``rad`` from J m**-2 to W m**-2.

    Follow ``cocip._rad_accumulation_to_average_instantaneous`` from pycontrails v0.48.0,
    which divides by 3600 s.

    From v0.6.0 to v0.47.3, CoCiP reads the values as W m**-2.
    """
    for name, da in rad.data.data_vars.items():
        if da.attrs.get("units") == "J m**-2":
            rad.data[name] = (da / 3600.0).assign_attrs({**da.attrs, "units": "W m**-2"})


def add_toa_incident_solar_radiation(rad: pycontrails.MetDataset) -> None:
    """Add the hourly ERA5 accumulation ``toa_incident_solar_radiation`` to ``rad`` in J m**-2.

    pycontrails v0.5.0 and earlier require this variable, which the staged ``rad`` does not have.
    """
    geo = importlib.import_module(f"{PHYSICS}.geo")

    ref = rad.data["top_net_solar_radiation"]
    longitude, latitude = np.meshgrid(ref["longitude"], ref["latitude"], indexing="ij")

    # Midpoints of the twelve 5-minute intervals in the hour before each time (the hour over
    # which ERA5 accumulates radiation). The 5-minute integration step is an arbitrary numerical
    # approximation choice
    offsets = (
        np.arange(12) * np.timedelta64(300, "s") + np.timedelta64(150, "s") - np.timedelta64(1, "h")
    )
    accumulation = np.empty((*longitude.shape, ref["time"].size))
    for i, valid_time in enumerate(ref["time"].values):
        sdr = [
            geo.solar_direct_radiation(
                longitude.ravel(), latitude.ravel(), np.full(longitude.size, valid_time + offset)
            )
            for offset in offsets
        ]
        accumulation[..., i] = np.mean(sdr, axis=0).reshape(longitude.shape) * 3600.0

    da = xr.DataArray(
        accumulation,
        coords={"longitude": ref["longitude"], "latitude": ref["latitude"], "time": ref["time"]},
        dims=("longitude", "latitude", "time"),
    )
    rad.data["toa_incident_solar_radiation"] = (
        da.broadcast_like(ref)
        .transpose(*ref.dims)
        .astype(ref.dtype)
        .assign_attrs(units="J m**-2", long_name="TOA incident solar radiation")
    )


def add_geopotential(met: pycontrails.MetDataset) -> None:
    """Add geopotential computed from the ISA altitude of each pressure level to ``met``.

    Follow ``tau_cirrus._geopotential_height`` of pycontrails v0.48.0.
    """
    constants = importlib.import_module(f"{PHYSICS}.constants")
    units = importlib.import_module(f"{PHYSICS}.units")

    air_temperature = met.data["air_temperature"]
    level = met.data["level"]
    altitude = units.pl_to_m(level.values)
    height = altitude * constants.radius_earth / (constants.radius_earth + altitude)
    geopotential = xr.DataArray(constants.g * height, coords={"level": level}, dims="level")
    met.data["geopotential"] = (
        geopotential.broadcast_like(air_temperature)
        .transpose(*air_temperature.dims)
        .astype(air_temperature.dtype)
    )


def bada3_covers(bada3: Any, aircraft_type: str, engine_uid: str | None) -> bool:
    """Return True if ``bada3`` has aircraft and engine parameters for the ``aircraft_type``."""
    if PYCONTRAILS_VERSION < (0, 11, 0):
        try:
            bada3.get_aircraft_engine_properties(aircraft_type)
        except KeyError:
            return False
        return True
    if not bada3.check_aircraft_type_availability(aircraft_type, False):
        return False
    try:
        if PYCONTRAILS_VERSION < (0, 29, 0):
            bada3.get_aircraft_engine_properties(aircraft_type)
        else:
            bada3.get_aircraft_engine_properties(aircraft_type, engine_uid)
        if PYCONTRAILS_VERSION < (0, 16, 0):
            # BADA3 has no get_aircraft_params before v0.16.0
            aircraft_params = bada3.aircraft_param_dict[bada3.synonym_dict[aircraft_type]]
        else:
            aircraft_params = bada3.get_aircraft_params(aircraft_type)
    except KeyError:
        return False
    return PYCONTRAILS_VERSION >= (0, 25, 9) or aircraft_params.engine_type != "Piston"


def copy_bada3_for_split_ptf_parser() -> str:
    """Copy the BADA3 directory into a form that pycontrails v0.12.0 and earlier can read.

    These versions split lines 8 to 10 of each PTF file on spaces to read the calibrated airspeeds.
    Remove the spaces after the slash in these lines.
    Return the path of the copy.
    """
    src = pathlib.Path(*pycontrails.__path__).parents[1] / "bada" / "bada3"
    dst = pathlib.Path(tempfile.mkdtemp()) / "bada3"
    shutil.copytree(src, dst)
    for path in dst.glob("*.PTF"):
        lines = path.read_bytes().split(b"\n")
        for i in (7, 8, 9):
            lines[i] = re.sub(rb"/ +", b"/", lines[i])
        path.write_bytes(b"\n".join(lines))
    return str(dst)


def extract_edb_from_v0_2_0() -> dict[str, str]:
    """Extract the ICAO EDB files of pycontrails 0.2.0 for pycontrails 0.1.0.

    (pycontrails 0.1.0 doesn't include the EDB files)
    """
    fs = gcsfs.GCSFileSystem()
    dst = pathlib.Path(tempfile.mkdtemp())
    url = "gs://contrails-301217-benchmark-data/src/pycontrails-archive-0.2.0.tar.gz"
    params = {}
    with fs.open(url, "rb") as f, tarfile.open(fileobj=f, mode="r:gz") as tar:
        for param, name in (("edb_engine_path", "edb.csv"), ("edb_nvpm_path", "nvpm.csv")):
            member = tar.extractfile(f"pycontrails/models/cocip/static/{name}")
            assert member is not None
            path = dst / name
            path.write_bytes(member.read())
            params[param] = str(path)
    PATCHES.append("pycontrails 0.2.0: static/edb.csv and static/nvpm.csv as the EDB paths")
    return params


def use_bada3_only() -> None:
    """Monkeypatch ``BADAFlight`` to look up aircraft types in BADA3 only.

    Before pycontrails v0.15.0, ``BADAFlight.get_bada`` loads BADA4 first and raises
    ``FileNotFoundError`` when the BADA4 files are missing.
    """
    bada_model = importlib.import_module("pycontrails.models.bada.bada_model")

    def get_bada(self: Any, aircraft_type: str) -> Any:
        """Return BADA3, or raise ``KeyError`` if it does not cover ``aircraft_type``."""
        bada3 = bada_model.BADA3(self.params["bada3_path"])
        bada3.check_aircraft_type_availability(aircraft_type)
        return bada3

    bada_model.BADAFlight.get_bada = get_bada


def treat_level_flight_as_cruise() -> None:
    """Correct the fuel flow of level flight waypoints with the cruise bounds.

    The ``BADA.correct_fuel_flow`` of pycontrails-bada 0.3.0 and 0.3.1 leaves the fuel flow of
    waypoints in the ``LEVEL_FLIGHT`` phase.

    Apply the fix of pycontrails-bada 0.4.0 (commit a82dae5), which corrects them as cruise.
    """
    bada_interface = importlib.import_module("pycontrails_bada.bada_interface")
    flight = importlib.import_module("pycontrails.core.flight")
    correct_fuel_flow = bada_interface.BADA.correct_fuel_flow
    signature = inspect.signature(correct_fuel_flow)

    @functools.wraps(correct_fuel_flow)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        phase = np.asarray(bound.arguments["flight_phase"])
        level = phase == flight.FlightPhase.LEVEL_FLIGHT
        bound.arguments["flight_phase"] = np.where(level, flight.FlightPhase.CRUISE, phase).astype(
            phase.dtype
        )
        return correct_fuel_flow(*bound.args, **bound.kwargs)

    bada_interface.BADA.correct_fuel_flow = wrapper
    PATCHES.append("pycontrails-bada a82dae5: correct level flight fuel flow as cruise")


def edb_fuel_flow_repeated(em: Any, bada3: Any, fl: pycontrails.Flight) -> bool:
    """Return True if an FFM2 emissions profile repeats a fuel flow.

    Before v0.17.1, interpolating a profile with a repeated fuel flow raises ``InvalidIndexError``;
    v0.17.1 drops the repeated values (commit 98f02aaa).
    """
    aircraft_type_bada = bada3.get_aircraft_engine_properties(fl.attrs["aircraft_type"]).atyp_bada
    edb = em.edb_engine_gaseous.get(aircraft_type_bada)
    if edb is None:
        return False

    profiles = (edb.log_ei_nox_profile, edb.log_ei_co_profile, edb.log_ei_hc_profile)
    return any(not profile.indexes["log_fuel_flow"].is_unique for profile in profiles)


def engine_not_in_edb(em: Any, bada3: Any, fl: pycontrails.Flight) -> bool:
    """Return True if the engine that Cocip uses for ``fl`` is not in the EDB."""
    engine_uid = fl.attrs.get("engine_uid")
    if engine_uid is None:
        aircraft_type = fl.attrs["aircraft_type"]
        engine_uid = bada3.get_aircraft_engine_properties(aircraft_type).engine_uid_edb
    return engine_uid not in em.edb_engine_gaseous


def sample_flight_ids() -> list[str]:
    """Return ``SAMPLE_SIZE`` staged flight IDs chosen at random with seed ``SAMPLE_SEED``."""
    metadata = pd.read_parquet(f"{INPUTS}/metadata.pq")

    rng = np.random.default_rng(SAMPLE_SEED)
    flight_ids = rng.choice(np.sort(metadata["flight_id"].to_numpy()), SAMPLE_SIZE, replace=False)

    SAMPLE.update(sample_size=SAMPLE_SIZE, n_staged_flights=len(metadata), met_processing_s=0.0)
    return flight_ids.tolist()


def load_flights(flight_ids: list[str] | None = None) -> list[pycontrails.Flight]:
    """Build Flights from the staged waypoints and metadata.

    If ``flight_ids`` is given, build only these flights.
    """
    waypoints = pd.read_parquet(f"{INPUTS}/waypoints.pq")
    metadata = pd.read_parquet(f"{INPUTS}/metadata.pq").set_index("flight_id")
    if flight_ids is not None:
        waypoints = waypoints[waypoints["flight_id"].isin(flight_ids)]

    flights = []
    for flight_id, df in waypoints.groupby("flight_id"):
        attrs = metadata.loc[flight_id].dropna().to_dict()
        flights.append(pycontrails.Flight(df, flight_id=flight_id, **attrs))
    return flights


def without_nvpm_interpolation_errors(
    met: pycontrails.MetDataset,
    flights: list[pycontrails.Flight],
    fleet_params: dict[str, Any],
    bada3_path: str | None,
) -> list[pycontrails.Flight]:
    """Return ``flights`` without those for which the EDB nvPM interpolation raises."""
    from pycontrails.models.emissions import emissions

    fleet_module = importlib.import_module("pycontrails.core.fleet")
    fleet = fleet_module.Fleet.from_seq(flights, **fleet_params)
    air_temperature = fleet.intersect_met(
        met["air_temperature"], method="linear", bounds_error=False, fill_value=np.nan
    )
    all_nan = pd.Series(np.isnan(air_temperature)).groupby(fleet["flight_id"]).all()
    no_met = set(all_nan.index[all_nan])

    em = emissions.Emissions()
    bada3 = importlib.import_module("pycontrails.models.bada.bada3").BADA3(bada3_path)

    def raises(fl: pycontrails.Flight) -> bool:
        """Return True if the EDB nvPM interpolation raises for ``fl``."""
        if fl.attrs["flight_id"] not in no_met:
            return False
        aircraft_type_bada = bada3.get_aircraft_engine_properties(
            fl.attrs["aircraft_type"]
        ).atyp_bada
        return em._check_edb_nvpm_availability(aircraft_type_bada, raise_error=False)

    return [fl for fl in flights if not raises(fl)]


# The columns of the flight and contrail outputs that summarize_flights reads
FLIGHT_SUMMARY_COLUMNS = (
    "flight_id",
    "time",
    "sac",
    "segment_length",
    "fuel_burn",
    "fuel_flow",
    "persistent_1",
    "ef",
)
CONTRAIL_SUMMARY_COLUMNS = ("flight_id", "ef", "segment_length", "continuous")


def summary_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy of the ``columns`` of ``df`` that it has."""
    return df[[c for c in columns if c in df]].copy()


def run_cocip_fleet(
    met: pycontrails.MetDataset,
    rad: pycontrails.MetDataset,
    ap_params: dict[str, Any] = AP_PARAMS,
    cocip_params: dict[str, Any] = COCIP_PARAMS,
    drop_flight: collections.abc.Callable[[pycontrails.Flight], bool] | None = None,
    meem: bool = True,
    humidity_scaling_class: str | None = "HistogramMatching",
    humidity_scaling_params: dict[str, Any] = HUMIDITY_SCALING_PARAMS,
    bada_only: bool = False,
    fleet_params: dict[str, Any] | None = None,
    builtin_bada: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP on the staged flights with pycontrails v0.11.0 and later.

    With the default arguments, run pycontrails v0.63.0 and later. Earlier versions
    pass their arguments.

    Pass ``ap_params`` to the PS and BADA3 aircraft performance models, ``cocip_params`` to
    CoCiP and ``humidity_scaling_params`` to the ``humidity_scaling_class`` model.

    If ``humidity_scaling_class`` is None, don't pass any humidity scaling model.

    Drop the flights for which ``drop_flight`` returns True, and for v0.11.0 to v0.20.1 those of
    :func:`without_nvpm_interpolation_errors`.

    If ``meem`` is False, do not pass an emissions model, so CoCiP uses its default.

    If ``bada_only`` is True, use BADA3 for every aircraft type it covers instead of the PS model.

    Pass ``fleet_params`` to ``Fleet.from_seq``.

    If ``builtin_bada`` is True, as for pycontrails v0.33 and earlier, CoCiP runs BADA itself from
    ``pycontrails.models.bada``.
    """
    from pycontrails.models.cocip import cocip

    ap_params = available_params(ap_params)
    cocip_params = available_params(cocip_params)
    humidity_scaling_params = available_params(humidity_scaling_params)
    ap: Any = None
    covers: collections.abc.Callable[[str, str | None], bool]
    bada = None if builtin_bada else importlib.import_module("pycontrails.ext.bada")
    if bada is not None and importlib.metadata.version("pycontrails-bada").startswith("0.3."):
        treat_level_flight_as_cruise()
    if builtin_bada:
        # The pycontrails.models.bada package exports BADA3 from v0.34.0
        bada3_module = importlib.import_module("pycontrails.models.bada.bada3")
        covers = functools.partial(bada3_covers, bada3_module.BADA3(cocip_params.get("bada3_path")))
    elif bada_only:
        ap = bada.BADAFlight(**ap_params)
        covers = functools.partial(bada3_covers, bada.BADA3(ap.params["bada3_path"]))
    else:
        from pycontrails.models import ps_model

        class PSFlightWithBADA3Backup(ps_model.PSFlight):
            """PS model with BADA3 for aircraft types the PS model does not cover."""

            def __init__(self, **params: Any) -> None:
                super().__init__(**params)
                self.bada_flight = bada.BADAFlight(**params)
                self.bada3 = bada.BADA3(self.bada_flight.params["bada3_path"])

            def eval_flight(self, fl: pycontrails.Flight) -> pycontrails.Flight:
                """Evaluate the PS model on ``fl``, or BADA3 if the PS model raises ``KeyError``."""
                try:
                    return super().eval_flight(fl)
                except KeyError:
                    # pycontrails-bada 0.7.2 to 0.7.7 read ``source``, which ``eval`` sets
                    self.bada_flight.source = fl
                    return self.bada_flight.eval_flight(fl)

            def _eval_flight(self, fl: pycontrails.Flight) -> pycontrails.Flight:
                """Do as :meth:`eval_flight` for pycontrails v0.54.5 and earlier.

                These versions evaluate each flight in ``PSFlight._eval_flight``, and their
                pycontrails-bada has no ``BADAFlight.eval_flight``.
                """
                try:
                    return super()._eval_flight(fl)  # type: ignore[misc]
                except KeyError:
                    return self.bada_flight.eval(source=fl, copy_source=False)

            def eval(self, source: Any = None, **params: Any) -> Any:
                """Do as :meth:`eval_flight` for pycontrails v0.54.1 and earlier.

                These versions evaluate the flight in ``PSFlight.eval``. In later versions, the
                ``KeyError`` is caught in :meth:`eval_flight` or :meth:`_eval_flight` instead.
                """
                try:
                    return super().eval(source, **params)
                except KeyError:
                    return self.bada_flight.eval(source, **params)

            def covers(self, aircraft_type: str, engine_uid: str | None) -> bool:
                """Return True if the PS model or BADA3 covers ``aircraft_type``."""
                if self.check_aircraft_type_availability(aircraft_type, raise_error=False):
                    return True
                return bada3_covers(self.bada3, aircraft_type, engine_uid)

        ap = PSFlightWithBADA3Backup(**ap_params)
        covers = ap.covers

    flights = load_flights(sample_flight_ids() if SAMPLE_FLIGHTS else None)
    keys = [(fl.attrs["aircraft_type"], fl.attrs.get("engine_uid")) for fl in flights]
    covered = {key for key in set(keys) if covers(*key)}
    kept = [fl for fl, key in zip(flights, keys) if key in covered]  # noqa: B905
    if drop_flight is not None:
        kept = [fl for fl in kept if not drop_flight(fl)]
    if (0, 11, 0) <= PYCONTRAILS_VERSION < (0, 20, 2):
        kept = without_nvpm_interpolation_errors(
            met, kept, fleet_params or {}, cocip_params.get("bada3_path")
        )

    models: dict[str, Any] = {}
    if humidity_scaling_class is not None:
        from pycontrails.models import humidity_scaling

        models["humidity_scaling"] = getattr(humidity_scaling, humidity_scaling_class)(
            **humidity_scaling_params
        )
    if ap is not None:
        models["aircraft_performance"] = ap
    if meem:
        from pycontrails.models import emissions

        models["emissions"] = emissions.Emissions(use_meem=True)

    # pycontrails exports Fleet from v0.28.1
    fleet_module = importlib.import_module("pycontrails.core.fleet")
    if PYCONTRAILS_VERSION < (0, 21, 0):
        # CoCiP takes a list of Flights and calls Fleet.from_seq itself, so make fleet_params the
        # defaults of Fleet.from_seq
        fleet_module.Fleet.from_seq = functools.partial(
            fleet_module.Fleet.from_seq, **(fleet_params or {})
        )
    dfs = []
    contrails = []
    runtime = 0.0
    for k, batch in enumerate(np.array_split(np.arange(len(kept)), N_BATCHES)):
        batch_flights = [kept[i] for i in batch]
        source: Any = batch_flights
        if PYCONTRAILS_VERSION >= (0, 21, 0):
            source = fleet_module.Fleet.from_seq(batch_flights, **(fleet_params or {}))

        model = cocip.Cocip(met=met, rad=rad, **models, **cocip_params)
        t0 = time.perf_counter()
        out = model.eval(source)
        batch_runtime = time.perf_counter() - t0
        runtime += batch_runtime
        print(
            f"Batch {k + 1} of {N_BATCHES}: {len(batch)} flights, {batch_runtime:.0f} s, "
            f"peak RSS {peak_rss_gb():.1f} GB",
            flush=True,
        )

        # Pycontrails v0.28.5 and earlier return a list of Flights for a Fleet
        fls = out if isinstance(out, list) else [out]
        dfs.extend(summary_columns(fl.dataframe, FLIGHT_SUMMARY_COLUMNS) for fl in fls)
        if model.contrail is not None:
            contrails.append(summary_columns(model.contrail, CONTRAIL_SUMMARY_COLUMNS))
        # Free this batch's model and outputs before the next batch runs
        del model, out, fls, source

    contrail = pd.concat(contrails) if contrails else None
    return pd.concat(dfs), contrail, runtime, len(flights) - len(kept)


def run_cocip_v0_61(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.61 and v0.62.

    The ``jet.cargo_load_factor`` of these versions raises ``ValueError`` for origin and
    destination regions missing from its cargo load factor table. Drop those flights.
    """
    from pycontrails.physics import jet

    def cargo_load_factor_raises(fl: pycontrails.Flight) -> bool:
        """Return True if ``jet.cargo_load_factor`` raises as called in ``estimate_payload``."""
        origin = fl.attrs.get("origin_airport_icao")
        destination = fl.attrs.get("destination_airport_icao")
        total_flight_dist = None
        if origin is None or destination is None:
            total_flight_dist = np.nansum(fl.segment_haversine()).item()
        aircraft_role = fl.attrs.get("aircraft_role")
        passenger_aircraft = pd.isna(aircraft_role) or aircraft_role == "Passenger"
        try:
            jet.cargo_load_factor(origin, destination, total_flight_dist, passenger_aircraft)
        except ValueError:
            return True
        return False

    return run_cocip_fleet(met, rad, drop_flight=cargo_load_factor_raises)


def run_cocip_v0_59_1(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.59.1 to v0.60.1.

    In v0.59.1 and v0.60.0, the MEEM2 nvPM number profile raises ``ValueError`` for engines
    with DAC, TAPS or TAPS II combustors, so drop flights with those engines for these two
    versions.

    Set ``vpm_activation`` to False, because the extended K15 model of v0.55.0 to v0.60.1
    runs out of memory.
    """
    from pycontrails.models import emissions

    em = emissions.Emissions()

    def staged_combustor(fl: pycontrails.Flight) -> bool:
        """Return True if the EDB nvPM engine that ``Emissions`` uses for ``fl`` is staged."""
        engine_uid = emissions_engine_uid(em, fl)
        edb_nvpm = em.edb_engine_nvpm.get(engine_uid) if engine_uid else None
        return edb_nvpm is not None and edb_nvpm.combustor in ("DAC", "TAPS", "TAPS II")

    cocip_params = {**COCIP_PARAMS, "vpm_activation": False}
    drop_flight = staged_combustor if PYCONTRAILS_VERSION < (0, 60, 1) else None
    return run_cocip_fleet(met, rad, cocip_params=cocip_params, drop_flight=drop_flight)


def run_cocip_v0_55(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.55 to v0.59.0.

    These versions also have no ``emissions`` CoCiP parameter and no MEEM2 nvPM model,
    so CoCiP uses its default emissions model.
    """
    cocip_params = {**COCIP_PARAMS, "vpm_activation": False}
    return run_cocip_fleet(met, rad, cocip_params=cocip_params, meem=False)


def run_cocip_v0_52_2(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.52.2 to v0.54.12.

    These versions have no ``emissions`` CoCiP parameter, so CoCiP uses its default.
    """
    return run_cocip_fleet(met, rad, meem=False)


def run_cocip_v0_44(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.44.0 to v0.52.1.

    These versions have no ``emissions`` CoCiP parameter, so CoCiP uses its default emissions
    model.

    Drop the flights whose EDB fuel flows are not increasing (see
    :func:`edb_fuel_flow_not_increasing`).

    Before v0.47.0, the PS model is ``PSModel`` rather than ``PSFlight``.

    Before v0.46.0, the ``max_depth`` CoCiP parameter is ``max_contrail_depth``.

    Before v0.45.0, ``Flight.copy`` returns a ``Flight`` for a ``Fleet``, which makes CoCiP
    reject the copied fleet's waypoints, so set ``copy_source`` to False.

    Up to v0.48.1, ``Fleet.from_seq`` copies numeric flight attributes to data variables
    and raises ``ValueError`` when flights have different ones, so keep them in attrs.
    """
    from pycontrails.models import emissions, ps_model

    if PYCONTRAILS_VERSION < (0, 47, 0):
        ps_model.PSFlight = ps_model.PSModel  # type: ignore[attr-defined,misc]
    em = emissions.Emissions()
    cocip_params = dict(COCIP_PARAMS)
    if PYCONTRAILS_VERSION < (0, 46, 0):
        cocip_params["max_contrail_depth"] = np.inf
    if PYCONTRAILS_VERSION < (0, 45, 0):
        cocip_params["copy_source"] = False
    return run_cocip_fleet(
        met,
        rad,
        cocip_params=cocip_params,
        drop_flight=lambda fl: edb_fuel_flow_not_increasing(em, fl),
        meem=False,
        fleet_params={"broadcast_numeric": False} if PYCONTRAILS_VERSION < (0, 49, 0) else None,
    )


def run_cocip_bada_only(
    met: pycontrails.MetDataset,
    rad: pycontrails.MetDataset,
    humidity_scaling_class: str | None,
    builtin_bada: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with BADA3 for every aircraft type it covers."""
    cocip_params = dict(COCIP_PARAMS)
    # The parameter is named copy_flight before v0.22.0
    cocip_params["copy_source" if PYCONTRAILS_VERSION >= (0, 22, 0) else "copy_flight"] = False
    if PYCONTRAILS_VERSION >= (0, 31, 4):
        cocip_params["max_contrail_depth"] = np.inf
    if PYCONTRAILS_VERSION < (0, 27, 0):
        cocip_params["rhi_adj"] = 0.97
    if PYCONTRAILS_VERSION < (0, 12, 3):
        cocip_params["bada3_path"] = copy_bada3_for_split_ptf_parser()
    if PYCONTRAILS_VERSION < (0, 15, 0):
        use_bada3_only()

    # The emissions package exports Emissions from v0.34.0
    from pycontrails.models.emissions import emissions

    em = emissions.Emissions()
    bada3: Any = None
    if PYCONTRAILS_VERSION == (0, 29, 0) or PYCONTRAILS_VERSION < (0, 17, 1):
        bada3_module = importlib.import_module("pycontrails.models.bada.bada3")
        bada3 = bada3_module.BADA3(cocip_params.get("bada3_path"))

    def drop_flight(fl: pycontrails.Flight) -> bool:
        """Return True if CoCiP raises for ``fl``."""
        if PYCONTRAILS_VERSION == (0, 29, 0) and engine_not_in_edb(em, bada3, fl):
            return True
        if PYCONTRAILS_VERSION < (0, 17, 1) and edb_fuel_flow_repeated(em, bada3, fl):
            return True
        return edb_fuel_flow_not_increasing(em, fl)

    return run_cocip_fleet(
        met,
        rad,
        cocip_params=cocip_params,
        drop_flight=drop_flight,
        meem=False,
        humidity_scaling_class=humidity_scaling_class,
        bada_only=True,
        fleet_params={"broadcast_numeric": False},
        builtin_bada=builtin_bada,
    )


def run_cocip_v0_43(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.43.0."""
    return run_cocip_bada_only(met, rad, "HistogramMatching")


def run_cocip_v0_34(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.34.0 to v0.42.2.

    These versions have no ``HistogramMatching``, so use
    ``ExponentialBoostLatitudeCorrectionHumidityScaling`` with its defaults.
    """
    return run_cocip_bada_only(met, rad, "ExponentialBoostLatitudeCorrectionHumidityScaling")


def run_cocip_v0_27(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails v0.27.0 to v0.33.8.

    These versions run BADA inside CoCiP, from ``pycontrails.models.bada``.
    """
    return run_cocip_bada_only(
        met, rad, "ExponentialBoostLatitudeCorrectionHumidityScaling", builtin_bada=True
    )


def run_cocip_v0_11(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP with pycontrails 0.11.0 to v0.26.0.

    These versions run BADA inside CoCiP, as :func:`run_cocip_v0_27` does,
    and have no humidity scaling models.
    """
    return run_cocip_bada_only(met, rad, None, builtin_bada=True)


def run_cocip_without_fleet(
    met: pycontrails.MetDataset, rad: pycontrails.MetDataset
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP on a sample of flights, one at a time, with pycontrails before v0.11.0.

    These versions have no Fleet, and a CoCiP model raises if it processes the same met twice.
    So process the met once, then run a new CoCiP model on each of `SAMPLE_SIZE` flights
    chosen at random with seed `SAMPLE_SEED`, and record the sample in `SAMPLE` for
    :func:`summarize`.

    Drop the flights for which CoCiP raises. Record the number of each error in `SAMPLE`.
    """
    from pycontrails.models.cocip import cocip

    flights = load_flights(sample_flight_ids())

    bada3_path = copy_bada3_for_split_ptf_parser()
    # BADA is in pycontrails.datalib before v0.5.0
    bada_package = (
        "pycontrails.models.bada"
        if PYCONTRAILS_VERSION >= (0, 5, 0)
        else "pycontrails.datalib.bada"
    )
    bada3 = importlib.import_module(f"{bada_package}.bada3").BADA3(bada3_path)
    # The emissions model is in pycontrails.models.emissions from v0.5.0
    em: Any = None
    if PYCONTRAILS_VERSION >= (0, 5, 0):
        em = importlib.import_module("pycontrails.models.emissions.emissions").Emissions()

    def drop_flight(fl: pycontrails.Flight) -> bool:
        """Return True if CoCiP raises for ``fl``."""
        if em is None:
            return False
        if hasattr(em, "edb_engine_gaseous") and edb_fuel_flow_repeated(em, bada3, fl):
            return True
        return edb_fuel_flow_not_increasing(em, fl)

    keys = [(fl.attrs["aircraft_type"], fl.attrs.get("engine_uid")) for fl in flights]
    covered = {key for key in set(keys) if bada3_covers(bada3, *key)}
    kept = [fl for fl, key in zip(flights, keys) if key in covered]  # noqa: B905
    kept = [fl for fl in kept if not drop_flight(fl)]

    rhi_adj = 0.97
    cocip_params = {
        "min_altitude_m": COCIP_PARAMS["min_altitude_m"],
        "max_altitude_m": COCIP_PARAMS["max_altitude_m"],
        "dt_integration": COCIP_PARAMS["dt_integration"],
        "max_age": COCIP_PARAMS["max_age"],
        "rhi_adj": rhi_adj,
        "copy_flight": False,
        "bada3_path": bada3_path,
        "process_met": False,
    }
    if PYCONTRAILS_VERSION < (0, 2, 0):
        cocip_params.update(extract_edb_from_v0_2_0())
    t0 = time.perf_counter()
    if hasattr(cocip.Cocip, "process_met_datasets"):
        met, rad = cocip.Cocip.process_met_datasets(met, rad, rhi_adj)
    else:
        # Before v0.3.0, a CoCiP model processes its own copies of met and rad
        processor = cocip.Cocip(met, rad, **{**cocip_params, "process_met": True})
        processor.process_met()
        met = processor.met
        rad = processor.rad
    SAMPLE["met_processing_s"] = time.perf_counter() - t0

    dfs = []
    contrails = []
    runtime = 0.0
    errors: collections.Counter[str] = collections.Counter()
    for fl in kept:
        flight_id = fl.attrs["flight_id"]
        model = cocip.Cocip(met, rad, **cocip_params)
        t0 = time.perf_counter()
        try:
            out = model.eval(fl)
        except Exception as err:
            errors[f"{type(err).__name__}: {err}"[:200]] += 1
            continue
        runtime += time.perf_counter() - t0
        # Before v0.3.0, a Flight holds its waypoints in the DataFrame ``data``
        out_df = out.dataframe if hasattr(out, "dataframe") else out.data
        dfs.append(out_df.assign(flight_id=flight_id))
        if model.contrail is not None:
            contrails.append(model.contrail.assign(flight_id=flight_id))
    SAMPLE["flight_errors"] = dict(errors)
    if not dfs:
        raise RuntimeError(f"CoCiP raised for every flight: {dict(errors)}")
    contrail = pd.concat(contrails) if contrails else None
    return pd.concat(dfs), contrail, runtime, len(flights) - len(dfs)


def run_cocip(
    met: pycontrails.MetDataset,
    rad: pycontrails.MetDataset,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float, int]:
    """Run CoCiP on the staged flights with the installed pycontrails version.

    Drop flights whose aircraft type the aircraft performance models do not cover.
    Return the flight output, the contrail output, the CoCiP runtime in seconds and
    the number of dropped flights.

    Raise ``RuntimeError`` for the releases in `BROKEN_RELEASES`.
    """
    if PYCONTRAILS_VERSION in BROKEN_RELEASES:
        raise RuntimeError(f"pycontrails {pycontrails.__version__} is left out of the benchmark")
    if PYCONTRAILS_VERSION < (0, 48, 0):
        add_geopotential(met)
        if PYCONTRAILS_VERSION >= (0, 6, 0):
            convert_rad_to_flux(rad)
        else:
            add_toa_incident_solar_radiation(rad)
    if PYCONTRAILS_VERSION >= (0, 63, 0):
        return run_cocip_fleet(met, rad)
    if PYCONTRAILS_VERSION >= (0, 61, 0):
        return run_cocip_v0_61(met, rad)
    if PYCONTRAILS_VERSION >= (0, 60, 2):
        return run_cocip_fleet(met, rad)
    if PYCONTRAILS_VERSION >= (0, 59, 1):
        return run_cocip_v0_59_1(met, rad)
    if PYCONTRAILS_VERSION >= (0, 55, 0):
        return run_cocip_v0_55(met, rad)
    if PYCONTRAILS_VERSION >= (0, 52, 2):
        return run_cocip_v0_52_2(met, rad)
    if PYCONTRAILS_VERSION >= (0, 44, 0):
        return run_cocip_v0_44(met, rad)
    if PYCONTRAILS_VERSION >= (0, 43, 0):
        return run_cocip_v0_43(met, rad)
    if PYCONTRAILS_VERSION >= (0, 34, 0):
        return run_cocip_v0_34(met, rad)
    if PYCONTRAILS_VERSION >= (0, 27, 0):
        return run_cocip_v0_27(met, rad)
    if PYCONTRAILS_VERSION >= (0, 11, 0):
        return run_cocip_v0_11(met, rad)
    return run_cocip_without_fleet(met, rad)


def peak_rss_gb() -> float:
    """Return the peak resident set size of this process in GB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def summarize_flights(df: pd.DataFrame, contrail: pd.DataFrame | None) -> pd.DataFrame:
    """Return per-flight totals of the CoCiP outputs ``df`` and ``contrail``, by ``flight_id``.

    Pycontrails v0.25.8 and earlier don't include ``fuel_burn``, so compute it as the
    ``Emissions`` model does. For a Fleet, pycontrails v0.19.0 and earlier include only
    the contrail results, so sum the contrail EF of each flight.
    """
    sac = df["sac"] == 1.0
    if "fuel_burn" in df:
        fuel_burn = df["fuel_burn"]
    else:
        dt_s = -df.groupby("flight_id")["time"].diff(-1).dt.total_seconds()
        fuel_burn = df["fuel_flow"] * dt_s

    waypoints = pd.DataFrame(
        {
            "flight_id": df["flight_id"],
            "n_waypoints": 1,
            "n_sac": sac,
            "sac_length_km": df["segment_length"].where(sac) / 1e3,
            "fuel_burn_kg": fuel_burn,
        }
    )
    if "persistent_1" in df:
        persistent = df["persistent_1"] == 1.0
        waypoints["n_persistent_waypoints"] = persistent
        waypoints["persistent_length_km"] = df["segment_length"].where(persistent) / 1e3
    if "ef" in df:
        waypoints["ef_J"] = df["ef"]

    flights = waypoints.groupby("flight_id").sum()

    if "persistent_1" not in df:
        flights["n_persistent_waypoints"] = np.nan
        flights["persistent_length_km"] = np.nan
    if "ef" not in df:
        flights["ef_J"] = 0.0
        if contrail is not None:
            by_flight = contrail["ef"].groupby(contrail["flight_id"]).sum()
            flights["ef_J"] = by_flight.reindex(flights.index, fill_value=0.0)

    flights["contrail_length_age_km_h"] = 0.0
    if contrail is not None:
        dt_h = (COCIP_PARAMS["dt_integration"] / np.timedelta64(1, "h")).item()
        length_age = contrail["segment_length"].where(contrail["continuous"]) / 1e3 * dt_h
        by_flight = length_age.groupby(contrail["flight_id"]).sum()
        flights["contrail_length_age_km_h"] = by_flight.reindex(flights.index, fill_value=0.0)

    columns = [
        "n_waypoints",
        "n_sac",
        "n_persistent_waypoints",
        "sac_length_km",
        "persistent_length_km",
        "fuel_burn_kg",
        "ef_J",
        "contrail_length_age_km_h",
    ]
    return flights[columns]


def dependency_version(name: str) -> str | None:
    """Return the installed version of ``name``, or None if the run did not use it.

    Count pycontrails-bada as used only if the run imported ``pycontrails.ext.bada``. Pycontrails
    v0.33.8 and earlier run BADA from ``pycontrails.models.bada``.
    """
    if name == "pycontrails-bada" and "pycontrails.ext.bada" not in sys.modules:
        return None
    return importlib.metadata.version(name)


def summarize(flights: pd.DataFrame, runtime: float, n_dropped: int) -> dict[str, Any]:
    """Return the run metadata and the totals of the per-flight summary ``flights``.

    For a run on a sample of flights, extrapolate to all staged flights:
    multiply the totals, the numbers of flights run and dropped and the CoCiP time of the flights
    by the number of staged flights over the sample size, add the met processing time to the
    runtime, and record the sample values under ``sample``. Do not scale the peak RSS.
    """
    sample = None
    scale = 1.0
    if SAMPLE:
        scale = SAMPLE["n_staged_flights"] / SAMPLE["sample_size"]
        sample = {
            **SAMPLE,
            "scale_factor": scale,
            "n_flights": len(flights),
            "n_dropped": n_dropped,
            "flights_runtime_s": runtime,
        }
        runtime = SAMPLE["met_processing_s"] + scale * runtime

    def extrapolate(value: float) -> float:
        """Return ``value`` multiplied by the scale for a run on a sample, else ``value``."""
        return value if sample is None else value * scale

    return {
        "pycontrails_version": pycontrails.__version__,
        # Commit time of the benchmarked pycontrails, set by vm_startup.sh
        "commit_time": os.environ.get("CUTOFF"),
        "python_version": platform.python_version(),
        **{f"{name.replace('-', '_')}_version": dependency_version(name) for name in DEPENDENCIES},
        "platform": platform.platform(),
        "machine_type": os.environ.get("MACHINE_TYPE"),
        "n_batches": N_BATCHES,
        "patches": PATCHES,
        "sample": sample,
        # CPUs available to this process; os.sched_getaffinity exists on Linux only
        "cpu_count": (
            len(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        ),
        "runtime_s": runtime,
        "peak_rss_gb": peak_rss_gb(),
        "n_flights": extrapolate(len(flights)),
        "n_dropped": extrapolate(n_dropped),
        # None where every flight is NaN
        **{
            name: None if flights[name].isna().all() else extrapolate(flights[name].sum().item())
            for name in flights
        },
    }


def main() -> None:
    """Run CoCiP on the staged flights and write the summaries."""
    add_empty_plotting_modules()
    skip_overrides_signature_check()

    fs = gcsfs.GCSFileSystem()
    met = open_met(fs, "met.nc")
    rad = open_met(fs, "rad.nc")
    df, contrail, runtime, n_dropped = run_cocip(met, rad)

    flights = summarize_flights(df, contrail)
    summary = summarize(flights, runtime, n_dropped)
    with fs.open(f"{OUTPUT}/flight_summary.csv", "w") as f:
        flights.to_csv(f)
    with fs.open(f"{OUTPUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
