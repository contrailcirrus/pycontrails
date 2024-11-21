
API Reference
=============

.. note this must be manually updated to refer to new/changed module names

.. currentmodule:: pycontrails


Data
----

Meteorology
"""""""""""

.. autosummary::
    :toctree: api/

    MetDataset
    MetDataArray

Vector Data
"""""""""""

.. autosummary::
    :toctree: api/

    VectorDataset
    GeoVectorDataset

Flight & Aircraft
"""""""""""""""""

.. autosummary::
    :toctree: api/

    Flight
    Fleet
    FlightPhase
    Fuel
    JetA
    SAFBlend
    HydrogenFuel



Datalib
-------


ECMWF
"""""

.. autosummary::
    :toctree: api/

    datalib.ecmwf.ERA5
    datalib.ecmwf.era5_model_level
    datalib.ecmwf.HRES
    datalib.ecmwf.hres_model_level
    datalib.ecmwf.hres.get_forecast_filename
    datalib.ecmwf.model_levels
    datalib.ecmwf.IFS
    datalib.ecmwf.variables


GFS
"""

.. autosummary::
    :toctree: api/

    datalib.gfs.GFSForecast
    datalib.gfs.variables


ARCO ERA5
"""""""""

.. autosummary::
    :toctree: api/

    datalib.ecmwf.arco_era5


GOES
""""

.. autosummary::
    :toctree: api/

    datalib.goes


Low Earth Orbit Satellites
""""""""""""""""""""""""""

.. autosummary::
    :toctree: api/

    datalib.landsat
    datalib.sentinel


Models
------

Base Classes
""""""""""""
.. autosummary::
    :toctree: api/

    Model
    ModelParams


SAC, ISSR & PCC
"""""""""""""""

.. autosummary::
    :toctree: api/

    models.issr
    models.sac
    models.pcr
    models.pcc


CoCiP
"""""

.. autosummary::
    :toctree: api/

    models.cocip.Cocip
    models.cocip.CocipParams
    models.cocip.CocipFlightParams
    models.cocip.contrail_properties
    models.cocip.radiative_forcing
    models.cocip.wake_vortex
    models.cocip.wind_shear


Gridded CoCiP
"""""""""""""

.. autosummary::
    :toctree: api/

    models.cocipgrid.CocipGrid
    models.cocipgrid.CocipGridParams


Dry Advection
"""""""""""""

.. autosummary::
    :toctree: api/

    models.dry_advection.DryAdvection
    models.dry_advection.DryAdvectionParams


ACCF
""""

    This model is an interface over the DLR / UMadrid `climaccf <https://github.com/dlr-pa/climaccf>`__ package.
    See :ref:`accf-install` for more information.

.. autosummary::
    :toctree: api/

    models.accf.ACCF
    models.accf.ACCFParams


APCEMM
""""""

    This model is an interface to the MIT `APCEMM <https://github.com/MIT-LAE/APCEMM>`__ plume model.

.. autosummary::
    :toctree: api/

    models.apcemm.APCEMM
    models.apcemm.APCEMMParams


Aircraft Performance
""""""""""""""""""""

.. autosummary::
    :toctree: api/

    core.aircraft_performance
    models.ps_model.PSFlightParams
    models.ps_model.PSFlight
    models.ps_model.PSGrid
    models.ps_model.PSAircraftEngineParams
    models.ps_model.ps_nominal_grid


Emissions
"""""""""

.. autosummary::
    :toctree: api/

    models.emissions.Emissions
    models.emissions.black_carbon
    models.emissions.ffm2


Cirrus Optical Depth (:math:`\tau_{cirrus}`)
""""""""""""""""""""""""""""""""""""""""""""

.. autosummary::
    :toctree: api/

    models.tau_cirrus

Humidity Scaling
""""""""""""""""

.. autosummary::
    :toctree: api/

    models.humidity_scaling


Physics
-------

.. autosummary::
    :toctree: api/

    physics.constants
    physics.thermo
    physics.jet
    physics.geo
    physics.units


Cache
-----

.. autosummary::
    :toctree: api/

    DiskCacheStore
    GCPCacheStore


Core
----

.. autosummary::
    :toctree: api/

    core.airports
    core.cache
    core.coordinates
    core.fleet
    core.flight
    core.fuel
    core.interpolation
    core.met
    core.met_var
    core.models
    core.polygon
    core.vector


Utilities
---------

.. autosummary::
    :toctree: api/

    utils.types
    utils.iteration
    utils.temp
    utils.json
    .. utils.synthetic_flight


Extensions
----------

.. _bada-extension:

BADA
""""

    Requires `pycontrails-bada <https://github.com/contrailcirrus/pycontrails-bada>`__ extension and data files obtained through `BADA license <https://www.eurocontrol.int/model/bada>`__.
    See :ref:`BADA Extension <bada-install>` for more information.

.. autosummary::
    :toctree: api/

    ext.bada.bada_model
    ext.bada.BADAFlight
    ext.bada.BADAFlightParams
    ext.bada.BADAGrid
    ext.bada.BADAGridParams
    ext.bada.BADA3
    ext.bada.BADA4
