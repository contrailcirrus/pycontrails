
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
    datalib.ecmwf.HRES
    datalib.ecmwf.IFS
    datalib.ecmwf.variables


GFS
"""

.. autosummary::
    :toctree: api/

    datalib.gfs.GFSForecast
    datalib.gfs.variables


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
    models.cocip.contrail_properties
    models.cocip.radiative_forcing
    models.cocip.wake_vortex
    models.cocip.wind_shear


ACCF
""""

    This model is a this interface over the DLR / UMadrid `climaccf <https://github.com/dlr-pa/climaccf>`__ package.
    See :ref:`accf-install` for more information.

.. autosummary::
    :toctree: api/

    models.accf.ACCF
    models.accf.ACCFParams


Aircraft Performance
""""""""""""""""""""

*In development*

.. autosummary::
    :toctree: api/

    models.aircraft_performance
    models.ps_model.PSModelParams
    models.ps_model.PSModel
    models.ps_model.PSAircraftEngineParams


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
    core.datalib
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
    ext.bada.bada3
    ext.bada.bada4
    ext.bada.BADAFlight
    ext.bada.BADAFlightParams
    ext.bada.BADAGrid
    ext.bada.BADAGridParams
    ext.bada.BADA3
    ext.bada.BADA4
