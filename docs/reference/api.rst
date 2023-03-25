
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
    Aircraft
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


.. Contrail Grid
.. """""""""""""

.. .. autosummary::
..     :toctree: api/

..     models.cocipgrid.CocipGrid
..     models.cocipgrid.CocipGridParams

Aircraft Performance
""""""""""""""""""""

*In development. Currently used as a base class for the `BADA extension <#bada>`_*

.. autosummary::
    :toctree: api/

    models.aircraft_performance

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


.. Extensions
.. ----------

.. BADA
.. """"

.. Requires `pycontrails-bada <https://github.com/contrailcirrus/pycontrails-bada>`_ extension and data files obtained `BADA license <https://www.eurocontrol.int/model/bada>`_.


.. .. autosummary::
..     :toctree: api/

..     ext.bada.bada_model
..     ext.bada.bada3
..     ext.bada.bada4
..     ext.bada.BADAFlight
..     ext.bada.BADAFlightParams
..     ext.bada.BADAGrid
..     ext.bada.BADAGridParams
..     ext.bada.BADA3
..     ext.bada.BADA4
