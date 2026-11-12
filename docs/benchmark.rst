CoCiP Benchmark
===============

..
   Figures and tables come from ``python docs/benchmark/generate.py``.
   Changes are in ``docs/benchmark/changes.toml``.

Running CoCiP periodically on a fixed scenario helps identify both intended and unintended changes.
Early in its development, pycontrails was evaluated against existing implementations of CoCiP
:cite:`schumannContrailCirrusPrediction2012`. Because bug fixes and model improvements have since
changed its results, this page compares pycontrails releases against one another. Every release
runs the identical CoCiP simulation, so changes in model output and runtime can be traced directly
to the release that made them.

The simulation covers 19,326 flights over Europe on 30 January 2020, using the ADS-B flight
trajectories of GAIA :cite:`teohHighresolutionGlobalAviation2024` and ERA5 model-level meteorology.
January 2020 predates the COVID-19 decline in air traffic, and according to ERA5, the 30th had the
largest share of persistent contrail regions over Europe of any day that month
(`select_day.py`_). For each release, the benchmark records CoCiP's output for every flight
alongside the totals across all flights (`run_benchmark.py`_).

.. _select_day.py:
   https://github.com/contrailcirrus/pycontrails/blob/main/tests/benchmark/select_day.py

.. _run_benchmark.py:
   https://github.com/contrailcirrus/pycontrails/blob/main/tests/benchmark/run_benchmark.py

The benchmark uses the CoCiP parameter choices of :cite:`teohGlobalAviationContrail2024`. Some
parameters, such as ``max_age``, have existed since the first pycontrails release, while others
were only added later, such as ``unterstrasser_ice_survival_fraction`` in 0.50.1. Each release runs
with the parameters available in it.

Every run uses a c3-highmem-4 Google Compute Engine instance (4 vCPUs, 32 GB of memory), with the
latest Python version the release supports and the package dependencies available on its release
date.

When a release fails on some flights, for example because it does not support their aircraft type,
the benchmark excludes them. If the skipped flights cannot be identified in advance, the benchmark
applies a patch from a later release instead. The table below lists the number of excluded flights,
and :download:`benchmark_results.csv <benchmark/_generated/benchmark_results.csv>` lists the patches
applied. Releases 0.9.0 and 0.15.1 are omitted entirely, as they fail on all or nearly all flights.

Releases before 0.11.0 have no ``Fleet``-mode and run CoCiP one flight at a time, which would take
about 40 hours for all flights. Instead, these releases run on a fixed random sample of 2,000
flights, and their totals and runtime are scaled by 19,326 / 2,000. When 0.11.0 runs on the same
sample, its scaled energy forcing is 5.5% above its result on all flights, which indicates the size
of the sampling error.

Releases 0.19.1 to 0.28.2 use too much memory to run ``Fleet``-mode on all flights at once, so they
run it on 64 batches of flights, one after another. Their runtime is the sum of the batch runtimes.

In the plots below, the numbered markers link to a row in the table of changes.

Results
-------

Flights excluded
~~~~~~~~~~~~~~~~

Number of flights excluded because the release fails on them.

.. raw:: html
   :file: benchmark/_generated/flights.svg

Totals
~~~~~~

Total energy forcing, contrail length times age, length of flight segments forming persistent
contrails, and fuel burn over all flights evaluated with CoCiP.

.. raw:: html
   :file: benchmark/_generated/results.svg

Energy forcing per flight
~~~~~~~~~~~~~~~~~~~~~~~~~

Distribution of the energy forcing per flight for selected releases. Each histogram shows the
fraction of the flights run in each bin. Flights with energy forcing between -10\ :sup:`10` and
10\ :sup:`10` J, which include those with none, are left out.

.. raw:: html
   :file: benchmark/_generated/ef_histograms.svg

Runtime Performance
-------------------

Time spent running CoCiP, not counting loading the flights and meteorology, and peak memory of the
process. Runtimes of releases run on the sample are scaled to all flights.

.. raw:: html
   :file: benchmark/_generated/performance.svg

Changes
-------

.. include:: benchmark/_generated/changes.inc

All results
-----------

Consecutive releases with equal results share a row, with the range of their runtimes and peak
memory. :download:`benchmark_results.csv <benchmark/_generated/benchmark_results.csv>` has every
field of every release's summary, including the Python and dependency versions, the machine, and
the sample details.

.. include:: benchmark/_generated/results.inc
