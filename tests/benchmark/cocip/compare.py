"""
Compare benchmark test outputs with previous outputs.

```
$ python compare.py
```
"""

import logging

import benchmark
import numpy as np
import pandas as pd

# relative tolerances for comparison
FLIGHT_TOLERANCE = 1e-6
CONTRAIL_TOLERANCE = 1e-3

# logging set up in benchmark.py
LOG = logging.getLogger("pycontrails")

LOG.info("Running Cocip over flights")

# run model on benchmark inputs
cocip = benchmark.run_cocip()

LOG.info("Comparing outputs with previous results")


# -----------------
# Flight comparison
# -----------------

fleet = benchmark.parse_flight_results(cocip.source)

for fid, df in fleet.groupby("flight_id"):
    df = df.reset_index(drop=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / "flight" / f"{fid}.pq")

    # test all number types
    for col in df.select_dtypes(include=np.number):
        np.testing.assert_allclose(
            df_prev[col].to_numpy(), df[col].to_numpy(), rtol=FLIGHT_TOLERANCE
        )

LOG.info(f"Successfully compared {len(fleet)} flight rows")

# -------------------
# Contrail comparison
# -------------------


contrail = benchmark.parse_contrail_results(cocip.contrail)

for fid, df in contrail.groupby("flight_id"):
    df = df.reset_index(drop=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / "contrail" / f"{fid}.pq")

    # test all number types
    for col in df.select_dtypes(include=np.number):
        np.testing.assert_allclose(
            df_prev[col].to_numpy(), df[col].to_numpy(), rtol=CONTRAIL_TOLERANCE
        )

LOG.info(f"Successfully compared {len(contrail)} contrail rows")
