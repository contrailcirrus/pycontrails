"""
Compare benchmark test outputs with previous outputs.

```
$ python compare-run.py
```
"""

import logging

import benchmark
import numpy as np
import pandas as pd

# logging set up in benchmark.py
LOG = logging.getLogger("pycontrails")

LOG.info("Running Cocip over flights")

# run model on benchmark inputs
cocip = benchmark.run_cocip()

LOG.info("Comparing outputs with previous results")

ATOL = 1e-8  # TODO: fix for smaller value columns


# -----------------
# Flight comparison
# -----------------

fleet = benchmark.parse_flight_results(cocip.source)

for fid, df in fleet.groupby("flight_id"):
    df = df.reset_index(drop=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / "flight" / f"{fid}.pq")

    try:
        # test all number types
        for col in df.select_dtypes(include=np.number):
            np.testing.assert_allclose(
                df[col].to_numpy(),
                df_prev[col].to_numpy(),
                atol=ATOL,
                rtol=benchmark.FLIGHT_TOLERANCE,
                equal_nan=True,
                err_msg=f"Column: {col}",
            )
    except AssertionError as e:
        print(f"Flight {fid} failed comparison")
        raise e

LOG.info(f"Successfully compared {len(fleet)} flight rows")

# -------------------
# Contrail comparison
# -------------------

contrail = benchmark.parse_contrail_results(cocip.contrail)

for fid, df in contrail.groupby("flight_id"):
    df = df.reset_index(drop=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / "contrail" / f"{fid}.pq")

    try:
        # test all number types
        for col in df.select_dtypes(include=np.number):
            np.testing.assert_allclose(
                df[col].to_numpy(),
                df_prev[col].to_numpy(),
                atol=ATOL,
                rtol=benchmark.CONTRAIL_TOLERANCE,
                equal_nan=True,
                err_msg=f"Column: {col}",
            )

    except AssertionError as e:
        print(f"Contrail for flight {fid} failed comparison")
        raise e

LOG.info(f"Successfully compared {len(contrail)} contrail rows")
