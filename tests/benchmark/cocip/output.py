"""
Output benchmark test results.

```
$ python output.py
```
"""

import logging
import pathlib
import sys

import benchmark

# logging set up in benchmark.py
LOG = logging.getLogger("pycontrails")

# get output path from cli
if len(sys.argv) > 1:
    OUTPUT_PATH = pathlib.Path(sys.argv[1])
else:
    OUTPUT_PATH = benchmark.OUTPUT_PATH

LOG.info("Running Cocip over flights")

# run model on benchmark inputs
cocip = benchmark.run_cocip()

LOG.info(f"Writing outputs to {OUTPUT_PATH}")

# -------------
# Flight output
# -------------

fleet = benchmark.parse_flight_results(cocip.source)

# output one file per flight id
pathlib.Path(OUTPUT_PATH / "flight").mkdir(parents=True, exist_ok=True)
for fid, df in fleet.groupby("flight_id"):
    # prefer parquet
    df.reset_index(drop=True).to_parquet(OUTPUT_PATH / "flight" / f"{fid}.pq", index=False)

    # # Output csv
    # # fill na with -9999
    # df.reset_index(drop=True) \
    #   .fillna(value=-9999) \
    #   .to_csv(OUTPUT_PATH / "flight" / f"{fid}.csv", index=False)


# ---------------
# Contrail output
# ---------------

contrail = benchmark.parse_contrail_results(cocip.contrail)

# output one file per flight id
pathlib.Path(OUTPUT_PATH / "contrail").mkdir(parents=True, exist_ok=True)
for fid, df in contrail.groupby("flight_id"):
    # prefer parquet
    df.reset_index(drop=True).to_parquet(OUTPUT_PATH / "contrail" / f"{fid}.pq", index=False)

    # # Output csv
    # # fill na with -9999
    # df.reset_index(drop=True) \
    #   .fillna(value=-9999) \
    #   .to_csv(OUTPUT_PATH / "contrail" / f"{fid}.csv", index=False)
