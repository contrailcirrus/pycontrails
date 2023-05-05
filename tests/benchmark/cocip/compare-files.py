"""
Compare existing benchmark test outputs with previous outputs.

```
$ python compare-files.py
```
"""
import logging
import pathlib
import sys

import benchmark
import numpy as np
import pandas as pd

# get the current output path from cli
if len(sys.argv) == 2:
    COMPARE_OUTPUT_PATH = pathlib.Path(sys.argv[1])
else:
    raise RuntimeError("Must prescribe output directory 'python compare-files.py <new-output-dir>'")

# logging set up in benchmark.py
LOG = logging.getLogger("pycontrails")

LOG.info(
    f"Comparing results in {COMPARE_OUTPUT_PATH} with previous results in {benchmark.OUTPUT_PATH}"
)

ATOL = 1e-8  # TODO: fix for smaller value columns
RAISE = True

# -----------------
# Flight comparison
# -----------------

result_type = "flight"
pathlib.Path(f"diff/{result_type}").mkdir(parents=True, exist_ok=True)

for path in pathlib.Path(COMPARE_OUTPUT_PATH / result_type).glob("*.pq"):
    # read new results
    df = pd.read_parquet(path)
    df.sort_index(axis=1, inplace=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / result_type / path.name)
    df_prev.sort_index(axis=1, inplace=True)

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
        # output comparison between two
        try:
            comp = df.compare(df_prev)
            comp.to_csv(f"diff/{result_type}/{path.name}.comp.csv", index=False)
        except ValueError:
            print(f"Failed to compare flight for {path.name}")

        # create diff for all number types
        diff = df.copy()
        for col in df.select_dtypes(include=np.number):
            diff[col] = df[col] - df_prev[col]

            # try to replace all values within tolerance with 0
            # this will fail if the rows don't align
            try:
                isclose = np.isclose(
                    df[col].to_numpy(),
                    df_prev[col].to_numpy(),
                    atol=ATOL,
                    rtol=benchmark.FLIGHT_TOLERANCE,
                    equal_nan=True,
                )
                diff.loc[isclose, [col]] = None
            except ValueError:
                pass

        diff.to_csv(f"diff/{result_type}/{path.name}.diff.csv", index=False)

        if RAISE:
            raise e

# -------------------
# Contrail comparison
# -------------------

result_type = "contrail"
pathlib.Path(f"diff/{result_type}").mkdir(parents=True, exist_ok=True)

for path in pathlib.Path(COMPARE_OUTPUT_PATH / result_type).glob("*.pq"):
    # read new results
    df = pd.read_parquet(path)
    df.sort_index(axis=1, inplace=True)

    # read previous results
    df_prev = pd.read_parquet(benchmark.OUTPUT_PATH / result_type / path.name)
    df_prev.sort_index(axis=1, inplace=True)

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
        # output comparison between two
        try:
            comp = df.compare(df_prev)
            comp.to_csv(f"diff/{result_type}/{path.name}.comp.csv", index=False)
        except ValueError:
            print(f"Failed to compare contrail for {path.name}")

        # create diff for all number types
        diff = df.copy()
        for col in df.select_dtypes(include=np.number):
            diff[col] = df[col] - df_prev[col]

            # try to replace all values within tolerance with 0
            # this will fail if the rows don't align
            try:
                isclose = np.isclose(
                    df[col].to_numpy(),
                    df_prev[col].to_numpy(),
                    atol=ATOL,
                    rtol=benchmark.CONTRAIL_TOLERANCE,
                    equal_nan=True,
                )
                diff.loc[isclose, [col]] = None
            except ValueError:
                pass

        diff.to_csv(f"diff/{result_type}/{path.name}.diff.csv", index=False)

        if RAISE:
            raise e
