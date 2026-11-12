"""Write the figures and tables of the benchmark page from the benchmark results.

    python docs/benchmark/generate.py

Read the ``summary.json`` of each pycontrails release from the benchmark results bucket and the
hand-written notes in ``changes.toml``, and write the files that ``docs/benchmark.rst`` includes
to ``docs/benchmark/_generated``. See ``tests/benchmark/run_benchmark.py`` for the contents of
``summary.json``.
"""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import re
import tomllib
from typing import Any

import gcsfs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging import version

RESULTS = "contrails-301217-benchmark-data/results"
TIMES = "\N{MULTIPLICATION SIGN}"
DASH = "\N{EN DASH}"
HERE = pathlib.Path(__file__).parent
OUTPUT = HERE / "_generated"

# Legend label and color of each way the benchmark runs a release
RUN_MODES = {
    "fleet": ("Fleet-mode", "C0"),
    "batched": ("Batched Fleet-mode", "C2"),
    "sampled": ("Extrapolated from sample", "C1"),
}


@dataclasses.dataclass(frozen=True)
class Panel:
    """A single panel of a figure."""

    field: str
    name: str
    unit: str
    exponent: int = 0
    log: bool = False
    lower_is_better: bool = False

    @property
    def label(self) -> str:
        """Return the axis label: the name, the unit, and whether lower is better."""
        scale = f"$10^{{{self.exponent}}}$ " if self.exponent else ""
        unit = f" ({scale}{self.unit})" if self.unit else ""
        better = "\n(lower is better)" if self.lower_is_better else ""
        return self.name + unit + better


PANELS = {
    "dropped": Panel("n_dropped", "Flights dropped", "", lower_is_better=True),
    "ef": Panel("ef_J", "Energy forcing", "J", 18),
    "length_age": Panel("contrail_length_age_km_h", f"Contrail length\n{TIMES} age", "km h", 7),
    "persistent_length": Panel("persistent_length_km", "Persistent contrail\nlength", "km", 6),
    "fuel_burn": Panel("fuel_burn_kg", "Fuel burn", "kg", 7),
    "runtime": Panel("runtime_s", "CoCiP runtime", "s", log=True, lower_is_better=True),
    "peak_rss": Panel("peak_rss_gb", "Peak memory", "GB", lower_is_better=True),
}
FLIGHT_PANELS = ("dropped",)
RESULT_PANELS = ("ef", "length_age", "persistent_length", "fuel_burn")

# Relative change in percent from the previous release at which a change is marked on a panel.
# The dropped panel marks any change in the number of flights dropped.
CHANGE_THRESHOLDS = {
    "ef": 1.0,
    "length_age": 1.0,
    "persistent_length": 1.0,
    "fuel_burn": 0.5,
    "runtime": 10.0,
    "peak_rss": 10.0,
}
PERFORMANCE_PANELS = ("runtime", "peak_rss")

FIGURE_WIDTH = 7.5
PANEL_HEIGHT = 2.7
FONT_SIZE = 9

# Energy forcing in J below which the per-flight histograms leave flights out
EF_LINTHRESH = 1e10

# Relative change in total energy forcing from the previous release for which a release gets a
# per-flight energy forcing histogram
EF_HISTOGRAM_CHANGE = 0.05


def load_results(fs: gcsfs.GCSFileSystem) -> pd.DataFrame:
    """Return the results of each pycontrails release, in release order.

    Leave out development versions.
    """
    contents = fs.cat(fs.glob(f"{RESULTS}/*/summary.json"))
    df = pd.DataFrame([json.loads(content) for content in contents.values()])

    df["parsed_version"] = df["pycontrails_version"].map(version.Version)
    df = df[~df["parsed_version"].map(lambda v: v.is_devrelease)]
    df = df.sort_values("parsed_version", ignore_index=True)

    df["sampled"] = df["sample"].notna() if "sample" in df else False
    df["n_batches"] = df.get("n_batches", pd.Series(1, index=df.index)).fillna(1).astype(int)
    df["mode"] = np.select([df["sampled"], df["n_batches"] > 1], ["sampled", "batched"], "fleet")
    df["patches"] = df.get("patches", pd.Series(None, index=df.index)).map(
        lambda p: p if isinstance(p, list) else []
    )
    for panel in PANELS.values():
        df[panel.field] = df[panel.field].astype(float)

    missing = list(df.loc[df["commit_time"].isna(), "pycontrails_version"])
    if missing:
        raise ValueError(f"No commit_time in the summaries of releases {missing}")

    df["date"] = pd.to_datetime(df["commit_time"], utc=True).dt.tz_localize(None)
    return df


def changed_panels(df: pd.DataFrame, release: str) -> set[str]:
    """Return the panels whose value changes at ``release`` from the previous release."""
    i = df.index[df["pycontrails_version"] == release][0]
    if i == 0:
        return set()

    previous = df.iloc[i - 1]
    current = df.iloc[i]
    panels = set()
    for key, panel in PANELS.items():
        old = previous[panel.field]
        new = current[panel.field]
        if np.isnan(new):
            continue
        if np.isnan(old):
            panels.add(key)
        elif key == "dropped":
            if round(new) != round(old):
                panels.add(key)
        elif old != 0.0 and abs(new / old - 1.0) * 100.0 >= CHANGE_THRESHOLDS[key]:
            panels.add(key)
    return panels


def load_changes(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the changes in ``changes.toml`` in release order, numbered from 1."""
    with (HERE / "changes.toml").open("rb") as f:
        changes = tomllib.load(f)["change"]

    releases = set(df["pycontrails_version"])
    for change in changes:
        if change["release"] not in releases:
            raise ValueError(f"No results for release {change['release']} in changes.toml")
        change["panels"] = changed_panels(df, change["release"])
    changes.sort(key=lambda c: version.Version(c["release"]))

    for number, change in enumerate(changes, start=1):
        change["number"] = number
    return changes


def release_ticks(df: pd.DataFrame, min_days: int = 140) -> tuple[list[float], list[str]]:
    """Return the x positions and labels of release ticks.

    Tick the first release and the first release of every fifth minor version, skipping a tick
    less than ``min_days`` after the previous one.
    """
    positions = [df["date"].iloc[0]]
    labels = [df["pycontrails_version"].iloc[0]]
    seen = {df["parsed_version"].iloc[0].minor}
    for v, release, date in zip(
        df["parsed_version"],
        df["pycontrails_version"],
        df["date"],
        strict=True,
    ):
        if v.minor % 5 or v.minor in seen:
            continue
        seen.add(v.minor)
        if date - positions[-1] >= pd.Timedelta(days=min_days):
            positions.append(date)
            labels.append(release)

    return [mdates.date2num(p) for p in positions], labels


def plot(
    df: pd.DataFrame,
    panels: tuple[str, ...],
    changes: list[dict[str, Any]],
    path: pathlib.Path,
) -> None:
    """Plot ``panels`` against release date, mark the numbers of ``changes``, and save to ``path``.

    Color each release by how the benchmark runs it.
    """
    x = df["date"].to_numpy()
    date = dict(zip(df["pycontrails_version"], df["date"], strict=True))
    index = {release: i for i, release in enumerate(df["pycontrails_version"])}
    mode = df["mode"].to_numpy()

    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        squeeze=False,
        figsize=(FIGURE_WIDTH, PANEL_HEIGHT * len(panels) + 0.8),
        layout="constrained",
    )

    axes = axes[:, 0]
    for ax, key in zip(axes, panels, strict=True):
        panel = PANELS[key]
        y = df[panel.field].to_numpy() / 10.0**panel.exponent
        ax.plot(x, y, color="0.6", lw=0.8, drawstyle="steps-post", zorder=1)
        for key_mode, (label, color) in RUN_MODES.items():
            mask = mode == key_mode
            ax.plot(x[mask], y[mask], "o", ms=3.5, color=color, label=label, zorder=2)
        for change in changes:
            if key not in change["panels"]:
                continue
            value = y[index[change["release"]]]
            if np.isnan(value):
                continue
            annotation = ax.annotate(
                str(change["number"]),
                (date[change["release"]], value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE - 2,
                bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": "0.4", "lw": 0.5},
            )
            annotation.set_url(f"#{change_anchor(change)}")
            annotation.get_bbox_patch().set_url(f"#{change_anchor(change)}")
        if panel.log:
            ax.set_yscale("log")
        ax.set_ylabel(panel.label, fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.grid(alpha=0.3, lw=0.5)
        ax.margins(y=0.25)

    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Release date", fontsize=FONT_SIZE)
    top = axes[0].secondary_xaxis("top")
    positions, labels = release_ticks(df)
    top.set_xticks(positions, labels)
    top.tick_params(labelsize=FONT_SIZE - 1)
    top.set_xlabel("pycontrails release", fontsize=FONT_SIZE)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncols=3,
        fontsize=FONT_SIZE - 1,
        frameon=False,
    )
    save_svg(fig, path)


def save_svg(fig: plt.Figure, path: pathlib.Path) -> None:
    """Save ``fig`` to ``path`` as an SVG that scales to the width of its container."""
    with plt.rc_context({"svg.hashsalt": path.stem}):
        fig.savefig(path)
    plt.close(fig)

    svg = path.read_text()
    svg = re.sub(r'(<svg [^>]*?)width="[^"]*" height="[^"]*"', r'\1width="100%"', svg, count=1)
    svg = svg.replace(' target="_blank"', "")
    path.write_text(svg)


def change_anchor(change: dict[str, Any]) -> str:
    """Return the HTML id of the row of ``change`` in the table of changes."""
    return f"change-{change['number']}"


def list_table(header: list[str], rows: list[list[str]], widths: list[int]) -> str:
    """Return an rst list table of ``rows`` under ``header``."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {' '.join(str(w) for w in widths)}",
        "",
    ]
    for row in [header, *rows]:
        for j, cell in enumerate(row):
            cell_lines = cell.strip().splitlines() or [""]
            prefix = "   * - " if j == 0 else "     - "
            lines.append(prefix + cell_lines[0])
            lines.extend("       " + line for line in cell_lines[1:])
    return "\n".join(lines) + "\n"


def changes_table(df: pd.DataFrame, changes: list[dict[str, Any]]) -> str:
    """Return the table of ``changes``, with the id of :func:`change_anchor` on each number."""
    date = dict(zip(df["pycontrails_version"], df["date"].dt.date, strict=True))
    role = ".. role:: raw-html(raw)\n   :format: html\n\n"
    rows = [
        [
            f':raw-html:`<span id="{change_anchor(c)}">{c["number"]}</span>`',
            c["release"],
            date[c["release"]].isoformat(),
            "\n".join(f"* {bullet}" for bullet in c["bullets"]),
        ]
        for c in changes
    ]
    return role + list_table(["#", "Release", "Release date", "Change"], rows, [4, 9, 13, 74])


def fmt(value: float, spec: str) -> str:
    """Format ``value`` with ``spec``, or return a dash for NaN."""
    return "—" if np.isnan(value) else format(value, spec)


def fmt_range(values: pd.Series, spec: str) -> str:
    """Format the minimum and maximum of ``values`` with ``spec``, or one value if they agree."""
    low = format(values.min(), spec)
    high = format(values.max(), spec)
    return low if low == high else f"{low}{DASH}{high}"


def result_groups(df: pd.DataFrame) -> pd.Series:
    """Return a group number for each release, shared by consecutive releases with equal results."""
    result_fields = ["n_flights", "n_dropped", *(PANELS[k].field for k in RESULT_PANELS)]
    key = df[result_fields].map(lambda v: fmt(v, ".4e")).agg(" ".join, axis=1)
    key = key + df["sampled"].astype(str) + df["n_batches"].astype(str)
    key = key + df["patches"].map(" ".join)
    return (key != key.shift()).cumsum()


def results_table(df: pd.DataFrame) -> str:
    """Return the table of results, with one row for each run of releases with equal results."""
    rows = []
    for _, g in df.groupby(result_groups(df), sort=False):
        first = g.iloc[0]
        releases = first["pycontrails_version"]
        if len(g) > 1:
            releases += f" {DASH} {g['pycontrails_version'].iloc[-1]}"
        rows.append(
            [
                releases,
                f"{first['n_dropped']:,.0f}",
                fmt(first["ef_J"], ".3e"),
                fmt(first["persistent_length_km"], ".3e"),
                fmt(first["fuel_burn_kg"], ".3e"),
                fmt_range(g["runtime_s"], ",.0f"),
                fmt_range(g["peak_rss_gb"], ".1f"),
            ]
        )
    header = [
        "Releases",
        "Flights dropped",
        "Energy forcing (J)",
        "Persistent length (km)",
        "Fuel burn (kg)",
        "Runtime (s)",
        "Peak memory (GB)",
    ]
    return list_table(header, rows, [16, 10, 14, 14, 14, 12, 10])


def results_csv(df: pd.DataFrame) -> str:
    """Return every ``summary.json`` field of every release as CSV, in release order."""
    out = df.drop(columns=["parsed_version", "sampled", "mode", "sample", "date"], errors="ignore")
    out = out.dropna(axis=1, how="all")
    out.insert(1, "commit_date", df["date"].dt.date)
    out["patches"] = df["patches"].map("; ".join)
    if "sample" in df:
        samples = df["sample"].map(lambda s: s if isinstance(s, dict) else {})
        sample = pd.DataFrame(list(samples), index=df.index).add_prefix("sample_")
        if "sample_flight_errors" in sample:
            sample["sample_flight_errors"] = sample["sample_flight_errors"].map(
                lambda e: json.dumps(e) if isinstance(e, dict) else None
            )
        out = pd.concat([out, sample], axis=1)
    return out.to_csv(index=False)


def symlog_bins(linthresh: float, vmax: float, n_log: int) -> np.ndarray:
    """Return histogram bin edges, log-spaced from ``linthresh`` to ``vmax`` on each side of 0.

    There are ``n_log`` bins on each side and one bin from ``-linthresh`` to ``linthresh``.
    """
    positive = np.geomspace(linthresh, vmax, n_log + 1)
    return np.concatenate([-positive[::-1], positive])


def plot_ef_histograms(fs: gcsfs.GCSFileSystem, df: pd.DataFrame, path: pathlib.Path) -> None:
    """Plot staggered histograms of the per-flight energy forcing and save to ``path``."""
    ef_total = df["ef_J"]
    keep = (ef_total / ef_total.shift() - 1.0).abs() >= EF_HISTOGRAM_CHANGE
    keep.iloc[[0, -1]] = True
    bins = symlog_bins(EF_LINTHRESH, 1e16, 48)
    paths = [
        f"{RESULTS}/{release}/flight_summary.csv" for release in df.loc[keep, "pycontrails_version"]
    ]
    contents = fs.cat(paths)
    histograms = []
    for csv_path, (_, row) in zip(paths, df[keep].iterrows(), strict=True):
        ef = pd.read_csv(io.BytesIO(contents[csv_path]), usecols=["ef_J"])["ef_J"].to_numpy()
        counts, _ = np.histogram(ef[np.isfinite(ef)], bins)
        counts[len(bins) // 2 - 1] = 0
        histograms.append((row["pycontrails_version"], row["mode"], counts / len(ef)))

    n_rows = len(histograms)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 0.55 * n_rows + 1.4), layout="constrained")
    # Height of the tallest bin, in rows, below 1 so that neighboring histograms do not overlap
    peak_height = 0.9
    peak = max(h[2].max() for h in histograms)
    for i, (_, mode, fraction) in enumerate(histograms):
        base = n_rows - 1 - i
        color = RUN_MODES[mode][1]
        y = base + peak_height * np.append(fraction, fraction[-1]) / peak
        ax.fill_between(bins, base, y, step="post", color=color, alpha=0.5, lw=0, zorder=i)
        ax.step(bins, y, where="post", color=color, lw=0.8, zorder=i)

    ax.set_xscale("symlog", linthresh=EF_LINTHRESH)
    ax.set_xlim(bins[0], bins[-1])
    ticks = [-1e15, -1e12, 0.0, 1e12, 1e15]
    ax.set_xticks(ticks, ["$-10^{15}$", "$-10^{12}$", "0", "$10^{12}$", "$10^{15}$"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Energy forcing per flight (J)", fontsize=FONT_SIZE)
    ax.set_yticks(range(n_rows), [h[0] for h in histograms][::-1])
    ax.set_ylim(-0.3, n_rows - 1 + peak_height + 0.3)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.5) for _, color in RUN_MODES.values()
    ]
    fig.legend(
        handles,
        [label for label, _ in RUN_MODES.values()],
        loc="outside upper center",
        ncols=3,
        fontsize=FONT_SIZE - 1,
        frameon=False,
    )
    save_svg(fig, path)


def main() -> None:
    """Write the figures and tables to :data:`OUTPUT`."""
    fs = gcsfs.GCSFileSystem()
    df = load_results(fs)
    changes = load_changes(df)

    OUTPUT.mkdir(exist_ok=True)
    plot(df, FLIGHT_PANELS, changes, OUTPUT / "flights.svg")
    plot(df, RESULT_PANELS, changes, OUTPUT / "results.svg")
    plot(df, PERFORMANCE_PANELS, changes, OUTPUT / "performance.svg")
    plot_ef_histograms(fs, df, OUTPUT / "ef_histograms.svg")
    (OUTPUT / "benchmark_results.csv").write_text(results_csv(df))
    (OUTPUT / "changes.inc").write_text(changes_table(df, changes))
    (OUTPUT / "results.inc").write_text(results_table(df))


if __name__ == "__main__":
    main()
