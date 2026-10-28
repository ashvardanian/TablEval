# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot TablEval JSON reports as SVG: load throughput, query times, speedups and memory."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEME = {
    "field": "#C44536",
    "ink": "#FFE6A7",
    "frame": "#E5E5E5",
    "font": ["Poppins", "Helvetica", "Arial", "sans-serif"],
}
TIMED = ("completed", "mismatch")
UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


def fact_rows(report):
    fact = {"taxi": "trips", "tpch": "lineitem", "clickbench": "hits"}[
        report["dataset"]["suite"]
    ]
    return next(t["rows"] for t in report["dataset"]["tables"] if t["name"] == fact)


def label(report):
    workload = report["workload"]
    return f"{report['config']['backend']} · {workload['threads'][0]} threads · {workload['source'][0]}"


def dataset(report):
    return f"{report['dataset']['suite']} · {Path(report['dataset']['path']).name}"


def seconds(query):
    """The fastest hot run, or the cold run when no hot run finished."""
    return min(query["hot_seconds"], default=query["cold_seconds"])


def server_bytes(usage):
    """The server's peak from cgroup sampling, or the used side of a Docker `MemUsage` like `1.2GiB / 62GiB`."""
    if not usage:
        return 0
    if "rss_max_bytes" in usage:
        return usage["rss_max_bytes"]
    if "MemUsage" not in usage:
        return 0
    used = usage["MemUsage"].split("/")[0].strip()
    unit = next(
        (u for u in sorted(UNITS, key=len, reverse=True) if used.endswith(u)), "B"
    )
    return float(used[: -len(unit)]) * UNITS[unit]


def peak_memory(report):
    phases = (
        [report["load"], *report["queries"]]
        if report.get("load")
        else report["queries"]
    )
    return max(
        (
            p["client_usage"]["rss_max_bytes"] + server_bytes(p.get("server_usage"))
            for p in phases
        ),
        default=0,
    )


def bars(output, name, xlabel, groups):
    """One horizontal bar panel per group; `groups` maps a title to (label, value, hatched) rows."""
    groups = {title: rows for title, rows in groups.items() if rows}
    if not groups:
        return
    heights = [max(1.8, 0.32 * len(rows) + 0.8) for rows in groups.values()]
    figure, axes = plt.subplots(
        len(groups),
        1,
        figsize=(12, sum(heights)),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    for axis, (title, rows) in zip(axes[:, 0], groups.items()):
        for index, (text, value, hatched) in enumerate(rows):
            axis.barh(
                index,
                value,
                color=THEME["field"] if hatched else THEME["ink"],
                edgecolor=THEME["ink"],
                hatch="//" if hatched else None,
            )
        axis.set_yticks(range(len(rows)), [text for text, _, _ in rows])
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline", default="duckdb", help="backend the speedup chart divides by"
    )
    args = parser.parse_args()
    paths = (
        sorted(args.reports.glob("*.json")) if args.reports.is_dir() else [args.reports]
    )
    reports = [json.loads(path.read_text()) for path in paths]
    for path, report in zip(paths, reports):
        if report.get("schema_version") != 1:
            raise ValueError(f"Unsupported report schema: {path}")
    if not reports:
        parser.error("No reports found")
    output = (
        args.output
        or (args.reports if args.reports.is_dir() else args.reports.parent) / "plots"
    )
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["field"],
            "axes.facecolor": THEME["field"],
            "axes.edgecolor": THEME["frame"],
            "axes.labelcolor": THEME["ink"],
            "axes.titlecolor": THEME["ink"],
            "text.color": THEME["ink"],
            "xtick.color": THEME["ink"],
            "ytick.color": THEME["ink"],
            "grid.color": THEME["ink"],
            "font.family": THEME["font"],
            "svg.fonttype": "none",
        }
    )

    loads, memory, queries = defaultdict(list), defaultdict(list), defaultdict(list)
    for report in reports:
        group = f"{dataset(report)} · {fact_rows(report):,} rows"
        load = report.get("load")
        if load and load["status"] == "completed":
            loads[group].append((label(report), load["throughput"], False))
        memory[group].append((label(report), peak_memory(report) / 2**30, False))
        for query in report["queries"]:
            if query["status"] in TIMED:
                hatched = query["status"] == "mismatch"
                queries[f"{group} · {query['query']}"].append(
                    (label(report), seconds(query), hatched)
                )
    bars(output, "load-throughput", "Fact-table rows loaded / second", loads)
    bars(
        output,
        "query-seconds",
        "Fastest hot run (s); hatched rows returned other answers",
        queries,
    )
    bars(
        output,
        "memory",
        "Peak client RSS plus the server's peak (GiB)",
        memory,
    )

    # Speedup over the baseline per query, against the fact-table rows of each run.
    timings = defaultdict(dict)
    for report in reports:
        for query in report["queries"]:
            if query["status"] == "completed":
                key = (dataset(report), query["query"], label(report))
                timings[key][fact_rows(report)] = seconds(query)
    panels = defaultdict(dict)
    for (name, query, text), points in timings.items():
        baseline = next(
            (
                v
                for (n, q, t), v in timings.items()
                if n == name and q == query and t.startswith(args.baseline + " ")
            ),
            None,
        )
        if baseline:
            panels[f"{name} · {query}"][text] = sorted(
                (rows, baseline[rows] / value)
                for rows, value in points.items()
                if rows in baseline and value
            )
    panels = {
        title: lines
        for title, lines in panels.items()
        if any(len(line) > 1 for line in lines.values())
    }
    if panels:
        columns = min(3, len(panels))
        figure, axes = plt.subplots(
            math.ceil(len(panels) / columns),
            columns,
            figsize=(6 * columns, 4 * math.ceil(len(panels) / columns)),
            squeeze=False,
        )
        for axis, (title, lines) in zip(axes.flat, panels.items()):
            for text, points in sorted(lines.items()):
                is_baseline = text.startswith(args.baseline + " ")
                axis.plot(
                    *zip(*points),
                    marker="o",
                    color=THEME["ink"],
                    alpha=1.0 if is_baseline else 0.6,
                    linestyle="--" if is_baseline else "-",
                    label=text,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(title, loc="left", fontsize=10)
            axis.set_xlabel("Fact-table rows")
            axis.set_ylabel(f"Speedup over {args.baseline}")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, facecolor=THEME["field"], edgecolor=THEME["frame"])
        for axis in list(axes.flat)[len(panels) :]:
            axis.set_visible(False)
        figure.tight_layout()
        figure.savefig(
            output / f"speedup-over-{args.baseline}.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
