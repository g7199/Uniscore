#!/usr/bin/env python3
"""Render the paper's main result table from canonical CSV values.

This script intentionally renders the submitted table exactly from
`data/main_table_values.csv`. The CSV stores the rounded values shown in the
paper, plus the best/second-best markers used in the LaTeX table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    ("dataset", "Dataset"),
    ("method", "Method"),
    ("spearman", "Spearman rho"),
    ("kendall", "Kendall tau"),
    ("pearson", "Pearson r"),
    ("f1", "F1"),
    ("acc", "Acc"),
    ("ncv", "nCV"),
    ("skew", "Skew."),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def emphases(row: dict[str, str]) -> dict[str, str]:
    out = {}
    raw = row.get("emphasis", "")
    if not raw:
        return out
    for item in raw.split(";"):
        if not item:
            continue
        key, _, value = item.partition(":")
        out[key] = value
    return out


def latex_value(row: dict[str, str], key: str) -> str:
    value = row[key].replace("+/-", r"$\pm$")
    mark = emphases(row).get(key)
    if mark == "best":
        return rf"\textbf{{{value}}}"
    if mark == "second":
        return rf"\underline{{{value}}}"
    return value


def latex_method(row: dict[str, str]) -> str:
    method = row["method"]
    if method.startswith("-- "):
        if "p=" in method:
            label = method.replace("-- ", r"\hspace*{3mm}-- $") + "$"
        elif "n=" in method:
            label = method.replace("-- ", r"\hspace*{3mm}-- $") + "$"
        else:
            label = r"\hspace*{3mm}" + method
    else:
        label = method
    if emphases(row).get("method") == "best":
        return rf"\textbf{{{label}}}"
    return label


def render_latex(rows: list[dict[str, str]]) -> str:
    lines = []
    lines.append(r"\begin{tabular}{llccc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Method} & \textbf{Spearman $\rho$} & "
        r"\textbf{Kendall $\tau$} & \textbf{Pearson $r$} & \textbf{F1} & "
        r"\textbf{Acc} & \textbf{nCV} & \textbf{Skew.} \\"
    )
    lines.append(r"\midrule")

    dataset_counts = {}
    for row in rows:
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1

    last_dataset = None
    for row in rows:
        dataset = row["dataset"]
        if last_dataset is not None and dataset != last_dataset:
            lines.append(r"\midrule")
        dataset_cell = ""
        if dataset != last_dataset:
            dataset_cell = rf"\multirow{{{dataset_counts[dataset]}}}{{*}}{{{dataset}}}"
        values = [
            dataset_cell,
            latex_method(row),
            latex_value(row, "spearman"),
            latex_value(row, "kendall"),
            latex_value(row, "pearson"),
            latex_value(row, "f1"),
            latex_value(row, "acc"),
            latex_value(row, "ncv"),
            latex_value(row, "skew"),
        ]
        lines.append("& " + " & ".join(values[1:]) + r" \\" if not dataset_cell else " & ".join(values) + r" \\")
        last_dataset = dataset

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def marked_text(row: dict[str, str], key: str) -> str:
    value = row[key].replace("+/-", "±")
    mark = emphases(row).get(key)
    if mark == "best":
        return f"**{value}**"
    if mark == "second":
        return f"__{value}__"
    return value


def text_method(row: dict[str, str]) -> str:
    method = row["method"]
    if emphases(row).get("method") == "best":
        return f"**{method}**"
    return method


def render_text(rows: list[dict[str, str]]) -> str:
    table_rows = []
    last_dataset = None
    for row in rows:
        dataset = row["dataset"] if row["dataset"] != last_dataset else ""
        table_rows.append(
            [
                dataset,
                text_method(row),
                marked_text(row, "spearman"),
                marked_text(row, "kendall"),
                marked_text(row, "pearson"),
                marked_text(row, "f1"),
                marked_text(row, "acc"),
                marked_text(row, "ncv"),
                marked_text(row, "skew"),
            ]
        )
        last_dataset = row["dataset"]

    headers = [label for _, label in COLUMNS]
    widths = [
        max(len(str(row[i])) for row in [headers] + table_rows)
        for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * width for width in widths)
    lines = [
        "Main Results Table",
        "Best values are wrapped in **...**; second-best values are wrapped in __...__.",
        "",
        fmt(headers),
        sep,
    ]
    lines.extend(fmt(row) for row in table_rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["txt", "latex"], default="txt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = repo_root()
    rows = read_rows(root / "data" / "main_table_values.csv")
    rendered = render_text(rows) if args.format == "txt" else render_latex(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
