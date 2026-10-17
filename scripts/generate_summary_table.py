#!/usr/bin/env python3
"""Generate the cross-dataset benchmark summary table from the results CSVs.

Protocol (see statistical_results_summary.md):
  1. Average every subject's metric over the three seeds (42, 1234, 777).
  2. Cohort aggregate = mean +/- SE over the subject-level averages.
  3. p-value per baseline = two-sided Wilcoxon signed-rank test on the
     subject-level averages against the proposed model (10,000 permutations).

Writes results/summary_table.txt: a plain-text metrics summary followed by the
LaTeX snippet with the fake table values replaced by the computed ones
(all metrics to 2 decimal places).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "summary_table.txt"

SUBJECT = "Subject"
METHOD = "Method"
METHOD_VALUE = "Simple 80/20 Split"
PERMUTATIONS = 10_000
SEEDS = (42, 1234, 777)


@dataclass(frozen=True)
class Metric:
    column: str  # CSV column name
    scale: float  # multiply to reach %
    header: str  # LaTeX header cell (without \textbf)
    has_p: bool = True  # whether a $p$-value column follows this metric


@dataclass(frozen=True)
class Model:
    label: str  # LaTeX row label
    csv: str  # path relative to the repository root
    params: str | None = None
    proposed: bool = False


@dataclass(frozen=True)
class Table:
    caption: str
    font: str
    tabcolsep: str
    spec: str
    metrics: tuple[Metric, ...]
    models: tuple[Model, ...]
    has_params: bool


@dataclass
class Cell:
    mean: float
    sem: float
    p: float | None
    bold: bool = False


TABLES: tuple[Table, ...] = (
    Table(
        caption="a) BCI-UT P300 dataset",
        font=r"\small",
        tabcolsep="4pt",
        spec="l ccccr",
        metrics=(
            Metric("Obj Acc (%)", 1.0, r"Obj. Dec. Acc. (\%)"),
            Metric("Macro F1", 100.0, r"Macro F1 (\%)"),
        ),
        models=(
            Model("BCIAUT EEGNet", "results/bciaut/eegnet/subject_results.csv", "1.4k"),
            Model("P3CNet", "results/bciaut/p3cnet/subject_results.csv", "4.3k"),
            Model(
                "MSR-P300Net (Proposed)",
                "results/bciaut/msr_p300net/subject_results.csv",
                "4.0k",
                proposed=True,
            ),
        ),
        has_params=True,
    ),
    Table(
        caption="b) GIB-UVA Dataset",
        font=r"\small",
        tabcolsep="4pt",
        spec="l ccccr",
        metrics=(
            Metric("Obj Acc (%)", 1.0, r"Spell. Acc.(\%)"),
            Metric("Macro F1", 100.0, r"Macro F1 (\%)"),
        ),
        models=(
            Model("P3CNet", "results/gib/p3cnet/subject_results.csv", "4.3k"),
            Model("EEG-Inception", "results/gib/eeginception/subject_results.csv", "15.3k"),
            Model(
                "MSR-P300Net (Proposed)",
                "results/gib/msr_p300net/subject_results.csv",
                "4.0k",
                proposed=True,
            ),
        ),
        has_params=True,
    ),
    Table(
        caption="c) i. GIST RSVP-P300 dataset  -- P300 Speller Task$^\\dagger$",
        font=r"\footnotesize",
        tabcolsep="2pt",
        spec="l ccccc r",
        metrics=(
            Metric("Pairwise Acc (%)", 1.0, r"P. Acc. (\%)"),
            Metric("Macro F1", 100.0, r"Macro F1 (\%)"),
            Metric("Trial AUC", 100.0, r"AUC (\%)", has_p=False),
        ),
        models=(
            Model("Standard EEGNet", "results/gist/eegnet/p300_cohort_results_eegnet.csv", "1.3k"),
            Model(
                "EEG-Inception",
                "results/gist/eeginception/p300_cohort_results_eeginception.csv",
                "15.1k",
            ),
            Model(
                "MSR-P300Net (Proposed)",
                "results/gist/msr_p300net/p300_cohort_results_msr_p300net.csv",
                "4.0k",
                proposed=True,
            ),
        ),
        has_params=True,
    ),
    Table(
        caption="c) ii. GIST RSVP-P300 dataset -- RSVP Task$^\\dagger$",
        font=r"\footnotesize",
        tabcolsep="2pt",
        spec="l ccccc",
        metrics=(
            Metric("Seq Top-1 Acc (%)", 1.0, r"Seq. Top-1 (\%)"),
            Metric("Macro F1", 100.0, r"Macro F1 (\%)"),
            Metric("AUC", 100.0, r"AUC(\%)", has_p=False),
        ),
        models=(
            Model("Standard EEGNet", "results/gist/eegnet/rsvp_cohort_results_eegnet.csv"),
            Model("EEG-Inception", "results/gist/eeginception/rsvp_cohort_results_eeginception.csv"),
            Model(
                "MSR-P300Net (Proposed)",
                "results/gist/msr_p300net/rsvp_cohort_results_msr_p300net.csv",
                proposed=True,
            ),
        ),
        has_params=False,
    ),
)


def _permutation_method() -> stats.PermutationMethod:
    kwargs: dict = {"n_resamples": PERMUTATIONS}
    params = inspect.signature(stats.PermutationMethod).parameters
    key = "rng" if "rng" in params else "random_state"
    kwargs[key] = 42
    return stats.PermutationMethod(**kwargs)


def load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / csv)
    if METHOD in df.columns:
        df = df[df[METHOD] == METHOD_VALUE]
        if df.empty:
            raise ValueError(f"{csv}: no rows with {METHOD} == {METHOD_VALUE!r}")
    return df


def subject_level(df: pd.DataFrame, metric: Metric) -> pd.Series:
    """Mean over seeds for each subject."""
    return (df.groupby(SUBJECT)[metric.column].mean() * metric.scale).sort_index()


def mean_se(values: pd.Series) -> tuple[float, float]:
    sem = values.std(ddof=1) / (len(values) ** 0.5)
    return float(values.mean()), float(sem)


def wilcoxon_p(proposed: pd.Series, baseline: pd.Series) -> float:
    paired = pd.concat({"proposed": proposed, "baseline": baseline}, axis=1).dropna()
    result = stats.wilcoxon(
        paired["proposed"],
        paired["baseline"],
        alternative="two-sided",
        method=_permutation_method(),
    )
    return float(result.pvalue)


def compute(table: Table) -> dict[str, dict[str, Cell]]:
    series = {
        model.label: {metric.column: subject_level(load(model.csv), metric) for metric in table.metrics}
        for model in table.models
    }
    proposed = next(model for model in table.models if model.proposed)
    cells: dict[str, dict[str, Cell]] = {model.label: {} for model in table.models}

    for metric in table.metrics:
        for model in table.models:
            values = series[model.label][metric.column]
            mean, sem = mean_se(values)
            p = None
            if metric.has_p and not model.proposed:
                p = wilcoxon_p(series[proposed.label][metric.column], values)
            cells[model.label][metric.column] = Cell(mean, sem, p)
        best = max(
            (model.label for model in table.models),
            key=lambda label: cells[label][metric.column].mean,
        )
        cells[best][metric.column].bold = True
    return cells


def format_metric(cell: Cell) -> str:
    text = f"{cell.mean:.2f} $\\pm$ {cell.sem:.2f}"
    return f"\\textbf{{{text}}}" if cell.bold else text


def format_p(cell: Cell, proposed: bool) -> str:
    if proposed or cell.p is None:
        return "--"
    return "$<$0.001" if cell.p < 0.001 else f"{cell.p:.3f}"


def ind(level: int, text: str) -> str:
    return "\t" * level + text


def header_row(table: Table) -> str:
    parts = [r"\textbf{Model Architecture}"]
    for metric in table.metrics:
        parts.append(rf"\textbf{{{metric.header}}}")
        if metric.has_p:
            parts.append(r"\textbf{$p$-value}")
    if table.has_params:
        parts.append(r"\textbf{Params}")
    return " & ".join(parts) + r" \\"


def data_row(table: Table, model: Model, cells: dict[str, Cell]) -> str:
    label = rf"\textbf{{{model.label}}}" if model.proposed else model.label
    parts = [label]
    for metric in table.metrics:
        cell = cells[metric.column]
        parts.append(format_metric(cell))
        if metric.has_p:
            parts.append(format_p(cell, model.proposed))
    if table.has_params:
        parts.append(model.params or "")
    return " & ".join(parts) + r" \\"


def latex_block(table: Table, cells: dict[str, dict[str, Cell]]) -> list[str]:
    begin_tabular = r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + table.spec + r"@{}}"
    lines = [
        ind(2, r"\begin{subtable}{\linewidth}"),
        ind(3, r"\centering"),
        ind(3, rf"\caption*{{ {table.caption}}}"),
        ind(3, table.font),
        ind(3, rf"\setlength{{\tabcolsep}}{{{table.tabcolsep}}}"),
        ind(3, begin_tabular),
        ind(4, r"\toprule"),
        ind(4, header_row(table)),
        ind(4, r"\midrule"),
    ]
    for model in table.models:
        lines.append(ind(4, data_row(table, model, cells[model.label])))
    lines += [
        ind(4, r"\bottomrule"),
        ind(3, r"\end{tabular*}"),
        ind(2, r"\end{subtable}"),
        ind(2, r"\vspace{1em}"),
    ]
    return lines


def build_latex(results: list[tuple[Table, dict[str, dict[str, Cell]]]]) -> str:
    lines = [
        ind(1, r"\begin{table}[H]"),
        ind(2, r"\centering"),
        ind(
            2,
            r"\caption{Summary of Cross-dataset benchmark performance. "
            r"All metrics are reported as Mean $\pm$ Standard Error (SE)  the test}",
        ),
        ind(2, r"\label{tab:p300_performance_tight}"),
        "\t\t",
    ]
    for table, cells in results:
        lines += latex_block(table, cells)
    lines += [
        ind(2, r"\raggedright"),
        ind(2, r"\footnotesize $^\dagger$First deep learning benchmark on GIST RSVP-P300.\\"),
        ind(
            2,
            r"\footnotesize \textbf{Obj. Dec. Acc.} = Object Decoding Accuracy; "
            r"\textbf{Spell. Acc.} = Spelling Accuracy; \textbf{Seq. Top-1} = Sequence Top-1 Accuracy; "
            r"\textbf{P. Acc.} = Pairwise Accuracy; \textbf{Params} = Parameter Count.\\",
        ),
        ind(
            2,
            r"\footnotesize \textbf{$p-values$} reported are associated with Wilcoxon signed-rank test "
            r"performed on the metric to the left calculated to statistically evaluate "
            r"the model's performance.",
        ),
        ind(1, r"\end{table}"),
    ]
    return "\n".join(lines)


def build_summary(results: list[tuple[Table, dict[str, dict[str, Cell]]]]) -> str:
    width = 78
    lines = [
        "Cross-dataset benchmark summary",
        "=" * width,
        f"Aggregation: metric averaged over seeds {SEEDS} per subject; cohort value",
        "is the mean +/- standard error over the subject-level averages.",
        f"p-values: two-sided Wilcoxon signed-rank vs. proposed model on the",
        f"subject-level averages, {PERMUTATIONS} permutations.",
        "",
    ]
    for table, cells in results:
        lines += ["-" * width, table.caption, "-" * width]
        for model in table.models:
            bits = []
            for metric in table.metrics:
                cell = cells[model.label][metric.column]
                text = f"{metric.column}: {cell.mean:.2f} +/- {cell.sem:.2f}"
                if cell.p is not None:
                    text += f" (p={cell.p:.4g})"
                bits.append(text)
            params = f" | params={model.params}" if model.params else ""
            lines.append(f"  {model.label:<26} " + " | ".join(bits) + params)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = [(table, compute(table)) for table in TABLES]
    summary = build_summary(results)
    latex = build_latex(results)
    separator = "=" * 78
    OUTPUT.write_text(
        f"{summary}\n{separator}\nLATEX SNIPPET\n{separator}\n\n{latex}\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
