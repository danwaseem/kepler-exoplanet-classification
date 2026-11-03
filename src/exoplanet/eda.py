"""Exploratory figures for the Stage 1 data audit. No models are fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exoplanet.config import CLASS_ORDER, FIGURES, TARGET_COLUMN

PALETTE = {
    "CONFIRMED": "#0072B2",
    "CANDIDATE": "#E69F00",
    "FALSE POSITIVE": "#009E73",
}

DISTRIBUTION_CANDIDATES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_steff",
    "koi_dicco_msky",
    "koi_kepmag",
]
LOG10_COLUMNS = {"koi_period", "koi_depth", "koi_prad", "koi_model_snr"}

CORRELATION_CANDIDATES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ror",
    "koi_prad",
    "koi_sma",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_smass",
    "koi_kepmag",
    "koi_dicco_msky",
]


def save_figures(
    frame: pd.DataFrame,
    unfitted: pd.Series,
    included: list[str],
    output_dir: Path | None = None,
) -> list[str]:
    output_dir = output_dir or FIGURES
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    written = [
        _class_balance(frame, output_dir / "class_balance.png"),
        _missingness(frame, output_dir / "missingness.png"),
        _unfitted_dispositions(frame, unfitted, output_dir / "unfitted_block_by_disposition.png"),
    ]
    distribution = _distributions(
        frame.loc[~unfitted],
        included,
        output_dir / "feature_distributions_by_class.png",
    )
    if distribution:
        written.append(distribution)
    correlation = _correlation(
        frame.loc[~unfitted],
        included,
        output_dir / "spearman_correlation.png",
    )
    if correlation:
        written.append(correlation)
    return written


def _class_balance(frame: pd.DataFrame, path: Path) -> str:
    counts = frame[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [PALETTE[name] for name in CLASS_ORDER]
    bars = ax.bar(CLASS_ORDER, counts.to_numpy(), color=colors)
    ax.set_ylabel("KOIs")
    ax.set_title("KOI disposition counts in the cumulative table")
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _missingness(frame: pd.DataFrame, path: Path) -> str:
    fraction = frame.isna().mean().sort_values(ascending=False)
    fraction = fraction[fraction > 0].head(40)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(fraction.index[::-1], fraction.to_numpy()[::-1], color="#4C78A8")
    ax.set_xlabel("Fraction missing")
    ax.set_title("40 columns with the most missing values")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _unfitted_dispositions(frame: pd.DataFrame, unfitted: pd.Series, path: Path) -> str:
    plot_frame = pd.DataFrame(
        {
            TARGET_COLUMN: frame[TARGET_COLUMN].to_numpy(),
            "sample": [
                "No transit depth" if flag else "Transit depth measured"
                for flag in unfitted.to_numpy()
            ],
        }
    )
    counts = (
        plot_frame.groupby(["sample", TARGET_COLUMN], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_ORDER)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fractions.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE[name] for name in CLASS_ORDER],
        rot=0,
    )
    ax.set_ylabel("Share of rows")
    ax.set_xlabel("")
    ax.set_title("Disposition mix for rows with and without a transit depth")
    ax.legend(title="", loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _distributions(frame: pd.DataFrame, included: list[str], path: Path) -> str | None:
    columns = [column for column in DISTRIBUTION_CANDIDATES if column in included]
    if not columns:
        return None
    plot = frame[[TARGET_COLUMN, *columns]].copy()
    for column in columns:
        plot[column] = pd.to_numeric(plot[column], errors="coerce")
        if column in LOG10_COLUMNS:
            plot[column] = np.log10(plot[column].where(plot[column] > 0))
    melted = plot.melt(id_vars=TARGET_COLUMN, var_name="feature", value_name="value")
    melted = melted.dropna()
    grid = sns.catplot(
        data=melted,
        x=TARGET_COLUMN,
        y="value",
        col="feature",
        kind="box",
        col_wrap=4,
        sharey=False,
        order=CLASS_ORDER,
        hue=TARGET_COLUMN,
        hue_order=CLASS_ORDER,
        palette=PALETTE,
        height=3.2,
        aspect=0.9,
        legend=False,
        fliersize=1,
    )
    grid.set_axis_labels("", "")
    grid.set_xticklabels(rotation=20)
    for axis, column in zip(grid.axes.flat, columns):
        label = f"log10 {column}" if column in LOG10_COLUMNS else column
        axis.set_title(label)
    grid.fig.suptitle(
        "Proposed features by disposition, rows with a measured transit depth",
        y=1.02,
    )
    grid.fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    return str(path)


def _correlation(frame: pd.DataFrame, included: list[str], path: Path) -> str | None:
    columns = [column for column in CORRELATION_CANDIDATES if column in included]
    if len(columns) < 3:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    # Rank correlation without requiring a separate SciPy call.
    correlation = numeric.rank(method="average").corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
    )
    ax.set_title("Spearman correlation, rows with a measured transit depth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
