# utils.py
# Minimal, student-friendly utilities for Kaplan–Meier plotting by group
# Dependencies: lifelines, matplotlib, pandas, numpy

from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test


def _coerce_series(x: Union[pd.Series, Iterable]) -> pd.Series:
    """Return a clean pandas Series"""
    return x if isinstance(x, pd.Series) else pd.Series(list(x))


def _format_p(p: float) -> str:
    """Pretty p-value formatting for titles or printouts"""
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.4f}"


# utils.py  — replace km_by_group with this updated version


def km_by_group(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = "duration",
    event_col: str = "event",
    xmax: Optional[float] = 90,
    title: Optional[str] = None,
    min_group_size: int = 5,
    label_map: Optional[Dict[Union[str, int, float], str]] = None,
    show_table: bool = True,
    show_ci: bool = True,
    order: Optional[bool] = "size",   # "size", "alpha", or None
    palette: Optional[Dict[Union[str, int, float], str]] = None,
    figsize: Tuple[float, float] = (7, 7.5),  # was (7, 5)
    bottom_pad: float = 0.30                  # extra space for risk table
) -> Tuple[plt.Figure, plt.Axes, Optional[object]]:
    """
    Plot Kaplan–Meier curves by a categorical column with an at-risk table
    figsize controls the overall height and width
    bottom_pad reserves space at the bottom for the table
    """
    missing_cols = [c for c in [group_col,
                                time_col, event_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    d = df[[group_col, time_col, event_col]].dropna(
        subset=[group_col, time_col, event_col]).copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce").clip(lower=0)
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce").astype(int)

    groups = d[group_col].dropna().unique().tolist()
    if order == "size":
        sizes = d[group_col].value_counts()
        groups = sorted(groups, key=lambda g: (-int(sizes.get(g, 0)), str(g)))
    elif order == "alpha":
        groups = sorted(groups, key=lambda g: str(g))

    fig, ax = plt.subplots(figsize=figsize)
    fitters, plotted_groups = [], []

    for g in groups:
        d_g = d.loc[d[group_col] == g]
        if len(d_g) < min_group_size:
            continue
        label = label_map.get(
            g, f"{group_col}={g}") if label_map else f"{group_col}={g}"
        km = KaplanMeierFitter(label=label)
        km.fit(durations=d_g[time_col], event_observed=d_g[event_col])
        style_kwargs = {}
        if palette and g in palette:
            style_kwargs["c"] = palette[g]
        km.plot_survival_function(ax=ax, ci_show=show_ci, **style_kwargs)
        fitters.append(km)
        plotted_groups.append(g)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Survival probability")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_title(title or f"Survival by {group_col}")
    ax.grid(True)
    ax.legend(title=None, loc="best")

    if show_table and fitters:
        add_at_risk_counts(*fitters, ax=ax)

    # leave room for the table
    plt.subplots_adjust(bottom=bottom_pad)
    plt.tight_layout()

    test_result = None
    if len(plotted_groups) >= 2:
        if len(plotted_groups) == 2:
            g1, g2 = plotted_groups[:2]
            d1, d2 = d.loc[d[group_col] == g1], d.loc[d[group_col] == g2]
            test_result = logrank_test(
                d1[time_col], d2[time_col],
                event_observed_A=d1[event_col],
                event_observed_B=d2[event_col]
            )
            ax.set_title(
                (title or f"Survival by {group_col}") + f"  [{_format_p(test_result.p_value)}]")
        else:
            test_result = multivariate_logrank_test(
                d[time_col], d[group_col], d[event_col])
            ax.set_title(
                (title or f"Survival by {group_col}") + f"  [{_format_p(test_result.p_value)}]")

    return fig, ax, test_result
