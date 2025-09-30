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
    order: Optional[str] = "size",  # "size", "alpha", or None to keep original
    palette: Optional[Dict[Union[str, int, float], str]] = None,
) -> Tuple[plt.Figure, plt.Axes, Optional[object]]:
    """
    Plot Kaplan–Meier survival curves by a categorical column with an at-risk table

    Parameters
    - df: dataframe with columns [time_col, event_col, group_col]
    - group_col: column that defines groups to compare
    - time_col: time to event or censoring, in days
    - event_col: 1 if event occurred in hospital, 0 if censored
    - xmax: x-axis limit in days for clarity
    - title: optional plot title
    - min_group_size: skip groups with fewer rows than this value
    - label_map: optional mapping {raw_value: label} for legend readability
    - show_table: whether to show the at-risk counts table
    - show_ci: whether to show KM confidence intervals
    - order: "size" sorts groups by descending size, "alpha" sorts alphanumerically, None keeps data order
    - palette: optional mapping {group_value: color}

    Returns
    - fig, ax: matplotlib Figure and Axes
    - test_result: lifelines log-rank object
        - logrank_test for 2 groups
        - multivariate_logrank_test for 3+ groups
        - None if < 2 usable groups
    """
    # Basic checks
    missing_cols = [c for c in [group_col, time_col, event_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only rows with non-missing essentials
    d = df[[group_col, time_col, event_col]].dropna(subset=[group_col, time_col, event_col]).copy()

    # Positive time scale for survival analysis
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce").clip(lower=0)
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce").astype(int)

    # Unique groups and ordering
    groups = d[group_col].dropna().unique().tolist()
    if order == "size":
        group_sizes = d[group_col].value_counts()
        groups = sorted(groups, key=lambda g: (-int(group_sizes.get(g, 0)), str(g)))
    elif order == "alpha":
        groups = sorted(groups, key=lambda g: str(g))

    # Fit and plot
    fig, ax = plt.subplots(figsize=(7, 5))
    fitters = []
    plotted_groups = []

    for g in groups:
        d_g = d.loc[d[group_col] == g]
        if len(d_g) < min_group_size:
            continue
        label = label_map.get(g, f"{group_col}={g}") if label_map else f"{group_col}={g}"
        km = KaplanMeierFitter(label=label)
        km.fit(durations=d_g[time_col], event_observed=d_g[event_col])
        style_kwargs = {}
        if palette and g in palette:
            style_kwargs["c"] = palette[g]
        km.plot_survival_function(ax=ax, ci_show=show_ci, **style_kwargs)
        fitters.append(km)
        plotted_groups.append(g)

    # Axes cosmetics
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Survival probability")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title(title or f"Survival by {group_col}")
    ax.grid(True)
    ax.legend(title=None)

    # At-risk table
    if show_table and fitters:
        add_at_risk_counts(*fitters, ax=ax)

    # Log-rank test
    test_result = None
    if len(plotted_groups) >= 2:
        if len(plotted_groups) == 2:
            g1, g2 = plotted_groups[:2]
            d1 = d.loc[d[group_col] == g1]
            d2 = d.loc[d[group_col] == g2]
            test_result = logrank_test(
                d1[time_col], d2[time_col],
                event_observed_A=d1[event_col],
                event_observed_B=d2[event_col]
            )
            # Append p-value to title for teaching
            ax.set_title((title or f"Survival by {group_col}") + f"  [{_format_p(test_result.p_value)}]")
        else:
            test_result = multivariate_logrank_test(d[time_col], d[group_col], d[event_col])
            ax.set_title((title or f"Survival by {group_col}") + f"  [{_format_p(test_result.p_value)}]")

    plt.tight_layout()
    return fig, ax, test_result
