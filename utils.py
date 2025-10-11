# utils.py
# Minimal, student-friendly utilities for Kaplan–Meier plotting by group
# Dependencies: lifelines, matplotlib, pandas, numpy

from lifelines import CoxPHFitter
from sksurv.nonparametric import cumulative_incidence_competing_risks
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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


def detect_feature_types(X: pd.DataFrame):
    """Return lists of numeric and categorical column names"""
    numeric_cols = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [
        c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols, num_strategy="median", cat_strategy="most_frequent"):
    """Build a ColumnTransformer for numeric and categorical preprocessing"""
    num_tf = Pipeline(
        steps=[("imputer", SimpleImputer(strategy=num_strategy))])
    if categorical_cols and len(categorical_cols) > 0:
        # Using sparse=False for broad sklearn compatibility
        cat_tf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=cat_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
    else:
        cat_tf = "drop"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor


def to_dataframe(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X with preprocessor and return a DataFrame with feature names preserved"""
    Xt = preprocessor.transform(X)
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(Xt, columns=cols, index=X.index)


def predict_cif_from_csh(
    cph_event: CoxPHFitter,
    cph_competing: CoxPHFitter,
    X_df: pd.DataFrame,
    times: np.ndarray
) -> pd.DataFrame:
    """
    Patient-level CIF for the event of interest by combining two cause-specific Cox models
    Implements F(t|x) ≈ Σ h_event(t_j|x) * S_all(t_{j-1}|x) * ΔH0_event(t_j)
    """
    # baseline cumulative hazards and common time grid
    H0e = cph_event.baseline_cumulative_hazard_.iloc[:, 0]
    H0c = cph_competing.baseline_cumulative_hazard_.iloc[:, 0]
    t_grid = np.union1d(H0e.index.values.astype(float),
                        H0c.index.values.astype(float))
    t_grid.sort()

    # baselines on grid and increments
    H0e_g = H0e.reindex(t_grid).ffill().fillna(0.0).to_numpy()
    H0c_g = H0c.reindex(t_grid).ffill().fillna(0.0).to_numpy()
    dH0e = np.diff(H0e_g, prepend=0.0)

    # patient-specific multipliers
    ph_e = np.asarray(cph_event.predict_partial_hazard(X_df)).ravel()
    ph_c = np.asarray(cph_competing.predict_partial_hazard(X_df)).ravel()

    He = np.outer(ph_e, H0e_g)
    Hc = np.outer(ph_c, H0c_g)

    # left Riemann sum uses survival just before the step
    S_prev = np.exp(-(He[:, :-1] + Hc[:, :-1]))      # n x (T-1)
    h_e_inc = np.outer(ph_e, dH0e[1:])               # n x (T-1)

    F = np.zeros((X_df.shape[0], t_grid.size), dtype=float)
    F[:, 1:] = np.cumsum(h_e_inc * S_prev, axis=1)

    # read CIF at requested times
    out = {}
    for t in times:
        j = np.searchsorted(t_grid, float(t), side="right") - 1
        j = int(np.clip(j, 0, t_grid.size - 1))
        out[t] = F[:, j]

    col_names = {t: f"CIF_{int(t)}d" for t in times}
    return pd.DataFrame(out, index=X_df.index).rename(columns=col_names)


def check_calibration_competing_risk(
    y_true: pd.DataFrame,
    predictions_60d: pd.Series,
    duration_col: str,
    event_col: str,
    competing_col: str,
    death_code: int = 1
) -> pd.DataFrame:
    """
    Calibration by risk quartile at 60 days using nonparametric CIF
    event coding must be 0=censored, 1=death, 2=discharge by default
    death_code selects which cause to treat as "death" in the CIF output:
      cif[0] = total, cif[death_code] = death, cif[2] = discharge for death_code=1
    """
    evt = np.where(y_true[event_col] == 1, death_code,
                   np.where(y_true[competing_col] == 1, 2, 0)).astype(int)
    dur = y_true[duration_col].astype(float).to_numpy()

    q = pd.qcut(predictions_60d, 4, labels=[
                "Q1 lowest", "Q2", "Q3", "Q4 highest"])

    rows = []
    for label in q.cat.categories:
        mask = (q == label).to_numpy()
        n = int(mask.sum())
        if n == 0:
            rows.append({"risk_quartile": label, "n": 0,
                         "pred_mean_risk_60d": np.nan, "obs_risk_60d": np.nan})
            continue

        pred_mean = float(predictions_60d[mask].mean())

        times, cif = cumulative_incidence_competing_risks(evt[mask], dur[mask])
        j = np.searchsorted(times, 60.0, side="right") - 1
        j = int(np.clip(j, 0, len(times) - 1))

        # correct curve for death under scikit-survival API
        obs_60 = float(cif[death_code][j])

        rows.append({"risk_quartile": label, "n": n,
                     "pred_mean_risk_60d": pred_mean, "obs_risk_60d": obs_60})

    return pd.DataFrame(rows)
