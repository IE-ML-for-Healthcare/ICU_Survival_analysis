# utils.py
# Shared survival-analysis and experiment-tracking utilities
# Dependencies: lifelines, scikit-survival, scikit-learn, matplotlib, pandas, numpy

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibrationDisplay
from lifelines import CoxPHFitter
from sksurv.nonparametric import cumulative_incidence_competing_risks
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


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
    Implements F(t|x) = Σ_j S_all(t_{j-1}|x) * (1 - exp(-ΔH_all(t_j|x))) * ΔH_event(t_j|x) / ΔH_all(t_j|x)
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

    # Discrete-time version that always stays within [0, 1]:
    # at each step, P(any event) = 1 - exp(-(dHe + dHc)), split between causes in proportion to dHe : dHc
    dH0c = np.diff(H0c_g, prepend=0.0)
    S_prev = np.exp(-(He[:, :-1] + Hc[:, :-1]))      # n x (T-1), survival just before the step
    dHe = np.outer(ph_e, dH0e[1:])                   # n x (T-1)
    dHc = np.outer(ph_c, dH0c[1:])
    dH = dHe + dHc
    with np.errstate(invalid="ignore", divide="ignore"):
        share_e = np.where(dH > 0, dHe / dH, 0.0)
    inc = S_prev * (1.0 - np.exp(-dH)) * share_e

    F = np.zeros((X_df.shape[0], t_grid.size), dtype=float)
    F[:, 1:] = np.cumsum(inc, axis=1)

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


def get_fixed_horizon_labels(y_df: pd.DataFrame, horizon_days: float, discharge_is_known: bool = True):
    """
    Fixed-horizon binary outcome and evaluability for in-hospital death

    Parameters
    ----------
    discharge_is_known : bool, default True
        True (recommended for this dataset): a patient discharged alive before the horizon
        is KNOWN not to die in hospital by the horizon, so they are evaluable with label 0.
        Discharge is a competing event, not loss to follow-up.
        False (legacy behaviour): anyone with event 0 before the horizon is dropped as
        "censored". This removes the healthiest patients and inflates the event rate.

    Returns
    -------
    y_binary : np.ndarray of shape (n,)
        1 if in-hospital death occurred by horizon_days, else 0
    evaluable_mask : np.ndarray of shape (n,)
        True if the outcome at the horizon is known
    """
    time = y_df["duration_days"].astype(float).to_numpy()
    death = y_df["event_death"].astype(bool).to_numpy()
    if discharge_is_known:
        evaluable_mask = np.ones_like(death, dtype=bool)
    else:
        evaluable_mask = (time >= horizon_days) | ((time < horizon_days) & death)
    y_binary = ((time <= horizon_days) & death).astype(int)
    return y_binary, evaluable_mask

# utils.py


def infer_competing_col(y_df: pd.DataFrame, death_col: str = "event_death", preferred: str | None = None) -> str:
    """
    Find a competing-event indicator column in y_df
    Requirements
      - binary 0/1 column
      - not the same column as death_col
      - minimal or zero overlap with death_col (no row should be both death and competing)
    Heuristics
      - look for common discharge/competing names
    """
    if preferred is not None:
        if preferred in y_df.columns:
            _validate_competing(y_df, death_col, preferred)
            return preferred
        raise ValueError(
            f"Preferred competing column '{preferred}' not found in y_df. Available: {list(y_df.columns)}")

    candidates = []
    patterns = ["discharge", "discharged", "alive",
                "home", "competing", "event_comp", "exit"]
    for c in y_df.columns:
        clow = c.lower()
        if c == death_col:
            continue
        if any(p in clow for p in patterns):
            candidates.append(c)

    # filter to binary columns
    def _is_binary(col):
        vals = pd.Series(y_df[col]).dropna().unique()
        return set(np.sort(vals)) <= {0, 1}

    candidates = [c for c in candidates if _is_binary(c)]

    # rank by fewest overlaps with death and most positives
    if len(candidates) == 0:
        raise ValueError(
            "No competing-event column found. Expected a 0/1 indicator such as 'event_discharge'. "
            f"Columns present: {list(y_df.columns)}"
        )

    death = y_df[death_col].astype(int).to_numpy()
    best = None
    best_key = None
    for c in candidates:
        comp = y_df[c].astype(int).to_numpy()
        overlap = int(((death == 1) & (comp == 1)).sum())
        positives = int(comp.sum())
        key = (overlap, -positives)  # prefer zero overlap, then more signal
        if best is None or key < best_key:
            best, best_key = c, key

    _validate_competing(y_df, death_col, best)
    return best


def _validate_competing(y_df: pd.DataFrame, death_col: str, comp_col: str) -> None:
    """Raise with a clear message if the setup is inconsistent"""
    death = y_df[death_col].astype(int).to_numpy()
    comp = y_df[comp_col].astype(int).to_numpy()
    if np.any((death == 1) & (comp == 1)):
        n = int(((death == 1) & (comp == 1)).sum())
        raise ValueError(
            f"Detected {n} rows with both death and competing=1. Check '{comp_col}' coding")
    # optional soft check: at least some 1s in comp_col
    if int(comp.sum()) == 0:
        raise ValueError(
            f"Competing column '{comp_col}' has no positives. Check coding")

# ==== utils.py additions ====


def predict_fixed_horizon_risk_from_cox(cph_model,
                                        X_df: pd.DataFrame,
                                        times: np.ndarray) -> pd.DataFrame:
    """
    Fixed-horizon absolute risk from a single Cox model when no competing event is available
    Risk(t) = 1 - S(t | x)
    Returns columns Risk_7d, Risk_30d, Risk_60d, etc
    """
    times = np.asarray(times, dtype=float)
    sf = cph_model.predict_survival_function(
        X_df, times=times)  # index=times, columns=patients
    risk = 1.0 - sf.T.to_numpy()  # n_patients x n_times
    colmap = {t: f"Risk_{int(t)}d" for t in times}
    return pd.DataFrame(risk, index=X_df.index, columns=[colmap[t] for t in times])


def check_calibration_fixed_horizon(y_true: pd.DataFrame,
                                    predictions: pd.Series,
                                    horizon_days: float,
                                    n_bins: int = 4) -> pd.DataFrame:
    """
    Calibration at a fixed horizon WITHOUT a competing-event column
    Uses evaluable patients only and compares mean predicted risk vs observed death rate
    """
    y_bin, eval_mask = get_fixed_horizon_labels(y_true, horizon_days)
    preds = pd.Series(predictions).loc[y_true.index]
    preds_e = preds[eval_mask]
    y_bin_e = y_bin[eval_mask]

    labels = ["Q1 lowest", "Q2", "Q3",
              "Q4 highest"] if n_bins == 4 else range(1, n_bins + 1)
    q = pd.qcut(preds_e, n_bins, labels=labels)

    rows = []
    for label in pd.Series(q).cat.categories:
        m = (q == label).to_numpy()
        n = int(m.sum())
        rows.append({
            "risk_quartile": label,
            "n": n,
            "pred_mean": float(preds_e[m].mean()) if n > 0 else np.nan,
            "obs_rate": float(np.mean(y_bin_e[m])) if n > 0 else np.nan
        })
    return pd.DataFrame(rows)


def plot_calibration_60d(y_true_bin: np.ndarray,
                         y_prob_raw: np.ndarray,
                         y_prob_cal: np.ndarray,
                         mask: np.ndarray,
                         n_bins: int = 8):
    """
    Familiar sklearn-style calibration plot at 60 days for evaluable patients only
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(
        y_true=y_true_bin[mask],
        y_prob=y_prob_raw[mask],
        n_bins=n_bins,
        name="Cox-CIF raw",
        ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_true=y_true_bin[mask],
        y_prob=y_prob_cal[mask],
        n_bins=n_bins,
        name="Cox-CIF calibrated",
        ax=ax,
    )
    ax.set_title("Calibration at 60 days")
    ax.set_xlabel("Predicted probability at 60 days")
    ax.set_ylabel("Observed fraction at 60 days")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


# ==== utils.py additions: audit + safe path selection ====


def audit_competing_event_feasibility(
    y_df: pd.DataFrame,
    duration_col: str = "duration_days",
    death_col: str = "event_death",
    cap_hint: float | None = None,
    spike_tol: float = 0.05
) -> pd.DataFrame:
    """
    Quick audit to judge if non-death likely equals discharge alive
    Heuristics
      - If many non-death cases end exactly at an apparent censoring cap, that suggests administrative censoring
      - If non-death durations spread broadly with no spike at a cap, dataset may be complete to discharge
    Returns a one-row summary for display
    """
    df = y_df.copy()
    t = df[duration_col].astype(float).to_numpy()
    d = df[death_col].astype(int).to_numpy()

    n = int(len(df))
    n_death = int(d.sum())
    n_nodeath = n - n_death

    # guess a cap as max duration unless provided
    cap = float(cap_hint) if cap_hint is not None else float(np.nanmax(t))
    at_cap = (np.isfinite(t)) & (np.isclose(t, cap))
    nodeath_at_cap = int(((d == 0) & at_cap).sum())
    frac_nodeath_at_cap = nodeath_at_cap / max(n_nodeath, 1)

    q_nodeath = np.percentile(
        t[d == 0], [25, 50, 75]) if n_nodeath > 0 else [np.nan]*3
    q_death = np.percentile(t[d == 1], [25, 50, 75]
                            ) if n_death > 0 else [np.nan]*3

    likely_complete = frac_nodeath_at_cap < spike_tol

    return pd.DataFrame([{
        "n": n,
        "n_death": n_death,
        "n_non_death": n_nodeath,
        "cap_days_checked": cap,
        "non_death_at_cap_frac": round(frac_nodeath_at_cap, 3),
        "non_death_q25": round(q_nodeath[0], 2),
        "non_death_q50": round(q_nodeath[1], 2),
        "non_death_q75": round(q_nodeath[2], 2),
        "death_q50": round(q_death[1], 2) if np.isfinite(q_death[1]) else np.nan,
        "likely_complete_followup": bool(likely_complete)
    }])


def add_competing_if_valid(
    y_df: pd.DataFrame,
    audit_summary: pd.DataFrame,
    death_col: str = "event_death",
    new_comp_col: str = "event_discharge"
) -> pd.DataFrame:
    """
    Adds event_discharge = 1 - event_death only when audit suggests complete follow-up
    Raises if follow-up likely censored
    """
    if not bool(audit_summary["likely_complete_followup"].iloc[0]):
        raise ValueError(
            "Follow-up likely includes administrative censoring. Do not set discharge = 1 - death")
    out = y_df.copy()
    out[new_comp_col] = (out[death_col] == 0).astype(int)
    return out

# --- utils.py additions for generic horizons ---


def check_calibration_competing_risk_at(
    y_true: pd.DataFrame,
    predictions: pd.Series,
    duration_col: str,
    event_col: str,
    competing_col: str,
    horizon_days: float,
    n_bins: int = 4
) -> pd.DataFrame:
    """
    Calibration by risk bin at a given horizon using Aalen Johansen (nonparametric CIF)
    Tie-robust: safely handles duplicate predictions by adapting the number of quantile bins
    Returns: risk_quartile, n, pred_mean_risk, obs_risk
    """
    import numpy as np
    import pandas as pd
    from sksurv.nonparametric import cumulative_incidence_competing_risks

    preds = pd.Series(predictions).astype(float).loc[y_true.index]

    # 1) Make quantile bins that never fail with duplicates
    #    First, ask qcut for bins w/out labels and allow duplicate edges to drop
    try:
        tmp_bins = pd.qcut(preds, n_bins, labels=None, duplicates="drop")
    except ValueError:
        # If still failing due to extreme ties, rank to break ties deterministically
        tmp_bins = pd.qcut(preds.rank(method="first"),
                           n_bins, labels=None, duplicates="drop")

    k = tmp_bins.cat.categories.size  # actual number of bins we can support
    base_labels = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    labels = base_labels[:k] if n_bins == 4 else [
        f"Bin {i+1}" for i in range(k)]

    # Recreate bins with human friendly labels and the final bin count k
    risk_bins = pd.qcut(preds.rank(method="first"), k,
                        labels=labels, duplicates="drop")

    # 2) Build multi-state event indicator: 0 censored, 1 death, 2 competing
    evt = np.where(y_true[event_col] == 1, 1,
                   np.where(y_true[competing_col] == 1, 2, 0)).astype(int)
    dur = y_true[duration_col].astype(float).to_numpy()

    # 3) Aggregate predicted vs observed CIF at the horizon by bin
    rows = []
    for label in pd.Series(risk_bins).cat.categories:
        m = (risk_bins == label).to_numpy()
        n = int(m.sum())
        if n == 0:
            rows.append({"risk_quartile": label, "n": 0,
                        "pred_mean_risk": np.nan, "obs_risk": np.nan})
            continue

        pred_mean = float(preds[m].mean())
        times, cif = cumulative_incidence_competing_risks(evt[m], dur[m])
        j = int(np.searchsorted(times, float(horizon_days), side="right") - 1)
        j = max(0, min(j, len(times) - 1))
        obs = float(cif[1, j])  # scikit-survival: cif[0] = all causes, cif[1] = event 1 (death), cif[2] = event 2

        rows.append({"risk_quartile": label, "n": n,
                    "pred_mean_risk": pred_mean, "obs_risk": obs})

    return pd.DataFrame(rows)


def plot_calibration_at_horizon(
    y_true_bin: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    mask: np.ndarray,
    horizon_days: float,
    n_bins: int = 8
):
    """Sklearn-style calibration plot for a fixed horizon on evaluable patients only"""
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay

    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(
        y_true=y_true_bin[mask],
        y_prob=y_prob_raw[mask],
        n_bins=n_bins,
        name=f"Cox-CIF raw {int(horizon_days)}d",
        ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_true=y_true_bin[mask],
        y_prob=y_prob_cal[mask],
        n_bins=n_bins,
        name=f"Cox-CIF calibrated {int(horizon_days)}d",
        ax=ax,
    )
    ax.set_title(f"Calibration at {int(horizon_days)} days")
    ax.set_xlabel(f"Predicted probability at {int(horizon_days)} days")
    ax.set_ylabel(f"Observed fraction at {int(horizon_days)} days")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def fit_isotonic_calibrator(y_val, p_val):
    """Fit an isotonic probability calibrator on validation predictions"""
    y = np.asarray(y_val, dtype=int)
    p = np.asarray(p_val, dtype=float)
    eps = 1e-6
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(np.clip(p, eps, 1 - eps), y)
    return cal


def apply_calibrator(cal, p):
    """Apply a fitted calibrator to probabilities"""
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    return np.clip(cal.transform(np.clip(p, eps, 1 - eps)), 0, 1)


def fixed_horizon_metrics(y_true, p_prob, mask):
    """AUROC, AUPRC, Brier, n_evaluable on the evaluable cohort for a horizon"""
    y = np.asarray(y_true)[mask]
    p = np.asarray(p_prob)[mask]
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    return {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier), "n_evaluable": int(mask.sum())}

# utils.py


def align_evaluable(labels_dict, pred_df, prob_col):
    """Return aligned arrays y, p for the evaluable cohort at a horizon"""
    mask = labels_dict["mask"]
    y = labels_dict["y_true"]
    df = pd.DataFrame({"y": y, "p": pred_df[prob_col]}).loc[mask].dropna()
    return df["y"].to_numpy().astype(int), df["p"].to_numpy().astype(float)


def decision_curve_df(y, p, thresholds):
    """Net benefit vs threshold and treat-all baseline"""
    N = len(y)
    out = []
    for t in thresholds:
        if t >= 1.0:
            continue
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        pt = t
        nb_model = (tp / N) - (fp / N) * (pt / (1 - pt))
        nb_all = (y.sum() / N) - ((N - y.sum()) / N) * (pt / (1 - pt))
        out.append({"threshold": t, "nb_model": nb_model, "nb_all": nb_all})
    return pd.DataFrame(out)


def select_threshold_by_net_benefit(y_val, p_val, thresholds):
    """Return t* that maximizes net benefit on validation"""
    nb = decision_curve_df(y_val, p_val, thresholds)
    return float(nb.loc[nb["nb_model"].idxmax(), "threshold"])


def fit_isotonic_calibrator(y_val, p_val):
    """Fit isotonic calibration on validation"""
    cal = IsotonicRegression(out_of_bounds="clip")
    eps = 1e-6
    cal.fit(np.clip(p_val, eps, 1 - eps), y_val.astype(int))
    return cal


def apply_calibrator(cal, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.clip(cal.transform(p), 0, 1)


def fixed_horizon_metrics(y_true, p_prob, mask):
    """AUROC, AUPRC, Brier, n_evaluable on the evaluable cohort"""
    y = np.asarray(y_true)[mask]
    p = np.asarray(p_prob)[mask]
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    return {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier), "n_evaluable": int(mask.sum())}


# Helpers for the tracked machine-learning notebook
# Tracking and explanation libraries are imported only when their helpers run


@dataclass
class WandbRunContext:
    """Shared run settings and input versions, updated when data are logged."""

    project: str
    environment: dict
    optional_surrogate: bool
    seed: int
    notebook: Path
    mode: str
    dataset_sha256: Optional[str] = None
    split_sha256: Optional[str] = None
    input_refs: list = field(default_factory=list)


def start_run(name, job_type, config=None, tags=None, group=None, inputs=True, *, context):
    """Record the environment, source code and pinned inputs for one run."""
    import wandb

    shared = {
        "environment": context.environment,
        "optional_surrogate": context.optional_surrogate,
        "split_seed": context.seed,
        "model_seed": context.seed,
        "dataset_sha256": context.dataset_sha256,
        "main_split_sha256": context.split_sha256,
    }
    run = wandb.init(
        project=context.project, name=name, job_type=job_type,
        config={**shared, **(config or {})}, tags=tags or [], group=group,
        reinit="finish_previous", save_code=True,
    )
    source_files = {context.notebook.name, Path(__file__).name}
    # Recorded runs predate this name; keep both files separate from automatic notebook snapshots
    run.log_code(root=".", name="icu-notebook-and-utils",
                 include_fn=lambda p: Path(p).name in source_files)
    if context.mode == "online" and inputs:
        for ref in context.input_refs:
            run.use_artifact(ref)
    return run


def calibration_table(y_true, y_prob, n_bins=10):
    """Quantile bins of predicted risk vs observed death rate, and the Expected Calibration Error (ECE):
    the patient-weighted average gap between the two"""
    q = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    if pd.isna(q).all():  # all predictions are equal: use one bin
        q = pd.Categorical(["all patients"] * len(y_prob))
    dfb = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": y_prob, "bin": q})
    agg = dfb.groupby("bin", observed=False).agg(
        mean_prob=("y_prob", "mean"), observed_rate=("y_true", "mean"), count=("y_true", "size")).reset_index()
    agg["bin"] = agg["bin"].astype(str)
    ece = float(np.sum(agg["count"] / agg["count"].sum() * np.abs(agg["observed_rate"] - agg["mean_prob"])))
    return agg, ece


def score(y_true, y_prob):
    """Return ranking, probability-error and calibration metrics for one split."""
    return {"auc": roc_auc_score(y_true, y_prob), "pr": average_precision_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob), "ece": calibration_table(y_true, y_prob)[1]}


def log_curves(y_true, y_prob, split):
    """Log receiver operating characteristic and precision-recall curves."""
    import wandb
    from wandb.plot import roc_curve as wandb_roc_curve
    from wandb.plot import pr_curve as wandb_pr_curve

    probas_2d = np.stack([1.0 - y_prob, y_prob], axis=1)
    wandb.log({
        f"roc_curve_{split}": wandb_roc_curve(np.asarray(y_true), probas_2d),
        f"pr_curve_{split}": wandb_pr_curve(np.asarray(y_true), probas_2d),
    })


def log_top_bar(df, label, value, key, title, k=15):
    """W&B bar chart of the k rows with the largest absolute value"""
    import wandb

    top = df.reindex(df[value].abs().sort_values(ascending=False).index).head(k)
    wandb.log({key: wandb.plot.bar(wandb.Table(dataframe=top[[label, value]]), label, value, title=title)})


def log_fig(key, fig, *, tight=True):
    """Log a matplotlib figure to W&B and also show it in the notebook"""
    import wandb

    wandb.log({key: wandb.Image(fig)})
    if tight:
        fig.tight_layout()
    plt.show()
    plt.close(fig)


def rf_shap(model, X_sample):
    """Return SHapley Additive exPlanations (SHAP) values and base risk for class 1.

    Values have shape (patients, features) and use probability units for a forest.
    """
    import shap

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X_sample)
    sv = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    base = np.atleast_1d(explainer.expected_value)
    return sv, float(base[1] if base.size > 1 else base[0])


def make_split(split_seed, *, idx_dev, idx_test, y):
    """Split development 75/25, retaining the same held-out test patients."""
    itr, iva = train_test_split(idx_dev, test_size=0.25, stratify=y.loc[idx_dev], random_state=split_seed)
    assert set(itr).isdisjoint(idx_test) and set(iva).isdisjoint(idx_test)
    return itr, iva, idx_test


def make_preprocessor(num_cols, cat_cols):
    """Replace source missing codes, impute and encode the supplied columns."""
    columns = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([
        ("source_missing", FunctionTransformer(pd.DataFrame.replace,
             kw_args={"to_replace": -1, "value": np.nan}, feature_names_out="one-to-one")),
        ("columns", columns),
    ])


def prepare(X, itr, iva, *, num_cols, cat_cols):
    """Fit preprocessing on training patients and transform validation inputs."""
    pre = make_preprocessor(num_cols, cat_cols)
    return pre.fit_transform(X.loc[itr]), pre.transform(X.loc[iva])


def split_manifest(itr, iva, ite, *, ICU, id_col):
    """List patient identifiers and their train, validation or test assignment."""
    return pd.concat([pd.DataFrame({id_col: ICU.loc[idx, id_col].values, "split": name})
                      for name, idx in [("train", itr), ("val", iva), ("test", ite)]], ignore_index=True)


def run_baseline(name, model, config, tags, *, X_train_t, y_train, X_val_t, y_val, seed, start_run):
    """Fit one model, log validation metrics and curves, return the open run, the fitted model and its validation risks"""
    import wandb

    run = start_run(name, "baseline", config={**config, "seed": seed}, tags=["baseline", *tags])
    model.fit(X_train_t, y_train)
    p_val = model.predict_proba(X_val_t)[:, 1]
    metrics = {f"val_{k}": v for k, v in score(y_val, p_val).items()}
    wandb.log(metrics)
    log_curves(y_val, p_val, "val")
    return run, model, p_val, metrics


def perm_drop(model, feature, X_val_t, y_val, *, seed, repeats=5):
    """Mean drop in validation average precision when one feature is shuffled"""
    base = average_precision_score(y_val, model.predict_proba(X_val_t)[:, 1])
    drops = []
    for r in range(repeats):
        Xp = X_val_t.copy()
        Xp[feature] = np.random.default_rng(seed + r).permutation(Xp[feature].to_numpy())
        drops.append(base - average_precision_score(y_val, model.predict_proba(Xp)[:, 1]))
    return float(np.mean(drops))


def train_rf_sweep(*, X_train_t, y_train, X_val_t, y_val, seed, start_run):
    """Fit and log one sweep trial using the supplied development data."""
    import wandb

    run = start_run(None, "sweep-agent", tags=["sweep"])
    cfg = wandb.config
    from time import perf_counter
    started = perf_counter()
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators, max_depth=None if cfg.max_depth == 0 else cfg.max_depth,
        max_features=cfg.max_features, min_samples_leaf=cfg.min_samples_leaf,
        oob_score=True, random_state=seed, n_jobs=-1).fit(X_train_t, y_train)
    fit_seconds = perf_counter() - started
    sc = score(y_val, model.predict_proba(X_val_t)[:, 1])
    oob = model.oob_decision_function_[:, 1]
    m = np.isfinite(oob) & (model.oob_decision_function_.sum(axis=1) > 0)
    train_pr = average_precision_score(y_train, model.predict_proba(X_train_t)[:, 1])
    wandb.log({**{f"val_{k}": v for k, v in sc.items()},
               "fit_seconds": fit_seconds, "train_pr": train_pr, "overfit_gap_pr": train_pr - sc["pr"],
               "oob_pr": average_precision_score(y_train.iloc[np.flatnonzero(m)], oob[m]),
               "mean_tree_depth": float(np.mean([e.get_depth() for e in model.estimators_])),
               "mean_leaves": float(np.mean([e.get_n_leaves() for e in model.estimators_]))})
    run.finish()


def display_details_table(frame, title):
    """Keep exact values available without making them the main visual output."""
    from html import escape
    from IPython.display import HTML, display

    display(HTML(
        f"<details><summary>{escape(title)}</summary>"
        + frame.to_html(border=0) + "</details>"
    ))


def plot_patient_split(train_rows, val_rows, test_rows):
    """Show the development pool and the test patients kept outside model choice."""
    fig, ax = plt.subplots(figsize=(10, 2.7))
    left = 0
    for name, size, color in zip(
        ["Training", "Validation", "Test"],
        [train_rows, val_rows, test_rows],
        ["#286A9F", "#BF7A25", "#68737D"],
    ):
        ax.barh(0, size, left=left, height=0.45, color=color, edgecolor="white")
        ax.text(left + size / 2, 0, f"{name}\n{size:,}", ha="center", va="center", color="white")
        left += size
    ax.set(xlim=(0, left), ylim=(-0.7, 0.7), yticks=[], xlabel="Number of ICU stays")
    ax.set_title("Separate development data from the fixed test patients", pad=18)
    ax.text((train_rows + val_rows) / 2, -0.43, "Development pool: models are compared here", ha="center")
    ax.text(train_rows + val_rows + test_rows / 2, -0.43, "Final evaluation", ha="center")
    ax.spines[["left", "top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_baseline_comparison(baseline_df, positive_rate):
    """Compare ranking and probability error on the same validation patients."""
    order = ["logistic_regression", "decision_tree", "random_forest"]
    labels = ["Logistic regression", "Shallow tree", "Random forest"]
    colors = ["#87929A", "#87929A", "#286A9F"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, metric, title, xlabel in zip(
        axes, ["val_pr", "val_brier"],
        ["Ranking: higher is better", "Probability error: lower is better"],
        ["Validation average precision", "Validation Brier score"],
    ):
        values = baseline_df.loc[order, metric].to_numpy(dtype=float)
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        upper = max(values) * 1.25
        ax.set(xlim=(0, upper), xlabel=xlabel, title=title)
        for i, value in enumerate(values):
            ax.text(value + upper * 0.02, i, f"{value:.3f}", va="center")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].axvline(positive_rate, color="#333333", linestyle="--")
    fig.suptitle("Three models, the same validation patients", fontsize=14)
    fig.text(0.5, 0.015, f"Dashed line: validation death fraction {positive_rate:.3f}, the average-precision reference", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_variance_comparison(var_df):
    """Show observed score spread and paired model differences without inference bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    groups = [("model_seed", "Forest seed changes", "#286A9F", "o"),
              ("data_split", "Patient split changes", "#BF7A25", "s")]
    for i, (source, label, color, marker) in enumerate(groups):
        values = var_df.loc[var_df.source == source, "pr"].to_numpy(dtype=float)
        offsets = np.linspace(-0.10, 0.10, len(values))
        axes[0].scatter(values, i + offsets, color=color, marker=marker, s=55, zorder=3)
        axes[0].text(float(np.mean(values)), i + 0.28, f"SD = {np.std(values, ddof=1):.3f}", ha="center")
    axes[0].set(yticks=[0, 1], yticklabels=[g[1] for g in groups],
                ylim=(1.6, -0.5), xlabel="Validation average precision", title="What moves the score?")
    axes[0].grid(axis="x", alpha=0.18)
    paired = var_df.loc[var_df.source == "data_split"].sort_values("seed")
    values = paired.rf_minus_lr_pr.to_numpy(dtype=float)
    y = np.arange(len(paired))
    axes[1].barh(y, values, color=["#286A9F" if v >= 0 else "#BF7A25" for v in values])
    axes[1].set(yticks=y, yticklabels=[f"Split {int(s)}" for s in paired.seed],
                xlabel="Forest minus logistic regression\naverage precision", title="Paired comparison on each split")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#333333", linewidth=1)
    margin = max(float(np.abs(values).max()), 0.01) * 0.22
    axes[1].set_xlim(min(0, float(values.min())) - margin, max(0, float(values.max())) + margin * 1.8)
    for i, value in enumerate(values):
        axes[1].text(value + (margin * 0.12 if value >= 0 else -margin * 0.12), i,
                     f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right")
    fig.suptitle("Five repetitions: score spread and model advantage", fontsize=14)
    fig.text(0.5, 0.015, "The splits overlap; the spread is descriptive, not a confidence interval", ha="center")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_sanity_comparison(null_comparison):
    """Put held-out prediction quality beside the shuffled-model feature ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(axes, ["AUROC", "average precision"],
                                ["Validation ranking area", "Validation average precision"]):
        values = null_comparison.loc[["real labels", "shuffled training labels"], metric].to_numpy(dtype=float)
        ax.barh(["Original outcomes", "Shuffled outcomes"], values, color=["#286A9F", "#87929A"])
        ax.invert_yaxis()
        reference = float(null_comparison.loc["chance reference", metric])
        ax.axvline(reference, color="#333333", linestyle="--")
        ax.set(xlim=(0, max(values.max(), reference) * 1.25), xlabel=title)
        for i, value in enumerate(values):
            ax.text(value + values.max() * 0.025, i, f"{value:.3f}", va="center")
    fig.suptitle("A feature ranking can remain when prediction becomes weak", fontsize=14)
    fig.text(0.5, 0.015, f"Dashed lines: chance ranking 0.500; chance average precision {null_comparison.loc['chance reference', 'average precision']:.3f}", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def sweep_results_frame(completed_runs):
    """Read a small plotting table from the same scored runs used for model selection."""
    return pd.DataFrame([
        {"run": run.name, "val_pr": float(run.summary["val_pr"]),
         "fit_seconds": float(run.summary.get("fit_seconds", np.nan)),
         "mean_leaves": float(run.summary.get("mean_leaves", np.nan))}
        for run in completed_runs
    ])


def plot_sweep_tradeoff(sweep_results, baseline_pr):
    """Plot recorded sweep results read back from W&B; do not fit or select a model."""
    data = sweep_results.dropna(subset=["fit_seconds", "val_pr"])
    if data.empty:
        raise ValueError("No completed sweep trials with recorded fit_seconds and val_pr")
    best = data.loc[data.val_pr.idxmax()]
    fastest = data.loc[data.fit_seconds.idxmin()]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.scatter(data.fit_seconds, data.val_pr, color="#87929A", s=60, label="Completed trials")
    ax.scatter([best.fit_seconds], [best.val_pr], color="#286A9F", marker="D", s=90, label="Highest validation score")
    ax.axhline(baseline_pr, color="#333333", linestyle="--", label=f"Baseline {baseline_pr:.3f}")
    ax.annotate(f"Best: {best.val_pr:.3f}\n{best.fit_seconds:.2f} s", (best.fit_seconds, best.val_pr),
                xytext=(12, 12), textcoords="offset points")
    if best.name != fastest.name:
        ax.annotate(f"Fastest: {fastest.val_pr:.3f}\n{fastest.fit_seconds:.2f} s", (fastest.fit_seconds, fastest.val_pr),
                    xytext=(12, -32), textcoords="offset points")
    ax.set(xlabel="Recorded training time [s]", ylabel="Validation average precision",
           title="W&B sweep results: performance against training time",
           xlim=(0, float(data.fit_seconds.max()) * 1.30),
           ylim=(min(float(data.val_pr.min()), baseline_pr) - 0.03,
                 max(float(data.val_pr.max()), baseline_pr) + 0.04))
    ax.grid(alpha=0.18)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


def plot_reproducibility_chain(input_refs, test_metrics, gap_file, gap_refit):
    """Summarise the recorded inputs, pipeline and reconstruction checks."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    inputs = "\n".join(ref.rsplit("/", 1)[-1] for ref in input_refs)
    inputs = "Versioned inputs\n" + inputs if inputs else "Local inputs\nDataset + patient split\nRecorded checksums"
    boxes = [(0.17, 0.74, inputs),
             (0.50, 0.74, f"Final evaluation\nTest average precision\n{test_metrics['test_pr']:.3f}"),
             (0.83, 0.74, "Pipeline artifact\nPreprocessing + forest\nSettings + library versions"),
             (0.40, 0.24, f"Reload saved file\nMaximum probability gap\n{gap_file:.1e}"),
             (0.80, 0.24, f"Rebuild from settings\nMaximum probability gap\n{gap_refit:.1e}")]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.145, y - 0.16), 0.29, 0.32,
                                   boxstyle="round,pad=0.008", facecolor="#F1F5F7", edgecolor="#286A9F"))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5)
    for start, end in [((0.32, 0.74), (0.345, 0.74)), ((0.65, 0.74), (0.675, 0.74)),
                       ((0.77, 0.57), (0.48, 0.42)), ((0.85, 0.57), (0.82, 0.42))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5})
    ax.set_title("The recorded experiment connects the data, model and checks", pad=15)
    fig.text(0.5, 0.015, "Rebuilding also uses the same logged input data; matching predictions do not establish clinical validity", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
