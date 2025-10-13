# utils.py
# Minimal, student-friendly utilities for Kaplan–Meier plotting by group
# Dependencies: lifelines, matplotlib, pandas, numpy

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


def get_fixed_horizon_labels(y_df: pd.DataFrame, horizon_days: float):
    """
    Fixed-horizon binary outcome and evaluability in survival data

    Returns
    -------
    y_binary : np.ndarray of shape (n,)
        1 if death occurred by horizon_days, else 0
    evaluable_mask : np.ndarray of shape (n,)
        True if not censored before the horizon (follow-up ≥ horizon or death before horizon)
    """
    time = y_df["duration_days"].astype(float).to_numpy()
    death = y_df["event_death"].astype(bool).to_numpy()
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
        obs = float(cif[0, j])  # index 0 corresponds to event type 1 (death)

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
