"""Statistical fidelity metrics for comparing real vs. synthetic tabular data."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def column_shapes(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """Column-wise distribution comparison.

    Returns a dict keyed by column name with:
    - Numerical: KS statistic, p-value, mean/std delta.
    - Categorical: Total Variation distance.
    """
    results: Dict[str, Dict] = {}
    for col in real_df.columns:
        if col not in synth_df.columns:
            continue
        r = real_df[col].dropna()
        s = synth_df[col].dropna()

        if pd.api.types.is_numeric_dtype(r):
            stat, pval = ks_2samp(r.values, s.values)
            results[col] = {
                "type": "numerical",
                "ks_statistic": round(float(stat), 4),
                "ks_pvalue": round(float(pval), 4),
                "mean_delta": round(float(r.mean() - s.mean()), 4),
                "std_delta": round(float(r.std() - s.std()), 4),
            }
        else:
            rc = r.value_counts(normalize=True)
            sc = s.value_counts(normalize=True)
            all_cats = set(rc.index) | set(sc.index)
            tv = 0.5 * sum(abs(rc.get(c, 0.0) - sc.get(c, 0.0)) for c in all_cats)
            results[col] = {
                "type": "categorical",
                "tv_distance": round(float(tv), 4),
            }
    return results


def column_pair_trends(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Pearson correlation comparison between column pairs.

    Returns the mean absolute difference in correlation coefficients across
    all numerical column pairs. Lower is better.
    """
    if numerical_cols is None:
        numerical_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) < 2:
        return {}

    r_corr = real_df[numerical_cols].corr().values
    s_corr = synth_df[numerical_cols].corr().values
    mask = np.triu(np.ones_like(r_corr, dtype=bool), k=1)
    diffs = np.abs(r_corr[mask] - s_corr[mask])
    return {"mean_corr_delta": round(float(diffs.mean()), 4)}


def ml_efficacy(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    n_estimators: int = 100,
) -> Dict[str, float]:
    """Train-on-Synthetic Test-on-Real (TSTR) vs. Train-on-Real Test-on-Real (TRTR).

    A high TSTR/TRTR ratio (ideally near 1.0) indicates the synthetic data
    preserves predictive structure.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in real_df.columns if c != target_col]

    le = LabelEncoder()
    le.fit(
        pd.concat([real_df[target_col], synth_df[target_col]]).astype(str)
    )

    def encode(df: pd.DataFrame):
        X = pd.get_dummies(df[feature_cols]).astype(np.float32)
        y = le.transform(df[target_col].astype(str))
        return X, y

    X_real, y_real = encode(real_df)
    X_synth, y_synth = encode(synth_df)
    X_synth = X_synth.reindex(columns=X_real.columns, fill_value=0.0)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    trtr = float(
        cross_val_score(clf, X_real.values, y_real, cv=5, scoring="accuracy").mean()
    )
    clf.fit(X_synth.values, y_synth)
    tstr = float((clf.predict(X_real.values) == y_real).mean())

    return {
        "TRTR": round(trtr, 4),
        "TSTR": round(tstr, 4),
        "efficacy_ratio": round(tstr / trtr, 4) if trtr > 0 else 0.0,
    }


def quality_report(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict:
    """One-call evaluation scorecard combining all fidelity/privacy metrics.

    Returns a nested dict with per-metric detail plus a single ``overall_score``
    in [0, 1] (higher is better), computed as the mean of:
      - shape fidelity  = 1 - mean(KS for numerical, TV for categorical)
      - correlation fidelity = 1 - mean absolute correlation delta
      - ML efficacy ratio (only if ``target_col`` is provided)
    Privacy (NNDR) is reported but not folded into the score, since it trades
    off against fidelity and should be inspected separately.
    """
    # ML efficacy needs the target in both frames; if it's missing from the
    # synthetic data we still report fidelity/privacy but skip efficacy.
    has_target = (
        target_col is not None
        and target_col in real_df.columns
        and target_col in synth_df.columns
    )

    def _drop_target(df):
        return df.drop(columns=[target_col]) if target_col and target_col in df.columns else df

    feature_df_real = _drop_target(real_df)
    feature_df_synth = _drop_target(synth_df)

    shapes = column_shapes(feature_df_real, feature_df_synth)
    distances = []
    for v in shapes.values():
        distances.append(v["ks_statistic"] if v["type"] == "numerical" else v["tv_distance"])
    shape_fidelity = max(0.0, 1.0 - (float(np.mean(distances)) if distances else 0.0))

    pair = column_pair_trends(feature_df_real, feature_df_synth)
    corr_fidelity = max(0.0, 1.0 - pair.get("mean_corr_delta", 0.0)) if pair else None

    privacy = privacy_distance(feature_df_real, feature_df_synth)

    components = [shape_fidelity]
    if corr_fidelity is not None:
        components.append(corr_fidelity)

    efficacy = None
    if has_target:
        efficacy = ml_efficacy(real_df, synth_df, target_col=target_col)
        components.append(min(1.0, efficacy["efficacy_ratio"]))

    return {
        "overall_score": round(float(np.mean(components)), 4),
        "shape_fidelity": round(shape_fidelity, 4),
        "correlation_fidelity": round(corr_fidelity, 4) if corr_fidelity is not None else None,
        "column_shapes": shapes,
        "column_pair_trends": pair,
        "ml_efficacy": efficacy,
        "privacy": privacy,
    }


def privacy_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    n_neighbors: int = 1,
) -> Dict[str, float]:
    """Nearest-neighbour distance ratio (NNDR) — a privacy proxy.

    Measures how close synthetic samples are to real samples relative to
    how close real samples are to *other* real samples. Values near 1.0
    indicate synthetic samples are no closer to real data than real-to-real
    distance, implying low memorisation risk.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    if numerical_cols is None:
        numerical_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_cols:
        return {}

    scaler = StandardScaler()
    R = scaler.fit_transform(real_df[numerical_cols].values.astype(np.float32))
    S = scaler.transform(synth_df[numerical_cols].values.astype(np.float32))

    nn_real = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto").fit(R)
    nn_synth = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(S)

    # Distance from each real point to its nearest real neighbour (excluding itself)
    d_rr = nn_real.kneighbors(R)[0][:, -1]
    # Distance from each real point to its nearest synthetic neighbour
    d_rs = nn_synth.kneighbors(R)[0][:, -1]

    # Avoid division by zero for identical real points
    with np.errstate(divide="ignore", invalid="ignore"):
        nndr = np.where(d_rs > 0, d_rr / d_rs, np.inf)

    return {
        "nndr_mean": round(float(np.nanmean(nndr)), 4),
        "nndr_median": round(float(np.nanmedian(nndr)), 4),
    }
