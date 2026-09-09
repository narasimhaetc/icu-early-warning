from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["heart_rate", "mean_arterial_pressure", "resp_rate", "spo2", "temperature"]
REQUIRED = {"patient_id", "time_hours", "outcome_time_hours", *FEATURES}


def read_hourly(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values(["patient_id", "time_hours"]).reset_index(drop=True)


def clean_hourly(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False):
    df = df.copy().sort_values(["patient_id", "time_hours"])
    # Physiologically implausible readings are treated as missing, not silently trusted.
    bounds = {"heart_rate": (20, 250), "mean_arterial_pressure": (20, 200),
              "resp_rate": (4, 80), "spo2": (40, 100), "temperature": (30, 43)}
    for col, (low, high) in bounds.items():
        df.loc[~df[col].between(low, high), col] = np.nan
    df[FEATURES] = df.groupby("patient_id", group_keys=False)[FEATURES].ffill()
    medians = df[FEATURES].median()
    df[FEATURES] = df[FEATURES].fillna(medians)
    if fit:
        scaler = StandardScaler().fit(df[FEATURES])
    if scaler is None:
        raise ValueError("Pass scaler or set fit=True")
    df[FEATURES] = scaler.transform(df[FEATURES])
    return df, scaler


def make_windows(df: pd.DataFrame, window_hours: int = 12, horizon_hours: int = 6, stride: int = 1):
    """Create causal fixed-length windows; y=1 iff a recorded event is in next horizon."""
    xs, dts, ys, meta = [], [], [], []
    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("time_hours")
        values = g[FEATURES].to_numpy(np.float32)
        times = g.time_hours.to_numpy(np.float32)
        outcomes = g.outcome_time_hours.to_numpy(np.float32)
        for end in range(window_hours - 1, len(g), stride):
            start = end - window_hours + 1
            # Require an hourly grid in this MVP. A future irregular raw-event loader can
            # preserve event times; delta_t is already carried through to CfC.
            if np.max(np.diff(times[start:end + 1])) > 1.01:
                continue
            outcome = outcomes[end]
            lead = outcome - times[end] if np.isfinite(outcome) else np.inf
            xs.append(values[start:end + 1])
            dts.append(np.r_[1.0, np.diff(times[start:end + 1])])
            ys.append(float(0 < lead <= horizon_hours))
            meta.append((patient_id, times[end], outcome))
    if not xs:
        raise ValueError("No usable windows; check time_hours and window length.")
    return (np.stack(xs), np.stack(dts).astype(np.float32), np.asarray(ys, dtype=np.float32),
            pd.DataFrame(meta, columns=["patient_id", "time_end", "outcome_time_hours"]))


def patient_split(df: pd.DataFrame, test_fraction=0.2, seed=42):
    ids = np.asarray(sorted(df.patient_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = set(ids[:n_test])
    return df[~df.patient_id.isin(test_ids)].copy(), df[df.patient_id.isin(test_ids)].copy()


def make_synthetic_hourly(n_patients=160, hours=36, seed=7) -> pd.DataFrame:
    """Only for plumbing/demo validation — never report these results as clinical evidence."""
    rng = np.random.default_rng(seed); rows = []
    for p in range(n_patients):
        event = rng.random() < 0.28
        event_time = float(rng.integers(18, hours)) if event else np.nan
        baseline = np.array([78, 78, 17, 97, 37.0]) + rng.normal(0, [8, 7, 2, 1, .25])
        for t in range(hours):
            proximity = max(0, 1 - (event_time - t) / 12) if event else 0
            trend = np.array([30, -28, 12, -6, 1.4]) * proximity
            v = baseline + trend + rng.normal(0, [5, 5, 1.5, .8, .15])
            rows.append([f"demo_{p:04d}", t, event_time, *v])
    return pd.DataFrame(rows, columns=["patient_id", "time_hours", "outcome_time_hours", *FEATURES])
