"""Public-safe Streamlit demo for the ICU early-warning research project."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
from src.data import FEATURES, make_synthetic_hourly

st.set_page_config(page_title="ICU Early Warning", page_icon="🫀", layout="wide")


def simulated_predictions():
    """Deterministic visual demo only; these are not model results or evidence."""
    vitals = make_synthetic_hourly(n_patients=24, hours=36, seed=17)
    rows = []
    for patient_id, g in vitals.groupby("patient_id", sort=False):
        for _, r in g[g.time_hours >= 11].iterrows():
            lead = r.outcome_time_hours - r.time_hours
            signal = 0.0 if pd.isna(lead) else max(0.0, 1 - lead / 13)
            cfc = float(np.clip(0.06 + 0.84 * signal + 0.02 * np.sin(r.time_hours), 0.01, 0.99))
            lstm = float(np.clip(0.07 + 0.68 * max(0, signal - 0.15), 0.01, 0.99))
            rows.append((patient_id, r.time_hours, r.outcome_time_hours, cfc, lstm))
    return pd.DataFrame(rows, columns=["patient_id", "time_end", "outcome_time_hours", "cfc_risk", "lstm_risk"]), vitals


@st.cache_data(show_spinner=False)
def load_demo_data():
    paths = {n: os.path.join(ROOT, "artifacts", n, "test_predictions.csv") for n in ("cfc", "lstm")}
    vitals_path = os.path.join(ROOT, "artifacts", "cfc", "test_vitals.csv")
    if all(os.path.exists(p) for p in paths.values()) and os.path.exists(vitals_path):
        cfc = pd.read_csv(paths["cfc"]).rename(columns={"risk": "cfc_risk"})
        lstm = pd.read_csv(paths["lstm"])[["patient_id", "time_end", "risk"]].rename(columns={"risk": "lstm_risk"})
        return cfc.merge(lstm, on=["patient_id", "time_end"], how="inner"), pd.read_csv(vitals_path), "held-out experiment outputs"
    predictions, vitals = simulated_predictions()
    return predictions, vitals, "illustrative synthetic simulation"


def fixed_alarm(vitals):
    """Transparent comparator, not a clinical alarm policy."""
    return vitals.heart_rate.gt(120) | vitals.mean_arterial_pressure.lt(65) | vitals.spo2.lt(90)


predictions, all_vitals, source = load_demo_data()
st.title("🫀 ICU Early Warning")
st.caption("Patient-adaptive deterioration-risk research prototype · CfC/Liquid Neural Network vs LSTM")
st.warning("Research and education demo only — not clinically validated and not for diagnosis, treatment, or patient care.")

with st.sidebar:
    st.header("Demo controls")
    patient = st.selectbox("Patient timeline", sorted(predictions.patient_id.astype(str).unique()))
    threshold = st.slider("Research alert threshold", 0.05, 0.95, 0.50, 0.05)
    st.divider()
    st.caption(f"Data source: **{source}**")
    st.caption("Public deployments use synthetic mode until private experiment artifacts are supplied.")

g = predictions[predictions.patient_id.astype(str) == patient].sort_values("time_end")
vitals = all_vitals[all_vitals.patient_id.astype(str) == patient].sort_values("time_hours")
at_end = vitals.set_index("time_hours").reindex(g.time_end)
threshold_hits = fixed_alarm(at_end)
event_time = g.outcome_time_hours.dropna().iloc[0] if g.outcome_time_hours.notna().any() else None

summary, timeline, methods = st.tabs(["Overview", "Patient timeline", "How it works"])
with summary:
    st.subheader("A smarter early-warning research question")
    st.write("Can a continuous-time model identify elevated deterioration risk from changing vital-sign patterns earlier and with fewer unnecessary alerts than simple fixed thresholds?")
    a, b, c, d = st.columns(4)
    a.metric("Latest CfC risk", f"{g.cfc_risk.iloc[-1]:.0%}")
    b.metric("Latest LSTM risk", f"{g.lstm_risk.iloc[-1]:.0%}")
    c.metric("CfC research alert", "YES" if g.cfc_risk.iloc[-1] >= threshold else "NO")
    d.metric("Fixed-rule alerts", int(threshold_hits.sum()))
    if source == "illustrative synthetic simulation":
        st.info("Synthetic mode uses visual simulation values, not clinical predictions or reported study results.")
    else:
        st.success("Showing saved held-out experiment outputs. Interpret metrics only within the documented research protocol.")

with timeline:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.time_end, y=g.cfc_risk, name="CfC / liquid risk", line=dict(color="#e15759", width=3)))
    fig.add_trace(go.Scatter(x=g.time_end, y=g.lstm_risk, name="LSTM risk", line=dict(color="#4e79a7", width=2)))
    fig.add_trace(go.Scatter(x=g.time_end[threshold_hits.to_numpy()], y=[0.02] * int(threshold_hits.sum()), mode="markers", name="Illustrative fixed-threshold alert", marker=dict(symbol="x", size=10, color="#f28e2b")))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#555", annotation_text="Research threshold")
    if event_time is not None:
        fig.add_vline(x=event_time, line_color="#222", line_dash="dot", annotation_text="Recorded/simulated outcome")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=35, b=10), yaxis=dict(tickformat=".0%", range=[0, 1], title="Risk score"), xaxis_title="ICU hour", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Fixed-rule comparator: HR > 120, MAP < 65, or SpO₂ < 90. It is illustrative, not a validated alarm policy.")
    st.subheader("Vital-sign trajectories")
    vf = go.Figure()
    for feature, color in zip(FEATURES, ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#b07aa1"]):
        vf.add_trace(go.Scatter(x=vitals.time_hours, y=vitals[feature], name=feature.replace("_", " ").title(), line=dict(color=color)))
    vf.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10), xaxis_title="ICU hour", yaxis_title="Native units (different scales)", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(vf, use_container_width=True)

with methods:
    st.subheader("Experimental design")
    st.markdown("""1. **Prepare approved retrospective ICU data:** clean implausible values and define the outcome before training.
2. **Create causal windows:** use 12 prior hours to forecast a recorded event in the next 6 hours.
3. **Compare fairly:** train CfC and LSTM models using the same patient-level split.
4. **Evaluate meaningful outcomes:** AUROC, AUPRC, calibration, false alarms per patient-hour, and warning lead time.
5. **Keep claims proportionate:** this studies retrospective decision support; it does not diagnose disease or replace clinicians.""")
    st.subheader("Run real experiments locally")
    st.code("python3 -m src.train --data data/raw/mimic_hourly.parquet --model cfc --epochs 30 --out artifacts/cfc\npython3 -m src.train --data data/raw/mimic_hourly.parquet --model lstm --epochs 30 --out artifacts/lstm", language="bash")
    st.caption("Never commit or deploy row-level clinical data, credentials, or patient-level prediction artifacts to a public repository.")

metrics_path = os.path.join(ROOT, "artifacts", "cfc", "metrics.json")
if source == "held-out experiment outputs" and os.path.exists(metrics_path):
    with st.expander("Held-out CfC metrics"):
        st.json(json.load(open(metrics_path)))
