import json, os, sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT=os.path.dirname(__file__); sys.path.insert(0,ROOT)
from src.data import FEATURES, make_synthetic_hourly

st.set_page_config(page_title="ICU Early Warning — Research Demo", layout="wide")
st.title("ICU Early Warning · Research Demo")
st.warning("Research/demo decision support only. Not validated for clinical use; do not use for diagnosis, treatment, or patient care.")
paths={name: os.path.join(ROOT,"artifacts",name,"test_predictions.csv") for name in ("cfc","lstm")}
if not all(os.path.exists(p) for p in paths.values()):
    st.info("Train both models first: `python3 -m src.train --model cfc --out artifacts/cfc` and the same command with `--model lstm --out artifacts/lstm`.")
    st.stop()
preds={name: pd.read_csv(path).rename(columns={"risk": f"{name}_risk"}) for name,path in paths.items()}
pred=preds["cfc"].merge(preds["lstm"][["patient_id","time_end","lstm_risk"]],on=["patient_id","time_end"],how="inner")
patients=sorted(pred.patient_id.astype(str).unique())
patient=st.sidebar.selectbox("Held-out patient",patients)
threshold=st.sidebar.slider("Research alert threshold",0.05,0.95,0.50,0.05)
g=pred[pred.patient_id.astype(str)==patient].sort_values("time_end")
vitals_path=os.path.join(ROOT,"artifacts","cfc","test_vitals.csv")
vitals=pd.read_csv(vitals_path) if os.path.exists(vitals_path) else make_synthetic_hourly()
vitals=vitals[vitals.patient_id.astype(str)==patient]
fig=go.Figure()
fig.add_trace(go.Scatter(x=g.time_end,y=g.cfc_risk,name="CfC / liquid risk",line=dict(color="#d62728",width=3)))
fig.add_trace(go.Scatter(x=g.time_end,y=g.lstm_risk,name="LSTM risk",line=dict(color="#1f77b4",width=2)))
# Deliberately illustrative, transparent fixed-rule comparator—not a clinical rule.
end_vitals=vitals.set_index("time_hours").reindex(g.time_end)
fixed=(end_vitals.heart_rate.gt(120) | end_vitals.mean_arterial_pressure.lt(65) | end_vitals.spo2.lt(90))
fig.add_trace(go.Scatter(x=g.time_end[fixed.to_numpy()],y=[0.02]*int(fixed.sum()),mode="markers",name="Illustrative fixed-threshold alert",marker=dict(symbol="x",size=10,color="#ff7f0e")))
fig.add_hline(y=threshold,line_dash="dash",annotation_text="Research threshold")
if g.outcome_time_hours.notna().any(): fig.add_vline(x=g.outcome_time_hours.dropna().iloc[0],line_color="black",annotation_text="Recorded outcome")
fig.update_layout(yaxis_tickformat=".0%",yaxis_range=[0,1],xaxis_title="ICU hour",yaxis_title="Risk score")
st.plotly_chart(fig,use_container_width=True)
st.caption("Fixed-threshold markers are an illustrative comparator (HR > 120, MAP < 65, or SpO₂ < 90), not a validated clinical alarm policy.")
cols=st.columns(3); cols[0].metric("Latest CfC risk",f"{g.cfc_risk.iloc[-1]:.0%}"); cols[1].metric("CfC research alert","YES" if g.cfc_risk.iloc[-1]>=threshold else "NO"); cols[2].metric("Windows shown",len(g))
st.subheader("Vital-sign timeline (demo data)")
vf=go.Figure()
for f in FEATURES: vf.add_trace(go.Scatter(x=vitals.time_hours,y=vitals[f],name=f.replace("_"," ").title()))
vf.update_layout(xaxis_title="ICU hour",yaxis_title="Native units (different scales)")
st.plotly_chart(vf,use_container_width=True)
if os.path.exists(os.path.join(ROOT,"artifacts","cfc","metrics.json")):
    st.subheader("Held-out experiment metrics")
    st.json({name: json.load(open(os.path.join(ROOT,"artifacts",name,"metrics.json"))) for name in ("cfc","lstm")})
