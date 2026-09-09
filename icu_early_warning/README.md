# ICU deterioration early-warning MVP

**Scope:** a research and competition-demo decision-support prototype. It is **not** a clinical diagnostic device, has not been externally validated, and must not be used for patient care or to make treatment decisions.

The MVP forecasts whether a recorded deterioration outcome occurs in the next six hours from the prior 12 hourly vital-sign observations. It compares a continuous-time CfC (Liquid Neural Network, `ncps`) with an `nn.LSTM` baseline. It uses a patient-level split to prevent a patient's windows leaking between training and test data.

## Fastest runnable path

```bash
cd icu_early_warning
python3 -m venv .venv
source .venv/bin/activate             # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python3 -m src.train --model lstm --epochs 5 --out artifacts/lstm
python3 -m src.train --model cfc --epochs 5 --out artifacts/cfc
streamlit run app.py
```

Omit `--data` only for a synthetic plumbing test. Synthetic metrics are deliberately not clinical evidence and must never appear in a project evaluation slide. The dashboard expects the two separate output folders shown above.

## Data access: choose this order

1. **MVP now:** use the synthetic generator to prove the pipeline, checkpointing, and dashboard.
2. **Competition evidence:** request the credentialed-access process for MIMIC-IV/PhysioNet, complete the required data-use training/agreement, and use only the approved environment. Do not commit, upload, or screen-share row-level patient data.
3. **Fallback:** use a dataset whose license expressly permits your intended competition use. Check its data dictionary and outcome definition before comparing results with MIMIC.

MIMIC-IV is credentialed rather than a public no-login download. Its tables need an extraction step; do not expect the raw database tables to run directly with this MVP.

## Input contract for the extraction step

Create one **hourly** CSV or Parquet file. Each row is one patient-hour; timestamps must be sorted and in ICU-relative hours. Required columns:

`patient_id, time_hours, outcome_time_hours, heart_rate, mean_arterial_pressure, resp_rate, spo2, temperature`

`outcome_time_hours` is the known time of your pre-specified research outcome for every row of an event patient and blank for non-events. Define it before training (for example: first onset of an approved sepsis label, or a carefully specified composite deterioration endpoint). Avoid labels constructed from the same measurements at the prediction instant; that is label leakage.

Example:

```bash
python3 -m src.train --data data/raw/mimic_hourly.parquet --model cfc --epochs 30 --tracker wandb --out artifacts/cfc
python3 -m src.train --data data/raw/mimic_hourly.parquet --model lstm --epochs 30 --tracker wandb --out artifacts/lstm
```

Use `--tracker wandb` only after `wandb login`. For offline/local experiment records, leave the default tracker `none`; each run still saves `metrics.json`, a scaler, a checkpoint, and held-out predictions.

## What the code does

- Cleans impossible vital values, forward-fills only within patient, then uses train-set medians.
- Fits the `StandardScaler` only on training patients.
- Builds causal 12-hour windows and labels an event only when it lies in the next six hours.
- Passes elapsed time (`delta_t`) to CfC; the LSTM also gets it as a feature for a fairer baseline.
- Uses a safe per-sequence CfC wrapper because the currently documented `ncps` PyTorch implementation has an open issue with batched `timespans`. This is correct but slower; pin and re-benchmark a fixed upstream release before scaling to the DGX.
- Saves the best validation/test-AUPRC checkpoint (`*_best.pt`) plus reproducible test predictions.
- Saves de-identified held-out vital timelines only inside the ignored artifact folder for the dashboard. Do not publish them or expose patient-level data in a public demo.
- Reports AUROC, AUPRC, precision, recall, Brier score, approximate false alarms/patient-hour, and median early-alert lead time. Threshold selection must be fixed on a validation set before final test reporting in a competition-grade study.

## Before claiming results

- Split by patient, then use a separate validation set to select alert threshold and model settings.
- State the exact outcome and label timing, cohort inclusion/exclusion, missingness strategy, and no-leakage checks.
- Compare against an LSTM **and** a transparent fixed-threshold vital-sign rule. This starter dashboard shows the learned model; add the threshold-rule series only after you document the clinical rule and its intended use.
- Report confidence intervals and subgroup performance, and describe this as retrospective research—not clinical deployment.

## Suggested project layout

```text
data/raw/          credentialed source export (ignored by git)
data/processed/    optional cached feature tables (ignored by git)
src/data.py        schema, cleaning, split and windowing
src/models.py      CfC and LSTM
src/train.py       training, tracking, checkpoints and metrics
artifacts/         models, predictions, metrics (ignored by git)
app.py             Streamlit/Plotly demo
```
