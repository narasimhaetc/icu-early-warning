import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss


def classification_metrics(y, p, threshold=0.5):
    pred = p >= threshold
    result = {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
              "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
              "precision": float(precision_score(y, pred, zero_division=0)),
              "recall": float(recall_score(y, pred, zero_division=0)),
              "brier": float(brier_score_loss(y, p))}
    return result


def alarm_metrics(meta, probabilities, threshold, horizon_hours=6, stride_hours=1):
    m = meta.copy(); m["risk"] = probabilities; m["alarm"] = m.risk >= threshold
    # A false alarm is an alarm outside the prospective prediction horizon.
    lead = m.outcome_time_hours - m.time_end
    false = m.alarm & ~lead.between(0, horizon_hours, inclusive="right")
    observation_hours = max(len(m) * stride_hours, 1)
    event_leads = []
    for _, g in m.dropna(subset=["outcome_time_hours"]).groupby("patient_id"):
        before = g[(g.alarm) & (g.time_end < g.outcome_time_hours)]
        if len(before):
            event_leads.append(float(g.outcome_time_hours.iloc[0] - before.time_end.min()))
    return {"false_alarms_per_patient_hour": float(false.sum() / observation_hours),
            "median_lead_hours": float(np.median(event_leads)) if event_leads else float("nan"),
            "events_with_early_alarm": int(len(event_leads))}
