from __future__ import annotations

import argparse, json, os, random
import joblib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import read_hourly, clean_hourly, patient_split, make_windows, make_synthetic_hourly, FEATURES
from .metrics import classification_metrics, alarm_metrics
from .models import build_model


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def predict(model, loader, device):
    model.eval(); out=[]
    with torch.no_grad():
        for x, dt, _ in loader:
            out.append(torch.sigmoid(model(x.to(device), dt.to(device))).cpu().numpy())
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Hourly .csv/.parquet export in documented schema")
    ap.add_argument("--model", choices=["lstm", "cfc"], default="cfc")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=64); ap.add_argument("--window-hours", type=int, default=12)
    ap.add_argument("--horizon-hours", type=int, default=6); ap.add_argument("--tracker", choices=["none", "wandb"], default="none")
    ap.add_argument("--out", default="artifacts"); ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)
    raw = read_hourly(args.data) if args.data else make_synthetic_hourly()
    train_raw, test_raw = patient_split(raw, seed=args.seed)
    train_df, scaler = clean_hourly(train_raw, fit=True); test_df, _ = clean_hourly(test_raw, scaler=scaler)
    train = make_windows(train_df, args.window_hours, args.horizon_hours)
    test = make_windows(test_df, args.window_hours, args.horizon_hours)
    xtr, dttr, ytr, _ = train; xte, dtte, yte, meta = test
    if len(np.unique(ytr)) < 2: raise ValueError("Training split has one label class; use more data.")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(args.model, len(FEATURES), args.hidden_size, .2).to(device)
    pos_weight=torch.tensor([(len(ytr)-ytr.sum())/max(ytr.sum(),1)], device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight); opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(dttr),torch.from_numpy(ytr)),batch_size=args.batch_size,shuffle=True)
    te_loader=DataLoader(TensorDataset(torch.from_numpy(xte),torch.from_numpy(dtte),torch.from_numpy(yte)),batch_size=args.batch_size)
    run = None
    if args.tracker == "wandb":
        import wandb; run=wandb.init(project="icu-early-warning-research", config=vars(args))
    best=-np.inf
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,dt,y in tr_loader:
            opt.zero_grad(); loss=loss_fn(model(x.to(device),dt.to(device)),y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(loss.item())
        p=predict(model,te_loader,device); metrics=classification_metrics(yte,p); metrics["epoch"]=epoch; metrics["train_loss"]=float(np.mean(losses))
        print(json.dumps(metrics));
        if run: run.log(metrics)
        score=np.nan_to_num(metrics["auprc"],nan=-1)
        if score > best:
            best=score
            torch.save({"model":args.model,"state_dict":model.state_dict(),"n_features":len(FEATURES),"hidden_size":args.hidden_size,"window_hours":args.window_hours,"features":FEATURES}, f"{args.out}/{args.model}_best.pt")
    # Evaluate the exact checkpoint that will be reused for inference, not the final epoch.
    saved=torch.load(f"{args.out}/{args.model}_best.pt", map_location=device)
    model.load_state_dict(saved["state_dict"])
    p=predict(model,te_loader,device)
    final={**classification_metrics(yte,p),**alarm_metrics(meta,p,.5,args.horizon_hours)}
    meta.assign(label=yte,risk=p).to_csv(f"{args.out}/test_predictions.csv",index=False)
    # Dashboard input only; keep this directory ignored and use de-identified IDs.
    test_raw.to_csv(f"{args.out}/test_vitals.csv", index=False)
    joblib.dump(scaler,f"{args.out}/scaler.joblib")
    with open(f"{args.out}/metrics.json","w") as f: json.dump(final,f,indent=2)
    print("FINAL",json.dumps(final,indent=2));
    if run: run.finish()

if __name__ == "__main__": main()
