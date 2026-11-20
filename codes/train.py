"""Train and evaluate a cost-sensitive, class-weighted fraud model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "result"
DATA = ROOT / "data" / "creditcard.csv"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run: python download_data.py")
    df = pd.read_csv(DATA)
    if "Class" not in df: raise ValueError("Expected Kaggle column 'Class'")
    df["Hour"] = ((df["Time"] / 3600) % 24).astype(float)
    X = df[FEATURES].astype(float)
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(n_estimators=450, max_depth=4, learning_rate=.045,
        subsample=.85, colsample_bytree=.85, min_child_weight=2,
        scale_pos_weight=ratio, eval_metric="aucpr", tree_method="hist",
        random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    # Threshold chosen by estimated expected loss. Demo assumptions are configurable in app.
    thresholds = np.linspace(.01, .99, 99)
    costs = [float(np.sum((y_test.to_numpy() == 1) & (p < t)) * 150 +
                   np.sum((y_test.to_numpy() == 0) & (p >= t)) * 10) for t in thresholds]
    threshold = float(thresholds[int(np.argmin(costs))])
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    metrics = {"rows": int(len(df)), "frauds": int(y.sum()), "fraud_rate": float(y.mean()),
       "test_rows": int(len(y_test)), "pr_auc": float(average_precision_score(y_test,p)),
       "roc_auc": float(roc_auc_score(y_test,p)), "threshold": threshold,
       "precision": float(precision_score(y_test,pred,zero_division=0)),
       "recall": float(recall_score(y_test,pred,zero_division=0)),
       "f1": float(f1_score(y_test,pred,zero_division=0)),
       "confusion_matrix": cm.tolist(), "test_fraud_count": int(y_test.sum()),
       "split": "stratified random 75/25; random_state=42"}
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold,
                 "test_X": X_test.reset_index(drop=True), "test_y": y_test.reset_index(drop=True),
                 "test_p": p}, OUT / "fraud_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true": y_test.to_numpy(), "risk": p}).to_csv(OUT / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved in {OUT}")

if __name__ == "__main__": main()
