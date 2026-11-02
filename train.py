"""Train and benchmark the XGBoost fraud classifier.

Reports: dataset size, fraud ratio, ROC-AUC, PR-AUC, precision, recall, false-positive rate,
and single-transaction scoring latency (model-only, in ms).

Usage: python train.py [--data path/to/transactions.csv]
"""
import argparse, json, time
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from data import load_transactions
from features import FEATURES, build_features


def pick_threshold(y, p, min_precision=0.80):
    """Highest-recall threshold whose precision stays >= min_precision (controls false positives)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= min_precision)[0]
    return float(thr[ok[np.argmax(rec[ok])]]) if len(ok) else 0.5


def main(args):
    df = build_features(load_transactions(args.data))
    n, fraud_ratio = len(df), df["is_fraud"].mean()
    print(f"Transactions: {n:,} | fraud ratio: {fraud_ratio:.4%}")

    # Time-ordered split: 70% train / 15% validation (threshold) / 15% test
    i1, i2 = int(n * 0.70), int(n * 0.85)
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    X_tr, y_tr = tr[FEATURES], tr["is_fraud"]

    # SMOTE only on the training fold
    sm = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=42)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f"After SMOTE: {len(X_res):,} rows, positive share {y_res.mean():.2%}")

    model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="aucpr", n_jobs=-1, random_state=42)
    model.fit(X_res, y_res, eval_set=[(va[FEATURES], va["is_fraud"])], verbose=False)

    thr = pick_threshold(va["is_fraud"], model.predict_proba(va[FEATURES])[:, 1])
    p = model.predict_proba(te[FEATURES])[:, 1]
    y = te["is_fraud"].values
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    # Latency: single-row scoring
    row = te[FEATURES].iloc[[0]]
    model.predict_proba(row)
    t = []
    for _ in range(300):
        s = time.perf_counter(); model.predict_proba(row); t.append((time.perf_counter() - s) * 1000)

    metrics = {
        "n_transactions": int(n), "fraud_ratio": float(fraud_ratio), "test_size": int(len(te)),
        "threshold": thr, "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "latency_ms_p50": float(np.percentile(t, 50)), "latency_ms_p95": float(np.percentile(t, 95)),
    }
    print(json.dumps(metrics, indent=2))
    joblib.dump({"model": model, "features": FEATURES, "threshold": thr}, "fraud_model.joblib")
    json.dump(metrics, open("metrics.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
    main(ap.parse_args())
