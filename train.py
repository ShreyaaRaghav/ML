"""
Train the Uncertainty-Aware Cascaded Ensemble on UNSW-NB15.

Expected layout
---------------
data/
    UNSW_NB15_training-set.csv
    UNSW_NB15_testing-set.csv

Run
---
    python train.py

Outputs
-------
models/stage1.joblib   — LightGBM binary screener + OrdinalEncoder
models/stage2.cbm      — CatBoost multiclass classifier
models/meta.json       — feature list, class names, thresholds
models/metrics.json    — full evaluation metrics (used by /api/metrics)
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from cascade import CAT_COLS, MODEL_DIR, Cascade, feature_frame, load_dataset

# ── config ────────────────────────────────────────────────────────────────────
DATA          = Path(__file__).parent / "data"
SEED          = 42
TARGET_RECALL = 0.98   # stage 1: keep at least this share of real attacks
TARGET_ACC    = 0.90   # stage 3: accuracy wanted on non-escalated predictions
MAX_ESCALATE  = 0.35   # stage 3: don't flag more than this share for review
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading datasets …")

TRAIN_CSV = DATA / "UNSW_NB15_training-set.csv"
TEST_CSV  = DATA / "UNSW_NB15_testing-set.csv"

# Fallback: if the training CSV is missing, split the testing CSV 80/20
if not TRAIN_CSV.exists():
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            "No dataset found. Place UNSW_NB15_training-set.csv (and optionally "
            "UNSW_NB15_testing-set.csv) in the data/ folder.\n"
            "Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        )
    print(
        "  WARNING: UNSW_NB15_training-set.csv not found.\n"
        "  Falling back: splitting UNSW_NB15_testing-set.csv 80/20 into train/test.\n"
        "  For best results, download the real training set."
    )
    full_df  = load_dataset(TEST_CSV)
    from sklearn.model_selection import train_test_split as _tts
    train_df, test_df = _tts(
        full_df, test_size=0.2,
        stratify=full_df["attack_cat"], random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
else:
    train_df = load_dataset(TRAIN_CSV)
    test_df  = load_dataset(TEST_CSV)   # NEVER used for tuning

print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"  Train label dist:\n{train_df['attack_cat'].value_counts().to_string()}\n")

# ── 1. Train / validation split (test_df untouched until final eval) ──────────
tr, va = train_test_split(
    train_df, test_size=0.2,
    stratify=train_df["attack_cat"], random_state=SEED
)

features = list(feature_frame(tr).columns)
Xtr = feature_frame(tr,       features)
Xva = feature_frame(va,       features)
Xte = feature_frame(test_df,  features)

ytr_bin = tr["label"].values
yva_bin = va["label"].values

print(f"Features: {len(features)}")

# ── 2. Stage 1 — LightGBM binary screener ────────────────────────────────────
print("\n[Stage 1] Training LightGBM binary classifier …")

enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
).fit(Xtr[CAT_COLS])


def for_lgb(X: "pd.DataFrame"):
    X2 = X.copy()
    X2[CAT_COLS] = enc.transform(X[CAT_COLS])
    return X2


binary = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1,
)
binary.fit(
    for_lgb(Xtr), ytr_bin,
    eval_set=[(for_lgb(Xva), yva_bin)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Select binary threshold: highest t that keeps attack recall >= TARGET_RECALL
p_va_bin = binary.predict_proba(for_lgb(Xva))[:, 1]
t_bin = 0.5
for t in np.linspace(0.95, 0.05, 91):
    if (p_va_bin[yva_bin == 1] >= t).mean() >= TARGET_RECALL:
        t_bin = float(t)
        break
print(f"  Binary threshold selected: {t_bin:.3f}")

# ── 3. Stage 2 — CatBoost multiclass (attacks only) ──────────────────────────
print("\n[Stage 2] Training CatBoost multiclass classifier …")

tr_a = tr[tr["label"] == 1]
va_a = va[va["label"] == 1]

multi = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.08,
    depth=6,
    loss_function="MultiClass",
    auto_class_weights="SqrtBalanced",
    random_seed=SEED,
    verbose=200,
    cat_features=CAT_COLS,
)
multi.fit(
    feature_frame(tr_a, features), tr_a["attack_cat"],
    eval_set=(feature_frame(va_a, features), va_a["attack_cat"]),
    early_stopping_rounds=100,
)

# ── 4. Stage 3 — Margin threshold ────────────────────────────────────────────
print("\n[Stage 3] Selecting confidence-margin threshold on validation data …")

pv      = multi.predict_proba(feature_frame(va_a, features))
cls     = np.array(multi.classes_)
order_v = np.argsort(pv, axis=1)[:, ::-1]
margin_va  = pv[np.arange(len(pv)), order_v[:, 0]] - pv[np.arange(len(pv)), order_v[:, 1]]
correct_va = cls[order_v[:, 0]] == va_a["attack_cat"].values

t_margin = 0.2
for t in np.linspace(0.0, 0.9, 91):
    keep = margin_va >= t
    if keep.mean() < 1 - MAX_ESCALATE:
        break
    if correct_va[keep].mean() >= TARGET_ACC:
        t_margin = float(t)
        break
print(f"  Margin threshold selected: {t_margin:.3f}")

# ── 5. Assemble and save the cascade ─────────────────────────────────────────
meta  = {
    "features":          features,
    "classes":           list(cls),
    "binary_threshold":  t_bin,
    "margin_threshold":  t_margin,
}
model = Cascade(binary, multi, enc, meta)
model.save()

# ── 6. Final evaluation on the untouched test set ────────────────────────────
print("\n[Eval] Running inference on test set …")

t_start  = time.perf_counter()
preds    = model.predict(Xte)
infer_ms = (time.perf_counter() - t_start) / len(Xte) * 1_000_000   # ms per 1 k

y_true  = test_df["attack_cat"].values
y_pred  = np.array([p["label"]     for p in preds])
y_bin   = test_df["label"].values
p_bin   = np.array([p["suspicious"] for p in preds], dtype=int)
flagged = np.array([p["escalated"]  for p in preds])

tp = int(((p_bin == 1) & (y_bin == 1)).sum())
fp = int(((p_bin == 1) & (y_bin == 0)).sum())
fn = int(((p_bin == 0) & (y_bin == 1)).sum())
tn = int(((p_bin == 0) & (y_bin == 0)).sum())

conf   = (~flagged) & (p_bin == 1)
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

# ── 7. Baseline — Random Forest ───────────────────────────────────────────────
print("\n[Baseline] Training Random Forest …")
Xtr_rf = for_lgb(Xtr).fillna(-1)
Xte_rf = for_lgb(Xte).fillna(-1)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
rf.fit(Xtr_rf, tr["attack_cat"])
rf_pred = rf.predict(Xte_rf)

# ── 8. Write metrics ──────────────────────────────────────────────────────────
n_suspicious = max(int(p_bin.sum()), 1)

metrics = {
    "thresholds": {
        "binary": t_bin,
        "margin": t_margin,
    },
    "test_records":       int(len(Xte)),
    "accuracy":           float(accuracy_score(y_true, y_pred)),
    "macro_f1":           float(f1_score(y_true, y_pred, average="macro")),
    "binary_precision":   tp / max(tp + fp, 1),
    "binary_recall":      tp / max(tp + fn, 1),
    "false_positive_rate": fp / max(fp + tn, 1),
    "escalation_rate":    float(flagged.sum() / n_suspicious),
    "confident_accuracy": float((y_pred[conf] == y_true[conf]).mean()) if conf.any() else None,
    "inference_ms_per_1000": float(infer_ms),
    "per_class": {
        k: {
            "precision": float(v["precision"]),
            "recall":    float(v["recall"]),
            "f1":        float(v["f1-score"]),
            "support":   int(v["support"]),
        }
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    },
    "baseline": {
        "name":     "Random Forest",
        "accuracy": float(accuracy_score(y_true, rf_pred)),
        "macro_f1": float(f1_score(y_true, rf_pred, average="macro")),
    },
}

(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

# ── 9. Summary print ─────────────────────────────────────────────────────────
summary = {k: v for k, v in metrics.items() if k != "per_class"}
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(json.dumps(summary, indent=2, default=float))
print("\nPer-class:")
for name, vals in sorted(metrics["per_class"].items(), key=lambda x: -x[1]["support"]):
    print(f"  {name:<20} P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
          f"F1={vals['f1']:.3f}  n={vals['support']}")
print("\nDone. Metrics saved to models/metrics.json")
