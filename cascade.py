"""
Uncertainty-Aware Cascaded Ensemble — UNSW-NB15

Stage 1 : LightGBM binary screen  (normal vs suspicious)
Stage 2 : CatBoost multiclass     (attack category, suspicious traffic only)
Stage 3 : Confidence margin       M = p1 - p2; low margin => escalated for review
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# ---- constants ---------------------------------------------------------------
CAT_COLS     = ["proto", "service", "state"]
NON_FEATURES = ["id", "label", "attack_cat"]
MODEL_DIR    = Path(__file__).parent / "models"

# ---- data helpers ------------------------------------------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_dataset(path) -> pd.DataFrame:
    """Load a UNSW-NB15 CSV, clean it, and normalise the attack_cat label."""
    df = clean_columns(pd.read_csv(path))
    df = df.drop(columns=["id"], errors="ignore").drop_duplicates().reset_index(drop=True)
    if "attack_cat" in df.columns:
        df["attack_cat"] = (
            df["attack_cat"]
            .astype(str)
            .str.strip()
            .replace({"Backdoors": "Backdoor", "nan": "Normal"})
        )
        # Rows with label==0 should read "Normal" for attack_cat
        mask = df["label"] == 0
        df.loc[mask, "attack_cat"] = "Normal"
    return df


def feature_frame(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    Drop label columns, fix dtypes, and (optionally) reindex to a fixed
    feature list. Never leaks target information into X.
    """
    df = clean_columns(df)
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
    if features is not None:
        X = X.reindex(columns=features)
    # Categorical cols: fill NaN, force str
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].fillna("-").astype(str).str.strip()
    # Numeric cols: coerce, replace inf
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    X[num_cols] = (
        X[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return X


# ---- model class -------------------------------------------------------------

class Cascade:
    """
    Wrapper around the two-stage pipeline.

    Attributes
    ----------
    binary  : LGBMClassifier
    multi   : CatBoostClassifier
    encoder : OrdinalEncoder  (fitted on training categorical cols for LightGBM)
    meta    : dict  (feature list, class names, thresholds)
    """

    def __init__(self, binary=None, multi=None, encoder=None, meta=None):
        self.binary  = binary
        self.multi   = multi
        self.encoder = encoder
        self.meta    = meta or {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, folder=MODEL_DIR):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"binary": self.binary, "encoder": self.encoder},
                    folder / "stage1.joblib")
        self.multi.save_model(str(folder / "stage2.cbm"))
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2))
        print(f"[Cascade] models saved to {folder}/")

    @classmethod
    def load(cls, folder=MODEL_DIR):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(
                f"Model directory not found: {folder}\n"
                "Run  python train.py  first to train and save the models."
            )
        s1    = joblib.load(folder / "stage1.joblib")
        multi = CatBoostClassifier()
        multi.load_model(str(folder / "stage2.cbm"))
        meta  = json.loads((folder / "meta.json").read_text())
        return cls(s1["binary"], multi, s1["encoder"], meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _for_lgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for LightGBM (uses OrdinalEncoder)."""
        X2 = X.copy()
        present = [c for c in CAT_COLS if c in X2.columns]
        if present:
            X2[present] = self.encoder.transform(X2[present])
        return X2

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> list[dict]:
        """
        Run the full cascade on a feature DataFrame.

        Returns a list of dicts (one per row):
            p_malicious  : float – stage 1 malicious probability
            suspicious   : bool  – True if stage 1 threshold exceeded
            label        : str   – predicted class ("Normal" or attack name)
            top          : list  – [[name, prob], [name, prob]] top-2 (attacks only)
            margin       : float | None – p1 - p2 (attacks only)
            escalated    : bool  – True if margin < threshold
            probs        : dict  – {class: prob} for all attack classes
        """
        classes   = list(self.multi.classes_)
        t_bin     = self.meta["binary_threshold"]
        t_margin  = self.meta["margin_threshold"]

        # Stage 1
        p_mal     = self.binary.predict_proba(self._for_lgb(X))[:, 1]
        suspicious = p_mal >= t_bin

        # Stage 2 (only for suspicious rows)
        probs = np.zeros((len(X), len(classes)), dtype=float)
        if suspicious.any():
            probs[suspicious] = self.multi.predict_proba(X[suspicious])

        rows = []
        for i in range(len(X)):
            if not suspicious[i]:
                rows.append({
                    "p_malicious": float(p_mal[i]),
                    "suspicious":  False,
                    "label":       "Normal",
                    "top":         [],
                    "margin":      None,
                    "escalated":   False,
                    "probs":       {},
                })
                continue

            order = np.argsort(probs[i])[::-1]
            p1    = float(probs[i][order[0]])
            p2    = float(probs[i][order[1]])
            margin = p1 - p2
            rows.append({
                "p_malicious": float(p_mal[i]),
                "suspicious":  True,
                "label":       classes[order[0]],
                "top":         [[classes[order[0]], p1], [classes[order[1]], p2]],
                "margin":      margin,
                "escalated":   margin < t_margin,
                "probs":       {classes[j]: float(probs[i][j]) for j in order},
            })
        return rows
