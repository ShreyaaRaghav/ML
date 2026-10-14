"""
Cyber Attack Detection — REST API server
Run:  python app.py  →  http://127.0.0.1:5000

Endpoints
---------
GET  /                     — Single-page web application
GET  /api/sample           — Random record from the test CSV
POST /api/predict          — Predict a single JSON record
POST /api/batch            — Predict an uploaded UNSW-NB15 CSV (≤ 20 000 rows)
GET  /api/metrics          — Model evaluation metrics from train.py
GET  /api/status           — Health check (models loaded, test CSV present)
GET  /api/features         — List of feature names expected by the model
"""
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from cascade import MODEL_DIR, NON_FEATURES, Cascade, clean_columns, feature_frame

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Flask setup ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
app  = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

BATCH_LIMIT = 20_000
TABLE_ROWS  = 300

# ── load model ────────────────────────────────────────────────────────────────
try:
    model    = Cascade.load()
    FEATURES = model.meta["features"]
    log.info("Model loaded. Features: %d | Classes: %s",
             len(FEATURES), model.meta.get("classes"))
except FileNotFoundError as exc:
    log.warning("Models not found — run train.py first.\n%s", exc)
    model    = None
    FEATURES = []

# ── test CSV (optional, used by /api/sample) ──────────────────────────────────
TEST_CSV = BASE / "data" / "UNSW_NB15_testing-set.csv"
try:
    test_df = clean_columns(pd.read_csv(TEST_CSV)) if TEST_CSV.exists() else None
    if test_df is not None:
        log.info("Test CSV loaded: %d rows.", len(test_df))
except Exception as exc:
    log.warning("Could not read test CSV: %s", exc)
    test_df = None


# ── helpers ───────────────────────────────────────────────────────────────────
def _model_required():
    if model is None:
        return jsonify(error="Models are not loaded. Run  python train.py  first."), 503
    return None


def _serialise(obj):
    """JSON serialiser that handles numpy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    """Serve the single-page application."""
    return send_from_directory(str(BASE / "static"), "index.html")


# ---------- health / introspection -------------------------------------------

@app.get("/api/status")
def status():
    return jsonify(
        model_loaded  = model is not None,
        test_csv      = test_df is not None,
        features      = len(FEATURES),
        classes       = model.meta.get("classes") if model else [],
        thresholds    = {
            "binary": model.meta.get("binary_threshold"),
            "margin": model.meta.get("margin_threshold"),
        } if model else {},
    )


@app.get("/api/features")
def features_list():
    err = _model_required()
    if err:
        return err
    return jsonify(features=FEATURES, count=len(FEATURES))


# ---------- sample record from test CSV --------------------------------------

@app.get("/api/sample")
def sample():
    err = _model_required()
    if err:
        return err
    if test_df is None:
        return jsonify(
            error="Test CSV not found. Place UNSW_NB15_testing-set.csv in data/."
        ), 404
    row    = test_df.sample(1).iloc[0]
    record = {
        k: v for k, v in json.loads(row.to_json()).items()
        if k not in NON_FEATURES
    }
    actual = str(row.get("attack_cat", "")).strip()
    return jsonify(record=record, actual=actual)


# ---------- single-record prediction -----------------------------------------

@app.post("/api/predict")
def predict():
    """
    Body: one traffic record as a flat JSON object.

    Response fields
    ---------------
    label          : predicted class
    suspicious     : bool
    p_malicious    : probability of malicious traffic (stage 1)
    margin         : confidence margin p1 - p2  (null for Normal)
    escalated      : bool – low-confidence flag
    top            : [[name, prob], [name, prob]]
    probs          : {class: prob}
    missing_features: list of features not supplied (model uses NaN for these)
    thresholds     : {binary, margin}
    """
    err = _model_required()
    if err:
        return err

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send one traffic record as a JSON object."), 400

    df = pd.DataFrame([{k.strip().lower(): v for k, v in body.items()}])
    missing = [f for f in FEATURES if f not in df.columns]

    # Reject if more than half the features are absent (likely wrong payload)
    if len(missing) > len(FEATURES) // 2:
        return jsonify(
            error=f"Record is missing {len(missing)} of {len(FEATURES)} features. "
                  "Use /api/sample to see the expected format."
        ), 400

    t0     = time.perf_counter()
    result = model.predict(feature_frame(df, FEATURES))[0]
    ms     = (time.perf_counter() - t0) * 1000

    result["missing_features"] = missing
    result["inference_ms"]     = round(ms, 3)
    result["thresholds"]       = {
        "binary": model.meta["binary_threshold"],
        "margin": model.meta["margin_threshold"],
    }
    return jsonify(result)


# ---------- batch prediction from CSV upload ---------------------------------

@app.post("/api/batch")
def batch():
    """
    Form-data: field name 'file', CSV in UNSW-NB15 format.
    First 20 000 rows are processed.

    Response
    --------
    total      : records processed
    suspicious : records flagged by stage 1
    escalated  : records flagged for review in stage 3
    counts     : {label: count}
    rows       : list of per-record summaries (first TABLE_ROWS only)
    truncated  : bool
    """
    err = _model_required()
    if err:
        return err

    file = request.files.get("file")
    if file is None:
        return jsonify(error="No file received. POST a CSV as form-data field 'file'."), 400

    try:
        df = clean_columns(pd.read_csv(io.BytesIO(file.read())))
    except Exception as exc:
        return jsonify(error=f"Could not parse CSV: {exc}"), 400

    if "proto" not in df.columns:
        return jsonify(
            error="File does not look like UNSW-NB15 data ('proto' column absent)."
        ), 400

    df      = df.head(BATCH_LIMIT).reset_index(drop=True)
    t0      = time.perf_counter()
    results = model.predict(feature_frame(df, FEATURES))
    ms      = (time.perf_counter() - t0) * 1000

    n          = len(results)
    suspicious = sum(r["suspicious"] for r in results)
    escalated  = sum(r["escalated"]  for r in results)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    rows = []
    for i, r in enumerate(results[:TABLE_ROWS]):
        rows.append({
            "row":         i + 1,
            "label":       r["label"],
            "p_malicious": round(r["p_malicious"], 4),
            "margin":      round(r["margin"], 4) if r["margin"] is not None else None,
            "escalated":   r["escalated"],
            "actual": (
                str(df.loc[i, "attack_cat"]).strip()
                if "attack_cat" in df.columns else None
            ),
        })

    return jsonify(
        total        = n,
        suspicious   = suspicious,
        escalated    = escalated,
        counts       = counts,
        rows         = rows,
        truncated    = n > TABLE_ROWS,
        inference_ms = round(ms, 1),
    )


# ---------- model metrics -----------------------------------------------------

@app.get("/api/metrics")
def metrics():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return jsonify(
            error="Metrics file not found. Run  python train.py  to generate it."
        ), 404
    data = json.loads(path.read_text())
    return jsonify(data)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info("Starting server on http://127.0.0.1:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
