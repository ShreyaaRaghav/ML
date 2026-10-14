# Cyber Attack Detection — Uncertainty-Aware Cascaded Ensemble

Network intrusion detection system built on the **UNSW-NB15** dataset.  
Three-stage pipeline: binary screening → attack classification → confidence escalation.

---

## Quick start

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place training CSV in data/
#    data/UNSW_NB15_training-set.csv   ← download from UNSW-NB15 site
#    data/UNSW_NB15_testing-set.csv    ← already copied here

# 3. Train the models (writes models/)
python train.py

# 4. Start the web server
python app.py

# 5. Open browser
#    http://127.0.0.1:5000
```

---

## Project layout

```
ML/
├── app.py              REST API + Flask server
├── cascade.py          Cascade model class (Stage 1–3 inference)
├── train.py            Training script — writes models/ and metrics.json
├── requirements.txt
├── data/
│   ├── UNSW_NB15_testing-set.csv
│   └── UNSW_NB15_training-set.csv   ← you add this
├── models/             Created by train.py
│   ├── stage1.joblib
│   ├── stage2.cbm
│   ├── meta.json
│   └── metrics.json
└── static/             Single-page web application
    ├── index.html
    ├── style.css
    └── app.js
```

---

## REST API

| Method | Path            | Description                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | `/`             | Web application (single-page UI)                   |
| GET    | `/api/status`   | Health check — model loaded, test CSV present      |
| GET    | `/api/features` | Feature names expected by the model                |
| GET    | `/api/sample`   | Random record from the test set                    |
| POST   | `/api/predict`  | Predict one traffic record (JSON body)             |
| POST   | `/api/batch`    | Predict an uploaded UNSW-NB15 CSV (≤ 20 000 rows) |
| GET    | `/api/metrics`  | Evaluation metrics from `train.py`                 |

### Single prediction

```
POST /api/predict
Content-Type: application/json

{ "dur": 0.12, "proto": "tcp", "service": "http", "state": "FIN", ... }
```

Response:
```json
{
  "label": "Exploits",
  "suspicious": true,
  "p_malicious": 0.9812,
  "margin": 0.4321,
  "escalated": false,
  "top": [["Exploits", 0.6543], ["Generic", 0.2222]],
  "probs": { "Exploits": 0.6543, "Generic": 0.2222, "DoS": 0.05, ... },
  "thresholds": { "binary": 0.45, "margin": 0.20 },
  "missing_features": [],
  "inference_ms": 2.1
}
```

### Batch upload

```
POST /api/batch
Content-Type: multipart/form-data

file=<CSV in UNSW-NB15 format>
```

---

## Methodology

1. **Binary screening (Stage 1)** — LightGBM classifies each record as Normal or Suspicious.  
   Threshold selected to keep attack recall ≥ 98 %.

2. **Multiclass classification (Stage 2)** — CatBoost assigns one of nine attack categories  
   to suspicious records only (DoS, Exploits, Fuzzers, Generic, Reconnaissance, Backdoor,  
   Analysis, Shellcode, Worms).

3. **Confidence escalation (Stage 3)** — Margin M = p₁ − p₂.  
   When M falls below a validation-selected threshold the prediction is flagged for review  
   rather than treated as a confident automated decision.

Baseline comparison: Random Forest trained on the same features, full 10-class prediction.
