# CrowdSentinel 
## AI-Based Real-Time Stampede Detection & Crowd Risk Analysis

> Final-Year Major Project · Computer Vision · Python · Flask · Auth-Protected

---

## What's in this?

| Area | Highlight |
|---|---|
| **Risk Logic** | Count-gated thresholds — 1–3 people always LOW regardless of density |
| **UI** | Clean modern white UI, no separate dashboard page |
| **Auth** | Login + Register with Werkzeug password hashing |
| **Detection** | CLAHE low-light enhancement, aspect-ratio box filter |
| **Visualization** | Gaussian heatmap replaced with colour-coded risk zone overlay |
| **Performance** | Reduced webcam buffer lag, EMA-smoothed risk score |
| **Config** | Single `config.py` for all tunable parameters |

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python run.py

# 4. Open browser
#    http://127.0.0.1:5000/login
#    Register an account, then sign in
```

The YOLOv8 model (`yolov8n.pt`, ~6 MB) downloads automatically on first run.

---

## Pages

| URL | Description |
|---|---|
| `/login` | Sign in page |
| `/register` | Create account |
| `/` | Home — Live Feed + Upload shortcuts |
| `/upload` | Video file upload + analysis |
| `/logout` | End session |


---

## Risk Thresholds 

| People detected | Maximum risk level possible |
|---|---|
| 1 – 3 | **LOW** (always) |
| 4 – 10 | Up to **MEDIUM** |
| 11 – 25 | Up to **HIGH** |
| 26 + | Up to **CRITICAL** |

Risk is calculated as:

```
RiskScore = α·Density + β·NormSpeed + γ·DirectionVariance
          = 0.45·Density + 0.30·Speed + 0.25·Variance
```

Then capped by the count gate above and smoothed with EMA.

---

## Project Structure

```
Stampede-Detection-System/
├── run.py                  ← Starts here
├── app.py                  ← Flask routes + capture thread
├── config.py               ← All tunable parameters
├── requirements.txt
├── .env.example
│
├── auth/
│   ├── __init__.py
│   └── models.py           ← UserManager (SQLite + Werkzeug hashing)
│
├── core/
│   ├── detector.py         ← YOLOv8n + CLAHE + aspect-ratio filter
│   ├── tracker.py          ← DeepSORT (centroid fallback)
│   ├── density.py          ← Grid density + inter-person distance
│   ├── optical_flow.py     ← Farneback flow → speed + direction variance
│   ├── risk_engine.py      ← Count-gated hybrid risk formula
│   ├── alerts.py           ← pygame audio with cooldown
│   └── zone_visualizer.py  ← Colour-coded risk zone overlay
│
├── utils/
│   ├── preprocessing.py    ← Frame resize
│   ├── visualization.py    ← Bounding boxes, HUD bar
│   ├── logger.py           ← CSV event logger
│   └── fps.py              ← Rolling FPS counter
│
├── database/
│   ├── db_manager.py       ← SQLite crowd events
│   ├── crowd_logs.db       ← Auto-created
│   └── users.db            ← Auto-created by auth
│
├── static/
│   ├── css/style.css       ← UI
│   ├── js/main.js          ← Charts + polling + webcam controls
│   ├── alerts/siren.wav    ← Alarm WAV
│   └── outputs/            ← Snapshots + CSV logs
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── index.html          ← Home (Live Feed + Upload cards)
│   └── upload.html
│
├── models/
│   └── yolov8n.pt          ← Auto-downloaded on first run
│
├── notebooks/
│   └── experiments.ipynb   ← Algorithm testing
│
└── datasets/               ← Test videos here
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/start` | ✓ | Start detection `{"source": 0}` |
| POST | `/api/stop` | ✓ | Stop detection |
| GET | `/api/status` | ✓ | Live metrics JSON |
| GET | `/api/history` | ✓ | Last 60 data points |
| GET | `/api/logs` | ✓ | Last 40 DB records |
| POST | `/api/clear_logs` | ✓ | Delete all DB records |
| GET | `/stream/detection` | ✓ | MJPEG detection feed |
| GET | `/stream/zones` | ✓ | MJPEG risk-zone feed |

---

## Tuning

Edit `config.py` to adjust:

- `COUNT_FOR_MEDIUM / HIGH / CRITICAL` — risk sensitivity
- `YOLO_CONFIDENCE` — detection threshold
- `MIN_SAFE_DISTANCE` — crowding pixel distance
- `RISK_ALPHA / BETA / GAMMA` — formula weights

---

## Conclusion

"Unlike basic crowd counters, CrowdSentinel combines density estimation, inter-person distance analysis, optical flow motion analysis, and a count-gated weighted formula to predict stampede risk — with a practical authentication-protected web interface deployable locally without any cloud infrastructure."
