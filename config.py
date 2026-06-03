
import os


# Flask
SECRET_KEY = os.environ.get("SECRET_KEY", "crowdsentinel-change-me-in-production")
DEBUG      = os.environ.get("DEBUG", "false").lower() == "true"
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", 5000))


# Paths
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "yolov8n.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
ALERT_SOUND   = os.path.join(BASE_DIR, "static", "alerts", "siren.wav")
DB_EVENTS     = os.path.join(BASE_DIR, "database", "crowd_logs.db")
DB_USERS      = os.path.join(BASE_DIR, "database", "users.db")


# Detection
YOLO_CONFIDENCE  = 0.45    # detection confidence threshold
YOLO_IOU         = 0.45    # NMS IoU threshold
YOLO_DEVICE      = "cpu"   # "cpu" or "0" (GPU index)
YOLO_IMG_SIZE    = 640     # inference resolution
FRAME_WIDTH      = 960     # frame width after preprocessing


# Tracking
TRACKER_MAX_AGE  = 10      # max frames to keep a lost track
TRACKER_N_INIT   = 2       # frames before track is confirmed

# Density Estimation
GRID_ROWS           = 4
GRID_COLS           = 4
CRITICAL_PER_CELL   = 8    # persons per cell = fully saturated
MIN_SAFE_DISTANCE   = 60   # pixels; below this = crowded


# Risk Engine 
RISK_ALPHA = 0.45   # density weight
RISK_BETA  = 0.30   # speed weight
RISK_GAMMA = 0.25   # direction-variance weight

# Count gates — minimum people required for each risk tier
COUNT_FOR_MEDIUM   = 4     # fewer than this → always LOW
COUNT_FOR_HIGH     = 11    # fewer than this → max MEDIUM
COUNT_FOR_CRITICAL = 26    # fewer than this → max HIGH


# Optical Flow
OF_WIN_SIZE      = 13
OF_LEVELS        = 3
OF_MAG_THRESHOLD = 0.8     # min pixel movement to include in flow analysis


# Alert System
ALERT_COOLDOWN_SECS = 10   # minimum seconds between audio alerts


# MJPEG Stream
STREAM_FPS    = 30         # max stream frame rate
JPEG_QUALITY  = 82         # JPEG compression quality (1–100)


# Logging
LOG_INTERVAL_SEC = 2.0     # throttle: seconds between CSV/DB writes
