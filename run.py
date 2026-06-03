
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="CrowdSentinel Server")
parser.add_argument("--host",  default="0.0.0.0",  help="Bind host  (default: 0.0.0.0)")
parser.add_argument("--port",  default=5000, type=int, help="Bind port  (default: 5000)")
parser.add_argument("--debug", action="store_true",    help="Enable Flask debug mode")
args = parser.parse_args()

os.makedirs("database",      exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs("models",        exist_ok=True)

print("=" * 56)
print("  CrowdSentinel — AI Stampede Detection System")
print("=" * 56)
print(f"  Host  : http://{args.host}:{args.port}")
print(f"  Debug : {args.debug}")
print("=" * 56)
print()
print("  ->  Open your browser at:")
print(f"     http://127.0.0.1:{args.port}/login")
print()

model_path = os.path.join("models", "yolov8n.pt")
if not os.path.exists(model_path):
    print("[run] YOLOv8 model not found — downloading yolov8n.pt …")
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        
        if os.path.exists("yolov8n.pt"):
            os.rename("yolov8n.pt", model_path)
        print(f"[run] Model saved to {model_path}")
    except Exception as e:
        print(f"[run] WARNING: Could not download model — {e}")
        print("[run] Detection will run in demo mode (no boxes).")
else:
    print(f"[run] Model found: {model_path}")

print()

from app import app
from database.db_manager import DatabaseManager

db = DatabaseManager("database/crowd_logs.db")
db.init()

app.run(
    host=args.host,
    port=args.port,
    debug=args.debug,
    threaded=True,
    use_reloader=False,   
)
