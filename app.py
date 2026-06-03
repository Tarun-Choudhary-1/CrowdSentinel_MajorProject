
import os
import cv2
import time
import threading
import numpy as np
import gc

from flask import (Flask, render_template, Response, jsonify,
                   request, redirect, url_for, flash)
from flask_login import (LoginManager, login_user, logout_user,
                          login_required, current_user)
from werkzeug.utils import secure_filename

from auth.models import UserManager, User
from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.density import DensityEstimator
from core.optical_flow import OpticalFlowAnalyzer
from core.risk_engine import RiskEngine
from core.alerts import AlertSystem
from core.zone_visualizer import ZoneVisualizer
from utils.preprocessing import preprocess_frame
from utils.visualization import draw_detections, draw_hud
from utils.logger import CrowdLogger
from utils.fps import FPSCounter
from database.db_manager import DatabaseManager



app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "crowdsentinel-dev-secret-2024")

UPLOAD_FOLDER     = "static/outputs"
ALLOWED_EXTS      = {"mp4", "avi", "mov", "mkv", "webm"}
app.config["UPLOAD_FOLDER"]        = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]   = 500 * 1024 * 1024   

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database",    exist_ok=True)

login_manager = LoginManager(app)
login_manager.login_view     = "login"
login_manager.login_message  = "Please sign in to continue."

user_mgr = UserManager()

@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    return user_mgr.get_by_id(int(user_id))


detector   = PersonDetector(model_path="models/yolov8n.pt",
                             confidence=0.45, iou=0.45)
tracker    = PersonTracker(max_age=10, n_init=3)
density_e  = DensityEstimator(grid_rows=4, grid_cols=4,
                               critical_per_cell=8,
                               min_safe_distance=60.0)
of_anal    = OpticalFlowAnalyzer(winsize=13, levels=3)
risk_eng   = RiskEngine(alpha=0.45, beta=0.30, gamma=0.25)
alert_sys  = AlertSystem(sound_path="static/alerts/siren.wav")
zone_vis   = ZoneVisualizer(grid_rows=4, grid_cols=4, critical_per_cell=8)
fps_ctr    = FPSCounter(window=30)
logger     = CrowdLogger(log_dir=UPLOAD_FOLDER)
db         = DatabaseManager("database/crowd_logs.db")



_state = {
    "running":            False,
    "source":             None,
    "raw_frame":          None,  
    "zone_frame":         None,  
    "people_count":       0,
    "risk_level":         "LOW",
    "risk_score":         0.0,
    "density":            0.0,
    "avg_speed":          0.0,
    "direction_variance": 0.0,
    "fps":                0.0,
    "alert_active":       False,
    "incident_count":     0,
    "history":            [],
    "lock":               threading.Lock(),
}


DETECT_INTERVAL_WEBCAM = 2   
DETECT_INTERVAL_VIDEO  = 2   
MAX_PROCESS_FPS        = 15  
FRAME_WIDTH_WEBCAM     = 640
FRAME_WIDTH_VIDEO      = 720


def _capture_loop_safe(source):
    
    retries = 0
    while _state["running"] and retries < 3:
        try:
            capture_loop(source)
        except Exception as e:
            print(f"[CaptureLoop] CRASHED (retry {retries+1}/3): {e}")
            import traceback
            traceback.print_exc()
            retries += 1
            time.sleep(1.0)
            gc.collect() 
    _state["running"] = False
    print("[CaptureLoop] Stopped.")


def capture_loop(source):
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Capture] Cannot open: {source}")
        _state["running"] = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

    is_video_file = isinstance(source, str)
    detect_interval = DETECT_INTERVAL_VIDEO if is_video_file else DETECT_INTERVAL_WEBCAM
    frame_width = FRAME_WIDTH_VIDEO if is_video_file else FRAME_WIDTH_WEBCAM
    min_frame_time = 1.0 / MAX_PROCESS_FPS

    prev_gray = None
    frame_counter = 0
    log_ticker = 0
    last_tracks = []
    last_centroids = []

    print(f"[Capture] Started: {'video' if is_video_file else 'webcam'}, "
          f"width={frame_width}, detect_every={detect_interval}")

    while _state["running"]:
        loop_start = time.perf_counter()

        ret, raw = cap.read()
        if not ret:
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_gray = None
                of_anal.reset()
                time.sleep(0.1)
                continue
            break

        fps_ctr.tick()
        frame = preprocess_frame(raw, target_width=frame_width)
        h, w  = frame.shape[:2]
        frame_counter += 1

        
        if frame_counter % detect_interval == 0:
            detections = detector.detect(frame, tiled=is_video_file)
            tracks     = tracker.update(detections, frame)
            last_tracks = tracks
        else:
            
            tracks = last_tracks

        count      = len(tracks)
        centroids  = [t["centroid"] for t in tracks]
        
        density_score, grid_counts = density_e.compute(centroids, w, h)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_speed, dir_var = 0.0, 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            avg_speed, dir_var = of_anal.analyze(prev_gray, gray)
        prev_gray = gray

        risk_score, risk_level = risk_eng.compute(
            count=count,
            density=density_score,
            speed=avg_speed,
            direction_variance=dir_var,
        )

        alert_active = alert_sys.check_and_trigger(risk_level, risk_score)

        vis_frame  = draw_detections(frame.copy(), tracks, risk_level)
        vis_frame  = draw_hud(vis_frame, count, risk_level,
                               risk_score, fps_ctr.get(), alert_active)

        zone_frame = zone_vis.render(frame.copy(), centroids, grid_counts)
        zone_frame = draw_hud(zone_frame, count, risk_level,
                               risk_score, fps_ctr.get(), alert_active)

        if risk_level == "CRITICAL" and frame_counter % 60 == 0:
            snap = f"{UPLOAD_FOLDER}/incident_{int(time.time())}.jpg"
            cv2.imwrite(snap, vis_frame)

        log_ticker += 1
        if log_ticker % 60 == 0:
            logger.log(count, risk_score, risk_level, density_score, avg_speed)
            db.insert(count, risk_score, risk_level, density_score, avg_speed)

        new_state = {
            "raw_frame":          vis_frame,
            "zone_frame":         zone_frame,
            "people_count":       count,
            "risk_level":         risk_level,
            "risk_score":         round(risk_score, 3),
            "density":            round(density_score, 3),
            "avg_speed":          round(avg_speed, 3),
            "direction_variance": round(dir_var, 3),
            "fps":                round(fps_ctr.get(), 1),
            "alert_active":       alert_active,
        }
        with _state["lock"]:
            _state.update(new_state)
            if risk_level in ("HIGH", "CRITICAL"):
                _state["incident_count"] += 1
            _state["history"].append({
                "t":     time.time(),
                "count": count,
                "risk":  risk_score,
            })
            
            if len(_state["history"]) > 120:
                _state["history"] = _state["history"][-120:]

        if frame_counter % 120 == 0:
            gc.collect()

        elapsed = time.perf_counter() - loop_start
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)

    cap.release()
    alert_sys.stop()
    of_anal.reset()
    _state["running"] = False


def _mjpeg_gen(key: str, placeholder_text: str):
    while True:
        try:
            with _state["lock"]:
                frame = _state.get(key)

            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, placeholder_text,
                            (80, 190), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (80, 80, 80), 1, cv2.LINE_AA)
                frame = blank

            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        except Exception as e:
            print(f"[MJPEG] stream error: {e}")
        time.sleep(0.066)  


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password   = request.form.get("password",   "")
        remember   = bool(request.form.get("remember"))

        user = user_mgr.authenticate(identifier, password)
        if user:
            login_user(user, remember=remember)
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        flash("Invalid username / email or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm",  "")

        if password != confirm:
            flash("Passwords do not match.", "error")
        else:
            ok, msg = user_mgr.register(username, email, password)
            if ok:
                flash("Account created! Please sign in.", "success")
                return redirect(url_for("login"))
            flash(msg, "error")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part"})
        f   = request.files["file"]
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXTS:
            return jsonify({"success": False, "error": "Unsupported file type"})
        fname = secure_filename(f.filename)
        path  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(path)
        return jsonify({"success": True, "path": path})
    return render_template("upload.html", user=current_user)


@app.route("/api/start", methods=["POST"])
@login_required
def api_start():
    if _state["running"]:
        return jsonify({"status": "already_running"})

    data   = request.get_json(silent=True) or {}
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)


    risk_eng.reset()
    of_anal.reset()
    fps_ctr._ts.clear()
    gc.collect()

    with _state["lock"]:
        _state["running"]        = True
        _state["source"]         = source
        _state["raw_frame"]      = None
        _state["zone_frame"]     = None
        _state["people_count"]   = 0
        _state["risk_level"]     = "LOW"
        _state["risk_score"]     = 0.0
        _state["density"]        = 0.0
        _state["avg_speed"]      = 0.0
        _state["incident_count"] = 0
        _state["history"]        = []

    threading.Thread(target=_capture_loop_safe, args=(source,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
@login_required
def api_stop():
    _state["running"] = False
    alert_sys.stop()
    gc.collect()
    return jsonify({"status": "stopped"})


@app.route("/api/status")
@login_required
def api_status():
    with _state["lock"]:
        return jsonify({
            "running":            _state["running"],
            "people_count":       _state["people_count"],
            "risk_level":         _state["risk_level"],
            "risk_score":         _state["risk_score"],
            "density":            _state["density"],
            "avg_speed":          _state["avg_speed"],
            "direction_variance": _state["direction_variance"],
            "fps":                _state["fps"],
            "alert_active":       _state["alert_active"],
            "incident_count":     _state["incident_count"],
        })


@app.route("/api/history")
@login_required
def api_history():
    with _state["lock"]:
        return jsonify(_state["history"][-60:])


@app.route("/api/logs")
@login_required
def api_logs():
    return jsonify(db.fetch_recent(40))


@app.route("/api/clear_logs", methods=["POST"])
@login_required
def api_clear_logs():
    db.clear()
    return jsonify({"status": "cleared"})


@app.route("/stream/detection")
@login_required
def stream_detection():
    return Response(
        _mjpeg_gen("raw_frame", "Detection feed -- press Start"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stream/zones")
@login_required
def stream_zones():
    return Response(
        _mjpeg_gen("zone_frame", "Risk zones -- press Start"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    db.init()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
