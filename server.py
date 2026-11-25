import sys
import threading
import time
from pathlib import Path

import cv2
import torch
from flask import Flask, Response, jsonify, request, render_template

ROOT = Path(__file__).resolve().parent
SIGN_DIR = ROOT / "sign_to_text"
sys.path.insert(0, str(SIGN_DIR))

import avatar as av              # noqa: E402
import extract_landmarks as el   # noqa: E402
import live_recognition as lr    # noqa: E402

app = Flask(__name__)

STATE = {
    "ready": False,
    "error": None,
    "mirror": True,
    "min_conf": 0.45,
    "use_profile": True,
    "has_profile": False,
    "avatar_text": "",
    "avatar_names": None,
    "avatar_play": None,
    "avatar_msg": "",
    "rec": None,
    "cap": None,
    "lib": None,
    "known": [],
    "rest": None,
    "latest_cam": None,
    "lock": threading.Lock(),
    "running": True,
}


def start_avatar_names(names, unknown=()):
    lib = STATE["lib"]
    if not names:
        STATE["avatar_msg"] = "No known word. Try: " + ", ".join(STATE["known"][:6])
        return
    E, H_, who = av.sentence_motion(lib, names)
    STATE["avatar_play"] = {
        "E": E, "H": H_, "who": who, "names": names, "t0": time.time()
    }
    STATE["avatar_names"] = names
    STATE["avatar_msg"] = ("skipped: " + ", ".join(unknown)) if unknown else ""


def _load_user_profile_if_available():
    prof_path = el.MODELS_DIR / "profile.npz"
    if prof_path.exists():
        return lr.load_profile(prof_path), True
    return None, False


def init_backend(source=0, backend="dshow", min_conf=0.45, use_profile=True):
    try:
        torch.set_num_threads(1)
        lib = av.load_library()
        known = sorted(m["name"] for m in lib["signs"].values())
        rest = av.rest_pose(lib)

        hand_model = el.ensure_model("hand_landmarker.task")
        pose_model = el.ensure_model("pose_landmarker_lite.task")
        ckpt = el.MODELS_DIR / "best.pt"
        model, classes, delta, _ = lr.load_model(ckpt)

        loaded_prof, has_prof = _load_user_profile_if_available()
        active_prof = loaded_prof if (use_profile and has_prof) else None

        rec = lr.Recognizer(
            model, classes, delta, hand_model, pose_model,
            calib_s=2.0, min_conf=min_conf, profile=active_prof
        )
        cap = lr.open_camera(int(source), backend)

        with STATE["lock"]:
            STATE["lib"] = lib
            STATE["known"] = known
            STATE["rest"] = rest
            STATE["rec"] = rec
            STATE["cap"] = cap
            STATE["min_conf"] = float(min_conf)
            STATE["has_profile"] = has_prof
            STATE["use_profile"] = bool(use_profile and has_prof)
            STATE["ready"] = True
            STATE["error"] = None

        def worker():
            t0 = time.monotonic()
            while STATE["running"]:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                t = time.monotonic() - t0
                try:
                    rec.step(frame, t)
                except Exception:
                    pass
                with STATE["lock"]:
                    STATE["latest_cam"] = frame.copy()
                time.sleep(0.01)

        threading.Thread(target=worker, daemon=True).start()
    except Exception as e:
        with STATE["lock"]:
            STATE["error"] = str(e)


def encode_jpg(bgr_img):
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


def cam_mjpeg_gen():
    while STATE["running"]:
        with STATE["lock"]:
            frame = STATE["latest_cam"].copy() if STATE["latest_cam"] is not None else None
            rec = STATE["rec"]
            mirror = STATE["mirror"]
        if frame is not None and rec is not None:
            img = lr.draw_frame(frame, rec.snapshot(), mirror)
            jpg = encode_jpg(img)
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)


def avatar_mjpeg_gen():
    while STATE["running"]:
        with STATE["lock"]:
            lib = STATE["lib"]
            pl = STATE["avatar_play"]
            names = STATE["avatar_names"]
            rest = STATE["rest"]
            msg = STATE["avatar_msg"]
            txt = STATE["avatar_text"]

        if lib is not None and rest is not None:
            if pl is not None:
                idx = int((time.time() - pl["t0"]) * av.FPS)
                if idx >= len(pl["E"]):
                    with STATE["lock"]:
                        STATE["avatar_play"] = None
                    pl = None
            if pl is not None:
                img = av.draw_avatar(lib, pl["E"][idx], pl["H"][idx], "", pl["names"], pl["who"][idx], idx)
            else:
                e, h = rest
                img = av.draw_avatar(lib, e, h, "", names, -1, int(time.time() * av.FPS))

            cv2.rectangle(img, (0, 0), (av.W, 52), (24, 28, 38), -1)
            lr.put(img, f"Sentence: {txt or ' '.join(names or [])}", 14, 22, 0.58, (255, 255, 255), 1)
            lr.put(img, msg or "Ready", 14, 43, 0.45, (0, 210, 255), 1)
            jpg = encode_jpg(img)
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream/camera")
def stream_camera():
    return Response(cam_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream/avatar")
def stream_avatar():
    return Response(avatar_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with STATE["lock"]:
        rec = STATE["rec"]
        snap = rec.snapshot() if rec else {}
        return jsonify({
            "ready": STATE["ready"],
            "error": STATE["error"],
            "state": snap.get("state", "LOADING"),
            "dev": f"{float(snap.get('dev', 0.0)):.2f}",
            "last": snap.get("last"),
            "top3": [[n, round(float(q) * 100)] for n, q in snap.get("top3", [])],
            "sentence": snap.get("sentence", []),
            "mirror": STATE["mirror"],
            "min_conf": STATE["min_conf"],
            "use_profile": STATE["use_profile"],
            "has_profile": STATE["has_profile"],
            "known_signs": STATE["known"],
        })


@app.route("/api/action", methods=["POST"])
def api_action():
    data = request.get_json(force=True) or {}
    act = data.get("action", "")
    with STATE["lock"]:
        rec = STATE["rec"]
        if act == "recalibrate" and rec:
            rec.recalibrate()
        elif act == "backspace" and rec:
            rec.pop_word()
        elif act == "clear" and rec:
            rec.clear()
        elif act == "speak" and rec:
            sent = " ".join(rec.snapshot().get("sentence", []))
            if sent:
                lr.speak(sent)
        elif act == "mirror":
            STATE["mirror"] = not STATE["mirror"]
        elif act == "set_min_conf":
            val = float(data.get("min_conf", 0.45))
            val = max(0.20, min(0.90, val))
            STATE["min_conf"] = val
            if rec:
                with rec.lock:
                    rec.min_conf = val
        elif act == "toggle_profile":
            requested = bool(data.get("use_profile", False))
            loaded_prof, has_prof = _load_user_profile_if_available()
            STATE["has_profile"] = has_prof
            enabled = bool(requested and has_prof)
            STATE["use_profile"] = enabled
            if rec:
                with rec.lock:
                    rec.profile = loaded_prof if enabled else None
        elif act == "sign_text":
            txt = (data.get("text") or "").strip()
            STATE["avatar_text"] = txt
            names, unknown = av.text_to_signs(txt, STATE["lib"])
            start_avatar_names(names, unknown)
        elif act == "sign_detected" and rec:
            words = [w for w in rec.snapshot().get("sentence", []) if w in STATE["known"]]
            STATE["avatar_text"] = " ".join(words)
            start_avatar_names(words)
    return jsonify({
        "ok": True,
        "min_conf": STATE["min_conf"],
        "use_profile": STATE["use_profile"],
        "has_profile": STATE["has_profile"],
    })


if __name__ == "__main__":
    init_backend(source=0, backend="dshow", min_conf=0.45, use_profile=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)