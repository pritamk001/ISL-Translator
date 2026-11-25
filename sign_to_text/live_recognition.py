"""
live_recognition.py  -  Step 5: live camera -> signs -> text (and speech)

Run from the project root (isl-translator) with the (islenv) environment active.

  1) Dry run on a dataset video (no camera needed):
       python sign_to_text\\live_recognition.py --source "data\\videos\\Greetings\\48. Hello\\MVI_0029.MOV"

  2) Live camera:
       python sign_to_text\\live_recognition.py

How it works
  * MediaPipe finds hands + body in every camera frame (same code as the dataset).
  * First 2 seconds: keep your hands DOWN / relaxed -> the program learns your "rest" pose.
  * When a hand moves away from the rest pose, a sign STARTS. When both hands come back
    to rest for half a second, the sign ENDS and the model names it.
  * Recognised words are collected into a sentence on the right side of the window.

Keys:   r = re-learn rest pose     BACKSPACE = delete last word     c = clear sentence
        s = speak the sentence     m = mirror the picture on/off    q or ESC = quit
"""

import argparse
import collections
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks.python import BaseOptions, vision
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as fx            # noqa: E402
import extract_landmarks as el   # noqa: E402  (same landmark code as the dataset)
import train_model as tm         # noqa: E402  (model definition)

WINDOW = "ISL Translator"
PANEL_W = 380


# ----------------------------------------------------------------------------- recogniser
class Recognizer:
    def __init__(self, model, classes, delta, hand_model, pose_model,
                 calib_s=2.0, min_conf=0.45, t_on=0.7, t_off=0.4,
                 post_s=0.5, preroll_s=0.5, min_sign_s=0.6, max_sign_s=4.5, profile=None):
        self.model, self.classes, self.delta = model, classes, delta
        self.profile = profile          # your own calibrated fingerprints (see adapt_model.py)
        self.on_segment = None          # recording tools plug in here instead of recognising
        self.calib_s, self.min_conf = calib_s, min_conf
        self.t_on, self.t_off = t_on, t_off
        self.post_s, self.preroll_s = post_s, preroll_s
        self.min_sign_s, self.max_sign_s = min_sign_s, max_sign_s

        h_opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        p_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.hl = vision.HandLandmarker.create_from_options(h_opts)
        self.pl = vision.PoseLandmarker.create_from_options(p_opts)

        self.lock = threading.Lock()
        self._last_ts = -1
        self.aspect = 16 / 9
        self.center = None
        self.scale = None
        self.sentence = []
        self.last_pred = None        # (label, confidence, accepted)
        self.top3 = []
        self.vis = None              # latest landmarks for drawing
        self.dev = 0.0
        self.events = []
        self.recalibrate()

    # ------------------------------------------------------------------ state
    def recalibrate(self):
        with self.lock:
            self.state = "CALIBRATING"
            self.calib = []
            self.calib_t0 = None
            self.rest = [None, None]
            self.preroll = collections.deque()
            self.seg = []
            self.seg_t0 = None
            self.quiet_since = None
            self.active_n = 0
            self.body_missing_since = None

    def snapshot(self):
        with self.lock:
            calib_p = 0.0
            if self.state == "CALIBRATING" and self.calib_t0 is not None and self.calib:
                calib_p = min(1.0, (self.calib[-1][0] - self.calib_t0) / max(self.calib_s, 1e-3))
            return {
                "state": self.state, "dev": self.dev, "vis": self.vis, "top3": list(self.top3),
                "sentence": list(self.sentence), "last": self.last_pred, "calib_p": calib_p,
                "body_missing": self.body_missing_since is not None,
            }

    def pop_word(self):
        with self.lock:
            if self.sentence:
                self.sentence.pop()

    def clear(self):
        with self.lock:
            self.sentence.clear()

    def close(self):
        for d in (self.hl, self.pl):
            try:
                d.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ one frame
    def step(self, frame, t):
        h0, w0 = frame.shape[:2]
        aspect = w0 / h0
        s = 640 / max(h0, w0)
        if s < 1:
            frame = cv2.resize(frame, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hands, hp, pose, pp = el.extract_features(self.hl.detect_for_video(img, ts),
                                                  self.pl.detect_for_video(img, ts))
        rec = {"t": t, "hands": hands, "hp": hp, "pose": pose, "pp": pp}

        with self.lock:
            self.aspect = aspect
            self.vis = rec
            ax = np.array([aspect, 1.0], np.float32)
            body_ok = False
            if pp:
                c = (pose[11, :2] + pose[12, :2]) / 2 * ax
                w = float(np.linalg.norm((pose[11, :2] - pose[12, :2]) * ax))
                if w > 1e-3:
                    body_ok = True
                    self.center = c
                    self.scale = w if self.scale is None else 0.8 * self.scale + 0.2 * w
            if self.center is None:
                self.body_missing_since = self.body_missing_since or t
                return
            self.body_missing_since = None if body_ok else (self.body_missing_since or t)

            wrist_rel = (hands[:, 0, :2] * ax - self.center) / self.scale       # (2, 2)
            dev = 0.0
            for k in range(2):
                if self.rest[k] is None:
                    d = 1.0 if hp[k] else 0.0
                else:
                    d = float(np.linalg.norm(wrist_rel[k] - self.rest[k])) if hp[k] else 0.0
                dev = max(dev, d)
            self.dev = dev

            if self.state == "CALIBRATING":
                self._calibrate(t, wrist_rel, hp)
                return

            # keep a short history so a sign can start slightly before the movement was noticed
            self.preroll.append(rec)
            while self.preroll and t - self.preroll[0]["t"] > self.preroll_s:
                self.preroll.popleft()

            if self.state == "READY":
                self.active_n = self.active_n + 1 if dev > self.t_on else 0
                if self.active_n >= 2:
                    self.state = "SIGNING"
                    self.seg = list(self.preroll)
                    self.seg_t0 = self.seg[0]["t"]
                    self.quiet_since = None
            else:  # SIGNING
                self.seg.append(rec)
                if dev < self.t_off:
                    self.quiet_since = self.quiet_since or t
                else:
                    self.quiet_since = None
                ended = self.quiet_since is not None and t - self.quiet_since >= self.post_s
                too_long = t - self.seg_t0 > self.max_sign_s
                if ended or too_long:
                    self._finish_segment()

    def _calibrate(self, t, wrist_rel, hp):
        if self.calib_t0 is None:
            self.calib_t0 = t
        self.calib.append((t, wrist_rel.copy(), hp.copy()))
        if t - self.calib_t0 >= self.calib_s and len(self.calib) >= 3:
            for k in range(2):
                seen = [(c[1][k]) for c in self.calib if c[2][k]]
                self.rest[k] = np.median(np.stack(seen), axis=0) if len(seen) >= 0.5 * len(self.calib) else None
            self.state = "READY"
            self.active_n = 0

    def _finish_segment(self):
        frames, self.seg = self.seg, []
        self.state = "READY"
        self.active_n = 0
        self.preroll.clear()
        dur = frames[-1]["t"] - frames[0]["t"]
        if dur < self.min_sign_s or len(frames) < 5:
            return
        if self.on_segment is not None:
            self.on_segment(frames, self.aspect)
            return
        x = fx.sequence_features(
            np.stack([f["hands"] for f in frames]), np.stack([f["hp"] for f in frames]),
            np.stack([f["pose"] for f in frames]), np.array([f["pp"] for f in frames], bool),
            self.aspect, delta=self.delta)
        with torch.no_grad():
            xt = torch.from_numpy(x[None]).float()
            if self.profile is not None:                      # match against YOUR fingerprints
                sims = self.model.embed(xt) @ self.profile["protos"].T
                p = F.softmax(16.0 * sims[0], dim=-1).numpy()
                names = self.profile["names"]
            else:
                p = F.softmax(self.model(xt)[0], dim=-1).numpy()
                names = self.classes
        order = np.argsort(-p)[:3]
        self.top3 = [(names[i], float(p[i])) for i in order]
        label, conf = self.top3[0]
        accepted = conf >= self.min_conf
        self.last_pred = (label, conf, accepted)
        if accepted:
            self.sentence.append(label)
        self.events.append((frames[0]["t"], label, conf, accepted))
        mark = "" if accepted else "   (not sure - ignored)"
        print(f"[{frames[0]['t']:5.1f}s]  {label:<16} {conf * 100:4.0f}%   "
              f"other guesses: {', '.join(f'{n} {q * 100:.0f}%' for n, q in self.top3[1:])}{mark}")

    def flush(self):
        """End of a video file: close a sign that is still open."""
        with self.lock:
            if self.state == "SIGNING" and self.seg:
                self._finish_segment()


# ----------------------------------------------------------------------------- drawing
def put(img, text, x, y, size=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def camera_view(frame, snap, mirror, H=480):
    """Camera picture with the hand / body points drawn on it."""
    img = frame.copy()
    v = snap["vis"]
    if v is not None:
        img = el.draw_overlay(img, v["hands"], v["hp"], v["pose"], v["pp"])
    if mirror:
        img = cv2.flip(img, 1)
    return cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))


def draw_frame(frame, snap, mirror):
    H = 480
    img = camera_view(frame, snap, mirror, H)

    panel = np.full((H, PANEL_W, 3), 32, np.uint8)
    state = snap["state"]
    if state == "CALIBRATING":
        put(panel, "Keep your hands DOWN", 14, 34, 0.75, (0, 200, 255), 2)
        put(panel, "learning your rest pose ...", 14, 62, 0.55, (200, 200, 200))
        put(panel, "Sit back: head to waist must be visible", 14, 128, 0.45, (0, 200, 255))
        cv2.rectangle(panel, (14, 76), (PANEL_W - 14, 92), (90, 90, 90), 1)
        cv2.rectangle(panel, (14, 76), (14 + int((PANEL_W - 28) * snap["calib_p"]), 92), (0, 200, 255), -1)
    elif snap["body_missing"]:
        put(panel, "Show face + shoulders", 14, 34, 0.75, (0, 0, 255), 2)
    elif state == "SIGNING":
        put(panel, "Signing ...", 14, 34, 0.8, (0, 165, 255), 2)
    else:
        put(panel, "Ready - show a sign", 14, 34, 0.8, (0, 220, 0), 2)
    put(panel, f"movement {snap['dev']:.2f}", 14, 112, 0.45, (150, 150, 150))

    put(panel, "LAST SIGN", 14, 150, 0.5, (150, 150, 150))
    if snap["last"]:
        label, conf, ok = snap["last"]
        put(panel, label, 14, 188, 1.0, (255, 255, 255) if ok else (120, 120, 120), 2)
        put(panel, f"{conf * 100:.0f}% sure" + ("" if ok else "  (ignored)"), 14, 212, 0.5, (200, 200, 200))
    for i, (name, q) in enumerate(snap["top3"]):
        y = 244 + i * 26
        put(panel, name, 14, y, 0.45, (200, 200, 200))
        cv2.rectangle(panel, (150, y - 12), (150 + int(150 * q), y + 2), (0, 160, 255), -1)
        put(panel, f"{q * 100:.0f}%", 308, y, 0.45, (200, 200, 200))

    put(panel, "SENTENCE", 14, 345, 0.5, (150, 150, 150))
    words, lines, cur = snap["sentence"], [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 24:
            lines.append(cur)
            cur = ""
        cur += (" " if cur else "") + w
    lines.append(cur)
    for i, line in enumerate(lines[-3:]):
        put(panel, line, 14, 376 + i * 28, 0.7, (255, 255, 255), 2)
    put(panel, "r rest  BKSP undo  c clear  s speak  q quit", 14, 468, 0.4, (130, 130, 130))
    return np.hstack([img, panel])


def speak(text):
    def _run():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as e:  # noqa: BLE001
            print("speech failed:", e)
    threading.Thread(target=_run, daemon=True).start()


def handle_key(key, rec, state):
    if key in (ord("q"), 27):
        return False
    if key == ord("r"):
        rec.recalibrate()
    elif key == 8:
        rec.pop_word()
    elif key == ord("c"):
        rec.clear()
    elif key == ord("s"):
        speak(" ".join(rec.snapshot()["sentence"]))
    elif key == ord("m"):
        state["mirror"] = not state["mirror"]
    return True


# ----------------------------------------------------------------------------- runners
def run_file(rec, path, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 < fps < 121 else 30.0
    state = {"mirror": False}
    idx, last_t = 0, 0.0
    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        if not cap.grab():
            break
        if idx % args.step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            last_t = idx / fps
            rec.step(frame, last_t)
            if not args.no_window:
                cv2.imshow(WINDOW, draw_frame(frame, rec.snapshot(), state["mirror"]))
                if not handle_key(cv2.waitKey(1) & 0xFF, rec, state):
                    break
        idx += 1
    cap.release()
    rec.flush()
    print("\nSentence:", " ".join(rec.snapshot()["sentence"]) or "(nothing recognised)")
    if not args.no_window:
        cv2.waitKey(1500)
        cv2.destroyAllWindows()


def run_camera(rec, cap, args, draw_fn=None, key_fn=None):
    latest = {"frame": None, "t": 0.0, "id": 0}
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    t0 = time.monotonic()

    def worker():
        seen = 0
        while running.is_set():
            with lock:
                fid, frame, t = latest["id"], latest["frame"], latest["t"]
            if frame is None or fid == seen:
                time.sleep(0.005)
                continue
            seen = fid
            try:
                rec.step(frame, t)
            except Exception as e:  # noqa: BLE001
                print("frame skipped:", e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    state = {"mirror": True}
    custom = draw_fn is not None
    draw_fn = draw_fn or (lambda frm, snap, st: draw_frame(frm, snap, st["mirror"]))
    key_fn = key_fn or (lambda key, st: handle_key(key, rec, st))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    misses = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            misses += 1
            if misses > 60:
                print("Camera stopped sending pictures - closing.")
                break
            time.sleep(0.03)
            continue
        misses = 0
        with lock:
            latest.update(frame=frame, t=time.monotonic() - t0, id=latest["id"] + 1)
        cv2.imshow(WINDOW, draw_fn(frame, rec.snapshot(), state))
        if not key_fn(cv2.waitKey(1) & 0xFF, state):
            break
    running.clear()
    th.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    if not custom:
        print("\nSentence:", " ".join(rec.snapshot()["sentence"]) or "(nothing recognised)")


BACKENDS = [("dshow", "DirectShow", getattr(cv2, "CAP_DSHOW", None)),
            ("msmf", "MediaFoundation", getattr(cv2, "CAP_MSMF", None)),
            ("any", "default", None)]


def _open(index, backend):
    return cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)


def frame_ok(fr):
    """A real picture: not empty, not (almost) black, not a flat colour."""
    return fr is not None and fr.size > 0 and float(fr.mean()) > 8 and float(fr.std()) > 6


def _wait_for_good_frame(cap, tries):
    """Cameras need a moment (auto-exposure): read several frames until one looks real."""
    last = None
    for _ in range(tries):
        ok, fr = cap.read()
        if ok:
            last = fr
            if frame_ok(fr):
                return True, fr
        time.sleep(0.1)
    return False, last


def try_open(index, backend, size=(640, 480), tries=30):
    """Returns (cap, reason). cap is None when no real picture could be read."""
    cap = _open(index, backend)
    if not cap.isOpened():
        cap.release()
        return None, "cannot open"
    ok, fr = _wait_for_good_frame(cap, tries)
    if not ok:
        cap.release()
        return None, ("black / empty picture" if fr is not None else "no pictures")
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        ok, fr = _wait_for_good_frame(cap, 10)
        if not ok:                              # camera did not like this size -> start again without it
            cap.release()
            cap = _open(index, backend)
            ok, fr = _wait_for_good_frame(cap, tries)
            if not ok:
                cap.release()
                return None, "black / empty picture"
    return cap, "ok"


CAMERA_HELP = """
Could not get a real picture from the camera. Things to check:
  1. Windows Camera app (Start > Camera): does it show your face? If not, the camera itself
     is off (look for a camera key on the keyboard, e.g. Fn+F10 / a sliding shutter).
  2. Windows Settings > Privacy > Camera: turn ON "Allow apps to access your camera" and
     "Allow desktop apps to access your camera"  (if this is OFF Python only gets a BLACK picture).
  3. Close every other app / tab that uses the camera (Zoom, Teams, Chrome tab).
  4. Run   python sign_to_text\\live_recognition.py --list-cameras   to see which camera works.
"""


def open_camera(index, backend_name="auto"):
    backends = [b for b in BACKENDS if backend_name in ("auto", b[0])]
    indices = [index] + ([i for i in (1, 2, 3) if i != index] if index == 0 else [])
    for idx in indices:
        for key, name, be in backends:
            cap, why = try_open(idx, be)
            if cap is not None:
                print(f"Camera {idx} opened ({name}).")
                return cap
            print(f"  camera {idx} with {name}: {why}")
    sys.exit(CAMERA_HELP)


def list_cameras():
    out_dir = el.ROOT / "data" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    good = []
    print("Testing cameras 0-3 (takes up to a minute) ...")
    for idx in range(4):
        for key, name, be in BACKENDS:
            cap, why = try_open(idx, be, size=None, tries=20)
            if cap is None:
                print(f"camera {idx}  {name:<16} -> {why}")
                continue
            ok, fr = cap.read()
            cap.release()
            if ok:
                path = out_dir / f"camera_{idx}_{key}.jpg"
                cv2.imwrite(str(path), fr)
                print(f"camera {idx}  {name:<16} -> OK  {fr.shape[1]}x{fr.shape[0]}  picture saved: {path}")
                good.append((idx, key))
    if good:
        idx, key = good[0]
        print(f"\nUse this:  python sign_to_text\\live_recognition.py --source {idx} --backend {key}")
    else:
        print(CAMERA_HELP)


def load_profile(path):
    d = np.load(path, allow_pickle=False)
    return {"names": [str(n) for n in d["names"]], "protos": torch.from_numpy(d["protos"]).float()}


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = tm.SignNet(ckpt["kind"], ckpt["in_dim"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["classes"], ckpt["delta"], ckpt["kind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera number (0) or path of a video file")
    ap.add_argument("--model", default="best.pt", help="file inside the models folder")
    ap.add_argument("--min-conf", type=float, default=0.45, help="ignore guesses below this")
    ap.add_argument("--no-profile", action="store_true", help="ignore models\\profile.npz (your calibration)")
    ap.add_argument("--calib-s", type=float, default=None, help="seconds to learn the rest pose")
    ap.add_argument("--step", type=int, default=2, help="(video files) use every N-th frame")
    ap.add_argument("--no-window", action="store_true", help="(video files) print only, no window")
    ap.add_argument("--list-cameras", action="store_true", help="test which camera numbers work and exit")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"],
                    help="camera driver to use (try msmf if dshow gives a black picture)")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    torch.set_num_threads(1)
    hand_model = el.ensure_model("hand_landmarker.task")
    pose_model = el.ensure_model("pose_landmarker_lite.task")
    ckpt_path = el.MODELS_DIR / args.model
    if not ckpt_path.exists():
        sys.exit(f"{ckpt_path} not found. Run train_model.py first.")
    model, classes, delta, kind = load_model(ckpt_path)
    print(f"Model: {kind}   signs: {len(classes)}")

    profile = None
    prof_path = el.MODELS_DIR / "profile.npz"
    if prof_path.exists() and not args.no_profile:
        profile = load_profile(prof_path)
        print(f"Using YOUR calibrated profile: {len(profile['names'])} signs")
    is_file = not args.source.isdigit()
    calib = args.calib_s if args.calib_s is not None else (0.3 if is_file else 2.0)
    rec = Recognizer(model, classes, delta, hand_model, pose_model, calib_s=calib,
                     min_conf=args.min_conf, profile=profile)
    try:
        if is_file:
            run_file(rec, Path(args.source), args)
        else:
            run_camera(rec, open_camera(int(args.source), args.backend), args)
    finally:
        rec.close()


if __name__ == "__main__":
    main()
