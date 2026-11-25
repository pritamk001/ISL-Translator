"""
record_samples.py  -  Step 6a: record YOUR OWN samples of each sign

Why: the model learned from other people on another camera. A few samples of YOU
on YOUR camera let the system adapt (calibration), and the same tool can add a
completely NEW sign (add-a-sign).

Run from the project root (isl-translator) with the (islenv) environment active.

  python sign_to_text\\record_samples.py                    # all 17 signs, 5 samples each
  python sign_to_text\\record_samples.py --signs "Hello,Thank you"
  python sign_to_text\\record_samples.py --new "Water"       # a brand-new sign (add-a-sign)

How to record
  * Sit so that head, shoulders AND elbows are visible (head to waist). Good light.
  * First 2 seconds: keep your hands DOWN (the program learns your rest pose).
  * The sign to record is shown on the right with a small reference video (the real signer).
  * Do the sign, then bring your hands back down. It is saved automatically.
  * After N samples the next sign comes up. You can stop with q and continue later.

Keys:  n = next sign   b = previous sign   u = undo last sample   m = mirror   q = quit
Samples are stored in  data\\own\\  (landmark numbers only, no pictures).
"""

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_recognition as lr   # noqa: E402
import extract_landmarks as el  # noqa: E402

OWN_DIR = el.ROOT / "data" / "own"
NEW_SIGN_ID_START = 1001


def slug(name):
    return el.slugify(name)


def load_dataset_signs():
    path = el.MODELS_DIR / "classes.json"
    if not path.exists():
        sys.exit("models\\classes.json not found. Run train_model.py first.")
    d = json.loads(path.read_text())
    return list(zip(d["label_ids"], d["names"]))


def existing_new_signs():
    """Signs already recorded by the user that are not in the dataset: {name: id}."""
    found = {}
    if OWN_DIR.exists():
        for d in OWN_DIR.iterdir():
            m = re.match(r"(\d+)_(.+)", d.name)
            if d.is_dir() and m and int(m.group(1)) >= NEW_SIGN_ID_START:
                f = next(d.glob("*.npz"), None)
                if f is not None:
                    found[str(np.load(f)["label"])] = int(m.group(1))
    return found


def new_sign_id(name):
    known = existing_new_signs()
    if name in known:
        return known[name]
    return max([NEW_SIGN_ID_START - 1] + list(known.values())) + 1


def own_folder(label_id, name):
    return OWN_DIR / f"{label_id:03d}_{slug(name)}"


def count_samples(label_id, name):
    d = own_folder(label_id, name)
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def load_reference_clips(label_ids, width=300, max_frames=40):
    """One small looping clip per dataset sign, so you can see how it should look."""
    clips = {}
    firsts = {}
    for it in el.find_videos():
        firsts.setdefault(it["label_id"], it["path"])
    for lid in label_ids:
        p = firsts.get(lid)
        if p is None:
            continue
        cap = cv2.VideoCapture(str(p))
        frames, i = [], 0
        while True:
            if not cap.grab():
                break
            if i % 3 == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                h = int(fr.shape[0] * width / fr.shape[1])
                frames.append(cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA))
            i += 1
        cap.release()
        if frames:
            step = max(1, len(frames) // max_frames)
            clips[lid] = frames[::step]
    return clips


class Session:
    """Keeps track of which sign is being recorded and saves every captured sample."""

    def __init__(self, signs, n_samples, folder_fn=own_folder):
        self.signs, self.n = signs, n_samples
        self.folder_fn = folder_fn
        self.saved = {lid: sorted(self.folder_fn(lid, nm).glob("*.npz")) if self.folder_fn(lid, nm).exists() else []
                      for lid, nm in signs}
        self.lock = threading.Lock()
        self.msg, self.lock_until = "", -1.0
        self.idx = self._first_unfinished(0)

    def _first_unfinished(self, start):
        for k in list(range(start, len(self.signs))) + list(range(0, start)):
            lid, nm = self.signs[k]
            if len(self.saved[lid]) < self.n:
                return k
        return min(start, len(self.signs) - 1)

    def done(self):
        return all(len(self.saved[lid]) >= self.n for lid, _ in self.signs)

    def current(self):
        return self.signs[self.idx]

    def on_segment(self, frames, aspect):
        with self.lock:
            if self.done() or frames[0]["t"] < self.lock_until:
                return
            lid, name = self.current()
            hp = np.stack([f["hp"] for f in frames])
            if hp.any(axis=1).mean() < 0.4:
                self.msg = "Hands not seen clearly - try again"
                return
            folder = self.folder_fn(lid, name)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"own_{int(time.time() * 1000)}.npz"
            np.savez_compressed(
                path, hands=np.stack([f["hands"] for f in frames]), hand_present=hp,
                pose=np.stack([f["pose"] for f in frames]),
                pose_present=np.array([f["pp"] for f in frames], bool),
                fps=0.0, step=1, label_id=lid, label=name, category="own", video="own", aspect=aspect)
            self.saved[lid].append(path)
            got = len(self.saved[lid])
            self.msg = f"Saved {name}  {got}/{self.n}"
            print(self.msg)
            if got >= self.n:
                self.idx = self._first_unfinished(self.idx + 1)
                self.lock_until = frames[-1]["t"] + 1.0
                if self.done():
                    self.msg = "ALL DONE - press q"

    def move(self, step):
        with self.lock:
            self.idx = (self.idx + step) % len(self.signs)

    def undo(self):
        with self.lock:
            lid, name = self.current()
            if self.saved[lid]:
                self.saved[lid].pop().unlink()
                self.msg = f"Removed last sample of {name}"


def draw_panel(snap, session, ref_frame, H=480, W=lr.PANEL_W):
    panel = np.full((H, W, 3), 32, np.uint8)
    lid, name = session.current()
    got = len(session.saved[lid])
    lr.put(panel, "RECORD THIS SIGN", 14, 24, 0.5, (150, 150, 150))
    lr.put(panel, name, 14, 60, 1.0, (255, 255, 255), 2)
    for k in range(session.n):
        cv2.circle(panel, (22 + k * 26, 82), 8, (0, 200, 0) if k < got else (90, 90, 90), -1 if k < got else 2)
    if ref_frame is not None:
        h, w = ref_frame.shape[:2]
        panel[100:100 + min(h, 175), 14:14 + min(w, W - 28)] = ref_frame[:175, :W - 28]
        lr.put(panel, "reference (dataset signer)", 14, 292, 0.4, (130, 130, 130))
    state = snap["state"]
    if state == "CALIBRATING":
        lr.put(panel, "Keep hands DOWN ...", 14, 322, 0.7, (0, 200, 255), 2)
        cv2.rectangle(panel, (14, 332), (W - 14, 344), (90, 90, 90), 1)
        cv2.rectangle(panel, (14, 332), (14 + int((W - 28) * snap["calib_p"]), 344), (0, 200, 255), -1)
    elif snap["body_missing"]:
        lr.put(panel, "Show face + shoulders", 14, 322, 0.7, (0, 0, 255), 2)
    elif state == "SIGNING":
        lr.put(panel, "Signing ... hands down when done", 14, 322, 0.6, (0, 165, 255), 2)
    else:
        lr.put(panel, "Ready - do the sign", 14, 322, 0.75, (0, 220, 0), 2)
    lr.put(panel, session.msg, 14, 372, 0.55, (255, 255, 255))
    total = sum(min(len(v), session.n) for v in session.saved.values())
    allv = session.n * len(session.signs)
    lr.put(panel, f"sign {session.idx + 1} of {len(session.signs)}   total {total}/{allv}", 14, 404, 0.5, (180, 180, 180))
    cv2.rectangle(panel, (14, 414), (W - 14, 424), (90, 90, 90), 1)
    cv2.rectangle(panel, (14, 414), (14 + int((W - 28) * total / max(allv, 1)), 424), (0, 200, 0), -1)
    lr.put(panel, "n next  b back  u undo  m mirror  q quit", 14, 462, 0.42, (130, 130, 130))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5, help="samples per sign (default 5)")
    ap.add_argument("--signs", default="", help='comma separated names, e.g. "Hello,Thank you"')
    ap.add_argument("--new", default="", help="record a brand-new sign with this name")
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    args = ap.parse_args()

    dataset_signs = load_dataset_signs()
    if args.new:
        signs = [(new_sign_id(args.new.strip()), args.new.strip())]
    elif args.signs:
        wanted = [s.strip().lower() for s in args.signs.split(",") if s.strip()]
        signs = [s for s in dataset_signs if s[1].lower() in wanted]
        if not signs:
            sys.exit("None of those names matched. Available: " + ", ".join(n for _, n in dataset_signs))
    else:
        signs = dataset_signs

    hand_model = el.ensure_model("hand_landmarker.task")
    pose_model = el.ensure_model("pose_landmarker_lite.task")
    session = Session(signs, args.samples)
    print(f"Recording {len(signs)} sign(s), {args.samples} samples each. Samples go to {OWN_DIR}")
    clips = {} if args.new else load_reference_clips([lid for lid, _ in signs])

    rec = lr.Recognizer(None, [], False, hand_model, pose_model, calib_s=2.0)
    rec.on_segment = session.on_segment

    def draw_fn(frame, snap, st):
        cam = lr.camera_view(frame, snap, st["mirror"])
        lid, _ = session.current()
        clip = clips.get(lid)
        ref = clip[int(time.time() * 8) % len(clip)] if clip else None
        return np.hstack([cam, draw_panel(snap, session, ref)])

    def key_fn(key, st):
        if key in (ord("q"), 27):
            return False
        if key == ord("n"):
            session.move(1)
        elif key == ord("b"):
            session.move(-1)
        elif key == ord("u"):
            session.undo()
        elif key == ord("m"):
            st["mirror"] = not st["mirror"]
        return True

    try:
        lr.run_camera(rec, lr.open_camera(int(args.source), args.backend), args, draw_fn, key_fn)
    finally:
        rec.close()
    print("\nSamples recorded so far:")
    for lid, nm in signs:
        print(f"  {nm:<16} {len(session.saved[lid])}/{args.samples}")
    print("\nNext:  python sign_to_text\\adapt_model.py")


if __name__ == "__main__":
    main()
