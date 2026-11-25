"""
extract_landmarks.py  -  Step 3: videos -> landmarks (hands + body pose)

Run from the project root (isl-translator) with the (islenv) environment active.

  1) Dry run on ONE video (opens a picture with hand/body points drawn on it):
       python sign_to_text\\extract_landmarks.py --debug

  2) Quick test, only 2 videos per sign:
       python sign_to_text\\extract_landmarks.py --limit 2

  3) Full run (all signs, resumable - already finished videos are skipped):
       python sign_to_text\\extract_landmarks.py

What it saves for every video  (data\\landmarks\\<id>_<sign>\\<video>.npz):
  hands         (T, 2, 21, 3)  x, y, z of both hands. slot 0 = hand MediaPipe calls "Left",
                               slot 1 = "Right". Zeros when that hand is not visible.
  hand_present  (T, 2)         True where that hand was detected
  pose          (T, 33, 4)     body points x, y, z, visibility
  pose_present  (T,)           True where body was detected
and one summary table:  data\\landmarks\\index.csv

NOTE: frames are NOT mirrored here. Live camera code must use the same convention.
"""

import argparse
import csv
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "data" / "landmarks"
DEBUG_DIR = ROOT / "data" / "debug"
MODELS_DIR = ROOT / "models"

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "hand_landmarker.task": [
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ],
    "pose_landmarker_lite.task": [
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ],
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
UPPER_POSE = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
POSE_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


# --------------------------------------------------------------------------- models
def ensure_model(name):
    """Return path of a MediaPipe model file, downloading it once if missing."""
    path = MODELS_DIR / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for url in MODEL_URLS[name]:
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, str(path))
            if path.stat().st_size > 1_000_000:
                print(f"  saved to {path}")
                return path
        except Exception as e:  # noqa: BLE001
            print(f"  failed ({e})")
    print("\nCould not download the model automatically.")
    print(f"Open this link in Chrome and save the file as:  {path}")
    print(f"  {MODEL_URLS[name][0]}")
    sys.exit(1)


# --------------------------------------------------------------------------- helpers
def parse_label(folder_name):
    """'48. Hello' -> (48, 'Hello')"""
    m = re.match(r"\s*(\d+)\.\s*(.+)", folder_name)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return -1, folder_name.strip()


def slugify(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def find_videos():
    """All videos under data/videos, as dicts with category / label info."""
    items = []
    for p in sorted(VIDEOS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label_id, label = parse_label(p.parent.name)
            items.append({
                "path": p,
                "category": p.parent.parent.name,
                "label_id": label_id,
                "label": label,
            })
    return items


def out_path_for(item):
    folder = f"{item['label_id']:03d}_{slugify(item['label'])}"
    return OUT_DIR / folder / f"{item['path'].stem}.npz"


def extract_features(hand_res, pose_res):
    """Turn MediaPipe results of one frame into fixed-size numpy arrays."""
    hands = np.zeros((2, 21, 3), np.float32)
    hand_present = np.zeros(2, bool)
    if hand_res.hand_landmarks:
        scores = [h[0].score for h in hand_res.handedness]
        for i in np.argsort(scores)[::-1]:  # most confident hand first
            label = hand_res.handedness[i][0].category_name
            slot = 0 if label == "Left" else 1
            if hand_present[slot]:          # slot already used -> try the other one
                slot = 1 - slot
                if hand_present[slot]:
                    continue
            hands[slot] = [[p.x, p.y, p.z] for p in hand_res.hand_landmarks[i]]
            hand_present[slot] = True

    pose = np.zeros((33, 4), np.float32)
    pose_present = False
    if pose_res.pose_landmarks:
        pose[:] = [[p.x, p.y, p.z, (p.visibility or 0.0)] for p in pose_res.pose_landmarks[0]]
        pose_present = True
    return hands, hand_present, pose, pose_present


def draw_overlay(frame, hands, hand_present, pose, pose_present):
    img = frame.copy()
    h, w = img.shape[:2]
    if pose_present:
        pts = {i: (int(pose[i, 0] * w), int(pose[i, 1] * h)) for i in UPPER_POSE}
        for a, b in POSE_LINES:
            cv2.line(img, pts[a], pts[b], (255, 200, 0), 2)
        for i in UPPER_POSE:
            cv2.circle(img, pts[i], 3, (0, 255, 255), -1)
    colors = [(0, 255, 0), (0, 0, 255)]  # slot 0 green, slot 1 red
    for s in range(2):
        if hand_present[s]:
            p = [(int(x * w), int(y * h)) for x, y, _ in hands[s]]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, p[a], p[b], colors[s], 2)
            for q in p:
                cv2.circle(img, q, 3, (255, 255, 255), -1)
    return img


# --------------------------------------------------------------------------- core
def process_video(path, hand_model, pose_model, step=2, max_side=640, keep_frames=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1 < fps < 121):
        fps = 30.0

    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    H, HP, P, PP, frames = [], [], [], [], []
    idx, last_ts = 0, -1
    with vision.HandLandmarker.create_from_options(hand_opts) as hl, \
            vision.PoseLandmarker.create_from_options(pose_opts) as pl:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                fh, fw = frame.shape[:2]
                scale = max_side / max(fh, fw)
                if scale < 1:
                    frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                       interpolation=cv2.INTER_AREA)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(idx / fps * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_res = hl.detect_for_video(mp_img, ts)
                pose_res = pl.detect_for_video(mp_img, ts)
                h, hp, p, pp = extract_features(hand_res, pose_res)
                H.append(h); HP.append(hp); P.append(p); PP.append(pp)
                if keep_frames:
                    frames.append(frame)
            idx += 1
    cap.release()

    if not H:
        raise ValueError("no frames could be read")
    return {
        "hands": np.stack(H), "hand_present": np.stack(HP),
        "pose": np.stack(P), "pose_present": np.array(PP, bool),
        "fps": float(fps), "step": int(step), "frames_total": idx, "frames": frames,
    }


def save_npz(item, res):
    out = out_path_for(item)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        hands=res["hands"], hand_present=res["hand_present"],
        pose=res["pose"], pose_present=res["pose_present"],
        fps=res["fps"], step=res["step"],
        label_id=item["label_id"], label=item["label"], category=item["category"],
        video=str(item["path"].relative_to(ROOT)),
    )
    return out


def build_index():
    """Rebuild data/landmarks/index.csv from all saved .npz files."""
    rows = []
    for f in sorted(OUT_DIR.rglob("*.npz")):
        d = np.load(f)
        hp = d["hand_present"]
        rows.append({
            "npz": str(f.relative_to(ROOT)),
            "video": str(d["video"]),
            "category": str(d["category"]),
            "label_id": int(d["label_id"]),
            "label": str(d["label"]),
            "frames": int(hp.shape[0]),
            "fps": round(float(d["fps"]), 1),
            "any_hand_%": round(100 * float(hp.any(axis=1).mean()), 1),
            "both_hands_%": round(100 * float(hp.all(axis=1).mean()), 1),
            "pose_%": round(100 * float(d["pose_present"].mean()), 1),
        })
    if rows:
        with open(OUT_DIR / "index.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- modes
def run_debug(video_arg, hand_model, pose_model, step, max_side):
    if video_arg and video_arg != "auto":
        path = Path(video_arg)
        item = {"path": path, "category": "?", "label_id": 0, "label": path.parent.name}
    else:
        items = find_videos()
        if not items:
            print(f"No videos found under {VIDEOS_DIR}")
            sys.exit(1)
        item = items[0]
        path = item["path"]
    print(f"Debug video: {path}")
    t0 = time.time()
    res = process_video(path, hand_model, pose_model, step, max_side, keep_frames=True)
    secs = time.time() - t0
    T = res["hands"].shape[0]
    print(f"frames in video: {res['frames_total']}   frames processed: {T}   fps: {res['fps']:.1f}")
    print(f"frame size used: {res['frames'][0].shape[1]} x {res['frames'][0].shape[0]}")
    print(f"hand detected: {100 * res['hand_present'].any(axis=1).mean():.0f}% of frames "
          f"| both hands: {100 * res['hand_present'].all(axis=1).mean():.0f}% "
          f"| body pose: {100 * res['pose_present'].mean():.0f}%")
    print(f"time taken: {secs:.1f} s  ({secs / T * 1000:.0f} ms per processed frame)")

    picks = np.linspace(0, T - 1, min(6, T)).astype(int)
    cells = []
    for i in picks:
        img = draw_overlay(res["frames"][i], res["hands"][i], res["hand_present"][i],
                           res["pose"][i], res["pose_present"][i])
        img = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))
        cv2.putText(img, f"frame {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cells.append(img)
    while len(cells) % 3:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    montage = np.vstack(rows)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / "debug_overlay.jpg"
    cv2.imwrite(str(out), montage)
    print(f"\nPicture saved: {out}")
    try:
        os.startfile(str(out))  # opens in the default image viewer (Windows)
    except Exception:  # noqa: BLE001
        pass


def run_all(hand_model, pose_model, step, max_side, limit, redo):
    items = find_videos()
    if not items:
        print(f"No videos found under {VIDEOS_DIR}")
        sys.exit(1)
    if limit:
        seen, kept = {}, []
        for it in items:
            key = (it["category"], it["label_id"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= limit:
                kept.append(it)
        items = kept

    todo = [it for it in items if redo or not out_path_for(it).exists()]
    signs = len({(i["category"], i["label_id"]) for i in items})
    print(f"Videos found: {len(items)}   signs: {signs}   still to process: {len(todo)}")

    failed, t_start = [], time.time()
    for n, it in enumerate(todo, 1):
        t0 = time.time()
        try:
            res = process_video(it["path"], hand_model, pose_model, step, max_side)
            save_npz(it, res)
            hp = 100 * res["hand_present"].any(axis=1).mean()
            elapsed = time.time() - t_start
            eta = elapsed / n * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)}] {it['label_id']}. {it['label']:<18} {it['path'].name:<14} "
                  f"{res['hands'].shape[0]:>3} frames  hands {hp:>3.0f}%  "
                  f"{time.time() - t0:>4.1f}s  ETA {eta:>4.1f} min")
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{len(todo)}] FAILED {it['path'].name}: {e}")
            failed.append(f"{it['path']}  ->  {e}")

    rows = build_index()
    print(f"\nDone. Saved landmark files: {len(rows)}   index: {OUT_DIR / 'index.csv'}")
    if failed:
        (OUT_DIR / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} videos failed, list in {OUT_DIR / 'failed.txt'}")


def main():
    ap = argparse.ArgumentParser(description="Extract hand + pose landmarks from sign videos")
    ap.add_argument("--debug", nargs="?", const="auto", default=None,
                    help="dry run on one video (optional path); saves a picture with points drawn")
    ap.add_argument("--limit", type=int, default=0, help="only first N videos of every sign")
    ap.add_argument("--step", type=int, default=2, help="use every N-th frame (default 2)")
    ap.add_argument("--max-side", type=int, default=640, help="resize frames so longest side is this")
    ap.add_argument("--redo", action="store_true", help="process again even if already saved")
    args = ap.parse_args()

    hand_model = ensure_model("hand_landmarker.task")
    pose_model = ensure_model("pose_landmarker_lite.task")

    if args.debug is not None:
        run_debug(args.debug, hand_model, pose_model, args.step, args.max_side)
    else:
        run_all(hand_model, pose_model, args.step, args.max_side, args.limit, args.redo)


if __name__ == "__main__":
    main()
