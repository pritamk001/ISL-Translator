"""
app.py  -  Step 8: ONE app for both directions

  Camera mode : you sign in front of the camera  ->  words appear as text (and can be spoken)
  Avatar mode : you type a sentence               ->  the avatar signs it

Run from the project root (isl-translator) with the (islenv) environment active:

  python sign_to_text\\app.py
  python sign_to_text\\app.py --source 0 --backend msmf      # if the camera gives a black picture

Keys in CAMERA mode
  TAB     switch to avatar mode
  ENTER   the avatar signs the sentence the camera just recognised  (sign -> text -> sign)
  r = re-learn rest pose    BACKSPACE = delete last word    c = clear    s = speak
  m = mirror on/off         f = fullscreen on/off           q or ESC = quit

Keys in AVATAR mode
  type a sentence, ENTER = sign it      BACKSPACE = delete a letter
  TAB = back to camera mode             ESC = quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avatar as av                  # noqa: E402
import extract_landmarks as el       # noqa: E402
import live_recognition as lr        # noqa: E402


def fit(img):
    """Scale the picture to the window (keeping its shape) and centre it - works in full screen too."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(lr.WINDOW)
    except cv2.error:
        return img
    if ww < 50 or wh < 50:
        return img
    h, w = img.shape[:2]
    s = min(ww / w, wh / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((wh, ww, 3), (35, 35, 35), np.uint8)
    y0, x0 = (wh - r.shape[0]) // 2, (ww - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def start_signing(ctx, names, unknown=()):
    """Prepare the avatar animation for a list of sign names."""
    lib = ctx["lib"]
    if not names:
        ctx["msg"] = "No known word. Try: " + ", ".join(ctx["known"][:6]) + " ..."
        return
    E, H_, who = av.sentence_motion(lib, names)
    ctx["play"] = {"E": E, "H": H_, "who": who, "names": names, "t0": time.time()}
    ctx["last_names"] = names
    ctx["msg"] = ("skipped: " + ", ".join(unknown)) if unknown else ""


def avatar_view(ctx):
    lib = ctx["lib"]
    pl = ctx["play"]
    if pl is not None:
        i = int((time.time() - pl["t0"]) * av.FPS)
        if i >= len(pl["E"]):
            ctx["play"], pl = None, None
    if pl is not None:
        img = av.draw_avatar(lib, pl["E"][i], pl["H"][i], "", pl["names"], pl["who"][i], i)
    else:
        e, h = ctx["rest"]
        img = av.draw_avatar(lib, e, h, "", ctx["last_names"], -1, int(time.time() * av.FPS))
    # input banner at the top
    cv2.rectangle(img, (0, 0), (av.W, 58), (35, 35, 35), -1)
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    lr.put(img, "Type: " + ctx["text"] + cursor, 14, 24, 0.6, (255, 255, 255), 1)
    hint = ctx["msg"] or "ENTER = sign    BKSP = delete    TAB = camera mode    ESC = quit"
    lr.put(img, hint, 14, 46, 0.45, (0, 200, 255) if ctx["msg"] else (150, 150, 150))
    return img


def camera_view(frame, snap, st, ctx):
    img = lr.draw_frame(frame, snap, st["mirror"])
    strip = np.full((28, img.shape[1], 3), 20, np.uint8)
    text = ctx["msg"] or "TAB avatar mode   ENTER avatar signs this sentence   f fullscreen"
    lr.put(strip, text, 10, 19, 0.45, (0, 200, 255))
    return np.vstack([strip, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera number")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--min-conf", type=float, default=0.45)
    ap.add_argument("--no-profile", action="store_true", help="ignore models\\profile.npz")
    ap.add_argument("--windowed", action="store_true", help="do not start in full screen")
    args = ap.parse_args()

    lib = av.load_library()
    known = sorted(m["name"] for m in lib["signs"].values())
    print("Avatar can sign:", ", ".join(known))

    torch.set_num_threads(1)
    hand_model = el.ensure_model("hand_landmarker.task")
    pose_model = el.ensure_model("pose_landmarker_lite.task")
    ckpt = el.MODELS_DIR / args.model
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found. Run train_model.py first.")
    model, classes, delta, kind = lr.load_model(ckpt)
    profile = None
    prof_path = el.MODELS_DIR / "profile.npz"
    if prof_path.exists() and not args.no_profile:
        profile = lr.load_profile(prof_path)
        print(f"Using YOUR calibrated profile: {len(profile['names'])} signs")
    rec = lr.Recognizer(model, classes, delta, hand_model, pose_model, calib_s=2.0,
                        min_conf=args.min_conf, profile=profile)

    ctx = {"mode": "camera", "text": "", "msg": "", "play": None, "last_names": None,
           "lib": lib, "known": known, "rest": av.rest_pose(lib),
           "fs": not args.windowed, "fs_applied": None}

    def draw_fn(frame, snap, st):
        if ctx["fs"] != ctx["fs_applied"]:
            cv2.setWindowProperty(lr.WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if ctx["fs"] else cv2.WINDOW_NORMAL)
            ctx["fs_applied"] = ctx["fs"]
        img = avatar_view(ctx) if ctx["mode"] == "avatar" else camera_view(frame, snap, st, ctx)
        return fit(img)

    def key_fn(key, st):
        if key == 255:
            return True
        if key == 9:                                            # TAB switches mode
            ctx["mode"] = "avatar" if ctx["mode"] == "camera" else "camera"
            ctx["msg"] = ""
            return True
        if ctx["mode"] == "camera":
            if key == 13:                                       # ENTER: avatar signs the recognised sentence
                words = [w for w in rec.snapshot()["sentence"] if w in known]
                if not words:
                    ctx["msg"] = "Nothing recognised yet - show a sign first"
                else:
                    ctx["mode"] = "avatar"
                    start_signing(ctx, words)
                return True
            if key == ord("f"):
                ctx["fs"] = not ctx["fs"]
                return True
            ctx["msg"] = ""
            return lr.handle_key(key, rec, st)
        # avatar mode: typing
        if key == 27:
            return False
        if key == 13:
            names, unknown = av.text_to_signs(ctx["text"], lib)
            start_signing(ctx, names, unknown)
            ctx["text"] = ""
        elif key == 8:
            ctx["text"] = ctx["text"][:-1]
        elif 32 <= key < 127:
            ctx["text"] += chr(key)
            ctx["msg"] = ""
        return True

    try:
        cap = lr.open_camera(int(args.source), args.backend)
        lr.run_camera(rec, cap, args, draw_fn=draw_fn, key_fn=key_fn)
    finally:
        rec.close()


if __name__ == "__main__":
    main()
