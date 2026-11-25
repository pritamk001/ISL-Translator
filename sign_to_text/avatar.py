"""
avatar.py  -  Step 7: text -> animated avatar that signs   (v2: polished look)

Run from the project root (isl-translator) with the (islenv) environment active.

  python sign_to_text\\avatar.py "Hello how are you"        # avatar signs the sentence in a window
  python sign_to_text\\avatar.py --interactive              # type sentence after sentence
  python sign_to_text\\avatar.py "Thank you" --save results\\thankyou.mp4   # save as a video
  python sign_to_text\\avatar.py --list                     # which words the avatar knows
  python sign_to_text\\avatar.py --roundtrip                # test: does our recogniser understand the avatar?

How it works (for the examiner)
  * For every sign we pick the most typical recording of the real signer (the one closest to the
    average of all recordings) and keep only its landmarks (hand + arm points), not the picture.
  * Points are normalised by the shoulders, smoothed, resting frames at the start / end are trimmed.
  * A stylised character is drawn from those points, so the avatar repeats the real movement.
  * A sentence = the signs one after the other with a short smooth transition between them.
  * The avatar does NOT invent new signs: it only shows signs we have recorded data for.
    (A text-to-sign generator that invents signs is still an open research problem.)

Keys while playing:  p = pause   SPACE = replay   q or ESC = close
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LM_DIR = ROOT / "data" / "landmarks"
OWN_DIR = ROOT / "data" / "own"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LIB_PATH = MODELS_DIR / "avatar_library.npz"

FPS = 30
W = 600                         # window width
H_BODY = 640                    # height of the picture part
CAP = 60                        # height of the caption bar under the picture
H = H_BODY + CAP
SW = 150                        # shoulder width in pixels
CX, CY = W // 2, 310            # shoulder centre
SHOW_HINT = True                # the website sets this to False (no keyboard hint text)
HAND_SCALE = 1.4                # hands are drawn a bit larger so the fingers are easy to see

# colours are BGR
SKIN, SKIN_D = (150, 185, 235), (90, 120, 170)
SKIN_M = (118, 152, 205)         # neck / chest shadow
NAIL = (185, 208, 248)
SHIRT, SHIRT_D = (170, 105, 55), (110, 62, 30)
HAIR = (40, 32, 30)
LIP = (75, 80, 170)
BG_TOP, BG_BOT = (246, 244, 240), (220, 217, 212)

# 21 hand points of an open hand pointing up (wrist at 0,0), same order as MediaPipe
OPEN_HAND = np.array([
    [0, 0], [-.22, -.14], [-.36, -.30], [-.46, -.44], [-.52, -.56],
    [-.15, -.42], [-.17, -.64], [-.18, -.80], [-.19, -.94],
    [-.02, -.45], [-.02, -.69], [-.02, -.86], [-.02, -1.0],
    [.11, -.42], [.13, -.64], [.14, -.79], [.15, -.92],
    [.22, -.36], [.26, -.53], [.28, -.65], [.29, -.76]], np.float32)
RELAXED = -OPEN_HAND * 0.5      # a hand hanging down (used when a hand is not part of the sign)
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM = [0, 1, 5, 9, 13, 17]


# ============================================================================ motion building
def _fill(arr, present):
    """Fill gaps (present == False) by interpolation over time. arr: (T, ...)."""
    if present.all():
        return arr.copy()
    if not present.any():
        return None
    t = np.arange(len(arr))
    flat = arr.reshape(len(arr), -1)
    out = np.stack([np.interp(t, t[present], flat[present, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(arr.shape).astype(np.float32)


def _smooth(arr, w=3):
    if len(arr) < w:
        return arr
    pad = w // 2
    p = np.concatenate([np.repeat(arr[:1], pad, 0), arr, np.repeat(arr[-1:], pad, 0)], 0)
    return np.mean([p[i:i + len(arr)] for i in range(w)], axis=0).astype(np.float32)


def canonical_motion(hands, hp, pose, pp, aspect, dt):
    """Raw landmarks of one recording -> avatar motion in 'shoulder-width' units. None if unusable."""
    ax = np.array([aspect, 1.0], np.float32)
    c = (pose[:, 11, :2] + pose[:, 12, :2]) / 2 * ax
    w = np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * ax, axis=1)
    valid = pp & (w > 1e-3)
    if not valid.any() or len(hands) < 6:
        return None
    center, scale = np.median(c[valid], axis=0), float(np.median(w[valid]))
    P = lambda xy: ((xy * ax - center) / scale).astype(np.float32)          # noqa: E731

    elbow = np.stack([P(pose[:, 13, :2]), P(pose[:, 14, :2])], 1)           # arm A, arm B  (T, 2, 2)
    pwrist = np.stack([P(pose[:, 15, :2]), P(pose[:, 16, :2])], 1)
    elbow = _fill(elbow, valid)
    pwrist = _fill(pwrist, valid)
    hand_slot = P(hands[..., :2])                                           # (T, 2, 21, 2)

    # which detected hand belongs to which arm: nearest wrist
    arm_hand = [None, None]
    cost = np.full((2, 2), np.inf)
    for s in range(2):
        m = hp[:, s] & valid
        if m.any():
            for a in range(2):
                cost[s, a] = np.mean(np.linalg.norm(hand_slot[m, s, 0] - pwrist[m, a], axis=1))
    if np.isfinite(cost).all():
        a_of = [0, 1] if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else [1, 0]
        for s in range(2):
            arm_hand[a_of[s]] = (hand_slot[:, s], hp[:, s])
    else:
        for s in range(2):
            if np.isfinite(cost[s]).any():
                a = int(np.argmin(cost[s]))
                if arm_hand[a] is None:
                    arm_hand[a] = (hand_slot[:, s], hp[:, s])

    T = len(hands)
    hand = np.zeros((T, 2, 21, 2), np.float32)
    for a in range(2):
        filled = _fill(arm_hand[a][0], arm_hand[a][1]) if arm_hand[a] is not None else None
        hand[:, a] = filled if filled is not None else pwrist[:, a, None, :] + RELAXED[None]
    elbow, hand = _smooth(elbow), _smooth(hand)

    # trim the resting frames at the start and the end
    rest = np.median(hand[:3, :, 0], axis=0)
    dev = np.linalg.norm(hand[:, :, 0] - rest, axis=-1).max(axis=1)
    act = np.where(dev > 0.5)[0]
    if len(act):
        a0, a1 = max(0, act[0] - 3), min(T, act[-1] + 4)
        if a1 - a0 >= 5:
            elbow, hand = elbow[a0:a1], hand[a0:a1]
    return {"elbow": elbow, "hand": hand, "dt": float(dt),
            "used": np.array([arm_hand[0] is not None, arm_hand[1] is not None]),
            "shoulder": np.stack([P(np.median(pose[valid, 11, :2], 0)), P(np.median(pose[valid, 12, :2], 0))]),
            "nose": P(np.median(pose[valid, 0, :2], 0))}


def build_library():
    """Pick the most typical recording of every sign and store its avatar motion."""
    aspects = {}
    ap = LM_DIR / "aspects.json"
    if ap.exists():
        aspects = json.loads(ap.read_text())
    groups = {}
    for f in sorted(LM_DIR.rglob("*.npz")):
        d = np.load(f)
        fps = float(d["fps"]) / max(int(d["step"]), 1) if float(d["fps"]) > 0 else 15.0
        groups.setdefault((int(d["label_id"]), str(d["label"])), []).append({
            "raw": (d["hands"], d["hand_present"], d["pose"], d["pose_present"]),
            "aspect": float(aspects.get(str(d["video"]), 16 / 9)), "dt": 1.0 / fps,
            "file": str(f.relative_to(ROOT))})
    if OWN_DIR.exists():                                    # only signs that are NOT in the dataset
        own = {}
        for f in sorted(OWN_DIR.rglob("*.npz")):
            d = np.load(f)
            if int(d["label_id"]) >= 1001:
                own.setdefault((int(d["label_id"]), str(d["label"])), []).append({
                    "raw": (d["hands"], d["hand_present"], d["pose"], d["pose_present"]),
                    "aspect": float(d["aspect"]), "dt": 0.125, "file": str(f.relative_to(ROOT))})
        groups.update(own)
    if not groups:
        sys.exit(f"No landmark files in {LM_DIR}. Run extract_landmarks.py first.")
    out, noses, shoulders = {}, [], []
    for (lid, name), samples in sorted(groups.items()):
        X = np.stack([fx.sequence_features(*s["raw"], s["aspect"]).reshape(-1) for s in samples])
        order = np.argsort(np.linalg.norm(X - X.mean(0), axis=1))
        for i in order:                                     # best first, skip unusable ones
            s = samples[int(i)]
            m = canonical_motion(*s["raw"], s["aspect"], s["dt"])
            if m is not None:
                out[lid] = {"name": name, **m, "source": s["file"]}
                noses.append(m["nose"])
                shoulders.append(m["shoulder"])
                break
    lib = {"signs": out, "head": np.median(np.stack(noses), 0), "shoulder": np.median(np.stack(shoulders), 0)}
    save_library(lib)
    return lib


def save_library(lib):
    MODELS_DIR.mkdir(exist_ok=True)
    data = {"head": lib["head"], "shoulder": lib["shoulder"], "ids": np.array(sorted(lib["signs"]))}
    for lid, m in lib["signs"].items():
        data[f"name_{lid}"] = np.array(m["name"])
        data[f"elbow_{lid}"] = m["elbow"]
        data[f"hand_{lid}"] = m["hand"]
        data[f"dt_{lid}"] = np.array(m["dt"])
        data[f"used_{lid}"] = m["used"]
        data[f"source_{lid}"] = np.array(m.get("source", ""))
    np.savez_compressed(LIB_PATH, **data)


def load_library(rebuild=False):
    if rebuild or not LIB_PATH.exists():
        print("Building avatar library from the landmark files (first time only) ...")
        return build_library()
    d = np.load(LIB_PATH, allow_pickle=False)
    signs = {}
    for lid in d["ids"]:
        lid = int(lid)
        signs[lid] = {"name": str(d[f"name_{lid}"]), "elbow": d[f"elbow_{lid}"], "hand": d[f"hand_{lid}"],
                      "dt": float(d[f"dt_{lid}"]), "used": d[f"used_{lid}"],
                      "source": str(d[f"source_{lid}"]) if f"source_{lid}" in d.files else ""}
    return {"signs": signs, "head": d["head"], "shoulder": d["shoulder"]}


# ============================================================================ text -> signs
ALIASES = {
    "hi": "Hello", "hey": "Hello", "hello": "Hello", "namaste": "Hello",
    "thanks": "Thank you", "thank you": "Thank you", "thankyou": "Thank you", "thank": "Thank you",
    "fine": "Alright", "ok": "Alright", "okay": "Alright", "alright": "Alright", "all right": "Alright",
    "morning": "Good Morning", "good morning": "Good Morning",
    "afternoon": "Good afternoon", "good afternoon": "Good afternoon",
    "evening": "Good evening", "good evening": "Good evening",
    "night": "Good night", "good night": "Good night",
    "pleased": "Pleased", "nice to meet you": "Pleased", "pleased to meet you": "Pleased",
    "how are you": "How are you", "how r u": "How are you",
    "i": "I", "me": "I", "you": "you", "he": "he", "she": "she", "it": "it", "we": "we", "they": "they",
    "you all": "you (plural)", "you guys": "you (plural)", "all of you": "you (plural)",
}


def text_to_signs(text, lib):
    """'Hello, how are you?' -> (['Hello', 'How are you'], [])  longest match first."""
    by_name = {m["name"].lower(): m["name"] for m in lib["signs"].values()}
    vocab = {**ALIASES, **by_name}
    vocab = {k: v for k, v in vocab.items() if v in by_name.values()}
    words = re.findall(r"[a-z0-9']+", text.lower())
    found, unknown, i = [], [], 0
    while i < len(words):
        for n in (4, 3, 2, 1):
            phrase = " ".join(words[i:i + n])
            if len(words[i:i + n]) == n and phrase in vocab:
                found.append(vocab[phrase])
                i += n
                break
        else:
            unknown.append(words[i])
            i += 1
    return found, unknown


# ============================================================================ motion for a sentence
def sign_arrays(lib, name):
    lid = next(k for k, m in lib["signs"].items() if m["name"] == name)
    m = lib["signs"][lid]
    n = max(2, int(round(len(m["elbow"]) * m["dt"] * FPS)))              # frames at 30 fps
    pos = np.linspace(0, len(m["elbow"]) - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(m["elbow"]) - 1)
    w = (pos - i0).reshape(-1, 1, 1)
    elbow = m["elbow"][i0] * (1 - w) + m["elbow"][i1] * w
    hand = m["hand"][i0] * (1 - w[..., None]) + m["hand"][i1] * w[..., None]
    return elbow.astype(np.float32), hand.astype(np.float32)


def rest_pose(lib):
    """Both hands rest in front of the body (visible, above the caption bar)."""
    sh = lib["shoulder"]
    elbow = np.stack([sh[0] + [0.08, 0.95], sh[1] + [-0.08, 0.95]]).astype(np.float32)
    hand = np.stack([elbow[a] + [0.0, 0.55] + RELAXED for a in range(2)]).astype(np.float32)
    return elbow, hand


def sentence_motion(lib, names):
    """Frames (elbow, hand) for the whole sentence + which word is shown in every frame."""
    r_e, r_h = rest_pose(lib)
    E, H_, who = [r_e[None].repeat(8, 0)], [r_h[None].repeat(8, 0)], [-1] * 8
    for k, nm in enumerate(names):
        e, h = sign_arrays(lib, nm)
        tr = 8                                                           # smooth transition
        a = np.linspace(0, 1, tr + 2)[1:-1].reshape(-1, 1, 1)
        E.append(E[-1][-1:] * (1 - a) + e[:1] * a)
        H_.append(H_[-1][-1:] * (1 - a[..., None]) + h[:1] * a[..., None])
        who += [k] * tr
        E.append(e)
        H_.append(h)
        who += [k] * len(e)
    tr = 10
    a = np.linspace(0, 1, tr + 2)[1:-1].reshape(-1, 1, 1)
    E.append(E[-1][-1:] * (1 - a) + r_e[None] * a)
    H_.append(H_[-1][-1:] * (1 - a[..., None]) + r_h[None] * a[..., None])
    E.append(r_e[None].repeat(10, 0))
    H_.append(r_h[None].repeat(10, 0))
    who += [-1] * (tr + 10)
    return np.concatenate(E), np.concatenate(H_), who


# ============================================================================ drawing
_BG = None


def background():
    """Soft vertical gradient (built once)."""
    global _BG
    if _BG is None:
        top, bot = np.array(BG_TOP, np.float32), np.array(BG_BOT, np.float32)
        t = np.linspace(0, 1, H_BODY, dtype=np.float32)[:, None, None]
        body = np.broadcast_to(top * (1 - t) + bot * t, (H_BODY, W, 3)).astype(np.uint8)
        img = np.zeros((H, W, 3), np.uint8)
        img[:H_BODY] = body
        img[H_BODY:] = (35, 35, 35)
        _BG = img
    return _BG.copy()


def px(p):
    return int(round(CX + float(p[0]) * SW)), int(round(CY + float(p[1]) * SW))


def capsule(img, a, b, th, color):
    th = max(int(th), 2)
    cv2.line(img, a, b, color, th, cv2.LINE_AA)
    cv2.circle(img, a, th // 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, b, th // 2, color, -1, cv2.LINE_AA)


def shade_sides(img, mask, strength=0.30):
    """Darken the left / right edges of a masked region a little -> looks rounded instead of flat."""
    m = mask.astype(bool)
    if not m.any():
        return
    xs = np.arange(W, dtype=np.float32)
    k = 1.0 - strength * np.clip(np.abs(xs - CX) / (0.95 * SW), 0, 1) ** 2
    cols = np.where(m)[1]
    img[m] = (img[m].astype(np.float32) * k[cols][:, None]).astype(np.uint8)


def draw_hand(img, pts):
    """Hand with 5 separate, tapered fingers (each with its own outline), palm and nails."""
    wr = pts[0]
    pts = wr + (pts - wr) * HAND_SCALE
    P = [px(p) for p in pts]

    # palm: outline first, then skin
    palm = np.array([P[i] for i in PALM], np.int32)
    cv2.polylines(img, [palm], True, SKIN_D, 6, cv2.LINE_AA)
    cv2.fillPoly(img, [palm], SKIN, cv2.LINE_AA)
    # thumb base (wrist -> thumb root)
    capsule(img, P[0], P[1], 0.105 * SW + 4, SKIN_D)
    capsule(img, P[0], P[1], 0.105 * SW, SKIN)

    # fingers one by one: pinky, ring, middle, index, thumb in front
    order = [FINGERS[4], FINGERS[3], FINGERS[2], FINGERS[1], FINGERS[0]]
    for f in order:
        thumb = f[0] == 1
        base, tip = (0.092, 0.066) if thumb else (0.074, 0.054)
        ths = np.linspace(base, tip, len(f) - 1) * SW
        segs = list(zip(f, f[1:]))
        for (u, v), t in zip(segs, ths):                # dark outline of this finger
            capsule(img, P[u], P[v], t + 4, SKIN_D)
        for (u, v), t in zip(segs, ths):                # skin of this finger
            capsule(img, P[u], P[v], t, SKIN)
        for j, t in zip(f[1:-1], ths[1:]):              # small knuckle crease on the joints
            cv2.circle(img, P[j], max(1, int(t * 0.13)), SKIN_D, -1, cv2.LINE_AA)
        cv2.circle(img, P[f[-1]], max(2, int(ths[-1] * 0.30)), NAIL, -1, cv2.LINE_AA)   # nail


# ---- facial expressions (hand-designed per sign; the dataset has no face data) --------------------
# each value: (eyebrows raised 0..1, smile 0..1, eyes open 0.3..1)
EXPR_NEUTRAL = np.array([0.0, 0.45, 1.0], np.float32)
EXPRESSIONS = {
    "hello": (0.3, 1.0, 1.0), "thank you": (0.2, 0.9, 0.85), "pleased": (0.3, 1.0, 0.7),
    "how are you": (1.0, 0.3, 1.0),              # raised eyebrows = question (as in ISL grammar)
    "good morning": (0.3, 0.9, 1.0), "good afternoon": (0.2, 0.8, 1.0),
    "good evening": (0.2, 0.7, 0.9), "good night": (0.0, 0.5, 0.35),
    "alright": (0.5, 0.6, 1.0),
}
_expr = EXPR_NEUTRAL.copy()


def current_expression(words, current):
    """Expression for the word being signed now; eases smoothly from one word to the next."""
    global _expr
    target = EXPR_NEUTRAL
    if current >= 0 and words:
        target = np.array(EXPRESSIONS.get(words[current].lower(), tuple(EXPR_NEUTRAL)), np.float32)
    _expr = _expr + (target - _expr) * 0.18
    return _expr


def draw_face(img, hx, hy, frame_no, expr=EXPR_NEUTRAL):
    # hair behind the head
    cv2.ellipse(img, px((hx, hy - 0.05)), (int(0.40 * SW), int(0.42 * SW)), 0, 0, 360, HAIR, -1, cv2.LINE_AA)
    # ears
    for sx in (-1, 1):
        e = px((hx + sx * 0.315, hy + 0.03))
        cv2.ellipse(img, e, (int(0.065 * SW), int(0.09 * SW)), 0, 0, 360, SKIN_D, -1, cv2.LINE_AA)
        cv2.ellipse(img, e, (int(0.045 * SW), int(0.07 * SW)), 0, 0, 360, SKIN, -1, cv2.LINE_AA)
    # face
    fc = px((hx, hy + 0.01))
    ax, ay = int(0.31 * SW), int(0.35 * SW)
    cv2.ellipse(img, fc, (ax + 2, ay + 2), 0, 0, 360, SKIN_D, -1, cv2.LINE_AA)
    cv2.ellipse(img, fc, (ax, ay), 0, 0, 360, SKIN, -1, cv2.LINE_AA)
    # fringe (hair over the forehead, slightly swept to one side)
    ang = np.radians(np.linspace(186, 354, 28))
    top = [(fc[0] + (ax + 1) * np.cos(a), fc[1] + (ay + 1) * np.sin(a)) for a in ang]
    low = [(fc[0] + ax * 0.98 * np.cos(a), fc[1] + ay * np.sin(a) * 0.78 + 0.05 * SW * np.cos(a)) for a in ang]
    cv2.fillPoly(img, [np.array(top + low[::-1], np.int32)], HAIR, cv2.LINE_AA)
    # eyes + eyebrows (expr = brow raise, smile, eye openness)
    brow, smile, eye = float(expr[0]), float(expr[1]), float(expr[2])
    blink = (frame_no % 100) < 3
    for sx in (-1, 1):
        c = px((hx + sx * 0.125, hy - 0.04))
        ry = max(1, int(round(6 * eye)))
        if blink or ry <= 2:
            cv2.line(img, (c[0] - 7, c[1]), (c[0] + 7, c[1]), HAIR, 2, cv2.LINE_AA)
        else:
            cv2.ellipse(img, c, (8, ry), 0, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.ellipse(img, c, (8, ry), 0, 0, 360, SKIN_D, 1, cv2.LINE_AA)
            cv2.circle(img, c, min(4, ry - 1), (60, 45, 35), -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] - 1, c[1] - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)
        lift = int(round(brow * 9))
        cv2.line(img, (c[0] - 10, c[1] - 13 - lift + int(brow * 2) * (-sx)),
                 (c[0] + 10, c[1] - 14 - lift + sx * 2 - int(brow * 2) * (-sx)), HAIR, 3, cv2.LINE_AA)
    # nose + mouth (wider and deeper smile when smile is high)
    nose = px((hx, hy + 0.075))
    cv2.ellipse(img, nose, (int(0.035 * SW), int(0.025 * SW)), 0, 10, 170, SKIN_D, 2, cv2.LINE_AA)
    mw, mh = int((0.06 + 0.05 * smile) * SW), int((0.02 + 0.06 * smile) * SW)
    cv2.ellipse(img, px((hx, hy + 0.15)), (mw, mh), 0, 15, 165, LIP, 3, cv2.LINE_AA)


def draw_avatar(lib, elbow, hand, label="", words=None, current=-1, frame_no=0, note=""):
    img = background()
    sh = lib["shoulder"]
    S = [px(sh[0]), px(sh[1])]
    hx, hy = float(lib["head"][0]), float(lib["head"][1])

    # torso (with soft side shading), shoulders, V-neck collar
    torso = np.array([px((-0.66, -0.06)), px((0.66, -0.06)), px((0.92, 3.0)), px((-0.92, 3.0))], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(img, [torso], SHIRT, cv2.LINE_AA)
    cv2.fillPoly(mask, [torso], 1)
    for s in S:
        cv2.circle(img, s, int(0.17 * SW), SHIRT, -1, cv2.LINE_AA)
        cv2.circle(mask, s, int(0.17 * SW), 1, -1)
    mask[H_BODY:] = 0
    shade_sides(img, mask)
    vneck = np.array([px((hx - 0.24, -0.07)), px((hx + 0.24, -0.07)), px((hx, 0.32))], np.int32)
    cv2.fillPoly(img, [vneck], SKIN_M, cv2.LINE_AA)
    cv2.polylines(img, [vneck], True, SHIRT_D, 3, cv2.LINE_AA)
    # neck (shadowed) + chin shadow
    cv2.rectangle(img, px((hx - 0.12, hy + 0.15)), px((hx + 0.12, -0.02)), SKIN_M, -1)

    # arms: sleeve (shirt) + forearm (skin), each with an outline
    for a in range(2):
        e, w = px(elbow[a]), px(hand[a, 0])
        capsule(img, S[a], e, 0.27 * SW + 4, SHIRT_D)
        capsule(img, e, w, 0.20 * SW + 4, SKIN_D)
    for a in range(2):
        e, w = px(elbow[a]), px(hand[a, 0])
        capsule(img, S[a], e, 0.27 * SW, SHIRT)
        capsule(img, e, w, 0.20 * SW, SKIN)

    draw_face(img, hx, hy, frame_no, current_expression(words, current))

    # hands on top of everything
    for a in range(2):
        draw_hand(img, hand[a])

    # caption bar (below the picture, does not hide the body)
    img[H_BODY:] = (35, 35, 35)
    if current >= 0 and words:
        cv2.putText(img, words[current], (14, H_BODY + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)
    x = 14
    for k, w in enumerate(words or []):
        col = (0, 215, 255) if k == current else (170, 170, 170)
        cv2.putText(img, w, (x, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        x += int(len(w) * 9.5) + 14
    if SHOW_HINT:
        cv2.putText(img, "p pause   SPACE replay   q close", (W - 262, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1, cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 200), 1, cv2.LINE_AA)
    return img


# ============================================================================ run
def show(img):
    """Fit the avatar into the window and keep it centred (works for any window size / full screen)."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect("ISL Avatar")
    except cv2.error:
        ww, wh = W, H
    if ww < 50 or wh < 50:
        ww, wh = W, H
    sc = min(ww / W, wh / H)
    r = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((wh, ww, 3), (35, 35, 35), np.uint8)
    y0, x0 = (wh - r.shape[0]) // 2, (ww - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    cv2.imshow("ISL Avatar", canvas)


def play(lib, names, note="", window=True, save=None, hold=False):
    E, H_, who = sentence_motion(lib, names)
    writer = None
    if window:
        cv2.namedWindow("ISL Avatar", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("ISL Avatar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    while True:
        t0 = time.time()
        for i in range(len(E)):
            img = draw_avatar(lib, E[i], H_[i], "", names, who[i], i, note)
            if writer is not None:
                writer.write(img)
            if window:
                show(img)
                wait = max(1, int((t0 + (i + 1) / FPS - time.time()) * 1000))
                k = cv2.waitKey(wait) & 0xFF
                if k in (ord("q"), 27):
                    return False
                if k == ord("p"):                      # pause / resume (handy for a screenshot)
                    t_pause = time.time()
                    while True:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 == ord("p"):
                            break
                        if k2 in (ord("q"), 27):
                            return False
                    t0 += time.time() - t_pause
        if writer is not None:
            writer.release()
            writer = None
            print(f"Video saved: {save}")
        if not window:
            return True
        if hold:
            last = draw_avatar(lib, E[-1], H_[-1], "", names, -1, len(E), "SPACE = replay    any other key = close")
            show(last)
            if cv2.waitKey(0) & 0xFF == ord(" "):
                continue
            return True
        k = cv2.waitKey(1200) & 0xFF
        if k == ord(" "):
            continue
        return True


def _load_source(rel):
    """The original recording an avatar motion was made from -> model input features."""
    d = np.load(ROOT / rel)
    if "aspect" in d.files:
        aspect = float(d["aspect"])
    else:
        ap = LM_DIR / "aspects.json"
        aspect = float(json.loads(ap.read_text()).get(str(d["video"]), 16 / 9)) if ap.exists() else 16 / 9
    return d["hands"], d["hand_present"], d["pose"], d["pose_present"], aspect


def roundtrip(lib):
    """Control experiment: does the recogniser name the ORIGINAL recording, and the AVATAR motion made from it?"""
    import torch
    import live_recognition as lr
    model, classes, delta, kind = lr.load_model(MODELS_DIR / "best.pt")

    def guess(x):
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(x[None]).float())[0], -1).numpy()
        return classes[int(p.argmax())], float(p.max())

    aspect, sw0, center = 16 / 9, 0.2, np.array([16 / 18, 0.45], np.float32)
    ids = [lid for lid in lib["signs"] if lib["signs"][lid]["name"] in classes]
    have_src = all(lib["signs"][lid].get("source") for lid in ids)
    if not have_src:
        print("(Run once with --rebuild to also test the original recordings as a control.)\n")
    ok_av, ok_or, rows = 0, 0, []
    print(f"  {'sign':<16}{'original recording':<26}{'avatar motion':<26}")
    for lid in ids:
        name = lib["signs"][lid]["name"]
        e, h = sign_arrays(lib, name)
        T = len(e)
        to_raw = lambda c: ((c * sw0 + center) / np.array([aspect, 1.0])).astype(np.float32)   # noqa: E731
        hands = np.zeros((T, 2, 21, 3), np.float32)
        pose = np.zeros((T, 33, 4), np.float32)
        for a in range(2):
            hands[:, a, :, :2] = to_raw(h[:, a])
            pose[:, 11 + a, :2] = to_raw(np.repeat(lib["shoulder"][a][None], T, 0))
            pose[:, 13 + a, :2] = to_raw(e[:, a])
            pose[:, 15 + a, :2] = to_raw(h[:, a, 0])
        pose[:, 0, :2] = to_raw(np.repeat(lib["head"][None], T, 0))
        pose[..., 3] = 1.0
        used = np.repeat(lib["signs"][lid]["used"][None], T, 0)
        hands[~used] = 0
        g_av, c_av = guess(fx.sequence_features(hands, used, pose, np.ones(T, bool), aspect, delta=delta))
        ok_av += g_av == name
        g_or, c_or = ("-", 0.0)
        if have_src:
            hh, hp, pp_, pose_p, asp = _load_source(lib["signs"][lid]["source"])
            g_or, c_or = guess(fx.sequence_features(hh, hp, pp_, pose_p, asp, delta=delta))
            ok_or += g_or == name
        rows.append((name, g_or, c_or, g_av, c_av))
        mark = lambda g: "ok " if g == name else "WRONG"          # noqa: E731
        orig = f"{g_or} {100 * c_or:.0f}% {mark(g_or)}" if have_src else "-"
        print(f"  {name:<16}{orig:<26}{g_av} {100 * c_av:.0f}% {mark(g_av)}")
    n = max(len(rows), 1)
    print()
    if have_src:
        print(f"Original recordings recognised : {ok_or}/{len(rows)} = {100 * ok_or / n:.0f}%")
    print(f"Avatar motion recognised       : {ok_av}/{len(rows)} = {100 * ok_av / n:.0f}%")
    if have_src:
        print("If both numbers are about the same, the avatar keeps the sign as well as the real recording does;")
        print("the remaining mistakes are the recogniser's (look-alike signs), not the avatar's.")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "avatar_roundtrip.csv", "w", encoding="utf-8") as fh:
        fh.write("sign,original_recognised_as,original_conf,avatar_recognised_as,avatar_conf\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]},{r[4]:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="", help="what the avatar should sign")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--save", default="", help="save the animation as an .mp4 file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the avatar library from the landmarks")
    ap.add_argument("--roundtrip", action="store_true")
    args = ap.parse_args()

    lib = load_library(args.rebuild)
    known = sorted(m["name"] for m in lib["signs"].values())
    if args.list:
        print("The avatar can sign:", ", ".join(known))
        return
    if args.roundtrip:
        roundtrip(lib)
        return

    def run_text(text):
        names, unknown = text_to_signs(text, lib)
        if unknown:
            print("Not in the vocabulary (skipped):", ", ".join(unknown))
        if not names:
            print("Nothing to sign. Words I know:", ", ".join(known))
            return True
        print("Signing:", " | ".join(names))
        return play(lib, names, ("skipped: " + ", ".join(unknown)) if unknown else "", save=args.save or None,
                    hold=not args.interactive)

    if args.interactive:
        print("Type a sentence and press Enter (empty line to stop). Known words:", ", ".join(known))
        while True:
            t = input("> ").strip()
            if not t or not run_text(t):
                break
    else:
        run_text(args.text or "Hello how are you")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
