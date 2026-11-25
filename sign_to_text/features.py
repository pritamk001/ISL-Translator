"""
features.py  -  shared by training, live camera recognition and the avatar.

Turns raw landmarks (hands + body pose) into a fixed-size array the model can read.
The SAME code must be used while training and while running live, otherwise the
model sees different numbers and fails. That is why it lives in its own file.

Raw input (what extract_landmarks.py saves):
  hands         (T, 2, 21, 3)   x, y, z of both hands (slot 0 / slot 1)
  hand_present  (T, 2)          True where that hand was detected
  pose          (T, 33, 4)      body points x, y, z, visibility
  pose_present  (T,)            True where the body was detected
  aspect        frame width / frame height  (x is stored as a fraction of width,
                                y as a fraction of height, so x must be stretched
                                by `aspect` to make distances comparable)

Output of sequence_features():  array of shape (T_FIXED, FEATURE_DIM)
"""

import numpy as np

T_FIXED = 32                      # every sign is resampled to 32 time steps
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]   # nose, shoulders, elbows, wrists
HAND_WRIST, HAND_MID_MCP = 0, 9

HAND_FEATS = 20 * 3 + 2 + 1       # 20 finger points (3), wrist position (2), present flag (1)
FEATURE_DIM = 2 * HAND_FEATS + len(POSE_KEEP) * 2     # = 140


def to_isotropic(hands, pose, aspect):
    """Stretch x (and z) by the aspect ratio so x and y use the same units."""
    hands = hands.astype(np.float32).copy()
    pose = pose.astype(np.float32).copy()
    hands[..., 0] *= aspect
    hands[..., 2] *= aspect
    pose[..., 0] *= aspect
    pose[..., 2] *= aspect
    return hands, pose


def _fill_missing(values, valid):
    """Copy the last valid value forward (and the first valid one backward)."""
    values = values.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return values
    last = None
    for t in range(len(values)):
        if valid[t]:
            last = values[t].copy()
        elif last is not None:
            values[t] = last
    values[: idx[0]] = values[idx[0]]
    return values


def frame_features(hands, hand_present, pose, pose_present):
    """Per-frame features from isotropic landmarks. Returns (T, FEATURE_DIM)."""
    T = hands.shape[0]

    # ---- body reference: centre = between shoulders, scale = shoulder width ----
    center = (pose[:, 11, :2] + pose[:, 12, :2]) / 2.0
    width = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    valid = pose_present & (width > 1e-3)
    center = _fill_missing(center, valid)
    scale = float(np.median(width[valid])) if valid.any() else 0.2

    # ---- hands ----
    wrist = hands[:, :, HAND_WRIST, :]                                   # (T, 2, 3)
    local = hands - wrist[:, :, None, :]                                 # (T, 2, 21, 3)
    size = np.linalg.norm(hands[:, :, HAND_MID_MCP, :] - wrist, axis=-1)  # (T, 2)
    size = np.where(size > 1e-4, size, 1.0)
    local = local / size[:, :, None, None]
    local = local[:, :, 1:, :].reshape(T, 2, 60)                         # drop the wrist itself
    wrist_pos = (wrist[:, :, :2] - center[:, None, :]) / scale           # (T, 2, 2)
    present = hand_present.astype(np.float32)                            # (T, 2)

    hand_feats = np.concatenate([local, wrist_pos, present[:, :, None]], axis=-1)   # (T, 2, 63)
    hand_feats = hand_feats * present[:, :, None]       # absent hand -> all zeros (flag too)
    hand_feats[:, :, -1] = present
    hand_feats = hand_feats.reshape(T, 2 * HAND_FEATS)

    # ---- upper body ----
    body = (pose[:, POSE_KEEP, :2] - center[:, None, :]) / scale         # (T, 7, 2)
    body = body.reshape(T, -1)
    if not valid.any():
        body[:] = 0.0

    out = np.concatenate([hand_feats, body], axis=1).astype(np.float32)
    return np.clip(out, -8.0, 8.0)


def resample(seq, length=T_FIXED):
    """Linear resampling of a (T_in, F) array to (length, F)."""
    t_in = seq.shape[0]
    if t_in == 1:
        return np.repeat(seq, length, axis=0)
    pos = np.linspace(0, t_in - 1, length)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t_in - 1)
    w = (pos - i0)[:, None].astype(np.float32)
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(np.float32)


def add_delta(x):
    """Append frame-to-frame change (motion) as extra features: (T, F) -> (T, 2F)."""
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return np.concatenate([x, d], axis=1)


def sequence_features(hands, hand_present, pose, pose_present, aspect, length=T_FIXED, delta=False):
    """Raw landmarks of one sign -> model input of shape (length, FEATURE_DIM [x2 if delta])."""
    h, p = to_isotropic(hands, pose, aspect)
    x = resample(frame_features(h, hand_present, p, pose_present), length)
    return add_delta(x) if delta else x
