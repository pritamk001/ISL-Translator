"""
train_model.py  -  Step 4: teach the computer to recognise signs from landmarks

Run from the project root (isl-translator) with the (islenv) environment active.

  python sign_to_text\\train_model.py                 # baseline + BiGRU + Transformer, compared
  python sign_to_text\\train_model.py --model gru     # only one model
  python sign_to_text\\train_model.py --cv            # 3-fold cross-validation (more reliable, slower)
  python sign_to_text\\train_model.py --epochs 10     # quick smoke test

What it does
  1. loads data\\landmarks\\*.npz  (made by extract_landmarks.py)
  2. splits videos into train / test with a GROUP split: takes recorded back-to-back
     (same session, consecutive MVI numbers) always stay together, so near-identical
     takes never sit in both train and test
  3. trains 3 models:  baseline (logistic regression), BiGRU, Transformer
  4. reports accuracy, confusion matrix, speed;  saves everything in  results\\
  5. trains a final copy of each neural model on ALL data and saves it in  models\\

Neural models use a cosine classifier: every sign becomes a point ("fingerprint")
in an embedding space. That is what later lets us add a NEW sign from just a few
examples, without retraining.
"""

import argparse
import copy
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LM_DIR = ROOT / "data" / "landmarks"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
ASPECT_CACHE = LM_DIR / "aspects.json"

EMB, HIDDEN, D_MODEL = 64, 96, 64
N_FOLDS = 3


# ----------------------------------------------------------------------------- data
def mvi_number(video):
    nums = re.findall(r"\d+", Path(video).stem)
    return int(nums[-1]) if nums else 0


def load_aspects(videos):
    """Width/height of every source video (read once from its first frame, then cached)."""
    cache = json.loads(ASPECT_CACHE.read_text()) if ASPECT_CACHE.exists() else {}
    changed, missing = False, 0
    for n, v in enumerate(videos):
        if v in cache:
            continue
        cap = cv2.VideoCapture(str(ROOT / v))
        ok, frame = cap.read()
        cap.release()
        if ok:
            cache[v] = frame.shape[1] / frame.shape[0]
        else:
            cache[v] = 16 / 9
            missing += 1
        changed = True
        if n % 50 == 0:
            print(f"  reading video shapes {n}/{len(videos)} ...")
    if changed:
        ASPECT_CACHE.write_text(json.dumps(cache))
    if missing:
        print(f"  (note: {missing} videos not found, assumed 16:9)")
    return cache


def load_records():
    files = sorted(LM_DIR.rglob("*.npz"))
    if not files:
        sys.exit(f"No .npz files in {LM_DIR}. Run extract_landmarks.py first.")
    raw = []
    for f in files:
        d = np.load(f)
        raw.append({
            "hands": d["hands"], "hp": d["hand_present"], "pose": d["pose"],
            "pp": d["pose_present"], "label_id": int(d["label_id"]),
            "label": str(d["label"]), "video": str(d["video"]),
        })
    aspects = load_aspects([r["video"] for r in raw])
    label_ids = sorted({r["label_id"] for r in raw})
    names = {r["label_id"]: r["label"] for r in raw}
    for r in raw:
        r["y"] = label_ids.index(r["label_id"])
        r["num"] = mvi_number(r["video"])
        r["aspect"] = float(aspects[r["video"]])
    return raw, label_ids, [names[i] for i in label_ids]


def assign_folds(recs, k=N_FOLDS, seed=0, gap=3):
    """Fold number for every video. Back-to-back takes (consecutive MVI numbers) stay together."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(recs), int)
    by_class = {}
    for i, r in enumerate(recs):
        by_class.setdefault(r["y"], []).append(i)
    for idxs in by_class.values():
        idxs = sorted(idxs, key=lambda i: recs[i]["num"])
        groups, cur = [], [idxs[0]]
        for a, b in zip(idxs, idxs[1:]):
            if recs[b]["num"] - recs[a]["num"] > gap:
                groups.append(cur)
                cur = []
            cur.append(b)
        groups.append(cur)
        if len(groups) < k:                       # too few sessions -> split per video
            groups = [[i] for i in idxs]
        for j, gi in enumerate(rng.permutation(len(groups))):
            for i in groups[gi]:
                fold[i] = j % k
    return fold


def clean_features(r, delta):
    x = fx.sequence_features(r["hands"], r["hp"], r["pose"], r["pp"], r["aspect"], delta=delta)
    return x


def augment_features(r, rng, mirror, delta):
    """Random but realistic variations of one training sign."""
    hands, hp, pose, pp = r["hands"], r["hp"], r["pose"], r["pp"]
    T = hands.shape[0]
    cut = max(1, int(0.12 * T))                                   # 1) trim start / end
    a, b = int(rng.integers(0, cut + 1)), T - int(rng.integers(0, cut + 1))
    if b - a >= 8:
        hands, hp, pose, pp = hands[a:b], hp[a:b], pose[a:b], pp[a:b]
    aspect = r["aspect"]
    h, p = fx.to_isotropic(hands, pose, aspect)
    s = rng.uniform(0.92, 1.08)                                   # 2) tiny zoom + tilt
    th = np.deg2rad(rng.uniform(-8, 8))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32) * s
    centre = np.array([aspect / 2, 0.5], np.float32)
    for arr in (h, p):
        arr[..., :2] = (arr[..., :2] - centre) @ R.T + centre
        arr[..., 2] *= s
        arr[..., :2] += rng.normal(0, 0.002, arr[..., :2].shape).astype(np.float32)   # 3) jitter
    if mirror and rng.random() < 0.5:                             # 4) left-handed signer
        h[..., 0] = aspect - h[..., 0]
        p[..., 0] = aspect - p[..., 0]
        h, hp = np.ascontiguousarray(h[:, ::-1]), np.ascontiguousarray(hp[:, ::-1])
        for u, v in ((11, 12), (13, 14), (15, 16)):
            p[:, [u, v]] = p[:, [v, u]]
    x = fx.resample(fx.frame_features(h, hp, p, pp), fx.T_FIXED)
    return fx.add_delta(x) if delta else x


# ----------------------------------------------------------------------------- models
class GRUEncoder(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, emb=EMB, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, emb)

    def forward(self, x):
        h, _ = self.gru(self.proj(self.norm(x)))
        h = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        return self.out(self.drop(h))


class TransformerEncoderNet(nn.Module):
    def __init__(self, in_dim, d_model=D_MODEL, emb=EMB, T=fx.T_FIXED, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, T, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, emb)

    def forward(self, x):
        h = self.enc(self.proj(self.norm(x)) + self.pos)
        return self.out(h.mean(1))


class SignNet(nn.Module):
    """encoder -> embedding ("fingerprint") -> cosine classifier."""

    def __init__(self, kind, in_dim, n_classes, scale=16.0):
        super().__init__()
        self.kind = kind
        self.encoder = GRUEncoder(in_dim) if kind == "gru" else TransformerEncoderNet(in_dim)
        self.W = nn.Parameter(torch.randn(n_classes, EMB) * 0.1)
        self.scale = scale

    def embed(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x):
        return self.scale * self.embed(x) @ F.normalize(self.W, dim=-1).T


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_logits(model, X):
    model.eval()
    return model(torch.from_numpy(np.stack(X)).float()).numpy()


@torch.no_grad()
def prototype_logits(model, X_support, y_support, X_query, n_classes):
    """Classify by distance to the average fingerprint of every sign."""
    model.eval()
    zs = model.embed(torch.from_numpy(np.stack(X_support)).float())
    zq = model.embed(torch.from_numpy(np.stack(X_query)).float())
    protos = torch.stack([zs[torch.as_tensor(y_support) == c].mean(0) for c in range(n_classes)])
    return (zq @ F.normalize(protos, dim=-1).T).numpy()


def topk_acc(logits, y, k=3):
    top = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([yy in t for yy, t in zip(y, top)]))


def train_neural(kind, recs, clean, train_idx, eval_idx, n_classes, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SignNet(kind, clean[0].shape[1], n_classes)
    lr = 3e-3 if kind == "gru" else 1.5e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist = {"loss": [], "eval_epoch": [], "eval_acc": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total = 0.0
        for s in range(0, len(perm), 32):
            batch = perm[s:s + 32]
            xb = torch.from_numpy(np.stack([augment_features(recs[i], rng, args.mirror, args.delta)
                                            for i in batch])).float()
            yb = torch.tensor([recs[i]["y"] for i in batch])
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        hist["loss"].append(total / len(perm))
        if eval_idx is not None and (ep % 5 == 4 or ep == args.epochs - 1):
            lg = predict_logits(model, [clean[i] for i in eval_idx])
            acc = float(np.mean(lg.argmax(1) == np.array([recs[i]["y"] for i in eval_idx])))
            hist["eval_epoch"].append(ep + 1)
            hist["eval_acc"].append(acc)
            print(f"    [{kind}] epoch {ep + 1:>3}/{args.epochs}  loss {hist['loss'][-1]:.3f}  "
                  f"test acc {acc * 100:.1f}%  ({time.time() - t0:.0f}s)")
        elif ep % 10 == 9:
            print(f"    [{kind}] epoch {ep + 1:>3}/{args.epochs}  loss {hist['loss'][-1]:.3f}  "
                  f"({time.time() - t0:.0f}s)")
    return model, hist, time.time() - t0


def run_baseline(clean, recs, train_idx, eval_idx):
    X = np.stack([c.reshape(-1) for c in clean])
    y = np.array([r["y"] for r in recs])
    t0 = time.time()
    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=2000, C=0.05).fit(scaler.transform(X[train_idx]), y[train_idx])
    train_s = time.time() - t0
    proba = np.zeros((len(eval_idx), int(y.max()) + 1))
    proba[:, clf.classes_] = clf.predict_proba(scaler.transform(X[eval_idx]))
    return clf, scaler, proba, train_s


def latency_ms(fn, n=40):
    for _ in range(5):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


# ----------------------------------------------------------------------------- plots
def plot_confusion(cm, names, title, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("Actual sign")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(hist, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(hist["loss"]) + 1), hist["loss"], label="training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(hist["eval_epoch"], [a * 100 for a in hist["eval_acc"]], "r.-", label="test accuracy %")
    ax2.set_ylabel("accuracy %")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "baseline", "gru", "transformer"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cv", action="store_true", help="evaluate on all 3 folds instead of one")
    ap.add_argument("--mirror", action="store_true", help="mirror augmentation (left-handed signers)")
    ap.add_argument("--delta", action="store_true", help="add motion (frame difference) features")
    ap.add_argument("--no-final", action="store_true", help="skip training the final all-data models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading landmarks ...")
    recs, label_ids, names = load_records()
    n_classes = len(names)
    clean = [clean_features(r, args.delta) for r in recs]
    y_all = np.array([r["y"] for r in recs])
    folds = assign_folds(recs, N_FOLDS, args.seed)

    print(f"\nVideos: {len(recs)}   signs: {n_classes}   feature size: {clean[0].shape}")
    print(f"{'sign':<16}{'videos':>7}{'frames':>8}{'hand%':>7}")
    for c, nm in enumerate(names):
        rs = [r for r in recs if r["y"] == c]
        print(f"{nm:<16}{len(rs):>7}{np.mean([len(r['hp']) for r in rs]):>8.0f}"
              f"{100 * np.mean([r['hp'].any(1).mean() for r in rs]):>7.0f}")
    print(f"\nFold sizes (videos): {[int((folds == k).sum()) for k in range(N_FOLDS)]}  "
          f"-> {'3-fold cross-validation' if args.cv else 'one split: train on 2 folds, test on 1'}")

    kinds = ["baseline", "gru", "transformer"] if args.model == "all" else [args.model]
    eval_folds = list(range(N_FOLDS)) if args.cv else [0]
    summary = []

    for kind in kinds:
        print(f"\n===== {kind} =====")
        accs, tops, protos, preds, trues, lat, secs, hist0, nparams = [], [], [], [], [], 0, 0, None, 0
        for k in eval_folds:
            te = [i for i in range(len(recs)) if folds[i] == k]
            tr = [i for i in range(len(recs)) if folds[i] != k]
            print(f"  fold {k}: train {len(tr)} videos, test {len(te)} videos")
            yte = y_all[te]
            if kind == "baseline":
                clf, scaler, logits, sec = run_baseline(clean, recs, tr, te)
                x1 = scaler.transform(clean[te[0]].reshape(1, -1))
                lat = latency_ms(lambda: clf.predict_proba(x1))
                nparams = clf.coef_.size + clf.intercept_.size
                pr = float("nan")
            else:
                model, hist, sec = train_neural(kind, recs, clean, tr, te, n_classes, args, args.seed + k)
                logits = predict_logits(model, [clean[i] for i in te])
                pl = prototype_logits(model, [clean[i] for i in tr], y_all[tr], [clean[i] for i in te], n_classes)
                pr = float(np.mean(pl.argmax(1) == yte))
                x1 = torch.from_numpy(clean[te[0]][None]).float()
                with torch.no_grad():
                    model.eval()
                    lat = latency_ms(lambda: model(x1))
                nparams = count_params(model)
                if hist0 is None:
                    hist0 = hist
            acc = float(np.mean(logits.argmax(1) == yte))
            accs.append(acc)
            tops.append(topk_acc(logits, yte))
            protos.append(pr)
            preds += list(logits.argmax(1))
            trues += list(yte)
            secs += sec
            print(f"  fold {k}: accuracy {acc * 100:.1f}%   top-3 {tops[-1] * 100:.1f}%"
                  + ("" if kind == "baseline" else f"   fingerprint-matching accuracy {pr * 100:.1f}%"))

        cm = confusion_matrix(trues, preds, labels=list(range(n_classes)))
        plot_confusion(cm, names, f"{kind}: confusion matrix ({'3-fold CV' if args.cv else 'test split'})",
                       RESULTS_DIR / f"confusion_{kind}.png")
        (RESULTS_DIR / f"report_{kind}.txt").write_text(
            classification_report(trues, preds, labels=list(range(n_classes)), target_names=names, zero_division=0))
        if hist0 is not None and hist0["eval_acc"]:
            plot_curves(hist0, f"{kind}: training curve", RESULTS_DIR / f"curve_{kind}.png")
        summary.append({
            "model": kind, "parameters": int(nparams),
            "test_accuracy_%": round(100 * float(np.mean(accs)), 1),
            "std_%": round(100 * float(np.std(accs)), 1) if len(accs) > 1 else 0.0,
            "top3_%": round(100 * float(np.mean(tops)), 1),
            "fingerprint_acc_%": "" if kind == "baseline" else round(100 * float(np.nanmean(protos)), 1),
            "latency_ms": round(lat, 2), "train_time_s": round(secs, 0),
        })

    # ---- comparison table
    print("\n================ MODEL COMPARISON ================")
    cols = list(summary[0].keys())
    print("  ".join(f"{c:<18}" for c in cols))
    for row in summary:
        print("  ".join(f"{str(row[c]):<18}" for c in cols))
    with open(RESULTS_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)

    # ---- final models on ALL data
    neural = [r for r in summary if r["model"] != "baseline"]
    if neural and not args.no_final:
        print("\nTraining final models on ALL videos (these are the ones the live app will use) ...")
        (MODELS_DIR / "classes.json").write_text(json.dumps({"label_ids": label_ids, "names": names}))
        for r in neural:
            kind = r["model"]
            model, _, _ = train_neural(kind, recs, clean, list(range(len(recs))), None, n_classes, args, args.seed)
            torch.save({"kind": kind, "state": model.state_dict(), "classes": names, "label_ids": label_ids,
                        "in_dim": clean[0].shape[1], "delta": args.delta, "T": fx.T_FIXED},
                       MODELS_DIR / f"{kind}.pt")
            print(f"  saved models\\{kind}.pt")
        best = max(neural, key=lambda r: r["test_accuracy_%"])["model"]
        torch.save(torch.load(MODELS_DIR / f"{best}.pt"), MODELS_DIR / "best.pt")
        print(f"  best neural model: {best}  ->  models\\best.pt")

    print(f"\nResults (pictures, tables) are in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
