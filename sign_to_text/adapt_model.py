"""
adapt_model.py  -  Step 6b: calibrate the system with YOUR samples (no retraining)

Run after record_samples.py, from the project root with (islenv) active:

    python sign_to_text\\adapt_model.py

Idea (for the examiner)
  The network turns every sign into a "fingerprint" (a point in space). Signs of the
  same kind land close together. So:
    * to adapt to YOU  -> move each sign's fingerprint towards the average of YOUR samples
    * to add a NEW sign -> its fingerprint is simply the average of a few samples
  Nothing is retrained, so it takes a second.

What this script does
  1. measures how accuracy grows with the number of your samples (k = 1, 2, 3 ...)
       - trained classifier, no calibration            (the starting point)
       - dataset fingerprints, no calibration
       - YOUR fingerprints only
       - dataset + YOUR fingerprints mixed
       - NEW signs (if you recorded any with --new)
  2. saves the table  results\\calibration.csv  and the graph  results\\calibration_curve.png
  3. saves your personal profile  models\\profile.npz  (live_recognition.py uses it automatically)

Honest note: the accuracy is measured on other takes recorded by you in the same session,
so it is more optimistic than a test on another day. Say this in the report.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as fx            # noqa: E402
import train_model as tm         # noqa: E402
import live_recognition as lr    # noqa: E402

ROOT = tm.ROOT
OWN_DIR = ROOT / "data" / "own"
RESULTS_DIR = ROOT / "results"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load_own():
    recs = []
    for f in sorted(OWN_DIR.rglob("*.npz")):
        d = np.load(f)
        recs.append({"hands": d["hands"], "hp": d["hand_present"], "pose": d["pose"], "pp": d["pose_present"],
                     "label_id": int(d["label_id"]), "label": str(d["label"]), "aspect": float(d["aspect"])})
    return recs


@torch.no_grad()
def embed(model, recs, delta):
    X = np.stack([fx.sequence_features(r["hands"], r["hp"], r["pose"], r["pp"], r["aspect"], delta=delta)
                  for r in recs])
    model.eval()
    return model.embed(torch.from_numpy(X).float()).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--repeats", type=int, default=40, help="random draws per k (more = smoother graph)")
    ap.add_argument("--max-k", type=int, default=5)
    args = ap.parse_args()

    ckpt_path = tm.MODELS_DIR / args.model
    if not ckpt_path.exists():
        sys.exit(f"{ckpt_path} not found. Run train_model.py first.")
    model, classes, delta, kind = lr.load_model(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_ids = list(ckpt["label_ids"])
    names = dict(zip(base_ids, classes))

    own = load_own()
    if not own:
        sys.exit("No samples in data\\own. Run record_samples.py first.")
    for r in own:
        names.setdefault(r["label_id"], r["label"])
    own_ids = np.array([r["label_id"] for r in own])
    counts = {int(i): int((own_ids == i).sum()) for i in sorted(set(own_ids))}
    print(f"Model: {kind}   your samples: {len(own)} for {len(counts)} signs")
    print("  " + ", ".join(f"{names[i]} {n}" for i, n in counts.items()))

    print("\nComputing fingerprints ...")
    recs, _, _ = tm.load_records()
    Zd = embed(model, recs, delta)
    yd = np.array([r["label_id"] for r in recs])
    proto_data = {lid: unit(Zd[yd == lid].mean(0)) for lid in base_ids if (yd == lid).any()}
    Zo = embed(model, own, delta)

    existing = [i for i in counts if i in base_ids]          # signs the model was trained on
    new = [i for i in counts if i not in base_ids]           # signs you added yourself
    all_ids = existing + new

    rows = []     # (method, k, accuracy, std)
    if existing:
        mask = np.isin(own_ids, existing)
        cols = [base_ids.index(i) for i in existing]
        with torch.no_grad():
            logits = model.scale * torch.from_numpy(Zo[mask]) @ F.normalize(model.W, dim=-1).T
        pred = np.array(existing)[logits.numpy()[:, cols].argmax(1)]
        rows.append(("Trained classifier (no calibration)", 0, float((pred == own_ids[mask]).mean()), 0.0))
        P = np.stack([proto_data[i] for i in existing])
        pred = np.array(existing)[(Zo[mask] @ P.T).argmax(1)]
        rows.append(("Dataset fingerprints (no calibration)", 0, float((pred == own_ids[mask]).mean()), 0.0))

    kmax = min(args.max_k, min(counts.values()) - 1)
    if kmax < 1:
        print("\nNeed at least 2 samples per sign to measure the curve (record more with record_samples.py).")
    rng = np.random.default_rng(0)
    for k in range(1, kmax + 1):
        res = {"own": [], "mix": [], "new": []}
        a = k / (k + 1)
        for _ in range(args.repeats):
            sup, qry = {}, {}
            for i in all_ids:
                perm = rng.permutation(np.where(own_ids == i)[0])
                sup[i], qry[i] = perm[:k], perm[k:]
            own_p = {i: unit(Zo[sup[i]].mean(0)) for i in all_ids}
            mix_p = {i: unit((1 - a) * proto_data[i] + a * own_p[i]) if i in proto_data else own_p[i] for i in all_ids}
            if existing:
                q = np.concatenate([qry[i] for i in existing])
                for key, protos in (("own", own_p), ("mix", mix_p)):
                    P = np.stack([protos[i] for i in existing])
                    res[key].append(float((np.array(existing)[(Zo[q] @ P.T).argmax(1)] == own_ids[q]).mean()))
            if new:
                q = np.concatenate([qry[i] for i in new])
                P = np.stack([own_p[i] for i in all_ids])
                res["new"].append(float((np.array(all_ids)[(Zo[q] @ P.T).argmax(1)] == own_ids[q]).mean()))
        label = {"own": "YOUR fingerprints only", "mix": "Dataset + YOUR fingerprints", "new": "NEW signs you added"}
        for key in ("own", "mix", "new"):
            if res[key]:
                rows.append((label[key], k, float(np.mean(res[key])), float(np.std(res[key]))))

    # ---- table
    methods = list(dict.fromkeys(r[0] for r in rows))
    ks = sorted({r[1] for r in rows if r[1] > 0})
    print("\n=========== ACCURACY ON YOUR SAMPLES ===========")
    print(f"{'method':<40}" + "".join(f"{('k=' + str(k)):>8}" for k in ks))
    for m in methods:
        line = f"{m:<40}"
        for k in ks:
            v = [r[2] for r in rows if r[0] == m and (r[1] == k or r[1] == 0)]
            line += f"{100 * v[0]:>7.0f}%" if v else f"{'-':>8}"
        print(line)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "calibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "k_own_samples", "accuracy_%", "std_%"])
        for m, k, acc, sd in rows:
            w.writerow([m, k, round(100 * acc, 1), round(100 * sd, 1)])

    if ks:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for m in methods:
            pts = [(r[1], r[2]) for r in rows if r[0] == m]
            if pts[0][0] == 0 and len(pts) == 1:
                ax.axhline(100 * pts[0][1], ls="--", lw=1.2, label=m)
            else:
                ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], "o-", label=m)
        ax.set_xlabel("number of YOUR samples per sign (k)")
        ax.set_ylabel("accuracy %")
        ax.set_ylim(0, 105)
        ax.set_xticks(ks)
        ax.set_title("Calibration: accuracy vs number of your samples")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "calibration_curve.png", dpi=140)
        plt.close(fig)
        print(f"\nSaved: results\\calibration.csv and results\\calibration_curve.png")

    # ---- your personal profile (uses ALL your samples)
    pnames, protos = [], []
    for lid in sorted(set(base_ids) | set(new)):
        n = counts.get(lid, 0)
        if n:
            mine = unit(Zo[own_ids == lid].mean(0))
            a = n / (n + 1)
            v = unit((1 - a) * proto_data[lid] + a * mine) if lid in proto_data else mine
        else:
            v = proto_data[lid]
        pnames.append(names[lid])
        protos.append(v)
    np.savez(tm.MODELS_DIR / "profile.npz", names=np.array(pnames), protos=np.stack(protos).astype(np.float32))
    print(f"Saved your profile: models\\profile.npz  ({len(pnames)} signs"
          + (f", {len(new)} new" if new else "") + ")")
    print("\nNext:  python sign_to_text\\live_recognition.py     (it uses your profile automatically)")


if __name__ == "__main__":
    main()
