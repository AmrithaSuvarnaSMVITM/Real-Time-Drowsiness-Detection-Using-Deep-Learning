"""
SUBJECT-WISE evaluation for the MRL Eye folder dataset (awake/sleepy).

The dataset's own train/val/test split was random per image, so the same 37 people
appear in all three splits. This script pools every image and re-splits BY PERSON,
so test people are never seen in training. Subject id = first part of the filename
(e.g. s0001_00001_0_0_0_0_0_01.png -> s0001).

  python train_subjectwise.py --data D:\\DrowsinessDetect\\data --out D:\\DrowsinessDetect\\results_subjectwise ^
      --models mobilenet_v3_small resnet18 --size 64 --epochs 3 --max_per_class 5000

Needs train_compare.py in the same folder. Label: sleepy = 1, awake = 0.
"""
import argparse
import re
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from train_compare import (IMG_EXT, MODELS, build_model, latency_ms,  # noqa: E402
                           load_images, metrics, predict, train_model)


def scan_all(root):
    rows = []
    for split in ("train", "val", "test"):
        for p in (Path(root) / split).rglob("*"):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel = str(p.relative_to(Path(root) / split)).lower()
            label = 1 if "sleepy" in rel else (0 if "awake" in rel else None)
            if label is None:
                continue
            parts = p.stem.split("_")
            subj = parts[0] if re.fullmatch(r"s\d+", parts[0]) else None
            fn_state = parts[4] if len(parts) > 4 and parts[4] in ("0", "1") else None
            rows.append((str(p), subj, label, fn_state))
    return pd.DataFrame(rows, columns=["path", "subject", "closed", "fn_state"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_subjectwise")
    ap.add_argument("--models", nargs="+", default=["mobilenet_v3_small", "resnet18"], choices=MODELS)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_per_class", type=int, default=5000, help="train cap per class; 0 = all")
    ap.add_argument("--max_val_per_class", type=int, default=2000, help="val cap per class; 0 = all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = scan_all(args.data)
    if df.empty:
        raise SystemExit("No images found. Check --data (it must contain train/ val/ test/).")
    if df.subject.isna().any():
        bad = df[df.subject.isna()].path.head(5).tolist()
        raise SystemExit("Could not read a subject id (like s0001) from some filenames, e.g.:\n  "
                         + "\n  ".join(bad) + "\nSubject-wise splitting needs the id in the filename.")
    chk = df[df.fn_state.notna()]
    if len(chk):
        agree = ((chk.closed == 1) == (chk.fn_state == "0")).mean()
        print(f"Folder label vs filename eye-state agreement: {agree * 100:.2f}% (should be ~100%)")
    print(f"Pooled {len(df)} images from {df.subject.nunique()} subjects")

    subs = sorted(df.subject.unique())
    random.Random(args.seed).shuffle(subs)
    n_te = max(1, round(0.15 * len(subs)))
    n_va = max(1, round(0.15 * len(subs)))
    te_s, va_s = set(subs[:n_te]), set(subs[n_te:n_te + n_va])
    dfs = {"test": df[df.subject.isin(te_s)],
           "val": df[df.subject.isin(va_s)],
           "train": df[~df.subject.isin(te_s | va_s)]}
    assert not (set(dfs["train"].subject) & set(dfs["test"].subject)), "subject overlap!"
    for s, d in dfs.items():
        print(f"{s}: {len(d)} images, {d.subject.nunique()} subjects, sleepy fraction {d.closed.mean():.2f}")
    json.dump({k: sorted(v.subject.unique()) for k, v in dfs.items()}, open(out / "subject_split.json", "w"), indent=2)

    if args.max_per_class > 0:
        dfs["train"] = pd.concat([g.sample(min(len(g), args.max_per_class), random_state=args.seed)
                                  for _, g in dfs["train"].groupby("closed")])
    if args.max_val_per_class > 0:
        dfs["val"] = pd.concat([g.sample(min(len(g), args.max_val_per_class), random_state=args.seed)
                                for _, g in dfs["val"].groupby("closed")])

    print("Loading images into RAM...")
    X, Y = {}, {}
    for s, d in dfs.items():
        X[s] = load_images(d.path.tolist(), args.size)
        Y[s] = d.closed.values.astype(np.int64)

    results, hists, cms = [], {}, {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        model, hist, mins = train_model(name, X["train"], Y["train"], X["val"], Y["val"], args, out)
        p = predict(model, X["test"])
        m = metrics(Y["test"], p)
        results.append(dict(model=name,
                            params_M=sum(q.numel() for q in model.parameters()) / 1e6,
                            size_MB=(out / f"{name}.pt").stat().st_size / 1e6,
                            test_acc=m["acc"], precision=m["precision"], recall=m["recall"],
                            f1=m["f1"], auc=m["auc"],
                            latency_ms=latency_ms(model, args.size), train_min=mins))
        hists[name] = hist
        cms[name] = confusion_matrix(Y["test"], (p >= 0.5).astype(int))
        pd.DataFrame(results).to_csv(out / "results.csv", index=False)

    res = pd.DataFrame(results).round(4)
    print("\n=============== FINAL COMPARISON (test set) ===============")
    print(res.to_string(index=False))
    json.dump(hists, open(out / "history.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(res))
    ax[0].bar(x - 0.2, res.test_acc, 0.4, label="Accuracy")
    ax[0].bar(x + 0.2, res.f1, 0.4, label="F1")
    ax[0].set_xticks(x, res.model, rotation=20)
    ax[0].set_ylim(max(0, res.test_acc.min() - 0.05), 1.0)
    ax[0].legend()
    ax[0].set_title("Test accuracy / F1")
    ax[1].bar(x, res.latency_ms, color="tab:orange")
    ax[1].set_xticks(x, res.model, rotation=20)
    ax[1].set_title("CPU latency per image (ms)")
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)

    fig, ax = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.2))
    ax = np.atleast_1d(ax)
    for a, (n_, cm) in zip(ax, cms.items()):
        a.imshow(cm, cmap="Blues")
        a.set_title(n_, fontsize=9)
        a.set_xticks([0, 1], ["awake", "sleepy"])
        a.set_yticks([0, 1], ["awake", "sleepy"])
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=150)
    print(f"\nDone. Results saved in {out.resolve()}")


if __name__ == "__main__":
    main()
