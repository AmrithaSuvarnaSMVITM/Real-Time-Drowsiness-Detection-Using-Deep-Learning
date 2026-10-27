"""
Train + compare models on a dataset laid out like:

  data/
    train/awake  train/sleepy
    val/awake    val/sleepy
    test/awake   test/sleepy

Label: sleepy = 1, awake = 0.   (Needs train_compare.py in the same folder.)

  python train_folders.py --data "C:/Users/YOU/Downloads/archive/data" --out D:/drowsy_out
"""
import argparse
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


def scan_split(root, split):
    rows = []
    for p in (Path(root) / split).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            rel = str(p.relative_to(Path(root) / split)).lower()
            if "sleepy" in rel:
                rows.append((str(p), 1))
            elif "awake" in rel:
                rows.append((str(p), 0))
    return pd.DataFrame(rows, columns=["path", "closed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = use all images")
    ap.add_argument("--max_val_per_class", type=int, default=2000,
                    help="cap val set per class (val runs every epoch, so a big val set is slow); 0 = all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dfs = {s: scan_split(args.data, s) for s in ("train", "val", "test")}
    for s, d in dfs.items():
        print(f"{s}: {len(d)} images  (sleepy fraction {d.closed.mean() if len(d) else 0:.2f})")
        if d.empty:
            raise SystemExit(f"No images found in {s}/ - check the --data path.")
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
