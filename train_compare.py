"""
Train + fine-tune + compare eye-state (open/closed) classifiers on CPU.

Label convention: closed = 1 (drowsy cue), open = 0.

Example:
  python train_compare.py --data D:/datasets/mrlEyes_2018_01 --out D:/drowsy_out
  python train_compare.py --data D:/datasets/mrlEyes_2018_01 --cross_root D:/datasets/CEW
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["small_cnn", "mobilenet_v3_small", "resnet18", "efficientnet_b0"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
CLOSED_KEYS = ("closed", "sleepy", "drowsy")
OPEN_KEYS = ("open", "awake", "alert")


# ----------------------------------------------------------------- models
class Wrap(nn.Module):
    """Takes 1-channel [0,1] images; handles channel repeat + ImageNet normalisation."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.net((x - self.mean) / self.std)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if name == "small_cnn":
        net = SmallCNN()
    elif name == "mobilenet_v3_small":
        net = tvm.mobilenet_v3_small(weights=w)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    elif name == "resnet18":
        net = tvm.resnet18(weights=w)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif name == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=w)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    else:
        raise ValueError(name)
    return Wrap(net)


# ------------------------------------------------------------------- data
def scan_mrl(root):
    """MRL filename: s0001_00001_<gender>_<glasses>_<eyeState>_<reflect>_<light>_<sensor>.png
    eyeState: 0 = closed, 1 = open."""
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            parts = p.stem.split("_")
            if len(parts) >= 5 and parts[0].startswith("s") and parts[4] in ("0", "1"):
                rows.append((str(p), parts[0], 1 - int(parts[4])))
    return pd.DataFrame(rows, columns=["path", "subject", "closed"])


def scan_folders(root):
    """Folder-labelled data: any path containing 'closed' -> 1, 'open' -> 0 (CEW, your own data)."""
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            rel = str(p.relative_to(root)).lower()
            if any(k in rel for k in CLOSED_KEYS):
                rows.append((str(p), 1))
            elif any(k in rel for k in OPEN_KEYS):
                rows.append((str(p), 0))
    return pd.DataFrame(rows, columns=["path", "closed"])


def cap(df, n, seed):
    if n <= 0:
        return df
    return pd.concat([g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby("closed")])


POS_WORDS = ("sleep", "drows", "closed", "fatigue", "tired")
NEG_WORDS = ("awake", "alert", "open", "active", "notdrowsy")


def scan_split(root, split):
    """root/<split>/<class>/*.jpg ; sleepy-like class -> 1, awake-like class -> 0."""
    rows = []
    d = Path(root) / split
    if not d.is_dir():
        return pd.DataFrame(columns=["path", "closed"])
    for p in d.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            cls = p.parent.name.lower()
            if any(w in cls for w in POS_WORDS):
                rows.append((str(p), 1))
            elif any(w in cls for w in NEG_WORDS):
                rows.append((str(p), 0))
    return pd.DataFrame(rows, columns=["path", "closed"])


def make_splits(args):
    root = Path(args.data)
    if (root / "train").is_dir() and (root / "test").is_dir():
        print("Layout: ready-made train/val/test folders")
        tr_df, va_df, te_df = (scan_split(root, s) for s in ("train", "val", "test"))
        if tr_df.empty or te_df.empty:
            raise SystemExit("Found train/test folders but no images with sleepy/awake class folders.")
        if va_df.empty:  # no val folder: hold out 15% of train
            va_df = tr_df.sample(frac=0.15, random_state=args.seed)
            tr_df = tr_df.drop(va_df.index)
        print(f"Images train/val/test: {len(tr_df)}/{len(va_df)}/{len(te_df)} | "
              f"sleepy ratio train {tr_df.closed.mean():.2f}")
        return (cap(tr_df, args.max_train_per_class, args.seed),
                cap(va_df, args.max_eval_per_class, args.seed),
                cap(te_df, args.max_eval_per_class, args.seed))


def load_images(paths, size):
    out = np.zeros((len(paths), 1, size, size), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            out[i, 0] = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return out


def augment(x):
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(3)
    b = (torch.rand(n, 1, 1, 1) - 0.5) * 0.4
    c = 0.7 + torch.rand(n, 1, 1, 1) * 0.6
    x = (x - 0.5) * c + 0.5 + b
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)


# --------------------------------------------------------------- evaluate
@torch.no_grad()
def predict(model, x, bs=256):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).float() / 255
        out.append(torch.softmax(model(xb), 1)[:, 1])
    return torch.cat(out).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return dict(acc=accuracy_score(y, pred), precision=pr, recall=rc, f1=f1, auc=auc)


def latency_ms(model, size, n=100):
    model.eval()
    x = torch.rand(1, 1, size, size)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t) / n * 1000


# ------------------------------------------------------------------ train
def train_model(name, xtr, ytr, xva, yva, args, out_dir):
    torch.manual_seed(args.seed)
    model = build_model(name)
    lr = 1e-3 if name == "small_cnn" else 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                    batch_size=args.bs, shuffle=True)
    best, hist, t0 = -1.0, [], time.time()
    ckpt = out_dir / f"{name}.pt"
    for ep in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in dl:
            xb = augment(xb.float() / 255)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        va = metrics(yva, predict(model, xva))
        hist.append(dict(epoch=ep + 1, train_loss=loss_sum / n, val_acc=va["acc"], val_f1=va["f1"]))
        print(f"  [{name}] epoch {ep + 1}/{args.epochs} loss {loss_sum / n:.4f} "
              f"val_acc {va['acc']:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if va["acc"] > best:
            best = va["acc"]
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model, hist, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="MRL Eye dataset root")
    ap.add_argument("--presplit", default=None, help="folder with train/ val/ test/, each with awake/ and sleepy/")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cross_root", default=None, help="folder-labelled dataset (CEW / own webcam data)")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_train_per_class", type=int, default=10000)
    ap.add_argument("--max_eval_per_class", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.presplit:
        root = Path(args.presplit)
        tr = cap(scan_folders(root / "train"), args.max_train_per_class, args.seed)
        va = cap(scan_folders(root / "val"), args.max_eval_per_class, args.seed)
        te = cap(scan_folders(root / "test"), args.max_eval_per_class, args.seed)
        if min(len(tr), len(va), len(te)) == 0:
            raise SystemExit("Empty split: check --presplit path and awake/sleepy folder names.")
        print(f"Presplit images train/val/test: {len(tr)}/{len(va)}/{len(te)} | "
              f"closed ratio train {tr.closed.mean():.2f}")
    else:
        if not args.data:
            raise SystemExit("Give --data (MRL) or --presplit.")
        splits = make_splits(args)
    if splits is not None:
        tr, va, te = splits
    else:
        print("Layout: MRL Eye dataset (subject-wise split)")
        df = scan_mrl(args.data)
        if df.empty:
            raise SystemExit("No images found. Check --data path.")
        subs = sorted(df.subject.unique())
        random.Random(args.seed).shuffle(subs)
        n_te = max(1, round(0.15 * len(subs)))
        n_va = max(1, round(0.15 * len(subs)))
        te_s, va_s = set(subs[:n_te]), set(subs[n_te:n_te + n_va])
        tr = cap(df[~df.subject.isin(te_s | va_s)], args.max_train_per_class, args.seed)
        va = cap(df[df.subject.isin(va_s)], args.max_eval_per_class, args.seed)
        te = cap(df[df.subject.isin(te_s)], args.max_eval_per_class, args.seed)
    print(f"Using images train/val/test: {len(tr)}/{len(va)}/{len(te)}")

    print("Loading images into RAM...")
    xtr, ytr = load_images(tr.path.tolist(), args.size), tr.closed.values.astype(np.int64)
    xva, yva = load_images(va.path.tolist(), args.size), va.closed.values.astype(np.int64)
    xte, yte = load_images(te.path.tolist(), args.size), te.closed.values.astype(np.int64)

    xcr = ycr = None
    if args.cross_root:
        cdf = scan_folders(args.cross_root)
        cdf = cap(cdf, args.max_eval_per_class, args.seed)
        print(f"Cross-dataset set: {len(cdf)} images (closed ratio {cdf.closed.mean():.2f})")
        xcr, ycr = load_images(cdf.path.tolist(), args.size), cdf.closed.values.astype(np.int64)

    results, histories, cms = [], {}, {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        model, hist, mins = train_model(name, xtr, ytr, xva, yva, args, out_dir)
        p = predict(model, xte)
        m = metrics(yte, p)
        row = dict(model=name,
                   params_M=sum(q.numel() for q in model.parameters()) / 1e6,
                   size_MB=(out_dir / f"{name}.pt").stat().st_size / 1e6,
                   test_acc=m["acc"], precision=m["precision"], recall=m["recall"],
                   f1=m["f1"], auc=m["auc"],
                   latency_ms=latency_ms(model, args.size), train_min=mins)
        if xcr is not None:
            row["cross_acc"] = metrics(ycr, predict(model, xcr))["acc"]
        results.append(row)
        histories[name] = hist
        cms[name] = confusion_matrix(yte, (p >= 0.5).astype(int))
        pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)  # save as we go

    res = pd.DataFrame(results).round(4)
    print("\n================ FINAL COMPARISON (held-out subjects) ================")
    print(res.to_string(index=False))
    json.dump(histories, open(out_dir / "history.json", "w"), indent=2)

    # Plots
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
    ax[1].set_title("CPU latency per eye crop (ms)")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=150)

    fig, ax = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3.2))
    ax = np.atleast_1d(ax)
    for a, (n_, cm) in zip(ax, cms.items()):
        a.imshow(cm, cmap="Blues")
        a.set_title(n_, fontsize=9)
        a.set_xticks([0, 1], ["open", "closed"])
        a.set_yticks([0, 1], ["open", "closed"])
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=150)
    print(f"\nSaved weights, results.csv and plots in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
