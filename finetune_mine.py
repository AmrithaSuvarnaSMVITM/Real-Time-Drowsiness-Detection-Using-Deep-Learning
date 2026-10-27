r"""
Fine-tune a trained eye-state model on YOUR OWN webcam crops.
Uses build_model / load_images / augment / predict / metrics from train_compare.py
(same folder). Label convention is the same: closed = 1, open = 0.

Folders:  my_data/s0/open/*.png   my_data/s0/closed/*.png
  python finetune_mine.py --weights results_subjectwise\resnet18.pt --model resnet18 ^
      --train my_data\s0 --out mine
Optional honest test:  --test my_data\s1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_compare import IMG_EXT, MODELS, augment, build_model, load_images, metrics, predict


def scan(root):
    paths, labels = [], []
    for cls, lab in (("open", 0), ("closed", 1)):
        d = Path(root) / cls
        if not d.is_dir():
            raise SystemExit(f"Missing folder: {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)  # timestamp names = time order
        paths.append([str(p) for p in files])
        labels.append(lab)
    return paths, labels


def split_last(files, frac=0.25):
    k = int(len(files) * (1 - frac))
    return files[:k], files[k:]


def report(tag, model, x, y):
    p = predict(model, x)
    m = metrics(y, p)
    pred = (p >= 0.5).astype(int)
    open_ok = (pred[y == 0] == 0).mean() if (y == 0).any() else float("nan")
    closed_ok = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
    print(f"{tag:>7}: acc {m['acc']*100:5.1f}% | open correct {open_ok*100:5.1f}% | "
          f"closed correct {closed_ok*100:5.1f}% | F1 {m['f1']:.3f}")
    return m["acc"], open_ok, closed_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="resnet18", choices=MODELS)
    ap.add_argument("--train", required=True, help="folder with open/ and closed/")
    ap.add_argument("--test", default=None, help="separate recording (optional but recommended)")
    ap.add_argument("--out", default="mine")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    op, cl = scan(a.train)[0]
    print(f"Found {len(op)} open and {len(cl)} closed crops in {a.train}")
    if min(len(op), len(cl)) < 40:
        raise SystemExit("Too few crops (need at least ~40 per class). Record more with realtime.py (o / c keys).")

    if a.test:
        top, tcl = scan(a.test)[0]
        tr_paths, te_paths = op + cl, top + tcl
        tr_lab = [0] * len(op) + [1] * len(cl)
        te_lab = [0] * len(top) + [1] * len(tcl)
    else:
        print("WARNING: no --test given. Last 25% of each class (by time) is held out. "
              "Frames are similar to training ones, so this is a weaker test.")
        o_tr, o_te = split_last(op)
        c_tr, c_te = split_last(cl)
        tr_paths, te_paths = o_tr + c_tr, o_te + c_te
        tr_lab = [0] * len(o_tr) + [1] * len(c_tr)
        te_lab = [0] * len(o_te) + [1] * len(c_te)
    print(f"Train {len(tr_paths)} | Test {len(te_paths)}")

    xtr, ytr = load_images(tr_paths, a.size), np.array(tr_lab, np.int64)
    xte, yte = load_images(te_paths, a.size), np.array(te_lab, np.int64)

    model = build_model(a.model, pretrained=False)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))

    print("\n--- Before fine-tuning (original model on YOUR webcam crops) ---")
    before = report("BEFORE", model, xte, yte)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                    batch_size=a.bs, shuffle=True)
    t0 = time.time()
    print("\n--- Fine-tuning ---")
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb = augment(xb.float() / 255)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        acc = metrics(yte, predict(model, xte))["acc"]
        print(f"epoch {ep+1}/{a.epochs}  loss {tot/n:.4f}  test acc {acc*100:.1f}%")

    print("\n--- After fine-tuning ---")
    after = report("AFTER", model, xte, yte)
    dst = out / f"{a.model}_mine.pt"
    torch.save(model.state_dict(), dst)

    print("\n================ BEFORE vs AFTER ================")
    print(f"Accuracy      : {before[0]*100:.1f}%  ->  {after[0]*100:.1f}%")
    print(f"Open correct  : {before[1]*100:.1f}%  ->  {after[1]*100:.1f}%   (looking-down frames live here)")
    print(f"Closed correct: {before[2]*100:.1f}%  ->  {after[2]*100:.1f}%")
    print(f"Time {(time.time()-t0)/60:.1f} min. Saved {dst}")


if __name__ == "__main__":
    main()
