"""Controlled smallcnn experiments (diagnostics only, never writes results.csv)."""
from __future__ import annotations
import argparse, time, json
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.transforms import v2
from hidrometros import train as T
from hidrometros.models import PROTOCOL, SmallCNN, SEED
from hidrometros.reproducibility import set_seed

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=0)
p.add_argument("--aug", default="protocol")  # none|protocol
p.add_argument("--sched", default="onecycle")  # onecycle|const
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--norm", default="imagenet")  # imagenet|none|half
p.add_argument("--ls", type=float, default=0.1)
p.add_argument("--arch", default="orig")
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--tag", default="")
p.add_argument("--dev", default="mps")
a = p.parse_args()

def main() -> None:
    global dev
    set_seed(SEED)
    dev = torch.device(a.dev)
    spec = PROTOCOL["smallcnn"]
    tr = T.split_index("train", T.BASE.index_csv)
    va = T.split_index("val", T.BASE.index_csv)
    if a.n:
        tr = tr.sample(n=a.n, random_state=SEED).reset_index(drop=True)
    ximg_tr, ximg_va = T.load_crops(tr), T.load_crops(va)
    ytr = tr["digit"].to_numpy(dtype=np.int64); yva = va["digit"].to_numpy(dtype=np.int64)

    def norm():
        if a.norm == "imagenet": return v2.Normalize(T.IMAGENET_MEAN, T.IMAGENET_STD)
        if a.norm == "half": return v2.Normalize((0.5,)*3, (0.5,)*3)
        return v2.Identity()
    ev = v2.Compose([v2.Resize(spec.input_hw, antialias=True), v2.ToDtype(torch.float32, scale=True), norm()])
    if a.aug == "protocol":
        trt = v2.Compose([v2.Resize(spec.input_hw, antialias=True),
            v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.3, contrast=0.3), v2.ToDtype(torch.float32, scale=True), norm()])
    else: trt = ev
    def mk(imgs, y, t, shuf):
        return torch.utils.data.DataLoader(T.CropDataset(imgs, y, t), batch_size=a.bs, shuffle=shuf,
            generator=torch.Generator().manual_seed(SEED), num_workers=4 if a.aug=="protocol" else 0,
            persistent_workers=a.aug=="protocol", worker_init_fn=T.worker_init_fn)
    trl, val = mk(ximg_tr, ytr, trt, True), mk(ximg_va, yva, ev, False)

    import importlib.util, sys
    def build():
        if a.arch == "orig": return SmallCNN()
        sys.path.insert(0, "local/exp"); import archs
        return archs.get(a.arch)
    m = build().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(trl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps) if a.sched == "onecycle" else None
    crit = nn.CrossEntropyLoss(label_smoothing=a.ls)
    print(vars(a), "steps/epoch", len(trl), "params", sum(q.numel() for q in m.parameters()), flush=True)
    for ep in range(1, a.epochs + 1):
        m.train(); ls=c=s=0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); o = m(x); l = crit(o, y); l.backward(); opt.step()
            if sch: sch.step()
            ls += l.item()*len(y); c += int((o.argmax(1)==y).sum()); s += len(y)
        vl, vacc, vf1, *_ = T.evaluate_loader(m, val, dev)
        print(f"ep {ep} tl={ls/s:.3f} ta={c/s:.3f} vl={vl:.3f} va={vacc:.3f} vf1={vf1:.3f}", flush=True)

if __name__ == '__main__':
    main()
