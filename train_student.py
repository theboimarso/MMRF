#!/usr/bin/env python3
"""Train the student: from the live surrogate plane (frames t-2, t-1, t), the reference plane and the pooled frame-0 encoder features, predict the
teacher's motion tokens of frame t (mean squared error on the standardised tokens). The teacher stays frozen; run dump_tokens.py first.
Standardisation: per-channel mean / std of the teacher tokens over the training frames t >= 1, and of the reference features over the training
subjects. Augmentation (identical for the four planes of a sample): in-plane shift up to +-3 pixels, gamma, gain and offset.
Outputs in --out: best.pth, stats.pt (both standardisations), starts_{val,test}.pt ({subject: (T-1, 216, 128) standardised token predictions}).

  python train_student.py --teacher runs/teacher_lung --data_dir /data/lung_ct_preprocessed --plane_axis 1 --out runs/teacher_lung/student"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mmrf import data
from mmrf.nets import Student


def build(a):
    """per (split, subject): planes (T, 96, 96), standardised reference features (216, 672), standardised target tokens (T, 216, 128)"""
    run = Path(a.teacher); tk = {s: torch.load(run / "tokens" / f"{s}.pt", weights_only=True) for s in ("train", "val", "test")}
    ref = {s: torch.load(run / "ref_feats" / f"{s}.pt", weights_only=True) for s in ("train", "val", "test")}
    Y = torch.cat([v[1:].float() for v in tk["train"].values()]); mu, sd = Y.mean((0, 1)), Y.std((0, 1)) + 1e-6; del Y
    H = torch.stack([ref["train"][p] for p in tk["train"]]); hmu, hsd = H.mean((0, 1)), H.std((0, 1)) + 1e-6; del H
    D = {}
    for s in ("train", "val", "test"):
        for p in tk[s]:
            V = data.load_sequence(a.data_dir, p); assert len(V) == len(tk[s][p]), p
            D[(s, p)] = dict(sl=torch.from_numpy(np.stack([data.plane(v, a.plane_axis) for v in V])).float(), F0=(ref[s][p] - hmu) / hsd, T=(tk[s][p].float() - mu) / sd)
    return D, dict(mu=mu, sd=sd, hmu=hmu, hsd=hsd)


def batch(D, s, its, aug, device):
    W, Rf, F0, Tg = [], [], [], []
    for p, t in its:
        d = D[(s, p)]; win, ref = d["sl"][[max(t - 2, 0), max(t - 1, 0), t]], d["sl"][0:1]
        if aug:
            sh = tuple(int(x) for x in np.random.randint(-3, 4, 2)); win, ref = torch.roll(win, sh, (1, 2)), torch.roll(ref, sh, (1, 2))
            gm, gn, of = np.random.uniform(0.85, 1.15), np.random.uniform(0.92, 1.08), np.random.uniform(-0.04, 0.04)
            win, ref = [(x.clamp(1e-6, 1) ** gm * gn + of).clamp(0, 1) for x in (win, ref)]
        W.append(win); Rf.append(ref); F0.append(d["F0"]); Tg.append(d["T"][t])
    return [torch.stack(x).to(device) for x in (W, Rf, F0, Tg)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True); ap.add_argument("--data_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--plane_axis", type=int, required=True, help="axis of the surrogate plane (lung, liver: 1 = coronal; cardiac: 0 = short axis)")
    ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--patience", type=int, default=20); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); device = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    D, stats = build(a); torch.save(stats, out / "stats.pt")
    items = {s: [(p, t) for (s_, p), d in D.items() if s_ == s for t in range(1, len(d["sl"]))] for s in ("train", "val", "test")}
    print(f"{len(items['train'])} / {len(items['val'])} / {len(items['test'])} train / val / test frames")
    net = Student(layers=a.layers).to(device); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)

    @torch.no_grad()
    def val_loss():
        net.eval(); its = items["val"]; L = 0.
        for i in range(0, len(its), 64): w, r, f, tg = batch(D, "val", its[i:i + 64], False, device); L += F.mse_loss(net(w, r, f), tg, reduction="sum").item()
        net.train(); return L / (len(its) * 216 * 128)

    best, best_ep, t0 = float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        perm = np.random.permutation(len(items["train"]))
        for i in range(0, len(perm), a.batch):
            w, r, f, tg = batch(D, "train", [items["train"][j] for j in perm[i:i + a.batch]], True, device)
            loss = F.mse_loss(net(w, r, f), tg); opt.zero_grad(); loss.backward(); opt.step()
        vl = val_loss()
        if vl < best: best, best_ep = vl, ep; torch.save(net.state_dict(), out / "best.pth")
        print(f"epoch {ep}: val token MSE {vl:.4f}{'  [best]' if ep == best_ep else ''}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if ep - best_ep >= a.patience: print(f"early stopping (best epoch {best_ep})"); break
    net.load_state_dict(torch.load(out / "best.pth", weights_only=True)); net.eval()
    with torch.no_grad():
        for s in ("val", "test"):
            SF = {}
            for (s_, p), d in D.items():
                if s_ == s: w, r, f, _ = batch(D, s, [(p, t) for t in range(1, len(d["sl"]))], False, device); SF[p] = net(w, r, f).cpu()
            torch.save(SF, out / f"starts_{s}.pt")
            P = torch.cat(list(SF.values())); Tg = torch.cat([D[(s, p)]["T"][1:] for p in SF])
            print(f"{s}: token R^2 {1 - float(((P - Tg) ** 2).sum() / ((Tg - Tg.mean((0, 1))) ** 2).sum()):.3f}")


if __name__ == "__main__":
    main()
