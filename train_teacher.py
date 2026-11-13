#!/usr/bin/env python3
"""Train the teacher: from the full volumes V_0 (reference) and V_t, a 3D CNN encoder E, a per-position token encoder g and a token-conditioned FiLM
decoder D learn the displacement field u_t with V_0(x + u_t(x)) ~ V_t(x), from image similarity alone (local NCC) with smoothness and folding
penalties. No registration field is used as a target.

One step = one sequence (all its frames; tokens of every frame computed once), gradients accumulated over --batch sequences. Checkpoint: the epoch
with the best validation NCC (the validation folding ratio is logged); early stopping after --patience epochs without improvement.
Several --data entries train one joint teacher (the checkpoint criterion is then the mean of the per-dataset validation NCCs).

  python train_teacher.py --data lung:/data/lung_ct_preprocessed:/data/lung_split.json --out runs/teacher_lung
  python train_teacher.py --data lung:...:... --data cardiac:...:... --out runs/teacher_joint           # joint teacher
  python train_teacher.py --data liver:...:... --init_from runs/teacher_joint/checkpoint_best.pth --lr 5e-5 --out runs/teacher_liver_ft"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmrf import data, losses, nets


@torch.no_grad()
def validate(enc, dec, stn, sets, n_samples, device):
    """per dataset: one random frame of each of up to n_samples validation sequences -> mean NCC, mean folding %"""
    enc.eval(); dec.eval(); out = {}
    for name, (ddir, ids) in sets.items():
        rng = np.random.RandomState(0); ids = list(ids); rng.shuffle(ids); nc, fo = [], []
        for s in ids[:n_samples]:
            V = torch.from_numpy(data.load_sequence(ddir, s))[:, None].to(device); t = int(rng.randint(1, len(V)))
            f = enc.features(V); u = dec({l: f[l][0:1] for l in (1, 2, 3)}, enc.tokens(f)[t][None])
            nc.append(-float(losses.lncc_loss(stn(V[0:1], u), V[t:t + 1]))); fo.append(losses.fold_pct(u))
        out[name] = (float(np.mean(nc)), float(np.mean(fo)))
    enc.train(); dec.train(); return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="name:data_dir:split_file (repeat for a joint teacher)")
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2, help="sequences per optimiser step"); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4); ap.add_argument("--lambda_smooth", type=float, default=0.01)
    ap.add_argument("--lambda_fold", type=float, default=1000.0); ap.add_argument("--fold_margin", type=float, default=0.05)
    ap.add_argument("--val_samples", type=int, default=40); ap.add_argument("--token_dim", type=int, default=128)
    ap.add_argument("--token_grid", type=int, default=6, help="token grid side (6 -> 216 tokens)"); ap.add_argument("--full_decoder", action="store_true",
                    help="predict the field at full resolution (default: at 48^3, upsampled)")
    ap.add_argument("--init_from", default=None, help="checkpoint to initialise from (fine-tuning)"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); device = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = dict(token_dim=a.token_dim, token_grid=a.token_grid, light_decoder=not a.full_decoder); json.dump(vars(a) | cfg, open(out / "config.json", "w"), indent=1)

    train, val = [], {}
    for spec in a.data:
        name, ddir, sf = spec.split(":"); sp = data.load_split(sf)
        train += [(name, ddir, s) for s in data.subjects(ddir, sp["train"])]; val[name] = (ddir, data.subjects(ddir, sp["val"]))
        print(f"{name}: {sum(1 for x in train if x[0] == name)} train / {len(val[name][1])} val sequences")

    enc, dec = nets.build_teacher(cfg); enc.to(device); dec.to(device); stn = nets.SpatialTransformer().to(device)
    if a.init_from:
        e0, d0, _ = nets.load_teacher(a.init_from, device); enc.load_state_dict(e0.state_dict()); dec.load_state_dict(d0.state_dict()); del e0, d0
        print(f"initialised from {a.init_from}")
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
    history, best, best_ep, start = [], -float("inf"), 0, 0
    if a.resume and (out / "checkpoint_latest.pth").exists():
        ck = torch.load(out / "checkpoint_latest.pth", map_location=device, weights_only=False)
        enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, best_ep = ck["epoch"], ck["best_val"], ck["best_epoch"]; history = json.load(open(out / "history.json")); print(f"resumed at epoch {start}")

    for ep in range(start, a.epochs):
        t0 = time.time(); order = np.random.RandomState(ep).permutation(len(train)); acc = {"ncc": 0., "smooth": 0., "fold": 0.}; opt.zero_grad()
        for i, j in enumerate(tqdm(order, desc=f"epoch {ep + 1}", leave=False)):
            _, ddir, s = train[j]; V = torch.from_numpy(data.load_sequence(ddir, s))[:, None].to(device)
            f = enc.features(V); tok = enc.tokens(f); ref = {l: f[l][0:1] for l in (1, 2, 3)}
            ts = list(range(1, len(V))); np.random.RandomState(ep * 100003 + i).shuffle(ts)
            for k, t in enumerate(ts):
                u = dec(ref, tok[t][None]); l_img = losses.lncc_loss(stn(V[0:1], u), V[t:t + 1]); l_sm = losses.smooth_loss(u); l_fo = losses.fold_penalty(u, a.fold_margin)
                ((l_img + a.lambda_smooth * l_sm + a.lambda_fold * l_fo) / len(ts) / a.batch).backward(retain_graph=k < len(ts) - 1)
                acc["ncc"] -= l_img.item() / len(ts) / len(train); acc["smooth"] += l_sm.item() / len(ts) / len(train); acc["fold"] += l_fo.item() / len(ts) / len(train)
            if (i + 1) % a.batch == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        sched.step()
        v = validate(enc, dec, stn, val, a.val_samples, device); vn = float(np.mean([x[0] for x in v.values()])); vf = float(max(x[1] for x in v.values()))
        improved = vn > best
        state = dict(encoder=enc.state_dict(), decoder=dec.state_dict(), config=cfg, epoch=ep + 1)
        if improved: best, best_ep = vn, ep + 1; torch.save(state | dict(best_val=best), out / "checkpoint_best.pth")
        torch.save(state | dict(opt=opt.state_dict(), sched=sched.state_dict(), best_val=best, best_epoch=best_ep), out / "checkpoint_latest.pth")
        history.append(dict(epoch=ep + 1, time_s=time.time() - t0, train=acc, val_ncc=vn, val_fold_pct=vf, val={k: dict(ncc=x[0], fold_pct=x[1]) for k, x in v.items()}))
        json.dump(history, open(out / "history.json", "w"), indent=1)
        print(f"epoch {ep + 1}: train NCC {acc['ncc']:.4f}  val NCC {vn:.4f}  val folding {vf:.4f} %" + "".join(f"  {k} {x[0]:.4f}" for k, x in v.items() if len(v) > 1)
              + ("  [best]" if improved else ""), flush=True)
        if best_ep and ep + 1 - best_ep >= a.patience: print(f"early stopping (best epoch {best_ep}, val NCC {best:.4f})"); break


if __name__ == "__main__":
    main()
