#!/usr/bin/env python3
"""Inference with test-time refinement, and evaluation.

Per frame t (sequentially): the student predicts the standardised tokens T_t from the live plane; the refinement fits one shift delta (128-d),
shared by all positions (T_t + 1 delta^T), by --steps Adam steps (learning rate --eta, fresh optimiser state each frame, delta warm-started from the
previous frame's) on
    1 - lNCC2D(P W_t(delta), S_t) + lambda mean(delta^2) + lambda_f fold(u_t(delta)) + lambda_r mean([(S_t - S_t-1) - (P W_t(delta) - P W_t-1)]^2)
where W_t(delta) is V_0 warped by the decoded field, P extracts the surrogate plane and P W_t-1 is the previous frame's refined plane (held fixed).
The frozen decoder runs in bfloat16; field, warp and loss in float32.

Evaluation on the --split_name sequences (frames 1..T-1 warp V_0 to V_t): lNCC, SSIM, PSNR, MSE and the folding ratio over the body (V_0 > 0.05),
per sequence = mean over frames, for no motion, the student, the refined student and the teacher (which sees V_t). Results -> --out (json).

  python refine.py --teacher runs/teacher_lung --student runs/teacher_lung/student --data_dir /data/lung_ct_preprocessed \\
                   --split /data/lung_split.json --plane_axis 1 --lambda_r 10 --out runs/teacher_lung/refined_test.json"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mmrf import data, losses, nets


def refine(V, planes, starts, ref, dec, stn, mu, sd, axis, eta=0.05, steps=1, warm=True, lam=1.0, lam_fold=1000.0, lam_r=0.0):
    """V (T, 1, 96^3) float32, planes (T, 1, 96, 96), starts (T-1, N, d) standardised student tokens -> refined standardised tokens (T-1, N, d)"""
    def decode(st):
        with torch.autocast("cuda", dtype=torch.bfloat16): return dec(ref, st * sd + mu).float()
    V0, prev, Wprev, out = V[0:1], None, planes[0:1], []
    for t in range(1, len(V)):
        st0, St = starts[t - 1:t], planes[t:t + 1]; dS = St - planes[t - 1:t]
        dl = (prev.clone() if warm and prev is not None else torch.zeros(1, st0.shape[-1], device=V.device)).requires_grad_(True)
        opt = torch.optim.Adam([dl], lr=eta)
        with torch.enable_grad():
            for _ in range(steps):
                u = decode(st0 + dl[:, None, :]); Wt = data.plane(stn(V0, u), axis)
                L = (1 - losses.lncc2d(Wt, St)).mean() + lam * (dl ** 2).mean() + lam_fold * losses.fold_penalty(u, 0.05)
                if lam_r: L = L + lam_r * ((dS - (Wt - Wprev)) ** 2).mean()
                opt.zero_grad(); L.backward(); opt.step()
        with torch.no_grad():
            st = st0 + dl.detach()[:, None, :]; Wprev = data.plane(stn(V0, decode(st)), axis); out.append(st); prev = dl.detach()
    return torch.cat(out)


@torch.no_grad()
def student_starts(net, planes, F0):
    """the student's standardised token predictions for frames 1..T-1"""
    T = len(planes); sl = planes[:, 0]
    W = torch.stack([sl[[max(t - 2, 0), max(t - 1, 0), t]] for t in range(1, T)]); Rf = sl[0:1][None].expand(T - 1, -1, -1, -1)
    return net(W, Rf, F0[None].expand(T - 1, -1, -1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="teacher run directory (checkpoint_best.pth)")
    ap.add_argument("--student", default=None, help="student directory (best.pth, stats.pt); or give --starts and --stats")
    ap.add_argument("--starts", default=None, help="precomputed standardised student predictions {subject: (T-1, 216, 128)}")
    ap.add_argument("--stats", default=None, help="token standardisation {mu, sd} (default: <student>/stats.pt)")
    ap.add_argument("--data_dir", required=True); ap.add_argument("--split", required=True); ap.add_argument("--split_name", default="test")
    ap.add_argument("--plane_axis", type=int, required=True); ap.add_argument("--eta", type=float, default=0.05); ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--cold", action="store_true", help="start every frame's shift from zero instead of the previous frame's")
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--lambda_fold", type=float, default=1000.0)
    ap.add_argument("--lambda_r", type=float, default=0.0, help="increment-consistency weight (paper: 10 lung, 0 cardiac)")
    ap.add_argument("--no_teacher", action="store_true", help="skip the teacher row"); ap.add_argument("--out", required=True)
    ap.add_argument("--save_fields", default=None, help="directory: refined fields per sequence (float16 npz)")
    a = ap.parse_args(); device = "cuda"; torch.backends.cudnn.benchmark = True
    enc, dec, _ = nets.load_teacher(Path(a.teacher) / "checkpoint_best.pth", device)
    for q in [*enc.parameters(), *dec.parameters()]: q.requires_grad_(False)
    stats = torch.load(a.stats or Path(a.student) / "stats.pt", weights_only=True); mu, sd = stats["mu"].to(device), stats["sd"].to(device)
    SF = torch.load(a.starts, weights_only=True) if a.starts else None
    if SF is None:
        net = nets.Student(layers=json.load(open(Path(a.student) / "config.json")).get("layers", 6)).to(device)
        net.load_state_dict(torch.load(Path(a.student) / "best.pth", map_location=device, weights_only=True)); net.eval()
        hmu, hsd = stats["hmu"].to(device), stats["hsd"].to(device)
    stn = nets.SpatialTransformer().to(device); res = {m: {} for m in ("no motion", "student", "refined", "teacher")}
    for s in data.subjects(a.data_dir, data.load_split(a.split)[a.split_name]):
        V = torch.from_numpy(data.load_sequence(a.data_dir, s))[:, None].to(device); planes = data.plane(V, a.plane_axis)
        with torch.no_grad():
            f = enc.features(V); ref = {l: f[l][0:1] for l in (1, 2, 3)}; tok = enc.tokens(f)
            if SF is not None: st = SF[s].to(device).float()
            else: F0 = torch.cat([torch.nn.functional.adaptive_avg_pool3d(f[l][0:1], 6)[0].reshape(-1, 216) for l in (1, 2, 3)]).T; st = student_starts(net, planes, (F0 - hmu) / hsd)
        rt = refine(V, planes, st, ref, dec, stn, mu, sd, a.plane_axis, a.eta, a.steps, not a.cold, a.lam, a.lambda_fold, a.lambda_r)
        body = V[0, 0] > 0.05; acc = {m: [] for m in res}; UR = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dS = lambda x: dec(ref, x * sd + mu).float()
            for t in range(1, len(V)):
                F_ = {"no motion": torch.zeros(1, 3, *V.shape[-3:], device=device), "student": dS(st[t - 1:t]), "refined": dS(rt[t - 1:t])}
                if not a.no_teacher:
                    with torch.autocast("cuda", enabled=False): F_["teacher"] = dec(ref, tok[t][None])
                for m, u in F_.items():
                    with torch.autocast("cuda", enabled=False): r = losses.metrics(stn(V[0:1], u.float()), V[t:t + 1]); r["fold_pct"] = losses.fold_pct(u, body)
                    acc[m].append(r)
                UR.append(F_["refined"][0].half().cpu().numpy())
        for m in res:
            if acc[m]: res[m][s] = {k: float(np.mean([r[k] for r in acc[m]])) for k in acc[m][0]}
        if a.save_fields: Path(a.save_fields).mkdir(parents=True, exist_ok=True); np.savez_compressed(Path(a.save_fields) / f"{s}.npz", u=np.stack(UR))
        print(f"{s}: " + "  ".join(f"{m} {res[m][s]['lncc']:.4f}" for m in res if s in res[m]), flush=True)
    summ = {m: {k: float(np.mean([v[k] for v in res[m].values()])) for k in next(iter(res[m].values()))} for m in res if res[m]}
    json.dump({"summary": summ, "per_sequence": res, "settings": vars(a)}, open(a.out, "w"), indent=1)
    for m, v in summ.items(): print(f"{m:10s} lNCC {v['lncc']:.4f}  SSIM {v['ssim']:.4f}  PSNR {v['psnr']:.2f}  folding {v['fold_pct']:.4f} %")


if __name__ == "__main__":
    main()
