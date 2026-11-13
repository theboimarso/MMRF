#!/usr/bin/env python3
"""After training a teacher: the motion tokens of every frame of every train / val / test sequence, and the pooled frame-0 encoder features the
student reads. Saved next to the checkpoint:
  tokens/{split}.pt     {subject: float16 (T, N, d)}   (frame 0 included; index = frame)
  ref_feats/{split}.pt  {subject: float32 (216, 672)}  the frame-0 features of the three levels average-pooled to 6^3 and concatenated

  python dump_tokens.py --teacher runs/teacher_lung --data_dir /data/lung_ct_preprocessed --split /data/lung_split.json"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mmrf import data, nets


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="teacher run directory (checkpoint_best.pth)"); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", required=True); a = ap.parse_args(); run = Path(a.teacher)
    enc, _, _ = nets.load_teacher(run / "checkpoint_best.pth"); sp = data.load_split(a.split)
    (run / "tokens").mkdir(exist_ok=True); (run / "ref_feats").mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        tok, ref = {}, {}
        for s in data.subjects(a.data_dir, sp[split]):
            V = torch.from_numpy(data.load_sequence(a.data_dir, s))[:, None].cuda(); f = enc.features(V)
            tok[s] = enc.tokens(f).half().cpu()
            ref[s] = torch.cat([F.adaptive_avg_pool3d(f[l][0:1], 6)[0].reshape(-1, 216) for l in (1, 2, 3)]).T.float().cpu()
        torch.save(tok, run / "tokens" / f"{split}.pt"); torch.save(ref, run / "ref_feats" / f"{split}.pt")
        print(f"{split}: {len(tok)} sequences, {sum(len(v) for v in tok.values())} frames", flush=True)


if __name__ == "__main__":
    main()
