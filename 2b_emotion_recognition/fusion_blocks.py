"""Block outputs inside the multimodal (GMU) model, as in the brief's figure: the speech temporal block, the text
contextual block and the fusion block of the same trained WavLM + BERT fusion model, on the test clips.

Writes Results/runs/fusion_gmu_wavlm+bert_<split>_s<seed>_blocks.npz (speech, text, fusion vectors + labels) and
Results/fusion_blocks.csv: silhouette of each block, and how much each GMU term varies across test clips.
python fusion_blocks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models/fusion_pipeline"))
import common as C  # noqa: E402
from train import build, name_of  # noqa: E402


@torch.no_grad()
def blocks(model, inputs, labels, idx):
    model.eval()
    S, T, H, ZS, ZT = [], [], [], [], []
    for xs, _, _ in C.batches(inputs, labels, idx, 64, False):
        s, t = model.speech(*xs[0]), model.text(*xs[1])
        f = model.fusion
        z = torch.sigmoid(f.z(torch.cat([s, t], -1)))
        zs, zt = z * torch.tanh(f.s(s)), (1 - z) * torch.tanh(f.t(t))      # the two terms of h = zs + zt
        for L, v in zip((S, T, H, ZS, ZT), (s, t, zs + zt, zs, zt)):
            L.append(v.cpu())
    return [torch.cat(L).numpy() for L in (S, T, H, ZS, ZT)]


def main():
    df = C.metadata()
    y_all = df.label.values
    rows = []
    for split in ["words", "spk_YAF", "spk_OAF"]:
        sp = C.split(df, split)
        for seed in range(3):
            cfg = dict(speech="wavlm", text="bert", fusion="gmu", split=split, seed=seed)
            ck = C.CKPT / f"{name_of(cfg)}.pt"
            if not ck.exists():
                sys.exit(f"missing {ck.name}: needs the checkpoints of all 3 seeds (only seed 0 is in git)")
            model, inputs = build(cfg, df, sp)
            model.load_state_dict(torch.load(ck, map_location="cpu")["state"])
            model.to(C.DEVICE)
            s, t, h, zs, zt = blocks(model, inputs, y_all, sp["test"])
            y = y_all[sp["test"]]
            np.savez_compressed(C.RESULTS / "runs" / f"{name_of(cfg)}_blocks.npz", speech=s, text=t, fusion=h, y=y)
            sil = lambda e: silhouette_score(e, y) if len(np.unique(e.round(5), axis=0)) > 7 else np.nan
            var = lambda e: float(e.var(0).sum())          # total variance across test clips
            rows.append(dict(split=split, seed=seed, speech_block=sil(s), text_block=sil(t), fusion_block=sil(h),
                             speech_term_var=var(zs), text_term_var=var(zt),
                             text_term_share=var(zt) / (var(zs) + var(zt))))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "fusion_blocks.csv", index=False)
    print(t.groupby("split")[["speech_block", "text_block", "fusion_block", "text_term_share"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
