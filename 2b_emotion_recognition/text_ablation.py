"""Does the fusion model use the text at all? Re-test each trained WavLM+BERT fusion model with the text input
(a) shuffled across test clips and (b) replaced by one fixed sentence. Same accuracy = text has no effect.

python text_ablation.py        # -> Results/text_ablation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models/fusion_pipeline"))
import common as C  # noqa: E402
from train import build, name_of  # noqa: E402


def acc(model, inputs, y, idx):
    return 100 * (C.predict(model, inputs, y, idx)["probs"].argmax(1) == y[idx]).mean()


def main():
    df = C.metadata()
    y = df.label.values
    rows = []
    for split in ["words", "spk_OAF", "spk_YAF"]:
        sp = C.split(df, split)
        te = sp["test"]
        for fusion in ["gmu", "concat"]:
            for seed in range(3):
                cfg = dict(speech="wavlm", text="bert", fusion=fusion, split=split, seed=seed)
                ck = C.CKPT / f"{name_of(cfg)}.pt"
                if not ck.exists():
                    sys.exit(f"missing {ck.name}: this needs the checkpoints of all 3 seeds (only seed 0 is in git); "
                             "retrain seeds 1 and 2 first (run_all.py --seeds 1 2)")
                model, (s_in, t_in) = build(cfg, df, sp)
                model.load_state_dict(torch.load(ck, map_location="cpu")["state"])
                model.to(C.DEVICE)
                shuffled, constant = list(t_in), list(t_in)
                perm = np.random.RandomState(seed).permutation(te)
                for i, j in zip(te, perm):
                    shuffled[i] = t_in[j]
                    constant[i] = t_in[te[0]]
                rows.append(dict(split=split, fusion=fusion, seed=seed,
                                 normal=acc(model, [s_in, t_in], y, te),
                                 text_shuffled=acc(model, [s_in, shuffled], y, te),
                                 text_constant=acc(model, [s_in, constant], y, te)))
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows).round(2)
    t.to_csv(C.RESULTS / "text_ablation.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
