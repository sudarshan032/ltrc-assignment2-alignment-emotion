"""Evaluate a trained fusion model on its test split.

python models/fusion_pipeline/test.py --speech wavlm --text bert --fusion gmu --split words --seed 0
Writes Results/runs/<name>_{metrics.json,predictions.csv,embeddings.npz}.
The embeddings are the fusion block's 256-d output; predictions.csv also has the mean speech gate (GMU).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
from train import args, build, name_of  # noqa: E402


def main():
    cfg = args()
    cfg.pop("epochs")
    ck = torch.load(C.CKPT / f"{name_of(cfg)}.pt", map_location="cpu")
    df = C.metadata()
    sp = C.split(df, cfg["split"])
    model, inputs = build(ck["config"], df, sp)
    model.load_state_dict(ck["state"])
    model.to(C.DEVICE)
    out = C.predict(model, inputs, df.label.values, sp["test"])
    C.write_test_outputs(name_of(cfg), df, sp, out, dict(pipeline="fusion", **cfg))


if __name__ == "__main__":
    main()
