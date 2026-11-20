"""Evaluate a trained text model on its test split.

python models/text_pipeline/test.py --features bert --split words --seed 0
Writes Results/runs/<name>_{metrics.json,predictions.csv,embeddings.npz}.
The embeddings are the contextual-modelling block's 256-d output (used for the cluster plots).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
from train import FEATURES, build, name_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="bert", choices=FEATURES)
    ap.add_argument("--split", default="words")
    ap.add_argument("--seed", type=int, default=0)
    cfg = vars(ap.parse_args())
    ck = torch.load(C.CKPT / f"{name_of(cfg)}.pt", map_location="cpu")
    df = C.metadata()
    sp = C.split(df, cfg["split"])
    model, inputs = build(ck["config"], df, sp)
    model.load_state_dict(ck["state"])
    model.to(C.DEVICE)
    out = C.predict(model, inputs, df.label.values, sp["test"])
    C.write_test_outputs(name_of(cfg), df, sp, out, dict(pipeline="text", **cfg))


if __name__ == "__main__":
    main()
