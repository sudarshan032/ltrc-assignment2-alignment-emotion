"""Speech-only emotion recognition: preprocessing -> features -> temporal modelling (BiLSTM + attention) -> classifier.

python models/speech_pipeline/train.py --features wavlm --split words --seed 0
  --features wavlm : frozen WavLM-base-plus, learned mix of its 13 layers (768-d, 50 frames/s)
  --features mfcc  : from-scratch baseline, 40 MFCC + delta + delta-delta (120-d, 100 frames/s)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402

DIMS = {"wavlm": (768, 13), "mfcc": (120, 1)}


def build(cfg, df, sp):
    feats = C.features(cfg["features"])
    dim, layers = DIMS[cfg["features"]]
    return C.UniModel(C.SpeechEncoder(dim, layers)), [feats]


def name_of(cfg):
    return f"speech_{cfg['features']}_{cfg['split']}_s{cfg['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="wavlm", choices=list(DIMS))
    ap.add_argument("--split", default="words")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    cfg = vars(ap.parse_args())
    C.set_seed(cfg["seed"])
    df = C.metadata()
    sp = C.split(df, cfg["split"])
    model, inputs = build(cfg, df, sp)
    hist = C.train(model, inputs, df.label.values, sp, epochs=cfg["epochs"], log=True)
    C.save_run(name_of(cfg), model, cfg, hist)


if __name__ == "__main__":
    main()
