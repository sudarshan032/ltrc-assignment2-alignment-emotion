"""Multimodal emotion recognition: speech branch + text branch -> fusion -> classifier, trained end to end.

python models/fusion_pipeline/train.py --speech wavlm --text bert --fusion gmu --split words --seed 0
  The branches are the same blocks as the unimodal pipelines (SpeechEncoder, TextEncoder).
  --fusion gmu    : gated multimodal unit; a per-dimension gate decides how much comes from each modality
  --fusion concat : concatenation + linear layer (ablation)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402

SPEECH_DIMS = {"wavlm": (768, 13), "mfcc": (120, 1)}
FUSIONS = {"gmu": C.GatedFusion, "concat": C.ConcatFusion}


def build(cfg, df, sp):
    speech = C.SpeechEncoder(*SPEECH_DIMS[cfg["speech"]])
    s_in = C.features(cfg["speech"])
    if cfg["text"] == "bert":
        text, t_in = C.TextEncoder(768, 13), C.features("bert")
    else:
        vocab = C.word_vocab(df, sp["train"])
        text, t_in = C.TextEncoder(vocab_size=len(vocab) + 2), C.word_ids(df, vocab)
    return C.FusionModel(speech, text, FUSIONS[cfg["fusion"]]()), [s_in, t_in]


def name_of(cfg):
    return f"fusion_{cfg['fusion']}_{cfg['speech']}+{cfg['text']}_{cfg['split']}_s{cfg['seed']}"


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", default="wavlm", choices=list(SPEECH_DIMS))
    ap.add_argument("--text", default="bert", choices=["bert", "scratch"])
    ap.add_argument("--fusion", default="gmu", choices=list(FUSIONS))
    ap.add_argument("--split", default="words")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    return vars(ap.parse_args())


def main():
    cfg = args()
    C.set_seed(cfg["seed"])
    df = C.metadata()
    sp = C.split(df, cfg["split"])
    model, inputs = build(cfg, df, sp)
    hist = C.train(model, inputs, df.label.values, sp, epochs=cfg["epochs"], log=True)
    C.save_run(name_of(cfg), model, cfg, hist)


if __name__ == "__main__":
    main()
