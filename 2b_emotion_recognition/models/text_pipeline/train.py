"""Text-only emotion recognition: cleaning/tokenizing -> features -> contextual modelling (Transformer + attention) -> classifier.

python models/text_pipeline/train.py --features bert --split words --seed 0
  --features bert    : frozen BERT-base-uncased, learned mix of its 13 layers (WordPiece tokens x 768)
  --features scratch : from-scratch baseline, word embeddings learned from the training transcripts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402

FEATURES = ["bert", "scratch"]


def build(cfg, df, sp):
    if cfg["features"] == "bert":
        return C.UniModel(C.TextEncoder(768, 13)), [C.features("bert")]
    vocab = C.word_vocab(df, sp["train"])
    return C.UniModel(C.TextEncoder(vocab_size=len(vocab) + 2)), [C.word_ids(df, vocab)]


def name_of(cfg):
    return f"text_{cfg['features']}_{cfg['split']}_s{cfg['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="bert", choices=FEATURES)
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
