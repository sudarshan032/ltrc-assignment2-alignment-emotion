"""Held-out accuracy of the trained G2P model.

The model is trained (mfa train_g2p) on the english_us_arpa lexicon minus 2,000 random words
(results/g2p_heldout_words.txt); its predictions for those words are compared with the lexicon.
python scripts/evaluate_g2p.py <lexicon.dict> <predicted.dict>   -> results/g2p_eval.md
"""
import sys
from collections import defaultdict
from pathlib import Path


def load(p):
    d = defaultdict(set)
    for line in Path(p).read_text().splitlines():
        if line.strip():
            parts = line.split("\t")
            d[parts[0]].add(" ".join(parts[-1].split()))
    return d


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main(lex, pred):
    ref, hyp = load(lex), load(pred)
    words = [w for w in Path(__file__).resolve().parent.parent.joinpath("results/g2p_heldout_words.txt").read_text().split() if w in hyp]
    nostress = lambda s: " ".join(p.rstrip("012") for p in s.split())
    wrong = wrong_ns = errs = n = 0
    for w in words:
        h = next(iter(hyp[w]))
        wrong += h not in ref[w]
        wrong_ns += nostress(h) not in {nostress(r) for r in ref[w]}
        best = min(ref[w], key=lambda r: edit(h.split(), r.split()))   # closest reference variant
        errs += edit(h.split(), best.split())
        n += len(best.split())
    txt = (f"| held-out words | word error rate | word error rate, ignoring stress | phone error rate |\n|---|---|---|---|\n"
           f"| {len(words)} | {100 * wrong / len(words):.1f} % | {100 * wrong_ns / len(words):.1f} % | {100 * errs / n:.1f} % |\n")
    Path(__file__).resolve().parent.parent.joinpath("results/g2p_eval.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main(*sys.argv[1:])
