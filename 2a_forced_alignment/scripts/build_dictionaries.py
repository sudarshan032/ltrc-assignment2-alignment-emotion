"""Build one small pronunciation dictionary per phone set, covering exactly the corpus words.

custom_<set>.dict = every corpus word found in the pretrained MFA lexicon (all its variants)
                  + G2P pronunciations for the words the lexicon lacks (oov_<set>_g2p.dict,
                    made by `mfa g2p` in run_pipeline.sh).

Usage: python build_dictionaries.py <pretrained_dict> <oov_g2p_dict> <out_dict>
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def corpus_words():
    words = set()
    for lab in (ROOT / "corpus").rglob("*.lab"):
        # same splitting MFA does: lowercase, hyphens split words, edge punctuation stripped
        for tok in re.split(r"[\s\-]+", lab.read_text().lower()):
            tok = tok.strip(".,:;!?\"")
            if tok:
                words.add(tok)
    return words


def main(base, oov, out):
    words = corpus_words()
    base_lines = Path(base).read_text(encoding="utf-8").splitlines()
    base_words = {l.split("\t", 1)[0] for l in base_lines}
    # english_mfa has no "x's" entries: MFA splits the clitic, so keep the stem and "'s" instead
    for w in [w for w in words if w.endswith("'s") and w not in base_words and "'s" in base_words]:
        words |= {w[:-2], "'s"}
        words.discard(w)
    lines, found = [], set()
    for line in base_lines:
        w = line.split("\t", 1)[0]
        if w in words:
            lines.append(line)
            found.add(w)
    g2p = Path(oov).read_text(encoding="utf-8").splitlines()
    lines += g2p
    found |= {l.split("\t", 1)[0] for l in g2p}
    missing = sorted(words - found)
    assert not missing, f"words with no pronunciation: {missing}"
    Path(out).write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")  # sorted: stable across runs
    print(f"{out}: {len(words)} words, {len(lines)} pronunciations, {len(g2p)} from G2P")


if __name__ == "__main__":
    main(*sys.argv[1:4])
