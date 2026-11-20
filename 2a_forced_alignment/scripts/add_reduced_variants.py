"""Add reduced pronunciations in which a /t/ after /n/ is dropped before a consonant or at the end of the word
(appointment, important, ...), so the aligner can skip an unreleased /t/ instead of squeezing it.
The reduced variant gets the same prior as the full one, so the acoustics decide.

python scripts/add_reduced_variants.py dictionaries/custom_arpa.dict dictionaries/custom_arpa_reduced.dict
"""
import sys
from pathlib import Path

VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")


def reduce(ph):
    """One variant per droppable /t/, plus one with all of them dropped (words like "appointment" have two)."""
    drop = [i for i, p in enumerate(ph) if p == "T" and i > 0 and ph[i - 1] == "N"
            and (i + 1 == len(ph) or not ph[i + 1].startswith(VOWELS))]
    sets = [{i} for i in drop] + ([set(drop)] if len(drop) > 1 else [])
    return [[p for i, p in enumerate(ph) if i not in d] for d in sets]


def main(src, dst):
    lines, have, added = Path(src).read_text().splitlines(), set(), []
    for line in lines:
        f = line.split("\t")
        have.add((f[0], f[-1]))
    for line in lines:
        f = line.split("\t")
        for v in reduce(f[-1].split()):
            r = " ".join(v)
            if (f[0], r) not in have:
                have.add((f[0], r))
                added.append("\t".join(f[:-1] + [r]))
    Path(dst).write_text("\n".join(lines + added) + "\n")
    print("\n".join(added))


if __name__ == "__main__":
    main(*sys.argv[1:])
