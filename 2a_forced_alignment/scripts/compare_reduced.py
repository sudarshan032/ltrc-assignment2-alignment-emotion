"""Which pronunciation did the aligner choose once reduced (no /t/ after /n/) variants were allowed?

python scripts/compare_reduced.py     # -> results/reduced_variants.md
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORDS = {"appointment", "appointments", "important", "president", "current", "retirement"}


def tiers(p):
    out = {}
    for name, body in re.findall(r'name = "([^"]+)"(.*?)(?=name = "|\Z)', p.read_text(), re.S):
        out[name] = [(t, float(a), float(b)) for a, b, t in re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', body) if t]
    return out


rows = ["| file | word | time (s) | phones, full dictionary (ms) | phones, with reduced variants (ms) |", "|---|---|---|---|---|"]
chosen = total = 0
for tg in sorted((ROOT / "outputs/arpa_reduced").rglob("*.TextGrid")):
    full = tiers(ROOT / "outputs/arpa" / tg.relative_to(ROOT / "outputs/arpa_reduced"))
    red = tiers(tg)
    for (w, a, b), (w2, a2, b2) in zip(full["words"], red["words"]):
        if w not in WORDS:
            continue
        show = lambda t, s, e: " ".join(f"{p}:{round(1000 * (y - x))}" for p, x, y in t["phones"] if x >= s - 1e-6 and y <= e + 1e-6)
        f, r = show(full, a, b), show(red, a2, b2)
        total += 1
        chosen += r.count("T") < f.count("T")
        rows.append(f"| {tg.stem} | {w} | {a:.2f} | {f} | {r} |")
rows.append(f"\nReduced variant chosen in {chosen} of {total} tokens.")
(ROOT / "results/reduced_variants.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
