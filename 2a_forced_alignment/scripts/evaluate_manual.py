"""Absolute accuracy: compare every run's boundaries with the hand-corrected TextGrids in manual_labels/.

python scripts/evaluate_manual.py      # -> results/manual_accuracy.{csv,md}
Words are matched by label order (as in analyze.py); phones only for ARPAbet runs (same phone set as the reference).
"""
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ["arpa", "arpa_finetune", "arpa_adapted", "arpa_reduced", "mfa_ipa", "scratch"]


def tiers(path):
    """{tier name: [(label, start, end), ...]} for non-empty intervals of a long-format TextGrid."""
    out = {}
    for name, body in re.findall(r'name = "([^"]+)"(.*?)(?=name = "|\Z)', Path(path).read_text(), re.S):
        out[name] = [(t, float(a), float(b)) for a, b, t in
                     re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', body) if t.strip()]
    return out


def diffs(ref, hyp):
    """|reference - hypothesis| for every distinct reference boundary (a boundary shared by two neighbouring
    intervals is counted once), over intervals matched by label order."""
    sm = difflib.SequenceMatcher(a=[x[0] for x in ref], b=[x[0] for x in hyp], autojunk=False)
    d = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            r, h = ref[blk.a + k], hyp[blk.b + k]
            d[round(r[1], 4)] = abs(r[1] - h[1])
            d[round(r[2], 4)] = abs(r[2] - h[2])
    return list(d.values())


def main():
    rows = []
    for run in RUNS:
        for tier in ("words", "phones"):
            if tier == "phones" and run == "mfa_ipa":
                continue
            d = []
            for ref in sorted((ROOT / "manual_labels").glob("*.TextGrid")):
                hyp = ROOT / "outputs" / run / "isle_sess0131" / ref.name
                if hyp.exists():
                    d += diffs(tiers(ref)[tier], tiers(hyp)[tier])
            if not d:
                continue
            d = np.round(np.array(d) * 1000)
            rows.append({"run": run, "tier": tier, "boundaries": len(d), "mean error (ms)": round(d.mean(), 1),
                         "median (ms)": float(np.median(d)), "within 20 ms (%)": round(100 * (d <= 20).mean(), 1),
                         "within 50 ms (%)": round(100 * (d <= 50).mean(), 1)})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results/manual_accuracy.csv", index=False)
    (ROOT / "results/manual_accuracy.md").write_text(t.to_markdown(index=False) + "\n")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
