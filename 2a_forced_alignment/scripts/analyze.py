"""Compare the alignment runs in outputs/<variant>/ and write tables.

Accuracy against hand-corrected boundaries is computed separately (evaluate_manual.py).
  * MFA's own per-file scores (alignment_analysis.csv: log-likelihood, phone duration deviation)
  * boundary agreement between runs (arpa = reference run)
  * duration-based flags (phones of 30 ms or less = 3 frames or fewer, very long phones)
Run inside the `mfa` env:  python scripts/analyze.py
"""
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from praatio import textgrid

ROOT = Path(__file__).resolve().parent.parent
OUT, RES = ROOT / "outputs", ROOT / "results"
VARIANTS = ["arpa", "arpa_finetune", "arpa_adapted", "arpa_reduced", "mfa_ipa", "scratch"]
SIL = {"", "sil", "sp", "spn"}
MIN_DUR = 0.031  # 3 frames or fewer; the true floor depends on each phone's HMM (some phones can be 1 frame)


def load():
    rows = []
    for v in VARIANTS:
        for tg_path in sorted((OUT / v).rglob("*.TextGrid")):
            tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
            for tier in ("words", "phones"):
                for s, e, lab in tg.getTier(tier).entries:
                    rows.append(dict(variant=v, utt=tg_path.stem, tier=tier, start=s, end=e,
                                     label=lab, dur=e - s, speech=lab not in SIL))
    return pd.DataFrame(rows)


def summary(df):
    out = []
    for v in VARIANTS:
        d = df[(df.variant == v) & df.speech]
        ph, w = d[d.tier == "phones"], d[d.tier == "words"]
        csv = OUT / v / "alignment_analysis.csv"
        a = pd.read_csv(csv) if csv.exists() else None
        out.append({
            "run": v,
            "words": len(w), "phones": len(ph),
            "mean phone (ms)": round(1000 * ph.dur.mean()),
            "phones <= 30 ms (%)": round(100 * (ph.dur <= MIN_DUR).mean(), 1),
            "phones > 250 ms": int((ph.dur > 0.25).sum()),
            "speech log-lik": round(a.speech_log_likelihood.mean(), 2) if a is not None else None,
            "phone dur. deviation": round(a.phone_duration_deviation.mean(), 2) if a is not None else None,
        })
    return pd.DataFrame(out)


def boundaries(df, v, utt, tier):
    d = df[(df.variant == v) & (df.utt == utt) & (df.tier == tier) & df.speech]
    return list(zip(d.label, d.start, d.end))


def agreement(df, ref="arpa"):
    """Mean |boundary difference| vs the reference run, over words matched by label order."""
    out = []
    for v in VARIANTS:
        if v == ref:
            continue
        for tier in ("words", "phones"):
            if tier == "phones" and v == "mfa_ipa":
                continue  # different phone set, phones can't be matched
            diffs = []
            for utt in df.utt.unique():
                a, b = boundaries(df, ref, utt, tier), boundaries(df, v, utt, tier)
                sm = difflib.SequenceMatcher(a=[x[0] for x in a], b=[x[0] for x in b], autojunk=False)
                for blk in sm.get_matching_blocks():
                    for k in range(blk.size):
                        x, y = a[blk.a + k], b[blk.b + k]
                        diffs += [abs(x[1] - y[1]), abs(x[2] - y[2])]
            diffs = np.round(np.array(diffs) * 1000)  # whole ms: boundaries sit on 10 ms frames, avoid float noise at 20 ms
            out.append({"run vs arpa": v, "tier": tier, "boundaries": len(diffs),
                        "mean |diff| (ms)": round(diffs.mean(), 1),
                        "median (ms)": round(float(np.median(diffs)), 1),
                        "within 20 ms (%)": round(100 * (diffs <= 20).mean(), 1),
                        "> 100 ms": int((diffs > 100).sum())})
    return pd.DataFrame(out)


def flags(df, v="arpa"):
    d = df[(df.variant == v) & (df.tier == "phones") & df.speech]
    f = d[(d.dur <= MIN_DUR) | (d.dur > 0.25)].copy()
    w = df[(df.variant == v) & (df.tier == "words")]
    f["word"] = [w[(w.utt == r.utt) & (w.start <= r.start) & (w.end >= r.end)].label.iloc[0] for r in f.itertuples()]
    f["why"] = np.where(f.dur <= MIN_DUR, "<= 30 ms", "very long")
    return f[["utt", "word", "label", "start", "end", "dur", "why"]].round(3)


def main():
    RES.mkdir(exist_ok=True)
    df = load()
    df.to_csv(RES / "all_intervals.csv", index=False)
    for name, table in [("summary", summary(df)), ("agreement_vs_arpa", agreement(df)), ("flags_arpa", flags(df))]:
        table.to_csv(RES / f"{name}.csv", index=False)
        (RES / f"{name}.md").write_text(table.to_markdown(index=False) + "\n")
        print(f"\n## {name}\n{table.to_markdown(index=False)}")


if __name__ == "__main__":
    main()
