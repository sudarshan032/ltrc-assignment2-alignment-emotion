"""Build every table and plot in Results/ from Results/runs/ (written by the pipelines' test.py).

python make_results.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models"))
import common as C  # noqa: E402

RUNS, OUT, PLOTS = ROOT / "Results/runs", ROOT / "Results", ROOT / "Results/plots"
NAMES = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
         "ps": "pleasant surprise", "sad": "sad"}
SPLITS = {"words": "Unseen words (both speakers)", "spk_OAF": "Unseen speaker: test OAF",
          "spk_YAF": "Unseen speaker: test YAF"}
COLORS = plt.cm.tab10(np.arange(7))
VARIANT_ORDER = ["speech_wavlm", "speech_mfcc", "text_bert", "text_scratch",
                 "fusion_gmu_wavlm+bert", "fusion_concat_wavlm+bert", "fusion_gmu_mfcc+scratch"]
LABEL = {"speech_wavlm": "Speech: WavLM + BiLSTM", "speech_mfcc": "Speech: MFCC + BiLSTM (from scratch)",
         "text_bert": "Text: BERT + Transformer", "text_scratch": "Text: word emb. + Transformer (from scratch)",
         "fusion_gmu_wavlm+bert": "Fusion: WavLM + BERT, gated (GMU)",
         "fusion_concat_wavlm+bert": "Fusion: WavLM + BERT, concat",
         "fusion_gmu_mfcc+scratch": "Fusion: MFCC + word emb., gated (from scratch)"}


def variant(name):
    return re.sub(r"_(words|spk_OAF|spk_YAF)_s\d+$", "", name)  # strip _<split>_s<seed>


def md(df):
    return df.to_markdown(index=False) + "\n"


# ----------------------------------------------------------------------------- tables
def accuracy_tables(m):
    m = m.assign(variant=m.name.map(variant))
    g = m.groupby(["variant", "split"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                                            f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"), n=("seed", "count"))
    rows = []
    for v in VARIANT_ORDER:
        r = {"model": LABEL[v]}
        for s in SPLITS:
            x = g.loc[(v, s)]
            r[SPLITS[s]] = f"{100 * x.acc:.1f} ± {100 * x.acc_sd:.1f}"
        sp = g.loc[[(v, "spk_OAF"), (v, "spk_YAF")]]
        r["Unseen speaker (mean)"] = f"{100 * sp.acc.mean():.1f}"
        rows.append(r)
    acc = pd.DataFrame(rows)
    rows = []
    for v in VARIANT_ORDER:
        r = {"model": LABEL[v]}
        for s in SPLITS:
            x = g.loc[(v, s)]
            r[SPLITS[s]] = f"{100 * x.f1:.1f} ± {100 * x.f1_sd:.1f}"
        rows.append(r)
    f1 = pd.DataFrame(rows)
    acc.to_csv(OUT / "accuracy_table.csv", index=False), f1.to_csv(OUT / "macro_f1_table.csv", index=False)
    return acc, f1


def preds(v, split):
    return pd.concat([pd.read_csv(RUNS / f"{v}_{split}_s{s}_predictions.csv").assign(seed=s) for s in range(3)])


def per_class(v, splits):
    p = pd.concat([preds(v, s) for s in splits])
    return (p.true == p.pred).groupby(p.true).mean()


def per_class_table():
    rows = []
    for v in ["speech_wavlm", "speech_mfcc", "fusion_gmu_wavlm+bert"]:
        for tag, splits in [("unseen words", ["words"]), ("unseen speaker", ["spk_OAF", "spk_YAF"])]:
            pc = per_class(v, splits)
            rows.append({"model": LABEL[v], "test": tag, **{NAMES[e]: f"{100 * pc[e]:.1f}" for e in C.EMOTIONS}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "per_class_accuracy.csv", index=False)
    return t


# ----------------------------------------------------------------------------- plots
def confusion_plots():
    """Speech WavLM model on the unseen speaker, one matrix per test direction (3 seeds pooled); CSVs for all models."""
    for v in ["speech_wavlm", "speech_mfcc", "fusion_gmu_wavlm+bert"]:
        for sp in ["spk_OAF", "spk_YAF"]:
            p = preds(v, sp)
            cm = confusion_matrix(p.true, p.pred, labels=C.EMOTIONS, normalize="true") * 100
            pd.DataFrame(cm.round(1), index=[NAMES[e] for e in C.EMOTIONS], columns=[NAMES[e] for e in C.EMOTIONS]) \
                .to_csv(OUT / f"confusion_{v}_{sp}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, (sp, title) in zip(axes, [("spk_YAF", "test YAF (trained on OAF)"), ("spk_OAF", "test OAF (trained on YAF)")]):
        p = preds("speech_wavlm", sp)
        cm = confusion_matrix(p.true, p.pred, labels=C.EMOTIONS, normalize="true") * 100
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{np.floor(cm[i, j] + 0.5):.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 60 else "black")
        labels = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
        ax.set_xticks(range(7), labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(7), labels, fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=8), ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"Speech WavLM, {title}", fontsize=9)
    fig.suptitle("Unseen speaker: % of each true emotion, 3 seeds pooled", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_unseen_speaker.png", dpi=300)
    plt.close(fig)


def fmt(v):
    """mean ± std, without "-0.00"."""
    m, sd = (0.0 if abs(x) < 0.005 else x for x in (v.mean(), v.std()))
    return f"{m:.2f} ± {sd:.2f}"


def tsne_plots():
    """Separability of the three blocks INSIDE the multimodal (GMU) model: speech temporal block, text contextual
    block, fusion block (written by fusion_blocks.py). t-SNE of seed 0; silhouette (256-d) over 3 seeds."""
    fb = pd.read_csv(OUT / "fusion_blocks.csv")
    sil = []
    names = [("speech", "Temporal block (speech)"), ("text", "Contextual block (text)"), ("fusion", "Fusion block (GMU)")]
    for split, tag in [("words", "unseen_words"), ("spk_YAF", "unseen_speaker_YAF"), ("spk_OAF", "unseen_speaker_OAF")]:
        z = np.load(RUNS / f"fusion_gmu_wavlm+bert_{split}_s0_blocks.npz")
        spk = preds("fusion_gmu_wavlm+bert", split).query("seed == 0").speaker.values if "seed" in preds("fusion_gmu_wavlm+bert", split) \
            else pd.read_csv(RUNS / f"fusion_gmu_wavlm+bert_{split}_s0_predictions.csv").speaker.values
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
        for ax, (key, title) in zip(axes, names):
            col = {"speech": "speech_block", "text": "text_block", "fusion": "fusion_block"}[key]
            v = fb[fb.split == split][col]
            sil.append({"block": title, "test set": SPLITS[split], "silhouette (mean ± std, 3 seeds)": fmt(v)})
            e, y = z[key], z["y"]
            xy = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(e + 1e-6 * np.random.RandomState(0).randn(*e.shape))
            if key == "text":  # the 7 emotions of a word share one point: shift each emotion a little so all are visible
                r = 0.012 * np.ptp(xy, axis=0)
                xy = xy + r * np.stack([np.cos(2 * np.pi * y / 7), np.sin(2 * np.pi * y / 7)], 1)
            for k, emo in enumerate(C.EMOTIONS):
                for sp_, mk in [("OAF", "o"), ("YAF", "^")]:
                    m = (y == k) & (spk == sp_)
                    if m.any():
                        ax.scatter(*xy[m].T, s=5 if mk == "o" else 7, color=COLORS[k], marker=mk, alpha=0.75, linewidths=0)
            ax.set_title(f"{title}\nsilhouette {fmt(v)}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if key == "text":
                ax.set_xlabel("each word: 7 emotions at one point (drawn as a small ring)", fontsize=7)
        handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], ms=4, label=NAMES[e]) for k, e in enumerate(C.EMOTIONS)]
        if split == "words":
            handles += [plt.Line2D([], [], marker="o", ls="", color="grey", ms=4, label="OAF"), plt.Line2D([], [], marker="^", ls="", color="grey", ms=5, label="YAF")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False, handletextpad=0.1, columnspacing=0.8)
        fig.suptitle(f"Blocks of the multimodal (GMU) model, test set: {SPLITS[split]}", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(PLOTS / f"tsne_{tag}.png", dpi=300)
        plt.close(fig)
    t = pd.DataFrame(sil)
    t.to_csv(OUT / "separability_silhouette.csv", index=False)
    return t


def extra_experiments(m):
    """Prosody baseline and few-shot recalibration (each written by its own script)."""
    out = ""
    m = m.assign(variant=m.name.map(variant))
    if (OUT / "prosody_baseline.csv").exists():
        out += "\n## Prosody baseline (16 hand-crafted features + logistic regression; accuracy %)\n\n" + md(pd.read_csv(OUT / "prosody_baseline.csv"))
    if (OUT / "fewshot.csv").exists():
        f = pd.read_csv(OUT / "fewshot.csv").groupby(["model", "k"])[["zero_shot", "recalibrated"]].mean().round(1).reset_index()
        out += ("\n## Few-shot recalibration on the unseen speaker (classifier refit on k labelled clips per emotion; "
                "accuracy % on her other clips, mean of both directions, 3 seeds, 20 draws)\n\n" + md(f))
    return out


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for v, ls in [("speech_wavlm", "-"), ("speech_mfcc", "--"), ("text_bert", ":"), ("fusion_gmu_wavlm+bert", "-.")]:
        for split, c in [("words", "C0"), ("spk_YAF", "C3")]:
            h = pd.DataFrame(torch.load(C.CKPT / f"{v}_{split}_s0.pt", map_location="cpu")["history"])
            axes[0].plot(h.epoch, h.train_loss, ls=ls, color=c, label=f"{LABEL[v].split(':')[0]} {v.split('_')[1]}, {split}")
            axes[1].plot(h.epoch, 100 * h.val_acc, ls=ls, color=c)
    axes[0].set_title("training loss"), axes[1].set_title("validation accuracy (%)")
    for a in axes:
        a.set_xlabel("epoch")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- error analysis
PROSODY = ["f0_mean", "f0_range", "dur"]


def prosody():
    """Per-clip mean F0 (Hz), F0 range (Hz, 5th-95th percentile) and trimmed duration (s) of all 2,800 clips,
    from Results/prosody_features.csv (written by prosody_baseline.py), plus the same 3 features z-scored."""
    f = pd.read_csv(OUT / "prosody_features.csv")
    meta = C.metadata()
    f = pd.DataFrame({"id": f.id, "speaker": meta.speaker, "emotion": meta.emotion, "f0_mean": f.f0_mean,
                      "f0_range": f.f0_p95 - f.f0_p5, "dur": f.dur})
    f[PROSODY] = f[PROSODY].fillna(f[PROSODY].median())
    z = (f[PROSODY] - f[PROSODY].mean()) / f[PROSODY].std()
    return f, pd.concat([f[["id", "speaker", "emotion"]], z], axis=1)


def acoustic_table():
    """Mean F0, F0 range and duration per speaker and emotion (all 200 clips per cell)."""
    f, _ = prosody()
    m = f.groupby(["speaker", "emotion"])[PROSODY].mean()
    rows = [{"emotion": NAMES[e],
             "mean F0 Hz (OAF / YAF)": f"{m.loc[('OAF', e)].f0_mean:.0f} / {m.loc[('YAF', e)].f0_mean:.0f}",
             "F0 range Hz (OAF / YAF)": f"{m.loc[('OAF', e)].f0_range:.0f} / {m.loc[('YAF', e)].f0_range:.0f}",
             "duration s (OAF / YAF)": f"{m.loc[('OAF', e)].dur:.2f} / {m.loc[('YAF', e)].dur:.2f}"} for e in C.EMOTIONS]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "acoustic_stats.csv", index=False)
    return t


def acoustic_nearest():
    """Does pitch / pitch range / duration explain the unseen-speaker confusions? For each emotion of the test
    speaker: the training speaker's acoustically nearest emotion (z-scored profile means), vs the speech WavLM
    model's most frequent prediction (3 seeds)."""
    _, z = prosody()
    prof = z.groupby(["speaker", "emotion"])[PROSODY].mean()
    rows = []
    for split, test, train in [("spk_YAF", "YAF", "OAF"), ("spk_OAF", "OAF", "YAF")]:
        p = preds("speech_wavlm", split)
        for e in C.EMOTIONS:
            near = ((prof.loc[train] - prof.loc[(test, e)]) ** 2).sum(axis=1).idxmin()
            vc = p[p.true == e].pred.value_counts(normalize=True)
            rows.append({"test speaker": test, "true emotion": NAMES[e],
                         "nearest training emotion (prosody)": NAMES[near],
                         "model's top prediction": f"{NAMES[vc.index[0]]} ({np.floor(100 * vc.iloc[0] + 0.5):.0f} %)",
                         "agree": bool(near == vc.index[0])})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "acoustic_nearest.csv", index=False)
    return t


def failure_cases(n=5):
    """Typical errors of the main fusion model on unseen speakers: the n most frequent confusion types (true ->
    predicted, both directions, 3 seeds pooled), and for each the clip that gets that wrong label in the most seeds
    (ties: highest mean probability of the wrong label). With the clip's prosody and the training speaker's
    acoustically nearest emotion for that clip (z-scored mean F0, F0 range, duration)."""
    f, z = prosody()
    f, z = f.set_index("id"), z.set_index("id")
    prof = z.groupby(["speaker", "emotion"])[PROSODY].mean()
    errs = []
    for split in ["spk_OAF", "spk_YAF"]:
        p = preds("fusion_gmu_wavlm+bert", split)
        sp = preds("speech_wavlm", split)
        e = p[p.true != p.pred].copy()
        e["p_wrong"] = e.apply(lambda r: r[f"p_{r.pred}"], axis=1)
        e["speech_wrong_same"] = (sp.set_index(["id", "seed"]).loc[list(zip(e.id, e.seed)), "pred"].values == e.pred.values)
        e["train_spk"] = "YAF" if split == "spk_OAF" else "OAF"
        errs.append(e)
    e = pd.concat(errs)
    types = e.groupby(["true", "pred"]).size().sort_values(ascending=False).head(n)
    rows = []
    for (t, pr), count in types.items():
        c = e[(e.true == t) & (e.pred == pr)].groupby("id").agg(seeds=("seed", "size"), p=("p_wrong", "mean"),
                                                                 speech=("speech_wrong_same", "sum"), spk=("speaker", "first"),
                                                                 train_spk=("train_spk", "first"))
        r = c.sort_values(["seeds", "p"], ascending=False).iloc[0]
        cid = c.sort_values(["seeds", "p"], ascending=False).index[0]
        near = ((prof.loc[r.train_spk] - z.loc[cid, PROSODY].astype(float)) ** 2).sum(axis=1).idxmin()
        mt = f[(f.speaker == r.spk) & (f.emotion == t)][PROSODY].mean()
        rows.append({"confusion": f"{NAMES[t]} -> {NAMES[pr]}", "errors of this type (3 seeds)": int(count),
                     "clip": cid, "seeds wrong": f"{int(r.seeds)}/3", "speech-only same error": f"{int(r.speech)}/3",
                     "mean F0 Hz (clip / her avg)": f"{f.loc[cid].f0_mean:.0f} / {mt.f0_mean:.0f}",
                     "F0 range Hz": f"{f.loc[cid].f0_range:.0f} / {mt.f0_range:.0f}",
                     "duration s": f"{f.loc[cid].dur:.2f} / {mt.dur:.2f}",
                     "nearest training emotion (prosody)": NAMES[near]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "failure_cases.csv", index=False)
    return t


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame([json.loads(p.read_text()) for p in sorted(RUNS.glob("*_metrics.json"))])
    acc, f1 = accuracy_tables(m)
    pc = per_class_table()
    confusion_plots()
    sil = tsne_plots()
    training_curves()
    fail = failure_cases()
    extra = extra_experiments(m)
    ac = acoustic_table()
    near = acoustic_nearest()
    abl = OUT / "text_ablation.csv"
    abl = pd.read_csv(abl).groupby(["split", "fusion"], sort=False)[["normal", "text_shuffled", "text_constant"]].mean().round(1).reset_index() if abl.exists() else None
    text = ("# Results\n\nAll numbers: test set, mean ± std over 3 seeds (0, 1, 2). Chance = 14.3 %.\n\n"
            "Names: `words` = unseen words (both speakers); `spk_OAF` = unseen speaker, tested on OAF (trained on YAF); "
            "`spk_YAF` = tested on YAF (trained on OAF). Run files: `runs/<model>_<split>_s<seed>_*`.\n\n"
            "Plots: [confusion](plots/confusion_unseen_speaker.png), [t-SNE unseen words](plots/tsne_unseen_words.png), "
            "[t-SNE test YAF](plots/tsne_unseen_speaker_YAF.png), [t-SNE test OAF](plots/tsne_unseen_speaker_OAF.png), "
            "[training curves](plots/training_curves.png).\n\n"
            "## Accuracy (%)\n\n" + md(acc) + "\n## Macro-F1 (%)\n\n" + md(f1)
            + "\n## Per-class accuracy (%) — recall of each emotion, 3 seeds pooled\n\n" + md(pc)
            + "\n## Separability of the blocks inside the multimodal (GMU) model (silhouette on 256-d test vectors, 3 seeds)\n\n" + md(sil)
            + "\n## Failure cases (GMU fusion, unseen speaker, seed 0, largest margin; stats = clip / her average for the true emotion)\n\n"
            + md(fail)
            + "\n## Acoustic profile per speaker and emotion (all 200 clips per cell)\n\n" + md(ac)
            + "\n## Does prosody explain the unseen-speaker confusions? (nearest training emotion by mean F0, F0 range, duration vs the speech WavLM model's top prediction)\n\n" + md(near)
            + ("\n## Text ablation: fusion accuracy with the text input shuffled or held constant (mean over seeds)\n\n" + md(abl)
               if abl is not None else "")
            + extra)
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
