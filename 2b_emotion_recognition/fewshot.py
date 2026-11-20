"""Few-shot recalibration on the unseen speaker: keep each trained speech model's encoder frozen and refit only
the classifier (logistic regression on its 256-d test embeddings) using k labelled clips per emotion of the new
speaker; test on her remaining clips. 20 random draws per k. Compares with the zero-shot model.

python fewshot.py      # -> Results/fewshot.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RES = Path(__file__).resolve().parent / "Results"
R = str(RES / "runs") + "/"
rows = []
for model in ["speech_wavlm", "speech_mfcc"]:
    for split in ["spk_OAF", "spk_YAF"]:
        for seed in range(3):
            name = f"{model}_{split}_s{seed}"
            e = np.load(R + f"{name}_embeddings.npz")
            X, y = e["emb"], e["y"]
            p = pd.read_csv(R + f"{name}_predictions.csv")
            zero = (p.true == p.pred).mean()
            for k in [1, 2, 5, 10, 20]:
                accs, zaccs = [], []
                for draw in range(20):
                    rng = np.random.RandomState(draw)
                    tr = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(7)])
                    te = np.setdiff1d(np.arange(len(y)), tr)
                    sc = StandardScaler().fit(X[tr])
                    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                    accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
                    zaccs.append((p.true.values[te] == p.pred.values[te]).mean())
                rows.append(dict(model=model, split=split, seed=seed, k=k, zero_shot=100 * np.mean(zaccs),
                                 recalibrated=100 * np.mean(accs), draw_std=100 * np.std(accs)))
t = pd.DataFrame(rows)
t.to_csv(RES / "fewshot.csv", index=False)
s = t.groupby(["model", "k"])[["zero_shot", "recalibrated"]].mean().round(1)
print(s.to_string())
