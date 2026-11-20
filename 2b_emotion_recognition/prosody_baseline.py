"""Hand-crafted prosody baseline: 16 pitch / duration / energy-contour features per clip + logistic regression,
evaluated on the same splits as the neural models. Shows how much plain prosody explains.

python prosody_baseline.py      # -> Results/prosody_baseline.csv, Results/prosody_features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import common as C  # noqa: E402


def feats(path):
    import librosa
    y = C.load_speech(path)                                   # same preprocessing as the models (normalized level)
    f0, vf, _ = librosa.pyin(y, fmin=70, fmax=600, sr=C.SR, frame_length=1024)
    f = f0[vf] if vf.any() else np.array([np.nan])
    st = 12 * np.log2(f / np.nanmedian(f)) if vf.any() else np.array([0.0])   # contour in semitones
    rms = librosa.feature.rms(y=y)[0]
    e = 20 * np.log10(rms + 1e-6)
    t = np.arange(len(st))
    slope = np.polyfit(t, st, 1)[0] if len(st) > 2 else 0.0
    return dict(dur=len(y) / C.SR, f0_mean=np.nanmean(f), f0_std=np.nanstd(f), f0_p5=np.nanpercentile(f, 5),
                f0_p95=np.nanpercentile(f, 95), f0_range_st=np.nanpercentile(st, 95) - np.nanpercentile(st, 5),
                f0_slope=slope, voiced_frac=vf.mean(), e_std=e.std(), e_range=np.percentile(e, 95) - np.percentile(e, 5),
                e_slope=np.polyfit(np.arange(len(e)), e, 1)[0], e_peak_pos=np.argmax(e) / len(e),
                zcr=librosa.feature.zero_crossing_rate(y)[0].mean(),
                centroid=librosa.feature.spectral_centroid(y=y, sr=C.SR)[0].mean(),
                flatness=librosa.feature.spectral_flatness(y=y)[0].mean(),
                rolloff=librosa.feature.spectral_rolloff(y=y, sr=C.SR)[0].mean())


def main():
    df = C.metadata()
    cache = C.RESULTS / "prosody_features.csv"     # one row per clip (metadata order); also used by make_results.py
    if cache.exists():
        F = pd.read_csv(cache)
    else:
        F = pd.DataFrame(Parallel(n_jobs=8)(delayed(feats)(p) for p in df.path))
        F.insert(0, "id", df.id.values)
        F.to_csv(cache, index=False)
    F = F.drop(columns="id")
    X, y = F.fillna(F.median()).values, df.label.values
    rows = []
    for split in ["words", "spk_OAF", "spk_YAF"]:
        sp = C.split(df, split)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[sp["train"]], y[sp["train"]])
        acc = (clf.predict(X[sp["test"]]) == y[sp["test"]]).mean()
        dur = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[sp["train"]][:, :1], y[sp["train"]])
        rows.append(dict(split=split, prosody_16_features=round(100 * acc, 1),
                         duration_only=round(100 * (dur.predict(X[sp["test"]][:, :1]) == y[sp["test"]]).mean(), 1)))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "prosody_baseline.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
