"""Shared code for the three pipelines: data, splits, preprocessing, feature caches, blocks, training loop.

Block map (assignment figure -> code):
  Preprocessing        load_speech() / clean_text()
  Feature extraction   speech: WavLM-base-plus (frozen, all 13 layers) or MFCC+deltas   -> T x F
                       text:   BERT-base-uncased (frozen, all 13 layers) or learned word embeddings -> tokens x F
  Temporal modelling   SpeechEncoder: layer mix -> projection -> 2-layer BiLSTM -> attention pooling (256-d)
  Contextual modelling TextEncoder:   layer mix -> projection -> 2-layer Transformer encoder -> attention pooling (256-d)
  Fusion               GatedFusion (gated multimodal unit) or concatenation
  Classifier           MLP 256 -> 128 -> 7
"""
import functools
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TESS_DIR", ROOT / "data/TESS"))
CACHE = ROOT / "cache"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "Results"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]  # ps = pleasant surprise
SR = 16000
DEVICE = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data + splits
def metadata():
    """One row per clip. File names are <speaker>_<word>_<emotion>.wav."""
    rows = []
    for p in sorted(DATA_DIR.glob("*/*.wav")):
        _, word, emo = p.stem.split("_")
        spk = p.parent.name.split("_")[0]  # folder, not file name: one file is misnamed "OA_bite_neutral"
        rows.append(dict(path=str(p), id=p.stem, speaker=spk, word=word.lower(), emotion=emo.lower(),
                         text=f"Say the word {word.lower()}"))
    df = pd.DataFrame(rows)
    assert len(df) == 2800, f"expected 2800 TESS clips in {DATA_DIR}, found {len(df)}"
    df["label"] = df.emotion.map(EMOTIONS.index)
    return df


def split(df, name, seed=0):
    """Return {'train','val','test'} -> index arrays.

    words   : word-disjoint split (140/30/30 of the 200 words), both speakers everywhere.
              No test word is ever heard in training, so the model can't use word identity.
    spk_OAF : speaker-independent, test = all OAF clips, train/val = YAF (val = 30 held-out words).
    spk_YAF : the reverse.
    The split is fixed (seed 0) for every model so all pipelines see the same test clips.
    """
    words = sorted(df.word.unique())
    rng = np.random.RandomState(0)
    rng.shuffle(words)
    if name == "words":
        tr, va, te = set(words[:140]), set(words[140:170]), set(words[170:])
        return {k: np.where(df.word.isin(s))[0] for k, s in [("train", tr), ("val", va), ("test", te)]}
    if name.startswith("spk_"):
        test_spk = name[4:]
        other = df.speaker != test_spk
        va = set(words[:30])
        return {"train": np.where(other & ~df.word.isin(va))[0],
                "val": np.where(other & df.word.isin(va))[0],
                "test": np.where(~other)[0]}
    raise ValueError(name)


# ----------------------------------------------------------------------------- preprocessing
def load_speech(path):
    """16 kHz mono, leading/trailing silence trimmed (30 dB below peak), zero-mean / unit-variance."""
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return (y - y.mean()) / (y.std() + 1e-7)


def clean_text(t):
    return re.sub(r"[^a-z' ]", " ", t.lower()).split()


# ----------------------------------------------------------------------------- feature caches
@functools.lru_cache(maxsize=None)
def features(kind):
    """List (one per clip, in metadata() order) of float tensors [layers, T, F] (layers=1 for non-layered).
    Cached on disk (cache/<kind>.pt) and in memory, so run_all.py loads WavLM features only once."""
    df = metadata()
    return [f.to(DEVICE) for f in torch.load(cache_path(kind, df))]  # kept on the GPU (fp16): batching is then cheap


def cache_path(kind, df):
    """cache/<kind>.pt, extracted on first use."""
    path = CACHE / f"{kind}.pt"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        torch.save({"wavlm": _wavlm, "mfcc": _mfcc, "bert": _bert}[kind](df), path)
    return path


@torch.no_grad()
def _wavlm(df):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()
    out = []
    for i, p in enumerate(df.path):
        x = torch.tensor(load_speech(p))[None].to(DEVICE)
        hs = m(x, output_hidden_states=True).hidden_states          # 13 x [1, T, 768], 50 frames/s
        out.append(torch.stack([h[0] for h in hs]).half().cpu())
        if i % 200 == 0:
            print(f"wavlm {i}/{len(df)}", flush=True)
    return out


def _mfcc(df):
    import librosa
    out = []
    for p in df.path:
        y = load_speech(p)
        m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=400, hop_length=160, n_mels=64)  # 25 ms / 10 ms
        f = np.concatenate([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])     # 120 x T
        out.append(torch.tensor(f.T, dtype=torch.float16)[None])
    return out


@torch.no_grad()
def _bert(df):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
    cache = {}
    for t in df.text.unique():
        enc = tok(" ".join(clean_text(t)), return_tensors="pt").to(DEVICE)
        hs = m(**enc, output_hidden_states=True).hidden_states       # 13 x [1, tokens, 768]
        cache[t] = torch.stack([h[0] for h in hs]).half().cpu()
    return [cache[t] for t in df.text]


def word_vocab(df, train_idx):
    """Word-level vocabulary from the training transcripts only (unseen words -> <unk>)."""
    words = sorted({w for t in df.text.iloc[train_idx] for w in clean_text(t)})
    return {w: i + 2 for i, w in enumerate(words)}                  # 0 = pad, 1 = <unk>


def word_ids(df, vocab):
    return [torch.tensor([vocab.get(w, 1) for w in clean_text(t)], device=DEVICE) for t in df.text]


# ----------------------------------------------------------------------------- blocks
class LayerMix(nn.Module):
    """Learned softmax-weighted sum over the encoder's hidden layers ([B, L, T, F] -> [B, T, F])."""

    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return (F.softmax(self.w, 0)[None, :, None, None] * x).sum(1)


class AttnPool(nn.Module):
    """Attention pooling: a learned score per time step, softmax over valid steps, weighted mean."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, h, mask):
        a = self.score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        return (a.unsqueeze(-1) * h).sum(1), a


class SpeechEncoder(nn.Module):
    """Temporal modelling block: frames -> 256-d utterance vector."""

    def __init__(self, in_dim, layers, d=256, dropout=0.3):
        super().__init__()
        self.mix = LayerMix(layers)
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(d)

    def forward(self, x, mask):
        h = self.inp(self.mix(x))
        lengths = mask.sum(1).cpu()
        h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=mask.shape[1])
        return self.pool(h, mask)[0]


class TextEncoder(nn.Module):
    """Contextual modelling block: tokens -> 256-d sentence vector."""

    def __init__(self, in_dim=None, layers=1, vocab_size=None, d=256, dropout=0.3):
        super().__init__()
        if vocab_size:  # from-scratch variant: learned word embeddings
            self.emb, self.mix = nn.Embedding(vocab_size, d, padding_idx=0), None
            self.inp = nn.Dropout(dropout)
        else:           # BERT features
            self.emb, self.mix = None, LayerMix(layers)
            self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.ctx = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pool = AttnPool(d)

    def forward(self, x, mask):
        h = self.emb(x) if self.emb is not None else self.mix(x)
        h = self.inp(h) + self.pos[:, : h.shape[1]]
        h = self.ctx(h, src_key_padding_mask=~mask)
        return self.pool(h, mask)[0]


def classifier(d=256, n=len(EMOTIONS), dropout=0.3):
    return nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n))


class GatedFusion(nn.Module):
    """Gated multimodal unit (Arevalo et al., 2017): h = z * tanh(Ws s) + (1 - z) * tanh(Wt t),
    z = sigmoid(Wz [s; t]) per dimension. z near 1 -> the model relies on speech."""

    def __init__(self, d=256):
        super().__init__()
        self.s, self.t, self.z = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(2 * d, d)
        self.last_gate = None

    def forward(self, s, t):
        z = torch.sigmoid(self.z(torch.cat([s, t], -1)))
        self.last_gate = z.detach()
        return z * torch.tanh(self.s(s)) + (1 - z) * torch.tanh(self.t(t))


class ConcatFusion(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.last_gate = None

    def forward(self, s, t):
        return self.f(torch.cat([s, t], -1))


class UniModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc, self.cls = encoder, classifier()

    def forward(self, batch):
        e = self.enc(*batch[0])
        return self.cls(e), e


class FusionModel(nn.Module):
    def __init__(self, speech, text, fusion):
        super().__init__()
        self.speech, self.text, self.fusion, self.cls = speech, text, fusion, classifier()

    def forward(self, batch):
        s, t = self.speech(*batch[0]), self.text(*batch[1])
        h = self.fusion(s, t)
        return self.cls(h), h


# ----------------------------------------------------------------------------- batching
def pad(seqs):
    """seqs: list of [L, T, F] float or [T] long tensors -> padded batch + bool mask [B, T].
    Built on the tensors' own device and dtype (features live on the GPU), then cast to float32."""
    dev = seqs[0].device
    lengths = torch.tensor([s.shape[-2] if s.dim() == 3 else s.shape[0] for s in seqs], device=dev)
    T = int(lengths.max())
    if seqs[0].dim() == 1:
        x = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = s
    else:
        L, _, Fd = seqs[0].shape
        x = torch.zeros(len(seqs), L, T, Fd, dtype=seqs[0].dtype, device=dev)
        for i, s in enumerate(seqs):
            x[i, :, : s.shape[1]] = s
        x = x.float()
    mask = torch.arange(T, device=dev)[None] < lengths[:, None]
    return x, mask


def batches(inputs, labels, idx, bs, shuffle):
    """inputs: list of modalities, each a list of per-clip tensors."""
    idx = np.array(idx)
    if shuffle:
        idx = idx[np.random.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xs = [tuple(t.to(DEVICE) for t in pad([m[j] for j in b])) for m in inputs]
        yield xs, torch.tensor(labels[b]).to(DEVICE), b


# ----------------------------------------------------------------------------- train / eval
def set_seed(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s)


@torch.no_grad()
def predict(model, inputs, labels, idx, bs=64):
    model.eval()
    P, E, G = [], [], []
    for xs, _, _ in batches(inputs, labels, idx, bs, False):
        logits, emb = model(xs)
        P.append(logits.softmax(-1).cpu()), E.append(emb.cpu())
        gate = getattr(getattr(model, "fusion", None), "last_gate", None)
        if gate is not None:
            G.append(gate.mean(-1).cpu())
    out = dict(probs=torch.cat(P).numpy(), emb=torch.cat(E).numpy())
    if G:
        out["gate"] = torch.cat(G).numpy()
    return out


def scores(y, probs):
    p = probs.argmax(1)
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"))


def train(model, inputs, labels, sp, epochs=15, bs=32, lr=1e-3, log=None):
    """AdamW + one-cycle LR over a fixed budget of `epochs`; the final weights are kept.

    No early stopping: in the unseen-speaker splits validation is the training speaker, which
    saturates at 100 % within a few epochs, so it can't pick an epoch that transfers. A fixed
    budget with the full LR schedule makes the result independent of when validation saturates.
    Validation is still logged for the training curves.
    """
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(sp["train"]) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    hist = []
    for ep in range(epochs):
        model.train()
        tot = 0
        for xs, y, b in batches(inputs, labels, sp["train"], bs, True):
            loss = F.cross_entropy(model(xs)[0], y, label_smoothing=0.05)
            opt.zero_grad(), loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step()
            tot += loss.item() * len(b)
        v = scores(labels[sp["val"]], predict(model, inputs, labels, sp["val"])["probs"])
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(sp["train"]), **{f"val_{k}": x for k, x in v.items()}))
        if log:
            print(f"ep {ep + 1:2d} loss {hist[-1]['train_loss']:.3f} val acc {v['acc']:.3f} f1 {v['macro_f1']:.3f}", flush=True)
    return hist


def save_run(name, model, config, hist):
    CKPT.mkdir(exist_ok=True)
    torch.save({"config": config, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "history": hist}, CKPT / f"{name}.pt")


def write_test_outputs(name, df, sp, out, config):
    """Per-clip predictions + embeddings + metrics for one trained model on its test split."""
    d = RESULTS / "runs"
    d.mkdir(parents=True, exist_ok=True)
    te = sp["test"]
    y = df.label.values[te]
    pred = pd.DataFrame(dict(id=df.id.values[te], speaker=df.speaker.values[te], word=df.word.values[te],
                             true=[EMOTIONS[i] for i in y], pred=[EMOTIONS[i] for i in out["probs"].argmax(1)],
                             conf=out["probs"].max(1)))
    for k, e in enumerate(EMOTIONS):
        pred[f"p_{e}"] = out["probs"][:, k]
    if "gate" in out:
        pred["speech_gate"] = out["gate"]
    pred.to_csv(d / f"{name}_predictions.csv", index=False)
    np.savez_compressed(d / f"{name}_embeddings.npz", emb=out["emb"], y=y)
    m = dict(name=name, **config, **scores(y, out["probs"]), n_test=len(te))
    (d / f"{name}_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m))
    return m
